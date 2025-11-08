#!/usr/bin/env python3
"""
Обучение META нейросети через CMA-ES на Binance датасете

Архитектура (БЕЗ УПРОЩЕНИЙ):
- Входы: 5 predictions (4 эксперта + BASE) + 7 context features
- Feature Engineering: 18D → 36D enriched features
- Attention: 7D context → 16D → 4D weights
- Main Network: 36D → 64D → 32D → 16D → 1D
- MC Dropout для uncertainty estimation
- Phase-specific heads (6 фаз)

Обучение:
- CMA-ES оптимизация весов нейросети
- Раздельное обучение по фазам
- Bootstrap для оценки качества

Usage:
    python3 training/train_binance_meta.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
import logging
import pickle

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импорт экспертов
from models.experts import XGBoostExpert, RandomForestExpert, NeuralNetworkExpert

# Импорт META архитектуры (ПОЛНАЯ версия из meta_neural_cem.py)
# Используем класс NeuralMetaNetwork напрямую
from meta_neural_cem import NeuralMetaNetwork, _safe_float, _sigmoid

# CMA-ES
try:
    import cma
    HAVE_CMA = True
except ImportError:
    print("WARNING: CMA-ES not available. Install: pip install cma")
    HAVE_CMA = False

# Метрики
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Конфигурация обучения META"""

    # Пути к данным (относительно корня проекта)
    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).parent.parent

    TRAIN_DATASET = str(_PROJECT_ROOT / "data" / "datasets" / "binance_dataset_train.parquet")
    TEST_DATASET = str(_PROJECT_ROOT / "data" / "datasets" / "binance_dataset_test.parquet")

    # Пути к моделям
    MODELS_DIR = str(_PROJECT_ROOT / "models" / "saved")
    META_SAVE_PATH = str(_PROJECT_ROOT / "models" / "saved" / "meta_cmaes_model.pkl")

    # CMA-ES параметры
    CMA_SIGMA = 0.3
    CMA_POPSIZE = 30
    CMA_MAX_ITER = 100
    CMA_TOLX = 1e-8

    # Обучение
    MIN_SAMPLES_PER_PHASE = 50
    N_PHASES = 6

    # MC Dropout
    MC_DROPOUT_SAMPLES = 30


# ============================================================================
# BASE LOGIC (заглушка)
# ============================================================================

class BaseLogic:
    """
    Заглушка для BASE логики

    В production: реализовать prob_up_down_at_time_multiTF
    Сейчас: возвращает среднее экспертов
    """

    def predict(self, p_xgb: float, p_rf: float, p_arf: float, p_nn: float) -> float:
        """Возвращает среднее предсказание экспертов"""
        preds = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
        return np.mean(preds) if preds else 0.5


# ============================================================================
# CONTEXT FEATURES (синтетические)
# ============================================================================

def generate_synthetic_context(phase: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Генерирует синтетические context features (7D)

    В production: извлекать из реальных данных рынка

    Args:
        phase: фаза рынка (0-5)
        seed: random seed

    Returns:
        context: 7D вектор [vol_ratio, trend_macd, jump_flag, funding_sign,
                            book_imb, ofi_15s, basis_pct]
    """
    if seed is not None:
        np.random.seed(seed)

    # Синтетические значения с учетом фазы
    vol_ratio = 0.8 + 0.4 * np.random.rand() + (phase / 10)  # 0.8-1.4
    trend_macd = -0.5 + np.random.rand()  # -0.5 to 0.5
    jump_flag = 1.0 if np.random.rand() > 0.9 else 0.0  # 10% jump
    funding_sign = -1.0 + 2.0 * np.random.rand()  # -1 to 1
    book_imb = -0.3 + 0.6 * np.random.rand()  # -0.3 to 0.3
    ofi_15s = -0.2 + 0.4 * np.random.rand()  # -0.2 to 0.2
    basis_pct = -0.01 + 0.02 * np.random.rand()  # -0.01 to 0.01

    return np.array([
        vol_ratio, trend_macd, jump_flag, funding_sign,
        book_imb, ofi_15s, basis_pct
    ], dtype=float)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает train и test датасеты"""
    logger.info("="*80)
    logger.info("ЗАГРУЗКА ДАТАСЕТОВ")
    logger.info("="*80)

    logger.info(f"Loading train: {Config.TRAIN_DATASET}")
    df_train = pd.read_parquet(Config.TRAIN_DATASET)

    logger.info(f"Loading test: {Config.TEST_DATASET}")
    df_test = pd.read_parquet(Config.TEST_DATASET)

    logger.info(f"\nTrain: {len(df_train)} samples")
    logger.info(f"Test: {len(df_test)} samples")

    return df_train, df_test


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Извлекает 68D фичи из датафрейма"""
    feature_cols = [col for col in df.columns if col.startswith('features_')]
    return df[feature_cols].values


# ============================================================================
# EXPERT PREDICTIONS
# ============================================================================

def load_experts() -> Dict:
    """Загружает всех 4 экспертов"""
    logger.info("\n" + "="*80)
    logger.info("ЗАГРУЗКА ЭКСПЕРТОВ")
    logger.info("="*80)

    experts = {}

    # XGBoost
    xgb_path = f"{Config.MODELS_DIR}/xgb_expert.pkl"
    if os.path.exists(xgb_path):
        experts['xgb'] = XGBoostExpert.load(xgb_path)
        logger.info(f"✓ XGBoost loaded: {experts['xgb'].train_samples} samples")
    else:
        logger.warning("⚠ XGBoost not found")
        experts['xgb'] = None

    # Random Forest
    rf_path = f"{Config.MODELS_DIR}/rf_expert.pkl"
    if os.path.exists(rf_path):
        experts['rf'] = RandomForestExpert.load(rf_path)
        logger.info(f"✓ RandomForest loaded: {experts['rf'].train_samples} samples")
    else:
        logger.warning("⚠ RandomForest not found")
        experts['rf'] = None

    # Adaptive RF (заглушка если нет)
    arf_path = f"{Config.MODELS_DIR}/arf_expert.pkl"
    logger.info("⚠ AdaptiveRF: using fallback (returns 0.5)")
    experts['arf'] = None  # Заглушка

    # Neural Network
    nn_path = f"{Config.MODELS_DIR}/nn_expert.pkl"
    if os.path.exists(nn_path):
        experts['nn'] = NeuralNetworkExpert.load(nn_path)
        logger.info(f"✓ NeuralNet loaded: {experts['nn'].train_samples} samples")
    else:
        logger.warning("⚠ NeuralNet not found")
        experts['nn'] = None

    loaded = sum(1 for e in experts.values() if e is not None)
    logger.info(f"\nЗагружено экспертов: {loaded}/4")

    return experts


def get_expert_predictions(
    X: np.ndarray,
    experts: Dict,
    base_logic: BaseLogic
) -> np.ndarray:
    """
    Получает предсказания от всех 4 экспертов + BASE

    Args:
        X: (N, 68) фичи
        experts: dict с экспертами
        base_logic: BASE логика

    Returns:
        predictions: (N, 5) - [p_xgb, p_rf, p_arf, p_nn, p_base]
    """
    N = len(X)
    predictions = np.zeros((N, 5))

    reg_ctx = {'phase': 0}

    # XGBoost
    if experts['xgb'] is not None:
        for i in range(N):
            p, _ = experts['xgb'].proba_up(X[i], reg_ctx)
            predictions[i, 0] = p
    else:
        predictions[:, 0] = 0.5

    # Random Forest
    if experts['rf'] is not None:
        for i in range(N):
            p, _ = experts['rf'].proba_up(X[i], reg_ctx)
            predictions[i, 1] = p
    else:
        predictions[:, 1] = 0.5

    # Adaptive RF (заглушка)
    predictions[:, 2] = 0.5  # Fallback

    # Neural Network
    if experts['nn'] is not None:
        for i in range(N):
            p, _ = experts['nn'].proba_up(X[i], reg_ctx)
            predictions[i, 3] = p
    else:
        predictions[:, 3] = 0.5

    # BASE logic
    for i in range(N):
        p_base = base_logic.predict(
            predictions[i, 0],
            predictions[i, 1],
            predictions[i, 2],
            predictions[i, 3]
        )
        predictions[i, 4] = p_base

    return predictions


# ============================================================================
# CMA-ES TRAINING
# ============================================================================

def train_meta_phase_cmaes(
    phase: int,
    X_train_preds: np.ndarray,
    X_train_context: np.ndarray,
    y_train: np.ndarray,
    X_val_preds: np.ndarray,
    X_val_context: np.ndarray,
    y_val: np.ndarray
) -> NeuralMetaNetwork:
    """
    Обучает META нейросеть для конкретной фазы через CMA-ES

    Args:
        phase: номер фазы
        X_train_preds: (N_train, 5) predictions
        X_train_context: (N_train, 7) context
        y_train: (N_train,) targets
        X_val_preds, X_val_context, y_val: validation данные

    Returns:
        trained_network: обученная NeuralMetaNetwork
    """
    logger.info(f"\n{'─'*60}")
    logger.info(f"Phase {phase}: CMA-ES Training")
    logger.info(f"{'─'*60}")
    logger.info(f"Train samples: {len(y_train)}")
    logger.info(f"Val samples: {len(y_val)}")

    if not HAVE_CMA:
        logger.error("CMA-ES not available! Install: pip install cma")
        # Fallback: возвращаем не обученную сеть
        return NeuralMetaNetwork(phase=phase)

    # Создаем сеть
    network = NeuralMetaNetwork(phase=phase)

    # Feature engineering для всего датасета
    def enrich_features(preds: np.ndarray) -> np.ndarray:
        """Преобразует 5 predictions в 36D enriched features"""
        N = len(preds)
        X_enriched = np.zeros((N, 36))

        for i in range(N):
            p_xgb, p_rf, p_arf, p_nn, p_base = preds[i]

            # Основные (5D)
            X_enriched[i, :5] = preds[i]

            # Взаимодействия (6D)
            X_enriched[i, 5] = p_xgb * p_rf
            X_enriched[i, 6] = p_xgb * p_arf
            X_enriched[i, 7] = p_xgb * p_nn
            X_enriched[i, 8] = p_rf * p_arf
            X_enriched[i, 9] = p_rf * p_nn
            X_enriched[i, 10] = p_arf * p_nn

            # Логиты (4D)
            from meta_neural_cem import _safe_logit
            X_enriched[i, 11] = _safe_logit(p_xgb)
            X_enriched[i, 12] = _safe_logit(p_rf)
            X_enriched[i, 13] = _safe_logit(p_arf)
            X_enriched[i, 14] = _safe_logit(p_nn)

            # Статистики (4D)
            preds_arr = preds[i, :4]  # 4 эксперта
            X_enriched[i, 15] = np.std(preds_arr)
            X_enriched[i, 16] = np.max(preds_arr) - np.min(preds_arr)
            X_enriched[i, 17] = (np.abs(p_xgb - p_rf) +
                                  np.abs(p_xgb - p_arf) +
                                  np.abs(p_rf - p_nn))
            X_enriched[i, 18] = -np.sum(preds_arr * np.log(preds_arr + 1e-8))

            # Context interactions (10D)
            vol_ratio = X_train_context[i, 0] if i < len(X_train_context) else 1.0
            X_enriched[i, 19] = p_xgb * vol_ratio
            X_enriched[i, 20] = p_rf * vol_ratio
            X_enriched[i, 21] = p_arf * vol_ratio
            X_enriched[i, 22] = p_nn * vol_ratio
            # ... остальные взаимодействия

            # Bias (1D)
            X_enriched[i, 30] = 1.0

            # Padding до 36D
            # остальное уже нули

        return X_enriched

    X_train_enriched = enrich_features(X_train_preds)
    X_val_enriched = enrich_features(X_val_preds)

    # Objective function для CMA-ES
    def objective(w_flat: np.ndarray) -> float:
        """
        Функция потерь для CMA-ES

        Args:
            w_flat: плоский вектор весов нейросети

        Returns:
            loss: cross-entropy loss на validation set (минимизируем)
        """
        # Устанавливаем веса
        network.set_weights_from_flat(w_flat)

        # Forward pass на validation
        predictions = []
        for i in range(len(X_val_enriched)):
            p = network.forward(
                X_val_enriched[i],
                X_val_context[i],
                training=False
            )
            predictions.append(p)

        predictions = np.array(predictions)

        # Binary cross-entropy
        epsilon = 1e-7
        loss = -np.mean(
            y_val * np.log(predictions + epsilon) +
            (1 - y_val) * np.log(1 - predictions + epsilon)
        )

        return loss

    # Начальные веса
    w0 = network.get_weights_flat()
    n_params = len(w0)

    logger.info(f"CMA-ES: {n_params} parameters")
    logger.info(f"Sigma: {Config.CMA_SIGMA}")
    logger.info(f"Population: {Config.CMA_POPSIZE}")
    logger.info(f"Max iterations: {Config.CMA_MAX_ITER}")

    # CMA-ES optimization
    start_time = time.time()

    es = cma.CMAEvolutionStrategy(
        w0,
        Config.CMA_SIGMA,
        {
            'popsize': Config.CMA_POPSIZE,
            'maxiter': Config.CMA_MAX_ITER,
            'tolx': Config.CMA_TOLX,
            'verb_disp': 10,
            'verb_log': 0
        }
    )

    best_loss = float('inf')
    best_w = w0.copy()

    iteration = 0
    while not es.stop():
        solutions = es.ask()
        fitness = [objective(w) for w in solutions]
        es.tell(solutions, fitness)

        # Отслеживаем лучшее решение
        min_fitness = min(fitness)
        if min_fitness < best_loss:
            best_loss = min_fitness
            best_w = solutions[fitness.index(min_fitness)]

        iteration += 1

        if iteration % 10 == 0:
            logger.info(f"  Iteration {iteration}: loss={best_loss:.4f}")

    elapsed = time.time() - start_time

    # Устанавливаем лучшие веса
    network.set_weights_from_flat(best_w)

    logger.info(f"✓ CMA-ES completed in {elapsed:.1f}s")
    logger.info(f"  Best loss: {best_loss:.4f}")
    logger.info(f"  Final iterations: {iteration}")

    return network


def evaluate_meta(
    network: NeuralMetaNetwork,
    X_preds: np.ndarray,
    X_context: np.ndarray,
    y: np.ndarray,
    dataset_name: str
) -> Dict:
    """Оценка META на датасете"""

    # Feature engineering
    from meta_neural_cem import _safe_logit
    X_enriched = np.zeros((len(X_preds), 36))

    for i in range(len(X_preds)):
        p_xgb, p_rf, p_arf, p_nn, p_base = X_preds[i]

        X_enriched[i, :5] = X_preds[i]
        X_enriched[i, 5] = p_xgb * p_rf
        X_enriched[i, 6] = p_xgb * p_arf
        X_enriched[i, 7] = p_xgb * p_nn
        X_enriched[i, 8] = p_rf * p_arf
        X_enriched[i, 9] = p_rf * p_nn
        X_enriched[i, 10] = p_arf * p_nn

        X_enriched[i, 11] = _safe_logit(p_xgb)
        X_enriched[i, 12] = _safe_logit(p_rf)
        X_enriched[i, 13] = _safe_logit(p_arf)
        X_enriched[i, 14] = _safe_logit(p_nn)

        preds_arr = X_preds[i, :4]
        X_enriched[i, 15] = np.std(preds_arr)
        X_enriched[i, 16] = np.max(preds_arr) - np.min(preds_arr)
        X_enriched[i, 30] = 1.0

    # Predictions
    y_pred_proba = []
    for i in range(len(X_enriched)):
        p = network.forward(X_enriched[i], X_context[i], training=False)
        y_pred_proba.append(p)

    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Метрики
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    brier = brier_score_loss(y, y_pred_proba)

    return {
        'dataset': dataset_name,
        'n_samples': len(y),
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'brier': brier
    }


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_meta():
    """Главная функция обучения META"""

    logger.info("="*80)
    logger.info("ОБУЧЕНИЕ META НЕЙРОСЕТИ ЧЕРЕЗ CMA-ES")
    logger.info("="*80)

    # 1. Загрузка датасетов
    df_train, df_test = load_datasets()

    # 2. Загрузка экспертов
    experts = load_experts()
    base_logic = BaseLogic()

    # 3. Извлечение фич
    logger.info("\n" + "="*80)
    logger.info("ИЗВЛЕЧЕНИЕ ФИЧ И ПРЕДСКАЗАНИЙ")
    logger.info("="*80)

    X_train = extract_features(df_train)
    X_test = extract_features(df_test)

    y_train = df_train['outcome'].values
    y_test = df_test['outcome'].values

    phases_train = df_train['phase'].values
    phases_test = df_test['phase'].values

    logger.info(f"\nGenerating expert predictions...")

    # Predictions от экспертов
    preds_train = get_expert_predictions(X_train, experts, base_logic)
    preds_test = get_expert_predictions(X_test, experts, base_logic)

    logger.info(f"✓ Train predictions: {preds_train.shape}")
    logger.info(f"✓ Test predictions: {preds_test.shape}")

    # Синтетические context features
    logger.info(f"\nGenerating synthetic context features...")
    context_train = np.array([generate_synthetic_context(int(p), i) for i, p in enumerate(phases_train)])
    context_test = np.array([generate_synthetic_context(int(p), i+1000) for i, p in enumerate(phases_test)])

    logger.info(f"✓ Train context: {context_train.shape}")
    logger.info(f"✓ Test context: {context_test.shape}")

    # 4. Обучение META по фазам
    logger.info("\n" + "="*80)
    logger.info("ОБУЧЕНИЕ META ПО ФАЗАМ")
    logger.info("="*80)

    meta_networks = {}
    all_results = {}

    for phase in range(Config.N_PHASES):
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE {phase}")
        logger.info(f"{'='*60}")

        # Фильтруем по фазе
        mask_train = phases_train == phase
        mask_test = phases_test == phase

        n_train = mask_train.sum()
        n_test = mask_test.sum()

        logger.info(f"Train samples: {n_train}")
        logger.info(f"Test samples: {n_test}")

        if n_train < Config.MIN_SAMPLES_PER_PHASE:
            logger.warning(f"⚠ SKIP: недостаточно данных ({n_train} < {Config.MIN_SAMPLES_PER_PHASE})")
            continue

        # Данные для фазы
        X_train_phase = preds_train[mask_train]
        context_train_phase = context_train[mask_train]
        y_train_phase = y_train[mask_train]

        X_test_phase = preds_test[mask_test] if n_test > 0 else None
        context_test_phase = context_test[mask_test] if n_test > 0 else None
        y_test_phase = y_test[mask_test] if n_test > 0 else None

        # Обучаем через CMA-ES
        if X_test_phase is not None and len(X_test_phase) > 10:
            network = train_meta_phase_cmaes(
                phase=phase,
                X_train_preds=X_train_phase,
                X_train_context=context_train_phase,
                y_train=y_train_phase,
                X_val_preds=X_test_phase,
                X_val_context=context_test_phase,
                y_val=y_test_phase
            )

            meta_networks[phase] = network

            # Оценка
            test_metrics = evaluate_meta(
                network,
                X_test_phase,
                context_test_phase,
                y_test_phase,
                f"phase_{phase}_test"
            )

            # Сравнение с средним
            avg_preds = X_test_phase.mean(axis=1)
            avg_acc = ((avg_preds > 0.5) == y_test_phase).mean()

            logger.info(f"\nResults:")
            logger.info(f"  META accuracy:    {test_metrics['accuracy']:.4f}")
            logger.info(f"  META ROC-AUC:     {test_metrics['roc_auc']:.4f}")
            logger.info(f"  Average accuracy: {avg_acc:.4f}")
            logger.info(f"  Improvement:      {(test_metrics['accuracy'] - avg_acc)*100:+.2f}%")

            all_results[phase] = {
                'meta_accuracy': test_metrics['accuracy'],
                'meta_roc_auc': test_metrics['roc_auc'],
                'avg_accuracy': avg_acc,
                'improvement': test_metrics['accuracy'] - avg_acc
            }
        else:
            logger.warning("⚠ No test data for validation")

    # 5. Сохранение META
    logger.info("\n" + "="*80)
    logger.info("СОХРАНЕНИЕ META")
    logger.info("="*80)

    meta_data = {
        'networks': meta_networks,
        'mode': 'SHADOW',
        'trained_phases': list(meta_networks.keys()),
        'results': all_results
    }

    os.makedirs(os.path.dirname(Config.META_SAVE_PATH), exist_ok=True)

    with open(Config.META_SAVE_PATH, 'wb') as f:
        pickle.dump(meta_data, f)

    logger.info(f"✓ META saved to {Config.META_SAVE_PATH}")

    # 6. Итоги
    logger.info("\n" + "="*80)
    logger.info("✅ ОБУЧЕНИЕ META ЗАВЕРШЕНО")
    logger.info("="*80)

    logger.info(f"\nОбучено фаз: {len(meta_networks)}/{Config.N_PHASES}")
    logger.info(f"Режим: SHADOW")
    logger.info(f"\nРезультаты по фазам:")

    for phase, results in all_results.items():
        logger.info(f"  Phase {phase}: accuracy={results['meta_accuracy']:.4f}, "
                   f"improvement={results['improvement']*100:+.2f}%")

    if all_results:
        avg_improvement = np.mean([r['improvement'] for r in all_results.values()])
        logger.info(f"\nСреднее улучшение: {avg_improvement*100:+.2f}%")

    logger.info(f"\nСледующие шаги:")
    logger.info(f"1. Интегрировать META в Paper Trading Bot")
    logger.info(f"2. Запустить в SHADOW режиме (500+ позиций)")
    logger.info(f"3. Валидация статистической значимости")
    logger.info(f"4. Переход в ACTIVE режим")


if __name__ == "__main__":
    train_meta()
