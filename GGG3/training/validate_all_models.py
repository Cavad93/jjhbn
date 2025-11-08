#!/usr/bin/env python3
"""
Валидация всех моделей (4 эксперта + META) на test dataset

Полная валидация с метриками:
- Accuracy
- ROC-AUC
- Precision / Recall
- ECE (Expected Calibration Error)

Сравнение META с Average Ensemble.

Usage:
    python3 training/validate_all_models.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple
import logging

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импорт экспертов
from models.experts import XGBoostExpert, RandomForestExpert, NeuralNetworkExpert

# Импорт META
from meta_neural_cem import NeuralMetaNetwork, _safe_float, _sigmoid

# Метрики
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    brier_score_loss
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Конфигурация валидации"""
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models" / "saved"
    DATA_DIR = PROJECT_ROOT / "data" / "datasets"

    # Dataset
    TEST_DATASET = DATA_DIR / "binance_dataset_test.parquet"

    # META
    META_MODEL = MODELS_DIR / "meta_cmaes_model.pkl"

    # Metrics
    N_BINS_ECE = 10  # Количество бинов для ECE


# ============================================================================
# CALIBRATION
# ============================================================================

def calculate_calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE)

    Args:
        y_true: истинные метки (0 или 1)
        y_pred_proba: предсказанные вероятности (0-1)
        n_bins: количество бинов для калибровки

    Returns:
        ECE значение (0-1, меньше = лучше)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            avg_pred = y_pred_proba[mask].mean()
            avg_true = y_true[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(avg_pred - avg_true)

    return ece


# ============================================================================
# CONTEXT GENERATION (SYNTHETIC)
# ============================================================================

def generate_synthetic_context(phase: int, seed: int = None) -> np.ndarray:
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
# FEATURE ENGINEERING (для META)
# ============================================================================

def enrich_features(preds: np.ndarray) -> np.ndarray:
    """
    Feature engineering: 5 predictions → 36D enriched features

    Args:
        preds: (5,) массив [p_xgb, p_rf, p_arf, p_nn, p_base]

    Returns:
        enriched: (36,) enriched features
    """
    p_xgb, p_rf, p_arf, p_nn, p_base = preds

    features = []

    # Основные (5D)
    features.extend([p_xgb, p_rf, p_arf, p_nn, p_base])

    # Взаимодействия (6D)
    features.append(p_xgb * p_rf)
    features.append(p_xgb * p_arf)
    features.append(p_xgb * p_nn)
    features.append(p_rf * p_arf)
    features.append(p_rf * p_nn)
    features.append(p_arf * p_nn)

    # Логиты (4D)
    epsilon = 1e-7
    features.append(np.log((p_xgb + epsilon) / (1 - p_xgb + epsilon)))
    features.append(np.log((p_rf + epsilon) / (1 - p_rf + epsilon)))
    features.append(np.log((p_arf + epsilon) / (1 - p_arf + epsilon)))
    features.append(np.log((p_nn + epsilon) / (1 - p_nn + epsilon)))

    # Статистики (4D)
    features.append(np.std([p_xgb, p_rf, p_arf, p_nn]))
    features.append(np.max([p_xgb, p_rf, p_arf, p_nn]) - np.min([p_xgb, p_rf, p_arf, p_nn]))
    features.append(np.max([abs(p_xgb - p_rf), abs(p_xgb - p_arf), abs(p_xgb - p_nn)]))

    probs = np.array([p_xgb, p_rf, p_arf, p_nn]) + epsilon
    entropy = -np.sum(probs * np.log(probs))
    features.append(entropy)

    # Padding до 36D (17 элементов пока)
    # Добавим еще взаимодействия с BASE
    features.append(p_xgb * p_base)
    features.append(p_rf * p_base)
    features.append(p_arf * p_base)
    features.append(p_nn * p_base)

    # Разности с BASE (4D)
    features.append(p_xgb - p_base)
    features.append(p_rf - p_base)
    features.append(p_arf - p_base)
    features.append(p_nn - p_base)

    # Квадраты (4D)
    features.append(p_xgb ** 2)
    features.append(p_rf ** 2)
    features.append(p_arf ** 2)
    features.append(p_nn ** 2)

    # Среднее и медиана (2D)
    features.append(np.mean([p_xgb, p_rf, p_arf, p_nn]))
    features.append(np.median([p_xgb, p_rf, p_arf, p_nn]))

    # Padding до 36D (если меньше)
    current_len = len(features)
    if current_len < 36:
        features.extend([0.0] * (36 - current_len))

    return np.array(features[:36], dtype=float)


# ============================================================================
# BASE LOGIC (FALLBACK)
# ============================================================================

class BaseLogic:
    """
    Заглушка для BASE logic (prob_up_down_at_time_multiTF)

    В production: полная реализация из BINANCE_BOT_MASTER_PLAN.md
    """
    def predict(self, p_xgb: float, p_rf: float, p_arf: float, p_nn: float) -> float:
        """Возвращает среднее экспертов"""
        preds = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
        if not preds:
            return 0.5
        return float(np.mean(preds))


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_experts() -> Dict:
    """Загружает всех 4 экспертов"""
    logger.info("="*80)
    logger.info("ЗАГРУЗКА ЭКСПЕРТОВ")
    logger.info("="*80)

    experts = {}

    # XGBoost
    xgb_path = Config.MODELS_DIR / "xgb_expert.pkl"
    if xgb_path.exists():
        experts['xgb'] = XGBoostExpert.load(str(xgb_path))
        logger.info(f"✓ XGBoost loaded: {experts['xgb'].train_samples} samples")
    else:
        logger.warning("⚠ XGBoost not found - using fallback")
        experts['xgb'] = None

    # Random Forest
    rf_path = Config.MODELS_DIR / "rf_expert.pkl"
    if rf_path.exists():
        experts['rf'] = RandomForestExpert.load(str(rf_path))
        logger.info(f"✓ RandomForest loaded: {experts['rf'].train_samples} samples")
    else:
        logger.warning("⚠ RandomForest not found - using fallback")
        experts['rf'] = None

    # Adaptive RF (заглушка)
    logger.info("⚠ AdaptiveRF: using fallback (returns 0.5)")
    experts['arf'] = None

    # Neural Network
    nn_path = Config.MODELS_DIR / "nn_expert.pkl"
    if nn_path.exists():
        experts['nn'] = NeuralNetworkExpert.load(str(nn_path))
        logger.info(f"✓ NeuralNet loaded: {experts['nn'].train_samples} samples")
    else:
        logger.warning("⚠ NeuralNet not found - using fallback")
        experts['nn'] = None

    loaded = sum(1 for e in experts.values() if e is not None)
    logger.info(f"\nЗагружено экспертов: {loaded}/4")

    return experts


def load_meta() -> Dict:
    """Загружает META модель"""
    logger.info("\n" + "="*80)
    logger.info("ЗАГРУЗКА META")
    logger.info("="*80)

    if not Config.META_MODEL.exists():
        logger.error(f"META model not found: {Config.META_MODEL}")
        return None

    with open(Config.META_MODEL, 'rb') as f:
        meta_data = pickle.load(f)

    logger.info(f"✓ META loaded")
    logger.info(f"  Mode: {meta_data['mode']}")
    logger.info(f"  Trained phases: {meta_data['trained_phases']}")

    return meta_data


# ============================================================================
# VALIDATION
# ============================================================================

def validate_all_models():
    """Полная валидация всех моделей на test set"""

    logger.info("\n" + "="*80)
    logger.info("ВАЛИДАЦИЯ ВСЕХ МОДЕЛЕЙ НА TEST DATASET")
    logger.info("="*80)

    # Загружаем модели
    experts = load_experts()
    meta_data = load_meta()

    if meta_data is None:
        logger.error("Cannot validate without META model")
        return

    meta_networks = meta_data['networks']
    base_logic = BaseLogic()

    # Загружаем test dataset
    logger.info("\n" + "="*80)
    logger.info("ЗАГРУЗКА TEST DATASET")
    logger.info("="*80)

    if not Config.TEST_DATASET.exists():
        logger.error(f"Test dataset not found: {Config.TEST_DATASET}")
        return

    df_test = pd.read_parquet(Config.TEST_DATASET)
    logger.info(f"Test samples: {len(df_test)}\n")

    # Извлекаем features
    feature_cols = [col for col in df_test.columns if col.startswith('features_')]
    X_test = df_test[feature_cols].values
    y_test = df_test['outcome'].values
    phases_test = df_test['phase'].values

    logger.info(f"Features: {X_test.shape}")
    logger.info(f"Outcomes: {y_test.shape}")
    logger.info(f"Phases: {np.unique(phases_test)}\n")

    # Результаты
    results = {
        'XGBoost': [],
        'RandomForest': [],
        'ARF': [],
        'NN': [],
        'Average': [],
        'META': []
    }

    # Валидация
    logger.info("="*80)
    logger.info("ГЕНЕРАЦИЯ ПРЕДСКАЗАНИЙ")
    logger.info("="*80)

    for idx in range(len(df_test)):
        features = X_test[idx].reshape(1, -1)
        phase = int(phases_test[idx])
        outcome = int(y_test[idx])

        # Predictions экспертов
        if experts['xgb'] is not None:
            p_xgb = float(experts['xgb'].predict_proba(features)[0])
        else:
            p_xgb = 0.5

        if experts['rf'] is not None:
            p_rf = float(experts['rf'].predict_proba(features)[0])
        else:
            p_rf = 0.5

        # ARF fallback
        p_arf = 0.5

        if experts['nn'] is not None:
            p_nn = float(experts['nn'].predict_proba(features)[0])
        else:
            p_nn = 0.5

        results['XGBoost'].append((p_xgb, outcome))
        results['RandomForest'].append((p_rf, outcome))
        results['ARF'].append((p_arf, outcome))
        results['NN'].append((p_nn, outcome))

        # Average
        p_avg = np.mean([p_xgb, p_rf, p_arf, p_nn])
        results['Average'].append((p_avg, outcome))

        # BASE logic
        p_base = base_logic.predict(p_xgb, p_rf, p_arf, p_nn)

        # META prediction
        if phase in meta_networks:
            # Генерируем context
            context = generate_synthetic_context(phase, seed=idx)

            # Feature engineering
            preds = np.array([p_xgb, p_rf, p_arf, p_nn, p_base])
            x_enriched = enrich_features(preds)

            # Forward pass
            meta_network = meta_networks[phase]
            p_meta = meta_network.forward(x_enriched, context, training=False)
        else:
            # Фаза не обучена - используем average
            p_meta = p_avg

        results['META'].append((p_meta, outcome))

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(df_test)} samples")

    logger.info(f"✓ Все {len(df_test)} примеров обработаны\n")

    # Вычисляем метрики
    logger.info("="*80)
    logger.info(f"{'Model':<15} | {'Accuracy':>8} | {'ROC-AUC':>8} | {'Precision':>9} | {'Recall':>8} | {'ECE':>8}")
    logger.info("="*80)

    metrics_table = []

    for model_name, preds in results.items():
        y_pred_proba = np.array([p[0] for p in preds])
        y_true = np.array([p[1] for p in preds])
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Accuracy
        acc = accuracy_score(y_true, y_pred)

        # ROC-AUC
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            auc = 0.5  # Если все предсказания одинаковые

        # Precision / Recall
        prec, rec, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        # ECE
        ece = calculate_calibration_error(y_true, y_pred_proba, n_bins=Config.N_BINS_ECE)

        logger.info(f"{model_name:<15} | {acc:8.4f} | {auc:8.4f} | {prec:9.4f} | {rec:8.4f} | {ece:8.4f}")

        metrics_table.append({
            'model': model_name,
            'accuracy': acc,
            'roc_auc': auc,
            'precision': prec,
            'recall': rec,
            'ece': ece
        })

    logger.info("="*80)

    # Сравнение META с Average
    meta_acc = metrics_table[-1]['accuracy']  # META последний
    avg_acc = metrics_table[-2]['accuracy']   # Average предпоследний

    improvement = meta_acc - avg_acc

    logger.info("\n" + "="*80)
    logger.info("META vs Average:")
    logger.info("="*80)
    logger.info(f"  META accuracy:    {meta_acc:.4f}")
    logger.info(f"  Average accuracy: {avg_acc:.4f}")
    logger.info(f"  Improvement:      {improvement:+.4f} ({improvement*100:+.2f}%)")
    logger.info("="*80)

    # Проверка значимости улучшения
    if meta_acc > avg_acc and improvement > 0.02:
        logger.info("\n✅ ALL MODELS VALIDATED")
        logger.info("✅ META SHOWS SIGNIFICANT IMPROVEMENT")
        logger.info("\nРекомендация: можно переводить META в ACTIVE режим")
    else:
        logger.info("\n⚠️  WARNING: META does not show significant improvement")
        logger.info("   Consider additional training or staying with average ensemble")
        logger.info("\nРекомендация: собрать больше данных, продолжить SHADOW режим")

    logger.info("\n" + "="*80)
    logger.info("ВАЛИДАЦИЯ ЗАВЕРШЕНА")
    logger.info("="*80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    validate_all_models()
