#!/usr/bin/env python3
"""
Обучение 4 ML экспертов на Binance датасете с TP/SL targets

Особенности:
- Раздельные модели по 6 фазам рынка
- Калибровка вероятностей
- Валидация на test set
- Метрики: Accuracy, ROC-AUC, Brier, ECE

Usage:
    python3 training/train_binance_experts.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
import logging

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импорт экспертов
from models.experts import XGBoostExpert, RandomForestExpert, AdaptiveRFExpert, NeuralNetworkExpert

# Метрики
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

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
    """Конфигурация обучения"""

    # Пути к данным
    TRAIN_DATASET = "/home/user/jjhbn/GGG3/data/datasets/binance_dataset_train.parquet"
    TEST_DATASET = "/home/user/jjhbn/GGG3/data/datasets/binance_dataset_test.parquet"

    # Пути к моделям
    MODELS_DIR = "/home/user/jjhbn/GGG3/models/saved"

    # Параметры обучения
    MIN_SAMPLES_PER_PHASE = 50  # Минимум samples для обучения фазы
    N_PHASES = 6  # Количество фаз рынка

    # Калибровка
    CALIBRATE_RF = True  # Калибровать Random Forest
    CALIBRATE_NN = True  # Temperature scaling для NN


# ============================================================================
# CALIBRATION ERROR
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
# DATA LOADING
# ============================================================================

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает train и test датасеты

    Returns:
        (df_train, df_test)
    """
    logger.info("="*80)
    logger.info("ЗАГРУЗКА ДАТАСЕТОВ")
    logger.info("="*80)

    logger.info(f"Loading train dataset: {Config.TRAIN_DATASET}")
    df_train = pd.read_parquet(Config.TRAIN_DATASET)

    logger.info(f"Loading test dataset: {Config.TEST_DATASET}")
    df_test = pd.read_parquet(Config.TEST_DATASET)

    logger.info(f"\nTrain size: {len(df_train)}")
    logger.info(f"Test size: {len(df_test)}")

    # Статистика по классам
    logger.info(f"\nClass distribution (train):")
    class_counts = df_train['outcome'].value_counts()
    for outcome, count in class_counts.items():
        pct = count / len(df_train) * 100
        label = "TP" if outcome == 1 else "SL"
        logger.info(f"  {label} ({outcome}): {count} ({pct:.1f}%)")

    # Статистика по фазам
    logger.info(f"\nPhase distribution (train):")
    phase_counts = df_train['phase'].value_counts().sort_index()
    for phase, count in phase_counts.items():
        pct = count / len(df_train) * 100
        logger.info(f"  Phase {phase}: {count} ({pct:.1f}%)")

    # Статистика по направлениям
    logger.info(f"\nDirection distribution (train):")
    dir_counts = df_train['direction'].value_counts()
    for direction, count in dir_counts.items():
        pct = count / len(df_train) * 100
        logger.info(f"  {direction}: {count} ({pct:.1f}%)")

    return df_train, df_test


def extract_features_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Извлекает фичи и target из датафрейма

    Args:
        df: датафрейм с features_0, features_1, ..., outcome

    Returns:
        (X, y) - фичи и таргеты
    """
    # Извлекаем фичи
    feature_cols = [col for col in df.columns if col.startswith('features_')]
    X = df[feature_cols].values

    # Таргет
    y = df['outcome'].values

    return X, y


# ============================================================================
# EXPERT TRAINING
# ============================================================================

def train_expert_by_phases(
    expert,
    expert_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Dict:
    """
    Обучает эксперта раздельно по фазам

    Args:
        expert: экземпляр эксперта
        expert_name: название ('xgb', 'rf', 'arf', 'nn')
        df_train: train датафрейм
        df_test: test датафрейм

    Returns:
        dict с результатами
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"ОБУЧЕНИЕ {expert_name.upper()} EXPERT")
    logger.info(f"{'='*80}")

    results = {
        'expert': expert_name,
        'phases': {},
        'overall_train': {},
        'overall_test': {}
    }

    # Получаем все уникальные фазы
    all_phases = sorted(df_train['phase'].unique())
    logger.info(f"\nPhases to train: {all_phases}")

    trained_phases = []

    # Обучаем по фазам
    for phase in all_phases:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Phase {phase}")
        logger.info(f"{'─'*60}")

        # Фильтруем данные по фазе
        df_phase_train = df_train[df_train['phase'] == phase]
        df_phase_test = df_test[df_test['phase'] == phase]

        # Проверяем достаточно ли данных
        if len(df_phase_train) < Config.MIN_SAMPLES_PER_PHASE:
            logger.warning(f"  ⚠ SKIP: недостаточно данных ({len(df_phase_train)} < {Config.MIN_SAMPLES_PER_PHASE})")
            results['phases'][phase] = {'skipped': True, 'reason': 'insufficient_data'}
            continue

        logger.info(f"  Train samples: {len(df_phase_train)}")
        logger.info(f"  Test samples: {len(df_phase_test)}")

        # Извлекаем фичи и target
        X_train, y_train = extract_features_and_target(df_phase_train)
        X_test, y_test = extract_features_and_target(df_phase_test) if len(df_phase_test) > 0 else (None, None)

        # Обучаем
        try:
            logger.info("  Training...")
            start_time = time.time()

            expert.fit(X_train, y_train)

            elapsed = time.time() - start_time
            logger.info(f"  ✓ Trained in {elapsed:.2f}s")

            trained_phases.append(phase)

            # Валидация на train
            train_metrics = evaluate_expert(expert, X_train, y_train, "train")
            results['phases'][phase] = {'train': train_metrics}

            # Валидация на test (если есть данные)
            if X_test is not None and len(X_test) > 0:
                test_metrics = evaluate_expert(expert, X_test, y_test, "test")
                results['phases'][phase]['test'] = test_metrics

                logger.info(f"  Test metrics:")
                logger.info(f"    Accuracy: {test_metrics['accuracy']:.4f}")
                logger.info(f"    ROC-AUC:  {test_metrics['roc_auc']:.4f}")
                logger.info(f"    Brier:    {test_metrics['brier']:.4f}")
                logger.info(f"    ECE:      {test_metrics['ece']:.4f}")
            else:
                logger.warning(f"  ⚠ No test data for phase {phase}")

        except Exception as e:
            logger.error(f"  ✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            results['phases'][phase] = {'error': str(e)}

    logger.info(f"\nTrained phases: {trained_phases}")

    # Общая валидация на всем train set
    logger.info(f"\n{'─'*60}")
    logger.info(f"Overall validation (all phases)")
    logger.info(f"{'─'*60}")

    X_train_all, y_train_all = extract_features_and_target(df_train)
    X_test_all, y_test_all = extract_features_and_target(df_test)

    train_metrics_all = evaluate_expert(expert, X_train_all, y_train_all, "train")
    test_metrics_all = evaluate_expert(expert, X_test_all, y_test_all, "test")

    results['overall_train'] = train_metrics_all
    results['overall_test'] = test_metrics_all

    logger.info(f"\nOverall test metrics:")
    logger.info(f"  Accuracy: {test_metrics_all['accuracy']:.4f}")
    logger.info(f"  ROC-AUC:  {test_metrics_all['roc_auc']:.4f}")
    logger.info(f"  Brier:    {test_metrics_all['brier']:.4f}")
    logger.info(f"  ECE:      {test_metrics_all['ece']:.4f}")

    return results


def evaluate_expert(expert, X: np.ndarray, y: np.ndarray, dataset_name: str) -> Dict:
    """
    Оценивает эксперта на датасете

    Args:
        expert: экземпляр эксперта
        X: фичи
        y: таргеты
        dataset_name: название датасета ('train' или 'test')

    Returns:
        dict с метриками
    """
    # Предсказания
    y_pred_proba = expert.predict_proba(X)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Метрики
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    brier = brier_score_loss(y, y_pred_proba)
    ece = calculate_calibration_error(y, y_pred_proba)

    return {
        'dataset': dataset_name,
        'n_samples': len(y),
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'brier': brier,
        'ece': ece
    }


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_all_experts():
    """Главная функция обучения всех экспертов"""

    logger.info("="*80)
    logger.info("ОБУЧЕНИЕ 4 ML ЭКСПЕРТОВ НА BINANCE ДАТАСЕТЕ")
    logger.info("="*80)

    # 1. Загружаем датасеты
    df_train, df_test = load_datasets()

    # 2. Создаем экспертов
    logger.info(f"\n{'='*80}")
    logger.info("ИНИЦИАЛИЗАЦИЯ ЭКСПЕРТОВ")
    logger.info(f"{'='*80}")

    experts = {
        'xgb': XGBoostExpert(),
        'rf': RandomForestExpert(),
        'nn': NeuralNetworkExpert()
    }

    # ARF пропускаем пока (проблемы с загрузкой)
    logger.info("\n⚠ Adaptive RF временно отключен (требует доработки)")
    logger.info(f"Обучаем: {list(experts.keys())}")

    # 3. Обучаем каждого эксперта
    all_results = {}

    for expert_name, expert in experts.items():
        results = train_expert_by_phases(
            expert=expert,
            expert_name=expert_name,
            df_train=df_train,
            df_test=df_test
        )

        all_results[expert_name] = results

        # Сохраняем модель
        save_path = f"{Config.MODELS_DIR}/{expert_name}_expert.pkl"
        logger.info(f"\n{'─'*60}")
        logger.info(f"Saving {expert_name.upper()} to {save_path}")

        try:
            expert.save(save_path)
            logger.info(f"  ✓ Saved successfully")
        except Exception as e:
            logger.error(f"  ✗ Save failed: {e}")

    # 4. Сводная таблица результатов
    logger.info(f"\n{'='*80}")
    logger.info("СВОДКА РЕЗУЛЬТАТОВ")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Expert':<10} {'Accuracy':<10} {'ROC-AUC':<10} {'Brier':<10} {'ECE':<10}")
    logger.info("─"*50)

    for expert_name, results in all_results.items():
        test_metrics = results['overall_test']
        logger.info(
            f"{expert_name.upper():<10} "
            f"{test_metrics['accuracy']:<10.4f} "
            f"{test_metrics['roc_auc']:<10.4f} "
            f"{test_metrics['brier']:<10.4f} "
            f"{test_metrics['ece']:<10.4f}"
        )

    # 5. Итоги
    logger.info(f"\n{'='*80}")
    logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    logger.info(f"{'='*80}")

    logger.info(f"\nОбучено экспертов: {len(experts)}/4")
    logger.info(f"Модели сохранены в: {Config.MODELS_DIR}")

    logger.info(f"\nСледующие шаги:")
    logger.info(f"1. Переобучить META на новых экспертах")
    logger.info(f"2. Интегрировать в Paper Trading Bot")
    logger.info(f"3. Запустить тестирование")

    return all_results


if __name__ == "__main__":
    train_all_experts()
