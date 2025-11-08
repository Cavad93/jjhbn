#!/usr/bin/env python3
"""
Обучение всех ML экспертов

Скрипт для обучения 4 экспертов на тренировочном датасете:
1. XGBoost
2. Random Forest
3. Adaptive Random Forest
4. Neural Network

После обучения модели сохраняются в models/saved/

Usage:
    python3 models/train_all_experts.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
from typing import Dict, List

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.experts import XGBoostExpert, RandomForestExpert, AdaptiveRFExpert, NeuralNetworkExpert

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Конфигурация обучения"""

    # Пути к данным (относительно корня проекта)
    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).parent.parent

    TRAIN_DATASET = str(_PROJECT_ROOT / "data" / "datasets" / "binance_dataset_train.parquet")
    TEST_DATASET = str(_PROJECT_ROOT / "data" / "datasets" / "binance_dataset_test.parquet")

    # Директория для сохранения моделей
    MODELS_DIR = str(_PROJECT_ROOT / "models" / "saved")

    # Параметры экспертов
    XGB_PARAMS = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }

    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt'
    }

    ARF_PARAMS = {
        'n_models': 10,
        'max_depth': 10,
        'drift_detector': True
    }

    NN_PARAMS = {
        'input_dim': 68,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'max_epochs': 50,
        'early_stopping_patience': 5
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(filepath: str) -> tuple:
    """
    Загружает датасет и разделяет на X и y

    Returns:
        (X, y, metadata)
    """
    print(f"Загрузка датасета: {filepath}")

    df = pd.read_parquet(filepath)

    print(f"  Размер: {df.shape}")
    print(f"  Колонки: {df.columns.tolist()}")

    # Извлекаем фичи (68D numpy array)
    # Фичи сохранены как отдельные колонки: features_0, features_1, ..., features_67
    feature_cols = [col for col in df.columns if col.startswith('features_')]
    if len(feature_cols) > 0:
        X = df[feature_cols].values
    elif 'features' in df.columns:
        # Альтернативный формат: фичи сохранены как array в одной колонке
        X = np.vstack(df['features'].values)
    else:
        raise ValueError("Колонки с фичами не найдены в датасете!")

    # Таргет
    y = df['outcome'].values

    # Метаданные
    metadata = {
        'n_samples': len(df),
        'n_features': X.shape[1],
        'class_distribution': {
            '0 (SL)': (y == 0).sum(),
            '1 (TP)': (y == 1).sum()
        },
        'direction_distribution': df['direction'].value_counts().to_dict() if 'direction' in df.columns else {},
        'phase_distribution': df['phase'].value_counts().to_dict() if 'phase' in df.columns else {}
    }

    print(f"  Фичи: {X.shape}")
    print(f"  Таргет: {y.shape}")
    print(f"  Класс 0: {metadata['class_distribution']['0 (SL)']} ({metadata['class_distribution']['0 (SL)']/len(y)*100:.1f}%)")
    print(f"  Класс 1: {metadata['class_distribution']['1 (TP)']} ({metadata['class_distribution']['1 (TP)']/len(y)*100:.1f}%)")

    return X, y, metadata


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost(X_train, y_train, X_val, y_val) -> XGBoostExpert:
    """Обучение XGBoost эксперта"""
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ XGBOOST")
    print("="*80)

    expert = XGBoostExpert(**Config.XGB_PARAMS)

    start_time = time.time()
    expert.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"\n✅ Обучение завершено за {elapsed:.1f}s")
    print(f"   Примеров: {expert.train_samples}")
    print(f"   Деревьев: {expert.model.num_boosted_rounds()}")

    # Валидация
    y_pred_proba = expert.predict_proba(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = (y_pred == y_val).mean()

    print(f"\nВалидация:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Средняя вероятность: {y_pred_proba.mean():.3f}")

    return expert


def train_random_forest(X_train, y_train, X_val, y_val) -> RandomForestExpert:
    """Обучение Random Forest эксперта"""
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ RANDOM FOREST")
    print("="*80)

    expert = RandomForestExpert(**Config.RF_PARAMS)

    start_time = time.time()
    expert.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"\n✅ Обучение завершено за {elapsed:.1f}s")
    print(f"   Примеров: {expert.train_samples}")
    print(f"   Деревьев: {len(expert.model.estimators_)}")

    # Валидация
    y_pred_proba = expert.predict_proba(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = (y_pred == y_val).mean()

    print(f"\nВалидация:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Средняя вероятность: {y_pred_proba.mean():.3f}")

    return expert


def train_adaptive_rf(X_train, y_train, X_val, y_val) -> AdaptiveRFExpert:
    """Обучение Adaptive Random Forest эксперта"""
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ ADAPTIVE RANDOM FOREST")
    print("="*80)
    print("⚠️  Онлайн обучение - может занять время...")

    expert = AdaptiveRFExpert(**Config.ARF_PARAMS)

    start_time = time.time()
    expert.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"\n✅ Обучение завершено за {elapsed:.1f}s")
    print(f"   Примеров: {expert.train_samples}")
    print(f"   Моделей: {expert.n_models}")

    # Валидация
    y_pred_proba = expert.predict_proba(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = (y_pred == y_val).mean()

    print(f"\nВалидация:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Средняя вероятность: {y_pred_proba.mean():.3f}")

    return expert


def train_neural_network(X_train, y_train, X_val, y_val) -> NeuralNetworkExpert:
    """Обучение Neural Network эксперта"""
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ NEURAL NETWORK")
    print("="*80)

    expert = NeuralNetworkExpert(**Config.NN_PARAMS)

    start_time = time.time()
    expert.fit(X_train, y_train, X_val, y_val)
    elapsed = time.time() - start_time

    print(f"\n✅ Обучение завершено за {elapsed:.1f}s")
    print(f"   Примеров: {expert.train_samples}")
    print(f"   Эпох: {expert.train_epochs}")
    print(f"   Best loss: {expert.best_loss:.4f}")

    # Валидация
    y_pred_proba = expert.predict_proba(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = (y_pred == y_val).mean()

    print(f"\nВалидация:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Средняя вероятность: {y_pred_proba.mean():.3f}")

    return expert


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Главная функция"""
    print("="*80)
    print("ОБУЧЕНИЕ ВСЕХ ML ЭКСПЕРТОВ")
    print("="*80)
    print()

    # Создаем директорию для моделей
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    # 1. Загружаем данные
    print("ЭТАП 1: ЗАГРУЗКА ДАННЫХ")
    print("-"*80)

    X_train, y_train, train_meta = load_dataset(Config.TRAIN_DATASET)
    X_test, y_test, test_meta = load_dataset(Config.TEST_DATASET)

    print(f"\n✅ Данные загружены")
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")

    # 2. Обучаем экспертов
    print("\n\nЭТАП 2: ОБУЧЕНИЕ ЭКСПЕРТОВ")
    print("-"*80)

    experts = {}

    # XGBoost
    try:
        experts['xgb'] = train_xgboost(X_train, y_train, X_test, y_test)
        save_path = f"{Config.MODELS_DIR}/xgb_expert.pkl"
        experts['xgb'].save(save_path)
        print(f"   💾 Сохранено: {save_path}")
    except Exception as e:
        print(f"   ❌ Ошибка обучения XGBoost: {e}")
        experts['xgb'] = None

    # Random Forest
    try:
        experts['rf'] = train_random_forest(X_train, y_train, X_test, y_test)
        save_path = f"{Config.MODELS_DIR}/rf_expert.pkl"
        experts['rf'].save(save_path)
        print(f"   💾 Сохранено: {save_path}")
    except Exception as e:
        print(f"   ❌ Ошибка обучения RF: {e}")
        experts['rf'] = None

    # Adaptive RF
    try:
        experts['arf'] = train_adaptive_rf(X_train, y_train, X_test, y_test)
        save_path = f"{Config.MODELS_DIR}/arf_expert.pkl"
        experts['arf'].save(save_path)
        print(f"   💾 Сохранено: {save_path}")
        print(f"   ⚠️  ARF требует retraining после загрузки!")
    except Exception as e:
        print(f"   ❌ Ошибка обучения ARF: {e}")
        experts['arf'] = None

    # Neural Network
    try:
        experts['nn'] = train_neural_network(X_train, y_train, X_test, y_test)
        save_path = f"{Config.MODELS_DIR}/nn_expert.pkl"
        experts['nn'].save(save_path)
        print(f"   💾 Сохранено: {save_path}")
    except Exception as e:
        print(f"   ❌ Ошибка обучения NN: {e}")
        experts['nn'] = None

    # 3. Итоговая оценка
    print("\n\nЭТАП 3: ИТОГОВАЯ ОЦЕНКА НА TEST SET")
    print("-"*80)

    results = {}

    for name, expert in experts.items():
        if expert is None:
            continue

        # Предсказания на test set
        y_pred_proba = expert.predict_proba(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Метрики
        accuracy = (y_pred == y_test).mean()
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_proba': y_pred_proba.mean()
        }

        print(f"\n{name.upper()}:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1 Score:  {f1*100:.2f}%")
        print(f"   Avg proba: {y_pred_proba.mean():.3f}")

    # 4. Сравнение экспертов
    print("\n\nЭТАП 4: СРАВНЕНИЕ ЭКСПЕРТОВ")
    print("-"*80)

    if results:
        # Сортируем по accuracy
        sorted_experts = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        print("\nРанжирование по Accuracy:")
        for i, (name, metrics) in enumerate(sorted_experts, 1):
            print(f"   {i}. {name.upper()}: {metrics['accuracy']*100:.2f}%")

        # Лучший эксперт
        best_expert = sorted_experts[0][0]
        print(f"\n🏆 Лучший эксперт: {best_expert.upper()}")
        print(f"   Accuracy: {results[best_expert]['accuracy']*100:.2f}%")

    # 5. Финальный отчет
    print("\n\n" + "="*80)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*80)

    print(f"\nОбучено экспертов: {len([e for e in experts.values() if e is not None])}/4")
    print(f"\nМодели сохранены в: {Config.MODELS_DIR}")

    print("\nСледующие шаги:")
    print("1. Обучить META нейросеть на предсказаниях экспертов")
    print("2. Протестировать ensemble на test set")
    print("3. Интегрировать с Paper Trading ботом")

    return experts, results


if __name__ == "__main__":
    experts, results = main()
