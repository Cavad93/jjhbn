#!/usr/bin/env python3
"""
Обучение META нейросети для Binance Bot

META нейросеть комбинирует предсказания 4 экспертов:
- XGBoost
- Random Forest
- Adaptive Random Forest
- Neural Network

Обучение через CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

Usage:
    python3 models/train_meta.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
import pickle
import logging

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импорт экспертов
from models.experts import XGBoostExpert, RandomForestExpert, AdaptiveRFExpert, NeuralNetworkExpert

# Импорт META (упрощенная версия без CMA-ES для начала)
try:
    import cma
    HAVE_CMA = True
except ImportError:
    print("WARNING: CMA-ES not available. Install: pip install cma")
    HAVE_CMA = False

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

    # Пути к данным
    TRAIN_DATASET = "/home/user/jjhbn/GGG3/data/datasets/binance_dataset_train.parquet"
    TEST_DATASET = "/home/user/jjhbn/GGG3/data/datasets/binance_dataset_test.parquet"

    # Пути к моделям
    MODELS_DIR = "/home/user/jjhbn/GGG3/models/saved"
    META_SAVE_PATH = "/home/user/jjhbn/GGG3/models/saved/meta_model.pkl"

    # Параметры META
    META_INPUT_SIZE = 4  # 4 эксперта (без базовой логики пока)
    META_HIDDEN_SIZE = 8

    # CMA-ES параметры
    CMA_SIGMA = 0.5
    CMA_POPSIZE = 20
    CMA_MAX_ITER = 100


# ============================================================================
# SIMPLE META NETWORK (без CMA-ES - для начала)
# ============================================================================

class SimpleMetaNetwork:
    """
    Простая META нейросеть для комбинирования экспертов

    Архитектура: 4 → 8 → 1
    - Input: 4 вероятности от экспертов
    - Hidden: 8 нейронов с ReLU
    - Output: финальная вероятность

    Обучение: простой gradient descent (можно заменить на CMA-ES)
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 8):
        """Инициализация сети"""
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier инициализация
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(1)

        self.is_trained = False
        self.train_samples = 0
        self.train_accuracy = 0.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            X: (N, 4) - предсказания экспертов

        Returns:
            (N,) - финальные вероятности
        """
        # Hidden layer
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU

        # Output layer
        z2 = a1 @ self.W2 + self.b2

        # Sigmoid
        proba = 1 / (1 + np.exp(-np.clip(z2, -20, 20)))

        return proba.flatten()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias для forward"""
        return self.forward(X)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100, learning_rate: float = 0.01):
        """
        Обучение сети через gradient descent

        Args:
            X_train: (N, 4) - предсказания экспертов
            y_train: (N,) - таргеты (0 или 1)
            X_val: валидационные данные
            y_val: валидационные таргеты
            epochs: количество эпох
            learning_rate: learning rate
        """
        logger.info(f"Training META network...")
        logger.info(f"  Train samples: {len(X_train)}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Learning rate: {learning_rate}")

        best_val_acc = 0.0

        for epoch in range(epochs):
            # Forward pass
            # Hidden layer
            z1 = X_train @ self.W1 + self.b1
            a1 = np.maximum(0, z1)  # ReLU

            # Output layer
            z2 = a1 @ self.W2 + self.b2
            proba = 1 / (1 + np.exp(-np.clip(z2, -20, 20)))
            proba = proba.flatten()

            # Loss (binary cross-entropy)
            epsilon = 1e-7
            loss = -np.mean(
                y_train * np.log(proba + epsilon) +
                (1 - y_train) * np.log(1 - proba + epsilon)
            )

            # Accuracy
            y_pred = (proba > 0.5).astype(int)
            train_acc = (y_pred == y_train).mean()

            # Backward pass
            # Output layer gradient
            d_loss = proba - y_train
            d_loss = d_loss.reshape(-1, 1)

            d_W2 = a1.T @ d_loss / len(X_train)
            d_b2 = np.mean(d_loss, axis=0)

            # Hidden layer gradient
            d_a1 = d_loss @ self.W2.T
            d_z1 = d_a1 * (z1 > 0)  # ReLU derivative

            d_W1 = X_train.T @ d_z1 / len(X_train)
            d_b1 = np.mean(d_z1, axis=0)

            # Update weights
            self.W1 -= learning_rate * d_W1
            self.b1 -= learning_rate * d_b1
            self.W2 -= learning_rate * d_W2
            self.b2 -= learning_rate * d_b2

            # Validation
            if X_val is not None and y_val is not None:
                val_proba = self.forward(X_val)
                val_pred = (val_proba > 0.5).astype(int)
                val_acc = (val_pred == y_val).mean()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs}: Loss={loss:.4f} "
                              f"Train_Acc={train_acc:.3f} Val_Acc={val_acc:.3f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs}: Loss={loss:.4f} Train_Acc={train_acc:.3f}")

        self.is_trained = True
        self.train_samples = len(X_train)
        self.train_accuracy = best_val_acc if X_val is not None else train_acc

        logger.info(f"✓ Training completed")
        logger.info(f"  Best accuracy: {self.train_accuracy:.3f}")

    def save(self, path: str):
        """Сохранение модели"""
        data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'is_trained': self.is_trained,
            'train_samples': self.train_samples,
            'train_accuracy': self.train_accuracy
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"META model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'SimpleMetaNetwork':
        """Загрузка модели"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        meta = cls(data['input_size'], data['hidden_size'])
        meta.W1 = data['W1']
        meta.b1 = data['b1']
        meta.W2 = data['W2']
        meta.b2 = data['b2']
        meta.is_trained = data['is_trained']
        meta.train_samples = data['train_samples']
        meta.train_accuracy = data['train_accuracy']

        return meta


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Загружает датасет

    Returns:
        (X, y, metadata)
    """
    logger.info(f"Loading dataset: {filepath}")

    df = pd.read_parquet(filepath)

    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Columns: {df.columns.tolist()}")

    # Извлекаем фичи (68D)
    feature_cols = [col for col in df.columns if col.startswith('features_')]
    if len(feature_cols) > 0:
        X = df[feature_cols].values
    elif 'features' in df.columns:
        X = np.vstack(df['features'].values)
    else:
        raise ValueError("Колонки с фичами не найдены!")

    # Таргет
    y = df['outcome'].values

    # Метаданные
    metadata = {
        'n_samples': len(df),
        'n_features': X.shape[1],
        'class_0': (y == 0).sum(),
        'class_1': (y == 1).sum()
    }

    logger.info(f"  Features shape: {X.shape}")
    logger.info(f"  Class 0: {metadata['class_0']} ({metadata['class_0']/len(y)*100:.1f}%)")
    logger.info(f"  Class 1: {metadata['class_1']} ({metadata['class_1']/len(y)*100:.1f}%)")

    return X, y, metadata


# ============================================================================
# EXPERT PREDICTIONS
# ============================================================================

def get_expert_predictions(X: np.ndarray, experts: dict) -> np.ndarray:
    """
    Получает предсказания от всех экспертов

    Args:
        X: (N, 68) - фичи
        experts: dict с экспертами

    Returns:
        (N, 4) - предсказания от 4 экспертов
    """
    N = len(X)
    predictions = np.zeros((N, 4))

    reg_ctx = {'phase': 0}

    # XGBoost
    if experts.get('xgb') is not None:
        for i in range(N):
            p, _ = experts['xgb'].proba_up(X[i], reg_ctx)
            predictions[i, 0] = p

    # Random Forest
    if experts.get('rf') is not None:
        for i in range(N):
            p, _ = experts['rf'].proba_up(X[i], reg_ctx)
            predictions[i, 1] = p

    # Adaptive RF
    if experts.get('arf') is not None:
        for i in range(N):
            p, _ = experts['arf'].proba_up(X[i], reg_ctx)
            predictions[i, 2] = p

    # Neural Network
    if experts.get('nn') is not None:
        for i in range(N):
            p, _ = experts['nn'].proba_up(X[i], reg_ctx)
            predictions[i, 3] = p

    return predictions


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    """Главная функция обучения"""
    logger.info("="*80)
    logger.info("META НЕЙРОСЕТЬ - ОБУЧЕНИЕ")
    logger.info("="*80)

    # 1. Загружаем обученных экспертов
    logger.info("\nЭТАП 1: ЗАГРУЗКА ЭКСПЕРТОВ")
    logger.info("-"*80)

    experts = {}

    # XGBoost
    xgb_path = f"{Config.MODELS_DIR}/xgb_expert.pkl"
    if os.path.exists(xgb_path):
        experts['xgb'] = XGBoostExpert.load(xgb_path)
        logger.info(f"✓ XGBoost loaded: {experts['xgb'].train_samples} samples")
    else:
        logger.error(f"✗ XGBoost not found: {xgb_path}")
        experts['xgb'] = None

    # Random Forest
    rf_path = f"{Config.MODELS_DIR}/rf_expert.pkl"
    if os.path.exists(rf_path):
        experts['rf'] = RandomForestExpert.load(rf_path)
        logger.info(f"✓ RandomForest loaded: {experts['rf'].train_samples} samples")
    else:
        logger.error(f"✗ RandomForest not found: {rf_path}")
        experts['rf'] = None

    # Adaptive RF
    arf_path = f"{Config.MODELS_DIR}/arf_expert.pkl"
    if os.path.exists(arf_path):
        # ARF может требовать retraining после загрузки
        try:
            experts['arf'] = AdaptiveRFExpert.load(arf_path)
            logger.info(f"✓ AdaptiveRF loaded: {experts['arf'].train_samples} samples")
        except Exception as e:
            logger.warning(f"⚠ AdaptiveRF load warning: {e}")
            experts['arf'] = None
    else:
        logger.error(f"✗ AdaptiveRF not found: {arf_path}")
        experts['arf'] = None

    # Neural Network
    nn_path = f"{Config.MODELS_DIR}/nn_expert.pkl"
    if os.path.exists(nn_path):
        experts['nn'] = NeuralNetworkExpert.load(nn_path)
        logger.info(f"✓ NeuralNet loaded: {experts['nn'].train_samples} samples")
    else:
        logger.error(f"✗ NeuralNet not found: {nn_path}")
        experts['nn'] = None

    loaded_experts = sum(1 for e in experts.values() if e is not None)
    logger.info(f"\nЗагружено экспертов: {loaded_experts}/4")

    if loaded_experts < 3:
        logger.error("Недостаточно экспертов для обучения META!")
        return

    # 2. Загружаем датасеты
    logger.info("\nЭТАП 2: ЗАГРУЗКА ДАТАСЕТОВ")
    logger.info("-"*80)

    X_train, y_train, train_meta = load_dataset(Config.TRAIN_DATASET)
    X_test, y_test, test_meta = load_dataset(Config.TEST_DATASET)

    # 3. Получаем предсказания экспертов
    logger.info("\nЭТАП 3: ПРЕДСКАЗАНИЯ ЭКСПЕРТОВ")
    logger.info("-"*80)

    logger.info("Получение предсказаний на train set...")
    start_time = time.time()
    expert_preds_train = get_expert_predictions(X_train, experts)
    elapsed = time.time() - start_time
    logger.info(f"  ✓ Train predictions: {expert_preds_train.shape} за {elapsed:.1f}s")

    logger.info("Получение предсказаний на test set...")
    start_time = time.time()
    expert_preds_test = get_expert_predictions(X_test, experts)
    elapsed = time.time() - start_time
    logger.info(f"  ✓ Test predictions: {expert_preds_test.shape} за {elapsed:.1f}s")

    # Статистика предсказаний экспертов
    logger.info("\nСтатистика предсказаний экспертов (train):")
    logger.info(f"  XGBoost: mean={expert_preds_train[:,0].mean():.3f} std={expert_preds_train[:,0].std():.3f}")
    logger.info(f"  RF:      mean={expert_preds_train[:,1].mean():.3f} std={expert_preds_train[:,1].std():.3f}")
    logger.info(f"  ARF:     mean={expert_preds_train[:,2].mean():.3f} std={expert_preds_train[:,2].std():.3f}")
    logger.info(f"  NN:      mean={expert_preds_train[:,3].mean():.3f} std={expert_preds_train[:,3].std():.3f}")

    # 4. Обучаем META
    logger.info("\nЭТАП 4: ОБУЧЕНИЕ META")
    logger.info("-"*80)

    meta = SimpleMetaNetwork(input_size=4, hidden_size=Config.META_HIDDEN_SIZE)

    meta.fit(
        X_train=expert_preds_train,
        y_train=y_train,
        X_val=expert_preds_test,
        y_val=y_test,
        epochs=100,
        learning_rate=0.01
    )

    # 5. Оценка на test set
    logger.info("\nЭТАП 5: ОЦЕНКА НА TEST SET")
    logger.info("-"*80)

    # META predictions
    meta_preds = meta.predict(expert_preds_test)
    meta_binary = (meta_preds > 0.5).astype(int)
    meta_acc = (meta_binary == y_test).mean()

    # Среднее экспертов (baseline)
    mean_preds = expert_preds_test.mean(axis=1)
    mean_binary = (mean_preds > 0.5).astype(int)
    mean_acc = (mean_binary == y_test).mean()

    # Лучший эксперт
    best_expert_idx = None
    best_expert_acc = 0.0
    expert_names = ['XGBoost', 'RF', 'ARF', 'NN']

    for i in range(4):
        expert_binary = (expert_preds_test[:, i] > 0.5).astype(int)
        expert_acc = (expert_binary == y_test).mean()
        if expert_acc > best_expert_acc:
            best_expert_acc = expert_acc
            best_expert_idx = i

    logger.info(f"\nРезультаты на test set:")
    logger.info(f"  Среднее экспертов:    {mean_acc*100:.2f}%")
    logger.info(f"  Лучший эксперт ({expert_names[best_expert_idx]}): {best_expert_acc*100:.2f}%")
    logger.info(f"  META:                {meta_acc*100:.2f}%")

    improvement = (meta_acc - mean_acc) * 100
    logger.info(f"\nУлучшение META над средним: {improvement:+.2f}%")

    # 6. Сохраняем META
    logger.info("\nЭТАП 6: СОХРАНЕНИЕ META")
    logger.info("-"*80)

    os.makedirs(os.path.dirname(Config.META_SAVE_PATH), exist_ok=True)
    meta.save(Config.META_SAVE_PATH)

    # Финальный отчет
    logger.info("\n" + "="*80)
    logger.info("✅ ОБУЧЕНИЕ META ЗАВЕРШЕНО")
    logger.info("="*80)
    logger.info(f"\nМодель сохранена: {Config.META_SAVE_PATH}")
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Meta accuracy: {meta_acc*100:.2f}%")
    logger.info(f"Improvement: {improvement:+.2f}%")
    logger.info("\nСледующие шаги:")
    logger.info("1. Интегрировать META в Paper Trading бот")
    logger.info("2. Запустить paper trading для тестирования")
    logger.info("3. Собрать статистику за 2-4 недели")


if __name__ == "__main__":
    main()
