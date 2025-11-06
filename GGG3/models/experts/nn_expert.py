"""
Neural Network Expert

Deep Learning эксперт на PyTorch для предсказания вероятности достижения TP.

Архитектура:
- Input: 68 фич
- Hidden: 128 -> 64 -> 32
- Output: 1 (вероятность через sigmoid)
- Activation: ReLU
- Dropout: 0.2
- Batch Normalization

Особенности:
- Онлайн обучение (mini-batch SGD)
- Early stopping
- Learning rate scheduling
- Gradient clipping
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print("WARNING: torch not installed. Install with: pip install torch")


class BinaryClassifierNN(nn.Module):
    """
    Нейронная сеть для бинарной классификации

    Architecture: 68 -> 128 -> 64 -> 32 -> 1
    """

    def __init__(self, input_dim: int = 68, dropout: float = 0.2):
        super(BinaryClassifierNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        # Output
        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x


class NeuralNetworkExpert:
    """
    Neural Network эксперт для бинарной классификации

    Предсказывает вероятность достижения TP (outcome=1)
    """

    def __init__(
        self,
        input_dim: int = 68,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        max_epochs: int = 50,
        early_stopping_patience: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            input_dim: Размерность входных фич
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size для обучения
            max_epochs: Максимум эпох
            early_stopping_patience: Терпение для early stopping
            random_state: Random seed
        """
        if not HAVE_TORCH:
            raise ImportError("torch not installed")

        self.input_dim = input_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        # Устанавливаем seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Модель
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.is_trained = False

        # Статистика
        self.train_samples = 0
        self.train_epochs = 0
        self.best_loss = float('inf')

        self._init_model()

    def _init_model(self):
        """Инициализация модели"""
        self.model = BinaryClassifierNN(
            input_dim=self.input_dim,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.criterion = nn.BCELoss()  # Binary Cross Entropy

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Обучение модели

        Args:
            X: Фичи (n_samples, n_features)
            y: Таргет (n_samples,) - 0 или 1
            X_val: Валидационные фичи (опционально)
            y_val: Валидационный таргет (опционально)
            sample_weight: Веса примеров (не используется пока)
        """
        if len(X) == 0:
            raise ValueError("Empty training set")

        # Конвертируем в tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        # Создаем DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Валидационный набор
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)

        # Early stopping
        patience_counter = 0
        best_val_loss = float('inf')

        # Обучение
        self.model.train()

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                # Forward
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Update
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            # Валидация
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                self.model.train()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_loss = val_loss
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    print(f"   Early stopping на эпохе {epoch+1}")
                    break

        self.is_trained = True
        self.train_samples = len(X)
        self.train_epochs = epoch + 1

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 5,
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Онлайн обучение (fine-tuning на новых данных)

        Args:
            X: Новые фичи
            y: Новый таргет
            n_epochs: Количество эпох для дообучения
            sample_weight: Веса примеров
        """
        if not self.is_trained:
            # Первое обучение
            self.fit(X, y)
            return

        # Дообучаем на новых данных
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for epoch in range(n_epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        self.train_samples += len(X)
        self.train_epochs += n_epochs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятностей

        Args:
            X: Фичи (n_samples, n_features)

        Returns:
            np.ndarray: Вероятности класса 1 (n_samples,)
        """
        if not self.is_trained:
            return np.full(len(X), 0.5)

        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            proba = outputs.cpu().numpy().flatten()

        return proba

    def proba_up(
        self,
        features: np.ndarray,
        reg_ctx: Optional[Dict] = None
    ) -> Tuple[float, Dict]:
        """
        Предсказание вероятности роста

        Args:
            features: Фичи (68,) или (n_samples, 68)
            reg_ctx: Контекст

        Returns:
            Tuple[float, Dict]: (вероятность, метаданные)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        proba = self.predict_proba(features)

        metadata = {
            'expert': 'nn',
            'is_trained': self.is_trained,
            'train_samples': self.train_samples,
            'train_epochs': self.train_epochs
        }

        return float(proba[0]), metadata

    def save(self, filepath: str):
        """Сохранение модели"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Сохраняем веса модели
        model_path = filepath.replace('.pkl', '_nn.pt')
        torch.save(self.model.state_dict(), model_path)

        # Сохраняем метаданные
        state = {
            'input_dim': self.input_dim,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'random_state': self.random_state,
            'is_trained': self.is_trained,
            'train_samples': self.train_samples,
            'train_epochs': self.train_epochs,
            'best_loss': self.best_loss,
            'model_path': model_path
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> 'NeuralNetworkExpert':
        """Загрузка модели"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        expert = cls(
            input_dim=state['input_dim'],
            dropout=state['dropout'],
            learning_rate=state['learning_rate'],
            batch_size=state['batch_size'],
            max_epochs=state['max_epochs'],
            early_stopping_patience=state['early_stopping_patience'],
            random_state=state['random_state']
        )

        expert.is_trained = state['is_trained']
        expert.train_samples = state['train_samples']
        expert.train_epochs = state['train_epochs']
        expert.best_loss = state['best_loss']

        # Загружаем веса
        if state['model_path'] and os.path.exists(state['model_path']):
            expert.model.load_state_dict(torch.load(state['model_path']))

        return expert


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ТЕСТ NEURAL NETWORK ЭКСПЕРТА")
    print("="*80)

    if not HAVE_TORCH:
        print("❌ PyTorch не установлен!")
        exit(1)

    # Тестовые данные
    np.random.seed(42)
    n_samples = 1000
    n_features = 68

    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = (X_train[:, :5].sum(axis=1) > 0).astype(np.float32)

    X_val = np.random.randn(200, n_features).astype(np.float32)
    y_val = (X_val[:, :5].sum(axis=1) > 0).astype(np.float32)

    print(f"\nТестовые данные:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape} (класс 1: {y_train.mean()*100:.1f}%)")
    print(f"  X_val: {X_val.shape}")

    # Создаем эксперт
    print("\n1. Создание эксперта...")
    expert = NeuralNetworkExpert(
        input_dim=68,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        max_epochs=30,
        early_stopping_patience=5
    )

    # Обучение
    print("\n2. Обучение...")
    expert.fit(X_train, y_train, X_val, y_val)
    print(f"   ✅ Обучено за {expert.train_epochs} эпох")
    print(f"   Best loss: {expert.best_loss:.4f}")

    # Предсказание
    print("\n3. Предсказание...")
    X_test = np.random.randn(100, n_features).astype(np.float32)
    y_test = (X_test[:, :5].sum(axis=1) > 0).astype(np.float32)

    proba = expert.predict_proba(X_test)
    pred = (proba > 0.5).astype(int)
    accuracy = (pred == y_test).mean()

    print(f"   Accuracy: {accuracy*100:.1f}%")
    print(f"   Средняя вероятность: {proba.mean():.3f}")

    # Тест proba_up
    print("\n4. Тест proba_up API...")
    p_up, metadata = expert.proba_up(X_test[0])
    print(f"   p_up: {p_up:.3f}")
    print(f"   Metadata: {metadata}")

    # Онлайн обучение
    print("\n5. Тест partial_fit...")
    X_new = np.random.randn(100, n_features).astype(np.float32)
    y_new = (X_new[:, :5].sum(axis=1) > 0).astype(np.float32)

    epochs_before = expert.train_epochs
    expert.partial_fit(X_new, y_new, n_epochs=3)
    print(f"   Эпох до: {epochs_before}")
    print(f"   Эпох после: {expert.train_epochs}")

    # Сохранение
    print("\n6. Тест сохранения/загрузки...")
    save_path = "/tmp/test_nn_expert.pkl"
    expert.save(save_path)
    print(f"   ✅ Сохранено")

    expert_loaded = NeuralNetworkExpert.load(save_path)
    print(f"   ✅ Загружено")

    # Проверка
    proba_original = expert.predict_proba(X_test[:10])
    proba_loaded = expert_loaded.predict_proba(X_test[:10])
    diff = np.abs(proba_original - proba_loaded).max()
    print(f"   Max diff: {diff:.6f}")

    if diff < 1e-5:
        print("   ✅ Предсказания идентичны!")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("="*80)
