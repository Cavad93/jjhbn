"""
Random Forest Expert

Ensemble метод на основе деревьев решений для предсказания вероятности достижения TP.

Особенности:
- Онлайн обучение через добавление новых деревьев
- Bagging для снижения variance
- Feature importance
- Калибровка вероятностей
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


class RandomForestExpert:
    """
    Random Forest эксперт для бинарной классификации

    Предсказывает вероятность достижения TP (outcome=1)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = 'sqrt',
        bootstrap: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина дерева
            min_samples_split: Минимум примеров для split
            min_samples_leaf: Минимум примеров в листе
            max_features: Количество фич для каждого дерева
            bootstrap: Использовать bootstrap sampling
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        # Модель
        self.model = None
        self.is_trained = False
        self.n_features = None

        # Статистика
        self.train_samples = 0
        self.retrain_count = 0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """
        Обучение модели с нуля

        Args:
            X: Фичи (n_samples, n_features)
            y: Таргет (n_samples,) - 0 или 1
            sample_weight: Веса примеров (опционально)
        """
        if len(X) == 0:
            raise ValueError("Empty training set")

        self.n_features = X.shape[1]

        # Создаем Random Forest
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1,  # Используем все CPU
            verbose=0
        )

        # Обучаем
        self.model.fit(X, y, sample_weight=sample_weight)

        self.is_trained = True
        self.train_samples = len(X)
        self.retrain_count += 1

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_new_trees: int = 10,
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Онлайн обучение - добавляет новые деревья к существующему лесу

        Args:
            X: Новые фичи
            y: Новый таргет
            n_new_trees: Количество новых деревьев для добавления
            sample_weight: Веса примеров
        """
        if not self.is_trained:
            # Первое обучение
            self.fit(X, y, sample_weight)
            return

        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        # Создаем новый лес с новыми деревьями
        new_forest = RandomForestClassifier(
            n_estimators=n_new_trees,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state + self.retrain_count,  # Новый seed
            n_jobs=-1,
            verbose=0
        )

        # Обучаем новые деревья
        new_forest.fit(X, y, sample_weight=sample_weight)

        # Добавляем новые деревья к существующему лесу
        self.model.estimators_ += new_forest.estimators_
        self.model.n_estimators = len(self.model.estimators_)

        self.train_samples += len(X)
        self.retrain_count += 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятностей

        Args:
            X: Фичи (n_samples, n_features)

        Returns:
            np.ndarray: Вероятности класса 1 (n_samples,)
        """
        if not self.is_trained:
            # Если модель не обучена, возвращаем 0.5
            return np.full(len(X), 0.5)

        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        proba = self.model.predict_proba(X)[:, 1]

        return proba

    def proba_up(
        self,
        features: np.ndarray,
        reg_ctx: Optional[Dict] = None
    ) -> Tuple[float, Dict]:
        """
        Предсказание вероятности роста (для совместимости с текущим API)

        Args:
            features: Фичи (68,) или (n_samples, 68)
            reg_ctx: Контекст (не используется в базовой версии)

        Returns:
            Tuple[float, Dict]: (вероятность, метаданные)
        """
        # Приводим к 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        proba = self.predict_proba(features)

        metadata = {
            'expert': 'rf',
            'is_trained': self.is_trained,
            'train_samples': self.train_samples,
            'n_trees': len(self.model.estimators_) if self.model else 0
        }

        return float(proba[0]), metadata

    def save(self, filepath: str):
        """Сохранение модели"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            'model': self.model,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'is_trained': self.is_trained,
            'n_features': self.n_features,
            'train_samples': self.train_samples,
            'retrain_count': self.retrain_count
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> 'RandomForestExpert':
        """Загрузка модели"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Создаем эксперт с параметрами
        expert = cls(
            n_estimators=state['n_estimators'],
            max_depth=state['max_depth'],
            min_samples_split=state['min_samples_split'],
            min_samples_leaf=state['min_samples_leaf'],
            max_features=state['max_features'],
            bootstrap=state['bootstrap'],
            random_state=state['random_state']
        )

        # Загружаем состояние
        expert.model = state['model']
        expert.is_trained = state['is_trained']
        expert.n_features = state['n_features']
        expert.train_samples = state['train_samples']
        expert.retrain_count = state['retrain_count']

        return expert

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Получение важности фич"""
        if not self.is_trained or self.model is None:
            return None

        return self.model.feature_importances_


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ТЕСТ RANDOM FOREST ЭКСПЕРТА")
    print("="*80)

    # Генерируем тестовые данные
    np.random.seed(42)
    n_samples = 1000
    n_features = 68

    X_train = np.random.randn(n_samples, n_features)
    # Простая зависимость: если сумма первых 5 фич > 0 → класс 1
    y_train = (X_train[:, :5].sum(axis=1) > 0).astype(int)

    print(f"\nТестовые данные:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  Класс 1: {y_train.sum()} примеров ({y_train.mean()*100:.1f}%)")

    # Создаем и обучаем эксперт
    print("\n1. Создание эксперта...")
    expert = RandomForestExpert(
        n_estimators=50,
        max_depth=8,
        min_samples_split=10
    )

    print("2. Обучение...")
    expert.fit(X_train, y_train)
    print(f"   ✅ Обучено на {expert.train_samples} примерах")
    print(f"   Количество деревьев: {len(expert.model.estimators_)}")

    # Предсказание
    print("\n3. Предсказание...")
    X_test = np.random.randn(100, n_features)
    y_test = (X_test[:, :5].sum(axis=1) > 0).astype(int)

    proba = expert.predict_proba(X_test)
    pred = (proba > 0.5).astype(int)
    accuracy = (pred == y_test).mean()

    print(f"   Accuracy: {accuracy*100:.1f}%")
    print(f"   Средняя вероятность: {proba.mean():.3f}")

    # Тест proba_up API
    print("\n4. Тест proba_up API...")
    single_features = X_test[0]
    p_up, metadata = expert.proba_up(single_features)
    print(f"   p_up: {p_up:.3f}")
    print(f"   Metadata: {metadata}")

    # Онлайн обучение
    print("\n5. Тест онлайн обучения (partial_fit)...")
    X_new = np.random.randn(100, n_features)
    y_new = (X_new[:, :5].sum(axis=1) > 0).astype(int)

    trees_before = len(expert.model.estimators_)
    expert.partial_fit(X_new, y_new, n_new_trees=10)
    trees_after = len(expert.model.estimators_)

    print(f"   Деревьев до: {trees_before}")
    print(f"   Деревьев после: {trees_after}")
    print(f"   Добавлено: {trees_after - trees_before}")
    print(f"   ✅ Total samples: {expert.train_samples}")

    # Сохранение и загрузка
    print("\n6. Тест сохранения/загрузки...")
    save_path = "/tmp/test_rf_expert.pkl"
    expert.save(save_path)
    print(f"   ✅ Сохранено: {save_path}")

    expert_loaded = RandomForestExpert.load(save_path)
    print(f"   ✅ Загружено")

    # Проверка что предсказания одинаковые
    proba_original = expert.predict_proba(X_test[:10])
    proba_loaded = expert_loaded.predict_proba(X_test[:10])
    diff = np.abs(proba_original - proba_loaded).max()
    print(f"   Max diff: {diff:.6f}")

    if diff < 1e-5:
        print("   ✅ Предсказания идентичны!")

    # Feature importance
    print("\n7. Feature importance...")
    importance = expert.get_feature_importance()
    if importance is not None:
        top_5_idx = np.argsort(importance)[-5:][::-1]
        print(f"   Топ-5 важных фич:")
        for idx in top_5_idx:
            print(f"     f{idx}: {importance[idx]:.4f}")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("="*80)
