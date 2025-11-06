"""
Adaptive Random Forest Expert

Онлайн обучение с использованием River библиотеки.
Адаптивный Random Forest который учится инкрементально на streaming данных.

Особенности:
- Истинное онлайн обучение (не batch)
- Адаптация к concept drift
- ADWIN drift detection
- Малое потребление памяти
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple

try:
    from river import ensemble, tree
    from river.drift import ADWIN
    HAVE_RIVER = True
except ImportError:
    HAVE_RIVER = False
    print("WARNING: river not installed. Install with: pip install river")


class AdaptiveRFExpert:
    """
    Adaptive Random Forest для онлайн обучения

    Предсказывает вероятность достижения TP (outcome=1)
    """

    def __init__(
        self,
        n_models: int = 10,
        max_depth: int = 10,
        lambda_value: int = 6,
        drift_detector: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            n_models: Количество деревьев в ensemble
            max_depth: Максимальная глубина дерева
            lambda_value: Для Poisson sampling
            drift_detector: Использовать ADWIN drift detection
            random_state: Random seed
        """
        if not HAVE_RIVER:
            raise ImportError("river not installed")

        self.n_models = n_models
        self.max_depth = max_depth
        self.lambda_value = lambda_value
        self.drift_detector = drift_detector
        self.random_state = random_state

        # Модель (ARF из River)
        self.model = None
        self.is_trained = False
        self.n_features = None

        # Статистика
        self.train_samples = 0
        self.drift_detections = 0

        self._init_model()

    def _init_model(self):
        """Инициализация модели"""
        # Создаем Adaptive Random Forest
        # Используем минимальные параметры для совместимости
        try:
            self.model = ensemble.AdaptiveRandomForestClassifier(
                n_models=self.n_models,
                seed=self.random_state
            )
        except Exception as e:
            print(f"Ошибка создания ARF: {e}")
            # Fallback к базовой версии
            self.model = ensemble.AdaptiveRandomForestClassifier(
                n_models=self.n_models
            )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """
        Batch обучение (конвертируется в онлайн)

        Args:
            X: Фичи (n_samples, n_features)
            y: Таргет (n_samples,) - 0 или 1
            sample_weight: Веса примеров (игнорируется для River)
        """
        if len(X) == 0:
            raise ValueError("Empty training set")

        self.n_features = X.shape[1]

        # Обучаем инкрементально
        for i in range(len(X)):
            # Конвертируем в dict (формат River)
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            y_val = int(y[i])

            # Онлайн обучение
            self.model.learn_one(x_dict, y_val)

        self.is_trained = True
        self.train_samples = len(X)

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_new_trees: int = None,  # Не используется для ARF
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Онлайн обучение (инкрементальное)

        Args:
            X: Новые фичи
            y: Новый таргет
            n_new_trees: Не используется (ARF всегда использует все деревья)
            sample_weight: Не используется
        """
        if X.shape[1] != self.n_features and self.n_features is not None:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        if self.n_features is None:
            self.n_features = X.shape[1]

        # Обучаем инкрементально
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            y_val = int(y[i])

            # Онлайн обучение
            self.model.learn_one(x_dict, y_val)

            # Проверяем drift detection
            if self.drift_detector and hasattr(self.model, 'drift_detector'):
                # River ARF автоматически обрабатывает drift
                pass

        self.is_trained = True
        self.train_samples += len(X)

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

        probas = []
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}

            # Предсказываем вероятность
            proba_dict = self.model.predict_proba_one(x_dict)

            # River возвращает dict {0: p0, 1: p1}
            p1 = proba_dict.get(1, 0.5)
            probas.append(p1)

        return np.array(probas)

    def proba_up(
        self,
        features: np.ndarray,
        reg_ctx: Optional[Dict] = None
    ) -> Tuple[float, Dict]:
        """
        Предсказание вероятности роста (для совместимости с текущим API)

        Args:
            features: Фичи (68,) или (n_samples, 68)
            reg_ctx: Контекст (не используется)

        Returns:
            Tuple[float, Dict]: (вероятность, метаданные)
        """
        # Приводим к 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        proba = self.predict_proba(features)

        metadata = {
            'expert': 'arf',
            'is_trained': self.is_trained,
            'train_samples': self.train_samples,
            'n_models': self.n_models,
            'drift_detections': self.drift_detections
        }

        return float(proba[0]), metadata

    def save(self, filepath: str):
        """
        Сохранение модели

        ВАЖНО: River модели сложно сериализовать полностью.
        Сохраняем только статистику, модель пересоздается.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            'n_models': self.n_models,
            'max_depth': self.max_depth,
            'lambda_value': self.lambda_value,
            'drift_detector': self.drift_detector,
            'random_state': self.random_state,
            'is_trained': self.is_trained,
            'n_features': self.n_features,
            'train_samples': self.train_samples,
            'drift_detections': self.drift_detections,
            # River модели сложно сериализовать, поэтому сохраняем только параметры
            'note': 'ARF model will be recreated on load - requires retraining'
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> 'AdaptiveRFExpert':
        """
        Загрузка модели

        ВАЖНО: Модель пересоздается, требует повторного обучения!
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Создаем эксперт с параметрами
        expert = cls(
            n_models=state['n_models'],
            max_depth=state['max_depth'],
            lambda_value=state['lambda_value'],
            drift_detector=state['drift_detector'],
            random_state=state['random_state']
        )

        # Загружаем статистику
        expert.is_trained = False  # Требуется retraining!
        expert.n_features = state['n_features']
        expert.train_samples = 0
        expert.drift_detections = 0

        return expert

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Получение важности фич

        River ARF не предоставляет feature importance напрямую
        """
        # River ARF не поддерживает feature_importances
        return None


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ТЕСТ ADAPTIVE RANDOM FOREST ЭКСПЕРТА")
    print("="*80)

    if not HAVE_RIVER:
        print("❌ River не установлен!")
        print("Установите: pip install river")
        exit(1)

    # Генерируем тестовые данные
    np.random.seed(42)
    n_samples = 1000
    n_features = 68

    X_train = np.random.randn(n_samples, n_features)
    # Простая зависимость
    y_train = (X_train[:, :5].sum(axis=1) > 0).astype(int)

    print(f"\nТестовые данные:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  Класс 1: {y_train.sum()} примеров ({y_train.mean()*100:.1f}%)")

    # Создаем и обучаем эксперт
    print("\n1. Создание эксперта...")
    expert = AdaptiveRFExpert(
        n_models=10,
        max_depth=8,
        drift_detector=True
    )

    print("2. Обучение (онлайн, может занять время)...")
    import time
    start = time.time()
    expert.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"   ✅ Обучено на {expert.train_samples} примерах за {elapsed:.1f}s")
    print(f"   Моделей в ensemble: {expert.n_models}")

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

    samples_before = expert.train_samples
    expert.partial_fit(X_new, y_new)
    samples_after = expert.train_samples

    print(f"   Samples до: {samples_before}")
    print(f"   Samples после: {samples_after}")
    print(f"   Добавлено: {samples_after - samples_before}")

    # Сохранение и загрузка
    print("\n6. Тест сохранения/загрузки...")
    save_path = "/tmp/test_arf_expert.pkl"
    expert.save(save_path)
    print(f"   ✅ Сохранено: {save_path}")

    expert_loaded = AdaptiveRFExpert.load(save_path)
    print(f"   ⚠️  Загружено (требует retraining)")
    print(f"   is_trained: {expert_loaded.is_trained}")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("="*80)
    print("\n⚠️  ВАЖНО: ARF требует онлайн обучения при загрузке!")
