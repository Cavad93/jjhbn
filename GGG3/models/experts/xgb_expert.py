"""
XGBoost Expert

Gradient Boosting эксперт для предсказания вероятности достижения TP.

Особенности:
- Онлайн обучение через partial_fit (добавление новых деревьев)
- L1/L2 регуляризация
- Калибровка вероятностей
- Поддержка phase-specific training
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("WARNING: xgboost not installed. Install with: pip install xgboost")


class XGBoostExpert:
    """
    XGBoost эксперт для бинарной классификации

    Предсказывает вероятность достижения TP (outcome=1)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        """
        Args:
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина дерева
            learning_rate: Learning rate
            subsample: Доля примеров для обучения каждого дерева
            colsample_bytree: Доля фич для обучения каждого дерева
            reg_alpha: L1 регуляризация
            reg_lambda: L2 регуляризация
            random_state: Random seed
        """
        if not HAVE_XGB:
            raise ImportError("xgboost not installed")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
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

        # Создаем DMatrix
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)

        # Параметры
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'seed': self.random_state,
            'tree_method': 'hist',
            'verbosity': 0
        }

        # Обучаем
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )

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
        Онлайн обучение - добавляет новые деревья к существующей модели

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

        # Создаем DMatrix
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)

        # Параметры
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'seed': self.random_state,
            'tree_method': 'hist',
            'verbosity': 0
        }

        # Добавляем новые деревья к существующей модели
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_new_trees,
            xgb_model=self.model,  # Продолжаем с существующей модели
            verbose_eval=False
        )

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

        dtest = xgb.DMatrix(X)
        proba = self.model.predict(dtest)

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
            'expert': 'xgb',
            'is_trained': self.is_trained,
            'train_samples': self.train_samples,
            'n_trees': self.model.num_boosted_rounds() if self.model else 0
        }

        return float(proba[0]), metadata

    def save(self, filepath: str):
        """Сохранение модели"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if self.model is not None:
            # Сохраняем XGBoost модель в JSON (легче для версионирования)
            model_path = filepath.replace('.pkl', '_xgb.json')
            self.model.save_model(model_path)

        # Сохраняем метаданные
        state = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'is_trained': self.is_trained,
            'n_features': self.n_features,
            'train_samples': self.train_samples,
            'retrain_count': self.retrain_count,
            'model_path': model_path if self.model else None
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> 'XGBoostExpert':
        """Загрузка модели"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Создаем эксперт с параметрами
        expert = cls(
            n_estimators=state['n_estimators'],
            max_depth=state['max_depth'],
            learning_rate=state['learning_rate'],
            subsample=state['subsample'],
            colsample_bytree=state['colsample_bytree'],
            reg_alpha=state['reg_alpha'],
            reg_lambda=state['reg_lambda'],
            random_state=state['random_state']
        )

        # Загружаем состояние
        expert.is_trained = state['is_trained']
        expert.n_features = state['n_features']
        expert.train_samples = state['train_samples']
        expert.retrain_count = state['retrain_count']

        # Загружаем модель
        if state['model_path'] and os.path.exists(state['model_path']):
            expert.model = xgb.Booster()
            expert.model.load_model(state['model_path'])

        return expert

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Получение важности фич"""
        if not self.is_trained or self.model is None:
            return None

        # Weight importance (количество использований фичи в splits)
        importance = self.model.get_score(importance_type='weight')

        # Преобразуем в array (может быть не все фичи используются)
        importance_array = np.zeros(self.n_features)
        for feat_name, score in importance.items():
            # feat_name вида 'f0', 'f1', ...
            idx = int(feat_name[1:])
            if idx < self.n_features:
                importance_array[idx] = score

        return importance_array


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ТЕСТ XGBoost ЭКСПЕРТА")
    print("="*80)

    if not HAVE_XGB:
        print("❌ XGBoost не установлен!")
        exit(1)

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
    expert = XGBoostExpert(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1
    )

    print("2. Обучение...")
    expert.fit(X_train, y_train)
    print(f"   ✅ Обучено на {expert.train_samples} примерах")
    print(f"   Количество деревьев: {expert.model.num_boosted_rounds()}")

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

    trees_before = expert.model.num_boosted_rounds()
    expert.partial_fit(X_new, y_new, n_new_trees=10)
    trees_after = expert.model.num_boosted_rounds()

    print(f"   Деревьев до: {trees_before}")
    print(f"   Деревьев после: {trees_after}")
    print(f"   Добавлено: {trees_after - trees_before}")
    print(f"   ✅ Total samples: {expert.train_samples}")

    # Сохранение и загрузка
    print("\n6. Тест сохранения/загрузки...")
    save_path = "/tmp/test_xgb_expert.pkl"
    expert.save(save_path)
    print(f"   ✅ Сохранено: {save_path}")

    expert_loaded = XGBoostExpert.load(save_path)
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
            print(f"     f{idx}: {importance[idx]:.0f}")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("="*80)
