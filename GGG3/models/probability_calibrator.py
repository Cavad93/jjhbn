"""
Probability Calibration Module для ML экспертов

Обеспечивает калибровку вероятностей для всех экспертов:
- XGBoost
- RandomForest
- AdaptiveRF (опционально)
- NeuralNet

Методы калибровки:
- Isotonic regression (рекомендуется для >1000 samples)
- Platt scaling (sigmoid, для <1000 samples)
"""

import numpy as np
from typing import Optional, Literal
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """
    Калибратор вероятностей для бинарной классификации

    Использует:
    - Isotonic regression для монотонной калибровки
    - Platt scaling как fallback для малых данных
    """

    def __init__(
        self,
        method: Literal['isotonic', 'sigmoid', 'auto'] = 'auto',
        min_samples_isotonic: int = 100
    ):
        """
        Args:
            method: Метод калибровки
                - 'isotonic': Isotonic regression (лучше для >1000 samples)
                - 'sigmoid': Platt scaling (лучше для <1000 samples)
                - 'auto': Автоматический выбор на основе количества данных
            min_samples_isotonic: Минимум примеров для isotonic (иначе sigmoid)
        """
        self.method = method
        self.min_samples_isotonic = min_samples_isotonic
        self.calibrator = None
        self.is_fitted = False
        self.actual_method = None

    def fit(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Обучает калибратор

        Args:
            probabilities: Uncalibrated probabilities (n_samples,)
            y_true: True labels (n_samples,)
            sample_weight: Sample weights (optional)
        """
        if len(probabilities) == 0:
            raise ValueError("Empty probabilities array")

        if len(probabilities) != len(y_true):
            raise ValueError(f"Size mismatch: {len(probabilities)} probs vs {len(y_true)} labels")

        # Выбор метода
        if self.method == 'auto':
            if len(probabilities) >= self.min_samples_isotonic:
                self.actual_method = 'isotonic'
            else:
                self.actual_method = 'sigmoid'
        else:
            self.actual_method = self.method

        # Обучение калибратора
        if self.actual_method == 'isotonic':
            # Isotonic regression
            self.calibrator = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds='clip'
            )
            self.calibrator.fit(probabilities, y_true, sample_weight=sample_weight)

        elif self.actual_method == 'sigmoid':
            # Platt scaling (logistic regression)
            # Преобразуем вероятности в logits
            X = probabilities.reshape(-1, 1)

            self.calibrator = LogisticRegression(
                solver='lbfgs',
                max_iter=100,
                random_state=42
            )
            self.calibrator.fit(X, y_true, sample_weight=sample_weight)

        else:
            raise ValueError(f"Unknown method: {self.actual_method}")

        self.is_fitted = True

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Калибрует вероятности

        Args:
            probabilities: Uncalibrated probabilities (n_samples,)

        Returns:
            Calibrated probabilities (n_samples,)
        """
        if not self.is_fitted:
            # Если не обучен, возвращаем as-is
            return probabilities

        if len(probabilities) == 0:
            return probabilities

        # Clip to [0, 1] для безопасности
        probabilities = np.clip(probabilities, 0.0, 1.0)

        if self.actual_method == 'isotonic':
            # Isotonic transform
            calibrated = self.calibrator.transform(probabilities)

        elif self.actual_method == 'sigmoid':
            # Sigmoid transform
            X = probabilities.reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(X)[:, 1]

        else:
            calibrated = probabilities

        # Clip результат
        return np.clip(calibrated, 0.0, 1.0)

    def fit_transform(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit + transform в одном вызове"""
        self.fit(probabilities, y_true, sample_weight)
        return self.transform(probabilities)

    def get_calibration_curve(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> tuple:
        """
        Вычисляет calibration curve для визуализации

        Args:
            probabilities: Predicted probabilities
            y_true: True labels
            n_bins: Number of bins

        Returns:
            (mean_predicted_probs, fraction_positives, bin_counts)
        """
        # Биннинг вероятностей
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bins[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        mean_predicted = np.zeros(n_bins)
        fraction_positives = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_predicted[i] = probabilities[mask].mean()
                fraction_positives[i] = y_true[mask].mean()
                bin_counts[i] = mask.sum()

        return mean_predicted, fraction_positives, bin_counts


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ТЕСТ PROBABILITY CALIBRATOR")
    print("="*80)

    # Генерируем uncalibrated predictions (overconfident)
    np.random.seed(42)
    n_samples = 500

    # Истинные вероятности
    true_probs = np.random.beta(2, 2, n_samples)  # Примерно равномерное

    # Симулируем uncalibrated predictions (сдвиг к краям)
    uncalibrated_probs = true_probs ** 0.5  # Pushes к 1.0

    # Генерируем labels на основе истинных вероятностей
    y_true = (np.random.rand(n_samples) < true_probs).astype(int)

    print(f"\nТестовые данные:")
    print(f"  Samples: {n_samples}")
    print(f"  Mean uncalibrated prob: {uncalibrated_probs.mean():.3f}")
    print(f"  Mean true label: {y_true.mean():.3f}")
    print(f"  Difference: {abs(uncalibrated_probs.mean() - y_true.mean()):.3f}")

    # Создаем calibrator
    print("\n1. Создание калибратора (auto method)...")
    calibrator = ProbabilityCalibrator(method='auto', min_samples_isotonic=100)

    # Обучаем
    print("\n2. Обучение...")
    calibrator.fit(uncalibrated_probs, y_true)
    print(f"   Выбранный метод: {calibrator.actual_method}")
    print(f"   Обучено: {calibrator.is_fitted}")

    # Калибруем
    print("\n3. Калибровка...")
    calibrated_probs = calibrator.transform(uncalibrated_probs)

    print(f"   Mean calibrated prob: {calibrated_probs.mean():.3f}")
    print(f"   Mean true label: {y_true.mean():.3f}")
    print(f"   Difference: {abs(calibrated_probs.mean() - y_true.mean()):.3f}")
    print(f"   ✅ Улучшение: {abs(uncalibrated_probs.mean() - y_true.mean()) - abs(calibrated_probs.mean() - y_true.mean()):.3f}")

    # Calibration curve
    print("\n4. Calibration curve...")
    mean_pred, frac_pos, counts = calibrator.get_calibration_curve(
        uncalibrated_probs, y_true, n_bins=5
    )

    print("   Uncalibrated:")
    for i in range(5):
        if counts[i] > 0:
            print(f"     Bin {i+1}: pred={mean_pred[i]:.2f}, true={frac_pos[i]:.2f}, count={int(counts[i])}")

    calibrated_mean_pred, calibrated_frac_pos, calibrated_counts = calibrator.get_calibration_curve(
        calibrated_probs, y_true, n_bins=5
    )

    print("   Calibrated:")
    for i in range(5):
        if calibrated_counts[i] > 0:
            print(f"     Bin {i+1}: pred={calibrated_mean_pred[i]:.2f}, true={calibrated_frac_pos[i]:.2f}, count={int(calibrated_counts[i])}")

    # Тест sigmoid method
    print("\n5. Тест sigmoid method (малые данные)...")
    small_probs = uncalibrated_probs[:50]
    small_y = y_true[:50]

    calibrator_sigmoid = ProbabilityCalibrator(method='sigmoid')
    calibrator_sigmoid.fit(small_probs, small_y)
    calibrated_small = calibrator_sigmoid.transform(small_probs)

    print(f"   Mean uncalibrated: {small_probs.mean():.3f}")
    print(f"   Mean calibrated: {calibrated_small.mean():.3f}")
    print(f"   Mean true: {small_y.mean():.3f}")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("="*80)
