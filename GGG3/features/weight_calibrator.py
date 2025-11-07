#!/usr/bin/env python3
"""
BASE Logic Weight Calibrator

Автоматическая калибровка весов сигналов [M, S, B, R] на основе
исторических результатов торговли с использованием логистической регрессии.

Идея:
- Собираем сигналы BASE Logic на момент входа в позицию
- Обучаем логистическую регрессию предсказывать исход сделки (win/loss)
- Коэффициенты регрессии становятся новыми весами

Usage:
    from features.weight_calibrator import BaseWeightCalibrator

    calibrator = BaseWeightCalibrator()
    w_new = calibrator.calibrate(closed_positions)
    base_logic.set_weights(w_new)
"""

import numpy as np
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Опциональные зависимости
try:
    from sklearn.linear_model import LogisticRegression
    HAVE_SKLEARN = True
except ImportError:
    LogisticRegression = None
    HAVE_SKLEARN = False


@dataclass
class CalibrationResult:
    """Результат калибровки весов"""
    weights: np.ndarray  # Новые веса [M, S, B, R]
    n_samples: int  # Количество сделок
    win_rate: float  # Win rate в обучающей выборке
    improvement: float  # Насколько новые веса лучше старых (%)
    coefficients: np.ndarray  # Исходные коэффициенты регрессии
    old_weights: np.ndarray  # Старые веса для сравнения


class BaseWeightCalibrator:
    """
    Калибратор весов BASE Logic сигналов

    Использует логистическую регрессию для подбора оптимальных весов
    на основе исторических результатов торговли.
    """

    def __init__(
        self,
        min_samples: int = 50,
        regularization: float = 1.0,
        normalize_method: str = "l1",  # "l1", "l2", "softmax"
        smoothing_alpha: float = 0.3,  # EMA сглаживание с предыдущими весами
    ):
        """
        Args:
            min_samples: Минимум сделок для калибровки
            regularization: Параметр C для LogisticRegression (выше = меньше регуляризация)
            normalize_method: Метод нормализации весов (l1, l2, softmax)
            smoothing_alpha: Коэффициент EMA сглаживания (0 = только новые, 1 = только старые)
        """
        self.min_samples = min_samples
        self.regularization = regularization
        self.normalize_method = normalize_method
        self.smoothing_alpha = smoothing_alpha

        # История калибровок
        self.calibration_history = []
        self.last_weights = None

    def extract_signals_and_outcomes(self, closed_positions) -> tuple:
        """
        Извлекает BASE сигналы и исходы из закрытых позиций

        Args:
            closed_positions: Список закрытых позиций (Position objects)

        Returns:
            (signals, outcomes): Массивы сигналов [N, 4] и исходов [N]
        """
        signals = []
        outcomes = []

        for pos in closed_positions:
            # Проверяем наличие BASE сигналов в entry_snapshot
            if 'base_signals' not in pos.entry_snapshot:
                continue

            base_sig = pos.entry_snapshot['base_signals']

            # Извлекаем значения сигналов
            M = base_sig.get('M', 0.0)
            S = base_sig.get('S', 0.0)
            B = base_sig.get('B', 0.0)
            R = base_sig.get('R', 0.0)

            # Проверяем валидность
            if not all(np.isfinite([M, S, B, R])):
                continue

            signals.append([M, S, B, R])

            # Исход: 1 если прибыль, 0 если убыток
            outcome = 1 if pos.pnl > 0 else 0
            outcomes.append(outcome)

        if len(signals) == 0:
            return np.array([]).reshape(0, 4), np.array([])

        return np.array(signals, dtype=np.float64), np.array(outcomes, dtype=np.int32)

    def normalize_weights(self, w: np.ndarray) -> np.ndarray:
        """
        Нормализует веса согласно выбранному методу

        Args:
            w: Исходные веса (могут быть отрицательными)

        Returns:
            Нормализованные положительные веса с суммой = 1
        """
        # Берем абсолютные значения (сигналы уже имеют направление)
        w_abs = np.abs(w)

        # Защита от нулевых весов
        if np.sum(w_abs) < 1e-8:
            return np.array([0.25, 0.25, 0.25, 0.25])

        if self.normalize_method == "l1":
            # L1 нормализация: сумма = 1
            return w_abs / np.sum(w_abs)

        elif self.normalize_method == "l2":
            # L2 нормализация: ||w|| = 1, потом масштабируем
            w_norm = w_abs / np.linalg.norm(w_abs)
            return w_norm / np.sum(w_norm)

        elif self.normalize_method == "softmax":
            # Softmax: более плавное распределение
            w_exp = np.exp(w_abs - np.max(w_abs))
            return w_exp / np.sum(w_exp)

        else:
            return w_abs / np.sum(w_abs)

    def apply_smoothing(self, w_new: np.ndarray, w_old: Optional[np.ndarray]) -> np.ndarray:
        """
        Применяет EMA сглаживание между новыми и старыми весами

        Args:
            w_new: Новые веса
            w_old: Старые веса (None если первая калибровка)

        Returns:
            Сглаженные веса
        """
        if w_old is None or self.smoothing_alpha < 1e-6:
            return w_new

        # EMA: w = alpha * w_old + (1 - alpha) * w_new
        w_smoothed = self.smoothing_alpha * w_old + (1 - self.smoothing_alpha) * w_new

        # Ре-нормализуем
        return w_smoothed / np.sum(w_smoothed)

    def evaluate_weights(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        Оценивает качество весов (accuracy на данных)

        Args:
            X: Сигналы [N, 4]
            y: Исходы [N]
            w: Веса [4]

        Returns:
            Accuracy (доля правильных предсказаний)
        """
        # Линейная комбинация сигналов
        scores = X @ w

        # Предсказание: score > 0 означает ожидаем прибыль
        predictions = (scores > np.median(scores)).astype(int)

        # Accuracy
        return np.mean(predictions == y)

    def calibrate(
        self,
        closed_positions,
        old_weights: Optional[np.ndarray] = None
    ) -> Optional[CalibrationResult]:
        """
        Калибрует веса BASE Logic на основе закрытых позиций

        Args:
            closed_positions: Список закрытых позиций
            old_weights: Текущие веса (для сравнения и сглаживания)

        Returns:
            CalibrationResult или None если недостаточно данных
        """
        # Извлекаем сигналы и исходы
        X, y = self.extract_signals_and_outcomes(closed_positions)

        if len(X) < self.min_samples:
            logger.info(
                f"[WeightCalibrator] Недостаточно данных для калибровки: "
                f"{len(X)}/{self.min_samples}"
            )
            return None

        # Проверка баланса классов
        win_rate = np.mean(y)
        if win_rate < 0.1 or win_rate > 0.9:
            logger.warning(
                f"[WeightCalibrator] Несбалансированные данные: win_rate={win_rate:.2%}. "
                "Калибровка может быть нестабильной."
            )

        # Веса по умолчанию
        w_default = np.array([0.35, 0.20, 0.20, 0.25])
        if old_weights is None:
            old_weights = w_default

        try:
            if HAVE_SKLEARN and LogisticRegression is not None:
                # Используем sklearn LogisticRegression
                clf = LogisticRegression(
                    fit_intercept=False,  # Без bias (веса должны быть чистыми)
                    C=self.regularization,
                    solver='lbfgs',
                    max_iter=500,
                    class_weight='balanced'  # Автоматическая балансировка классов
                )

                clf.fit(X, y)

                # Извлекаем коэффициенты
                coefficients = clf.coef_[0]

                # Проверка на вырожденность
                if not np.all(np.isfinite(coefficients)):
                    logger.error("[WeightCalibrator] Получены невалидные коэффициенты")
                    return None

            else:
                # Fallback: простая корреляция с исходом
                logger.warning("[WeightCalibrator] sklearn недоступен, использую простую корреляцию")

                coefficients = np.zeros(4)
                for i in range(4):
                    # Корреляция между сигналом и исходом
                    if np.std(X[:, i]) > 1e-8:
                        coefficients[i] = np.corrcoef(X[:, i], y)[0, 1]
                    else:
                        coefficients[i] = 0.25

                # Если все корреляции ~0, используем дефолт
                if np.max(np.abs(coefficients)) < 0.01:
                    coefficients = w_default

            # Нормализуем веса
            w_new = self.normalize_weights(coefficients)

            # Применяем сглаживание
            w_smoothed = self.apply_smoothing(w_new, old_weights)

            # Оценка улучшения
            acc_old = self.evaluate_weights(X, y, old_weights)
            acc_new = self.evaluate_weights(X, y, w_smoothed)
            improvement = (acc_new - acc_old) * 100

            # Создаём результат
            result = CalibrationResult(
                weights=w_smoothed,
                n_samples=len(X),
                win_rate=win_rate,
                improvement=improvement,
                coefficients=coefficients,
                old_weights=old_weights
            )

            # Сохраняем в историю
            self.calibration_history.append(result)
            self.last_weights = w_smoothed

            logger.info(
                f"[WeightCalibrator] Калибровка завершена: "
                f"samples={len(X)}, win_rate={win_rate:.2%}, "
                f"improvement={improvement:+.1f}%"
            )
            logger.info(
                f"[WeightCalibrator] Новые веса: "
                f"M={w_smoothed[0]:.3f}, S={w_smoothed[1]:.3f}, "
                f"B={w_smoothed[2]:.3f}, R={w_smoothed[3]:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"[WeightCalibrator] Ошибка калибровки: {e}", exc_info=True)
            return None

    def should_calibrate(self, n_closed: int, last_calibration_at: int) -> bool:
        """
        Проверяет, нужна ли калибровка

        Args:
            n_closed: Текущее количество закрытых позиций
            last_calibration_at: Количество закрытых позиций при последней калибровке

        Returns:
            True если нужно калибровать
        """
        # Калибруем каждые min_samples новых позиций
        return (n_closed - last_calibration_at) >= self.min_samples

    def get_stats(self) -> dict:
        """Возвращает статистику калибровок"""
        if not self.calibration_history:
            return {
                'total_calibrations': 0,
                'current_weights': None
            }

        latest = self.calibration_history[-1]

        return {
            'total_calibrations': len(self.calibration_history),
            'current_weights': latest.weights.tolist(),
            'last_n_samples': latest.n_samples,
            'last_win_rate': latest.win_rate,
            'last_improvement': latest.improvement,
            'weights_history': [
                {
                    'weights': r.weights.tolist(),
                    'n_samples': r.n_samples,
                    'improvement': r.improvement
                }
                for r in self.calibration_history[-5:]  # Последние 5
            ]
        }
