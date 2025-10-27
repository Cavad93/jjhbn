# -*- coding: utf-8 -*-
from __future__ import annotations
import os, pickle, math
from typing import Optional, List, Tuple
import numpy as np

# Опциональные зависимости
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    HAVE_SK = True
except Exception:
    IsotonicRegression = None
    LogisticRegression = None
    HAVE_SK = False

def _logit(p: float) -> float:
    """Безопасное вычисление logit с защитой от крайних значений"""
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    return math.log(p/(1.0 - p))

def _sigmoid(z: float) -> float:
    """Численно стабильная sigmoid функция"""
    z = max(min(z, 60.0), -60.0)  # Клипируем для избежания overflow
    if z >= 0:
        exp_neg_z = math.exp(-z)
        return 1.0 / (1.0 + exp_neg_z)
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

class _BaseCal:
    """Базовый класс калибратора с улучшенной валидацией данных"""
    
    def __init__(self, window: int = 2000):
        self.window = int(window)
        self.P: List[float] = []
        self.Y: List[int] = []
        self.ready: bool = False
        # Добавляем счетчики для мониторинга
        self._total_observations = 0
        self._successful_fits = 0
        self._failed_fits = 0

    def observe(self, p_raw: float, y: int):
        """Добавление наблюдения с валидацией"""
        # Валидация входных данных
        if not (0.0 <= p_raw <= 1.0):
            print(f"[Calibrator] Warning: invalid probability {p_raw}, skipping")
            return
        
        if y not in (0, 1):
            print(f"[Calibrator] Warning: invalid label {y}, skipping")
            return
        
        self.P.append(float(p_raw))
        self.Y.append(int(y))
        self._total_observations += 1
        
        # Ограничение размера окна
        if len(self.P) > self.window:
            self.P = self.P[-self.window:]
            self.Y = self.Y[-self.window:]

    def check_data_quality(
        self, 
        min_per_class_ratio: float = 0.2,
        min_std: float = 0.01,
        min_absolute: int = 20
    ) -> Tuple[bool, str]:
        """
        Проверяет качество данных перед обучением.
        
        Returns:
            (is_valid, reason) - флаг валидности и причина отказа
        """
        if len(self.P) == 0:
            return False, "no_data"
        
        # Проверка баланса классов
        n_pos = sum(self.Y)
        n_neg = len(self.Y) - n_pos
        total = len(self.Y)
        
        # Минимум примеров каждого класса
        min_per_class = max(min_absolute, int(total * min_per_class_ratio))
        
        if n_pos < min_per_class:
            return False, f"insufficient_positive_samples_{n_pos}<{min_per_class}"
        if n_neg < min_per_class:
            return False, f"insufficient_negative_samples_{n_neg}<{min_per_class}"
        
        # Проверка разнообразия предсказаний
        p_array = np.array(self.P)
        p_std = np.std(p_array)
        
        if p_std < min_std:
            return False, f"predictions_too_uniform_std={p_std:.4f}"
        
        # Проверка на экстремальные значения
        p_mean = np.mean(p_array)
        if p_mean < 0.05 or p_mean > 0.95:
            return False, f"predictions_too_extreme_mean={p_mean:.4f}"
        
        # Проверка на вырожденность (все предсказания в узком диапазоне)
        p_range = np.max(p_array) - np.min(p_array)
        if p_range < 0.1:
            return False, f"predictions_range_too_narrow={p_range:.4f}"
        
        return True, "ok"

    def save(self, path: str):
        """Атомарное сохранение состояния"""
        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(self, f)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[Calibrator] Failed to save: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    @staticmethod
    def load(path: str) -> Optional["_BaseCal"]:
        """Безопасная загрузка состояния"""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[Calibrator] Failed to load from {path}: {e}")
            return None

    def get_stats(self) -> dict:
        """Получить статистику калибратора"""
        if len(self.Y) == 0:
            return {
                "n_samples": 0,
                "n_positive": 0,
                "n_negative": 0,
                "ready": self.ready,
                "success_rate": 0.0
            }
        
        n_pos = sum(self.Y)
        n_neg = len(self.Y) - n_pos
        success_rate = (self._successful_fits / max(1, self._successful_fits + self._failed_fits))
        
        return {
            "n_samples": len(self.Y),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "ready": self.ready,
            "success_rate": success_rate,
            "total_observations": self._total_observations,
            "successful_fits": self._successful_fits,
            "failed_fits": self._failed_fits
        }

    def transform(self, p_raw: float) -> float:
        """Трансформация вероятности (должна быть переопределена)"""
        raise NotImplementedError

    def maybe_fit(self, min_samples: int = 200, every: int = 100) -> bool:
        """Попытка обучения (должна быть переопределена)"""
        raise NotImplementedError


# --- 1) Platt Calibration: sigmoid(A * logit(p_raw) + B) ---
class PlattCalibrator(_BaseCal):
    """Platt scaling с улучшенной стабильностью"""
    
    def __init__(self, window: int = 2000):
        super().__init__(window=window)
        self.A: float = 1.0
        self.B: float = 0.0
        self._since = 0
        self._last_loss = float('inf')

    def __setstate__(self, state):
        """Обратная совместимость при загрузке старых pickle"""
        self.__dict__.update(state)
        if not hasattr(self, '_since'):
            self._since = 0
        if not hasattr(self, '_total_observations'):
            self._total_observations = len(self.P)
        if not hasattr(self, '_successful_fits'):
            self._successful_fits = 0
        if not hasattr(self, '_failed_fits'):
            self._failed_fits = 0

    def transform(self, p_raw: float) -> float:
        """Применение калибровки"""
        if not self.ready:
            return float(min(max(p_raw, 1e-6), 1.0 - 1e-6))
        
        z = self.A * _logit(p_raw) + self.B
        return _sigmoid(z)

    def maybe_fit(self, min_samples: int = 200, every: int = 100) -> bool:
        """Обучение с проверкой качества данных"""
        self._since += 1
        
        # Проверка частоты обучения
        if len(self.P) < min_samples or self._since < every:
            return False
        
        # НОВОЕ: Проверка качества данных
        is_valid, reason = self.check_data_quality()
        if not is_valid:
            if self._since >= every * 2:  # Логируем только если долго не обучались
                print(f"[PlattCalibrator] Skipping fit: {reason}")
            return False
        
        self._since = 0
        
        try:
            X = np.array([[_logit(p)] for p in self.P], dtype=np.float64)
            y = np.array(self.Y, dtype=np.int32)
            
            # НОВОЕ: Проверка на NaN/Inf в фичах
            if not np.all(np.isfinite(X)):
                print("[PlattCalibrator] Warning: non-finite values in features")
                self._failed_fits += 1
                return False
            
            if HAVE_SK and LogisticRegression is not None:
                # Используем sklearn с регуляризацией
                clf = LogisticRegression(
                    solver="lbfgs",
                    max_iter=300,
                    C=1.0,  # Регуляризация
                    class_weight='balanced'  # НОВОЕ: балансировка классов
                )
                clf.fit(X, y)
                
                # Проверка на вырожденность коэффициентов
                if not np.isfinite(clf.coef_[0,0]) or not np.isfinite(clf.intercept_[0]):
                    print("[PlattCalibrator] Warning: non-finite coefficients")
                    self._failed_fits += 1
                    return False
                
                self.A = float(clf.coef_[0,0])
                self.B = float(clf.intercept_[0])
                
            else:
                # Ручной градиентный спуск с адаптивным learning rate
                A, B = self.A, self.B
                lr = 0.05
                best_A, best_B = A, B
                best_loss = float('inf')
                
                for iteration in range(200):
                    z = np.clip(A * X[:,0] + B, -60, 60)
                    p = 1.0 / (1.0 + np.exp(-z))
                    
                    # Защита от численных проблем
                    p = np.clip(p, 1e-7, 1 - 1e-7)
                    
                    # Взвешенная loss для балансировки классов
                    weights = np.ones_like(y, dtype=float)
                    n_pos = np.sum(y == 1)
                    n_neg = np.sum(y == 0)
                    weights[y == 1] = 1.0 / max(1, n_pos)
                    weights[y == 0] = 1.0 / max(1, n_neg)
                    weights /= weights.sum()
                    
                    # Cross-entropy loss
                    loss = -np.sum(weights * (y * np.log(p) + (1-y) * np.log(1-p)))
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_A, best_B = A, B
                    
                    # Градиенты с весами
                    gA = np.sum(weights * (p - y) * X[:,0])
                    gB = np.sum(weights * (p - y))
                    
                    # Адаптивный learning rate
                    if iteration > 50 and loss > self._last_loss:
                        lr *= 0.95  # Уменьшаем скорость если loss растет
                    
                    # Обновление с клипированием
                    A -= lr * np.clip(gA, -10, 10)
                    B -= lr * np.clip(gB, -10, 10)
                    
                    # Ограничение диапазона параметров
                    A = np.clip(A, -10, 10)
                    B = np.clip(B, -10, 10)
                
                self.A, self.B = float(best_A), float(best_B)
                self._last_loss = best_loss
            
            self.ready = True
            self._successful_fits += 1
            return True
            
        except Exception as e:
            print(f"[PlattCalibrator] Fit failed: {e}")
            self._failed_fits += 1
            return False


# --- 2) Isotonic Regression Calibration ---
class IsotonicCalibrator(_BaseCal):
    """Isotonic regression с проверками монотонности"""
    
    def __init__(self, window: int = 4000):
        super().__init__(window=window)
        if not HAVE_SK or IsotonicRegression is None:
            raise RuntimeError("IsotonicCalibrator requires sklearn. Install with: pip install scikit-learn")
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self._since = 0

    def __setstate__(self, state):
        """Обратная совместимость при загрузке старых pickle"""
        self.__dict__.update(state)
        if not hasattr(self, '_since'):
            self._since = 0
        if not hasattr(self, '_total_observations'):
            self._total_observations = len(self.P)
        if not hasattr(self, '_successful_fits'):
            self._successful_fits = 0
        if not hasattr(self, '_failed_fits'):
            self._failed_fits = 0

    def transform(self, p_raw: float) -> float:
        """Применение калибровки"""
        if not self.ready:
            return float(min(max(p_raw, 1e-6), 1.0 - 1e-6))
        
        try:
            calibrated = float(self.iso.predict([p_raw])[0])
            # Дополнительная защита от экстремальных значений
            return float(min(max(calibrated, 1e-6), 1.0 - 1e-6))
        except Exception:
            return float(min(max(p_raw, 1e-6), 1.0 - 1e-6))

    def maybe_fit(self, min_samples: int = 400, every: int = 200) -> bool:
        """Обучение с проверкой монотонности"""
        self._since += 1
        
        if len(self.P) < min_samples or self._since < every:
            return False
        
        # НОВОЕ: Проверка качества данных с более строгими требованиями для isotonic
        is_valid, reason = self.check_data_quality(min_per_class_ratio=0.25, min_absolute=50)
        if not is_valid:
            if self._since >= every * 2:
                print(f"[IsotonicCalibrator] Skipping fit: {reason}")
            return False
        
        self._since = 0
        
        try:
            X = np.array(self.P, dtype=np.float64)
            y = np.array(self.Y, dtype=np.int32)
            
            # НОВОЕ: Проверка достаточного покрытия диапазона вероятностей
            bins = np.histogram(X, bins=10)[0]
            non_empty_bins = np.sum(bins > 0)
            if non_empty_bins < 5:  # Минимум 5 непустых бинов из 10
                print(f"[IsotonicCalibrator] Insufficient probability coverage: {non_empty_bins}/10 bins")
                self._failed_fits += 1
                return False
            
            self.iso.fit(X, y)
            
            # НОВОЕ: Проверка монотонности результата
            test_points = np.linspace(0.01, 0.99, 20)
            calibrated = self.iso.predict(test_points)
            
            # Проверка что калибровка не слишком экстремальная
            if np.all(calibrated < 0.1) or np.all(calibrated > 0.9):
                print("[IsotonicCalibrator] Calibration too extreme, reverting")
                self._failed_fits += 1
                return False
            
            self.ready = True
            self._successful_fits += 1
            return True
            
        except Exception as e:
            print(f"[IsotonicCalibrator] Fit failed: {e}")
            self._failed_fits += 1
            return False


# --- 3) Temperature Scaling Calibration ---
class TemperatureCalibrator(_BaseCal):
    """Temperature scaling с защитой от вырождения"""
    
    def __init__(self, window: int = 2000):
        super().__init__(window=window)
        self.T: float = 1.0
        self._since = 0
        self._T_history: List[float] = []  # История температур для стабилизации

    def __setstate__(self, state):
        """Обратная совместимость при загрузке старых pickle"""
        self.__dict__.update(state)
        if not hasattr(self, '_since'):
            self._since = 0
        if not hasattr(self, '_T_history'):
            self._T_history = []
        if not hasattr(self, '_total_observations'):
            self._total_observations = len(self.P)
        if not hasattr(self, '_successful_fits'):
            self._successful_fits = 0
        if not hasattr(self, '_failed_fits'):
            self._failed_fits = 0

    def transform(self, p_raw: float) -> float:
        """Применение температурной калибровки"""
        if not self.ready or self.T <= 0:
            return float(min(max(p_raw, 1e-6), 1.0 - 1e-6))
        
        z = _logit(p_raw) / max(1e-6, self.T)
        return _sigmoid(z)

    def maybe_fit(self, min_samples: int = 200, every: int = 100) -> bool:
        """Обучение с защитой от экстремальных температур"""
        self._since += 1
        
        if len(self.P) < min_samples or self._since < every:
            return False
        
        # НОВОЕ: Проверка качества данных
        is_valid, reason = self.check_data_quality()
        if not is_valid:
            if self._since >= every * 2:
                print(f"[TemperatureCalibrator] Skipping fit: {reason}")
            return False
        
        self._since = 0
        
        try:
            z = np.array([_logit(p) for p in self.P], dtype=np.float64)
            y = np.array(self.Y, dtype=np.float64)
            
            # Проверка на NaN/Inf
            if not np.all(np.isfinite(z)):
                print("[TemperatureCalibrator] Warning: non-finite logits")
                self._failed_fits += 1
                return False
            
            best_T, best_loss = self.T, float("inf")
            
            # НОВОЕ: Адаптивный диапазон поиска на основе истории
            if len(self._T_history) > 0:
                mean_T = np.mean(self._T_history[-5:])  # Последние 5 значений
                std_T = np.std(self._T_history[-5:]) if len(self._T_history) > 1 else 0.5
                T_min = max(0.1, mean_T - 2 * std_T)
                T_max = min(5.0, mean_T + 2 * std_T)
            else:
                T_min, T_max = 0.3, 3.0
            
            # Поиск оптимальной температуры
            for T in np.linspace(T_min, T_max, 31):
                zz = np.clip(z / T, -60, 60)
                p = 1.0 / (1.0 + np.exp(-zz))
                p = np.clip(p, 1e-9, 1 - 1e-9)
                
                # Взвешенная loss для балансировки классов
                weights = np.ones_like(y)
                weights[y == 1] = 1.0 / max(1, np.sum(y == 1))
                weights[y == 0] = 1.0 / max(1, np.sum(y == 0))
                weights /= weights.sum()
                
                loss = -np.sum(weights * (y * np.log(p) + (1-y) * np.log(1-p)))
                
                if loss < best_loss:
                    best_loss, best_T = loss, float(T)
            
            # НОВОЕ: Проверка на разумность температуры
            if best_T < 0.1 or best_T > 10.0:
                print(f"[TemperatureCalibrator] Extreme temperature {best_T:.3f}, using default")
                best_T = 1.0
            
            # НОВОЕ: Сглаживание температуры с историей
            if len(self._T_history) > 0:
                # Экспоненциальное сглаживание
                alpha = 0.7  # Вес нового значения
                self.T = alpha * best_T + (1 - alpha) * self.T
            else:
                self.T = best_T
            
            self._T_history.append(best_T)
            if len(self._T_history) > 20:  # Храним только последние 20
                self._T_history = self._T_history[-20:]
            
            self.ready = True
            self._successful_fits += 1
            return True
            
        except Exception as e:
            print(f"[TemperatureCalibrator] Fit failed: {e}")
            self._failed_fits += 1
            return False


def make_calibrator(method: str = "platt", **kwargs) -> _BaseCal:
    """
    Создаёт калибратор по имени метода с дополнительными параметрами.
    
    Args:
        method: "platt", "isotonic", "temperature" или "logistic" (алиас для platt)
        **kwargs: дополнительные параметры для конструктора
    
    Returns:
        Инстанс калибратора
    
    Raises:
        ValueError: Если метод неизвестен
        RuntimeError: Если метод требует sklearn, но он не установлен
    """
    # Алиасы для обратной совместимости
    if method in ("logistic", "sigmoid"):
        method = "platt"
    
    if method == "isotonic":
        window = kwargs.get('window', 4000)
        return IsotonicCalibrator(window=window)
    elif method == "temperature":
        window = kwargs.get('window', 2000)
        return TemperatureCalibrator(window=window)
    elif method == "platt":
        window = kwargs.get('window', 2000)
        return PlattCalibrator(window=window)
    else:
        raise ValueError(f"Unknown calibrator method: {method}. Use 'platt', 'isotonic' or 'temperature'")


# Экспорт публичного API
__all__ = [
    'make_calibrator',
    'PlattCalibrator', 
    'IsotonicCalibrator',
    'TemperatureCalibrator',
    '_BaseCal',  # Для наследования пользовательских калибраторов
    'HAVE_SK'  # Для проверки доступности sklearn
]
