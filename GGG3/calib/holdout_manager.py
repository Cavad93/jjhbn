# calib/holdout_manager.py
# -*- coding: utf-8 -*-
"""
Калибратор с hold-out валидацией: обучаем на 80%, валидируем на 20%

Цель: Избежать переобучения при выборе калибратора.
Последние 20% данных (самые свежие) используются для валидации.
"""
import numpy as np
from typing import Tuple, Optional
from .selector import CalibratorSelector

class HoldoutCalibManager:
    """
    Менеджер калибровки с hold-out валидацией.
    
    Принцип:
    - Данные НЕ перемешиваются (важно для временных рядов)
    - Первые 80% = train set
    - Последние 20% = hold-out set (самые свежие данные)
    - Калибратор выбирается на train, оценивается на hold-out
    """
    
    def __init__(self, train_ratio=0.8):
        """
        Args:
            train_ratio: Доля данных для обучения (по умолчанию 0.8 = 80%)
        """
        self.train_ratio = train_ratio
        self.selector = CalibratorSelector()
        self.calibrator = None
        self.holdout_nll = None
        self.train_nll = None
        self.n_train = 0
        self.n_holdout = 0
        
    def fit_with_holdout(self, p_raw: np.ndarray, y: np.ndarray) -> Tuple[Optional[object], Optional[float]]:
        """
        Обучить калибратор с hold-out валидацией.
        
        Args:
            p_raw: Сырые вероятности (до калибровки)
            y: Истинные метки (0 или 1)
            
        Returns:
            (calibrator, holdout_nll): Обученный калибратор и NLL на hold-out
            Если недостаточно данных: (None, None)
        """
        n = len(y)
        n_train = int(n * self.train_ratio)
        
        # Важно: НЕ перемешиваем, берем последовательно
        # Причина: временные ряды, самые свежие данные должны быть в holdout
        p_train = p_raw[:n_train]
        y_train = y[:n_train]
        p_holdout = p_raw[n_train:]
        y_holdout = y[n_train:]
        
        # Проверяем минимальные требования
        if len(p_train) < 50:
            print(f"[holdout] Not enough train data: {len(p_train)} < 50")
            return None, None
            
        if len(p_holdout) < 10:
            print(f"[holdout] Not enough holdout data: {len(p_holdout)} < 10")
            return None, None
        
        # Обучаем калибратор на train set
        try:
            method, cal, metrics = self.selector.fit_pick(p_train, y_train)
            self.train_nll = metrics.get('nll', None)
        except Exception as e:
            print(f"[holdout] Calibrator training failed: {e}")
            return None, None
        
        # Оцениваем на hold-out set
        try:
            p_cal = cal.predict_proba(p_holdout)
            holdout_nll = self._nll(y_holdout, p_cal)
        except Exception as e:
            print(f"[holdout] Holdout evaluation failed: {e}")
            return None, None
        
        # Сохраняем результаты
        self.calibrator = cal
        self.holdout_nll = holdout_nll
        self.n_train = len(p_train)
        self.n_holdout = len(p_holdout)
        
        # Проверка на переобучение
        if self.train_nll is not None and holdout_nll > self.train_nll * 1.3:
            print(f"[holdout] WARNING: Overfitting detected! "
                  f"Train NLL={self.train_nll:.4f}, Holdout NLL={holdout_nll:.4f}")
        
        print(f"[holdout] Calibrator '{method}' trained: "
              f"train_n={self.n_train}, holdout_n={self.n_holdout}, "
              f"train_nll={self.train_nll:.4f}, holdout_nll={holdout_nll:.4f}")
        
        return cal, holdout_nll
    
    def transform(self, p_raw: float) -> float:
        """
        Применить калибровку к сырой вероятности.
        
        Args:
            p_raw: Сырая вероятность (до калибровки)
            
        Returns:
            Калиброванная вероятность
        """
        if self.calibrator is None:
            return float(np.clip(p_raw, 1e-6, 1.0 - 1e-6))
        
        try:
            p_cal = self.calibrator.predict_proba([p_raw])[0]
            return float(np.clip(p_cal, 1e-6, 1.0 - 1e-6))
        except Exception:
            return float(np.clip(p_raw, 1e-6, 1.0 - 1e-6))
    
    @staticmethod
    def _nll(y: np.ndarray, p: np.ndarray) -> float:
        """
        Вычислить Negative Log-Likelihood (NLL).
        
        Args:
            y: Истинные метки (0 или 1)
            p: Предсказанные вероятности
            
        Returns:
            NLL (меньше = лучше)
        """
        p = np.clip(p, 1e-12, 1 - 1e-12)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(nll)
    
    def get_stats(self) -> dict:
        """
        Получить статистику по последнему обучению.
        
        Returns:
            Словарь с метриками
        """
        return {
            'n_train': self.n_train,
            'n_holdout': self.n_holdout,
            'train_nll': self.train_nll,
            'holdout_nll': self.holdout_nll,
            'overfitting_ratio': (self.holdout_nll / self.train_nll 
                                  if self.train_nll and self.train_nll > 0 
                                  else None),
            'is_ready': self.calibrator is not None
        }