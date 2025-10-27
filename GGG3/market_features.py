# market_features.py
"""
Модуль для расчета рыночных фич для META-модели
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import time


@dataclass
class MarketSnapshot:
    """Снимок рыночных данных в момент времени"""
    timestamp: float
    price: float
    volume: float = 0.0
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    funding_rate: Optional[float] = None
    futures_price: Optional[float] = None


class MarketFeaturesCalculator:
    """
    Вычисляет рыночные фичи для META на основе истории цен и объемов
    """
    
    def __init__(self, 
                 max_history: int = 300,  # 5 часов при 1-минутных данных
                 trend_window: int = 5,    # Окно для тренда (минуты)
                 vol_window: int = 20,     # Окно для волатильности
                 jump_threshold: float = 0.005):  # Порог для jump (0.5%)
        
        self.max_history = max_history
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.jump_threshold = jump_threshold
        
        # История снимков рынка
        self.history: deque[MarketSnapshot] = deque(maxlen=max_history)
        
        # Кэш для EMA волатильности (ускорение)
        self.ema_vol = None
        self.ema_alpha = 0.1
        
        print(f"[MarketFeatures] Инициализирован: "
              f"history={max_history}, trend_win={trend_window}, "
              f"vol_win={vol_window}, jump_thr={jump_threshold}")
    
    def update(self, snapshot: MarketSnapshot) -> None:
        """Добавляет новый снимок рынка в историю"""
        self.history.append(snapshot)
    
    def calculate_features(self) -> Dict[str, float]:
        """
        Вычисляет все 8 рыночных фич
        
        Returns:
            Словарь с фичами для reg_ctx
        """
        if len(self.history) < 2:
            # Недостаточно данных - возвращаем нули
            return self._default_features()
        
        current = self.history[-1]
        
        # 1. Trend features (тренд за последние N минут)
        trend_sign, trend_abs = self._calculate_trend()
        
        # 2. Volatility ratio (волатильность относительно средней)
        vol_ratio = self._calculate_volatility_ratio()
        
        # 3. Jump flag (резкое движение цены)
        jump_flag = self._calculate_jump_flag()
        
        # 4. Order Flow Imbalance (если есть bid/ask данные)
        ofi_sign = self._calculate_ofi()
        
        # 5. Book Imbalance (дисбаланс ордербука)
        book_imb = self._calculate_book_imbalance()
        
        # 6. Basis (спред spot-futures)
        basis_sign = self._calculate_basis()
        
        # 7. Funding rate sign
        funding_sign = self._calculate_funding_sign()
        
        return {
            "trend_sign": trend_sign,
            "trend_abs": trend_abs,
            "vol_ratio": vol_ratio,
            "jump_flag": jump_flag,
            "ofi_sign": ofi_sign,
            "book_imb": book_imb,
            "basis_sign": basis_sign,
            "funding_sign": funding_sign,
        }
    
    def _calculate_trend(self) -> Tuple[float, float]:
        """
        Вычисляет тренд за последние trend_window периодов
        
        Returns:
            (trend_sign, trend_abs) - знак и абсолютная величина тренда
        """
        if len(self.history) < self.trend_window + 1:
            return 0.0, 0.0
        
        current_price = self.history[-1].price
        past_price = self.history[-(self.trend_window + 1)].price
        
        price_change = (current_price - past_price) / past_price
        
        trend_sign = float(np.sign(price_change))
        trend_abs = float(abs(price_change))
        
        return trend_sign, trend_abs
    
    def _calculate_volatility_ratio(self) -> float:
        """
        Вычисляет волатильность относительно средней исторической
        
        Returns:
            vol_ratio - текущая волатильность / средняя волатильность
        """
        if len(self.history) < self.vol_window + 1:
            return 1.0
        
        # Берем последние vol_window цен
        prices = np.array([s.price for s in list(self.history)[-self.vol_window-1:]])
        
        # Вычисляем returns
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) < 2:
            return 1.0
        
        # Текущая волатильность (последние 5 периодов)
        if len(returns) >= 5:
            current_vol = float(np.std(returns[-5:]))
        else:
            current_vol = float(np.std(returns))
        
        # Средняя историческая волатильность
        avg_vol = float(np.std(returns))
        
        # Обновляем EMA волатильности для сглаживания
        if self.ema_vol is None:
            self.ema_vol = avg_vol
        else:
            self.ema_vol = self.ema_alpha * avg_vol + (1 - self.ema_alpha) * self.ema_vol
        
        # Ratio
        if self.ema_vol > 1e-8:
            vol_ratio = float(current_vol / self.ema_vol)
        else:
            vol_ratio = 1.0
        
        # Ограничиваем разумными пределами
        vol_ratio = float(np.clip(vol_ratio, 0.1, 10.0))
        
        return vol_ratio
    
    def _calculate_jump_flag(self) -> float:
        """
        Определяет был ли резкий скачок цены
        
        Returns:
            1.0 если был jump, 0.0 иначе
        """
        if len(self.history) < 2:
            return 0.0
        
        current_price = self.history[-1].price
        prev_price = self.history[-2].price
        
        price_move = abs((current_price - prev_price) / prev_price)
        
        if price_move > self.jump_threshold:
            return 1.0
        else:
            return 0.0
    
    def _calculate_ofi(self) -> float:
        """
        Order Flow Imbalance - дисбаланс потока ордеров
        
        Требует bid/ask данных. Если недоступно, возвращает 0.0
        
        Returns:
            Знак OFI (-1, 0, +1)
        """
        if len(self.history) < 2:
            return 0.0
        
        current = self.history[-1]
        prev = self.history[-2]
        
        # Проверяем наличие bid/ask данных
        if (current.bid_volume is None or current.ask_volume is None or
            prev.bid_volume is None or prev.ask_volume is None):
            return 0.0
        
        # OFI = Δbid_volume - Δask_volume
        delta_bid = current.bid_volume - prev.bid_volume
        delta_ask = current.ask_volume - prev.ask_volume
        
        ofi = delta_bid - delta_ask
        
        # Нормализуем и возвращаем знак
        if abs(ofi) < 1e-6:
            return 0.0
        else:
            return float(np.sign(ofi))
    
    def _calculate_book_imbalance(self) -> float:
        """
        Book Imbalance - дисбаланс ордербука
        
        book_imb = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        
        Returns:
            Значение от -1 (сильный перевес ask) до +1 (сильный перевес bid)
        """
        current = self.history[-1]
        
        if current.bid_volume is None or current.ask_volume is None:
            return 0.0
        
        total = current.bid_volume + current.ask_volume
        
        if total < 1e-6:
            return 0.0
        
        imbalance = (current.bid_volume - current.ask_volume) / total
        
        return float(imbalance)
    
    def _calculate_basis(self) -> float:
        """
        Basis - спред между spot и futures ценой
        
        basis = (futures_price - spot_price) / spot_price
        
        Returns:
            Знак базиса (-1, 0, +1)
        """
        current = self.history[-1]
        
        if current.futures_price is None:
            return 0.0
        
        basis = (current.futures_price - current.price) / current.price
        
        if abs(basis) < 1e-6:
            return 0.0
        else:
            return float(np.sign(basis))
    
    def _calculate_funding_sign(self) -> float:
        """
        Funding Rate Sign - знак ставки фандинга
        
        Returns:
            Знак funding rate (-1, 0, +1)
        """
        current = self.history[-1]
        
        if current.funding_rate is None:
            return 0.0
        
        if abs(current.funding_rate) < 1e-6:
            return 0.0
        else:
            return float(np.sign(current.funding_rate))
    
    def _default_features(self) -> Dict[str, float]:
        """Возвращает дефолтные значения фич (когда недостаточно данных)"""
        return {
            "trend_sign": 0.0,
            "trend_abs": 0.0,
            "vol_ratio": 1.0,
            "jump_flag": 0.0,
            "ofi_sign": 0.0,
            "book_imb": 0.0,
            "basis_sign": 0.0,
            "funding_sign": 0.0,
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Возвращает статистику для мониторинга"""
        return {
            "history_size": len(self.history),
            "max_history": self.max_history,
            "has_orderbook": any(s.bid_volume is not None for s in self.history),
            "has_futures": any(s.futures_price is not None for s in self.history),
            "has_funding": any(s.funding_rate is not None for s in self.history),
        }


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========

if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ MarketFeaturesCalculator")
    print("=" * 60)
    
    # Создаем калькулятор
    calc = MarketFeaturesCalculator()
    
    # Симулируем поступление данных
    base_price = 100000.0
    
    for i in range(30):
        # Симулируем случайное движение цены
        price_change = np.random.randn() * 0.001  # 0.1% std
        price = base_price * (1 + price_change)
        
        # Симулируем объемы
        bid_vol = 10.0 + np.random.rand() * 5.0
        ask_vol = 10.0 + np.random.rand() * 5.0
        
        snapshot = MarketSnapshot(
            timestamp=time.time() + i * 60,  # Каждую минуту
            price=price,
            volume=bid_vol + ask_vol,
            bid_price=price * 0.9999,
            ask_price=price * 1.0001,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            funding_rate=0.0001,
            futures_price=price * 1.0002
        )
        
        calc.update(snapshot)
        base_price = price
    
    # Вычисляем фичи
    features = calc.calculate_features()
    
    print("\nВычисленные фичи:")
    print("-" * 60)
    for key, value in features.items():
        print(f"  {key:<20} = {value:+.6f}")
    
    print("\nСтатистика:")
    print("-" * 60)
    stats = calc.get_stats()
    for key, value in stats.items():
        print(f"  {key:<20} = {value}")
    
    print("\n" + "=" * 60)
    print("✅ Тест пройден!")