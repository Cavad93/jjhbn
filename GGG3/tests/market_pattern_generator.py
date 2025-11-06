#!/usr/bin/env python3
"""
Market Pattern Generator

Генератор синтетических рыночных паттернов для тестирования
способности бота находить закономерности
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple


class MarketPatternGenerator:
    """
    Генератор синтетических рыночных данных с известными паттернами

    Создает OHLCV данные с различными рыночными фазами:
    - Восходящий тренд (бот должен находить LONG возможности)
    - Нисходящий тренд (бот должен находить SHORT возможности)
    - Breakout (прорыв уровня)
    - Mean reversion (возврат к средней)
    - Боковое движение (бот должен избегать входов)
    """

    def __init__(self, base_price: float = 35000.0, seed: int = 42):
        self.base_price = base_price
        self.rng = np.random.RandomState(seed)

    def generate_uptrend(self, periods: int = 200) -> pd.DataFrame:
        """
        Генерирует восходящий тренд

        Паттерн: Четкий восходящий тренд с постепенным ростом
        Ожидание: BASE логика должна находить LONG сигналы
        """
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='5min')

        # Восходящий тренд: +0.1% в среднем за свечу
        trend = np.cumsum(self.rng.normal(0.001, 0.003, periods))

        prices = self.base_price * (1 + trend)

        df = self._create_ohlcv(timestamps, prices, trend_type='up')

        return df

    def generate_downtrend(self, periods: int = 200) -> pd.DataFrame:
        """
        Генерирует нисходящий тренд

        Паттерн: Четкий нисходящий тренд с постепенным падением
        Ожидание: BASE логика должна находить SHORT сигналы
        """
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='5min')

        # Нисходящий тренд: -0.1% в среднем за свечу
        trend = np.cumsum(self.rng.normal(-0.001, 0.003, periods))

        prices = self.base_price * (1 + trend)

        df = self._create_ohlcv(timestamps, prices, trend_type='down')

        return df

    def generate_breakout(self, periods: int = 200, breakout_at: int = 150) -> pd.DataFrame:
        """
        Генерирует breakout паттерн

        Паттерн: Боковое движение, затем резкий прорыв вверх
        Ожидание: BASE логика (signal B) должна определить breakout
        """
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='5min')

        prices = np.zeros(periods)

        # Фаза 1: Боковое движение (consolidation)
        for i in range(breakout_at):
            prices[i] = self.base_price + self.rng.normal(0, 50)

        # Фаза 2: Breakout вверх
        breakout_momentum = 0
        for i in range(breakout_at, periods):
            breakout_momentum += self.rng.uniform(0.002, 0.005)  # Сильный рост
            prices[i] = self.base_price * (1 + breakout_momentum) + self.rng.normal(0, 30)

        df = self._create_ohlcv(timestamps, prices, trend_type='breakout')

        return df

    def generate_mean_reversion(self, periods: int = 200) -> pd.DataFrame:
        """
        Генерирует mean reversion паттерн

        Паттерн: Цена отклоняется от средней, затем возвращается
        Ожидание: BASE логика (signal R) должна находить реверсии
        """
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='5min')

        # Среднее значение
        mean_price = self.base_price

        # Цена отклоняется и возвращается к средней
        prices = np.zeros(periods)
        current_deviation = 0

        for i in range(periods):
            # Случайный шок
            if self.rng.random() < 0.05:
                current_deviation += self.rng.uniform(-0.02, 0.02)

            # Возврат к средней (mean reversion force)
            reversion_force = -current_deviation * 0.1
            current_deviation += reversion_force + self.rng.normal(0, 0.002)

            prices[i] = mean_price * (1 + current_deviation)

        df = self._create_ohlcv(timestamps, prices, trend_type='reversion')

        return df

    def generate_sideways(self, periods: int = 200) -> pd.DataFrame:
        """
        Генерирует боковое движение (ranging market)

        Паттерн: Цена двигается в узком диапазоне без тренда
        Ожидание: Бот НЕ должен открывать позиции (низкие сигналы)
        """
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='5min')

        # Боковое движение в диапазоне ±0.5%
        prices = self.base_price + self.rng.normal(0, self.base_price * 0.005, periods)

        df = self._create_ohlcv(timestamps, prices, trend_type='sideways')

        return df

    def generate_volatile_trend(self, periods: int = 200, direction: str = 'up') -> pd.DataFrame:
        """
        Генерирует тренд с высокой волатильностью

        Паттерн: Тренд присутствует, но с сильными колебаниями
        Ожидание: BASE логика должна находить тренд несмотря на шум
        """
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='5min')

        # Базовый тренд
        if direction == 'up':
            trend = np.cumsum(self.rng.normal(0.001, 0.001, periods))
        else:
            trend = np.cumsum(self.rng.normal(-0.001, 0.001, periods))

        # Добавляем высокую волатильность
        volatility = self.rng.normal(0, 0.01, periods)

        prices = self.base_price * (1 + trend + volatility)

        df = self._create_ohlcv(timestamps, prices, trend_type=f'volatile_{direction}')

        return df

    def _create_ohlcv(self, timestamps: pd.DatetimeIndex, close_prices: np.ndarray,
                      trend_type: str) -> pd.DataFrame:
        """
        Создает OHLCV данные из массива цен закрытия
        """
        periods = len(close_prices)

        # Генерируем OHLC на основе close
        open_prices = np.zeros(periods)
        high_prices = np.zeros(periods)
        low_prices = np.zeros(periods)

        for i in range(periods):
            close = close_prices[i]

            # Open - предыдущий close с небольшим гэпом
            if i == 0:
                open_prices[i] = close
            else:
                gap = self.rng.normal(0, close * 0.0005)
                open_prices[i] = close_prices[i-1] + gap

            # High и Low - создаем свечи
            wick_size = abs(self.rng.normal(0, close * 0.003))

            if trend_type in ['up', 'breakout', 'volatile_up']:
                # Бычья свеча
                high_prices[i] = max(open_prices[i], close) + wick_size
                low_prices[i] = min(open_prices[i], close) - wick_size * 0.5
            elif trend_type in ['down', 'volatile_down']:
                # Медвежья свеча
                high_prices[i] = max(open_prices[i], close) + wick_size * 0.5
                low_prices[i] = min(open_prices[i], close) - wick_size
            else:
                # Нейтральная свеча
                high_prices[i] = max(open_prices[i], close) + wick_size
                low_prices[i] = min(open_prices[i], close) - wick_size

        # Volume - больше в трендовых движениях
        base_volume = 1000000
        if trend_type in ['breakout', 'volatile_up', 'volatile_down']:
            volume = self.rng.uniform(base_volume * 1.5, base_volume * 3, periods)
        else:
            volume = self.rng.uniform(base_volume * 0.5, base_volume * 1.5, periods)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })

        df.set_index('timestamp', inplace=True)

        return df

    def generate_all_patterns(self) -> Dict[str, pd.DataFrame]:
        """
        Генерирует все паттерны для тестирования

        Returns:
            Dict с ключами: 'uptrend', 'downtrend', 'breakout',
                           'reversion', 'sideways', 'volatile_up', 'volatile_down'
        """
        patterns = {
            'uptrend': self.generate_uptrend(200),
            'downtrend': self.generate_downtrend(200),
            'breakout': self.generate_breakout(200),
            'reversion': self.generate_mean_reversion(200),
            'sideways': self.generate_sideways(200),
            'volatile_up': self.generate_volatile_trend(200, 'up'),
            'volatile_down': self.generate_volatile_trend(200, 'down'),
        }

        return patterns


def test_pattern_quality():
    """Тест качества генерируемых паттернов"""
    print("="*80)
    print("ТЕСТ ГЕНЕРАТОРА РЫНОЧНЫХ ПАТТЕРНОВ")
    print("="*80)

    generator = MarketPatternGenerator()
    patterns = generator.generate_all_patterns()

    for name, df in patterns.items():
        print(f"\n{name.upper()}:")
        print(f"  Periods: {len(df)}")
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        # Проверяем тренд
        returns = df['close'].pct_change().dropna()
        avg_return = returns.mean()
        volatility = returns.std()

        print(f"  Avg return: {avg_return*100:.4f}%")
        print(f"  Volatility: {volatility*100:.4f}%")

        # Проверяем соответствие ожиданиям
        if name == 'uptrend':
            assert avg_return > 0, "Uptrend должен иметь положительную доходность"
            print("  ✓ Восходящий тренд подтвержден")
        elif name == 'downtrend':
            assert avg_return < 0, "Downtrend должен иметь отрицательную доходность"
            print("  ✓ Нисходящий тренд подтвержден")
        elif name == 'sideways':
            assert abs(avg_return) < 0.0001, "Sideways должен иметь околонулевую доходность"
            print("  ✓ Боковое движение подтверждено")

    print("\n" + "="*80)
    print("✅ ВСЕ ПАТТЕРНЫ СГЕНЕРИРОВАНЫ КОРРЕКТНО")
    print("="*80)


if __name__ == '__main__':
    test_pattern_quality()
