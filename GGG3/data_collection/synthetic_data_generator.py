#!/usr/bin/env python3
"""
Генератор синтетических OHLCV данных для тестирования

ВАЖНО: Это НЕ для production! Используется только для:
1. Тестирования pipeline без доступа к Binance API
2. Проверки работы Feature Builder
3. Тестирования Target Calculator
4. Обучения моделей на синтетических данных

Для реального использования нужны РЕАЛЬНЫЕ данные с Binance!
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import os

# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================

class SyntheticOHLCVGenerator:
    """
    Генератор реалистичных синтетических OHLCV данных

    Использует модель Geometric Brownian Motion с трендами,
    волатильностью и сезонностью для имитации реальных рынков
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed для воспроизводимости
        """
        np.random.seed(seed)
        self.seed = seed

    def generate_price_series(
        self,
        initial_price: float,
        n_candles: int,
        timeframe_minutes: int,
        trend: float = 0.0,
        volatility: float = 0.02,
        mean_reversion: float = 0.1
    ) -> pd.Series:
        """
        Генерирует серию цен с трендом и волатильностью

        Args:
            initial_price: Начальная цена
            n_candles: Количество свечей
            timeframe_minutes: Таймфрейм в минутах (5, 15, 30, 240)
            trend: Тренд (drift) - например, 0.001 = 0.1% рост за свечу
            volatility: Волатильность (std dev) - например, 0.02 = 2%
            mean_reversion: Сила возврата к среднему (0-1)

        Returns:
            pd.Series: Цены закрытия
        """
        # Geometric Brownian Motion с mean reversion
        prices = [initial_price]

        for i in range(n_candles - 1):
            # Random walk component
            random_shock = np.random.normal(trend, volatility)

            # Mean reversion component (тянет к начальной цене)
            mean_rev = -mean_reversion * (prices[-1] / initial_price - 1)

            # Combine
            returns = random_shock + mean_rev

            # Calculate new price
            new_price = prices[-1] * (1 + returns)
            prices.append(new_price)

        return pd.Series(prices)

    def generate_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        n_candles: int = 500,
        initial_price: float = None,
        trend: float = 0.0,
        volatility: float = 0.02
    ) -> pd.DataFrame:
        """
        Генерирует OHLCV данные для одного символа и таймфрейма

        Args:
            symbol: Символ (например, 'BTCUSDT')
            timeframe: Таймфрейм ('5m', '15m', '30m', '4h')
            n_candles: Количество свечей
            initial_price: Начальная цена (если None - используется типичная)
            trend: Тренд (0 = боковик, + = восходящий, - = нисходящий)
            volatility: Волатильность

        Returns:
            pd.DataFrame: OHLCV данные
        """
        # Типичные цены для популярных монет (для реалистичности)
        typical_prices = {
            'BTCUSDT': 60000,
            'ETHUSDT': 3000,
            'BNBUSDT': 400,
            'SOLUSDT': 150,
            'ADAUSDT': 0.5,
            'XRPUSDT': 0.6,
            'DOTUSDT': 7,
            'DOGEUSDT': 0.08,
            'AVAXUSDT': 35,
            'MATICUSDT': 0.8,
        }

        if initial_price is None:
            initial_price = typical_prices.get(symbol, 100)

        # Определяем таймфрейм в минутах
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        tf_minutes = timeframe_map.get(timeframe, 5)

        # Генерируем close prices
        close = self.generate_price_series(
            initial_price=initial_price,
            n_candles=n_candles,
            timeframe_minutes=tf_minutes,
            trend=trend,
            volatility=volatility
        )

        # Генерируем high, low, open с реалистичными отклонениями
        open_prices = []
        high_prices = []
        low_prices = []

        for i in range(n_candles):
            c = close.iloc[i]

            if i == 0:
                o = c
            else:
                # Open близко к предыдущему close, но может быть gap
                gap = np.random.normal(0, volatility * 0.3)
                o = close.iloc[i-1] * (1 + gap)

            # High и Low относительно open и close
            price_range = abs(c - o) / o if o != 0 else volatility
            range_multiplier = 1.5 + np.random.exponential(0.5)

            h = max(o, c) * (1 + price_range * range_multiplier * np.random.random())
            l = min(o, c) * (1 - price_range * range_multiplier * np.random.random())

            # Убеждаемся что high >= low
            h = max(h, l + abs(l) * 0.0001)

            open_prices.append(o)
            high_prices.append(h)
            low_prices.append(l)

        # Генерируем volume (коррелирует с волатильностью)
        base_volume = 1000000  # Базовый объем
        volume = []
        for i in range(n_candles):
            # Объем растет при больших движениях цены
            price_change = abs(close.iloc[i] / open_prices[i] - 1) if open_prices[i] != 0 else 0
            vol_multiplier = 1 + price_change * 10

            # Random component
            vol = base_volume * vol_multiplier * (0.5 + np.random.random())
            volume.append(vol)

        # Генерируем timestamps (в прошлое)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=tf_minutes * n_candles)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=n_candles)

        # Создаем DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close.values,
            'volume': volume
        })

        return df


# ============================================================================
# BATCH GENERATION
# ============================================================================

def generate_synthetic_dataset(
    output_dir: str = None,
    n_symbols: int = 50,
    timeframes: List[str] = ['5m', '15m', '30m', '4h'],
    n_candles: int = 500
):
    """
    Генерирует синтетический датасет для множества символов

    Args:
        output_dir: Директория для сохранения (по умолчанию: относительный путь)
        n_symbols: Количество символов для генерации
        timeframes: Список таймфреймов
        n_candles: Количество свечей на таймфрейм
    """
    # Используем относительный путь если не указан явно
    if output_dir is None:
        project_root = Path(__file__).parent.parent  # GGG3/
        output_dir = str(project_root / 'data' / 'historical')

    os.makedirs(output_dir, exist_ok=True)

    # Популярные символы
    top_symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
        'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
        'LINKUSDT', 'UNIUSDT', 'LTCUSDT', 'ATOMUSDT', 'ETCUSDT',
        'XLMUSDT', 'VETUSDT', 'TRXUSDT', 'FILUSDT', 'ICPUSDT',
        'AAVEUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'THETAUSDT',
        'ALGOUSDT', 'EGLDUSDT', 'FTMUSDT', 'HBARUSDT', 'NEARUSDT',
        'FLOWUSDT', 'ENJUSDT', 'CHZUSDT', 'RUNEUSDT', 'COMPUSDT',
        'ZRXUSDT', 'SNXUSDT', 'MKRUSDT', 'CRVUSDT', 'SUSHIUSDT',
        'YFIUSDT', '1INCHUSDT', 'BATUSDT', 'ZECUSDT', 'DASHUSDT',
        'IOTAUSDT', 'WAVESUSDT', 'OMGUSDT', 'ZILUSDT', 'ONTUSDT'
    ]

    symbols = top_symbols[:n_symbols]

    generator = SyntheticOHLCVGenerator(seed=42)

    print(f"Генерация синтетических данных для {len(symbols)} символов...")
    print(f"Таймфреймы: {timeframes}")
    print(f"Свечей на таймфрейм: {n_candles}")
    print(f"Директория: {output_dir}\n")

    from tqdm import tqdm

    for symbol in tqdm(symbols, desc="Symbols"):
        # Случайные параметры для разнообразия
        trend = np.random.uniform(-0.0005, 0.0005)  # Слабый тренд
        volatility = np.random.uniform(0.015, 0.030)  # 1.5-3% волатильность

        for tf in timeframes:
            try:
                df = generator.generate_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    n_candles=n_candles,
                    trend=trend,
                    volatility=volatility
                )

                # Сохраняем в parquet
                filename = f"{output_dir}/{symbol}_{tf}.parquet"
                df.to_parquet(filename, index=False)

            except Exception as e:
                print(f"❌ Ошибка для {symbol} {tf}: {e}")

    # Создаем индекс
    create_data_index(output_dir)

    print(f"\n✅ Синтетические данные сгенерированы!")
    print(f"   Символов: {len(symbols)}")
    print(f"   Файлов: {len(symbols) * len(timeframes)}")
    print(f"   Директория: {output_dir}")


def create_data_index(data_dir: str):
    """
    Создает индекс данных (data_index.json)

    Формат:
    {
      "BTCUSDT": {
        "5m": "path/to/file.parquet",
        "15m": "...",
        "last_update": "2025-11-06T12:00:00",
        "rows_4h": 500
      },
      ...
    }
    """
    import glob
    import json

    index = {}

    for file_path in glob.glob(f"{data_dir}/*.parquet"):
        filename = os.path.basename(file_path)
        # BTCUSDT_5m.parquet -> BTCUSDT, 5m
        symbol, tf = filename.replace('.parquet', '').rsplit('_', 1)

        if symbol not in index:
            index[symbol] = {}

        df = pd.read_parquet(file_path)
        index[symbol][tf] = file_path
        index[symbol][f'rows_{tf}'] = len(df)
        index[symbol]['last_update'] = df['timestamp'].iloc[-1].isoformat()

    # Сохраняем индекс
    index_path = f"{data_dir}/data_index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\n✅ Индекс создан: {index_path}")
    print(f"   Символов в индексе: {len(index)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic OHLCV data for testing')

    # Вычисляем относительный путь по умолчанию
    default_output_dir = str(Path(__file__).parent.parent / 'data' / 'historical')

    parser.add_argument('--output-dir', type=str,
                       default=default_output_dir,
                       help='Output directory for data')
    parser.add_argument('--n-symbols', type=int, default=50,
                       help='Number of symbols to generate')
    parser.add_argument('--n-candles', type=int, default=500,
                       help='Number of candles per timeframe')
    parser.add_argument('--timeframes', type=str, nargs='+',
                       default=['5m', '15m', '30m', '4h'],
                       help='Timeframes to generate')

    args = parser.parse_args()

    print("=" * 80)
    print("SYNTHETIC DATA GENERATOR")
    print("=" * 80)
    print("\n⚠️  ВАЖНО: Это СИНТЕТИЧЕСКИЕ данные для ТЕСТИРОВАНИЯ!")
    print("   Для production используйте РЕАЛЬНЫЕ данные с Binance!\n")
    print("=" * 80)
    print()

    generate_synthetic_dataset(
        output_dir=args.output_dir,
        n_symbols=args.n_symbols,
        timeframes=args.timeframes,
        n_candles=args.n_candles
    )

    print("\n" + "=" * 80)
    print("✅ ГОТОВО!")
    print("=" * 80)
    print(f"\nДанные сохранены в: {args.output_dir}")
    print("\nСледующие шаги:")
    print("1. Проверьте данные: ls -lh data/historical/")
    print("2. Сгенерируйте датасет: python features/target_calculator.py")
    print("3. Обучите модели: python models/train_experts.py")
    print("4. Запустите paper trading: python paper_trading/paper_bot_main.py")
