"""
Target Calculator для Binance Futures Bot

Модуль для расчёта TP/SL исходов для LONG и SHORT позиций
и генерации датасета для обучения ML моделей.

Автор: Claude Code
Дата: 2025-11-06
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from .builder import BinanceFeatureBuilder
except ImportError:
    import sys
    from pathlib import Path
    # Добавляем путь относительно текущего файла (features/ -> GGG3/)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from features.builder import BinanceFeatureBuilder


def calculate_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Рассчитывает ATR (Average True Range)

    Args:
        df: DataFrame с OHLCV данными
        period: Период для расчёта ATR

    Returns:
        Series с значениями ATR
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_tp_sl_outcome_bidirectional(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    atr_value: float,
    direction: str,
    tp_mult: float = 1.2,
    sl_mult: float = 0.8,
    max_bars: int = 100
) -> Optional[Dict]:
    """
    Определяет исход для LONG или SHORT позиции

    Проверяет достижение TP или SL на каждой последующей свече,
    используя high/low для корректного определения порядка срабатывания.

    Args:
        df: DataFrame с OHLCV данными
        entry_idx: Индекс свечи входа
        entry_price: Цена входа
        atr_value: Значение ATR на момент входа
        direction: 'LONG' или 'SHORT'
        tp_mult: Множитель ATR для TP (default: 1.2)
        sl_mult: Множитель ATR для SL (default: 0.8)
        max_bars: Максимальное количество свечей до таймаута

    Returns:
        dict с результатами:
        {
            'outcome': 1 или 0 (1=TP достигнут, 0=SL достигнут),
            'direction': 'LONG' или 'SHORT',
            'bars_held': количество свечей до выхода,
            'exit_price': цена выхода,
            'pnl_pct': P&L в процентах
        }
        или None если таймаут (не достигли ни TP ни SL)

    Примеры:
        >>> # LONG пример
        >>> result = calculate_tp_sl_outcome_bidirectional(
        ...     df, entry_idx=100, entry_price=50000, atr_value=1000,
        ...     direction='LONG', tp_mult=1.2, sl_mult=0.8
        ... )
        >>> # TP = 50000 + 1.2*1000 = 51200
        >>> # SL = 50000 - 0.8*1000 = 49200

        >>> # SHORT пример
        >>> result = calculate_tp_sl_outcome_bidirectional(
        ...     df, entry_idx=100, entry_price=50000, atr_value=1000,
        ...     direction='SHORT', tp_mult=1.2, sl_mult=0.8
        ... )
        >>> # TP = 50000 - 1.2*1000 = 48800
        >>> # SL = 50000 + 0.8*1000 = 50800
    """

    if direction not in ['LONG', 'SHORT']:
        raise ValueError(f"direction должен быть 'LONG' или 'SHORT', получено: {direction}")

    if entry_idx < 0 or entry_idx >= len(df):
        raise ValueError(f"entry_idx {entry_idx} вне допустимого диапазона [0, {len(df)-1}]")

    if atr_value <= 0:
        raise ValueError(f"atr_value должен быть > 0, получено: {atr_value}")

    # Рассчитываем уровни TP и SL
    if direction == 'LONG':
        tp_price = entry_price + tp_mult * atr_value
        sl_price = entry_price - sl_mult * atr_value

        # Проверяем каждую последующую свечу
        for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            # Проверяем TP (достигнут раньше?)
            # Для LONG: если high >= tp_price, то TP достигнут
            if high >= tp_price:
                # Проверяем, мог ли SL сработать на той же свече
                # SL срабатывает только если low <= sl_price И это произошло раньше
                # В рамках одной свечи предполагаем: сначала low, потом high
                if low <= sl_price:
                    # SL сработал раньше на этой свече
                    return {
                        'outcome': 0,
                        'direction': 'LONG',
                        'bars_held': i - entry_idx,
                        'exit_price': sl_price,
                        'pnl_pct': (sl_price - entry_price) / entry_price * 100
                    }
                else:
                    # TP достигнут
                    return {
                        'outcome': 1,
                        'direction': 'LONG',
                        'bars_held': i - entry_idx,
                        'exit_price': tp_price,
                        'pnl_pct': (tp_price - entry_price) / entry_price * 100
                    }

            # Проверяем SL (если TP не был достигнут)
            if low <= sl_price:
                return {
                    'outcome': 0,
                    'direction': 'LONG',
                    'bars_held': i - entry_idx,
                    'exit_price': sl_price,
                    'pnl_pct': (sl_price - entry_price) / entry_price * 100
                }

    elif direction == 'SHORT':
        tp_price = entry_price - tp_mult * atr_value
        sl_price = entry_price + sl_mult * atr_value

        # Проверяем каждую последующую свечу
        for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            # Проверяем TP (цена падает до tp_price)
            # Для SHORT: если low <= tp_price, то TP достигнут
            if low <= tp_price:
                # Проверяем, мог ли SL сработать на той же свече
                # SL срабатывает если high >= sl_price
                if high >= sl_price:
                    # SL сработал раньше на этой свече
                    return {
                        'outcome': 0,
                        'direction': 'SHORT',
                        'bars_held': i - entry_idx,
                        'exit_price': sl_price,
                        'pnl_pct': (entry_price - sl_price) / entry_price * 100
                    }
                else:
                    # TP достигнут
                    return {
                        'outcome': 1,
                        'direction': 'SHORT',
                        'bars_held': i - entry_idx,
                        'exit_price': tp_price,
                        'pnl_pct': (entry_price - tp_price) / entry_price * 100
                    }

            # Проверяем SL (цена растет до sl_price)
            if high >= sl_price:
                return {
                    'outcome': 0,
                    'direction': 'SHORT',
                    'bars_held': i - entry_idx,
                    'exit_price': sl_price,
                    'pnl_pct': (entry_price - sl_price) / entry_price * 100
                }

    # Таймаут: не достигли ни TP ни SL за max_bars свечей
    return None


def detect_market_phase(df: pd.DataFrame) -> int:
    """
    Определяет фазу рынка по Wyckoff

    Фазы:
    0 - Ranging (флэт)
    1 - Accumulation (накопление)
    2 - Markup (рост)
    3 - Distribution (распределение)
    4 - Markdown (падение)
    5 - Volatility spike (резкая волатильность)

    Args:
        df: DataFrame с OHLCV данными

    Returns:
        int: номер фазы (0-5)
    """
    if len(df) < 20:
        return 0  # Ranging по умолчанию

    # Рассчитываем индикаторы
    close = df['close']
    volume = df['volume']

    # Тренд (EMA 9 vs EMA 21)
    ema9 = close.ewm(span=9).mean()
    ema21 = close.ewm(span=21).mean()
    trend = ema9.iloc[-1] - ema21.iloc[-1]

    # Волатильность (ATR относительно цены)
    atr = calculate_atr(df, period=10)
    atr_pct = (atr.iloc[-1] / close.iloc[-1]) * 100

    # Объём (текущий vs средний)
    vol_ratio = volume.iloc[-1] / volume.iloc[-20:].mean()

    # Изменение цены за последние 10 свечей
    price_change = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]

    # Логика определения фазы (упрощённая и более чувствительная к трендам)
    if atr_pct > 8.0:
        return 5  # Volatility spike

    if abs(price_change) < 0.02 and atr_pct < 3.0:
        return 0  # Ranging

    # Проверяем сильный рост (price_change > 5% OR trend > 0)
    if price_change > 0.05 or (price_change > 0.02 and trend > 0):
        if vol_ratio > 1.3:
            return 2  # Markup (рост с объёмом)
        else:
            return 1  # Accumulation (начало роста)

    # Проверяем сильное падение (price_change < -5% OR trend < 0)
    if price_change < -0.05 or (price_change < -0.02 and trend < 0):
        if vol_ratio > 1.3:
            return 4  # Markdown (падение с объёмом)
        else:
            return 3  # Distribution (начало падения)

    return 0  # По умолчанию Ranging


def prepare_training_dataset(
    data_dir: str,
    output_dir: str,
    tp_mult: float = 1.2,
    sl_mult: float = 0.8,
    max_bars: int = 100,
    min_candles: int = 200,
    train_split: float = 0.8
) -> Tuple[str, str]:
    """
    Создаёт датасет для обучения ML моделей

    Для каждой монеты и каждой свечи (4h):
    1. Рассчитывает 68 фич (multi-TF)
    2. Рассчитывает ATR
    3. Определяет фазу рынка
    4. Для ОБОИХ направлений (LONG и SHORT):
       - Определяет outcome (TP/SL)
       - Если outcome не None: добавляет в датасет

    Args:
        data_dir: Директория с историческими данными (parquet files)
        output_dir: Директория для сохранения датасетов
        tp_mult: Множитель ATR для TP
        sl_mult: Множитель ATR для SL
        max_bars: Максимальное количество свечей до таймаута
        min_candles: Минимальное количество свечей для обработки
        train_split: Доля данных для обучения (остальное - тест)

    Returns:
        Tuple[str, str]: Пути к train и test parquet файлам

    Сохраняет файлы:
        - {output_dir}/binance_dataset_train.parquet
        - {output_dir}/binance_dataset_test.parquet

    Структура датасета:
        - features_0 ... features_67: 68 технических фич
        - outcome: 0 (SL) или 1 (TP)
        - direction: 'LONG' или 'SHORT'
        - phase: фаза рынка (0-5)
        - symbol: название монеты
        - timestamp: временная метка
        - bars_held: количество свечей до выхода
        - pnl_pct: P&L в процентах
        - entry_price: цена входа
        - atr_value: значение ATR
        - atr_pct: ATR в процентах от цены
    """

    print("="*80)
    print("ГЕНЕРАЦИЯ ДАТАСЕТА ДЛЯ BINANCE FUTURES BOT")
    print("="*80)
    print(f"Директория данных: {data_dir}")
    print(f"Директория выходных файлов: {output_dir}")
    print(f"TP множитель: {tp_mult}× ATR")
    print(f"SL множитель: {sl_mult}× ATR")
    print(f"Таймаут: {max_bars} свечей")
    print(f"Train/Test split: {int(train_split*100)}/{int((1-train_split)*100)}%")
    print("="*80)

    # Создаём директорию для выходных файлов
    os.makedirs(output_dir, exist_ok=True)

    # Инициализируем feature builder
    feature_builder = BinanceFeatureBuilder()

    # Находим все символы с полными данными (4 таймфрейма)
    all_files = glob.glob(f"{data_dir}/*_4h.parquet")
    symbols = [os.path.basename(f).replace('_4h.parquet', '') for f in all_files]

    print(f"\nНайдено {len(symbols)} символов с данными 4h")

    # Список для хранения всех примеров
    all_samples = []

    # Статистика
    stats = {
        'total_symbols': len(symbols),
        'processed_symbols': 0,
        'skipped_symbols': 0,
        'total_candles': 0,
        'long_tp': 0,
        'long_sl': 0,
        'short_tp': 0,
        'short_sl': 0,
        'timeouts': 0
    }

    # Обрабатываем каждый символ
    print("\nОбработка символов...")
    for symbol in tqdm(symbols, desc="Символы"):

        try:
            # Загружаем данные всех таймфреймов
            df_5m = pd.read_parquet(f"{data_dir}/{symbol}_5m.parquet")
            df_15m = pd.read_parquet(f"{data_dir}/{symbol}_15m.parquet")
            df_30m = pd.read_parquet(f"{data_dir}/{symbol}_30m.parquet")
            df_4h = pd.read_parquet(f"{data_dir}/{symbol}_4h.parquet")

            # Проверяем минимальную длину
            if len(df_4h) < min_candles:
                stats['skipped_symbols'] += 1
                continue

            # Рассчитываем ATR на 4h
            atr_4h = calculate_atr(df_4h, period=10)

            # Обрабатываем каждую свечу 4h (начиная с min_candles)
            for idx in range(min_candles, len(df_4h) - max_bars):

                # Текущая временная метка и цена
                current_time = df_4h['timestamp'].iloc[idx]
                entry_price = df_4h['close'].iloc[idx]
                atr_value = atr_4h.iloc[idx]

                # Пропускаем если ATR не рассчитан
                if pd.isna(atr_value) or atr_value <= 0:
                    continue

                # Получаем соответствующие данные для других таймфреймов
                # (последние свечи до current_time)
                df_5m_slice = df_5m[df_5m['timestamp'] <= current_time].tail(200)
                df_15m_slice = df_15m[df_15m['timestamp'] <= current_time].tail(200)
                df_30m_slice = df_30m[df_30m['timestamp'] <= current_time].tail(200)
                df_4h_slice = df_4h.iloc[:idx+1].tail(100)

                # Проверяем достаточно ли данных
                if (len(df_5m_slice) < 100 or len(df_15m_slice) < 100 or
                    len(df_30m_slice) < 100 or len(df_4h_slice) < 100):
                    continue

                # Рассчитываем фичи
                try:
                    features = feature_builder.build_extended_features(
                        df_5m=df_5m_slice,
                        df_15m=df_15m_slice,
                        df_30m=df_30m_slice,
                        df_4h=df_4h_slice,
                        current_time=current_time
                    )
                except Exception as e:
                    # Пропускаем если ошибка в расчёте фич
                    continue

                # Определяем фазу рынка
                phase = detect_market_phase(df_4h_slice)

                # ATR в процентах от цены
                atr_pct = (atr_value / entry_price) * 100

                stats['total_candles'] += 1

                # Проверяем BOTH направления (LONG и SHORT)
                for direction in ['LONG', 'SHORT']:

                    # Рассчитываем outcome
                    outcome_data = calculate_tp_sl_outcome_bidirectional(
                        df=df_4h,
                        entry_idx=idx,
                        entry_price=entry_price,
                        atr_value=atr_value,
                        direction=direction,
                        tp_mult=tp_mult,
                        sl_mult=sl_mult,
                        max_bars=max_bars
                    )

                    # Если таймаут (не достигли ни TP ни SL) - пропускаем
                    if outcome_data is None:
                        stats['timeouts'] += 1
                        continue

                    # Обновляем статистику
                    if direction == 'LONG':
                        if outcome_data['outcome'] == 1:
                            stats['long_tp'] += 1
                        else:
                            stats['long_sl'] += 1
                    else:  # SHORT
                        if outcome_data['outcome'] == 1:
                            stats['short_tp'] += 1
                        else:
                            stats['short_sl'] += 1

                    # Формируем пример для датасета
                    sample = {}

                    # Добавляем фичи
                    for i in range(68):
                        sample[f'features_{i}'] = features[i]

                    # Добавляем мета-информацию
                    sample['outcome'] = outcome_data['outcome']
                    sample['direction'] = direction
                    sample['phase'] = phase
                    sample['symbol'] = symbol
                    sample['timestamp'] = current_time
                    sample['bars_held'] = outcome_data['bars_held']
                    sample['pnl_pct'] = outcome_data['pnl_pct']
                    sample['entry_price'] = entry_price
                    sample['atr_value'] = atr_value
                    sample['atr_pct'] = atr_pct

                    all_samples.append(sample)

            stats['processed_symbols'] += 1

        except Exception as e:
            print(f"\n⚠️ Ошибка при обработке {symbol}: {e}")
            stats['skipped_symbols'] += 1
            continue

    # Создаём DataFrame
    print("\n\nСоздание DataFrame...")
    df_dataset = pd.DataFrame(all_samples)

    if len(df_dataset) == 0:
        raise ValueError("Датасет пуст! Проверьте данные и параметры.")

    print(f"Всего примеров: {len(df_dataset)}")
    print(f"\nРаспределение по направлениям:")
    print(df_dataset['direction'].value_counts())
    print(f"\nРаспределение по исходам:")
    print(df_dataset['outcome'].value_counts())

    # Сортируем по времени
    df_dataset = df_dataset.sort_values('timestamp').reset_index(drop=True)

    # Разделяем на train/test по времени (первые 80% - train, последние 20% - test)
    split_idx = int(len(df_dataset) * train_split)

    df_train = df_dataset.iloc[:split_idx].copy()
    df_test = df_dataset.iloc[split_idx:].copy()

    print(f"\nTrain set: {len(df_train)} примеров")
    print(f"Test set: {len(df_test)} примеров")

    # Сохраняем в parquet
    train_path = f"{output_dir}/binance_dataset_train.parquet"
    test_path = f"{output_dir}/binance_dataset_test.parquet"

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    print(f"\n✅ Train датасет сохранён: {train_path}")
    print(f"✅ Test датасет сохранён: {test_path}")

    # Выводим статистику
    print("\n" + "="*80)
    print("СТАТИСТИКА ГЕНЕРАЦИИ")
    print("="*80)
    print(f"Обработано символов: {stats['processed_symbols']}/{stats['total_symbols']}")
    print(f"Пропущено символов: {stats['skipped_symbols']}")
    print(f"Обработано свечей: {stats['total_candles']}")
    print(f"\nРезультаты:")
    print(f"  LONG TP: {stats['long_tp']} ({stats['long_tp']/(stats['long_tp']+stats['long_sl'])*100:.1f}% WR)" if stats['long_tp']+stats['long_sl'] > 0 else "  LONG: нет данных")
    print(f"  LONG SL: {stats['long_sl']}")
    print(f"  SHORT TP: {stats['short_tp']} ({stats['short_tp']/(stats['short_tp']+stats['short_sl'])*100:.1f}% WR)" if stats['short_tp']+stats['short_sl'] > 0 else "  SHORT: нет данных")
    print(f"  SHORT SL: {stats['short_sl']}")
    print(f"  Таймауты: {stats['timeouts']}")

    total_tp = stats['long_tp'] + stats['short_tp']
    total_sl = stats['long_sl'] + stats['short_sl']
    total_valid = total_tp + total_sl

    if total_valid > 0:
        overall_wr = (total_tp / total_valid) * 100
        print(f"\nОбщий Win Rate: {overall_wr:.2f}%")
        print(f"Break-even WR (для R/R={tp_mult/sl_mult:.2f}): 40%")

        if overall_wr > 40:
            print(f"✅ Датасет показывает прибыльность! ({overall_wr:.2f}% > 40%)")
        else:
            print(f"⚠️ Датасет показывает убыточность ({overall_wr:.2f}% < 40%)")

    print("="*80)

    return train_path, test_path


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":

    print("="*80)
    print("ТЕСТ TARGET CALCULATOR")
    print("="*80)

    # Создаём тестовые данные
    np.random.seed(42)
    n_candles = 200

    # Генерируем случайный прайс с трендом
    base_price = 50000
    trend = np.cumsum(np.random.randn(n_candles) * 100)
    prices = base_price + trend

    # OHLCV с реалистичными high/low
    df_test = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_candles, freq='4H'),
        'open': prices,
        'high': prices + np.abs(np.random.randn(n_candles) * 50),
        'low': prices - np.abs(np.random.randn(n_candles) * 50),
        'close': prices + np.random.randn(n_candles) * 20,
        'volume': np.random.randint(1000, 10000, n_candles)
    })

    # Рассчитываем ATR
    atr = calculate_atr(df_test, period=10)

    print("\n1️⃣ Тест LONG позиции")
    print("-" * 80)

    entry_idx = 100
    entry_price = df_test['close'].iloc[entry_idx]
    atr_value = atr.iloc[entry_idx]

    print(f"Entry index: {entry_idx}")
    print(f"Entry price: ${entry_price:.2f}")
    print(f"ATR: ${atr_value:.2f}")

    result_long = calculate_tp_sl_outcome_bidirectional(
        df=df_test,
        entry_idx=entry_idx,
        entry_price=entry_price,
        atr_value=atr_value,
        direction='LONG',
        tp_mult=1.2,
        sl_mult=0.8,
        max_bars=50
    )

    if result_long:
        print(f"\n✅ Результат LONG:")
        print(f"   Outcome: {result_long['outcome']} ({'TP' if result_long['outcome'] == 1 else 'SL'})")
        print(f"   Bars held: {result_long['bars_held']}")
        print(f"   Exit price: ${result_long['exit_price']:.2f}")
        print(f"   P&L: {result_long['pnl_pct']:.2f}%")
    else:
        print("⏱️ Таймаут (не достигли ни TP ни SL)")

    print("\n2️⃣ Тест SHORT позиции")
    print("-" * 80)

    result_short = calculate_tp_sl_outcome_bidirectional(
        df=df_test,
        entry_idx=entry_idx,
        entry_price=entry_price,
        atr_value=atr_value,
        direction='SHORT',
        tp_mult=1.2,
        sl_mult=0.8,
        max_bars=50
    )

    if result_short:
        print(f"\n✅ Результат SHORT:")
        print(f"   Outcome: {result_short['outcome']} ({'TP' if result_short['outcome'] == 1 else 'SL'})")
        print(f"   Bars held: {result_short['bars_held']}")
        print(f"   Exit price: ${result_short['exit_price']:.2f}")
        print(f"   P&L: {result_short['pnl_pct']:.2f}%")
    else:
        print("⏱️ Таймаут (не достигли ни TP ни SL)")

    print("\n3️⃣ Тест фаз рынка")
    print("-" * 80)

    phase = detect_market_phase(df_test.tail(50))
    phase_names = {
        0: 'Ranging (флэт)',
        1: 'Accumulation (накопление)',
        2: 'Markup (рост)',
        3: 'Distribution (распределение)',
        4: 'Markdown (падение)',
        5: 'Volatility spike'
    }

    print(f"Текущая фаза рынка: {phase} - {phase_names[phase]}")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("="*80)

    print("\n💡 Для генерации полного датасета используйте:")
    print("   from features.target_calculator import prepare_training_dataset")
    print("   train_path, test_path = prepare_training_dataset(")
    print("       data_dir='data/historical',  # или используйте относительные пути")
    print("       output_dir='data/datasets'")
    print("   )")
