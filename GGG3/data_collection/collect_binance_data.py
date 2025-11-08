#!/usr/bin/env python3
"""
Сборщик исторических данных для Binance

Функционал:
- Первичная загрузка данных для всех пар
- Обновление существующих данных (дозагрузка новых свечей)
- Поддержка нескольких таймфреймов
- Создание индекса данных
- Rate limiting для защиты от бана
"""

import sys
import os
import time
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

import pandas as pd
from tqdm import tqdm

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))

from binance_api.client import BinanceClient
from binance_api.exceptions import BinanceAPIError

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ФИЛЬТРЫ ДЛЯ ТОРГУЕМЫХ ПАР
# ============================================================================

# Список стейблкоинов (исключаем из торговли)
STABLECOINS = [
    'USDCUSDT',
    'BUSDUSDT',
    'TUSDUSDT',
    'USDPUSDT',
    'DAIUSDT',
    'FDUSDUSDT',
    'EURUSDT',
]

# Blacklist (проблемные монеты)
BLACKLIST = [
    # Добавьте проблемные пары сюда
]


def get_all_tradeable_pairs(
    client: BinanceClient,
    min_volume_24h: float = 1_000_000,
    exclude_stablecoins: bool = True
) -> List[str]:
    """
    Получает список всех торгуемых пар с фильтрами

    Args:
        client: Binance API клиент
        min_volume_24h: Минимальный объём торгов за 24ч (в USDT)
        exclude_stablecoins: Исключить стейблкоины

    Фильтры:
    - Только USDT пары
    - Объем 24h > min_volume_24h
    - Исключить стейблкоины
    - Исключить пары из blacklist

    Returns:
        List[str]: Список символов (например, ['BTCUSDT', 'ETHUSDT', ...])
        Обычно ~300-400 пар
    """
    logger.info(f"Получение списка торгуемых пар (min_volume={min_volume_24h:,.0f} USDT)...")

    try:
        # Получаем все USDT пары через API
        pairs = client.get_all_usdt_pairs(min_volume_24h=min_volume_24h)

        logger.info(f"Получено {len(pairs)} пар от API")

        # Фильтр: исключаем стейблкоины
        if exclude_stablecoins:
            pairs = [p for p in pairs if p not in STABLECOINS]
            logger.info(f"После исключения стейблкоинов: {len(pairs)} пар")

        # Фильтр: исключаем blacklist
        pairs = [p for p in pairs if p not in BLACKLIST]
        logger.info(f"После исключения blacklist: {len(pairs)} пар")

        # Сортируем для стабильности
        pairs = sorted(pairs)

        logger.info(f"✅ Итого торгуемых пар: {len(pairs)}")

        return pairs

    except Exception as e:
        logger.error(f"Ошибка при получении списка пар: {e}")
        raise


# ============================================================================
# ЗАГРУЗКА OHLCV ДАННЫХ
# ============================================================================

def download_ohlcv_data(
    client: BinanceClient,
    symbol: str,
    timeframes: List[str],
    output_dir: str,
    limit: int = 500
) -> bool:
    """
    Скачивает OHLCV данные для всех таймфреймов

    Args:
        client: Binance API клиент
        symbol: Символ пары (например, 'BTCUSDT')
        timeframes: Список таймфреймов (например, ['5m', '15m', '30m', '4h'])
        output_dir: Директория для сохранения
        limit: Количество свечей для загрузки (max 1000, default 500)

    Saves:
        {output_dir}/{symbol}_5m.parquet
        {output_dir}/{symbol}_15m.parquet
        {output_dir}/{symbol}_30m.parquet
        {output_dir}/{symbol}_4h.parquet

    Returns:
        bool: True если все таймфреймы загружены успешно
    """
    os.makedirs(output_dir, exist_ok=True)

    success = True

    for tf in timeframes:
        try:
            # Скачиваем OHLCV данные
            df = client.get_ohlcv(symbol, tf, limit=limit)

            if df is None or len(df) == 0:
                logger.warning(f"✗ {symbol} {tf}: пустой результат")
                success = False
                continue

            # Сохраняем в parquet (быстрый формат с сжатием)
            file_path = os.path.join(output_dir, f"{symbol}_{tf}.parquet")
            df.to_parquet(file_path, index=False, compression='snappy')

            logger.debug(f"✓ {symbol} {tf}: {len(df)} свечей")

            # Rate limiting
            time.sleep(0.1)

        except BinanceAPIError as e:
            logger.error(f"✗ {symbol} {tf}: Binance API ошибка - {e}")
            success = False

        except Exception as e:
            logger.error(f"✗ {symbol} {tf}: {e}")
            success = False

    return success


# ============================================================================
# ОБНОВЛЕНИЕ СУЩЕСТВУЮЩИХ ДАННЫХ
# ============================================================================

def update_ohlcv_data(
    client: BinanceClient,
    symbol: str,
    timeframes: List[str],
    output_dir: str
) -> bool:
    """
    Обновляет существующие OHLCV данные (дозагружает новые свечи)

    Логика:
    1. Читает существующий файл
    2. Определяет последнюю временную метку
    3. Загружает новые свечи после последней метки
    4. Объединяет старые и новые данные
    5. Удаляет дубликаты
    6. Сохраняет обновлённый файл

    Args:
        client: Binance API клиент
        symbol: Символ пары
        timeframes: Список таймфреймов
        output_dir: Директория с данными

    Returns:
        bool: True если все таймфреймы обновлены успешно
    """
    success = True

    for tf in timeframes:
        try:
            file_path = os.path.join(output_dir, f"{symbol}_{tf}.parquet")

            # Проверяем, существует ли файл
            if not os.path.exists(file_path):
                logger.warning(f"⚠️  {symbol} {tf}: файл не существует, пропускаем обновление")
                continue

            # Читаем существующие данные
            existing_df = pd.read_parquet(file_path)

            if len(existing_df) == 0:
                logger.warning(f"⚠️  {symbol} {tf}: файл пуст, пропускаем обновление")
                continue

            # Получаем последнюю временную метку
            last_timestamp = existing_df['timestamp'].max()
            logger.debug(f"{symbol} {tf}: последняя метка {last_timestamp}")

            # Загружаем новые данные (последние 500 свечей)
            # Это гарантирует, что мы получим пересечение и новые данные
            new_df = client.get_ohlcv(symbol, tf, limit=500)

            if new_df is None or len(new_df) == 0:
                logger.warning(f"✗ {symbol} {tf}: не удалось загрузить новые данные")
                success = False
                continue

            # Фильтруем только новые свечи (после последней метки)
            new_df = new_df[new_df['timestamp'] > last_timestamp]

            if len(new_df) == 0:
                logger.debug(f"✓ {symbol} {tf}: нет новых данных")
                continue

            # Объединяем старые и новые данные
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Удаляем дубликаты по timestamp (оставляем последние значения)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')

            # Сортируем по timestamp
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

            # Сохраняем обновлённые данные
            combined_df.to_parquet(file_path, index=False, compression='snappy')

            logger.info(f"✓ {symbol} {tf}: добавлено {len(new_df)} новых свечей (всего {len(combined_df)})")

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"✗ {symbol} {tf}: ошибка обновления - {e}")
            success = False

    return success


# ============================================================================
# ПЕРВИЧНАЯ ЗАГРУЗКА ВСЕХ ДАННЫХ
# ============================================================================

def collect_all_data(
    output_dir: str = None,
    timeframes: Optional[List[str]] = None,
    min_volume_24h: float = 1_000_000,
    limit: int = 500,
    use_testnet: bool = False
) -> Dict[str, Any]:
    """
    Главная функция первичной загрузки всех данных

    Args:
        output_dir: Директория для сохранения данных (по умолчанию: относительный путь)
        timeframes: Список таймфреймов (по умолчанию ['5m', '15m', '30m', '4h'])
        min_volume_24h: Минимальный объём торгов за 24ч
        limit: Количество свечей для загрузки
        use_testnet: Использовать testnet (для тестирования)

    Returns:
        Dict: Статистика загрузки
            {
                'total_pairs': int,
                'success_pairs': int,
                'failed_pairs': List[str],
                'timeframes': List[str],
                'output_dir': str,
                'timestamp': str
            }
    """
    logger.info("=" * 80)
    logger.info("ПЕРВИЧНАЯ ЗАГРУЗКА ДАННЫХ С BINANCE")
    logger.info("=" * 80)

    # Параметры по умолчанию
    if output_dir is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent  # GGG3/
        output_dir = str(project_root / 'data' / 'historical')

    if timeframes is None:
        timeframes = ['5m', '15m', '30m', '4h']

    logger.info(f"Таймфреймы: {timeframes}")
    logger.info(f"Лимит свечей: {limit}")
    logger.info(f"Минимальный объём: {min_volume_24h:,.0f} USDT")
    logger.info(f"Директория: {output_dir}")

    # Создаём Binance клиент
    # Для загрузки данных API ключи не нужны
    client = BinanceClient(
        api_key="",
        secret_key="",
        testnet=use_testnet
    )

    # Получаем список торгуемых пар
    logger.info("\n" + "=" * 80)
    logger.info("ЭТАП 1: Получение списка торгуемых пар")
    logger.info("=" * 80)

    pairs = get_all_tradeable_pairs(
        client=client,
        min_volume_24h=min_volume_24h
    )

    logger.info(f"\n✅ Найдено {len(pairs)} торгуемых пар")

    # Скачиваем данные для всех пар
    logger.info("\n" + "=" * 80)
    logger.info("ЭТАП 2: Загрузка OHLCV данных")
    logger.info("=" * 80)
    logger.info(f"Загрузка данных для {len(pairs)} пар × {len(timeframes)} таймфреймов...")

    success_count = 0
    failed_pairs = []

    # Используем tqdm для прогресс-бара
    for symbol in tqdm(pairs, desc="Загрузка данных"):
        try:
            success = download_ohlcv_data(
                client=client,
                symbol=symbol,
                timeframes=timeframes,
                output_dir=output_dir,
                limit=limit
            )

            if success:
                success_count += 1
            else:
                failed_pairs.append(symbol)

            # Rate limiting между парами
            time.sleep(0.2)

        except Exception as e:
            logger.error(f"Ошибка при загрузке {symbol}: {e}")
            failed_pairs.append(symbol)

    # Создаём индекс данных
    logger.info("\n" + "=" * 80)
    logger.info("ЭТАП 3: Создание индекса данных")
    logger.info("=" * 80)

    index_info = create_data_index(output_dir)

    # Итоговая статистика
    logger.info("\n" + "=" * 80)
    logger.info("ИТОГИ ЗАГРУЗКИ")
    logger.info("=" * 80)
    logger.info(f"✅ Успешно загружено: {success_count}/{len(pairs)} пар")

    if failed_pairs:
        logger.warning(f"❌ Не удалось загрузить: {len(failed_pairs)} пар")
        logger.warning(f"   Список: {failed_pairs[:10]}..." if len(failed_pairs) > 10 else f"   Список: {failed_pairs}")

    logger.info(f"📁 Директория: {output_dir}")
    logger.info(f"📊 Индекс: {index_info['index_path']}")
    logger.info(f"🔢 Всего символов в индексе: {index_info['total_symbols']}")

    return {
        'total_pairs': len(pairs),
        'success_pairs': success_count,
        'failed_pairs': failed_pairs,
        'timeframes': timeframes,
        'output_dir': output_dir,
        'timestamp': datetime.now().isoformat(),
        'index_info': index_info
    }


# ============================================================================
# ОБНОВЛЕНИЕ ВСЕХ ДАННЫХ
# ============================================================================

def update_all_data(
    data_dir: str = None,
    timeframes: Optional[List[str]] = None,
    use_testnet: bool = False
) -> Dict[str, Any]:
    """
    Обновление всех существующих данных (дозагрузка новых свечей)

    Логика:
    1. Читает индекс данных (data_index.json)
    2. Для каждого символа обновляет данные по всем таймфреймам
    3. Обновляет индекс

    Args:
        data_dir: Директория с данными (по умолчанию: относительный путь)
        timeframes: Список таймфреймов (по умолчанию ['5m', '15m', '30m', '4h'])
        use_testnet: Использовать testnet

    Returns:
        Dict: Статистика обновления
    """
    logger.info("=" * 80)
    logger.info("ОБНОВЛЕНИЕ ДАННЫХ BINANCE")
    logger.info("=" * 80)

    # Параметры по умолчанию
    if data_dir is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent  # GGG3/
        data_dir = str(project_root / 'data' / 'historical')

    if timeframes is None:
        timeframes = ['5m', '15m', '30m', '4h']

    logger.info(f"Таймфреймы: {timeframes}")
    logger.info(f"Директория: {data_dir}")

    # Создаём Binance клиент
    client = BinanceClient(
        api_key="",
        secret_key="",
        testnet=use_testnet
    )

    # Читаем индекс данных
    index_path = os.path.join(data_dir, 'data_index.json')

    if not os.path.exists(index_path):
        logger.error(f"❌ Индекс не найден: {index_path}")
        logger.info("Сначала выполните первичную загрузку: collect_all_data()")
        return {
            'error': 'Index not found',
            'index_path': index_path
        }

    with open(index_path, 'r') as f:
        index = json.load(f)

    symbols = list(index.keys())
    logger.info(f"Найдено символов в индексе: {len(symbols)}")

    # Обновляем данные для всех символов
    logger.info("\n" + "=" * 80)
    logger.info("ОБНОВЛЕНИЕ OHLCV ДАННЫХ")
    logger.info("=" * 80)

    success_count = 0
    failed_symbols = []

    for symbol in tqdm(symbols, desc="Обновление данных"):
        try:
            success = update_ohlcv_data(
                client=client,
                symbol=symbol,
                timeframes=timeframes,
                output_dir=data_dir
            )

            if success:
                success_count += 1
            else:
                failed_symbols.append(symbol)

            # Rate limiting
            time.sleep(0.2)

        except Exception as e:
            logger.error(f"Ошибка при обновлении {symbol}: {e}")
            failed_symbols.append(symbol)

    # Обновляем индекс
    logger.info("\n" + "=" * 80)
    logger.info("ОБНОВЛЕНИЕ ИНДЕКСА")
    logger.info("=" * 80)

    index_info = create_data_index(data_dir)

    # Итоговая статистика
    logger.info("\n" + "=" * 80)
    logger.info("ИТОГИ ОБНОВЛЕНИЯ")
    logger.info("=" * 80)
    logger.info(f"✅ Успешно обновлено: {success_count}/{len(symbols)} символов")

    if failed_symbols:
        logger.warning(f"❌ Не удалось обновить: {len(failed_symbols)} символов")
        logger.warning(f"   Список: {failed_symbols[:10]}..." if len(failed_symbols) > 10 else f"   Список: {failed_symbols}")

    logger.info(f"📁 Директория: {data_dir}")
    logger.info(f"📊 Индекс: {index_info['index_path']}")
    logger.info(f"🔢 Всего символов в индексе: {index_info['total_symbols']}")

    return {
        'total_symbols': len(symbols),
        'success_symbols': success_count,
        'failed_symbols': failed_symbols,
        'timeframes': timeframes,
        'data_dir': data_dir,
        'timestamp': datetime.now().isoformat(),
        'index_info': index_info
    }


# ============================================================================
# СОЗДАНИЕ ИНДЕКСА ДАННЫХ
# ============================================================================

def create_data_index(data_dir: str) -> Dict[str, Any]:
    """
    Создает индекс data_index.json

    Индекс содержит информацию о всех загруженных данных:
    - Список файлов для каждого символа
    - Количество строк в каждом файле
    - Временной диапазон данных
    - Дата последнего обновления

    Args:
        data_dir: Директория с данными

    Returns:
        Dict: Информация об индексе

    Saves:
        {data_dir}/data_index.json

    Example index structure:
    {
      "BTCUSDT": {
        "5m": "BTCUSDT_5m.parquet",
        "15m": "BTCUSDT_15m.parquet",
        "30m": "BTCUSDT_30m.parquet",
        "4h": "BTCUSDT_4h.parquet",
        "rows_5m": 500,
        "rows_15m": 500,
        "rows_30m": 500,
        "rows_4h": 500,
        "first_timestamp": "2025-08-15T10:00:00",
        "last_timestamp": "2025-11-06T18:00:00",
        "last_update": "2025-11-06T19:00:00"
      },
      ...
    }
    """
    logger.info(f"Создание индекса данных: {data_dir}")

    index = {}

    # Поиск всех parquet файлов
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

    logger.info(f"Найдено файлов: {len(parquet_files)}")

    for file_path in parquet_files:
        try:
            filename = os.path.basename(file_path)

            # Парсинг имени файла: BTCUSDT_5m.parquet -> BTCUSDT, 5m
            name_without_ext = filename.replace('.parquet', '')
            parts = name_without_ext.rsplit('_', 1)

            if len(parts) != 2:
                logger.warning(f"Некорректное имя файла: {filename}")
                continue

            symbol, timeframe = parts

            # Создаём запись для символа, если её ещё нет
            if symbol not in index:
                index[symbol] = {}

            # Читаем данные для получения метаинформации
            df = pd.read_parquet(file_path)

            if len(df) == 0:
                logger.warning(f"Пустой файл: {filename}")
                continue

            # Добавляем информацию о файле
            index[symbol][timeframe] = filename
            index[symbol][f'rows_{timeframe}'] = len(df)

            # Временной диапазон (используем таймфрейм 4h как основной)
            if timeframe == '4h':
                index[symbol]['first_timestamp'] = df['timestamp'].iloc[0].isoformat()
                index[symbol]['last_timestamp'] = df['timestamp'].iloc[-1].isoformat()

            # Дата обновления
            index[symbol]['last_update'] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Ошибка при обработке {file_path}: {e}")

    # Сохраняем индекс
    index_path = os.path.join(data_dir, 'data_index.json')

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    logger.info(f"✅ Индекс создан: {index_path}")
    logger.info(f"   Всего символов: {len(index)}")

    # Статистика по таймфреймам
    timeframe_stats = {}
    for symbol_data in index.values():
        for key in symbol_data.keys():
            if not key.startswith('rows_') and key not in ['first_timestamp', 'last_timestamp', 'last_update']:
                timeframe_stats[key] = timeframe_stats.get(key, 0) + 1

    logger.info(f"   Статистика по таймфреймам:")
    for tf, count in sorted(timeframe_stats.items()):
        logger.info(f"     - {tf}: {count} символов")

    return {
        'index_path': index_path,
        'total_symbols': len(index),
        'timeframe_stats': timeframe_stats
    }


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_data_stats(data_dir: str = None) -> Dict[str, Any]:
    """
    Получает статистику по загруженным данным

    Args:
        data_dir: Директория с данными (по умолчанию: относительный путь)

    Returns:
        Dict: Статистика
    """
    # Используем относительный путь если не указан явно
    if data_dir is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent  # GGG3/
        data_dir = str(project_root / 'data' / 'historical')

    index_path = os.path.join(data_dir, 'data_index.json')

    if not os.path.exists(index_path):
        return {'error': 'Index not found', 'index_path': index_path}

    with open(index_path, 'r') as f:
        index = json.load(f)

    # Подсчёт файлов
    total_files = sum(
        len([k for k in v.keys() if not k.startswith('rows_') and k not in ['first_timestamp', 'last_timestamp', 'last_update']])
        for v in index.values()
    )

    # Подсчёт общего размера
    total_size_mb = 0
    for file_path in glob.glob(os.path.join(data_dir, "*.parquet")):
        total_size_mb += os.path.getsize(file_path) / (1024 * 1024)

    return {
        'total_symbols': len(index),
        'total_files': total_files,
        'total_size_mb': round(total_size_mb, 2),
        'data_dir': data_dir,
        'index_path': index_path
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse

    # Вычисляем относительный путь по умолчанию
    default_data_dir = str(Path(__file__).parent.parent / 'data' / 'historical')

    parser = argparse.ArgumentParser(description='Binance Data Collector')
    parser.add_argument(
        '--mode',
        type=str,
        default='collect',
        choices=['collect', 'update', 'stats'],
        help='Режим работы: collect (первичная загрузка), update (обновление), stats (статистика)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=default_data_dir,
        help='Директория для сохранения данных'
    )
    parser.add_argument(
        '--min-volume',
        type=float,
        default=1_000_000,
        help='Минимальный объём торгов за 24ч (USDT)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=500,
        help='Количество свечей для загрузки'
    )
    parser.add_argument(
        '--testnet',
        action='store_true',
        help='Использовать testnet'
    )

    args = parser.parse_args()

    if args.mode == 'collect':
        # Первичная загрузка данных
        result = collect_all_data(
            output_dir=args.output_dir,
            min_volume_24h=args.min_volume,
            limit=args.limit,
            use_testnet=args.testnet
        )
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТ ЗАГРУЗКИ:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'update':
        # Обновление существующих данных
        result = update_all_data(
            data_dir=args.output_dir,
            use_testnet=args.testnet
        )
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТ ОБНОВЛЕНИЯ:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == 'stats':
        # Статистика по данным
        stats = get_data_stats(data_dir=args.output_dir)
        print("\n" + "=" * 80)
        print("СТАТИСТИКА ДАННЫХ:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
