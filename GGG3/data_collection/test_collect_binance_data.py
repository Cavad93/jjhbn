#!/usr/bin/env python3
"""
Тесты для collect_binance_data.py

Проверка функционала сбора и обновления данных
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

# Добавляем родительскую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.collect_binance_data import (
    get_all_tradeable_pairs,
    download_ohlcv_data,
    update_ohlcv_data,
    create_data_index,
    get_data_stats,
    STABLECOINS,
    BLACKLIST
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_data_dir():
    """Временная директория для тестов"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_client():
    """Мок Binance клиента"""
    client = Mock()
    return client


@pytest.fixture
def sample_ohlcv_df():
    """Пример OHLCV данных"""
    dates = pd.date_range(start='2025-11-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [100.0] * 100,
        'high': [101.0] * 100,
        'low': [99.0] * 100,
        'close': [100.5] * 100,
        'volume': [1000.0] * 100
    })
    return df


# ============================================================================
# ТЕСТЫ
# ============================================================================

def test_imports():
    """Тест 1: Проверка импортов"""
    print("Тест 1: Проверка импортов...", end=" ")

    try:
        from data_collection import collect_binance_data
        assert hasattr(collect_binance_data, 'get_all_tradeable_pairs')
        assert hasattr(collect_binance_data, 'download_ohlcv_data')
        assert hasattr(collect_binance_data, 'update_ohlcv_data')
        assert hasattr(collect_binance_data, 'collect_all_data')
        assert hasattr(collect_binance_data, 'update_all_data')
        assert hasattr(collect_binance_data, 'create_data_index')

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_stablecoins_list():
    """Тест 2: Проверка списка стейблкоинов"""
    print("Тест 2: Проверка списка стейблкоинов...", end=" ")

    try:
        assert isinstance(STABLECOINS, list)
        assert len(STABLECOINS) > 0

        # Проверяем, что основные стейблкоины есть в списке
        required_stablecoins = ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT']
        for coin in required_stablecoins:
            assert coin in STABLECOINS, f"{coin} отсутствует в списке стейблкоинов"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_get_all_tradeable_pairs(mock_client):
    """Тест 3: Получение списка торгуемых пар"""
    print("Тест 3: Получение списка торгуемых пар...", end=" ")

    try:
        # Мокаем ответ API
        mock_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT',
            'USDCUSDT',  # Стейблкоин - должен быть исключён
        ]
        mock_client.get_all_usdt_pairs.return_value = mock_pairs

        # Вызываем функцию
        pairs = get_all_tradeable_pairs(
            client=mock_client,
            min_volume_24h=1_000_000
        )

        # Проверки
        assert isinstance(pairs, list)
        assert len(pairs) > 0

        # Стейблкоин должен быть исключён
        assert 'USDCUSDT' not in pairs

        # Обычные монеты должны быть
        assert 'BTCUSDT' in pairs
        assert 'ETHUSDT' in pairs

        # Проверка, что список отсортирован
        assert pairs == sorted(pairs)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_download_ohlcv_data(mock_client, sample_ohlcv_df, temp_data_dir):
    """Тест 4: Загрузка OHLCV данных"""
    print("Тест 4: Загрузка OHLCV данных...", end=" ")

    try:
        # Мокаем ответ API
        mock_client.get_ohlcv.return_value = sample_ohlcv_df

        # Вызываем функцию
        success = download_ohlcv_data(
            client=mock_client,
            symbol='BTCUSDT',
            timeframes=['5m', '15m'],
            output_dir=temp_data_dir,
            limit=500
        )

        # Проверки
        assert success == True

        # Проверяем, что файлы созданы
        assert os.path.exists(os.path.join(temp_data_dir, 'BTCUSDT_5m.parquet'))
        assert os.path.exists(os.path.join(temp_data_dir, 'BTCUSDT_15m.parquet'))

        # Читаем файл и проверяем данные
        df = pd.read_parquet(os.path.join(temp_data_dir, 'BTCUSDT_5m.parquet'))
        assert len(df) == 100
        assert 'timestamp' in df.columns
        assert 'open' in df.columns
        assert 'close' in df.columns

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_download_ohlcv_data_empty_response(mock_client, temp_data_dir):
    """Тест 5: Загрузка с пустым ответом"""
    print("Тест 5: Загрузка с пустым ответом...", end=" ")

    try:
        # Мокаем пустой ответ
        mock_client.get_ohlcv.return_value = None

        # Вызываем функцию
        success = download_ohlcv_data(
            client=mock_client,
            symbol='TESTUSDT',
            timeframes=['5m'],
            output_dir=temp_data_dir
        )

        # Должно вернуть False
        assert success == False

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_update_ohlcv_data(mock_client, sample_ohlcv_df, temp_data_dir):
    """Тест 6: Обновление OHLCV данных"""
    print("Тест 6: Обновление OHLCV данных...", end=" ")

    try:
        # Создаём начальный файл со старыми данными
        old_dates = pd.date_range(start='2025-11-01', periods=50, freq='5min')
        old_df = pd.DataFrame({
            'timestamp': old_dates,
            'open': [100.0] * 50,
            'high': [101.0] * 50,
            'low': [99.0] * 50,
            'close': [100.5] * 50,
            'volume': [1000.0] * 50
        })

        file_path = os.path.join(temp_data_dir, 'BTCUSDT_5m.parquet')
        old_df.to_parquet(file_path, index=False)

        # Мокаем новые данные (с пересечением и новыми свечами)
        new_dates = pd.date_range(start='2025-11-01 03:00:00', periods=100, freq='5min')
        new_df = pd.DataFrame({
            'timestamp': new_dates,
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100
        })
        mock_client.get_ohlcv.return_value = new_df

        # Вызываем функцию обновления
        success = update_ohlcv_data(
            client=mock_client,
            symbol='BTCUSDT',
            timeframes=['5m'],
            output_dir=temp_data_dir
        )

        # Проверки
        assert success == True

        # Читаем обновлённый файл
        updated_df = pd.read_parquet(file_path)

        # Должно быть больше данных
        assert len(updated_df) > len(old_df)

        # Не должно быть дубликатов
        assert len(updated_df) == len(updated_df['timestamp'].unique())

        # Данные должны быть отсортированы
        assert updated_df['timestamp'].is_monotonic_increasing

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_create_data_index(temp_data_dir, sample_ohlcv_df):
    """Тест 7: Создание индекса данных"""
    print("Тест 7: Создание индекса данных...", end=" ")

    try:
        # Создаём тестовые файлы
        symbols = ['BTCUSDT', 'ETHUSDT']
        timeframes = ['5m', '15m', '4h']

        for symbol in symbols:
            for tf in timeframes:
                file_path = os.path.join(temp_data_dir, f'{symbol}_{tf}.parquet')
                sample_ohlcv_df.to_parquet(file_path, index=False)

        # Создаём индекс
        index_info = create_data_index(temp_data_dir)

        # Проверки
        assert 'index_path' in index_info
        assert 'total_symbols' in index_info
        assert index_info['total_symbols'] == 2

        # Проверяем, что индекс создан
        index_path = os.path.join(temp_data_dir, 'data_index.json')
        assert os.path.exists(index_path)

        # Читаем индекс
        with open(index_path, 'r') as f:
            index = json.load(f)

        # Проверяем структуру индекса
        assert 'BTCUSDT' in index
        assert 'ETHUSDT' in index

        assert '5m' in index['BTCUSDT']
        assert '15m' in index['BTCUSDT']
        assert '4h' in index['BTCUSDT']

        assert 'rows_5m' in index['BTCUSDT']
        assert index['BTCUSDT']['rows_5m'] == 100

        assert 'last_update' in index['BTCUSDT']

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_get_data_stats(temp_data_dir, sample_ohlcv_df):
    """Тест 8: Получение статистики данных"""
    print("Тест 8: Получение статистики данных...", end=" ")

    try:
        # Создаём тестовые файлы
        file_path = os.path.join(temp_data_dir, 'BTCUSDT_5m.parquet')
        sample_ohlcv_df.to_parquet(file_path, index=False)

        # Создаём индекс
        create_data_index(temp_data_dir)

        # Получаем статистику
        stats = get_data_stats(temp_data_dir)

        # Проверки
        assert 'total_symbols' in stats
        assert 'total_files' in stats
        assert 'total_size_mb' in stats
        assert 'data_dir' in stats

        assert stats['total_symbols'] == 1
        assert stats['total_files'] == 1
        assert stats['total_size_mb'] > 0

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_get_data_stats_no_index(temp_data_dir):
    """Тест 9: Статистика без индекса"""
    print("Тест 9: Статистика без индекса...", end=" ")

    try:
        # Получаем статистику без индекса
        stats = get_data_stats(temp_data_dir)

        # Должна быть ошибка
        assert 'error' in stats
        assert stats['error'] == 'Index not found'

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_file_naming_convention(temp_data_dir, sample_ohlcv_df, mock_client):
    """Тест 10: Проверка соглашения о именовании файлов"""
    print("Тест 10: Проверка соглашения о именовании файлов...", end=" ")

    try:
        mock_client.get_ohlcv.return_value = sample_ohlcv_df

        symbol = 'BTCUSDT'
        timeframe = '4h'

        download_ohlcv_data(
            client=mock_client,
            symbol=symbol,
            timeframes=[timeframe],
            output_dir=temp_data_dir
        )

        # Проверяем имя файла
        expected_filename = f'{symbol}_{timeframe}.parquet'
        file_path = os.path.join(temp_data_dir, expected_filename)

        assert os.path.exists(file_path), f"Файл {expected_filename} не найден"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_parquet_format(temp_data_dir, sample_ohlcv_df, mock_client):
    """Тест 11: Проверка формата Parquet"""
    print("Тест 11: Проверка формата Parquet...", end=" ")

    try:
        mock_client.get_ohlcv.return_value = sample_ohlcv_df

        download_ohlcv_data(
            client=mock_client,
            symbol='BTCUSDT',
            timeframes=['5m'],
            output_dir=temp_data_dir
        )

        file_path = os.path.join(temp_data_dir, 'BTCUSDT_5m.parquet')

        # Читаем файл
        df = pd.read_parquet(file_path)

        # Проверяем структуру
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in df.columns, f"Колонка {col} отсутствует"

        # Проверяем типы данных
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        assert pd.api.types.is_numeric_dtype(df['open'])
        assert pd.api.types.is_numeric_dtype(df['close'])

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_data_integrity_after_update(temp_data_dir, mock_client):
    """Тест 12: Целостность данных после обновления"""
    print("Тест 12: Целостность данных после обновления...", end=" ")

    try:
        # Создаём начальные данные
        old_dates = pd.date_range(start='2025-11-01', periods=50, freq='5min')
        old_df = pd.DataFrame({
            'timestamp': old_dates,
            'open': [100.0] * 50,
            'high': [101.0] * 50,
            'low': [99.0] * 50,
            'close': [100.5] * 50,
            'volume': [1000.0] * 50
        })

        file_path = os.path.join(temp_data_dir, 'TESTUSDT_5m.parquet')
        old_df.to_parquet(file_path, index=False)

        # Мокаем новые данные
        new_dates = pd.date_range(start='2025-11-01 02:00:00', periods=100, freq='5min')
        new_df = pd.DataFrame({
            'timestamp': new_dates,
            'open': [105.0] * 100,  # Другие значения
            'high': [106.0] * 100,
            'low': [104.0] * 100,
            'close': [105.5] * 100,
            'volume': [2000.0] * 100
        })
        mock_client.get_ohlcv.return_value = new_df

        # Обновляем данные
        update_ohlcv_data(
            client=mock_client,
            symbol='TESTUSDT',
            timeframes=['5m'],
            output_dir=temp_data_dir
        )

        # Читаем обновлённые данные
        updated_df = pd.read_parquet(file_path)

        # Проверяем целостность
        # 1. Нет дубликатов по timestamp
        assert len(updated_df) == len(updated_df['timestamp'].unique())

        # 2. Данные отсортированы
        assert updated_df['timestamp'].is_monotonic_increasing

        # 3. Нет пропущенных значений
        assert updated_df.isnull().sum().sum() == 0

        # 4. Старые данные сохранились
        first_timestamp = updated_df['timestamp'].iloc[0]
        assert first_timestamp == old_dates[0]

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ DATA COLLECTION MODULE")
    print("=" * 80)
    print()

    # Создаём fixtures
    temp_dir = tempfile.mkdtemp()
    mock_client = Mock()

    dates = pd.date_range(start='2025-11-01', periods=100, freq='5min')
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': [100.0] * 100,
        'high': [101.0] * 100,
        'low': [99.0] * 100,
        'close': [100.5] * 100,
        'volume': [1000.0] * 100
    })

    tests = [
        lambda: test_imports(),
        lambda: test_stablecoins_list(),
        lambda: test_get_all_tradeable_pairs(mock_client),
        lambda: test_download_ohlcv_data(mock_client, sample_df, temp_dir),
        lambda: test_download_ohlcv_data_empty_response(mock_client, temp_dir),
        lambda: test_update_ohlcv_data(mock_client, sample_df, temp_dir),
        lambda: test_create_data_index(temp_dir, sample_df),
        lambda: test_get_data_stats(temp_dir, sample_df),
        lambda: test_get_data_stats_no_index(temp_dir),
        lambda: test_file_naming_convention(temp_dir, sample_df, mock_client),
        lambda: test_parquet_format(temp_dir, sample_df, mock_client),
        lambda: test_data_integrity_after_update(temp_dir, mock_client),
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Тест провален с ошибкой: {e}")
            results.append(False)

    # Очистка
    shutil.rmtree(temp_dir, ignore_errors=True)

    print()
    print("=" * 80)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"Пройдено: {passed}/{total}")

    if passed == total:
        print()
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
        return True
    else:
        print()
        print(f"❌ ПРОВАЛЕНО ТЕСТОВ: {total - passed}")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
