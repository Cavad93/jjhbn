"""
Комплексное тестирование доработок Paper Exchange:
1. Атомарное сохранение/загрузка состояния
2. Восстановление из backup при повреждении
3. Параллельная загрузка OHLCV (get_ohlcv_batch)
4. Интеграционные тесты полного цикла
"""

import pytest
import json
import os
import sys
import tempfile
import time
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import signal
import subprocess

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_trading.paper_exchange import PaperExchange


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Создаёт временную директорию для тестов"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def exchange():
    """Создаёт экземпляр PaperExchange для тестов"""
    return PaperExchange(initial_capital=1000.0)


@pytest.fixture
def exchange_with_data(exchange):
    """PaperExchange с несколькими сделками"""
    # Мокаем get_binance_price чтобы избежать реальных запросов
    with patch('paper_trading.paper_exchange.get_binance_price') as mock_price:
        mock_price.side_effect = lambda sym: {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0
        }.get(sym, 1000.0)

        # Открываем несколько позиций
        exchange.create_market_order('BTCUSDT', 'BUY', 0.01)
        exchange.create_market_order('ETHUSDT', 'BUY', 0.1)

        # Закрываем одну
        positions = list(exchange.positions.keys())
        if positions:
            pos = exchange.positions[positions[0]]
            exchange.create_market_order(pos['symbol'], 'SELL', pos['amount'])

    return exchange


# ============================================================================
# UNIT ТЕСТЫ: Атомарное сохранение состояния
# ============================================================================

class TestAtomicStateSaving:
    """Тесты атомарного сохранения состояния"""

    def test_save_creates_backup(self, exchange_with_data, temp_dir):
        """Проверка создания backup файла"""
        filepath = os.path.join(temp_dir, 'state.json')

        # Сохраняем первый раз
        exchange_with_data.save_state(filepath)
        assert os.path.exists(filepath), "Основной файл не создан"

        # Сохраняем второй раз - должен создаться backup
        time.sleep(0.1)  # Небольшая задержка для уникальности
        exchange_with_data.balance = 999.0
        exchange_with_data.save_state(filepath)

        backup_path = filepath + '.backup'
        assert os.path.exists(backup_path), "Backup файл не создан"

        # Проверяем что backup содержит старые данные
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        assert backup_data['balance'] != 999.0, "Backup содержит новые данные"

    def test_save_is_atomic(self, exchange_with_data, temp_dir):
        """Проверка атомарности сохранения через временный файл"""
        filepath = os.path.join(temp_dir, 'state.json')
        temp_filepath = filepath + '.tmp'

        # Патчим os.replace чтобы проверить что он вызывается
        with patch('os.replace') as mock_replace:
            exchange_with_data.save_state(filepath)

            # Проверяем что os.replace был вызван с правильными аргументами
            mock_replace.assert_called_once()
            args = mock_replace.call_args[0]
            assert args[0] == temp_filepath, "Первый аргумент должен быть temp файл"
            assert args[1] == filepath, "Второй аргумент должен быть основной файл"

    def test_save_and_load_consistency(self, exchange_with_data, temp_dir):
        """Проверка консистентности: сохранили → загрузили → данные идентичны"""
        filepath = os.path.join(temp_dir, 'state.json')

        # Сохраняем
        original_balance = exchange_with_data.balance
        original_positions = len(exchange_with_data.positions)
        original_trades = len(exchange_with_data.trades_history)

        exchange_with_data.save_state(filepath)

        # Загружаем в новый экземпляр
        loaded = PaperExchange.load_state(filepath)

        # Проверяем что всё совпадает
        assert loaded.balance == original_balance, "Баланс не совпадает"
        assert len(loaded.positions) == original_positions, "Количество позиций не совпадает"
        assert len(loaded.trades_history) == original_trades, "Количество сделок не совпадает"
        assert loaded.initial_capital == exchange_with_data.initial_capital, "Начальный капитал не совпадает"

    def test_save_handles_directory_creation(self, exchange, temp_dir):
        """Проверка автоматического создания директории"""
        nested_path = os.path.join(temp_dir, 'nested', 'deep', 'state.json')

        # Директории не существует
        assert not os.path.exists(os.path.dirname(nested_path))

        # Сохраняем - директория должна создаться автоматически
        exchange.save_state(nested_path)

        assert os.path.exists(nested_path), "Файл не создан в вложенной директории"
        assert os.path.exists(os.path.dirname(nested_path)), "Директория не создана"


# ============================================================================
# UNIT ТЕСТЫ: Восстановление из backup
# ============================================================================

class TestBackupRecovery:
    """Тесты восстановления из backup при повреждении"""

    def test_load_from_corrupted_main_uses_backup(self, exchange_with_data, temp_dir):
        """Проверка загрузки из backup при повреждённом основном файле"""
        filepath = os.path.join(temp_dir, 'state.json')
        backup_path = filepath + '.backup'

        # Сохраняем нормальное состояние
        exchange_with_data.save_state(filepath)

        # Копируем основной файл в backup вручную
        shutil.copy2(filepath, backup_path)

        # Повреждаем основной файл
        with open(filepath, 'w') as f:
            f.write('{"balance": 1000, "invalid json without closing')

        # Загружаем - должен использовать backup
        loaded = PaperExchange.load_state(filepath)

        assert loaded is not None, "Не удалось загрузить из backup"
        assert loaded.balance == exchange_with_data.balance, "Данные из backup не совпадают"

    def test_load_from_empty_main_uses_backup(self, exchange_with_data, temp_dir):
        """Проверка загрузки из backup при пустом основном файле"""
        filepath = os.path.join(temp_dir, 'state.json')
        backup_path = filepath + '.backup'

        # Сохраняем нормальное состояние
        exchange_with_data.save_state(filepath)

        # Копируем в backup
        shutil.copy2(filepath, backup_path)

        # Делаем основной файл пустым
        with open(filepath, 'w') as f:
            f.write('')

        # Загружаем - должен использовать backup
        loaded = PaperExchange.load_state(filepath)

        assert loaded is not None, "Не удалось загрузить из backup"
        assert loaded.balance == exchange_with_data.balance

    def test_load_fails_if_both_corrupted(self, temp_dir):
        """Проверка что загрузка падает если оба файла повреждены"""
        filepath = os.path.join(temp_dir, 'state.json')
        backup_path = filepath + '.backup'

        # Создаём повреждённые файлы
        with open(filepath, 'w') as f:
            f.write('invalid json')
        with open(backup_path, 'w') as f:
            f.write('also invalid')

        # Загрузка должна упасть
        with pytest.raises(Exception):
            PaperExchange.load_state(filepath)

    def test_backup_restores_main_file(self, exchange_with_data, temp_dir):
        """Проверка что при загрузке из backup основной файл восстанавливается"""
        filepath = os.path.join(temp_dir, 'state.json')
        backup_path = filepath + '.backup'

        # Сохраняем нормальное состояние
        exchange_with_data.save_state(filepath)
        shutil.copy2(filepath, backup_path)

        # Повреждаем основной файл
        with open(filepath, 'w') as f:
            f.write('corrupted')

        # Загружаем из backup
        loaded = PaperExchange.load_state(filepath)

        # Проверяем что основной файл восстановлен
        with open(filepath, 'r') as f:
            main_data = json.load(f)

        assert main_data['balance'] == exchange_with_data.balance, "Основной файл не восстановлен"


# ============================================================================
# UNIT ТЕСТЫ: Параллельная загрузка OHLCV
# ============================================================================

class TestOHLCVBatchLoading:
    """Тесты параллельной загрузки OHLCV"""

    def test_get_ohlcv_batch_exists(self, exchange):
        """Проверка что метод get_ohlcv_batch существует"""
        assert hasattr(exchange, 'get_ohlcv_batch'), "Метод get_ohlcv_batch не существует"
        assert callable(exchange.get_ohlcv_batch), "get_ohlcv_batch не является callable"

    def test_get_ohlcv_batch_returns_dict(self, exchange):
        """Проверка что метод возвращает правильную структуру"""
        # Мокаем get_ohlcv чтобы избежать реальных запросов
        with patch.object(exchange, 'get_ohlcv') as mock_get_ohlcv:
            import pandas as pd

            # Создаём мок DataFrame
            mock_df = pd.DataFrame({
                'open': [100, 101],
                'high': [102, 103],
                'low': [99, 100],
                'close': [101, 102],
                'volume': [1000, 1100]
            })
            mock_df.index = pd.DatetimeIndex(['2025-01-01', '2025-01-02'])
            mock_df.index.name = 'timestamp'
            mock_df['timestamp'] = mock_df.index

            mock_get_ohlcv.return_value = mock_df

            # Вызываем batch метод
            requests = [
                ('BTCUSDT', '5m', 200),
                ('ETHUSDT', '5m', 200),
            ]
            result = exchange.get_ohlcv_batch(requests, max_workers=2)

            # Проверяем структуру результата
            assert isinstance(result, dict), "Результат должен быть dict"
            assert 'BTCUSDT' in result, "Результат должен содержать BTCUSDT"
            assert 'ETHUSDT' in result, "Результат должен содержать ETHUSDT"
            assert '5m' in result['BTCUSDT'], "Результат должен содержать timeframe"

    def test_get_ohlcv_batch_parallel_execution(self, exchange):
        """Проверка что запросы выполняются параллельно"""
        call_times = []

        def mock_get_ohlcv(symbol, timeframe, limit):
            import pandas as pd
            import time

            call_times.append(time.time())
            time.sleep(0.1)  # Симулируем задержку сети

            mock_df = pd.DataFrame({
                'open': [100], 'high': [102], 'low': [99],
                'close': [101], 'volume': [1000]
            })
            mock_df.index = pd.DatetimeIndex(['2025-01-01'])
            mock_df.index.name = 'timestamp'
            mock_df['timestamp'] = mock_df.index
            return mock_df

        with patch.object(exchange, 'get_ohlcv', side_effect=mock_get_ohlcv):
            # 5 запросов, если последовательно = 0.5s, если параллельно < 0.3s
            requests = [
                ('SYM1', '5m', 200),
                ('SYM2', '5m', 200),
                ('SYM3', '5m', 200),
                ('SYM4', '5m', 200),
                ('SYM5', '5m', 200),
            ]

            start = time.time()
            result = exchange.get_ohlcv_batch(requests, max_workers=5)
            duration = time.time() - start

            # Параллельное выполнение должно занять < 0.3s (не 0.5s)
            assert duration < 0.3, f"Выполнение заняло {duration}s, похоже на последовательное"

            # Проверяем что вызовы происходили примерно одновременно
            time_spread = max(call_times) - min(call_times)
            assert time_spread < 0.15, "Вызовы не были параллельными"

    def test_get_ohlcv_batch_handles_failures(self, exchange):
        """Проверка обработки ошибок при batch загрузке"""
        def mock_get_ohlcv(symbol, timeframe, limit):
            import pandas as pd

            if symbol == 'FAIL':
                raise Exception("Network error")

            mock_df = pd.DataFrame({
                'open': [100], 'high': [102], 'low': [99],
                'close': [101], 'volume': [1000]
            })
            mock_df.index = pd.DatetimeIndex(['2025-01-01'])
            mock_df.index.name = 'timestamp'
            mock_df['timestamp'] = mock_df.index
            return mock_df

        with patch.object(exchange, 'get_ohlcv', side_effect=mock_get_ohlcv):
            requests = [
                ('BTCUSDT', '5m', 200),
                ('FAIL', '5m', 200),
                ('ETHUSDT', '5m', 200),
            ]

            result = exchange.get_ohlcv_batch(requests, max_workers=3)

            # Успешные запросы должны быть в результате
            assert 'BTCUSDT' in result, "BTCUSDT должен быть в результате"
            assert 'ETHUSDT' in result, "ETHUSDT должен быть в результате"

            # Неудачный запрос не должен быть в результате
            assert 'FAIL' not in result, "FAIL не должен быть в результате"


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

class TestIntegration:
    """Интеграционные тесты полного цикла"""

    def test_full_trading_cycle_with_state_persistence(self, temp_dir):
        """Полный цикл: создание → торговля → сохранение → загрузка → продолжение"""
        filepath = os.path.join(temp_dir, 'state.json')

        # Мокаем get_binance_price глобально
        with patch('paper_trading.paper_exchange.get_binance_price') as mock_price:
            mock_price.side_effect = lambda sym: {'BTCUSDT': 50000.0, 'ETHUSDT': 3000.0}.get(sym, 1000.0)

            # Создаём exchange и делаем несколько сделок
            exchange1 = PaperExchange(initial_capital=1000.0)

            # Открываем позиции
            order1 = exchange1.create_market_order('BTCUSDT', 'BUY', 0.01)
            order2 = exchange1.create_market_order('ETHUSDT', 'BUY', 0.1)

            balance_before_save = exchange1.balance
            positions_count = len(exchange1.positions)

            # Сохраняем состояние
            exchange1.save_state(filepath)

            # Загружаем в новый экземпляр
            exchange2 = PaperExchange.load_state(filepath)

            # Проверяем что всё восстановилось
            assert exchange2.balance == balance_before_save, "Баланс не восстановился"
            assert len(exchange2.positions) == positions_count, "Позиции не восстановились"

            # Продолжаем торговать только если есть позиции
            if len(exchange2.positions) > 0:
                # Меняем цену для закрытия
                mock_price.side_effect = lambda sym: {'BTCUSDT': 51000.0, 'ETHUSDT': 3100.0}.get(sym, 1100.0)

                # Продолжаем торговать на восстановленном exchange
                pos_id = list(exchange2.positions.keys())[0]
                pos = exchange2.positions[pos_id]
                exchange2.create_market_order(pos['symbol'], 'SELL', pos['amount'])

                # Проверяем что торговля работает
                assert len(exchange2.closed_positions) > 0, "Позиция не закрылась"
            else:
                # Если позиций нет, просто проверяем что есть история сделок
                assert len(exchange2.trades_history) > 0, "История сделок пуста"

    def test_state_survives_corruption_during_save(self, temp_dir):
        """Симуляция повреждения во время сохранения"""
        filepath = os.path.join(temp_dir, 'state.json')

        # Мокаем get_binance_price
        with patch('paper_trading.paper_exchange.get_binance_price', return_value=50000.0):
            # Создаём exchange с данными
            exchange1 = PaperExchange(initial_capital=1000.0)
            exchange1.create_market_order('BTCUSDT', 'BUY', 0.01)

            # Первое сохранение
            exchange1.save_state(filepath)

            # Изменяем баланс
            original_balance = exchange1.balance
            exchange1.balance = 999.0

            # Второе сохранение (создаст backup)
            exchange1.save_state(filepath)

            # Симулируем повреждение основного файла
            with open(filepath, 'w') as f:
                f.write('{"corrupted": true')

            # Загружаем - должен использовать backup
            exchange2 = PaperExchange.load_state(filepath)

            # Проверяем что данные из backup (старый баланс или новый)
            assert exchange2 is not None, "Не удалось восстановить из backup"
            assert exchange2.balance > 0, "Баланс повреждён"

    def test_concurrent_operations_thread_safety(self, exchange):
        """Проверка thread-safety при параллельных операциях"""
        errors = []

        # Мокаем get_binance_price глобально
        with patch('paper_trading.paper_exchange.get_binance_price') as mock_price:
            mock_price.return_value = 50000.0

            def create_order(symbol, side, amount):
                try:
                    exchange.create_market_order(symbol, side, amount)
                except Exception as e:
                    errors.append(e)

            # Создаём 10 параллельных заказов
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for i in range(10):
                    symbol = f'SYM{i}USDT'
                    future = executor.submit(create_order, symbol, 'BUY', 0.01)
                    futures.append(future)

                # Ждём завершения всех
                for future in futures:
                    future.result()

            # Не должно быть ошибок
            assert len(errors) == 0, f"Произошли ошибки при параллельных операциях: {errors}"

            # Проверяем что все заказы прошли
            assert len(exchange.trades_history) == 10, "Не все заказы выполнены"


# ============================================================================
# НАГРУЗОЧНЫЕ ТЕСТЫ
# ============================================================================

class TestPerformance:
    """Нагрузочные тесты производительности"""

    def test_batch_loading_performance(self, exchange):
        """Проверка что batch загрузка быстрее последовательной"""
        import pandas as pd

        def mock_get_ohlcv(symbol, timeframe, limit):
            time.sleep(0.01)  # Симулируем задержку сети 10ms

            mock_df = pd.DataFrame({
                'open': [100], 'high': [102], 'low': [99],
                'close': [101], 'volume': [1000]
            })
            mock_df.index = pd.DatetimeIndex(['2025-01-01'])
            mock_df.index.name = 'timestamp'
            mock_df['timestamp'] = mock_df.index
            return mock_df

        # Создаём 50 запросов
        requests = [
            (f'SYM{i}USDT', '5m', 200)
            for i in range(50)
        ]

        with patch.object(exchange, 'get_ohlcv', side_effect=mock_get_ohlcv):
            # Параллельная загрузка
            start = time.time()
            result = exchange.get_ohlcv_batch(requests, max_workers=10)
            parallel_duration = time.time() - start

            print(f"\n📊 Batch загрузка 50 монет: {parallel_duration:.2f}s")

            # Последовательная загрузка заняла бы ~0.5s (50 * 0.01)
            # Параллельная должна занять ~0.05s (50 / 10 workers * 0.01)
            assert parallel_duration < 0.2, f"Batch загрузка слишком медленная: {parallel_duration}s"

            # Проверяем что все данные загружены
            assert len(result) == 50, f"Загружено только {len(result)} из 50"

    def test_large_state_save_load(self, temp_dir):
        """Проверка сохранения/загрузки большого состояния"""
        filepath = os.path.join(temp_dir, 'large_state.json')

        # Создаём exchange с большим количеством сделок
        exchange = PaperExchange(initial_capital=10000.0)

        # Создаём 1000 сделок
        with patch.object(exchange, 'get_current_price', return_value=50000.0):
            for i in range(1000):
                exchange.trades_history.append({
                    'id': f'TRADE_{i}',
                    'symbol': f'SYM{i % 10}USDT',
                    'side': 'BUY' if i % 2 == 0 else 'SELL',
                    'amount': 0.01,
                    'price': 50000.0 + i,
                    'timestamp': time.time(),
                })

        # Измеряем время сохранения
        start = time.time()
        exchange.save_state(filepath)
        save_duration = time.time() - start

        print(f"\n💾 Сохранение 1000 сделок: {save_duration:.3f}s")

        # Измеряем время загрузки
        start = time.time()
        loaded = PaperExchange.load_state(filepath)
        load_duration = time.time() - start

        print(f"📥 Загрузка 1000 сделок: {load_duration:.3f}s")

        # Проверяем корректность
        assert len(loaded.trades_history) == 1000, "Не все сделки загружены"
        assert loaded.balance == exchange.balance, "Баланс не совпадает"

        # Операции должны быть быстрыми (< 1 секунды каждая)
        assert save_duration < 1.0, f"Сохранение слишком медленное: {save_duration}s"
        assert load_duration < 1.0, f"Загрузка слишком медленная: {load_duration}s"


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================

if __name__ == "__main__":
    # Запускаем pytest с подробным выводом
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes',
        '-s'  # Показывать print() в консоли
    ])
