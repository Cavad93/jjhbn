#!/usr/bin/env python3
"""
Тесты для Position и PositionManager

Полное тестирование функциональности управления позициями.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Тест 1: Импорт компонентов"""
    print("Тест 1: Импорт компонентов...", end=" ")

    try:
        from portfolio import Position, PositionManager
        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_creation():
    """Тест 2: Создание Position"""
    print("Тест 2: Создание Position...", end=" ")

    try:
        from portfolio import Position

        # Создаем LONG позицию
        position = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=45000.0,
            amount=0.001,
            position_value=45.0,
            tp_price=46350.0,
            sl_price=44100.0,
            atr_value=1000.0,
            entry_order_id='ORDER_123',
            tp_order_id='TP_123',
            sl_order_id='SL_123'
        )

        # Проверки
        assert position.symbol == 'BTCUSDT'
        assert position.direction == 'LONG'
        assert position.entry_price == 45000.0
        assert position.status == 'OPEN'
        assert position.pnl == 0.0

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_with_snapshot():
    """Тест 3: Position с entry_snapshot"""
    print("Тест 3: Position с entry_snapshot...", end=" ")

    try:
        from portfolio import Position

        # Создаем snapshot
        snapshot = {
            'features_68d': np.array([0.1, 0.2, 0.3] + [0.0] * 65),
            'predictions': {
                'p_xgb': 0.65,
                'p_rf': 0.62,
                'p_arf': 0.64,
                'p_nn': 0.63,
                'p_base': 0.60
            },
            'meta_context': {
                'feature1': 1.0,
                'feature2': 2.0
            },
            'p_meta': 0.68,
            'ev_long': 0.0700,
            'ev_short': 0.0200,
            'selected_ev': 0.0700,
            'timestamp': datetime.now().timestamp()
        }

        position = Position(
            symbol='ETHUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=2500.0,
            amount=0.1,
            position_value=250.0,
            tp_price=2625.0,
            sl_price=2425.0,
            atr_value=50.0,
            entry_snapshot=snapshot
        )

        # Проверки snapshot
        assert 'features_68d' in position.entry_snapshot
        assert 'predictions' in position.entry_snapshot
        assert position.entry_snapshot['p_meta'] == 0.68
        assert position.entry_snapshot['predictions']['p_xgb'] == 0.65

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_pnl_calculation():
    """Тест 4: Расчет PnL для LONG и SHORT"""
    print("Тест 4: Расчет PnL для LONG и SHORT...", end=" ")

    try:
        from portfolio import Position

        # LONG позиция
        long_pos = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=45000.0,
            amount=0.001,
            position_value=45.0,
            tp_price=46000.0,
            sl_price=44000.0,
            atr_value=1000.0
        )

        # Тест calculate_pnl для LONG
        pnl, pnl_pct = long_pos.calculate_pnl(46000.0)
        assert abs(pnl - 1.0) < 0.01, f"Expected PnL ~1.0, got {pnl}"
        assert abs(pnl_pct - 2.22) < 0.1, f"Expected PnL% ~2.22, got {pnl_pct}"

        # SHORT позиция
        short_pos = Position(
            symbol='ETHUSDT',
            direction='SHORT',
            entry_time=datetime.now(),
            entry_price=2500.0,
            amount=0.1,
            position_value=250.0,
            tp_price=2400.0,
            sl_price=2600.0,
            atr_value=50.0
        )

        # Тест calculate_pnl для SHORT
        pnl, pnl_pct = short_pos.calculate_pnl(2400.0)
        assert abs(pnl - 10.0) < 0.01, f"Expected PnL ~10.0, got {pnl}"
        assert abs(pnl_pct - 4.0) < 0.1, f"Expected PnL% ~4.0, got {pnl_pct}"

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_tp_sl_check():
    """Тест 5: Проверка достижения TP/SL"""
    print("Тест 5: Проверка достижения TP/SL...", end=" ")

    try:
        from portfolio import Position

        # LONG позиция
        long_pos = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=45000.0,
            amount=0.001,
            position_value=45.0,
            tp_price=46000.0,
            sl_price=44000.0,
            atr_value=1000.0
        )

        # Проверки для LONG
        assert long_pos.is_tp_hit(46500.0) == True
        assert long_pos.is_tp_hit(45500.0) == False
        assert long_pos.is_sl_hit(43500.0) == True
        assert long_pos.is_sl_hit(45500.0) == False

        # SHORT позиция
        short_pos = Position(
            symbol='ETHUSDT',
            direction='SHORT',
            entry_time=datetime.now(),
            entry_price=2500.0,
            amount=0.1,
            position_value=250.0,
            tp_price=2400.0,
            sl_price=2600.0,
            atr_value=50.0
        )

        # Проверки для SHORT
        assert short_pos.is_tp_hit(2300.0) == True
        assert short_pos.is_tp_hit(2450.0) == False
        assert short_pos.is_sl_hit(2700.0) == True
        assert short_pos.is_sl_hit(2550.0) == False

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_serialization():
    """Тест 6: Сериализация/десериализация Position"""
    print("Тест 6: Сериализация/десериализация Position...", end=" ")

    try:
        from portfolio import Position

        # Создаем позицию с snapshot
        original = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=45000.0,
            amount=0.001,
            position_value=45.0,
            tp_price=46000.0,
            sl_price=44000.0,
            atr_value=1000.0,
            entry_snapshot={
                'features_68d': np.array([0.1, 0.2, 0.3]),
                'p_meta': 0.65
            }
        )

        # Сериализация
        data = original.to_dict()
        assert 'symbol' in data
        assert 'entry_snapshot' in data
        assert isinstance(data['entry_snapshot']['features_68d'], list)

        # Десериализация
        restored = Position.from_dict(data)
        assert restored.symbol == original.symbol
        assert restored.entry_price == original.entry_price
        assert restored.direction == original.direction
        assert isinstance(restored.entry_snapshot['features_68d'], np.ndarray)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_manager_init():
    """Тест 7: Инициализация PositionManager"""
    print("Тест 7: Инициализация PositionManager...", end=" ")

    try:
        from portfolio import PositionManager

        # Создаем временную директорию
        temp_dir = tempfile.mkdtemp()

        manager = PositionManager(max_positions=10, data_dir=temp_dir)

        assert manager.max_positions == 10
        assert len(manager.positions) == 0
        assert len(manager.closed_positions) == 0
        assert manager.get_open_count() == 0
        assert manager.get_available_slots() == 10

        # Очистка
        shutil.rmtree(temp_dir)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_manager_add_position():
    """Тест 8: Добавление позиций"""
    print("Тест 8: Добавление позиций...", end=" ")

    try:
        from portfolio import Position, PositionManager

        temp_dir = tempfile.mkdtemp()
        manager = PositionManager(max_positions=3, data_dir=temp_dir)

        # Добавляем 3 позиции
        for i in range(3):
            position = Position(
                symbol=f'COIN{i}USDT',
                direction='LONG',
                entry_time=datetime.now(),
                entry_price=100.0 + i,
                amount=0.1,
                position_value=10.0,
                tp_price=110.0 + i,
                sl_price=90.0 + i,
                atr_value=5.0
            )
            manager.add_position(position)

        # Проверки
        assert manager.get_open_count() == 3
        assert manager.get_available_slots() == 0
        assert manager.has_position('COIN0USDT') == True
        assert manager.has_position('COIN3USDT') == False

        # Попытка добавить 4-ю позицию (должна упасть)
        try:
            position4 = Position(
                symbol='COIN3USDT',
                direction='LONG',
                entry_time=datetime.now(),
                entry_price=100.0,
                amount=0.1,
                position_value=10.0,
                tp_price=110.0,
                sl_price=90.0,
                atr_value=5.0
            )
            manager.add_position(position4)
            print(f"✗ ПРОВАЛЕН: Должна была упасть ошибка при добавлении 4-й позиции")
            return False
        except ValueError:
            pass  # Ожидаемая ошибка

        # Очистка
        shutil.rmtree(temp_dir)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_manager_close_position():
    """Тест 9: Закрытие позиций"""
    print("Тест 9: Закрытие позиций...", end=" ")

    try:
        from portfolio import Position, PositionManager

        temp_dir = tempfile.mkdtemp()
        manager = PositionManager(max_positions=10, data_dir=temp_dir)

        # Добавляем позицию
        position = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=45000.0,
            amount=0.001,
            position_value=45.0,
            tp_price=46000.0,
            sl_price=44000.0,
            atr_value=1000.0
        )
        manager.add_position(position)

        assert manager.get_open_count() == 1

        # Закрываем позицию по TP
        closed = manager.close_position('BTCUSDT', exit_price=46000.0, exit_reason='TP')

        # Проверки
        assert closed.status == 'CLOSED'
        assert closed.exit_reason == 'TP'
        assert closed.pnl > 0  # Должна быть прибыль
        assert manager.get_open_count() == 0
        assert len(manager.closed_positions) == 1

        # Очистка
        shutil.rmtree(temp_dir)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_manager_pnl_summary():
    """Тест 10: Статистика PnL"""
    print("Тест 10: Статистика PnL...", end=" ")

    try:
        from portfolio import Position, PositionManager

        temp_dir = tempfile.mkdtemp()
        manager = PositionManager(max_positions=10, data_dir=temp_dir)

        # Открываем и закрываем несколько позиций
        # 3 wins
        for i in range(3):
            pos = Position(
                symbol=f'WIN{i}USDT',
                direction='LONG',
                entry_time=datetime.now(),
                entry_price=100.0,
                amount=0.1,
                position_value=10.0,
                tp_price=110.0,
                sl_price=90.0,
                atr_value=5.0
            )
            manager.add_position(pos)
            manager.close_position(f'WIN{i}USDT', exit_price=110.0, exit_reason='TP')

        # 2 losses
        for i in range(2):
            pos = Position(
                symbol=f'LOSS{i}USDT',
                direction='LONG',
                entry_time=datetime.now(),
                entry_price=100.0,
                amount=0.1,
                position_value=10.0,
                tp_price=110.0,
                sl_price=90.0,
                atr_value=5.0
            )
            manager.add_position(pos)
            manager.close_position(f'LOSS{i}USDT', exit_price=90.0, exit_reason='SL')

        # Получаем статистику
        summary = manager.get_pnl_summary()

        # Проверки
        assert summary['total_trades'] == 5
        assert summary['wins'] == 3
        assert summary['losses'] == 2
        assert summary['win_rate'] == 60.0
        assert summary['total_pnl'] > 0  # Общая прибыль должна быть положительной

        # Очистка
        shutil.rmtree(temp_dir)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_manager_save_load():
    """Тест 11: Сохранение и загрузка из JSON"""
    print("Тест 11: Сохранение и загрузка из JSON...", end=" ")

    try:
        from portfolio import Position, PositionManager

        temp_dir = tempfile.mkdtemp()

        # Создаем менеджер и добавляем позиции
        manager1 = PositionManager(max_positions=10, data_dir=temp_dir)

        position1 = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=45000.0,
            amount=0.001,
            position_value=45.0,
            tp_price=46000.0,
            sl_price=44000.0,
            atr_value=1000.0,
            entry_snapshot={'p_meta': 0.65}
        )
        manager1.add_position(position1)

        position2 = Position(
            symbol='ETHUSDT',
            direction='SHORT',
            entry_time=datetime.now(),
            entry_price=2500.0,
            amount=0.1,
            position_value=250.0,
            tp_price=2400.0,
            sl_price=2600.0,
            atr_value=50.0
        )
        manager1.add_position(position2)

        # Закрываем одну позицию
        manager1.close_position('ETHUSDT', exit_price=2400.0, exit_reason='TP')

        # Сохраняем
        manager1.save_to_file()

        # Создаем новый менеджер и загружаем
        manager2 = PositionManager(max_positions=10, data_dir=temp_dir)
        success = manager2.load_from_file()

        # Проверки
        assert success == True
        assert manager2.get_open_count() == 1
        assert len(manager2.closed_positions) == 1
        assert manager2.has_position('BTCUSDT') == True
        assert manager2.has_position('ETHUSDT') == False

        restored_pos = manager2.get_position('BTCUSDT')
        assert restored_pos.entry_price == 45000.0
        assert restored_pos.entry_snapshot['p_meta'] == 0.65

        # Очистка
        shutil.rmtree(temp_dir)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_manager_duplicate_symbol():
    """Тест 12: Проверка дублирования символа"""
    print("Тест 12: Проверка дублирования символа...", end=" ")

    try:
        from portfolio import Position, PositionManager

        temp_dir = tempfile.mkdtemp()
        manager = PositionManager(max_positions=10, data_dir=temp_dir)

        # Добавляем позицию
        position1 = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=45000.0,
            amount=0.001,
            position_value=45.0,
            tp_price=46000.0,
            sl_price=44000.0,
            atr_value=1000.0
        )
        manager.add_position(position1)

        # Попытка добавить вторую позицию на тот же символ
        try:
            position2 = Position(
                symbol='BTCUSDT',  # Тот же символ!
                direction='SHORT',
                entry_time=datetime.now(),
                entry_price=45000.0,
                amount=0.001,
                position_value=45.0,
                tp_price=44000.0,
                sl_price=46000.0,
                atr_value=1000.0
            )
            manager.add_position(position2)

            print(f"✗ ПРОВАЛЕН: Должна была упасть ошибка при дублировании символа")
            return False
        except ValueError:
            pass  # Ожидаемая ошибка

        # Очистка
        shutil.rmtree(temp_dir)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def test_position_manager_total_exposure():
    """Тест 13: Расчет общей экспозиции"""
    print("Тест 13: Расчет общей экспозиции...", end=" ")

    try:
        from portfolio import Position, PositionManager

        temp_dir = tempfile.mkdtemp()
        manager = PositionManager(max_positions=10, data_dir=temp_dir)

        # Добавляем 3 позиции с разной стоимостью
        for i in range(3):
            position = Position(
                symbol=f'COIN{i}USDT',
                direction='LONG',
                entry_time=datetime.now(),
                entry_price=100.0,
                amount=0.1,
                position_value=100.0 * (i + 1),  # 100, 200, 300
                tp_price=110.0,
                sl_price=90.0,
                atr_value=5.0
            )
            manager.add_position(position)

        # Проверка общей экспозиции
        total_exposure = manager.get_total_exposure()
        assert abs(total_exposure - 600.0) < 0.01, f"Expected 600, got {total_exposure}"

        # Очистка
        shutil.rmtree(temp_dir)

        print("✓ ПРОЙДЕН")
        return True
    except Exception as e:
        print(f"✗ ПРОВАЛЕН: {e}")
        return False


def run_all_tests():
    """Запускает все тесты"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ POSITION И POSITIONMANAGER")
    print("=" * 80)
    print()

    tests = [
        test_imports,
        test_position_creation,
        test_position_with_snapshot,
        test_position_pnl_calculation,
        test_position_tp_sl_check,
        test_position_serialization,
        test_position_manager_init,
        test_position_manager_add_position,
        test_position_manager_close_position,
        test_position_manager_pnl_summary,
        test_position_manager_save_load,
        test_position_manager_duplicate_symbol,
        test_position_manager_total_exposure,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

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
