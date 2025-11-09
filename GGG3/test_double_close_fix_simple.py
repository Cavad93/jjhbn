#!/usr/bin/env python3
"""
Упрощенный тест защиты от двойного закрытия позиций

Работает без обращений к Binance API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_trading.paper_exchange import PaperExchange
import time
import uuid


def create_mock_position(exchange, symbol, side, amount, entry_price):
    """Создает мок-позицию без обращения к API"""
    pos_id = f"TEST_{uuid.uuid4().hex[:12].upper()}"
    order_id = f"ORDER_{uuid.uuid4().hex[:12].upper()}"

    position = {
        'id': pos_id,
        'symbol': symbol,
        'side': side,
        'amount': amount,
        'entry_price': entry_price,
        'entry_order_id': order_id,
        'status': 'open',
        'opened_at': time.time(),
        'opened_datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tp_price': None,
        'sl_price': None,
        'tp_order_id': None,
        'sl_order_id': None,
        'metadata': {}
    }

    # Добавляем позицию
    exchange.positions[pos_id] = position

    # Моделируем entry order в истории торгов
    commission = amount * entry_price * exchange.commission
    exchange.trades_history.append({
        'id': order_id,
        'symbol': symbol,
        'side': 'BUY' if side == 'LONG' else 'SELL',
        'amount': amount,
        'price': entry_price,
        'commission': commission,
        'timestamp': time.time()
    })

    return position


def mock_close_position(exchange, position, exit_price):
    """
    Моделирует создание closing market order для close_position()

    Мокаем get_binance_price, чтобы вернуть нужную exit_price
    """
    # Патчим get_binance_price для этого символа
    original_get_price = None
    try:
        from paper_trading import paper_exchange
        original_get_price = paper_exchange.get_binance_price

        def mock_get_price(symbol):
            if symbol == position['symbol']:
                return exit_price
            return original_get_price(symbol)

        paper_exchange.get_binance_price = mock_get_price

        # Теперь вызываем close_position
        result = exchange.close_position(position['id'], reason='manual')

        return result
    finally:
        # Восстанавливаем оригинальную функцию
        if original_get_price:
            paper_exchange.get_binance_price = original_get_price


def test_1_normal_close():
    """Тест 1: Нормальное закрытие позиции"""
    print("\n" + "="*80)
    print("ТЕСТ 1: Нормальное закрытие позиции")
    print("="*80)

    exchange = PaperExchange(initial_capital=1000.0)

    # Создаем мок-позицию
    print("\n[1.1] Создаем мок-позицию: LONG 0.01 BTC @ $50,000")
    pos = create_mock_position(exchange, 'BTCUSDT', 'LONG', 0.01, 50000.0)

    balance_before = exchange.balance
    print(f"   Balance before: ${balance_before:.2f}")

    # Закрываем с прибылью (exit @ $51,000)
    print("\n[1.2] Закрываем позицию @ $51,000 (прибыль)")
    closed = mock_close_position(exchange, pos, 51000.0)

    balance_after = exchange.balance
    pnl = closed['pnl_after_commission']
    balance_change = balance_after - balance_before

    print(f"   Balance after: ${balance_after:.2f}")
    print(f"   PnL: ${pnl:.2f}")
    print(f"   Balance change: ${balance_change:.2f}")

    # Проверка: баланс изменился на PnL
    assert abs(balance_change - pnl) < 0.01, f"Balance change should equal PnL"
    print("✅ Баланс изменился корректно")

    # Проверка: позиция удалена
    assert pos['id'] not in exchange.positions, "Position should be removed"
    print("✅ Позиция удалена из открытых")

    # Валидация
    validation = exchange.validate_balance()
    assert validation['is_valid'], "Balance validation should pass"
    print(f"✅ Валидация баланса пройдена\n")

    return True


def test_2_double_close_protection():
    """Тест 2: Защита от двойного закрытия"""
    print("\n" + "="*80)
    print("ТЕСТ 2: Защита от двойного закрытия")
    print("="*80)

    exchange = PaperExchange(initial_capital=1000.0)

    # Создаем мок-позицию
    print("\n[2.1] Создаем мок-позицию")
    pos = create_mock_position(exchange, 'BTCUSDT', 'LONG', 0.01, 50000.0)
    pos_id = pos['id']

    # Первое закрытие
    print("\n[2.2] Первое закрытие позиции")
    balance_before_first = exchange.balance
    closed = mock_close_position(exchange, pos, 51000.0)
    balance_after_first = exchange.balance

    pnl = closed['pnl_after_commission']
    first_change = balance_after_first - balance_before_first

    print(f"   PnL: ${pnl:.2f}")
    print(f"   Balance: ${balance_before_first:.2f} → ${balance_after_first:.2f}")
    print(f"   Change: ${first_change:.2f}")
    print("✅ Первое закрытие выполнено")

    # Попытка закрыть второй раз
    print("\n[2.3] Попытка закрыть ВТОРОЙ раз")
    balance_before_second = exchange.balance

    try:
        # Позиция уже удалена, должна возникнуть ошибка
        exchange.close_position(pos_id, reason='manual')
        print("❌ ОШИБКА: Должна была возникнуть ошибка OrderNotFound")
        return False
    except Exception as e:
        print(f"✅ Ожидаемая ошибка: {type(e).__name__}")

    balance_after_second = exchange.balance
    second_change = balance_after_second - balance_before_second

    print(f"   Balance: ${balance_before_second:.2f} → ${balance_after_second:.2f}")
    print(f"   Change: ${second_change:.2f}")

    # КРИТИЧНО: Баланс не должен измениться
    assert abs(second_change) < 0.01, "Balance should not change on failed second close"
    print("✅ Баланс не изменился при второй попытке")

    # Валидация
    validation = exchange.validate_balance()
    assert validation['is_valid'], "Balance validation should pass"
    print(f"✅ Валидация баланса пройдена\n")

    return True


def test_3_is_closing_flag():
    """Тест 3: Проверка флага is_closing"""
    print("\n" + "="*80)
    print("ТЕСТ 3: Флаг is_closing защищает от одновременного закрытия")
    print("="*80)

    exchange = PaperExchange(initial_capital=1000.0)

    # Создаем мок-позицию
    print("\n[3.1] Создаем мок-позицию")
    pos = create_mock_position(exchange, 'BTCUSDT', 'LONG', 0.01, 50000.0)
    pos_id = pos['id']

    # Устанавливаем флаг is_closing вручную
    print("\n[3.2] Устанавливаем флаг is_closing=True вручную")
    exchange.positions[pos_id]['is_closing'] = True

    balance_before = exchange.balance

    # Пытаемся закрыть с установленным флагом
    print("[3.3] Пытаемся закрыть позицию")
    result = exchange.close_position(pos_id, reason='manual')

    balance_after = exchange.balance
    balance_change = balance_after - balance_before

    print(f"   Balance: ${balance_before:.2f} → ${balance_after:.2f}")
    print(f"   Change: ${balance_change:.2f}")

    # Проверка: баланс НЕ должен измениться
    assert abs(balance_change) < 0.01, "Balance should not change with is_closing=True"
    print("✅ Баланс не изменился - флаг сработал")

    # Проверка: позиция все еще открыта
    assert pos_id in exchange.positions, "Position should still be open"
    print("✅ Позиция все еще открыта")

    # Сбрасываем флаг и закрываем нормально
    print("\n[3.4] Сбрасываем флаг и закрываем нормально")
    exchange.positions[pos_id]['is_closing'] = False

    balance_before = exchange.balance
    closed = mock_close_position(exchange, pos, 51000.0)
    balance_after = exchange.balance

    pnl = closed['pnl_after_commission']
    balance_change = balance_after - balance_before

    print(f"   PnL: ${pnl:.2f}")
    print(f"   Balance change: ${balance_change:.2f}")

    assert abs(balance_change - pnl) < 0.01, "Balance should change by PnL"
    print("✅ Позиция закрыта корректно")

    # Валидация
    validation = exchange.validate_balance()
    assert validation['is_valid'], "Balance validation should pass"
    print(f"✅ Валидация баланса пройдена\n")

    return True


def test_4_multiple_positions():
    """Тест 4: Множественные позиции"""
    print("\n" + "="*80)
    print("ТЕСТ 4: Закрытие нескольких позиций")
    print("="*80)

    exchange = PaperExchange(initial_capital=10000.0)

    # Создаем 5 позиций
    print("\n[4.1] Создаем 5 позиций")
    positions = []
    for i in range(5):
        pos = create_mock_position(
            exchange,
            f'TEST{i}USDT',
            'LONG',
            1.0,
            100.0 + i * 10
        )
        positions.append(pos)
        print(f"   [{i+1}/5] Создана {pos['symbol']} @ ${pos['entry_price']:.2f}")

    print(f"✅ Создано {len(positions)} позиций")

    # Закрываем все
    print("\n[4.2] Закрываем все позиции")
    balance_before = exchange.balance
    total_pnl = 0

    for i, pos in enumerate(positions):
        # Закрываем с небольшой прибылью
        exit_price = pos['entry_price'] + 5.0
        closed = mock_close_position(exchange, pos, exit_price)

        pnl = closed['pnl_after_commission']
        total_pnl += pnl
        print(f"   [{i+1}/5] {closed['symbol']}: PnL ${pnl:.2f}")

    balance_after = exchange.balance
    balance_change = balance_after - balance_before

    print(f"\n   Total PnL: ${total_pnl:.2f}")
    print(f"   Balance change: ${balance_change:.2f}")

    assert abs(balance_change - total_pnl) < 0.10, "Total balance change should equal total PnL"
    print("✅ Общее изменение баланса корректно")

    # Валидация
    validation = exchange.validate_balance()
    assert validation['is_valid'], "Balance validation should pass"
    print(f"✅ Валидация баланса пройдена")
    print(f"   Итоговый баланс: ${validation['actual_balance']:.2f}\n")

    return True


def test_5_validation_accuracy():
    """Тест 5: Точность валидации баланса"""
    print("\n" + "="*80)
    print("ТЕСТ 5: Точность validate_balance()")
    print("="*80)

    exchange = PaperExchange(initial_capital=1000.0)

    # Создаем и закрываем несколько позиций
    print("\n[5.1] Создаем и закрываем позиции с разными результатами")

    test_cases = [
        ('BTC', 50000.0, 51000.0, "прибыль"),
        ('ETH', 3000.0, 2950.0, "убыток"),
        ('SOL', 100.0, 105.0, "прибыль"),
    ]

    for symbol_base, entry, exit, outcome in test_cases:
        symbol = f'{symbol_base}USDT'
        pos = create_mock_position(exchange, symbol, 'LONG', 0.01, entry)
        closed = mock_close_position(exchange, pos, exit)

        pnl = closed['pnl_after_commission']
        print(f"   {symbol}: Entry=${entry}, Exit=${exit} → PnL ${pnl:.2f} ({outcome})")

    # Валидация
    print("\n[5.2] Проверка валидации")
    validation = exchange.validate_balance()

    print(f"   Initial capital: ${validation['initial_capital']:.2f}")
    print(f"   Total PnL: ${validation['total_pnl_from_history']:.2f}")
    print(f"   Expected balance: ${validation['expected_balance']:.2f}")
    print(f"   Actual balance: ${validation['actual_balance']:.2f}")
    print(f"   Difference: ${validation['difference']:.2f}")

    assert validation['is_valid'], "Validation should pass"
    assert abs(validation['difference']) < 0.01, "Difference should be less than $0.01"
    print(f"✅ Валидация точная, разница < $0.01\n")

    return True


def main():
    """Запуск всех тестов"""
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ ЗАЩИТЫ ОТ ДВОЙНОГО ЗАКРЫТИЯ ПОЗИЦИЙ")
    print("="*80)

    tests = [
        ("Нормальное закрытие", test_1_normal_close),
        ("Защита от двойного закрытия", test_2_double_close_protection),
        ("Флаг is_closing", test_3_is_closing_flag),
        ("Множественные позиции", test_4_multiple_positions),
        ("Точность валидации", test_5_validation_accuracy),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except AssertionError as e:
            print(f"\n❌ ТЕСТ '{test_name}' ПРОВАЛЕН:")
            print(f"   {e}")
            results.append((test_name, False))
        except Exception as e:
            print(f"\n❌ ТЕСТ '{test_name}' ПРОВАЛЕН С ОШИБКОЙ:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Итоги
    print("\n" + "="*80)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*80 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}  {test_name}")

    print(f"\n{'='*80}")
    print(f"Пройдено: {passed}/{total} тестов ({passed/total*100:.1f}%)")
    print(f"{'='*80}\n")

    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("✅ Проблема двойного закрытия полностью устранена")
        print("✅ Баланс считается корректно")
        print("✅ Валидация работает точно")
        return 0
    else:
        print("⚠️  НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
