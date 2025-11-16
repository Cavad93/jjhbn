#!/usr/bin/env python3
"""
Тест: проверка сохранения и восстановления состояния PaperExchange
"""

import sys
import os
import json
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paper_trading.paper_exchange import PaperExchange


def test_state_persistence():
    """Тест: сохранение и загрузка состояния"""
    print("\n=== Test: State Persistence ===")

    # Путь к файлу состояния
    state_file = Path(__file__).parent / 'test_exchange_state.json'

    # Удаляем файл если существует
    if state_file.exists():
        state_file.unlink()

    # Шаг 1: Создаем новую биржу
    print("\n1. Creating new exchange...")
    exchange1 = PaperExchange(initial_capital=1000.0)

    assert exchange1.get_balance() == 1000.0, "Initial balance should be 1000"
    assert exchange1.get_equity() == 1000.0, "Initial equity should be 1000"
    assert len(exchange1.get_open_positions()) == 0, "Should have no positions"

    print(f"  ✓ Balance: {exchange1.get_balance():.2f} USDT")
    print(f"  ✓ Equity: {exchange1.get_equity():.2f} USDT")

    # Шаг 2: Симулируем некоторую активность
    print("\n2. Simulating trading activity...")

    # Открываем фейковую позицию (для теста)
    # Напрямую меняем баланс для упрощения
    exchange1.balance = 950.0  # Потратили 50 USDT на позицию

    print(f"  ✓ Modified balance to: {exchange1.get_balance():.2f} USDT")

    # Шаг 3: Сохраняем состояние
    print("\n3. Saving state...")
    exchange1.save_state(str(state_file))

    assert state_file.exists(), f"State file should exist at {state_file}"
    print(f"  ✓ State saved to {state_file}")

    # Шаг 4: Загружаем состояние в новый экземпляр
    print("\n4. Loading state into new instance...")
    exchange2 = PaperExchange.load_state(str(state_file))

    print(f"  ✓ Balance: {exchange2.get_balance():.2f} USDT")
    print(f"  ✓ Equity: {exchange2.get_equity():.2f} USDT")

    # Шаг 5: Проверяем что состояние восстановлено
    print("\n5. Verifying restored state...")

    assert exchange2.get_balance() == 950.0, \
        f"Balance should be 950.0, got {exchange2.get_balance()}"
    assert exchange2.initial_capital == 1000.0, \
        f"Initial capital should be 1000.0, got {exchange2.initial_capital}"

    print("  ✓ Balance matches: 950.0 USDT")
    print("  ✓ Initial capital matches: 1000.0 USDT")

    # Очистка
    if state_file.exists():
        state_file.unlink()
    if (state_file.parent / f"{state_file.name}.backup").exists():
        (state_file.parent / f"{state_file.name}.backup").unlink()

    print("\n✅ STATE PERSISTENCE TEST PASSED!")
    return True


def test_bot_initialization_with_saved_state():
    """Тест: инициализация бота с сохраненным состоянием"""
    print("\n=== Test: Bot Initialization with Saved State ===")

    # Создаем директорию для данных
    data_dir = Path(__file__).parent / 'test_bot_data'
    data_dir.mkdir(exist_ok=True)

    state_file = data_dir / 'paper_exchange_state.json'

    # Удаляем старое состояние
    if state_file.exists():
        state_file.unlink()

    # Шаг 1: Создаем и сохраняем exchange с балансом 1200 USDT
    print("\n1. Creating initial exchange state...")
    exchange = PaperExchange(initial_capital=1000.0)
    exchange.balance = 1200.0  # Прибыль +200 USDT
    exchange.save_state(str(state_file))

    print(f"  ✓ Saved state: balance=1200.0 USDT, equity=1200.0 USDT")

    # Шаг 2: Проверяем что файл существует
    assert state_file.exists(), "State file should exist"

    # Шаг 3: Загружаем состояние (симулируем рестарт бота)
    print("\n2. Loading state (simulating bot restart)...")

    if state_file.exists():
        try:
            loaded_exchange = PaperExchange.load_state(str(state_file))
            loaded_balance = loaded_exchange.get_balance()
            loaded_equity = loaded_exchange.get_equity()

            print(f"  ✓ State loaded successfully!")
            print(f"  ✓ Balance: {loaded_balance:.2f} USDT")
            print(f"  ✓ Equity: {loaded_equity:.2f} USDT")

            # Проверяем что баланс восстановлен
            assert loaded_balance == 1200.0, \
                f"Balance should be 1200.0, got {loaded_balance}"

            print("\n✅ Balance and equity preserved after restart!")

        except Exception as e:
            print(f"  ✗ Failed to load state: {e}")
            return False
    else:
        print("  ✗ State file does not exist")
        return False

    # Очистка
    if state_file.exists():
        state_file.unlink()
    if (state_file.parent / f"{state_file.name}.backup").exists():
        (state_file.parent / f"{state_file.name}.backup").unlink()
    if data_dir.exists():
        try:
            data_dir.rmdir()
        except:
            pass

    print("\n✅ BOT INITIALIZATION TEST PASSED!")
    return True


if __name__ == '__main__':
    print("=" * 80)
    print("PAPER EXCHANGE STATE PERSISTENCE TESTS")
    print("=" * 80)

    try:
        result1 = test_state_persistence()
        result2 = test_bot_initialization_with_saved_state()

        if result1 and result2:
            print("\n" + "=" * 80)
            print("✅ ALL STATE PERSISTENCE TESTS PASSED!")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("❌ SOME TESTS FAILED")
            print("=" * 80)
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
