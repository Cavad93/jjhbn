#!/usr/bin/env python3
"""
Тест для проверки сохранения и загрузки состояния PaperExchange
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from paper_trading.paper_exchange import PaperExchange

def test_save_load():
    """Тест сохранения и загрузки состояния"""

    print("="*80)
    print("ТЕСТ: Сохранение и загрузка состояния PaperExchange")
    print("="*80)

    # 1. Создаем новую биржу с начальным капиталом
    print("\n1. Создание новой PaperExchange с $10000...")
    exchange1 = PaperExchange(initial_capital=10000.0)
    print(f"   Начальный баланс: ${exchange1.balance:.2f}")

    # 2. Симулируем некоторые изменения баланса
    print("\n2. Изменение баланса (симуляция сделки)...")
    exchange1.balance = 12500.50  # Симуляция прибыли
    print(f"   Новый баланс: ${exchange1.balance:.2f}")

    # 3. Сохраняем состояние
    print("\n3. Сохранение состояния в test_exchange_state.json...")
    exchange1.save_state('test_exchange_state.json')
    print("   ✅ Состояние сохранено")

    # 4. Загружаем состояние в новый объект
    print("\n4. Загрузка состояния из test_exchange_state.json...")
    exchange2 = PaperExchange.load_state('test_exchange_state.json')
    print(f"   Загруженный баланс: ${exchange2.balance:.2f}")

    # 5. Проверяем, что баланс совпадает
    print("\n5. Проверка корректности...")
    if abs(exchange1.balance - exchange2.balance) < 0.01:
        print("   ✅ УСПЕХ: Баланс корректно сохранен и загружен!")
        print(f"   Сохранен: ${exchange1.balance:.2f}")
        print(f"   Загружен: ${exchange2.balance:.2f}")
    else:
        print("   ❌ ОШИБКА: Балансы не совпадают!")
        print(f"   Сохранен: ${exchange1.balance:.2f}")
        print(f"   Загружен: ${exchange2.balance:.2f}")
        return False

    # 6. Проверяем, что initial_capital тоже сохранен
    print("\n6. Проверка initial_capital...")
    if abs(exchange1.initial_capital - exchange2.initial_capital) < 0.01:
        print(f"   ✅ УСПЕХ: initial_capital корректно сохранен (${exchange2.initial_capital:.2f})")
    else:
        print(f"   ❌ ОШИБКА: initial_capital не совпадает!")
        return False

    # 7. Удаляем тестовый файл
    print("\n7. Очистка тестовых файлов...")
    if os.path.exists('test_exchange_state.json'):
        os.remove('test_exchange_state.json')
        print("   ✅ Тестовый файл удален")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("="*80)

    return True

if __name__ == "__main__":
    try:
        success = test_save_load()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ОШИБКА ПРИ ВЫПОЛНЕНИИ ТЕСТА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
