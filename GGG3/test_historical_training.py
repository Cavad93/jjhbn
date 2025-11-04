#!/usr/bin/env python3
"""
Тестовый скрипт для проверки обучения на реальных данных

Собирает небольшое количество данных (100 раундов) и тестирует обучение
"""
import sys
import os
import json
import time
from datetime import datetime

# Добавляем путь
sys.path.insert(0, '/home/user/jjhbn/GGG3')

from collect_historical_data import connect_web3, collect_data, PREDICTION_ABI, PREDICTION_ADDR
from web3 import Web3

print(f"{'='*70}")
print("ТЕСТОВЫЙ ЗАПУСК СБОРА ДАННЫХ")
print(f"{'='*70}\n")

try:
    # Подключаемся к BSC
    print("📡 Подключение к BSC...")
    w3 = connect_web3()

    # Создаем контракт
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(PREDICTION_ADDR),
        abi=PREDICTION_ABI
    )

    # Собираем только 100 раундов для теста
    print("\n📊 Сбор 100 раундов для теста...")
    start_time = time.time()

    rounds = collect_data(
        w3,
        contract,
        n_rounds=100,
        output_file='test_historical_100.json'
    )

    elapsed = time.time() - start_time

    if rounds and len(rounds) >= 10:
        print(f"\n{'='*70}")
        print("✅ ТЕСТ УСПЕШЕН!")
        print(f"{'='*70}")
        print(f"Собрано раундов: {len(rounds)}")
        print(f"Время: {elapsed:.1f}s")
        print(f"Файл: test_historical_100.json")
        print(f"\n💡 Теперь можно запустить полный сбор данных с 2022 года")
        print(f"{'='*70}\n")
    else:
        print(f"\n❌ Ошибка: собрано мало данных ({len(rounds) if rounds else 0})")

except KeyboardInterrupt:
    print(f"\n\n⚠️  Прервано пользователем")
except Exception as e:
    print(f"\n❌ ОШИБКА: {e}")
    import traceback
    traceback.print_exc()
