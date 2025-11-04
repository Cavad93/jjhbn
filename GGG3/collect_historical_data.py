#!/usr/bin/env python3
"""
Скрипт сбора исторических данных PancakeSwap Prediction для обучения META Neural

ПЛАН:
1. Получить последние N завершенных раундов через BSC RPC
2. Получить исторические цены BNB из Binance API
3. Симулировать работу бота (расчет фич, предсказания экспертов)
4. Сохранить данные для обучения META Neural

ЦЕЛЬ: Собрать минимум 5,000 примеров (для начала)
"""
import sys
import os
import json
import time
from datetime import datetime, timedelta
import requests
import numpy as np
from web3 import Web3, HTTPProvider

# Настройки
RPC_URLS = [
    "https://bsc-dataseed.binance.org/",
    "https://bsc-dataseed1.defibit.io/",
    "https://bsc-dataseed1.ninicoin.io/",
    "https://bsc-rpc.publicnode.com",
    "https://rpc.ankr.com/bsc"
]
PREDICTION_ADDR = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"
BINANCE_API = "https://api.binance.com/api/v3/klines"

# Упрощенный ABI (только нужные функции)
PREDICTION_ABI = json.loads(r"""[
    {
        "inputs": [{"internalType": "uint256", "name": "epoch", "type": "uint256"}],
        "name": "rounds",
        "outputs": [
            {"internalType": "uint256", "name": "epoch", "type": "uint256"},
            {"internalType": "uint256", "name": "startTimestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "lockTimestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "closeTimestamp", "type": "uint256"},
            {"internalType": "int256", "name": "lockPrice", "type": "int256"},
            {"internalType": "int256", "name": "closePrice", "type": "int256"},
            {"internalType": "uint256", "name": "lockOracleId", "type": "uint256"},
            {"internalType": "uint256", "name": "closeOracleId", "type": "uint256"},
            {"internalType": "uint256", "name": "totalAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "bullAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "bearAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "rewardBaseCalAmount", "type": "uint256"},
            {"internalType": "uint256", "name": "rewardAmount", "type": "uint256"},
            {"internalType": "bool", "name": "oracleCalled", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "currentEpoch",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]""")

def connect_web3():
    """Подключение к BSC (пробует несколько RPC)"""
    print("🔗 Подключение к BSC RPC...")

    for i, rpc_url in enumerate(RPC_URLS):
        try:
            print(f"  Попытка {i+1}/{len(RPC_URLS)}: {rpc_url[:40]}...")
            w3 = Web3(HTTPProvider(rpc_url, request_kwargs={'timeout': 30}))

            if w3.is_connected():
                block_num = w3.eth.block_number
                print(f"✅ Подключено к BSC (block: {block_num:,})")
                return w3
        except Exception as e:
            print(f"  ❌ Не удалось: {str(e)[:50]}")
            continue

    raise Exception("Не удалось подключиться ни к одному BSC RPC")

def get_current_epoch(contract):
    """Получить текущий epoch"""
    try:
        epoch = contract.functions.currentEpoch().call()
        return epoch
    except Exception as e:
        print(f"❌ Ошибка получения currentEpoch: {e}")
        return None

def fetch_round(contract, epoch):
    """Получить данные раунда"""
    try:
        round_data = contract.functions.rounds(epoch).call()

        # Парсим данные
        return {
            'epoch': round_data[0],
            'startTimestamp': round_data[1],
            'lockTimestamp': round_data[2],
            'closeTimestamp': round_data[3],
            'lockPrice': round_data[4],
            'closePrice': round_data[5],
            'totalAmount': round_data[8],
            'bullAmount': round_data[9],
            'bearAmount': round_data[10],
            'oracleCalled': round_data[13]
        }
    except Exception as e:
        # print(f"  Ошибка раунда {epoch}: {e}")
        return None

def fetch_historical_prices(start_time, end_time, interval='5m'):
    """Получить исторические цены BNB из Binance"""
    print(f"📊 Загрузка цен BNB с Binance ({start_time} - {end_time})...")

    try:
        params = {
            'symbol': 'BNBUSDT',
            'interval': interval,
            'startTime': int(start_time * 1000),
            'endTime': int(end_time * 1000),
            'limit': 1000
        }

        response = requests.get(BINANCE_API, params=params, timeout=10)
        response.raise_for_status()

        klines = response.json()

        # Конвертируем в словарь timestamp -> OHLCV
        prices = {}
        for k in klines:
            ts = int(k[0]) // 1000  # timestamp в секундах
            prices[ts] = {
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            }

        print(f"✅ Загружено {len(prices)} свечей")
        return prices

    except Exception as e:
        print(f"❌ Ошибка загрузки цен: {e}")
        return {}

def calculate_simple_features(prices_window):
    """Упрощенный расчет фич (без полного бота)"""
    if len(prices_window) < 20:
        return None

    closes = np.array([p['close'] for p in prices_window])
    volumes = np.array([p['volume'] for p in prices_window])

    # Momentum
    momentum_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
    momentum_10 = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0

    # Volatility
    volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) if len(closes) >= 10 else 0

    # Volume Z-score
    vol_mean = np.mean(volumes[-20:])
    vol_std = np.std(volumes[-20:])
    vol_zscore = (volumes[-1] - vol_mean) / vol_std if vol_std > 0 else 0

    # RSI упрощенный
    deltas = np.diff(closes[-14:])
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    rs = gains / losses if losses > 0 else 1
    rsi = 100 - (100 / (1 + rs))

    return {
        'momentum_5m': momentum_5,
        'momentum_10m': momentum_10,
        'volatility': volatility,
        'vol_zscore': vol_zscore,
        'rsi': rsi,
        'close': closes[-1]
    }

def collect_data(w3, contract, n_rounds=1000, output_file='historical_data.json'):
    """Основная функция сбора данных"""
    print(f"\n{'='*60}")
    print(f"СБОР ИСТОРИЧЕСКИХ ДАННЫХ")
    print(f"Цель: {n_rounds:,} раундов")
    print(f"{'='*60}\n")

    # Получаем текущий epoch
    current_epoch = get_current_epoch(contract)
    if not current_epoch:
        print("❌ Не удалось получить current epoch")
        return

    print(f"📍 Текущий epoch: {current_epoch:,}")

    # Определяем диапазон
    start_epoch = max(1, current_epoch - n_rounds - 100)  # +100 запас
    end_epoch = current_epoch - 10  # -10 чтобы не брать активные

    print(f"📥 Загрузка раундов {start_epoch:,} - {end_epoch:,}...")

    # Собираем раунды
    rounds = []
    valid_rounds = 0

    for epoch in range(start_epoch, end_epoch + 1):
        if len(rounds) >= n_rounds:
            break

        round_data = fetch_round(contract, epoch)

        if round_data and round_data['oracleCalled'] and round_data['closePrice'] > 0:
            rounds.append(round_data)
            valid_rounds += 1

        if (epoch - start_epoch + 1) % 100 == 0:
            print(f"  Обработано: {epoch - start_epoch + 1:,}, валидных: {valid_rounds:,}", end='\r')

        time.sleep(0.1)  # Не спамим RPC

    print(f"\n✅ Собрано {len(rounds):,} валидных раундов")

    if len(rounds) == 0:
        print("❌ Нет данных для обработки")
        return

    # Получаем временной диапазон
    min_ts = min(r['lockTimestamp'] for r in rounds)
    max_ts = max(r['closeTimestamp'] for r in rounds)

    # Загружаем цены
    prices = fetch_historical_prices(min_ts - 3600, max_ts + 3600)  # +1h запас

    # Обрабатываем раунды
    print(f"\n📊 Обработка раундов с фичами...")
    processed_data = []

    for i, round_data in enumerate(rounds):
        lock_ts = round_data['lockTimestamp']
        close_ts = round_data['closeTimestamp']

        # Получаем окно цен перед lock
        price_window = []
        for ts in range(lock_ts - 1200, lock_ts, 300):  # 20 минут по 5 мин
            if ts in prices:
                price_window.append(prices[ts])

        if len(price_window) < 3:
            continue

        # Рассчитываем фичи
        features = calculate_simple_features(price_window)
        if not features:
            continue

        # Определяем outcome
        lock_price = round_data['lockPrice'] / 1e8
        close_price = round_data['closePrice'] / 1e8
        outcome = 1 if close_price > lock_price else 0

        # Определяем фазу рынка (упрощенно)
        volatility = features['volatility']
        if volatility < 0.005:
            phase = 0  # low volatility
        elif volatility < 0.01:
            phase = 1  # medium
        elif volatility < 0.015:
            phase = 2  # medium-high
        elif volatility < 0.02:
            phase = 3  # high
        elif volatility < 0.03:
            phase = 4  # very high
        else:
            phase = 5  # extreme

        processed_data.append({
            'epoch': round_data['epoch'],
            'timestamp': lock_ts,
            'lock_price': lock_price,
            'close_price': close_price,
            'outcome': outcome,
            'phase': phase,
            'features': features,
            'bull_amount': round_data['bullAmount'] / 1e18,
            'bear_amount': round_data['bearAmount'] / 1e18
        })

        if (i + 1) % 100 == 0:
            print(f"  Обработано: {i+1}/{len(rounds)}", end='\r')

    print(f"\n✅ Обработано {len(processed_data):,} раундов с фичами")

    # Сохраняем
    print(f"\n💾 Сохранение в {output_file}...")
    with open(output_file, 'w') as f:
        json.dump({
            'meta': {
                'collected_at': datetime.now().isoformat(),
                'total_rounds': len(processed_data),
                'start_epoch': processed_data[0]['epoch'] if processed_data else 0,
                'end_epoch': processed_data[-1]['epoch'] if processed_data else 0
            },
            'rounds': processed_data
        }, f, indent=2)

    print(f"✅ Данные сохранены!")

    # Статистика
    print(f"\n{'='*60}")
    print("СТАТИСТИКА")
    print(f"{'='*60}")
    print(f"Всего раундов: {len(processed_data):,}")

    outcomes = [r['outcome'] for r in processed_data]
    print(f"UP раундов: {sum(outcomes):,} ({sum(outcomes)/len(outcomes)*100:.1f}%)")
    print(f"DOWN раундов: {len(outcomes) - sum(outcomes):,} ({(1-sum(outcomes)/len(outcomes))*100:.1f}%)")

    phases = [r['phase'] for r in processed_data]
    for phase in range(6):
        count = sum(1 for p in phases if p == phase)
        print(f"Фаза {phase}: {count:,} раундов ({count/len(phases)*100:.1f}%)")

    return processed_data

def main():
    """Главная функция"""
    try:
        # Подключаемся
        w3 = connect_web3()

        # Создаем контракт
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(PREDICTION_ADDR),
            abi=PREDICTION_ABI
        )

        # Собираем данные
        data = collect_data(
            w3,
            contract,
            n_rounds=5000,  # Начинаем с 5000
            output_file='pancakeswap_historical_5k.json'
        )

        if data and len(data) >= 1000:
            print(f"\n🎉 УСПЕХ! Собрано {len(data):,} примеров")
            print(f"📁 Файл: pancakeswap_historical_5k.json")
            print(f"\n💡 Следующий шаг: Обучение экспертов и META Neural")
        else:
            print(f"\n⚠️  Собрано мало данных: {len(data) if data else 0}")
            print(f"💡 Попробуйте увеличить n_rounds или проверить RPC")

    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱️  Время: {elapsed/60:.1f} минут")
