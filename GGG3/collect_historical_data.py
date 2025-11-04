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

def _ema(series, n):
    """Exponential Moving Average"""
    alpha = 2.0 / (n + 1)
    ema_vals = np.zeros(len(series))
    ema_vals[0] = series[0]
    for i in range(1, len(series)):
        ema_vals[i] = alpha * series[i] + (1 - alpha) * ema_vals[i-1]
    return ema_vals

def _macd_hist(closes, fast=12, slow=26, sig=9):
    """MACD Histogram calculation"""
    if len(closes) < slow:
        return 0.0

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd = ema_fast - ema_slow
    signal = _ema(macd, sig)
    hist = macd - signal
    return hist[-1]

def calculate_simple_features(prices_window):
    """
    Расширенный расчет фич из OHLCV данных

    Генерирует максимум фич из доступных исторических данных.
    Недоступные фичи (order book, funding, gas) будут заполнены нулями в prepare_features().
    """
    if len(prices_window) < 20:
        return None

    opens = np.array([p['open'] for p in prices_window])
    highs = np.array([p['high'] for p in prices_window])
    lows = np.array([p['low'] for p in prices_window])
    closes = np.array([p['close'] for p in prices_window])
    volumes = np.array([p['volume'] for p in prices_window])

    # ========== БАЗОВЫЕ ИНДИКАТОРЫ ==========

    # Momentum на разных таймфреймах
    momentum_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
    momentum_10 = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
    momentum_15 = (closes[-1] - closes[-15]) / closes[-15] if len(closes) >= 15 else 0

    # Volatility
    volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) if len(closes) >= 10 else 0
    volatility_20 = np.std(closes[-20:]) / np.mean(closes[-20:]) if len(closes) >= 20 else 0

    # Volume Z-score
    vol_mean = np.mean(volumes[-20:])
    vol_std = np.std(volumes[-20:])
    vol_zscore = (volumes[-1] - vol_mean) / vol_std if vol_std > 0 else 0

    # RSI (текущий и лаги)
    def calc_rsi(prices, period=14):
        deltas = np.diff(prices[-period-1:])
        gains = deltas[deltas > 0].sum()
        losses = -deltas[deltas < 0].sum()
        rs = gains / losses if losses > 0 else 1
        return 100 - (100 / (1 + rs))

    rsi = calc_rsi(closes, 14)
    rsi_lag1 = calc_rsi(closes[:-1], 14) if len(closes) > 15 else rsi
    rsi_lag5 = calc_rsi(closes[:-5], 14) if len(closes) > 19 else rsi

    # ATR (Average True Range)
    tr = np.maximum(highs[1:] - lows[1:],
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1]))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    atr_norm = atr / closes[-1] if closes[-1] > 0 else 0

    # ATR SMA (для vol_ratio как в meta_ctx.py)
    if len(tr) >= 34:  # Нужно минимум 14 + 20 значений
        atr_series = []
        for i in range(14, len(tr) + 1):
            atr_series.append(np.mean(tr[i-14:i]))
        atr_sma = np.mean(atr_series[-20:])
    else:
        atr_sma = atr

    # ========== ПРОИЗВОДНЫЕ ==========

    # RSI volatility
    rsi_values = []
    for i in range(max(0, len(closes) - 20), len(closes)):
        if i >= 14:
            rsi_values.append(calc_rsi(closes[:i+1], 14))
    rsi_volatility = np.std(rsi_values) if len(rsi_values) > 1 else 0

    # Returns coefficient of variation
    returns = np.diff(closes[-20:]) / closes[-20:-1]
    returns_cv = np.std(returns) / np.abs(np.mean(returns)) if np.abs(np.mean(returns)) > 1e-12 else 0
    returns_cv = min(returns_cv, 5.0)  # Клипируем экстремумы

    # ========== TREND & VOL RATIO (КАК В META_CTX.PY) ==========
    # Resample to 15min for MACD histogram calculation
    # Для упрощения используем скользящее окно: 15 свечей 1m = 1 свеча 15m
    if len(closes) >= 15 * 3:  # Минимум 3 свечи 15m
        closes_15m = []
        for i in range(14, len(closes), 15):
            closes_15m.append(closes[i])
        closes_15m = np.array(closes_15m)

        if len(closes_15m) >= 30:  # Нужно для MACD + z-norm окна
            # MACD histogram на 15m данных
            macd_hist_vals = []
            for i in range(26, len(closes_15m)):
                h = _macd_hist(closes_15m[:i+1])
                macd_hist_vals.append(h)

            if len(macd_hist_vals) > 0:
                # Z-нормализация последнего значения
                macd_hist_current = macd_hist_vals[-1]
                window = min(100, len(macd_hist_vals))
                hist_window = np.array(macd_hist_vals[-window:])
                hist_std = np.std(hist_window) if len(hist_window) > 1 else 0.0

                trend_sign = 1.0 if macd_hist_current > 0 else (-1.0 if macd_hist_current < 0 else 0.0)
                trend_abs = 0.0 if hist_std == 0.0 else min(3.0, max(0.0, abs(macd_hist_current) / hist_std))
            else:
                trend_sign = 0.0
                trend_abs = 0.0
        else:
            trend_sign = 0.0
            trend_abs = 0.0
    else:
        trend_sign = 0.0
        trend_abs = 0.0

    # vol_ratio из ATR (как в meta_ctx.py:155)
    vol_ratio = min(5.0, max(0.0, atr / atr_sma)) if atr_sma > 0 else 0.0

    # ========== ВРЕМЕННЫЕ ЛАГИ ==========

    close_lag1 = closes[-2] if len(closes) >= 2 else closes[-1]
    close_lag5 = closes[-6] if len(closes) >= 6 else closes[-1]
    close_lag15 = closes[-16] if len(closes) >= 16 else closes[-1]

    returns_1 = (closes[-1] - close_lag1) / close_lag1 if close_lag1 > 0 else 0
    returns_5 = (closes[-1] - close_lag5) / close_lag5 if close_lag5 > 0 else 0
    returns_15 = (closes[-1] - close_lag15) / close_lag15 if close_lag15 > 0 else 0

    volume_lag1 = volumes[-2] if len(volumes) >= 2 else volumes[-1]
    volume_change = (volumes[-1] - volume_lag1) / volume_lag1 if volume_lag1 > 0 else 0

    volatility_lag5 = np.std(closes[-15:-5]) / np.mean(closes[-15:-5]) if len(closes) >= 15 else volatility

    # ========== РАСШИРЕННЫЕ ФИЧИ ДЛЯ ПОЛНОГО СООТВЕТСТВИЯ ==========

    # Time features (8)
    # В историческом режиме заполняем нулями, т.к. нет timestamp контекста
    tod_sin = 0.0
    tod_cos = 1.0
    EU = 0.0
    US = 0.0
    ASIA = 0.0
    dow_0, dow_1, dow_2, dow_3, dow_4, dow_5, dow_6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # ИТОГОВЫЙ СЛОВАРЬ
    # Содержит максимум фич извлекаемых из OHLCV
    # Недоступные фичи (order book, funding, gas) заполнятся 0 в prepare_features_exact()
    return {
        # Базовые индикаторы (14)
        'momentum_5m': float(momentum_5),
        'momentum_10m': float(momentum_10),
        'momentum_15m': float(momentum_15),
        'volatility': float(volatility),
        'vol_zscore': float(vol_zscore),
        'rsi': float(rsi),
        'rsi_lag1': float(rsi_lag1),
        'rsi_lag5': float(rsi_lag5),
        'close': float(closes[-1]),
        'high': float(highs[-1]),
        'low': float(lows[-1]),
        'volume': float(volumes[-1]),
        'atr': float(atr),
        'atr_norm': float(atr_norm),

        # Производные (6)
        'rsi_volatility': float(rsi_volatility),
        'returns_cv': float(returns_cv),
        'trend_sign': float(trend_sign),
        'trend_abs': float(trend_abs),
        'vol_ratio': float(vol_ratio),
        'volatility_20': float(volatility_20),

        # Временные лаги (9)
        'close_lag1': float(close_lag1),
        'close_lag5': float(close_lag5),
        'close_lag15': float(close_lag15),
        'returns_1': float(returns_1),
        'returns_5': float(returns_5),
        'returns_15': float(returns_15),
        'volume_lag1': float(volume_lag1),
        'volume_change': float(volume_change),
        'volatility_lag5': float(volatility_lag5),

        # Дополнительные (3)
        'price_range': float(highs[-1] - lows[-1]),
        'hl_ratio': float(highs[-1] / lows[-1]) if lows[-1] > 0 else 1.0,
        'volume_ma': float(np.mean(volumes[-10:])),

        # Normalized position (1)
        'normalized_position': float((closes[-1] - lows[-1]) / max(1e-12, highs[-1] - lows[-1])),

        # Time features (13) - заполняются нулями в историческом режиме
        'tod_sin': tod_sin,
        'tod_cos': tod_cos,
        'EU': EU,
        'US': US,
        'ASIA': ASIA,
        'dow_0': dow_0,
        'dow_1': dow_1,
        'dow_2': dow_2,
        'dow_3': dow_3,
        'dow_4': dow_4,
        'dow_5': dow_5,
        'dow_6': dow_6,
    }
    # ИТОГО: ~46 фич из OHLCV + time features
    # Остальные (order book, funding, gas, P_up) будут заполнены дефолтами в prepare_features_exact()

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

        # Определяем фазу рынка (КАК В META_CTX.PY:260-282)
        # VOL_THRESHOLD = 1.3
        trend_sign = features['trend_sign']
        vol_ratio = features['vol_ratio']

        high_vol = (vol_ratio >= 1.3)

        if trend_sign > 0:
            phase = 1 if high_vol else 0  # bull_high or bull_low
        elif trend_sign < 0:
            phase = 3 if high_vol else 2  # bear_high or bear_low
        else:
            phase = 5 if high_vol else 4  # flat_high or flat_low

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
