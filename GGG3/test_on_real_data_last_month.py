#!/usr/bin/env python3
"""
Тест базовой модели на РЕАЛЬНЫХ данных за последний месяц

Этапы:
1. Собрать последние ~9000 раундов из PancakeSwap Prediction (≈ 1 месяц)
2. Получить исторические цены BNB из Binance
3. Применить базовую модель из bnbusdrt6.py
4. Вычислить метрики: accuracy, winrate, ROI, Brier score
"""
import sys
import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import requests
import numpy as np
import pandas as pd

# Добавляем путь к GGG3
sys.path.insert(0, os.path.dirname(__file__))

# Web3 для BSC
from web3 import Web3, HTTPProvider

# Импорт функций базовой модели
try:
    from bnbusdrt6 import (
        features_from_binance,
        prob_up_down_at_time
    )
    print("✅ Импортированы функции базовой модели")
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    IMPORTS_OK = False

# Настройки
RPC_URLS = [
    "https://bsc-dataseed.binance.org/",
    "https://bsc-dataseed1.defibit.io/",
    "https://bsc-dataseed1.ninicoin.io/",
]
PREDICTION_ADDR = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"
BINANCE_API = "https://api.binance.com/api/v3/klines"

MAX_WORKERS = 20
BATCH_SIZE = 100

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
    """Подключение к BSC"""
    print("🔗 Подключение к BSC...")
    for rpc_url in RPC_URLS:
        try:
            w3 = Web3(HTTPProvider(rpc_url, request_kwargs={'timeout': 30}))
            if w3.is_connected():
                print(f"✅ Подключено к {rpc_url[:40]}...")
                return w3
        except Exception as e:
            print(f"  ❌ {rpc_url[:40]}... - {str(e)[:50]}")
            continue
    raise Exception("Не удалось подключиться к BSC")


def fetch_round_batch(contract, epochs: List[int]) -> List[Optional[Dict]]:
    """Загружает батч раундов"""
    results = []
    for epoch in epochs:
        try:
            round_data = contract.functions.rounds(epoch).call()
            results.append({
                'epoch': round_data[0],
                'startTimestamp': round_data[1],
                'lockTimestamp': round_data[2],
                'closeTimestamp': round_data[3],
                'lockPrice': round_data[4],
                'closePrice': round_data[5],
                'oracleCalled': round_data[13]
            })
        except Exception:
            results.append(None)
    return results


def collect_last_month_rounds(w3, contract, n_rounds=9000):
    """Собирает последние N раундов"""
    print(f"\n📥 Сбор последних {n_rounds:,} раундов из BSC...")

    current_epoch = contract.functions.currentEpoch().call()
    print(f"   Текущий epoch: {current_epoch:,}")

    start_epoch = max(1, current_epoch - n_rounds - 100)
    end_epoch = current_epoch - 2  # Пропускаем последние 2 (могут быть незавершенные)

    all_epochs = list(range(start_epoch, end_epoch + 1))
    batches = [all_epochs[i:i + BATCH_SIZE] for i in range(0, len(all_epochs), BATCH_SIZE)]

    rounds = []
    processed_batches = 0

    print(f"   Загрузка {len(batches)} батчей...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_round_batch, contract, batch): batch for batch in batches}

        for future in as_completed(futures):
            batch_results = future.result()

            for round_data in batch_results:
                if round_data and round_data['oracleCalled'] and round_data['closePrice'] > 0:
                    rounds.append(round_data)

            processed_batches += 1
            print(f"     Батчей: {processed_batches}/{len(batches)}, Валидных: {len(rounds):,}", end='\r')

            if len(rounds) >= n_rounds:
                break

            time.sleep(0.05)

    print(f"\n   ✅ Собрано {len(rounds):,} валидных раундов")

    # Оставляем только последние n_rounds
    if len(rounds) > n_rounds:
        rounds = rounds[-n_rounds:]
        print(f"   Обрезано до последних {len(rounds):,} раундов")

    return rounds


def fetch_binance_prices_batch(start_time, end_time, interval='1m'):
    """Загружает цены BNB с Binance"""
    print(f"\n📊 Загрузка цен BNB с Binance...")
    print(f"   Период: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d')} - "
          f"{datetime.fromtimestamp(end_time).strftime('%Y-%m-%d')}")

    all_prices = {}
    current_start = start_time
    chunk_duration = 3.4 * 24 * 3600  # ~3.4 дня (1000 свечей * 5 минут)

    chunks_loaded = 0
    total_chunks = int((end_time - start_time) / chunk_duration) + 1

    while current_start < end_time:
        chunk_end = min(current_start + int(chunk_duration), end_time)

        try:
            params = {
                'symbol': 'BNBUSDT',
                'interval': interval,
                'startTime': int(current_start * 1000),
                'endTime': int(chunk_end * 1000),
                'limit': 1000
            }

            response = requests.get(BINANCE_API, params=params, timeout=10)
            response.raise_for_status()
            klines = response.json()

            for k in klines:
                ts = int(k[0]) // 1000
                all_prices[ts] = {
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                }

            chunks_loaded += 1
            print(f"     Чанков: {chunks_loaded}/{total_chunks}, Свечей: {len(all_prices):,}", end='\r')

        except Exception as e:
            print(f"\n   ⚠️  Ошибка чанка: {e}")

        current_start = chunk_end
        time.sleep(0.2)

    print(f"\n   ✅ Загружено {len(all_prices):,} 1-минутных свечей")
    return all_prices


def get_nearest_price(prices_dict, target_ts, tolerance=60):
    """Находит ближайшую цену к target_ts"""
    best_ts = None
    best_diff = float('inf')

    for ts in range(target_ts - tolerance, target_ts + tolerance + 1, 10):
        if ts in prices_dict:
            diff = abs(ts - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_ts = ts

    if best_ts is not None and best_diff <= tolerance:
        return prices_dict[best_ts]
    return None


def build_ohlcv_dataframe(prices_dict, min_ts, max_ts):
    """Строит DataFrame OHLCV из словаря цен"""
    print(f"\n🔨 Построение OHLCV DataFrame...")

    # Создаем список timestamp с 1-минутным интервалом
    timestamps = []
    data = []

    current_ts = min_ts
    while current_ts <= max_ts:
        price_data = get_nearest_price(prices_dict, current_ts, tolerance=60)

        if price_data:
            timestamps.append(pd.Timestamp(current_ts, unit='s', tz='UTC'))
            data.append({
                'open': price_data['open'],
                'high': price_data['high'],
                'low': price_data['low'],
                'close': price_data['close'],
                'volume': price_data['volume']
            })

        current_ts += 60  # 1 минута

    df = pd.DataFrame(data, index=timestamps)
    print(f"   ✅ DataFrame: {len(df):,} свечей")
    print(f"   Период: {df.index[0]} - {df.index[-1]}")

    return df


def test_baseline_model_on_rounds(rounds, df_ohlcv):
    """Тестирует базовую модель на реальных раундах"""
    print(f"\n🧪 Тестирование базовой модели...")

    if not IMPORTS_OK:
        print("❌ Функции не импортированы")
        return None

    # Вычисляем фичи
    print("   Вычисление фич из OHLCV...")
    try:
        feats = features_from_binance(df_ohlcv)
        print(f"   ✅ Фичи вычислены")
    except Exception as e:
        print(f"   ❌ Ошибка вычисления фич: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Прогнозы для каждого раунда
    print(f"   Применение базовой модели к {len(rounds):,} раундам...")

    predictions = []
    skipped = 0

    for i, rd in enumerate(rounds):
        try:
            lock_ts = pd.Timestamp(rd['lockTimestamp'], unit='s', tz='UTC')

            # Проверяем что timestamp в диапазоне df_ohlcv
            if lock_ts < df_ohlcv.index[0] or lock_ts > df_ohlcv.index[-1]:
                skipped += 1
                continue

            # БАЗОВАЯ МОДЕЛЬ
            P_up, P_dn, _ = prob_up_down_at_time(feats, lock_ts, w_dyn=None)

            # Outcome
            lock_price = rd['lockPrice'] / 1e8
            close_price = rd['closePrice'] / 1e8
            outcome = 1 if close_price > lock_price else 0

            predictions.append({
                'epoch': rd['epoch'],
                'lock_ts': rd['lockTimestamp'],
                'p_up': P_up,
                'p_dn': P_dn,
                'pred_binary': 1 if P_up > 0.5 else 0,
                'outcome': outcome,
                'lock_price': lock_price,
                'close_price': close_price
            })

            if (i + 1) % 1000 == 0:
                print(f"     Обработано: {i+1:,}/{len(rounds):,} (пропущено: {skipped})", end='\r')

        except Exception as e:
            skipped += 1
            if skipped < 5:
                print(f"\n   ⚠️  Ошибка раунда {i}: {e}")
            continue

    print(f"\n   ✅ Получено {len(predictions):,} прогнозов (пропущено: {skipped})")

    return predictions


def evaluate_results(predictions):
    """Оценивает результаты"""
    if not predictions or len(predictions) < 100:
        print(f"\n❌ Недостаточно прогнозов: {len(predictions) if predictions else 0}")
        return None

    print(f"\n{'='*70}")
    print(f"📊 РЕЗУЛЬТАТЫ НА РЕАЛЬНЫХ ДАННЫХ ЗА ПОСЛЕДНИЙ МЕСЯЦ")
    print(f"{'='*70}")

    n = len(predictions)

    # Accuracy
    correct = sum(1 for p in predictions if p['pred_binary'] == p['outcome'])
    accuracy = correct / n

    # Precision / Recall / F1
    tp = sum(1 for p in predictions if p['pred_binary'] == 1 and p['outcome'] == 1)
    fp = sum(1 for p in predictions if p['pred_binary'] == 1 and p['outcome'] == 0)
    fn = sum(1 for p in predictions if p['pred_binary'] == 0 and p['outcome'] == 1)
    tn = sum(1 for p in predictions if p['pred_binary'] == 0 and p['outcome'] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Winrate
    up_winrate = tp / (tp + fp) if (tp + fp) > 0 else 0

    # ROI с комиссией 3%
    commission = 0.03
    roi_total = 0.0
    for p in predictions:
        if p['pred_binary'] == 1:
            if p['outcome'] == 1:
                roi_total += (1 - commission)
            else:
                roi_total -= 1.0

    roi_per_bet = roi_total / n

    # Brier Score
    brier = sum((p['p_up'] - p['outcome'])**2 for p in predictions) / n

    # Log Loss
    eps = 1e-7
    log_loss = 0.0
    for p in predictions:
        prob = max(eps, min(1-eps, p['p_up']))
        log_loss += -(p['outcome'] * np.log(prob) + (1-p['outcome']) * np.log(1-prob))
    log_loss /= n

    # Статистика
    p_ups = [p['p_up'] for p in predictions]
    outcomes = [p['outcome'] for p in predictions]

    print(f"\n📈 МЕТРИКИ НА {n:,} РЕАЛЬНЫХ РАУНДАХ:\n")
    print(f"   Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision (UP):     {precision:.4f}")
    print(f"   Recall (UP):        {recall:.4f}")
    print(f"   F1 Score:           {f1:.4f}")
    print(f"\n   Winrate (UP ставки): {up_winrate:.4f} ({up_winrate*100:.2f}%)")
    print(f"   Break-even:         50.76%")

    if up_winrate > 0.5076:
        print(f"   ✅ Winrate выше break-even на +{(up_winrate-0.5076)*100:.2f}%")
    else:
        print(f"   ❌ Winrate ниже break-even на {(0.5076-up_winrate)*100:.2f}%")

    print(f"\n   Brier Score:        {brier:.4f}")
    print(f"   Log Loss:           {log_loss:.4f}")

    print(f"\n   Total ROI:          {roi_total:+.2f} USDT (на {n:,} ставок)")
    print(f"   ROI per bet:        {roi_per_bet:+.5f} USDT")

    if roi_total > 0:
        print(f"   ✅ ПРИБЫЛЬНО (+{roi_total:.2f} USDT)")
    else:
        print(f"   ❌ УБЫТОЧНО ({roi_total:.2f} USDT)")

    print(f"\n   Confusion Matrix:")
    print(f"                Predicted")
    print(f"                UP      DOWN")
    print(f"   Actual UP    {tp:5d}   {fn:5d}")
    print(f"   Actual DOWN  {fp:5d}   {tn:5d}")

    print(f"\n📊 СТАТИСТИКА:\n")
    print(f"   Средняя P(UP):      {np.mean(p_ups):.4f}")
    print(f"   Std P(UP):          {np.std(p_ups):.4f}")
    print(f"   Min/Max P(UP):      {min(p_ups):.4f} / {max(p_ups):.4f}")
    print(f"   Фактический UP rate: {np.mean(outcomes):.4f}")

    # Месячная экстраполяция
    print(f"\n💰 ЭКСТРАПОЛЯЦИЯ (если торговать каждый раунд):\n")
    bets_per_day = 288
    monthly_roi = roi_per_bet * bets_per_day * 30
    monthly_pct = (monthly_roi / 100) * 100

    print(f"   Ставок в день:      {bets_per_day}")
    print(f"   ROI per bet:        {roi_per_bet:+.5f} USDT")
    print(f"   Месячный ROI:       {monthly_roi:+.2f} USDT")
    print(f"   Месячная доходность: {monthly_pct:+.1f}% (на 100 USDT)")

    print(f"\n{'='*70}\n")

    return {
        'accuracy': accuracy,
        'winrate': up_winrate,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier': brier,
        'log_loss': log_loss,
        'roi_total': roi_total,
        'roi_per_bet': roi_per_bet,
        'n_samples': n
    }


def main():
    """Главная функция"""
    print("="*70)
    print("🧪 ТЕСТ БАЗОВОЙ МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ (ПОСЛЕДНИЙ МЕСЯЦ)")
    print("="*70)

    if not IMPORTS_OK:
        print("\n❌ Невозможно выполнить: функции не импортированы")
        return

    try:
        # Подключение
        w3 = connect_web3()
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(PREDICTION_ADDR),
            abi=PREDICTION_ABI
        )

        # Сбор раундов
        rounds = collect_last_month_rounds(w3, contract, n_rounds=9000)

        if len(rounds) < 100:
            print(f"\n❌ Недостаточно раундов: {len(rounds)}")
            return

        # Получаем временной диапазон
        min_ts = min(r['lockTimestamp'] for r in rounds) - 3600  # -1 час для истории
        max_ts = max(r['closeTimestamp'] for r in rounds) + 3600

        # Загрузка цен
        prices = fetch_binance_prices_batch(min_ts, max_ts, interval='1m')

        if len(prices) < 1000:
            print(f"\n❌ Недостаточно цен: {len(prices)}")
            return

        # Построение DataFrame
        df_ohlcv = build_ohlcv_dataframe(prices, min_ts, max_ts)

        # Тестирование
        predictions = test_baseline_model_on_rounds(rounds, df_ohlcv)

        if not predictions:
            print("\n❌ Не удалось получить прогнозы")
            return

        # Оценка
        metrics = evaluate_results(predictions)

        # Сохранение результатов
        if metrics:
            output_file = "test_results_real_data.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'test_date': datetime.now().isoformat(),
                    'n_rounds': len(rounds),
                    'n_predictions': len(predictions),
                    'metrics': metrics,
                    'first_epoch': predictions[0]['epoch'] if predictions else None,
                    'last_epoch': predictions[-1]['epoch'] if predictions else None
                }, f, indent=2)
            print(f"💾 Результаты сохранены в {output_file}")

        # Выводы
        print("\n🎯 ВЫВОДЫ:")
        print("="*70 + "\n")

        if metrics:
            if metrics['accuracy'] > 0.52:
                print("✅ БАЗОВАЯ МОДЕЛЬ РАБОТАЕТ НА РЕАЛЬНЫХ ДАННЫХ!")
                print(f"   Accuracy {metrics['accuracy']*100:.1f}% > 52%")
            else:
                print("⚠️  Accuracy недостаточен для стабильной прибыли")
                print(f"   Accuracy {metrics['accuracy']*100:.1f}% < 52%")

            if metrics['winrate'] > 0.5076:
                print(f"\n✅ Winrate {metrics['winrate']*100:.1f}% выше break-even (50.76%)")
                print(f"   Теоретически прибыльно с правильным MM")
            else:
                print(f"\n❌ Winrate {metrics['winrate']*100:.1f}% ниже break-even")

            if metrics['roi_total'] > 0:
                print(f"\n✅ Положительный ROI: {metrics['roi_total']:+.2f} USDT")
            else:
                print(f"\n❌ Отрицательный ROI: {metrics['roi_total']:+.2f} USDT")

            print(f"\n💡 РЕКОМЕНДАЦИИ:")
            if metrics['accuracy'] >= 0.54:
                print(f"   • Базовая модель эффективна - можно добавлять ML экспертов")
                print(f"   • Ожидаемое улучшение с XGB/RF/ARF/NN: +2-4% accuracy")
                print(f"   • С META слоем: еще +1-2% accuracy")
            elif metrics['accuracy'] >= 0.52:
                print(f"   • Базовая модель на грани прибыльности")
                print(f"   • ML эксперты КРИТИЧЕСКИ важны для улучшения")
                print(f"   • Обязательно обучить все 4 эксперта + META")
            else:
                print(f"   • Базовая модель недостаточно эффективна")
                print(f"   • Требуется полная ML система для прибыльности")
                print(f"   • Рекомендуется paper trading после обучения")

        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
