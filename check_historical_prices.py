#!/usr/bin/env python3
"""
Проверка исторических цен для позиций, закрытых MANUAL
"""

import requests
from datetime import datetime

# Позиции закрытые MANUAL при ребалансировке
positions = [
    {
        'symbol': 'COTIUSDT',
        'entry': 0.051766,
        'exit': 0.04808,
        'tp': 0.06055374397996071,
        'sl': 0.04512969201502947,
        'closed_time': '2024-11-10 14:53:29',
        'pnl': -7.10
    },
    {
        'symbol': 'VETUSDT',
        'entry': 0.017849,
        'exit': 0.01784,
        'tp': 0.019441295737608805,
        'sl': 0.01687922255743472,
        'closed_time': '2024-11-10 14:53:32',
        'pnl': -1.67
    },
    {
        'symbol': 'BATUSDT',
        'entry': 0.234917,
        'exit': 0.23457,
        'tp': 0.261746274013388,
        'sl': 0.2186322355919672,
        'closed_time': '2024-11-10 14:53:34',
        'pnl': -0.14
    },
    {
        'symbol': 'STXUSDT',
        'entry': 0.427113,
        'exit': 0.42071,
        'tp': 0.4653135769469632,
        'sl': 0.4038518538318221,
        'closed_time': '2024-11-10 18:58:49',
        'pnl': -1.27
    },
    {
        'symbol': 'CRVUSDT',
        'entry': 0.500650,
        'exit': 0.48816,
        'tp': 0.5427715943792546,
        'sl': 0.4751370433724471,
        'closed_time': '2024-11-10 18:58:52',
        'pnl': -2.43
    },
    {
        'symbol': 'MANTAUSDT',
        'entry': 0.123662,
        'exit': 0.11765,
        'tp': 0.13848546752234403,
        'sl': 0.112435899358242,
        'closed_time': '2024-11-10 18:58:54',
        'pnl': -4.90
    },
    {
        'symbol': 'ENAUSDT',
        'entry': 0.353076,
        'exit': 0.33560,
        'tp': 0.38773941793530814,
        'sl': 0.3318363492388151,
        'closed_time': '2024-11-10 18:58:57',
        'pnl': -4.95
    }
]

def get_binance_klines(symbol, start_time, end_time, interval='4h'):
    """Получить свечи с Binance"""
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(start_time),
        'endTime': int(end_time),
        'limit': 1000
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  ❌ Ошибка для {symbol}: {e}")
        return None

def check_if_tp_or_sl_hit(symbol, entry, tp, sl, closed_time):
    """Проверить, достиг ли бы цена TP или SL"""

    # Конвертируем время закрытия в timestamp
    dt = datetime.strptime(closed_time, '%Y-%m-%d %H:%M:%S')
    start_ts = int(dt.timestamp() * 1000)

    # Проверяем следующие 48 часов (12 свечей по 4 часа)
    end_ts = start_ts + (48 * 60 * 60 * 1000)

    print(f"\n{'='*70}")
    print(f"🔍 {symbol}")
    print(f"  Entry: ${entry:.6f}")
    print(f"  TP:    ${tp:.6f} (+{((tp/entry - 1) * 100):.2f}%)")
    print(f"  SL:    ${sl:.6f} ({((sl/entry - 1) * 100):.2f}%)")
    print(f"  Закрыта MANUAL: {closed_time}")

    # Получаем свечи
    klines = get_binance_klines(symbol, start_ts, end_ts, interval='4h')

    if not klines:
        return None

    tp_hit = False
    sl_hit = False
    hit_time = None
    hit_price = None
    hit_type = None

    for kline in klines:
        # [0] = Open time, [1] = Open, [2] = High, [3] = Low, [4] = Close
        candle_time = datetime.fromtimestamp(int(kline[0]) / 1000)
        high = float(kline[2])
        low = float(kline[3])

        # Проверяем TP (LONG позиция)
        if high >= tp and not tp_hit:
            tp_hit = True
            hit_time = candle_time
            hit_price = tp
            hit_type = 'TP'
            break

        # Проверяем SL
        if low <= sl and not sl_hit:
            sl_hit = True
            hit_time = candle_time
            hit_price = sl
            hit_type = 'SL'
            break

    if hit_type:
        pnl_pct = ((hit_price / entry) - 1) * 100
        print(f"  ✅ Результат: {hit_type} достигнут!")
        print(f"  ⏰ Время: {hit_time}")
        print(f"  💰 Цена: ${hit_price:.6f} ({pnl_pct:+.2f}%)")
        return {
            'type': hit_type,
            'price': hit_price,
            'time': hit_time,
            'pnl_pct': pnl_pct
        }
    else:
        # Проверим последнюю цену
        last_close = float(klines[-1][4]) if klines else None
        if last_close:
            pnl_pct = ((last_close / entry) - 1) * 100
            print(f"  ⏳ TP/SL не достигнут за 48ч")
            print(f"  📊 Последняя цена: ${last_close:.6f} ({pnl_pct:+.2f}%)")
        return None

# Анализ всех позиций
print("\n" + "="*70)
print("📊 АНАЛИЗ: Что было бы, если не закрывать позиции MANUAL?")
print("="*70)

results = {
    'tp_count': 0,
    'sl_count': 0,
    'no_hit': 0,
    'total_pnl': 0,
    'actual_pnl': 0
}

for pos in positions:
    result = check_if_tp_or_sl_hit(
        pos['symbol'],
        pos['entry'],
        pos['tp'],
        pos['sl'],
        pos['closed_time']
    )

    if result:
        if result['type'] == 'TP':
            results['tp_count'] += 1
            results['total_pnl'] += result['pnl_pct']
        else:
            results['sl_count'] += 1
            results['total_pnl'] += result['pnl_pct']
    else:
        results['no_hit'] += 1

    results['actual_pnl'] += pos['pnl']

# Итоговая статистика
print("\n" + "="*70)
print("📈 ИТОГОВАЯ СТАТИСТИКА")
print("="*70)
print(f"✅ TP достигнут: {results['tp_count']}/{len(positions)}")
print(f"❌ SL достигнут: {results['sl_count']}/{len(positions)}")
print(f"⏳ Не достигнут: {results['no_hit']}/{len(positions)}")
print(f"\n💰 Фактический PnL (MANUAL): {results['actual_pnl']:.2f}%")
print(f"💰 Потенциальный PnL (TP/SL): {results['total_pnl']:.2f}%")
print(f"📊 Разница: {results['total_pnl'] - results['actual_pnl']:.2f}%")
print("="*70)
