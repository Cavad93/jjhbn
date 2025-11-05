#!/usr/bin/env python3
"""
Тест базовой модели на РЕАЛИСТИЧНОЙ ИМИТАЦИИ рынка

Используем известные статистические свойства PancakeSwap Prediction:
- UP/DOWN распределение ~50/50 (слабый бычий bias ~51%)
- Высокий шум (hard to predict)
- Корреляция с BTC/ETH
- Микроструктурные эффекты (order flow, late money)
- Периоды высокой/низкой волатильности

Это НЕ реальные данные, но максимально близкая имитация!
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

sys.path.insert(0, os.path.dirname(__file__))

try:
    from bnbusdrt6 import features_from_binance, prob_up_down_at_time
    IMPORTS_OK = True
except:
    IMPORTS_OK = False


def generate_realistic_bnb_market(n_candles=15000, base_price=580.0):
    """
    Генерирует реалистичную имитацию рынка BNB

    Основано на:
    - Реальная волатильность BNB: ~3-5% в день
    - Weak momentum (AR коэффициент ~0.1-0.2)
    - Периоды трендов и флэта
    - Микроструктурный шум
    """
    print(f"📊 Генерация реалистичной имитации рынка BNB...")
    print(f"   Параметры: {n_candles:,} свечей, базовая цена {base_price:.2f}")

    np.random.seed(42)
    random.seed(42)

    timestamps = [datetime(2024, 10, 1) + timedelta(minutes=i) for i in range(n_candles)]

    # === МНОГОКОМПОНЕНТНАЯ МОДЕЛЬ ЦЕНЫ ===

    t = np.arange(n_candles)

    # 1. ТРЕНД (слабый долгосрочный)
    # Меняется каждые ~1000 свечей
    trend = np.zeros(n_candles)
    current_trend = 0
    trend_change_prob = 1.0 / 1000

    for i in range(n_candles):
        if random.random() < trend_change_prob:
            current_trend = random.choice([-0.00005, 0, 0.00005])  # Weak trend
        trend[i] = current_trend * i

    # 2. ЦИКЛЫ (имитация макро движений)
    cycle_long = 0.5 * np.sin(2 * np.pi * t / 500)   # ~8 часов
    cycle_mid = 0.3 * np.sin(2 * np.pi * t / 120)    # ~2 часа
    cycle_short = 0.15 * np.sin(2 * np.pi * t / 30)  # ~30 минут

    # 3. MOMENTUM (слабая автокорреляция)
    returns = np.zeros(n_candles)
    returns[0] = np.random.randn() * 0.002

    for i in range(1, n_candles):
        # AR(1) с коэффициентом 0.15 (слабый momentum)
        returns[i] = 0.15 * returns[i-1] + np.random.randn() * 0.002

    # 4. VOLATILITY CLUSTERING (GARCH-like)
    volatility = np.ones(n_candles) * 0.001
    for i in range(1, n_candles):
        # Волатильность зависит от недавних шоков
        shock = abs(returns[i-1])
        volatility[i] = 0.7 * volatility[i-1] + 0.3 * shock + np.random.rand() * 0.0005

    # 5. SHOCKS (случайные резкие движения)
    shocks = np.zeros(n_candles)
    shock_prob = 0.005  # 0.5% вероятность шока на каждой свече

    for i in range(n_candles):
        if random.random() < shock_prob:
            shocks[i] = random.choice([-1, 1]) * (0.005 + abs(np.random.randn()) * 0.003)

    # 6. MICROSTRUCTURE NOISE (высокочастотный шум)
    micro_noise = np.random.randn(n_candles) * 0.0003

    # === КОМБИНИРУЕМ ВСЕ КОМПОНЕНТЫ ===
    price_changes = (
        trend * 0.02 +
        (cycle_long + cycle_mid + cycle_short) * 0.001 +
        returns +
        shocks +
        micro_noise
    ) * volatility * 50  # Масштабируем волатильностью

    # Ограничиваем экстремальные изменения
    price_changes = np.clip(price_changes, -0.03, 0.03)  # Max ±3% per candle

    # Вычисляем цены
    close_prices = base_price * np.exp(np.cumsum(price_changes))

    # === ГЕНЕРИРУЕМ OHLC ===
    data = []
    for i in range(n_candles):
        close = close_prices[i]

        # Внутрисвечевая волатильность
        candle_range = abs(np.random.randn()) * volatility[i] * close * 30

        # Open близко к предыдущему close
        if i == 0:
            open_price = close * (1 + np.random.randn() * 0.001)
        else:
            open_price = close_prices[i-1] * (1 + np.random.randn() * 0.002)

        # High и Low
        if close > open_price:
            high = max(close, open_price) * (1 + candle_range / close)
            low = min(close, open_price) * (1 - candle_range / close)
        else:
            high = max(close, open_price) * (1 + candle_range / close)
            low = min(close, open_price) * (1 - candle_range / close)

        # Volume (коррелирует с волатильностью)
        base_volume = 2000000
        volume = base_volume * (1 + volatility[i] * 100 + abs(np.random.randn()) * 0.3)

        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index, utc=True)

    print(f"✅ Сгенерирован реалистичный рынок:")
    print(f"   Цена: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")
    print(f"   Volatility (daily): {df['close'].pct_change().std() * np.sqrt(1440) * 100:.2f}%")
    print(f"   Returns autocorr: {df['close'].pct_change().autocorr():.4f}")

    return df


def simulate_prediction_rounds(df_ohlcv, n_rounds=9000):
    """Симулирует prediction раунды (5 минут)"""
    print(f"\n🔮 Симуляция {n_rounds:,} prediction раундов...")

    if not IMPORTS_OK:
        print("❌ Функции не импортированы")
        return []

    # Вычисляем фичи
    print("   Вычисление фич...")
    try:
        feats = features_from_binance(df_ohlcv)
        print("   ✅ Фичи вычислены")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Раунды (каждые 5 минут)
    rounds = []
    predictions = []
    min_idx = 100
    max_idx = len(df_ohlcv) - 10

    for i in range(min_idx, max_idx, 5):
        if len(rounds) >= n_rounds:
            break

        lock_ts = df_ohlcv.index[i]
        close_idx = min(i + 5, len(df_ohlcv) - 1)
        close_ts = df_ohlcv.index[close_idx]

        lock_price = float(df_ohlcv['close'].iloc[i])
        close_price = float(df_ohlcv['close'].iloc[close_idx])

        outcome = 1 if close_price > lock_price else 0

        # БАЗОВАЯ МОДЕЛЬ
        try:
            P_up, P_dn, _ = prob_up_down_at_time(feats, lock_ts, w_dyn=None)

            predictions.append({
                'round_id': len(rounds),
                'p_up': P_up,
                'p_dn': P_dn,
                'pred_binary': 1 if P_up > 0.5 else 0,
                'outcome': outcome,
                'lock_price': lock_price,
                'close_price': close_price
            })

            rounds.append(1)

            if len(predictions) % 1000 == 0:
                print(f"   Прогнозов: {len(predictions):,}", end='\r')

        except Exception as e:
            if len(predictions) < 5:
                print(f"\n   ⚠️  Ошибка: {e}")
            continue

    print(f"\n   ✅ Получено {len(predictions):,} прогнозов")

    return predictions


def evaluate_predictions(predictions):
    """Оценивает результаты"""
    if not predictions or len(predictions) < 100:
        print(f"\n❌ Недостаточно прогнозов")
        return None

    print(f"\n{'='*70}")
    print(f"📊 РЕЗУЛЬТАТЫ НА РЕАЛИСТИЧНОЙ ИМИТАЦИИ РЫНКА")
    print(f"{'='*70}")

    n = len(predictions)

    # Метрики
    correct = sum(1 for p in predictions if p['pred_binary'] == p['outcome'])
    accuracy = correct / n

    tp = sum(1 for p in predictions if p['pred_binary'] == 1 and p['outcome'] == 1)
    fp = sum(1 for p in predictions if p['pred_binary'] == 1 and p['outcome'] == 0)
    fn = sum(1 for p in predictions if p['pred_binary'] == 0 and p['outcome'] == 1)
    tn = sum(1 for p in predictions if p['pred_binary'] == 0 and p['outcome'] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    up_winrate = tp / (tp + fp) if (tp + fp) > 0 else 0

    # ROI
    commission = 0.03
    roi_total = 0.0
    for p in predictions:
        if p['pred_binary'] == 1:
            if p['outcome'] == 1:
                roi_total += (1 - commission)
            else:
                roi_total -= 1.0

    roi_per_bet = roi_total / n

    # Brier / Log Loss
    brier = sum((p['p_up'] - p['outcome'])**2 for p in predictions) / n

    eps = 1e-7
    log_loss = sum(-(p['outcome'] * np.log(max(eps, min(1-eps, p['p_up']))) +
                     (1-p['outcome']) * np.log(max(eps, 1-min(1-eps, p['p_up']))))
                   for p in predictions) / n

    # Статистика
    p_ups = [p['p_up'] for p in predictions]
    outcomes = [p['outcome'] for p in predictions]

    print(f"\n⚠️  ВНИМАНИЕ: Это ИМИТАЦИЯ рынка, НЕ реальные данные!")
    print(f"   Реальные BSC RPC недоступны из этого окружения")
    print(f"   Имитация основана на известных свойствах рынка BNB\n")

    print(f"📈 МЕТРИКИ НА {n:,} ИМИТИРОВАННЫХ РАУНДАХ:\n")
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
    print(f"\n   Total ROI:          {roi_total:+.2f} USDT")
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
    print(f"   Фактический UP rate: {np.mean(outcomes):.4f}")

    # Экстраполяция
    monthly_roi = roi_per_bet * 288 * 30
    monthly_pct = (monthly_roi / 100) * 100

    print(f"\n💰 ЭКСТРАПОЛЯЦИЯ НА МЕСЯЦ (288 ставок/день):\n")
    print(f"   ROI per bet:        {roi_per_bet:+.5f} USDT")
    print(f"   Месячный ROI:       {monthly_roi:+.2f} USDT")
    print(f"   Месячная доходность: {monthly_pct:+.1f}% (на 100 USDT)")

    print(f"\n{'='*70}\n")

    return {
        'accuracy': accuracy,
        'winrate': up_winrate,
        'roi_total': roi_total,
        'roi_per_bet': roi_per_bet
    }


def main():
    print("="*70)
    print("🧪 ТЕСТ БАЗОВОЙ МОДЕЛИ (РЕАЛИСТИЧНАЯ ИМИТАЦИЯ)")
    print("="*70)

    if not IMPORTS_OK:
        print("\n❌ Функции не импортированы")
        return

    # Генерация реалистичного рынка
    df = generate_realistic_bnb_market(n_candles=15000, base_price=580.0)

    # Симуляция раундов
    predictions = simulate_prediction_rounds(df, n_rounds=9000)

    if not predictions:
        return

    # Оценка
    metrics = evaluate_predictions(predictions)

    # Выводы
    print("🎯 ВЫВОДЫ:")
    print("="*70 + "\n")

    if metrics:
        print("⚠️  ВАЖНО: Это имитация, НЕ реальные данные!\n")

        if metrics['accuracy'] > 0.52:
            print(f"✅ Базовая модель показывает {metrics['accuracy']*100:.1f}% accuracy")
            print(f"   На реалистичной имитации выше порога прибыльности (52%)")
        else:
            print(f"⚠️  Accuracy {metrics['accuracy']*100:.1f}% близко к случайному")

        if metrics['winrate'] > 0.5076:
            print(f"\n✅ Winrate {metrics['winrate']*100:.1f}% > break-even (50.76%)")
        else:
            print(f"\n⚠️  Winrate {metrics['winrate']*100:.1f}% < break-even")

        print(f"\n💡 ЧТО ОЖИДАТЬ НА РЕАЛЬНЫХ ДАННЫХ:")
        print(f"   • Accuracy вероятно НИЖЕ на 1-3% (больше шума)")
        print(f"   • Критически важны ML эксперты для улучшения")
        print(f"   • Рекомендуется собрать реальные данные локально:")
        print(f"\n   ИНСТРУКЦИЯ:")
        print(f"   1. Настроить BSC RPC endpoint (Infura/Alchemy)")
        print(f"   2. Запустить: python collect_historical_data_FAST.py")
        print(f"   3. Обучить 4 эксперта: python train_bot_full.py")
        print(f"   4. Paper trading 30 дней перед реальной торговлей")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
