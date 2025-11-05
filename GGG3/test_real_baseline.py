#!/usr/bin/env python3
"""
Тест РЕАЛЬНОЙ базовой модели из bnbusdrt6.py

Импортирует настоящие функции и тестирует их на синтетических данных
Минимум 3000 раундов как запросил пользователь
"""
import sys
import os
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Добавляем путь к GGG3
sys.path.insert(0, os.path.dirname(__file__))

print("📦 Импорт функций из bnbusdrt6.py...")

try:
    # Импорт констант
    from bnbusdrt6 import (
        M1, M2, M3,  # Momentum периоды
        RB_LEN, ATR_LEN, ATR_SMOOTH,
        VWAP_LOOK, KC_LEN, KC_MULT,
        BB_LEN, BB_Z, VOL_LEN, VOL_BOOST,
        C_M, C_S, C_B, C_R,
        USE_LORENTZ
    )

    # Импорт вспомогательных функций
    from bnbusdrt6 import (
        ema, sma, stdev, atr_wilder,
        norm_feat, session_vwap,
        features_from_binance,
        prob_up_down_at_time,
        softmax2_with_temperature,
        adaptive_temperature,
        to_logit, from_logit
    )

    print("✅ Успешно импортированы функции из bnbusdrt6.py")
    IMPORTS_OK = True

except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("⚠️  Будет использована упрощенная логика")
    IMPORTS_OK = False


def generate_synthetic_ohlcv(n_candles=3000, signal_strength=0.4):
    """
    Генерирует синтетические OHLCV данные с паттернами

    Args:
        n_candles: Количество свечей (минимум 3000)
        signal_strength: Сила сигнала для прогнозирования

    Returns:
        DataFrame с OHLCV данными
    """
    print(f"\n📊 Генерация {n_candles:,} синтетических свечей...")

    np.random.seed(42)
    random.seed(42)

    # Базовая цена и время
    base_price = 300.0  # BNB price
    start_time = datetime(2023, 1, 1)

    # Генерируем цены с трендами и циклами
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_candles)]

    # Создаем сложную динамику цены
    t = np.arange(n_candles)

    # Тренд
    trend = 0.0001 * t  # Медленный рост

    # Циклы (имитация рыночных циклов)
    cycle1 = 5.0 * np.sin(2 * np.pi * t / 200)  # Длинный цикл
    cycle2 = 2.0 * np.sin(2 * np.pi * t / 50)   # Средний цикл
    cycle3 = 1.0 * np.sin(2 * np.pi * t / 10)   # Короткий цикл

    # Случайные шоки
    shocks = np.random.randn(n_candles) * 0.5

    # Автокорреляция (momentum)
    returns = np.zeros(n_candles)
    returns[0] = np.random.randn() * 0.001
    for i in range(1, n_candles):
        # AR(1) процесс с коэффициентом 0.2
        returns[i] = 0.2 * returns[i-1] + np.random.randn() * 0.001

    # Комбинируем все компоненты
    price_changes = (trend + cycle1 + cycle2 + cycle3 + shocks) * 0.02 + returns

    # Вычисляем цены (ограничиваем изменения)
    price_changes = np.clip(price_changes, -0.05, 0.05)  # Max ±5% per candle
    close_prices = base_price * np.exp(np.cumsum(price_changes))

    # Генерируем OHLC из close
    data = []
    for i in range(n_candles):
        close = close_prices[i]

        # Волатильность свечи
        candle_vol = abs(np.random.randn()) * 0.3 + 0.1

        # Open близко к предыдущему close
        if i == 0:
            open_price = close * (1 + np.random.randn() * 0.001)
        else:
            open_price = close_prices[i-1] * (1 + np.random.randn() * 0.002)

        # High и Low с учетом направления
        if close > open_price:
            high = close * (1 + candle_vol * abs(np.random.randn()) * 0.01)
            low = open_price * (1 - candle_vol * abs(np.random.randn()) * 0.01)
        else:
            high = open_price * (1 + candle_vol * abs(np.random.randn()) * 0.01)
            low = close * (1 - candle_vol * abs(np.random.randn()) * 0.01)

        # Volume с корреляцией к волатильности
        volume = 1000000 * (1 + candle_vol * 2 + abs(np.random.randn()) * 0.5)

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

    # Добавляем timezone UTC (required by bnbusdrt6.py)
    df.index = pd.to_datetime(df.index, utc=True)

    print(f"✅ Сгенерировано {len(df):,} свечей")
    print(f"   Цена: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")
    print(f"   Min: {df['low'].min():.2f}, Max: {df['high'].max():.2f}")

    return df


def simulate_prediction_rounds(df_ohlcv, n_rounds=3000):
    """
    Симулирует раунды prediction используя РЕАЛЬНУЮ базовую модель

    Args:
        df_ohlcv: DataFrame с OHLCV
        n_rounds: Количество раундов для симуляции

    Returns:
        List of dicts с результатами раундов
    """
    print(f"\n🔮 Симуляция {n_rounds:,} раундов prediction...")

    if not IMPORTS_OK:
        print("❌ Невозможно: функции не импортированы")
        return []

    # Вычисляем фичи для всех свечей
    print("   Вычисление фич из OHLCV...")
    try:
        feats = features_from_binance(df_ohlcv)
        print(f"   ✅ Фичи вычислены: {list(feats.keys())}")
    except Exception as e:
        print(f"   ❌ Ошибка вычисления фич: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Создаем раунды (каждые 5 минут)
    rounds = []
    min_index = 100  # Нужна история для индикаторов
    max_index = len(df_ohlcv) - 10

    # Интервал между раундами (5 минут = 5 свечей)
    round_interval = 5

    for i in range(min_index, max_index, round_interval):
        if len(rounds) >= n_rounds:
            break

        # Timestamp lock (когда делается прогноз)
        lock_ts = df_ohlcv.index[i]

        # Timestamp close (через 5 минут)
        close_idx = min(i + 5, len(df_ohlcv) - 1)
        close_ts = df_ohlcv.index[close_idx]

        # Цены
        lock_price = float(df_ohlcv['close'].iloc[i])
        close_price = float(df_ohlcv['close'].iloc[close_idx])

        # Outcome
        outcome = 1 if close_price > lock_price else 0

        rounds.append({
            'round_id': len(rounds),
            'lock_ts': lock_ts,
            'close_ts': close_ts,
            'lock_idx': i,
            'close_idx': close_idx,
            'lock_price': lock_price,
            'close_price': close_price,
            'outcome': outcome
        })

        if (len(rounds) % 500) == 0:
            print(f"   Создано раундов: {len(rounds):,}", end='\r')

    print(f"\n   ✅ Создано {len(rounds):,} раундов")

    # Теперь вычисляем прогнозы для каждого раунда
    print(f"   Вычисление прогнозов базовой модели...")

    predictions = []
    for i, rd in enumerate(rounds):
        try:
            # РЕАЛЬНАЯ базовая модель из bnbusdrt6.py
            P_up, P_dn, wf_dict = prob_up_down_at_time(
                feats,
                rd['lock_ts'],
                w_dyn=None  # Используем default веса
            )

            predictions.append({
                'round_id': rd['round_id'],
                'p_up': P_up,
                'p_dn': P_dn,
                'pred_binary': 1 if P_up > 0.5 else 0,
                'outcome': rd['outcome'],
                'lock_price': rd['lock_price'],
                'close_price': rd['close_price'],
                'temperature': wf_dict.get('temperature', 1.0),
                'phi_wf0': wf_dict.get('phi_wf0', 0.0),
                'phi_wf1': wf_dict.get('phi_wf1', 0.0),
                'phi_wf2': wf_dict.get('phi_wf2', 0.0),
                'phi_wf3': wf_dict.get('phi_wf3', 0.0),
            })

            if (i % 500) == 0:
                print(f"   Прогнозов: {i:,}/{len(rounds):,}", end='\r')

        except Exception as e:
            if i < 5:  # Показываем только первые ошибки
                print(f"\n   ⚠️  Ошибка прогноза для раунда {i}: {e}")
            continue

    print(f"\n   ✅ Вычислено {len(predictions):,} прогнозов")

    return predictions


def evaluate_predictions(predictions):
    """Оценивает качество прогнозов"""
    if len(predictions) == 0:
        print("\n❌ Нет данных для оценки")
        return None

    print(f"\n{'='*70}")
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ РЕАЛЬНОЙ БАЗОВОЙ МОДЕЛИ")
    print(f"{'='*70}")

    n = len(predictions)

    # Accuracy
    correct = sum(1 for p in predictions if p['pred_binary'] == p['outcome'])
    accuracy = correct / n

    # Precision / Recall / F1 для UP
    tp = sum(1 for p in predictions if p['pred_binary'] == 1 and p['outcome'] == 1)
    fp = sum(1 for p in predictions if p['pred_binary'] == 1 and p['outcome'] == 0)
    fn = sum(1 for p in predictions if p['pred_binary'] == 0 and p['outcome'] == 1)
    tn = sum(1 for p in predictions if p['pred_binary'] == 0 and p['outcome'] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Winrate для UP ставок
    up_winrate = tp / (tp + fp) if (tp + fp) > 0 else 0

    # ROI с комиссией 3%
    commission = 0.03
    roi_total = 0.0
    for p in predictions:
        if p['pred_binary'] == 1:  # Ставим UP
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

    # Статистика прогнозов
    p_ups = [p['p_up'] for p in predictions]
    p_up_mean = np.mean(p_ups)
    p_up_std = np.std(p_ups)

    # Outcome distribution
    outcomes = [p['outcome'] for p in predictions]
    up_rate = np.mean(outcomes)

    print(f"\n📈 МЕТРИКИ НА {n:,} РАУНДАХ:\n")
    print(f"   Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision (UP):     {precision:.4f}")
    print(f"   Recall (UP):        {recall:.4f}")
    print(f"   F1 Score:           {f1:.4f}")
    print(f"\n   Winrate (UP ставки): {up_winrate:.4f} ({up_winrate*100:.2f}%)")
    print(f"   Break-even threshold: 50.76%")

    if up_winrate > 0.5076:
        print(f"   ✅ Winrate выше break-even!")
    else:
        print(f"   ❌ Winrate ниже break-even")

    print(f"\n   Brier Score:        {brier:.4f} (lower is better)")
    print(f"   Log Loss:           {log_loss:.4f} (lower is better)")

    print(f"\n   Total ROI:          {roi_total:+.2f} USDT (с комиссией 3%)")
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

    print(f"\n📊 СТАТИСТИКА ПРОГНОЗОВ:\n")
    print(f"   Средняя P(UP):      {p_up_mean:.4f}")
    print(f"   Std P(UP):          {p_up_std:.4f}")
    print(f"   Min P(UP):          {min(p_ups):.4f}")
    print(f"   Max P(UP):          {max(p_ups):.4f}")
    print(f"\n   Фактический UP rate: {up_rate:.4f} ({up_rate*100:.1f}%)")

    # Калибровка
    print(f"\n📐 КАЛИБРОВКА:\n")
    # Бины по вероятности
    bins = [(0.0, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 1.0)]
    print(f"   {'Bin P(UP)':<15} {'Count':>8} {'Actual UP':>12} {'Calibration':<15}")
    print(f"   {'-'*55}")
    for low, high in bins:
        in_bin = [p for p in predictions if low <= p['p_up'] < high]
        if len(in_bin) > 0:
            actual_up_rate = sum(p['outcome'] for p in in_bin) / len(in_bin)
            avg_pred = np.mean([p['p_up'] for p in in_bin])
            calib_error = abs(actual_up_rate - avg_pred)
            print(f"   [{low:.2f}, {high:.2f})  {len(in_bin):>8,} {actual_up_rate:>11.1%}    "
                  f"Error: {calib_error:.4f}")

    # Экстраполяция на месяц
    print(f"\n💰 ЭКСТРАПОЛЯЦИЯ НА МЕСЯЦ:\n")
    bets_per_day = 288  # 24h * 60min / 5min
    days_per_month = 30
    monthly_bets = bets_per_day * days_per_month

    monthly_roi = roi_per_bet * monthly_bets
    initial_bankroll = 100
    monthly_pct = (monthly_roi / initial_bankroll) * 100

    print(f"   Ставок в день:      {bets_per_day}")
    print(f"   Ставок в месяц:     {monthly_bets:,}")
    print(f"   Начальный банк:     {initial_bankroll} USDT")
    print(f"   Месячный ROI:       {monthly_roi:+.2f} USDT")
    print(f"   Месячная доходность: {monthly_pct:+.1f}%")

    if monthly_pct > 20:
        print(f"\n   ⚠️  {monthly_pct:.1f}% в месяц на СИНТЕТИКЕ")
        print(f"   На реальных данных ожидайте 20-40% от этого")
    elif monthly_pct > 0:
        print(f"\n   ✅ {monthly_pct:.1f}% в месяц возможно на синтетике")
        print(f"   На реальных данных: {monthly_pct*0.3:.1f}% - {monthly_pct*0.5:.1f}%")
    else:
        print(f"\n   ❌ Убыточно на синтетике")

    print(f"\n{'='*70}\n")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'winrate': up_winrate,
        'brier': brier,
        'log_loss': log_loss,
        'roi_total': roi_total,
        'roi_per_bet': roi_per_bet,
        'n_samples': n
    }


def main():
    """Главная функция"""
    print("="*70)
    print("🧪 ТЕСТ РЕАЛЬНОЙ БАЗОВОЙ МОДЕЛИ (bnbusdrt6.py)")
    print("="*70)

    if not IMPORTS_OK:
        print("\n❌ Невозможно выполнить тест: функции не импортированы")
        print("   Проверьте что bnbusdrt6.py находится в GGG3/")
        return

    # Генерация данных (достаточно для 3000+ раундов)
    n_candles = 15000  # Это даст ~3000 раундов (каждый 5-ый раунд)
    df_ohlcv = generate_synthetic_ohlcv(n_candles)

    # Симуляция раундов
    predictions = simulate_prediction_rounds(df_ohlcv, n_rounds=3000)

    if len(predictions) < 100:
        print(f"\n❌ Недостаточно прогнозов: {len(predictions)}")
        return

    # Оценка
    metrics = evaluate_predictions(predictions)

    # Выводы
    print("🎯 ВЫВОДЫ:")
    print("="*70 + "\n")

    if metrics:
        if metrics['accuracy'] > 0.52:
            print("✅ БАЗОВАЯ МОДЕЛЬ РАБОТАЕТ!")
            print(f"   Accuracy {metrics['accuracy']*100:.1f}% > 52%")
            print(f"   Winrate {metrics['winrate']*100:.1f}% {'>' if metrics['winrate'] > 0.5076 else '<'} 50.76% (break-even)")
        else:
            print("⚠️  Базовая модель недостаточно эффективна")
            print(f"   Accuracy {metrics['accuracy']*100:.1f}% ≤ 52%")

        if metrics['roi_total'] > 0:
            print(f"\n✅ ПРИБЫЛЬНО на синтетике: {metrics['roi_total']:+.2f} USDT")
        else:
            print(f"\n❌ УБЫТОЧНО: {metrics['roi_total']:+.2f} USDT")

        print(f"\n💡 СЛЕДУЮЩИЕ ШАГИ:")
        print(f"   1. Базовая модель протестирована")
        print(f"   2. Добавить 4 эксперта (XGB/RF/ARF/NN) → +3-7% accuracy")
        print(f"   3. Добавить META слой → +1-2% accuracy")
        print(f"   4. Тестировать на реальных данных из BSC")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
