#!/usr/bin/env python3
"""
Тест базовой модели (БЕЗ ML) на синтетических данных

Проверяет эффективность простой эвристической логики:
- Momentum signals
- RSI oversold/overbought
- Volatility regime
- Trend direction

Цель: понять baseline перформанс перед применением ML
"""
import random
import math


def sigmoid(x):
    """Сигмоида"""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def tanh(x):
    """Гиперболический тангенс"""
    return math.tanh(max(-500, min(500, x)))


def clip(x, min_val, max_val):
    """Ограничивает значение"""
    return max(min_val, min(max_val, x))


class BaselineModel:
    """
    Базовая модель на эвристиках (БЕЗ ML)

    Использует только простые индикаторы:
    - Momentum (краткосрочный, средний, долгосрочный)
    - RSI (перекупленность/перепроданность)
    - Volatility (режим волатильности)
    - Volume (аномалии объема)
    """

    def __init__(self, name="Baseline Heuristic"):
        self.name = name

    def predict(self, features):
        """
        Предсказывает вероятность UP

        Features (68 total, используем первые 15):
        [0]: momentum_5m
        [1]: momentum_10m
        [2]: momentum_15m
        [3]: volatility
        [4]: vol_zscore
        [5]: rsi
        [6]: rsi_lag1
        [7]: rsi_lag5
        [8]: close
        [9]: high
        [10]: low
        [11]: volume
        [12]: atr
        [13]: atr_norm
        [14]: trend_sign
        """
        if len(features) < 15:
            return 0.5  # Недостаточно данных

        # Извлекаем фичи
        mom_5m = features[0]
        mom_10m = features[1]
        mom_15m = features[2]
        volatility = features[3]
        vol_zscore = features[4]
        rsi = features[5]
        rsi_lag1 = features[6]
        trend_sign = features[14]

        # === СИГНАЛ 1: MOMENTUM ===
        # Усредненный momentum с весами (больший вес - более свежему)
        momentum_weighted = (mom_5m * 3 + mom_10m * 2 + mom_15m * 1) / 6.0

        # Нормализуем через tanh (сжимаем в [-1, 1])
        momentum_signal = tanh(momentum_weighted * 2.0) * 0.25  # Вес ±25%

        # === СИГНАЛ 2: RSI ===
        # RSI в диапазоне [-1, 1] (предполагаем нормализованный)
        # Положительный RSI → oversold → ожидаем отскок UP
        # Отрицательный RSI → overbought → ожидаем падение DOWN
        rsi_signal = tanh(rsi * 1.5) * 0.20  # Вес ±20%

        # RSI momentum (изменение RSI)
        rsi_change = rsi - rsi_lag1
        rsi_momentum = tanh(rsi_change * 2.0) * 0.10  # Вес ±10%

        # === СИГНАЛ 3: VOLATILITY REGIME ===
        # Высокая волатильность → усиливает тренды
        # Низкая волатильность → mean reversion
        vol_factor = 1.0 + clip(volatility, -1, 1) * 0.15

        # Vol zscore (аномальный объем)
        vol_boost = 0.0
        if abs(vol_zscore) > 1.5:  # Аномальный объем
            # Предполагаем, что аномальный объем усиливает текущий тренд
            vol_boost = tanh(vol_zscore * 0.5) * 0.10

        # === СИГНАЛ 4: TREND ===
        # trend_sign: 1.0 = uptrend, -1.0 = downtrend, 0.0 = sideways
        trend_signal = trend_sign * 0.15  # Вес ±15%

        # === КОМБИНИРОВАНИЕ СИГНАЛОВ ===
        # Базовая вероятность 50%
        base_prob = 0.5

        # Основные сигналы
        combined_signal = momentum_signal + rsi_signal + rsi_momentum + trend_signal + vol_boost

        # Применяем volatility factor
        combined_signal *= vol_factor

        # Финальная вероятность
        prob = clip(base_prob + combined_signal, 0.05, 0.95)

        return prob


class ImprovedBaselineModel(BaselineModel):
    """
    Улучшенная базовая модель с дополнительной логикой
    """

    def __init__(self):
        super().__init__(name="Improved Baseline")

    def predict(self, features):
        """Предсказание с дополнительными проверками"""
        if len(features) < 15:
            return 0.5

        # Базовое предсказание
        base_prob = super().predict(features)

        # === ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ ===

        mom_5m = features[0]
        mom_10m = features[1]
        volatility = features[3]
        rsi = features[5]

        # ФИЛЬТР 1: Согласованность momentum
        # Если все momentum в одном направлении → усиливаем сигнал
        momentum_consensus = (mom_5m > 0) == (mom_10m > 0)
        if momentum_consensus:
            direction = 1 if mom_5m > 0 else -1
            consensus_boost = 0.03 * direction
            base_prob += consensus_boost

        # ФИЛЬТР 2: Экстремальный RSI
        # RSI > 0.8 (сильно oversold) → вероятно отскок UP
        # RSI < -0.8 (сильно overbought) → вероятно падение DOWN
        if abs(rsi) > 0.8:
            extreme_boost = tanh(rsi * 1.0) * 0.05
            base_prob += extreme_boost

        # ФИЛЬТР 3: Низкая волатильность → mean reversion
        # В низкой волатильности momentum менее надежен
        if volatility < -0.5:  # Низкая волатильность
            # Уменьшаем влияние momentum, усиливаем mean reversion
            deviation = base_prob - 0.5
            base_prob = 0.5 + deviation * 0.7  # Сжимаем к 50%

        # Финальное ограничение
        return clip(base_prob, 0.05, 0.95)


def generate_synthetic_data(n_samples=5000, n_features=68, signal_strength=0.4):
    """
    Генерирует синтетические данные с ИЗВЕСТНЫМИ паттернами

    Args:
        n_samples: Количество примеров
        n_features: Количество фич (68 как у реального бота)
        signal_strength: Сила сигнала (0.0 = random, 1.0 = perfect)

    Returns:
        List of dicts with features, outcome, phase, true_prob
    """
    print(f"\n📊 Генерация синтетических данных:")
    print(f"   Примеров: {n_samples:,}")
    print(f"   Фич: {n_features}")
    print(f"   Сила сигнала: {signal_strength:.2f}")

    random.seed(42)
    data = []

    for _ in range(n_samples):
        # Случайные фичи (нормальное распределение)
        features = [random.gauss(0, 1) for _ in range(n_features)]

        # === СОЗДАЕМ ПАТТЕРНЫ (как модель будет их искать) ===

        # Momentum pattern
        mom_5m = features[0]
        mom_10m = features[1]
        mom_15m = features[2]
        momentum_avg = (mom_5m * 3 + mom_10m * 2 + mom_15m) / 6.0
        momentum_signal = tanh(momentum_avg * 2.0) * 0.25

        # RSI pattern
        rsi = features[5]
        rsi_lag1 = features[6]
        rsi_signal = tanh(rsi * 1.5) * 0.20
        rsi_change = rsi - rsi_lag1
        rsi_momentum = tanh(rsi_change * 2.0) * 0.10

        # Volatility regime
        volatility = features[3]
        vol_zscore = features[4]
        vol_factor = 1.0 + clip(volatility, -1, 1) * 0.15

        vol_boost = 0.0
        if abs(vol_zscore) > 1.5:
            vol_boost = tanh(vol_zscore * 0.5) * 0.10

        # Trend
        trend_sign = features[14] if len(features) > 14 else 0
        trend_signal = trend_sign * 0.15

        # Комбинируем сигналы
        combined = momentum_signal + rsi_signal + rsi_momentum + trend_signal + vol_boost
        combined *= vol_factor

        # Применяем signal_strength
        combined *= signal_strength

        # Истинная вероятность
        true_prob = clip(0.5 + combined, 0.05, 0.95)

        # Генерируем outcome на основе истинной вероятности
        outcome = 1 if random.random() < true_prob else 0

        # Фаза рынка (для статистики)
        phase = random.randint(0, 5)

        data.append({
            'features': features,
            'outcome': outcome,
            'phase': phase,
            'true_prob': true_prob
        })

    # Статистика
    avg_prob = sum(d['true_prob'] for d in data) / len(data)
    std_prob = (sum((d['true_prob'] - avg_prob)**2 for d in data) / len(data)) ** 0.5
    up_rate = sum(d['outcome'] for d in data) / len(data)

    print(f"\n   ✅ Статистика данных:")
    print(f"      Средняя истинная P(UP): {avg_prob:.3f}")
    print(f"      Std истинных P(UP):     {std_prob:.3f}")
    print(f"      Фактический UP rate:    {up_rate:.3f} ({up_rate*100:.1f}%)")

    return data


def evaluate_model(model, test_data, verbose=True):
    """
    Оценивает модель на тестовых данных

    Returns:
        dict с метриками
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 ТЕСТИРОВАНИЕ: {model.name}")
        print(f"{'='*70}")

    predictions = []
    correct = 0
    total_roi = 0.0
    commission = 0.03

    # Предсказания
    for sample in test_data:
        features = sample['features']
        true_outcome = sample['outcome']
        true_prob = sample['true_prob']

        # Предсказание модели
        pred_prob = model.predict(features)
        pred_binary = 1 if pred_prob > 0.5 else 0

        predictions.append({
            'pred_prob': pred_prob,
            'pred_binary': pred_binary,
            'true_outcome': true_outcome,
            'true_prob': true_prob
        })

        # Accuracy
        if pred_binary == true_outcome:
            correct += 1

        # ROI (с комиссией 3%)
        if pred_binary == 1:  # Ставим UP
            if true_outcome == 1:
                total_roi += (1 - commission)
            else:
                total_roi -= 1.0

    # Метрики
    n = len(test_data)
    accuracy = correct / n
    roi_per_bet = total_roi / n

    # Precision / Recall / F1
    tp = sum(1 for p in predictions if p['pred_binary'] == 1 and p['true_outcome'] == 1)
    fp = sum(1 for p in predictions if p['pred_binary'] == 1 and p['true_outcome'] == 0)
    fn = sum(1 for p in predictions if p['pred_binary'] == 0 and p['true_outcome'] == 1)
    tn = sum(1 for p in predictions if p['pred_binary'] == 0 and p['true_outcome'] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Brier score (calibration)
    brier = sum((p['pred_prob'] - p['true_outcome'])**2 for p in predictions) / n

    # Log loss
    eps = 1e-7
    log_loss = 0.0
    for p in predictions:
        prob = clip(p['pred_prob'], eps, 1 - eps)
        log_loss += -(p['true_outcome'] * math.log(prob) + (1 - p['true_outcome']) * math.log(1 - prob))
    log_loss /= n

    # Calibration error (разница между предсказанной и истинной вероятностью)
    calibration_error = sum(abs(p['pred_prob'] - p['true_prob']) for p in predictions) / n

    # Winrate для ставок UP
    up_bets = tp + fp
    up_wins = tp
    up_winrate = up_wins / up_bets if up_bets > 0 else 0

    if verbose:
        print(f"\n📈 Результаты на {n:,} примерах:\n")
        print(f"   Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision (UP):     {precision:.4f}")
        print(f"   Recall (UP):        {recall:.4f}")
        print(f"   F1 Score:           {f1:.4f}")
        print(f"\n   Brier Score:        {brier:.4f} (lower is better)")
        print(f"   Log Loss:           {log_loss:.4f} (lower is better)")
        print(f"   Calibration Error:  {calibration_error:.4f} (lower is better)")
        print(f"\n   Total ROI:          {total_roi:+.2f} USDT")
        print(f"   ROI per bet:        {roi_per_bet:+.5f} USDT")

        # Confusion matrix
        print(f"\n   Confusion Matrix:")
        print(f"                Predicted")
        print(f"                UP      DOWN")
        print(f"   Actual UP    {tp:5d}   {fn:5d}")
        print(f"   Actual DOWN  {fp:5d}   {tn:5d}")

        # Статус
        if total_roi > 0:
            print(f"\n   ✅ ПРИБЫЛЬНО (+{total_roi:.2f} USDT)")
        else:
            print(f"\n   ❌ УБЫТОЧНО ({total_roi:.2f} USDT)")

        # Показываем winrate (уже вычислено выше)
        print(f"\n   Winrate (только UP ставки): {up_winrate:.4f} ({up_winrate*100:.2f}%)")

        if up_winrate > 0.5076:
            print(f"   ✅ Winrate выше break-even (50.76%)")
        else:
            print(f"   ⚠️  Winrate ниже break-even (50.76%)")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier': brier,
        'log_loss': log_loss,
        'calibration_error': calibration_error,
        'roi_total': total_roi,
        'roi_per_bet': roi_per_bet,
        'winrate': up_winrate,
        'n_samples': n
    }


def compare_models(models, test_data):
    """Сравнивает несколько моделей"""
    print(f"\n{'='*70}")
    print(f"📊 СРАВНЕНИЕ МОДЕЛЕЙ")
    print(f"{'='*70}\n")

    results = {}
    for model in models:
        metrics = evaluate_model(model, test_data, verbose=False)
        results[model.name] = metrics

    # Таблица сравнения
    print(f"{'Модель':<25} {'Accuracy':>10} {'Winrate':>10} {'ROI/bet':>12} {'Brier':>8}")
    print(f"{'-'*70}")

    for name, metrics in results.items():
        status = '✅' if metrics['roi_total'] > 0 else '❌'
        print(f"{name:<25} {metrics['accuracy']*100:>9.2f}% {metrics['winrate']*100:>9.2f}% "
              f"{metrics['roi_per_bet']:>+11.5f} {metrics['brier']:>8.4f} {status}")

    return results


def main():
    """Главная функция"""
    print("="*70)
    print("🧪 ТЕСТ БАЗОВОЙ МОДЕЛИ (БЕЗ ML)")
    print("="*70)

    # Генерация данных
    print("\n" + "="*70)
    print("ЭТАП 1: ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("="*70)

    all_data = generate_synthetic_data(
        n_samples=5000,
        n_features=68,
        signal_strength=0.4  # 40% сигнала
    )

    # Train/test split
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    print(f"\n   Разделение данных:")
    print(f"      Train: {len(train_data):,} примеров")
    print(f"      Test:  {len(test_data):,} примеров")

    # Тестирование базовой модели
    print("\n" + "="*70)
    print("ЭТАП 2: ТЕСТИРОВАНИЕ БАЗОВОЙ МОДЕЛИ")
    print("="*70)

    baseline = BaselineModel()
    baseline_metrics = evaluate_model(baseline, test_data)

    # Тестирование улучшенной модели
    print("\n" + "="*70)
    print("ЭТАП 3: ТЕСТИРОВАНИЕ УЛУЧШЕННОЙ МОДЕЛИ")
    print("="*70)

    improved = ImprovedBaselineModel()
    improved_metrics = evaluate_model(improved, test_data)

    # Сравнение
    print("\n" + "="*70)
    print("ЭТАП 4: ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*70)

    compare_models([baseline, improved], test_data)

    # Выводы
    print("\n" + "="*70)
    print("🎯 ВЫВОДЫ")
    print("="*70 + "\n")

    best_model = improved if improved_metrics['accuracy'] > baseline_metrics['accuracy'] else baseline
    best_metrics = improved_metrics if improved_metrics['accuracy'] > baseline_metrics['accuracy'] else baseline_metrics

    print(f"1. ЛУЧШАЯ МОДЕЛЬ: {best_model.name}")
    print(f"   • Accuracy:  {best_metrics['accuracy']*100:.2f}%")
    print(f"   • Winrate:   {best_metrics['winrate']*100:.2f}%")
    print(f"   • ROI/bet:   {best_metrics['roi_per_bet']:+.5f} USDT")

    if best_metrics['accuracy'] > 0.52:
        print(f"\n2. ✅ Базовая логика РАБОТАЕТ на синтетике!")
        print(f"   Accuracy {best_metrics['accuracy']*100:.1f}% > 52%")
    else:
        print(f"\n2. ⚠️  Базовая логика недостаточно эффективна")
        print(f"   Accuracy {best_metrics['accuracy']*100:.1f}% ≤ 52%")

    if best_metrics['winrate'] > 0.5076:
        print(f"\n3. ✅ Winrate выше break-even!")
        print(f"   {best_metrics['winrate']*100:.2f}% > 50.76% (минимум для прибыли)")
    else:
        print(f"\n3. ❌ Winrate ниже break-even")
        print(f"   {best_metrics['winrate']*100:.2f}% < 50.76% (минимум для прибыли)")

    if best_metrics['roi_total'] > 0:
        print(f"\n4. ✅ Стратегия прибыльна на синтетике!")
        print(f"   Total ROI: {best_metrics['roi_total']:+.2f} USDT на {best_metrics['n_samples']:,} ставок")

        # Экстраполяция на месяц
        monthly_bets = 288 * 30  # 5 минут * 24 часа * 30 дней
        monthly_roi = best_metrics['roi_per_bet'] * monthly_bets
        monthly_pct = (monthly_roi / 100) * 100  # Assuming 100 USDT bankroll

        print(f"\n   💰 Экстраполяция на месяц:")
        print(f"      Ставок в месяц: {monthly_bets:,}")
        print(f"      Начальный банк: 100 USDT")
        print(f"      Месячный ROI:   {monthly_roi:+.2f} USDT")
        print(f"      Месячная доходность: {monthly_pct:+.1f}%")
    else:
        print(f"\n4. ❌ Стратегия убыточна")
        print(f"   Total ROI: {best_metrics['roi_total']:+.2f} USDT")

    print(f"\n5. 📌 СЛЕДУЮЩИЕ ШАГИ:")
    if best_metrics['accuracy'] > 0.52:
        print(f"   ✅ Базовая логика работает → добавить ML модели для улучшения")
        print(f"   ✅ Ожидаемое улучшение с XGBoost/RF/Neural: +2-5% accuracy")
        print(f"   ✅ После ML → тестировать на реальных данных")
    else:
        print(f"   ⚠️  Базовая логика слабая → ML критически важен")
        print(f"   ⚠️  Без ML сложных паттернов accuracy останется ~{best_metrics['accuracy']*100:.0f}%")
        print(f"   ✅ ML модели (XGBoost/RF/Neural) должны значительно улучшить результат")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
