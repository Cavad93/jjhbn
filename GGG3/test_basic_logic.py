#!/usr/bin/env python3
"""
Упрощенный тест базовой логики без ML зависимостей

Проверяет:
1. Генерацию синтетических данных с паттернами
2. Простую логику предсказания (без ML моделей)
3. Базовые метрики: accuracy, ROI
"""
import random
import math

def sigmoid(x):
    """Сигмоида"""
    return 1.0 / (1.0 + math.exp(-x))

def tanh(x):
    """Гиперболический тангенс"""
    return math.tanh(x)

def clip(x, min_val, max_val):
    """Ограничивает значение"""
    return max(min_val, min(max_val, x))

class SimplePredictor:
    """Простой предиктор на основе эвристик"""

    def __init__(self):
        self.name = "Simple Heuristic"

    def predict(self, features):
        """
        Предсказывает вероятность UP на основе простых правил

        Features (68 total):
        [0-2]: momentum (5m, 10m, 15m)
        [3]: volatility
        [4]: vol_zscore
        [5]: rsi
        """
        # Momentum signal (avg of first 3 features)
        momentum = sum(features[:3]) / 3.0 if len(features) >= 3 else 0
        momentum_signal = tanh(momentum) * 0.2  # ±20%

        # RSI signal (feature 5)
        rsi = features[5] if len(features) > 5 else 0
        rsi_signal = tanh(rsi) * 0.15  # ±15%

        # Volatility factor (feature 3)
        vol = features[3] if len(features) > 3 else 0
        vol_factor = 1.0 + clip(vol, -1, 1) * 0.1

        # Combine signals
        signal = (momentum_signal + rsi_signal) * vol_factor

        # Base probability 50%
        prob = clip(0.5 + signal, 0.1, 0.9)

        return prob


def generate_synthetic_data(n_samples=1000, n_features=68, signal_strength=0.3):
    """Генерирует синтетические данные"""
    print(f"📊 Генерация {n_samples:,} синтетических примеров...")
    print(f"   Фич: {n_features}, Сила сигнала: {signal_strength:.2f}")

    random.seed(42)

    data = []

    for _ in range(n_samples):
        # Случайные фичи (нормальное распределение)
        features = [random.gauss(0, 1) for _ in range(n_features)]

        # Истинная вероятность UP (на основе паттернов)
        momentum = sum(features[:3]) / 3.0
        rsi = features[5]
        vol = features[3]

        signal = (tanh(momentum) * 0.2 + tanh(rsi) * 0.15) * (1.0 + clip(vol, -1, 1) * 0.1)
        signal *= signal_strength

        true_prob = clip(0.5 + signal, 0.1, 0.9)

        # Генерируем outcome
        outcome = 1 if random.random() < true_prob else 0

        # Фаза рынка
        phase = random.randint(0, 5)

        data.append({
            'features': features,
            'outcome': outcome,
            'phase': phase,
            'true_prob': true_prob
        })

    return data


def evaluate(predictor, test_data):
    """Оценка предиктора"""
    print(f"\n{'='*60}")
    print(f"📊 ТЕСТИРОВАНИЕ: {predictor.name}")
    print(f"{'='*60}")

    correct = 0
    total_roi = 0.0
    predictions = []

    for sample in test_data:
        features = sample['features']
        true_outcome = sample['outcome']

        # Предсказание
        pred_prob = predictor.predict(features)
        pred_binary = 1 if pred_prob > 0.5 else 0

        predictions.append({
            'prob': pred_prob,
            'binary': pred_binary,
            'true': true_outcome
        })

        # Accuracy
        if pred_binary == true_outcome:
            correct += 1

        # ROI (с комиссией 3%)
        if pred_binary == 1:  # Ставим UP
            if true_outcome == 1:  # Выигрыш
                total_roi += 0.97  # +97% (минус комиссия)
            else:  # Проигрыш
                total_roi -= 1.0

    # Метрики
    accuracy = correct / len(test_data)
    roi_per_bet = total_roi / len(test_data)

    # Precision/Recall для UP
    tp = sum(1 for p in predictions if p['binary'] == 1 and p['true'] == 1)
    fp = sum(1 for p in predictions if p['binary'] == 1 and p['true'] == 0)
    fn = sum(1 for p in predictions if p['binary'] == 0 and p['true'] == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Brier score
    brier = sum((p['prob'] - p['true'])**2 for p in predictions) / len(predictions)

    print(f"\nРезультаты на {len(test_data):,} примерах:")
    print(f"  Accuracy:     {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Precision:    {precision:.3f}")
    print(f"  Recall:       {recall:.3f}")
    print(f"  F1 Score:     {f1:.3f}")
    print(f"  Brier Score:  {brier:.4f}")
    print(f"  Total ROI:    {total_roi:+.2f} USDT")
    print(f"  ROI per bet:  {roi_per_bet:+.4f} USDT")

    if total_roi > 0:
        print(f"  ✅ ПРИБЫЛЬНО!")
    else:
        print(f"  ❌ УБЫТОЧНО")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier': brier,
        'roi_total': total_roi,
        'roi_per_bet': roi_per_bet
    }


def main():
    """Главная функция"""
    print(f"\n{'='*60}")
    print(f"🧪 ТЕСТ БАЗОВОЙ ЛОГИКИ (упрощенный)")
    print(f"{'='*60}\n")

    # Генерация данных
    all_data = generate_synthetic_data(
        n_samples=2000,
        n_features=68,
        signal_strength=0.3
    )

    # Train/test split (80/20)
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    print(f"\n✅ Данные разбиты:")
    print(f"   Train: {len(train_data):,} примеров")
    print(f"   Test:  {len(test_data):,} примеров")

    # Статистика
    train_up = sum(1 for d in train_data if d['outcome'] == 1)
    test_up = sum(1 for d in test_data if d['outcome'] == 1)

    print(f"\n📈 Распределение UP/DOWN:")
    print(f"   Train UP: {train_up:,} ({train_up/len(train_data)*100:.1f}%)")
    print(f"   Test UP:  {test_up:,} ({test_up/len(test_data)*100:.1f}%)")

    # Тестирование простого предиктора
    predictor = SimplePredictor()
    metrics = evaluate(predictor, test_data)

    # Выводы
    print(f"\n{'='*60}")
    print(f"🎯 ВЫВОДЫ")
    print(f"{'='*60}\n")

    if metrics['accuracy'] > 0.52:
        print(f"✅ Базовая логика РАБОТАЕТ!")
        print(f"   Accuracy {metrics['accuracy']*100:.1f}% > 52%")
        print(f"   Это выше порога прибыльности с учетом комиссии")
    else:
        print(f"⚠️  Accuracy {metrics['accuracy']*100:.1f}% < 52%")
        print(f"   Недостаточно для прибыльности с комиссией 3%")

    if metrics['roi_total'] > 0:
        print(f"\n✅ Стратегия ПРИБЫЛЬНА на синтетике!")
        print(f"   Total ROI: {metrics['roi_total']:+.2f} USDT")
        print(f"   ROI per bet: {metrics['roi_per_bet']:+.4f} USDT")

        # Экстраполяция
        bets_per_day = 288  # 24h * 60min / 5min
        days_per_month = 30
        initial_bankroll = 100  # USDT

        monthly_roi = metrics['roi_per_bet'] * bets_per_day * days_per_month
        monthly_pct = (monthly_roi / initial_bankroll) * 100

        print(f"\n💰 Экстраполяция на месяц:")
        print(f"   Ставок в день:    {bets_per_day}")
        print(f"   Ставок в месяц:   {bets_per_day * days_per_month:,}")
        print(f"   Начальный банк:   {initial_bankroll} USDT")
        print(f"   Месячный ROI:     {monthly_roi:+.2f} USDT")
        print(f"   Месячная доходность: {monthly_pct:+.1f}%")

        if monthly_pct > 10:
            print(f"\n   ⚠️  ВНИМАНИЕ: {monthly_pct:.1f}% в месяц - НЕРЕАЛИСТИЧНО высоко!")
            print(f"   Это результат на идеальной синтетике")
            print(f"   Реальная доходность будет ЗНАЧИТЕЛЬНО ниже")
        elif monthly_pct > 0:
            print(f"\n   ✅ {monthly_pct:.1f}% в месяц - возможно на синтетике")
            print(f"   На реальных данных ожидайте 30-50% от этого")
    else:
        print(f"\n❌ Стратегия УБЫТОЧНА на синтетике")
        print(f"   Total ROI: {metrics['roi_total']:+.2f} USDT")

    print(f"\n📌 Следующие шаги:")
    print(f"   1. Дождаться установки ML библиотек")
    print(f"   2. Запустить полный тест с XGBoost/RF/Neural")
    print(f"   3. Проверить на реальных исторических данных")
    print(f"   4. Paper trading перед реальной торговлей")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
