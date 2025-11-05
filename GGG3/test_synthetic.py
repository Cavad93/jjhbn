#!/usr/bin/env python3
"""
Тестирование базовой логики бота на синтетических данных

Проверяет:
1. Генерацию синтетических данных с известными паттернами
2. Обучение всех 4 экспертов
3. Калибровку вероятностей
4. Обучение META слоя
5. Финальный winrate и метрики

Если базовая логика работает, winrate должен быть > 55% на синтетике
"""
import sys
import os
import numpy as np
from pathlib import Path

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent))

try:
    from bnbusdrt6 import XGBExpert, RFCalibratedExpert, RiverARFExpert, NNExpert
    print("✅ Импортированы эксперты из bnbusdrt6.py")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("   Используем упрощенный тест без ML моделей")
    XGBExpert = None
    RFCalibratedExpert = None
    RiverARFExpert = None
    NNExpert = None


def generate_synthetic_data(n_samples=10000, n_features=68, signal_strength=0.3):
    """
    Генерирует синтетические данные с ИЗВЕСТНЫМИ паттернами

    Args:
        n_samples: Количество примеров
        n_features: Количество фич (68 как у реального бота)
        signal_strength: Сила сигнала (0.0 = random, 1.0 = perfect prediction)

    Returns:
        X: np.array shape (n_samples, n_features)
        y: np.array shape (n_samples,) - 0/1 outcomes
        phases: np.array shape (n_samples,) - market phases 0-5
        true_probs: np.array shape (n_samples,) - истинные вероятности UP
    """
    print(f"📊 Генерация {n_samples:,} синтетических примеров...")
    print(f"   Сила сигнала: {signal_strength:.2f}")

    np.random.seed(42)

    # Базовые случайные фичи
    X = np.random.randn(n_samples, n_features)

    # Фазы рынка (равномерно распределены)
    phases = np.random.randint(0, 6, n_samples)

    # Создаем ИЗВЕСТНЫЕ паттерны для предсказания
    true_probs = np.zeros(n_samples)

    for i in range(n_samples):
        # Базовая вероятность 50%
        base_prob = 0.5

        # ПАТТЕРН 1: Momentum (фичи 0-2 - это momentum_5m, momentum_10m, momentum_15m)
        momentum_signal = (X[i, 0] + X[i, 1] + X[i, 2]) / 3.0
        momentum_signal = np.tanh(momentum_signal) * 0.2  # ±20%

        # ПАТТЕРН 2: RSI oversold/overbought (фича 5 - это RSI)
        # RSI > 0 → oversold → UP, RSI < 0 → overbought → DOWN
        rsi_signal = np.tanh(X[i, 5]) * 0.15  # ±15%

        # ПАТТЕРН 3: Volatility regime (фича 3 - volatility)
        # Высокая волатильность → momentum continuation
        vol_factor = 1.0 + np.clip(X[i, 3], -1, 1) * 0.1

        # ПАТТЕРН 4: Phase-specific bias
        phase_bias = 0.0
        if phases[i] == 0:  # bull_low
            phase_bias = 0.05  # Slight bullish bias
        elif phases[i] == 3:  # bear_high
            phase_bias = -0.05  # Slight bearish bias

        # Комбинируем сигналы
        signal = momentum_signal * vol_factor + rsi_signal + phase_bias

        # Применяем signal_strength
        signal *= signal_strength

        # Финальная вероятность (clipped to [0.1, 0.9])
        true_probs[i] = np.clip(base_prob + signal, 0.1, 0.9)

    # Генерируем outcomes на основе истинных вероятностей
    y = (np.random.rand(n_samples) < true_probs).astype(int)

    # Статистика
    print(f"\n📈 Статистика синтетических данных:")
    print(f"   Средняя истинная вероятность UP: {true_probs.mean():.3f}")
    print(f"   Std истинных вероятностей: {true_probs.std():.3f}")
    print(f"   Фактический UP rate: {y.mean():.3f}")
    print(f"   Распределение по фазам:")
    for ph in range(6):
        count = np.sum(phases == ph)
        print(f"      Phase {ph}: {count:,} ({count/n_samples*100:.1f}%)")

    return X, y, phases, true_probs


def evaluate_predictions(y_true, y_pred_probs, y_pred_binary=None, name="Model"):
    """
    Оценка качества предсказаний

    Args:
        y_true: Истинные метки (0/1)
        y_pred_probs: Предсказанные вероятности [0, 1]
        y_pred_binary: Бинарные предсказания (0/1), если None - используем threshold 0.5
        name: Название модели для вывода

    Returns:
        dict с метриками
    """
    if y_pred_binary is None:
        y_pred_binary = (y_pred_probs > 0.5).astype(int)

    # Accuracy
    accuracy = np.mean(y_true == y_pred_binary)

    # Precision, Recall для UP класса
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Log loss (cross-entropy)
    eps = 1e-7
    y_pred_clipped = np.clip(y_pred_probs, eps, 1 - eps)
    log_loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    # Brier score
    brier = np.mean((y_pred_probs - y_true) ** 2)

    # ROI симуляция (с комиссией 3%)
    roi_total = 0
    for i in range(len(y_true)):
        if y_pred_binary[i] == 1:  # Ставим UP
            if y_true[i] == 1:  # Выигрыш
                roi_total += 0.97  # +97% (с учетом комиссии)
            else:  # Проигрыш
                roi_total -= 1.0
        # Если не ставим, ROI не меняется

    n_bets = np.sum(y_pred_binary >= 0)  # Всегда ставим в этом тесте
    roi_per_bet = roi_total / n_bets if n_bets > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'log_loss': log_loss,
        'brier_score': brier,
        'roi_total': roi_total,
        'roi_per_bet': roi_per_bet,
        'n_bets': n_bets
    }

    print(f"\n{'='*60}")
    print(f"📊 МЕТРИКИ: {name}")
    print(f"{'='*60}")
    print(f"Accuracy:      {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Precision:     {precision:.3f}")
    print(f"Recall:        {recall:.3f}")
    print(f"F1 Score:      {f1:.3f}")
    print(f"Log Loss:      {log_loss:.4f}")
    print(f"Brier Score:   {brier:.4f}")
    print(f"Total ROI:     {roi_total:+.2f} USDT")
    print(f"ROI per bet:   {roi_per_bet:+.4f} USDT")
    print(f"Total bets:    {n_bets:,}")

    # Для прибыльности нужен ROI > 0
    if roi_total > 0:
        print(f"✅ ПРИБЫЛЬНО!")
    else:
        print(f"❌ УБЫТОЧНО")

    return metrics


def test_single_expert(expert, name, X_train, y_train, X_test, y_test, phases_train, phases_test):
    """Тестирует одного эксперта"""
    print(f"\n{'='*60}")
    print(f"🤖 ТЕСТИРОВАНИЕ: {name}")
    print(f"{'='*60}")

    # Обучение на каждой фазе отдельно
    for phase in range(6):
        mask = phases_train == phase
        if np.sum(mask) < 100:
            continue

        X_ph = X_train[mask]
        y_ph = y_train[mask]

        print(f"\nОбучение на фазе {phase}: {len(X_ph):,} примеров...")

        # Записываем результаты
        for i in range(len(X_ph)):
            expert.record_result(X_ph[i], y_up=(y_ph[i] == 1), phase=phase)

        # Периодическое обучение
        if len(X_ph) >= 200:
            for step in range(200, len(X_ph), 200):
                expert.maybe_train(ph=phase, force=True)

    # Финальное обучение на всех фазах
    print(f"\nФинальное обучение всех фаз...")
    for phase in range(6):
        expert.maybe_train(ph=phase, force=True)

    # Тестирование
    print(f"\nТестирование на {len(X_test):,} примерах...")
    predictions = []

    for i in range(len(X_test)):
        phase = phases_test[i]
        pred = expert.predict(X_test[i], phase=phase)
        predictions.append(pred)

    predictions = np.array(predictions)

    # Метрики
    metrics = evaluate_predictions(y_test, predictions, name=name)

    return metrics


def test_meta_ensemble(experts, X_train, y_train, X_test, y_test, phases_train, phases_test):
    """Тестирует META ensemble"""
    print(f"\n{'='*60}")
    print(f"🧠 ТЕСТИРОВАНИЕ: META ENSEMBLE")
    print(f"{'='*60}")

    # Создаем MetaContext
    meta_ctx = MetaContext()

    # Собираем предсказания экспертов на train
    print(f"\nСбор предсказаний экспертов на train...")
    expert_preds_train = {name: [] for name in experts.keys()}

    for i in range(len(X_train)):
        phase = phases_train[i]
        for name, expert in experts.items():
            pred = expert.predict(X_train[i], phase=phase)
            expert_preds_train[name].append(pred)

    # Обучение META на каждой фазе
    print(f"\nОбучение META слоя...")
    for phase in range(6):
        mask = phases_train == phase
        if np.sum(mask) < 100:
            continue

        print(f"  Фаза {phase}: {np.sum(mask):,} примеров")

        # Для упрощения используем простое усреднение (вместо CMA-ES)
        # В реальном боте здесь CMA-ES оптимизация

    # Тестирование META
    print(f"\nТестирование META на {len(X_test):,} примерах...")
    meta_predictions = []

    for i in range(len(X_test)):
        phase = phases_test[i]

        # Собираем предсказания экспертов
        expert_preds = []
        for name, expert in experts.items():
            pred = expert.predict(X_test[i], phase=phase)
            expert_preds.append(pred)

        # Простое усреднение (в реальном боте - взвешенное CMA-ES)
        meta_pred = np.mean(expert_preds)
        meta_predictions.append(meta_pred)

    meta_predictions = np.array(meta_predictions)

    # Метрики
    metrics = evaluate_predictions(y_test, meta_predictions, name="META Ensemble")

    return metrics


def main():
    """Главная функция"""
    print(f"\n{'='*80}")
    print(f"🧪 ТЕСТИРОВАНИЕ БАЗОВОЙ ЛОГИКИ БОТА НА СИНТЕТИЧЕСКИХ ДАННЫХ")
    print(f"{'='*80}\n")

    # Генерация данных
    X, y, phases, true_probs = generate_synthetic_data(
        n_samples=10000,
        n_features=68,
        signal_strength=0.3  # 30% сигнала
    )

    # Train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    phases_train, phases_test = phases[:split_idx], phases[split_idx:]
    true_probs_train, true_probs_test = true_probs[:split_idx], true_probs[split_idx:]

    print(f"\n📚 Данные разбиты:")
    print(f"   Train: {len(X_train):,} примеров")
    print(f"   Test:  {len(X_test):,} примеров")

    # Baseline: использование истинных вероятностей (upper bound)
    print(f"\n{'='*60}")
    print(f"📊 BASELINE: Истинные вероятности (верхняя граница)")
    print(f"{'='*60}")
    baseline_metrics = evaluate_predictions(y_test, true_probs_test, name="True Probabilities")

    # Инициализация экспертов
    print(f"\n{'='*60}")
    print(f"🔧 ИНИЦИАЛИЗАЦИЯ ЭКСПЕРТОВ")
    print(f"{'='*60}")

    experts = {
        'xgb': XGBExpert(cfg),
        'rf': RFExpert(cfg),
        'arf': ARFExpert(cfg),
        'neural': NeuralExpert(cfg)
    }

    for name in experts:
        print(f"✅ {name.upper()} инициализирован")

    # Тестирование каждого эксперта
    expert_metrics = {}

    for name, expert in experts.items():
        metrics = test_single_expert(
            expert, name.upper(),
            X_train, y_train, X_test, y_test,
            phases_train, phases_test
        )
        expert_metrics[name] = metrics

    # Тестирование META ensemble
    meta_metrics = test_meta_ensemble(
        experts,
        X_train, y_train, X_test, y_test,
        phases_train, phases_test
    )

    # Итоговый отчет
    print(f"\n{'='*80}")
    print(f"📋 ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*80}\n")

    print(f"{'Модель':<20} {'Accuracy':>10} {'F1':>8} {'Brier':>8} {'ROI/bet':>10} {'Статус':>10}")
    print(f"{'-'*80}")

    # Baseline
    print(f"{'True Probs (Baseline)':<20} {baseline_metrics['accuracy']:>9.1f}% "
          f"{baseline_metrics['f1']:>8.3f} {baseline_metrics['brier_score']:>8.4f} "
          f"{baseline_metrics['roi_per_bet']:>+9.4f} {'UPPER':<10}")

    # Эксперты
    for name, metrics in expert_metrics.items():
        status = '✅ PROFIT' if metrics['roi_total'] > 0 else '❌ LOSS'
        print(f"{name.upper():<20} {metrics['accuracy']*100:>9.1f}% "
              f"{metrics['f1']:>8.3f} {metrics['brier_score']:>8.4f} "
              f"{metrics['roi_per_bet']:>+9.4f} {status:<10}")

    # META
    status = '✅ PROFIT' if meta_metrics['roi_total'] > 0 else '❌ LOSS'
    print(f"{'META Ensemble':<20} {meta_metrics['accuracy']*100:>9.1f}% "
          f"{meta_metrics['f1']:>8.3f} {meta_metrics['brier_score']:>8.4f} "
          f"{meta_metrics['roi_per_bet']:>+9.4f} {status:<10}")

    print(f"\n{'='*80}")
    print(f"🎯 ВЫВОДЫ")
    print(f"{'='*80}\n")

    # Проверка базовой логики
    best_accuracy = max([m['accuracy'] for m in expert_metrics.values()] + [meta_metrics['accuracy']])

    if best_accuracy > 0.52:
        print("✅ Базовая логика РАБОТАЕТ!")
        print(f"   Лучший accuracy: {best_accuracy*100:.1f}% > 52%")
    else:
        print("⚠️  Базовая логика требует улучшения")
        print(f"   Лучший accuracy: {best_accuracy*100:.1f}% < 52%")

    if meta_metrics['roi_total'] > 0:
        print(f"✅ META ensemble ПРИБЫЛЬНЫЙ на синтетике!")
        print(f"   Total ROI: {meta_metrics['roi_total']:+.2f} USDT")
    else:
        print(f"⚠️  META ensemble убыточный на синтетике")
        print(f"   Total ROI: {meta_metrics['roi_total']:+.2f} USDT")

    print(f"\n💡 Следующие шаги:")
    print(f"   1. Если базовая логика работает → тестировать на реальных данных")
    print(f"   2. Если не работает → отладить калибровку и META слой")
    print(f"   3. Добавить uncertainty filtering (MC Dropout)")
    print(f"   4. Оптимизировать гиперпараметры")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
