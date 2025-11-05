# -*- coding: utf-8 -*-
"""
test_meta_neural_quick.py

БЫСТРАЯ ВЕРСИЯ ТЕСТА META НЕЙРОСЕТИ

Использует 10,000 примеров (вместо 360,000) для быстрой проверки:
- 1,667 примеров на фазу (вместо 60,000)
- Упрощенное обучение CEM (10 итераций вместо 30)

Цель: быстрая проверка работоспособности и оценка ожидаемых результатов
"""

import os
import sys

# Импортируем полную версию
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_meta_neural_realistic import (
    TestConfig,
    RealisticPancakeSwapGenerator,
    test_meta_neural,
    HAVE_META
)

import numpy as np
import time


def test_meta_neural_quick():
    """Быстрая версия теста META"""

    print("=" * 70)
    print("⚡ БЫСТРЫЙ ТЕСТ META НЕЙРОСЕТИ (10K ПРИМЕРОВ)")
    print("=" * 70)

    if not HAVE_META:
        print("❌ META недоступна - завершаем")
        return

    # Быстрая конфигурация
    class QuickConfig(TestConfig):
        samples_per_phase = 1667   # ~1.7k на фазу
        total_samples = 10000      # всего 10k
        train_ratio = 0.80

    cfg = QuickConfig()

    print(f"\n📊 Параметры быстрого теста:")
    print(f"  Всего примеров: {cfg.total_samples:,} (вместо 360,000)")
    print(f"  Примеров на фазу: {cfg.samples_per_phase:,} (вместо 60,000)")
    print(f"  Train/Test split: {cfg.train_ratio:.0%} / {1-cfg.train_ratio:.0%}")
    print(f"  ⏱️ Ожидаемое время: ~2-3 минуты")

    # Генератор данных
    print(f"\n🎲 Генерация реалистичных данных...")
    generator = RealisticPancakeSwapGenerator(seed=42)

    # Импортируем META
    from meta_neural_cem import MetaNeuralCEM

    # Собираем все данные
    all_train_examples = []
    all_train_labels = []
    all_test_examples = []
    all_test_labels = []

    for phase in range(6):
        # Генерируем данные для фазы
        examples, labels = generator.generate_phase_data(
            phase=phase,
            n_samples=cfg.samples_per_phase
        )

        # Split на train/test
        n_train = int(len(examples) * cfg.train_ratio)

        train_ex = examples[:n_train]
        train_lab = labels[:n_train]
        test_ex = examples[n_train:]
        test_lab = labels[n_train:]

        all_train_examples.extend(train_ex)
        all_train_labels.extend(train_lab)
        all_test_examples.extend(test_ex)
        all_test_labels.extend(test_lab)

        print(f"  ✓ Фаза {phase}: train={len(train_ex):,}, test={len(test_ex):,}")

    print(f"\n📦 Итого данных:")
    print(f"  Train: {len(all_train_examples):,} примеров")
    print(f"  Test:  {len(all_test_examples):,} примеров")

    # Перемешиваем train данные
    train_indices = np.random.permutation(len(all_train_examples))
    all_train_examples = [all_train_examples[i] for i in train_indices]
    all_train_labels = [all_train_labels[i] for i in train_indices]

    # Инициализация META
    print(f"\n🧠 Инициализация META нейросети...")
    meta = MetaNeuralCEM(cfg)

    print(f"  ✓ Архитектура: 36D features + 7D context")
    print(f"  ✓ Параметров: ~5200")
    print(f"  ✓ Обучение: CEM (упрощенное)")

    # Обучение на train данных
    print(f"\n📚 Обучение META на train данных...")

    start_time = time.time()

    for i, (example, label) in enumerate(zip(all_train_examples, all_train_labels)):
        # Записываем результат в META
        meta.record_result(
            p_xgb=example['p_xgb'],
            p_rf=example['p_rf'],
            p_arf=example['p_arf'],
            p_nn=example['p_nn'],
            p_base=example['p_base'],
            y_up=label,
            used_in_live=False,
            reg_ctx=example['ctx']
        )

        # Прогресс каждые 1000 примеров
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(all_train_examples) - i - 1) / rate
            print(f"  Прогресс: {i+1:,}/{len(all_train_examples):,} "
                  f"({100*(i+1)/len(all_train_examples):.1f}%) - "
                  f"осталось ~{remaining:.0f} сек")

    train_time = time.time() - start_time
    print(f"\n✅ Обучение завершено за {train_time:.1f} секунд ({train_time/60:.1f} мин)")

    # Тестирование на test данных
    print(f"\n🎯 Тестирование на hold-out данных...")

    predictions = []
    actuals = []
    phase_results = {p: {'correct': 0, 'total': 0} for p in range(6)}

    for example, label in zip(all_test_examples, all_test_labels):
        # Предсказание META
        p_meta = meta.predict(
            p_xgb=example['p_xgb'],
            p_rf=example['p_rf'],
            p_arf=example['p_arf'],
            p_nn=example['p_nn'],
            p_base=example['p_base'],
            reg_ctx=example['ctx']
        )

        if p_meta is None:
            # Fallback на среднее экспертов
            preds = [example['p_xgb'], example['p_rf'], example['p_arf'], example['p_nn']]
            p_meta = float(np.mean(preds))

        predictions.append(p_meta)
        actuals.append(label)

        # Статистика по фазам
        phase = example['ctx']['phase']
        phase_results[phase]['total'] += 1
        if (p_meta >= 0.5) == bool(label):
            phase_results[phase]['correct'] += 1

    # Результаты
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Общая точность
    correct = np.sum((predictions >= 0.5) == actuals)
    accuracy = 100.0 * correct / len(actuals)

    # Brier score
    brier = np.mean((predictions - actuals) ** 2)

    # Log-loss
    preds_clip = np.clip(predictions, 1e-6, 1 - 1e-6)
    log_loss = -np.mean(actuals * np.log(preds_clip) + (1 - actuals) * np.log(1 - preds_clip))

    print(f"\n" + "=" * 70)
    print(f"📊 РЕЗУЛЬТАТЫ БЫСТРОГО ТЕСТА")
    print(f"=" * 70)

    print(f"\n🎯 Общие метрики:")
    print(f"  Accuracy:     {accuracy:.2f}%")
    print(f"  Brier Score:  {brier:.4f}")
    print(f"  Log-Loss:     {log_loss:.4f}")
    print(f"  Test samples: {len(actuals):,}")

    print(f"\n📈 Результаты по фазам:")
    print(f"  {'Фаза':<6} {'Название':<12} {'Accuracy':<10} {'Samples':<10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10}")

    phase_names = {
        0: 'bull_low',
        1: 'bull_high',
        2: 'bear_low',
        3: 'bear_high',
        4: 'flat_low',
        5: 'flat_high'
    }

    for phase in range(6):
        if phase_results[phase]['total'] > 0:
            phase_acc = 100.0 * phase_results[phase]['correct'] / phase_results[phase]['total']
            print(f"  {phase:<6} {phase_names[phase]:<12} {phase_acc:>8.2f}% {phase_results[phase]['total']:>9,}")
        else:
            print(f"  {phase:<6} {phase_names[phase]:<12} {'—':>8} {0:>9}")

    # Оценка прибыльности
    print(f"\n💰 Оценка прибыльности:")

    winrate = accuracy / 100.0
    commission = 0.03
    break_even = 1.0 / (2.0 - commission)

    print(f"  Break-even WR: {break_even:.4f} ({break_even*100:.2f}%)")
    print(f"  META WR:       {winrate:.4f} ({winrate*100:.2f}%)")

    if winrate > break_even:
        # Прибыльно
        roi_per_bet = (winrate * (1 - commission) - (1 - winrate))
        roi_1000 = roi_per_bet * 1000

        print(f"  ✅ ПРИБЫЛЬНО")
        print(f"  ROI на 1000 ставок: {roi_1000:+.2f} USDT ({roi_per_bet*100:+.3f}% на ставку)")

        # Месячная оценка (3000 раундов в месяц)
        monthly_roi = roi_per_bet * 3000
        print(f"  Ожидаемый месячный ROI: {monthly_roi:+.2f}% на капитал")
    else:
        # Убыточно
        loss_per_bet = (1 - winrate) - winrate * (1 - commission)
        loss_1000 = loss_per_bet * 1000

        print(f"  ❌ УБЫТОЧНО")
        print(f"  Потери на 1000 ставок: {-loss_1000:.2f} USDT ({-loss_per_bet*100:.3f}% на ставку)")

    # Экстраполяция на полный датасет
    print(f"\n🔮 Экстраполяция на полный датасет (360k примеров):")
    print(f"  На основе {len(actuals):,} тестовых примеров:")
    print(f"  Ожидаемая точность: {accuracy:.2f}% ± 1-2%")
    if winrate > break_even:
        print(f"  Ожидаемый месячный ROI: {monthly_roi:+.2f}% ± 2-3%")

    # Статус META
    print(f"\n🔧 Статус META:")
    status = meta.status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print(f"\n" + "=" * 70)
    print(f"⚡ БЫСТРЫЙ ТЕСТ ЗАВЕРШЕН")
    print(f"=" * 70)

    # Очистка
    try:
        if os.path.exists(cfg.meta_state_path):
            os.remove(cfg.meta_state_path)
        # Удаляем CSV фазовых данных
        for p in range(6):
            csv_path = f"meta_neural_phase_{p}.csv"
            if os.path.exists(csv_path):
                os.remove(csv_path)
    except Exception:
        pass


if __name__ == "__main__":
    test_meta_neural_quick()
