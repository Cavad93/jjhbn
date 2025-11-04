#!/usr/bin/env python3
"""
Тест гипотезы: META не работает из-за недостатка данных

АРХИТЕКТУРЫ:
1. МАЛЕНЬКАЯ [8]:        ~500 параметров  → нужно 5,000-10,000 примеров
2. СТАНДАРТНАЯ [64,32,16]: ~5,200 параметров → нужно 52,000-104,000 примеров

ТЕСТИРОВАНИЕ:
- Маленькая сеть с 500, 2,000, 5,000, 10,000 примеров
- Стандартная сеть с 500, 2,000, 10,000, 50,000 примеров
"""
import sys
import os
import time
import tempfile
import shutil
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from meta_neural_cem import MetaNeuralCEM

class SmallNetConfig:
    """Маленькая сеть [8] ~500 параметров"""
    def __init__(self):
        self.meta_nn_hidden = [8]
        self.meta_retrain_every = 200
        self.meta_min_train = 200
        self.meta_use_cma_es = False  # CEM быстрее для теста
        self.meta_cem_iters = 30
        self.meta_cem_pop = 20
        self.meta_bootstrap_reps = 3
        self.meta_dropout_rate = 0.1
        self.meta_state_path = None
        self.meta_exp4_phases = 6
        self.meta_mc_n_inference = 10

class StandardNetConfig:
    """Стандартная сеть [64,32,16] ~5200 параметров"""
    def __init__(self):
        # Оставляем параметры по умолчанию (сеть создается внутри NeuralMetaNetwork)
        self.meta_nn_hidden = None  # Использует стандартную архитектуру из NeuralMetaNetwork
        self.meta_retrain_every = 500
        self.meta_min_train = 500
        self.meta_use_cma_es = False  # CEM быстрее
        self.meta_cem_iters = 30
        self.meta_cem_pop = 20
        self.meta_bootstrap_reps = 3
        self.meta_dropout_rate = 0.1
        self.meta_state_path = None
        self.meta_exp4_phases = 6
        self.meta_mc_n_inference = 10

def generate_pattern_data(force_pattern=None):
    """Генерирует данные с паттернами"""
    if force_pattern == 'high_up':
        preds = [np.random.uniform(0.70, 0.95) for _ in range(4)]
    elif force_pattern == 'low_down':
        preds = [np.random.uniform(0.05, 0.30) for _ in range(4)]
    elif force_pattern == 'spread':
        preds = [
            np.random.uniform(0.1, 0.3),
            np.random.uniform(0.7, 0.9),
            np.random.uniform(0.2, 0.4),
            np.random.uniform(0.6, 0.8),
        ]
    else:
        preds = [np.random.uniform(0.1, 0.9) for _ in range(4)]

    return preds

def get_true_outcome(preds):
    """Истинная закономерность"""
    if all(p > 0.7 for p in preds):
        return 1 if np.random.rand() < 0.80 else 0
    if all(p < 0.3 for p in preds):
        return 0 if np.random.rand() < 0.80 else 1
    spread = max(preds) - min(preds)
    if spread > 0.4:
        return np.random.randint(0, 2)
    return 1 if np.random.rand() < np.mean(preds) else 0

def count_network_parameters(meta):
    """Подсчитывает количество параметров в сети"""
    net = meta.networks[0]
    weights = net.get_weights_flat()
    return len(weights)

def run_experiment(config, n_train, n_test=500, name="Test"):
    """Запускает эксперимент с заданной конфигурацией и количеством данных"""
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print(f"   Обучающих примеров: {n_train}")
    print(f"   Тестовых примеров: {n_test}")
    print(f"{'='*70}")

    np.random.seed(42)
    tmpdir = tempfile.mkdtemp()

    try:
        config.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(config)

        # Подсчет параметров
        n_params = count_network_parameters(meta)
        ratio = n_train / n_params
        print(f"   Параметров сети: {n_params:,}")
        print(f"   Соотношение данные/параметры: {ratio:.2f}x")

        if ratio < 10:
            print(f"   ⚠️  КРИТИЧНО: нужно минимум {n_params*10:,} примеров!")
        elif ratio < 20:
            print(f"   ⚠️  МАЛО: рекомендуется {n_params*20:,} примеров")
        else:
            print(f"   ✅ ДОСТАТОЧНО: данных хватает")

        # Обучение
        print(f"\n   Обучение...")
        t0 = time.time()
        for i in range(n_train):
            rand = np.random.rand()
            if rand < 0.30:
                preds = generate_pattern_data('high_up')
            elif rand < 0.60:
                preds = generate_pattern_data('low_down')
            elif rand < 0.80:
                preds = generate_pattern_data('spread')
            else:
                preds = generate_pattern_data()

            outcome = get_true_outcome(preds)

            meta.record_result(
                p_xgb=preds[0],
                p_rf=preds[1],
                p_arf=preds[2],
                p_nn=preds[3],
                p_base=np.mean(preds),
                y_up=outcome,
                used_in_live=True,
                reg_ctx={'phase': 0}
            )

            if (i + 1) % max(1, n_train // 5) == 0:
                print(f"   ...{i+1}/{n_train}", end='\r')

        train_time = time.time() - t0
        print(f"   Обучение завершено за {train_time:.1f}s")

        # Тестирование
        meta.mode = "ACTIVE"

        test_outcomes = []
        meta_preds = []
        baseline_preds = []

        pattern_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'meta_preds': []})

        for i in range(n_test):
            rand = np.random.rand()
            if rand < 0.25:
                preds = generate_pattern_data('high_up')
                pattern_type = 'high_up'
            elif rand < 0.50:
                preds = generate_pattern_data('low_down')
                pattern_type = 'low_down'
            elif rand < 0.75:
                preds = generate_pattern_data('spread')
                pattern_type = 'spread'
            else:
                preds = generate_pattern_data()
                pattern_type = 'other'

            outcome = get_true_outcome(preds)

            meta_pred = meta.predict(
                p_xgb=preds[0],
                p_rf=preds[1],
                p_arf=preds[2],
                p_nn=preds[3],
                p_base=np.mean(preds),
                reg_ctx={'phase': 0}
            )

            if meta_pred is not None:
                test_outcomes.append(outcome)
                meta_preds.append(meta_pred)
                baseline_preds.append(np.mean(preds))

                pattern_stats[pattern_type]['total'] += 1
                pattern_stats[pattern_type]['meta_preds'].append(meta_pred)
                if (meta_pred > 0.5 and outcome == 1) or (meta_pred <= 0.5 and outcome == 0):
                    pattern_stats[pattern_type]['correct'] += 1

        # Результаты
        if len(meta_preds) > 0:
            meta_correct = sum(1 for i in range(len(meta_preds)) if
                             (meta_preds[i] > 0.5 and test_outcomes[i] == 1) or
                             (meta_preds[i] <= 0.5 and test_outcomes[i] == 0))
            meta_acc = meta_correct / len(meta_preds) * 100

            baseline_correct = sum(1 for i in range(len(baseline_preds)) if
                                 (baseline_preds[i] > 0.5 and test_outcomes[i] == 1) or
                                 (baseline_preds[i] <= 0.5 and test_outcomes[i] == 0))
            baseline_acc = baseline_correct / len(baseline_preds) * 100

            improvement = meta_acc - baseline_acc

            print(f"\n   📈 РЕЗУЛЬТАТЫ:")
            print(f"      META Neural:     {meta_acc:.1f}%")
            print(f"      Baseline:        {baseline_acc:.1f}%")
            print(f"      Улучшение:       {improvement:+.1f}%")

            # Распознавание паттернов
            patterns_found = 0
            if 'high_up' in pattern_stats and pattern_stats['high_up']['total'] > 5:
                avg = np.mean(pattern_stats['high_up']['meta_preds'])
                if avg > 0.65:
                    patterns_found += 1
                    print(f"      ✅ Паттерн HIGH_UP распознан (p={avg:.3f})")
                else:
                    print(f"      ❌ Паттерн HIGH_UP НЕ распознан (p={avg:.3f})")

            if 'low_down' in pattern_stats and pattern_stats['low_down']['total'] > 5:
                avg = np.mean(pattern_stats['low_down']['meta_preds'])
                if avg < 0.35:
                    patterns_found += 1
                    print(f"      ✅ Паттерн LOW_DOWN распознан (p={avg:.3f})")
                else:
                    print(f"      ❌ Паттерн LOW_DOWN НЕ распознан (p={avg:.3f})")

            return {
                'n_params': n_params,
                'n_train': n_train,
                'ratio': ratio,
                'meta_acc': meta_acc,
                'baseline_acc': baseline_acc,
                'improvement': improvement,
                'patterns_found': patterns_found,
                'train_time': train_time
            }
        else:
            print(f"   ❌ META не дала предсказаний")
            return None

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def main():
    print("="*70)
    print("ТЕСТ ГИПОТЕЗЫ: META не работает из-за недостатка данных")
    print("="*70)

    results = []

    # ========== ЭКСПЕРИМЕНТ 1: МАЛЕНЬКАЯ СЕТЬ [8] ==========
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 1: МАЛЕНЬКАЯ СЕТЬ [8]")
    print("Ожидаемые параметры: ~500")
    print("="*70)

    for n_train in [500, 2000, 5000, 10000]:
        result = run_experiment(
            SmallNetConfig(),
            n_train=n_train,
            n_test=500,
            name=f"Маленькая сеть [8], {n_train:,} примеров"
        )
        if result:
            results.append(('small', result))

    # ========== ЭКСПЕРИМЕНТ 2: СТАНДАРТНАЯ СЕТЬ [64,32,16] ==========
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ 2: СТАНДАРТНАЯ СЕТЬ [64,32,16]")
    print("Ожидаемые параметры: ~5,200")
    print("="*70)

    for n_train in [500, 2000, 10000]:
        result = run_experiment(
            StandardNetConfig(),
            n_train=n_train,
            n_test=500,
            name=f"Стандартная сеть [64,32,16], {n_train:,} примеров"
        )
        if result:
            results.append(('standard', result))

    # ========== ИТОГОВЫЙ АНАЛИЗ ==========
    print("\n" + "="*70)
    print("ИТОГОВЫЙ АНАЛИЗ")
    print("="*70)

    print("\n📊 Маленькая сеть [8]:")
    small_results = [r for net, r in results if net == 'small']
    for r in small_results:
        status = "✅" if r['improvement'] > 0 else "❌"
        print(f"   {status} {r['n_train']:,} примеров (ratio={r['ratio']:.1f}x): "
              f"META={r['meta_acc']:.1f}% vs {r['baseline_acc']:.1f}% "
              f"({r['improvement']:+.1f}%), паттернов={r['patterns_found']}/2")

    print("\n📊 Стандартная сеть [64,32,16]:")
    std_results = [r for net, r in results if net == 'standard']
    for r in std_results:
        status = "✅" if r['improvement'] > 0 else "❌"
        print(f"   {status} {r['n_train']:,} примеров (ratio={r['ratio']:.1f}x): "
              f"META={r['meta_acc']:.1f}% vs {r['baseline_acc']:.1f}% "
              f"({r['improvement']:+.1f}%), паттернов={r['patterns_found']}/2")

    # Выводы
    print("\n" + "="*70)
    print("ВЫВОДЫ")
    print("="*70)

    small_best = max(small_results, key=lambda r: r['improvement']) if small_results else None
    std_best = max(std_results, key=lambda r: r['improvement']) if std_results else None

    if small_best:
        print(f"\n✅ МАЛЕНЬКАЯ СЕТЬ:")
        print(f"   Лучший результат: {small_best['n_train']:,} примеров")
        print(f"   Улучшение: {small_best['improvement']:+.1f}%")
        print(f"   Ratio: {small_best['ratio']:.1f}x (нужно 10-20x)")

        if small_best['improvement'] > 5:
            print(f"   ✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА для маленькой сети!")
        else:
            print(f"   ❌ Даже с достаточными данными улучшение минимально")

    if std_best:
        print(f"\n📊 СТАНДАРТНАЯ СЕТЬ:")
        print(f"   Лучший результат: {std_best['n_train']:,} примеров")
        print(f"   Улучшение: {std_best['improvement']:+.1f}%")
        print(f"   Ratio: {std_best['ratio']:.1f}x (нужно 10-20x)")

        if std_best['ratio'] < 10:
            print(f"   ⚠️  Все еще недостаточно данных!")
            print(f"   📌 Нужно минимум {std_best['n_params']*10:,} примеров")

        if std_best['improvement'] > 5:
            print(f"   ✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА для стандартной сети!")
        else:
            print(f"   ❌ Проблема не только в данных")

if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱️  Общее время: {elapsed/60:.1f} минут")
