#!/usr/bin/env python3
"""
БЫСТРЫЙ тест гипотезы: META не работает из-за недостатка данных

ПЛАН:
1. Маленькая сеть [8] с 1000 примеров (хватает)
2. Стандартная сеть [64,32,16] с 1000 примеров (мало!)
3. Сравниваем результаты

БЫСТРЫЕ ПАРАМЕТРЫ:
- CEM 10 итераций (вместо 30)
- Bootstrap 2 (вместо 5)
- Популяция 10 (вместо 20)
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

class FastSmallConfig:
    """Маленькая сеть [8] ~500 параметров"""
    def __init__(self):
        self.meta_nn_hidden = [8]
        self.meta_retrain_every = 300
        self.meta_min_train = 300
        self.meta_use_cma_es = False
        self.meta_cem_iters = 10  # БЫСТРО!
        self.meta_cem_pop = 10
        self.meta_bootstrap_reps = 2
        self.meta_dropout_rate = 0.1
        self.meta_state_path = None
        self.meta_exp4_phases = 6
        self.meta_mc_n_inference = 5

class FastStandardConfig:
    """Стандартная сеть (по умолчанию) ~5200 параметров"""
    def __init__(self):
        self.meta_nn_hidden = None  # Использует стандарт [64,32,16]
        self.meta_retrain_every = 500
        self.meta_min_train = 500
        self.meta_use_cma_es = False
        self.meta_cem_iters = 10  # БЫСТРО!
        self.meta_cem_pop = 10
        self.meta_bootstrap_reps = 2
        self.meta_dropout_rate = 0.1
        self.meta_state_path = None
        self.meta_exp4_phases = 6
        self.meta_mc_n_inference = 5

def generate_data(pattern_type=None):
    """Генерация данных с паттернами"""
    if pattern_type == 'high':
        preds = [np.random.uniform(0.70, 0.95) for _ in range(4)]
        outcome = 1 if np.random.rand() < 0.80 else 0
    elif pattern_type == 'low':
        preds = [np.random.uniform(0.05, 0.30) for _ in range(4)]
        outcome = 0 if np.random.rand() < 0.80 else 1
    else:
        preds = [np.random.uniform(0.1, 0.9) for _ in range(4)]
        outcome = 1 if np.random.rand() < np.mean(preds) else 0

    return preds, outcome

def run_test(config, n_samples, name):
    """Быстрый тест"""
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")

    tmpdir = tempfile.mkdtemp()
    try:
        config.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(config)

        # Подсчет параметров
        net = meta.networks[0]
        n_params = len(net.get_weights_flat())
        ratio = n_samples / n_params

        print(f"Параметров: {n_params:,}")
        print(f"Примеров: {n_samples:,}")
        print(f"Ratio: {ratio:.2f}x", end="")

        if ratio < 10:
            print(" ❌ МАЛО (нужно >10x)")
        elif ratio < 20:
            print(" ⚠️ НЕДОСТАТОЧНО (нужно >20x)")
        else:
            print(" ✅ ХВАТАЕТ")

        # Обучение
        print(f"\nОбучение на {n_samples:,} примерах...")
        t0 = time.time()

        for i in range(n_samples):
            # 40% high, 40% low, 20% random
            rand = np.random.rand()
            if rand < 0.40:
                preds, outcome = generate_data('high')
            elif rand < 0.80:
                preds, outcome = generate_data('low')
            else:
                preds, outcome = generate_data()

            meta.record_result(
                p_xgb=preds[0], p_rf=preds[1],
                p_arf=preds[2], p_nn=preds[3],
                p_base=np.mean(preds),
                y_up=outcome, used_in_live=True,
                reg_ctx={'phase': 0}
            )

        train_time = time.time() - t0
        print(f"Обучено за {train_time:.1f}s")

        # Тестирование
        meta.mode = "ACTIVE"

        test_high, test_low = [], []
        base_high, base_low = [], []
        outcomes_high, outcomes_low = [], []

        # Тест на паттерне HIGH
        for _ in range(100):
            preds, outcome = generate_data('high')
            p = meta.predict(
                p_xgb=preds[0], p_rf=preds[1],
                p_arf=preds[2], p_nn=preds[3],
                p_base=np.mean(preds),
                reg_ctx={'phase': 0}
            )
            if p is not None:
                test_high.append(p)
                base_high.append(np.mean(preds))
                outcomes_high.append(outcome)

        # Тест на паттерне LOW
        for _ in range(100):
            preds, outcome = generate_data('low')
            p = meta.predict(
                p_xgb=preds[0], p_rf=preds[1],
                p_arf=preds[2], p_nn=preds[3],
                p_base=np.mean(preds),
                reg_ctx={'phase': 0}
            )
            if p is not None:
                test_low.append(p)
                base_low.append(np.mean(preds))
                outcomes_low.append(outcome)

        # Результаты
        if len(test_high) > 0 and len(test_low) > 0:
            # Средние вероятности для паттернов
            meta_high_avg = np.mean(test_high)
            meta_low_avg = np.mean(test_low)
            base_high_avg = np.mean(base_high)
            base_low_avg = np.mean(base_low)

            # Точность
            meta_correct = (
                sum(1 for i in range(len(test_high)) if
                    (test_high[i] > 0.5) == outcomes_high[i]) +
                sum(1 for i in range(len(test_low)) if
                    (test_low[i] > 0.5) == outcomes_low[i])
            )
            meta_acc = meta_correct / (len(test_high) + len(test_low)) * 100

            base_correct = (
                sum(1 for i in range(len(base_high)) if
                    (base_high[i] > 0.5) == outcomes_high[i]) +
                sum(1 for i in range(len(base_low)) if
                    (base_low[i] > 0.5) == outcomes_low[i])
            )
            base_acc = base_correct / (len(base_high) + len(base_low)) * 100

            print(f"\n📈 РЕЗУЛЬТАТЫ:")
            print(f"META точность:     {meta_acc:.1f}%")
            print(f"Baseline точность: {base_acc:.1f}%")
            print(f"Улучшение:         {meta_acc - base_acc:+.1f}%")

            print(f"\n🔍 ПАТТЕРНЫ:")
            print(f"HIGH (ожидаем >0.65): META={meta_high_avg:.3f}, Base={base_high_avg:.3f}")
            if meta_high_avg > 0.65:
                print(f"  ✅ META распознала HIGH")
            else:
                print(f"  ❌ META НЕ распознала HIGH")

            print(f"LOW  (ожидаем <0.35): META={meta_low_avg:.3f}, Base={base_low_avg:.3f}")
            if meta_low_avg < 0.35:
                print(f"  ✅ META распознала LOW")
            else:
                print(f"  ❌ META НЕ распознала LOW")

            return {
                'n_params': n_params,
                'n_samples': n_samples,
                'ratio': ratio,
                'meta_acc': meta_acc,
                'base_acc': base_acc,
                'improvement': meta_acc - base_acc,
                'high_found': meta_high_avg > 0.65,
                'low_found': meta_low_avg < 0.35,
                'train_time': train_time
            }
        else:
            print("❌ Нет предсказаний")
            return None

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def main():
    print("="*60)
    print("БЫСТРЫЙ ТЕСТ: Гипотеза о недостатке данных")
    print("="*60)

    np.random.seed(42)

    # Тест 1: Маленькая сеть с достаточными данными
    result1 = run_test(
        FastSmallConfig(),
        n_samples=1000,
        name="Маленькая сеть [8] + 1000 примеров"
    )

    # Тест 2: Стандартная сеть с недостаточными данными
    result2 = run_test(
        FastStandardConfig(),
        n_samples=1000,
        name="Стандартная сеть [64,32,16] + 1000 примеров"
    )

    # Выводы
    print("\n" + "="*60)
    print("ВЫВОДЫ")
    print("="*60)

    if result1 and result2:
        print(f"\n1️⃣ МАЛЕНЬКАЯ СЕТЬ [8]:")
        print(f"   Параметров: {result1['n_params']:,}")
        print(f"   Ratio: {result1['ratio']:.1f}x {'✅' if result1['ratio'] >= 10 else '❌'}")
        print(f"   Улучшение: {result1['improvement']:+.1f}% {'✅' if result1['improvement'] > 0 else '❌'}")
        print(f"   Паттерны: {sum([result1['high_found'], result1['low_found']])}/2")

        print(f"\n2️⃣ СТАНДАРТНАЯ СЕТЬ [64,32,16]:")
        print(f"   Параметров: {result2['n_params']:,}")
        print(f"   Ratio: {result2['ratio']:.1f}x {'✅' if result2['ratio'] >= 10 else '❌'}")
        print(f"   Улучшение: {result2['improvement']:+.1f}% {'✅' if result2['improvement'] > 0 else '❌'}")
        print(f"   Паттерны: {sum([result2['high_found'], result2['low_found']])}/2")

        print(f"\n" + "="*60)
        print("ИТОГ:")
        print("="*60)

        # Гипотеза: маленькая сеть должна работать лучше
        if result1['improvement'] > result2['improvement']:
            diff = result1['improvement'] - result2['improvement']
            print(f"\n✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА!")
            print(f"   Маленькая сеть лучше на {diff:.1f}%")
            print(f"   Причина: достаточно данных (ratio={result1['ratio']:.1f}x)")
            print(f"\n💡 РЕКОМЕНДАЦИЯ:")
            print(f"   Для стандартной сети нужно минимум {result2['n_params']*10:,} примеров")
            print(f"   Сейчас есть только {result2['n_samples']:,} ({result2['ratio']:.1f}x)")
        else:
            print(f"\n❌ ГИПОТЕЗА НЕ ПОДТВЕРЖДЕНА")
            print(f"   Обе сети показывают похожие результаты")
            print(f"   Проблема может быть не только в данных")
            print(f"\n💡 Возможные причины:")
            print(f"   - Алгоритм CEM застревает в локальных минимумах")
            print(f"   - Недостаточно итераций оптимизации")
            print(f"   - Паттерны слишком сложные для эволюционного алгоритма")

if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱️  Время: {elapsed:.1f}s")
