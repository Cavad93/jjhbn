#!/usr/bin/env python3
"""
Тест способности META находить скрытые закономерности

СЦЕНАРИЙ:
- 4 эксперта дают случайные предсказания (шум)
- НО есть скрытый паттерн: когда все эксперты согласны → результат предсказуем
- Паттерн 1: Высокое согласие (все >0.7) → UP с 80% вероятностью
- Паттерн 2: Низкое согласие (все <0.3) → DOWN с 80% вероятностью
- Паттерн 3: Разброс >0.4 → результат случаен

ЗАДАЧА META: Найти эти закономерности и научиться их использовать
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

class TestConfig:
    def __init__(self):
        self.meta_nn_hidden = [32, 16]  # БОЛЬШЕ нейронов для сложных паттернов
        self.meta_retrain_every = 80  # Обучение каждые 80 примеров
        self.meta_min_train = 150  # Минимум для первого обучения
        self.meta_use_cma_es = True  # ✅ ВКЛЮЧАЕМ CMA-ES - лучший алгоритм!
        self.meta_cem_iters = 50  # Не используется для CMA-ES
        self.meta_cem_pop = 30  # Не используется для CMA-ES
        self.meta_bootstrap_reps = 5  # Больше bootstrap
        self.meta_dropout_rate = 0.1
        self.meta_state_path = None
        self.meta_exp4_phases = 6
        self.meta_mc_n_inference = 10

def generate_expert_predictions(force_pattern=None):
    """
    Генерирует предсказания 4 экспертов

    Args:
        force_pattern: None (случайно) или 'high_up', 'low_down', 'spread'
    """
    if force_pattern == 'high_up':
        # Все эксперты дают высокие значения (>0.7)
        return [
            np.random.uniform(0.70, 0.95),
            np.random.uniform(0.70, 0.95),
            np.random.uniform(0.70, 0.95),
            np.random.uniform(0.70, 0.95),
        ]
    elif force_pattern == 'low_down':
        # Все эксперты дают низкие значения (<0.3)
        return [
            np.random.uniform(0.05, 0.30),
            np.random.uniform(0.05, 0.30),
            np.random.uniform(0.05, 0.30),
            np.random.uniform(0.05, 0.30),
        ]
    elif force_pattern == 'spread':
        # Большой разброс между экспертами
        return [
            np.random.uniform(0.1, 0.3),
            np.random.uniform(0.7, 0.9),
            np.random.uniform(0.2, 0.4),
            np.random.uniform(0.6, 0.8),
        ]
    else:
        # Случайные значения
        return [
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.1, 0.9),
        ]

def get_true_outcome(preds):
    """
    Истинная закономерность (скрытая от META):
    - Высокое согласие (все >0.7) → UP с 80% вероятностью
    - Низкое согласие (все <0.3) → DOWN с 80% вероятностью
    - Разброс >0.4 → случайный результат 50/50
    """
    p_min = min(preds)
    p_max = max(preds)
    spread = p_max - p_min

    # ПАТТЕРН 1: Высокое согласие → UP
    if all(p > 0.7 for p in preds):
        return 1 if np.random.rand() < 0.80 else 0

    # ПАТТЕРН 2: Низкое согласие → DOWN
    if all(p < 0.3 for p in preds):
        return 0 if np.random.rand() < 0.80 else 1

    # ПАТТЕРН 3: Большой разброс → случайность
    if spread > 0.4:
        return np.random.randint(0, 2)

    # Иначе слабый паттерн по среднему
    p_mean = np.mean(preds)
    return 1 if np.random.rand() < p_mean else 0

def evaluate_predictions(true_outcomes, meta_preds):
    """Оценка качества предсказаний META"""
    if len(meta_preds) == 0:
        return None

    # Бинаризуем предсказания META
    meta_binary = [1 if p > 0.5 else 0 for p in meta_preds]

    # Метрики
    correct = sum(1 for i in range(len(meta_binary)) if meta_binary[i] == true_outcomes[i])
    accuracy = correct / len(meta_binary) * 100

    # Калибровка (разница между предсказанной вероятностью и истинной частотой)
    bins = np.linspace(0, 1, 11)
    calibration_error = 0
    for i in range(len(bins) - 1):
        mask = [(bins[i] <= p < bins[i+1]) for p in meta_preds]
        if sum(mask) > 0:
            pred_prob = np.mean([meta_preds[j] for j in range(len(mask)) if mask[j]])
            true_freq = np.mean([true_outcomes[j] for j in range(len(mask)) if mask[j]])
            calibration_error += abs(pred_prob - true_freq)

    return {
        'accuracy': accuracy,
        'calibration_error': calibration_error / 10,
        'n_samples': len(meta_preds)
    }

def analyze_pattern_recognition(meta, test_cases):
    """
    Анализирует: научилась ли META распознавать паттерны?
    Тестирует на специально подобранных случаях
    """
    results = {
        'high_consensus_up': [],  # Все >0.7
        'low_consensus_down': [],  # Все <0.3
        'high_spread': [],  # Разброс >0.4
    }

    meta.mode = "ACTIVE"  # Включаем предсказания

    for case_type, test_preds in test_cases.items():
        for preds in test_preds:
            pred = meta.predict(
                p_xgb=preds[0],
                p_rf=preds[1],
                p_arf=preds[2],
                p_nn=preds[3],
                p_base=np.mean(preds),
                reg_ctx={'phase': 0}
            )
            if pred is not None:
                results[case_type].append(pred)

    return results

def main():
    print("="*70)
    print("ТЕСТ СПОСОБНОСТИ META НАХОДИТЬ СКРЫТЫЕ ЗАКОНОМЕРНОСТИ")
    print("="*70)

    np.random.seed(42)
    tmpdir = tempfile.mkdtemp()

    try:
        cfg = TestConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(cfg)

        print("\n📊 ПАТТЕРНЫ В ДАННЫХ:")
        print("  1. Все эксперты >0.7 → UP (80% вероятность)")
        print("  2. Все эксперты <0.3 → DOWN (80% вероятность)")
        print("  3. Разброс >0.4 → Случайность (50/50)")
        print("  4. Остальное → Слабая корреляция со средним")

        # ================== ФАЗА 1: ОБУЧЕНИЕ (500 примеров) ==================
        print("\n" + "="*70)
        print("ФАЗА 1: ОБУЧЕНИЕ НА 500 ПРИМЕРАХ")
        print("="*70)

        training_outcomes = []
        training_preds_before = []

        for i in range(500):
            # Контролируемая генерация: 30% high_up, 30% low_down, 20% spread, 20% random
            rand = np.random.rand()
            if rand < 0.30:
                expert_preds = generate_expert_predictions('high_up')
            elif rand < 0.60:
                expert_preds = generate_expert_predictions('low_down')
            elif rand < 0.80:
                expert_preds = generate_expert_predictions('spread')
            else:
                expert_preds = generate_expert_predictions()

            # Получаем истинный результат (с паттерном)
            outcome = get_true_outcome(expert_preds)
            training_outcomes.append(outcome)

            # Пробуем предсказать ДО обучения
            if i < 10:  # Только первые 10 для сравнения
                meta.mode = "ACTIVE"
                pred_before = meta.predict(
                    p_xgb=expert_preds[0],
                    p_rf=expert_preds[1],
                    p_arf=expert_preds[2],
                    p_nn=expert_preds[3],
                    p_base=np.mean(expert_preds),
                    reg_ctx={'phase': 0}
                )
                if pred_before is not None:
                    training_preds_before.append((expert_preds, pred_before, outcome))
                meta.mode = "SHADOW"

            # Записываем результат для обучения
            meta.record_result(
                p_xgb=expert_preds[0],
                p_rf=expert_preds[1],
                p_arf=expert_preds[2],
                p_nn=expert_preds[3],
                p_base=np.mean(expert_preds),
                y_up=outcome,
                used_in_live=True,
                reg_ctx={'phase': 0}
            )

            if (i + 1) % 100 == 0:
                print(f"  📝 Записано {i+1}/500 примеров...")

        print(f"\n✅ Обучение завершено: {meta.seen_ph[0]} примеров")
        print(f"  🔄 Счетчик new_since_train_ph[0]: {meta.new_since_train_ph[0]}")
        print(f"  🧠 Сеть обучена: {'ДА' if meta.networks[0] else 'НЕТ'}")

        # Проверка что обучение действительно произошло
        if meta.new_since_train_ph[0] > 80:
            print(f"  ⚠️  ВНИМАНИЕ: Обучение НЕ сработало! Счетчик {meta.new_since_train_ph[0]} > 80")

        # ================== ФАЗА 2: ТЕСТИРОВАНИЕ (200 примеров) ==================
        print("\n" + "="*70)
        print("ФАЗА 2: ТЕСТИРОВАНИЕ НА 200 НОВЫХ ПРИМЕРАХ")
        print("="*70)

        meta.mode = "ACTIVE"  # Включаем предсказания

        test_outcomes = []
        test_meta_preds = []
        test_expert_avg_preds = []

        # Счетчики по типам паттернов
        pattern_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'meta_preds': []})

        for i in range(200):
            # Контролируемая генерация для теста: 25% каждого типа
            rand = np.random.rand()
            if rand < 0.25:
                expert_preds = generate_expert_predictions('high_up')
            elif rand < 0.50:
                expert_preds = generate_expert_predictions('low_down')
            elif rand < 0.75:
                expert_preds = generate_expert_predictions('spread')
            else:
                expert_preds = generate_expert_predictions()

            outcome = get_true_outcome(expert_preds)

            # Определяем тип паттерна
            p_min, p_max = min(expert_preds), max(expert_preds)
            spread = p_max - p_min
            if all(p > 0.7 for p in expert_preds):
                pattern_type = "high_consensus_up"
            elif all(p < 0.3 for p in expert_preds):
                pattern_type = "low_consensus_down"
            elif spread > 0.4:
                pattern_type = "high_spread"
            else:
                pattern_type = "other"

            # Предсказание META
            meta_pred = meta.predict(
                p_xgb=expert_preds[0],
                p_rf=expert_preds[1],
                p_arf=expert_preds[2],
                p_nn=expert_preds[3],
                p_base=np.mean(expert_preds),
                reg_ctx={'phase': 0}
            )

            if meta_pred is not None:
                test_outcomes.append(outcome)
                test_meta_preds.append(meta_pred)
                test_expert_avg_preds.append(np.mean(expert_preds))

                # Статистика по паттернам
                pattern_stats[pattern_type]['total'] += 1
                pattern_stats[pattern_type]['meta_preds'].append(meta_pred)
                if (meta_pred > 0.5 and outcome == 1) or (meta_pred <= 0.5 and outcome == 0):
                    pattern_stats[pattern_type]['correct'] += 1

        # ================== АНАЛИЗ РЕЗУЛЬТАТОВ ==================
        print("\n" + "="*70)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА")
        print("="*70)

        # 1. Общая точность
        meta_eval = evaluate_predictions(test_outcomes, test_meta_preds)
        expert_eval = evaluate_predictions(test_outcomes, test_expert_avg_preds)

        print(f"\n📈 ОБЩАЯ ТОЧНОСТЬ:")
        print(f"  META Neural:     {meta_eval['accuracy']:.2f}%")
        print(f"  Среднее экспертов: {expert_eval['accuracy']:.2f}%")
        print(f"  Улучшение META:   {meta_eval['accuracy'] - expert_eval['accuracy']:+.2f}%")

        # 2. Калибровка
        print(f"\n🎯 КАЛИБРОВКА (ниже = лучше):")
        print(f"  META Neural:     {meta_eval['calibration_error']:.3f}")
        print(f"  Среднее экспертов: {expert_eval['calibration_error']:.3f}")

        # 3. Распознавание паттернов
        print(f"\n🔍 РАСПОЗНАВАНИЕ ПАТТЕРНОВ:")
        for pattern_type, stats in sorted(pattern_stats.items()):
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
                avg_pred = np.mean(stats['meta_preds'])

                pattern_names = {
                    'high_consensus_up': 'Высокое согласие (>0.7) → UP ожидается',
                    'low_consensus_down': 'Низкое согласие (<0.3) → DOWN ожидается',
                    'high_spread': 'Большой разброс (>0.4) → Случайность',
                    'other': 'Остальные случаи'
                }

                print(f"\n  {pattern_names.get(pattern_type, pattern_type)}:")
                print(f"    Примеров: {stats['total']}")
                print(f"    Точность: {acc:.1f}%")
                print(f"    Средняя вероятность META: {avg_pred:.3f}")

                # Оценка распознавания
                if pattern_type == 'high_consensus_up':
                    if avg_pred > 0.65:
                        print(f"    ✅ META РАСПОЗНАЛА: высокая вероятность UP")
                    else:
                        print(f"    ❌ META НЕ РАСПОЗНАЛА: вероятность слишком низкая")
                elif pattern_type == 'low_consensus_down':
                    if avg_pred < 0.35:
                        print(f"    ✅ META РАСПОЗНАЛА: низкая вероятность UP")
                    else:
                        print(f"    ❌ META НЕ РАСПОЗНАЛА: вероятность слишком высокая")
                elif pattern_type == 'high_spread':
                    if 0.45 < avg_pred < 0.55:
                        print(f"    ✅ META РАСПОЗНАЛА: предсказывает случайность (~0.5)")
                    else:
                        print(f"    ⚠️  META ЧАСТИЧНО РАСПОЗНАЛА: отклонение от 0.5")

        # 4. Итоговая оценка
        print("\n" + "="*70)
        print("ИТОГОВАЯ ОЦЕНКА СПОСОБНОСТИ НАХОДИТЬ ЗАКОНОМЕРНОСТИ")
        print("="*70)

        score = 0
        max_score = 3

        # Критерий 1: Точность > среднего экспертов
        if meta_eval['accuracy'] > expert_eval['accuracy']:
            score += 1
            print(f"✅ [1/3] META точнее среднего экспертов (+{meta_eval['accuracy'] - expert_eval['accuracy']:.2f}%)")
        else:
            print(f"❌ [0/3] META не точнее среднего экспертов")

        # Критерий 2: Распознает паттерн high_consensus_up
        if 'high_consensus_up' in pattern_stats and pattern_stats['high_consensus_up']['total'] > 5:
            avg_high = np.mean(pattern_stats['high_consensus_up']['meta_preds'])
            if avg_high > 0.65:
                score += 1
                print(f"✅ [2/3] META распознала паттерн 'высокое согласие → UP' (p={avg_high:.3f})")
            else:
                print(f"❌ [0/3] META не распознала паттерн 'высокое согласие → UP' (p={avg_high:.3f})")

        # Критерий 3: Распознает паттерн low_consensus_down
        if 'low_consensus_down' in pattern_stats and pattern_stats['low_consensus_down']['total'] > 5:
            avg_low = np.mean(pattern_stats['low_consensus_down']['meta_preds'])
            if avg_low < 0.35:
                score += 1
                print(f"✅ [3/3] META распознала паттерн 'низкое согласие → DOWN' (p={avg_low:.3f})")
            else:
                print(f"❌ [0/3] META не распознала паттерн 'низкое согласие → DOWN' (p={avg_low:.3f})")

        print(f"\n{'='*70}")
        print(f"ФИНАЛЬНАЯ ОЦЕНКА: {score}/{max_score} ({score/max_score*100:.0f}%)")
        print(f"{'='*70}")

        if score == max_score:
            print("🏆 ОТЛИЧНО: META успешно находит все скрытые закономерности!")
        elif score >= 2:
            print("✅ ХОРОШО: META находит большинство закономерностей")
        elif score >= 1:
            print("⚠️  УДОВЛЕТВОРИТЕЛЬНО: META находит некоторые закономерности")
        else:
            print("❌ ПЛОХО: META не находит закономерности в данных")

        return score == max_score

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    start = time.time()
    success = main()
    elapsed = time.time() - start

    print(f"\n⏱️  Время выполнения: {elapsed:.1f}s")
    sys.exit(0 if success else 1)
