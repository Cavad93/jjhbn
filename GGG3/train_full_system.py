#!/usr/bin/env python3
"""
Полное обучение системы: 4 эксперта + META Neural на реальных данных

СИМУЛЯЦИЯ ПОЛНОГО ФУНКЦИОНАЛА БОТА:
1. Обучение XGBoost на исторических данных
2. Обучение RandomForest
3. Обучение ARF (Adaptive Random Forest)
4. Обучение NN (Neural Network)
5. Сбор предсказаний всех экспертов
6. Обучение META Neural на предсказаниях экспертов
7. Оценка качества всей системы
"""
import sys
import os
import json
import time
import tempfile
import shutil
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Импорт META Neural
from meta_neural_cem import MetaNeuralCEM

class SimpleConfig:
    """Конфигурация для META (по умолчанию из бота)"""
    def __init__(self):
        self.meta_nn_hidden = None  # Используется стандартная [64,32,16]
        self.meta_retrain_every = 200  # Переобучение каждые 200 примеров
        self.meta_min_train = 300  # Минимум для первого обучения
        self.meta_use_cma_es = False  # CEM быстрее
        self.meta_cem_iters = 20
        self.meta_cem_pop = 15
        self.meta_bootstrap_reps = 3
        self.meta_dropout_rate = 0.1
        self.meta_state_path = None
        self.meta_exp4_phases = 6
        self.meta_mc_n_inference = 10

class SimpleExpert:
    """Упрощенная симуляция эксперта (XGBoost, RF, ARF, NN)"""
    def __init__(self, name, bias=0.0, noise=0.1):
        self.name = name
        self.bias = bias  # Смещение предсказаний
        self.noise = noise  # Уровень шума
        self.history = []

    def predict(self, features, true_outcome=None):
        """
        Симуляция предсказания эксперта
        Использует реальные фичи + добавляет свой паттерн
        """
        # Базовое предсказание на основе momentum
        base_pred = 0.5

        if 'momentum_10m' in features:
            # Эксперты реагируют на momentum
            momentum = features['momentum_10m']
            base_pred = 0.5 + momentum * 10  # Усиливаем сигнал

        if 'rsi' in features:
            # RSI влияние
            rsi = features['rsi']
            if rsi > 70:  # Перекупленность
                base_pred -= 0.1
            elif rsi < 30:  # Перепроданность
                base_pred += 0.1

        # Добавляем bias эксперта
        pred = base_pred + self.bias

        # Добавляем шум
        pred += np.random.randn() * self.noise

        # Ограничиваем [0.1, 0.9]
        pred = np.clip(pred, 0.1, 0.9)

        # Сохраняем историю для "обучения"
        if true_outcome is not None:
            self.history.append({
                'prediction': pred,
                'outcome': true_outcome,
                'error': abs(pred - true_outcome)
            })

            # Адаптация (упрощенная)
            if len(self.history) > 100:
                recent_errors = [h['error'] for h in self.history[-100:]]
                avg_error = np.mean(recent_errors)

                # Если ошибка большая - снижаем bias
                if avg_error > 0.4:
                    self.bias *= 0.99

        return float(pred)

    def get_accuracy(self):
        """Точность эксперта"""
        if len(self.history) < 10:
            return 0.5

        correct = sum(1 for h in self.history
                     if (h['prediction'] > 0.5 and h['outcome'] == 1) or
                        (h['prediction'] <= 0.5 and h['outcome'] == 0))
        return correct / len(self.history)

def load_data(filename='realistic_synthetic_60k.json'):
    """Загрузка данных"""
    print(f"📂 Загрузка данных из {filename}...")

    with open(filename, 'r') as f:
        data = json.load(f)

    rounds = data['rounds']
    print(f"✅ Загружено {len(rounds):,} раундов")

    return rounds

def train_full_system(rounds, test_split=0.2):
    """Полное обучение системы"""
    print(f"\n{'='*70}")
    print("ПОЛНОЕ ОБУЧЕНИЕ СИСТЕМЫ: 4 ЭКСПЕРТА + META NEURAL")
    print(f"{'='*70}\n")

    # Разделение на train/test
    split_idx = int(len(rounds) * (1 - test_split))
    train_rounds = rounds[:split_idx]
    test_rounds = rounds[split_idx:]

    print(f"📊 Train: {len(train_rounds):,} раундов")
    print(f"📊 Test:  {len(test_rounds):,} раундов\n")

    # ========== ШАГ 1: СОЗДАНИЕ ЭКСПЕРТОВ ==========
    print(f"{'='*70}")
    print("ШАГ 1: ИНИЦИАЛИЗАЦИЯ 4 ЭКСПЕРТОВ")
    print(f"{'='*70}\n")

    experts = {
        'XGBoost': SimpleExpert('XGBoost', bias=0.02, noise=0.08),
        'RandomForest': SimpleExpert('RandomForest', bias=-0.01, noise=0.10),
        'ARF': SimpleExpert('ARF', bias=0.00, noise=0.12),
        'NN': SimpleExpert('NN', bias=0.01, noise=0.09)
    }

    for name, expert in experts.items():
        print(f"✅ {name:15} инициализирован (bias={expert.bias:+.2f}, noise={expert.noise:.2f})")

    # ========== ШАГ 2: ОБУЧЕНИЕ ЭКСПЕРТОВ ==========
    print(f"\n{'='*70}")
    print("ШАГ 2: ОБУЧЕНИЕ ЭКСПЕРТОВ НА TRAIN DATA")
    print(f"{'='*70}\n")

    print(f"Обработка {len(train_rounds):,} обучающих раундов...")

    for i, round_data in enumerate(train_rounds):
        features = round_data['features']
        outcome = round_data['outcome']

        # Каждый эксперт делает предсказание и "учится"
        for expert in experts.values():
            expert.predict(features, true_outcome=outcome)

        if (i + 1) % 500 == 0:
            print(f"  Обработано: {i+1:,}/{len(train_rounds):,}", end='\r')

    print(f"\n\n✅ Эксперты обучены!")

    # Статистика экспертов
    print(f"\n📊 ТОЧНОСТЬ ЭКСПЕРТОВ НА TRAIN:")
    for name, expert in experts.items():
        acc = expert.get_accuracy() * 100
        print(f"  {name:15}: {acc:.1f}%")

    # ========== ШАГ 3: ОБУЧЕНИЕ META NEURAL ==========
    print(f"\n{'='*70}")
    print("ШАГ 3: ОБУЧЕНИЕ META NEURAL НА ПРЕДСКАЗАНИЯХ ЭКСПЕРТОВ")
    print(f"{'='*70}\n")

    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "meta_state.json")
        meta = MetaNeuralCEM(cfg)

        print(f"META Neural инициализирована")
        print(f"Параметров сети: {len(meta.networks[0].get_weights_flat()):,}")
        print(f"Минимум примеров для обучения: {cfg.meta_min_train:,}")
        print(f"Частота переобучения: каждые {cfg.meta_retrain_every:,} примеров\n")

        print(f"Обработка {len(train_rounds):,} раундов...")

        for i, round_data in enumerate(train_rounds):
            features = round_data['features']
            outcome = round_data['outcome']
            phase = round_data['phase']

            # Получаем предсказания всех экспертов
            p_xgb = experts['XGBoost'].predict(features)
            p_rf = experts['RandomForest'].predict(features)
            p_arf = experts['ARF'].predict(features)
            p_nn = experts['NN'].predict(features)
            p_base = np.mean([p_xgb, p_rf, p_arf, p_nn])

            # META учится на предсказаниях экспертов
            meta.record_result(
                p_xgb=p_xgb,
                p_rf=p_rf,
                p_arf=p_arf,
                p_nn=p_nn,
                p_base=p_base,
                y_up=outcome,
                used_in_live=True,
                reg_ctx={'phase': phase}
            )

            if (i + 1) % 500 == 0:
                print(f"  Обработано: {i+1:,}/{len(train_rounds):,}", end='\r')

        print(f"\n\n✅ META Neural обучена!")
        print(f"   Примеров в фазе 0: {meta.seen_ph[0]:,}")
        print(f"   Счетчик обучения: {meta.new_since_train_ph[0]:,}")

        # ========== ШАГ 4: ТЕСТИРОВАНИЕ ==========
        print(f"\n{'='*70}")
        print("ШАГ 4: ТЕСТИРОВАНИЕ НА TEST DATA")
        print(f"{'='*70}\n")

        meta.mode = "ACTIVE"  # Включаем предсказания

        # Результаты по фазам
        results_by_phase = defaultdict(lambda: {
            'expert_avg': [],
            'meta': [],
            'outcomes': []
        })

        print(f"Тестирование на {len(test_rounds):,} раундах...")

        for i, round_data in enumerate(test_rounds):
            features = round_data['features']
            outcome = round_data['outcome']
            phase = round_data['phase']

            # Предсказания экспертов
            p_xgb = experts['XGBoost'].predict(features)
            p_rf = experts['RandomForest'].predict(features)
            p_arf = experts['ARF'].predict(features)
            p_nn = experts['NN'].predict(features)
            p_base = np.mean([p_xgb, p_rf, p_arf, p_nn])

            # Предсказание META
            p_meta = meta.predict(
                p_xgb=p_xgb,
                p_rf=p_rf,
                p_arf=p_arf,
                p_nn=p_nn,
                p_base=p_base,
                reg_ctx={'phase': phase}
            )

            if p_meta is not None:
                results_by_phase[phase]['expert_avg'].append(p_base)
                results_by_phase[phase]['meta'].append(p_meta)
                results_by_phase[phase]['outcomes'].append(outcome)

            if (i + 1) % 200 == 0:
                print(f"  Протестировано: {i+1:,}/{len(test_rounds):,}", end='\r')

        print(f"\n\n✅ Тестирование завершено!")

        # ========== ШАГ 5: АНАЛИЗ РЕЗУЛЬТАТОВ ==========
        print(f"\n{'='*70}")
        print("ШАГ 5: АНАЛИЗ РЕЗУЛЬТАТОВ")
        print(f"{'='*70}\n")

        total_meta_correct = 0
        total_expert_correct = 0
        total_count = 0

        print(f"📊 РЕЗУЛЬТАТЫ ПО ФАЗАМ:")
        for phase in sorted(results_by_phase.keys()):
            data = results_by_phase[phase]

            if len(data['outcomes']) < 10:
                continue

            # Точность среднего экспертов
            expert_correct = sum(1 for i in range(len(data['outcomes']))
                               if (data['expert_avg'][i] > 0.5) == data['outcomes'][i])
            expert_acc = expert_correct / len(data['outcomes']) * 100

            # Точность META
            meta_correct = sum(1 for i in range(len(data['outcomes']))
                             if (data['meta'][i] > 0.5) == data['outcomes'][i])
            meta_acc = meta_correct / len(data['outcomes']) * 100

            improvement = meta_acc - expert_acc

            total_meta_correct += meta_correct
            total_expert_correct += expert_correct
            total_count += len(data['outcomes'])

            status = "✅" if improvement > 0 else "❌"
            print(f"\n  Фаза {phase} ({len(data['outcomes']):,} примеров):")
            print(f"    Среднее экспертов: {expert_acc:.1f}%")
            print(f"    META Neural:       {meta_acc:.1f}%")
            print(f"    {status} Улучшение:       {improvement:+.1f}%")

        # Общая статистика
        if total_count > 0:
            total_expert_acc = total_expert_correct / total_count * 100
            total_meta_acc = total_meta_correct / total_count * 100
            total_improvement = total_meta_acc - total_expert_acc

            print(f"\n{'='*70}")
            print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
            print(f"{'='*70}\n")
            print(f"  Всего примеров: {total_count:,}")
            print(f"  Среднее экспертов: {total_expert_acc:.1f}%")
            print(f"  META Neural:       {total_meta_acc:.1f}%")
            print(f"  Улучшение:         {total_improvement:+.1f}%")

            # Расчет соотношения данные/параметры
            n_params = 5189  # Известное количество параметров META Neural
            data_ratio = len(train_rounds) / n_params

            if total_improvement > 5:
                print(f"\n  🎉 ОТЛИЧНО! META улучшает результаты на {total_improvement:.1f}%")
                print(f"  📊 Соотношение данные/параметры: {data_ratio:.1f}x")
            elif total_improvement > 0:
                print(f"\n  ✅ ХОРОШО! META немного улучшает результаты (+{total_improvement:.1f}%)")
                print(f"  📊 Соотношение данные/параметры: {data_ratio:.1f}x")
            else:
                print(f"\n  ⚠️  META не улучшает результаты")
                print(f"  📊 Соотношение данные/параметры: {data_ratio:.1f}x")
                if data_ratio < 10:
                    print(f"  ❌ КРИТИЧНО: нужно минимум {n_params*10:,} примеров (сейчас {len(train_rounds):,})")
                elif data_ratio < 20:
                    print(f"  ⚠️  МАЛО: рекомендуется {n_params*20:,} примеров (сейчас {len(train_rounds):,})")
                else:
                    print(f"  💡 Данных достаточно, проблема может быть в другом")

            return {
                'total_examples': total_count,
                'expert_acc': total_expert_acc,
                'meta_acc': total_meta_acc,
                'improvement': total_improvement,
                'train_size': len(train_rounds)
            }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def main():
    """Главная функция"""
    try:
        # Загрузка данных
        rounds = load_data('realistic_synthetic_60k.json')

        # Полное обучение
        results = train_full_system(rounds, test_split=0.2)

        print(f"\n{'='*70}")
        print("ЗАВЕРШЕНО УСПЕШНО!")
        print(f"{'='*70}\n")

        return results

    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start = time.time()
    results = main()
    elapsed = time.time() - start

    print(f"⏱️  Общее время: {elapsed/60:.1f} минут")
