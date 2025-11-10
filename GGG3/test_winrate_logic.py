#!/usr/bin/env python3
"""
Тест логики расчета винрейта для экспертов и META

Проверяет:
1. Правильность оценки экспертов по их предсказаниям
2. Различные сценарии LONG/SHORT позиций
3. Режимы META (SHADOW/ACTIVE)
"""

import sys
from datetime import datetime
from typing import Dict


class TestWinRateLogic:
    """Тестирование логики расчета винрейта"""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.expert_stats = {
            'base': {'wins': 0, 'losses': 0, 'total': 0},
            'xgb': {'wins': 0, 'losses': 0, 'total': 0},
            'rf': {'wins': 0, 'losses': 0, 'total': 0},
            'arf': {'wins': 0, 'losses': 0, 'total': 0},
            'nn': {'wins': 0, 'losses': 0, 'total': 0},
            'meta': {'wins': 0, 'losses': 0, 'total': 0}
        }

    def reset_stats(self):
        """Сброс статистики"""
        for expert in self.expert_stats:
            self.expert_stats[expert] = {'wins': 0, 'losses': 0, 'total': 0}

    def assert_equal(self, actual, expected, test_name):
        """Проверка равенства"""
        if actual == expected:
            print(f"  ✅ {test_name}: {actual} == {expected}")
            self.tests_passed += 1
            return True
        else:
            print(f"  ❌ {test_name}: {actual} != {expected}")
            self.tests_failed += 1
            return False

    def simulate_ml_expert_evaluation(self, direction: str, ml_preds: Dict[str, float], price_went_up: bool):
        """
        Симуляция оценки ML экспертов (копия из binance_bot_main.py)

        Args:
            direction: 'LONG' или 'SHORT'
            ml_preds: {'xgb': 0.7, 'rf': 0.3, ...}
            price_went_up: True если цена выросла
        """
        for expert_name in ['xgb', 'rf', 'arf', 'nn']:
            if expert_name in ml_preds and expert_name in self.expert_stats:
                p_up = ml_preds[expert_name]

                # Логика из binance_bot_main.py:1338-1346
                if direction == 'LONG':
                    expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)
                else:  # SHORT
                    expert_was_right = (p_up < 0.5 and price_went_up) or (p_up >= 0.5 and not price_went_up)

                self.expert_stats[expert_name]['total'] += 1
                if expert_was_right:
                    self.expert_stats[expert_name]['wins'] += 1
                else:
                    self.expert_stats[expert_name]['losses'] += 1

    def simulate_base_evaluation(self, direction: str, p_base: float, price_went_up: bool):
        """
        Симуляция оценки BASE логики

        Args:
            direction: 'LONG' или 'SHORT'
            p_base: Предсказание BASE (0-1)
            price_went_up: True если цена выросла
        """
        if direction == 'LONG':
            base_was_right = (p_base > 0.5 and price_went_up) or (p_base <= 0.5 and not price_went_up)
        else:  # SHORT
            base_was_right = (p_base < 0.5 and price_went_up) or (p_base >= 0.5 and not price_went_up)

        self.expert_stats['base']['total'] += 1
        if base_was_right:
            self.expert_stats['base']['wins'] += 1
        else:
            self.expert_stats['base']['losses'] += 1

    def simulate_meta_evaluation(self, direction: str, p_meta: float, price_went_up: bool,
                                 meta_mode: str):
        """
        Симуляция оценки META

        Args:
            direction: 'LONG' или 'SHORT'
            p_meta: Предсказание META (0-1)
            price_went_up: True если цена выросла
            meta_mode: 'SHADOW' или 'ACTIVE'
        """
        if meta_mode == 'ACTIVE' and p_meta is not None:
            if direction == 'LONG':
                meta_was_right = (p_meta > 0.5 and price_went_up) or (p_meta <= 0.5 and not price_went_up)
            else:  # SHORT
                meta_was_right = (p_meta < 0.5 and price_went_up) or (p_meta >= 0.5 and not price_went_up)

            self.expert_stats['meta']['total'] += 1
            if meta_was_right:
                self.expert_stats['meta']['wins'] += 1
            else:
                self.expert_stats['meta']['losses'] += 1
        # В SHADOW режиме не обновляем статистику

    def test_case_1_long_position_profit(self):
        """
        Тест 1: LONG позиция в прибыли

        Сценарий:
        - Позиция: LONG
        - Цена выросла (PnL > 0)
        - XGB предсказал рост (0.7)
        - RF предсказал падение (0.3)
        - ARF предсказал рост (0.6)
        - NN предсказал падение (0.4)

        Ожидается:
        - XGB: WIN (правильно предсказал рост)
        - RF: LOSS (неправильно предсказал падение)
        - ARF: WIN (правильно предсказал рост)
        - NN: LOSS (неправильно предсказал падение)
        """
        print("\n" + "="*80)
        print("ТЕСТ 1: LONG позиция в прибыли (цена выросла)")
        print("="*80)

        self.reset_stats()

        ml_preds = {
            'xgb': 0.7,   # Предсказал рост
            'rf': 0.3,    # Предсказал падение
            'arf': 0.6,   # Предсказал рост
            'nn': 0.4     # Предсказал падение
        }

        print("\nВходные данные:")
        print(f"  Направление: LONG")
        print(f"  Цена: выросла (PnL > 0)")
        print(f"  Предсказания: XGB=0.7, RF=0.3, ARF=0.6, NN=0.4")

        self.simulate_ml_expert_evaluation('LONG', ml_preds, price_went_up=True)

        print("\nРезультаты:")
        self.assert_equal(self.expert_stats['xgb']['wins'], 1, "XGB wins")
        self.assert_equal(self.expert_stats['xgb']['losses'], 0, "XGB losses")

        self.assert_equal(self.expert_stats['rf']['wins'], 0, "RF wins")
        self.assert_equal(self.expert_stats['rf']['losses'], 1, "RF losses")

        self.assert_equal(self.expert_stats['arf']['wins'], 1, "ARF wins")
        self.assert_equal(self.expert_stats['arf']['losses'], 0, "ARF losses")

        self.assert_equal(self.expert_stats['nn']['wins'], 0, "NN wins")
        self.assert_equal(self.expert_stats['nn']['losses'], 1, "NN losses")

    def test_case_2_short_position_profit(self):
        """
        Тест 2: SHORT позиция в прибыли

        Сценарий:
        - Позиция: SHORT
        - Цена упала (PnL > 0 для SHORT)
        - XGB предсказал падение (0.3)
        - RF предсказал рост (0.7)
        - ARF предсказал падение (0.4)
        - NN предсказал рост (0.6)

        Ожидается:
        - XGB: WIN (правильно предсказал падение для SHORT)
        - RF: LOSS (неправильно предсказал рост)
        - ARF: WIN (правильно предсказал падение)
        - NN: LOSS (неправильно предсказал рост)
        """
        print("\n" + "="*80)
        print("ТЕСТ 2: SHORT позиция в прибыли (цена упала)")
        print("="*80)

        self.reset_stats()

        ml_preds = {
            'xgb': 0.3,   # Предсказал падение
            'rf': 0.7,    # Предсказал рост
            'arf': 0.4,   # Предсказал падение
            'nn': 0.6     # Предсказал рост
        }

        print("\nВходные данные:")
        print(f"  Направление: SHORT")
        print(f"  Цена: упала (PnL > 0 для SHORT)")
        print(f"  Предсказания: XGB=0.3, RF=0.7, ARF=0.4, NN=0.6")

        self.simulate_ml_expert_evaluation('SHORT', ml_preds, price_went_up=True)

        print("\nРезультаты:")
        self.assert_equal(self.expert_stats['xgb']['wins'], 1, "XGB wins")
        self.assert_equal(self.expert_stats['xgb']['losses'], 0, "XGB losses")

        self.assert_equal(self.expert_stats['rf']['wins'], 0, "RF wins")
        self.assert_equal(self.expert_stats['rf']['losses'], 1, "RF losses")

        self.assert_equal(self.expert_stats['arf']['wins'], 1, "ARF wins")
        self.assert_equal(self.expert_stats['arf']['losses'], 0, "ARF losses")

        self.assert_equal(self.expert_stats['nn']['wins'], 0, "NN wins")
        self.assert_equal(self.expert_stats['nn']['losses'], 1, "NN losses")

    def test_case_3_long_position_loss(self):
        """
        Тест 3: LONG позиция в убытке

        Сценарий:
        - Позиция: LONG
        - Цена упала (PnL < 0)
        - XGB предсказал падение (0.3)
        - RF предсказал рост (0.7)

        Ожидается:
        - XGB: WIN (правильно предсказал падение)
        - RF: LOSS (неправильно предсказал рост)
        """
        print("\n" + "="*80)
        print("ТЕСТ 3: LONG позиция в убытке (цена упала)")
        print("="*80)

        self.reset_stats()

        ml_preds = {
            'xgb': 0.3,   # Предсказал падение
            'rf': 0.7,    # Предсказал рост
        }

        print("\nВходные данные:")
        print(f"  Направление: LONG")
        print(f"  Цена: упала (PnL < 0)")
        print(f"  Предсказания: XGB=0.3, RF=0.7")

        self.simulate_ml_expert_evaluation('LONG', ml_preds, price_went_up=False)

        print("\nРезультаты:")
        self.assert_equal(self.expert_stats['xgb']['wins'], 1, "XGB wins")
        self.assert_equal(self.expert_stats['xgb']['losses'], 0, "XGB losses")

        self.assert_equal(self.expert_stats['rf']['wins'], 0, "RF wins")
        self.assert_equal(self.expert_stats['rf']['losses'], 1, "RF losses")

    def test_case_4_base_logic(self):
        """
        Тест 4: BASE логика

        Сценарий:
        - Позиция: LONG
        - Цена выросла
        - BASE предсказал рост (0.65)

        Ожидается:
        - BASE: WIN
        """
        print("\n" + "="*80)
        print("ТЕСТ 4: BASE логика")
        print("="*80)

        self.reset_stats()

        print("\nВходные данные:")
        print(f"  Направление: LONG")
        print(f"  Цена: выросла")
        print(f"  BASE предсказание: 0.65")

        self.simulate_base_evaluation('LONG', p_base=0.65, price_went_up=True)

        print("\nРезультаты:")
        self.assert_equal(self.expert_stats['base']['wins'], 1, "BASE wins")
        self.assert_equal(self.expert_stats['base']['losses'], 0, "BASE losses")
        self.assert_equal(self.expert_stats['base']['total'], 1, "BASE total")

    def test_case_5_meta_shadow_mode(self):
        """
        Тест 5: META в SHADOW режиме

        Сценарий:
        - META в SHADOW режиме
        - META предсказала рост (0.75)
        - Цена выросла

        Ожидается:
        - META: total = 0 (не обновляется в SHADOW)
        """
        print("\n" + "="*80)
        print("ТЕСТ 5: META в SHADOW режиме")
        print("="*80)

        self.reset_stats()

        print("\nВходные данные:")
        print(f"  META режим: SHADOW")
        print(f"  Направление: LONG")
        print(f"  Цена: выросла")
        print(f"  META предсказание: 0.75")

        self.simulate_meta_evaluation('LONG', p_meta=0.75, price_went_up=True, meta_mode='SHADOW')

        print("\nРезультаты:")
        self.assert_equal(self.expert_stats['meta']['total'], 0, "META total (должен быть 0 в SHADOW)")
        self.assert_equal(self.expert_stats['meta']['wins'], 0, "META wins")
        self.assert_equal(self.expert_stats['meta']['losses'], 0, "META losses")

    def test_case_6_meta_active_mode(self):
        """
        Тест 6: META в ACTIVE режиме

        Сценарий:
        - META в ACTIVE режиме
        - META предсказала рост (0.75)
        - Цена выросла

        Ожидается:
        - META: WIN (правильно предсказала)
        """
        print("\n" + "="*80)
        print("ТЕСТ 6: META в ACTIVE режиме")
        print("="*80)

        self.reset_stats()

        print("\nВходные данные:")
        print(f"  META режим: ACTIVE")
        print(f"  Направление: LONG")
        print(f"  Цена: выросла")
        print(f"  META предсказание: 0.75")

        self.simulate_meta_evaluation('LONG', p_meta=0.75, price_went_up=True, meta_mode='ACTIVE')

        print("\nРезультаты:")
        self.assert_equal(self.expert_stats['meta']['wins'], 1, "META wins")
        self.assert_equal(self.expert_stats['meta']['losses'], 0, "META losses")
        self.assert_equal(self.expert_stats['meta']['total'], 1, "META total")

    def test_case_7_mixed_predictions(self):
        """
        Тест 7: Смешанные предсказания (реалистичный сценарий)

        Сценарий: Несколько позиций с разными результатами
        """
        print("\n" + "="*80)
        print("ТЕСТ 7: Смешанные предсказания (несколько позиций)")
        print("="*80)

        self.reset_stats()

        positions = [
            # (direction, ml_preds, price_went_up, description)
            ('LONG', {'xgb': 0.7, 'rf': 0.6, 'arf': 0.8, 'nn': 0.5}, True, "LONG прибыль, все угадали кроме NN"),
            ('LONG', {'xgb': 0.3, 'rf': 0.4, 'arf': 0.2, 'nn': 0.6}, False, "LONG убыток, все угадали кроме NN"),
            ('SHORT', {'xgb': 0.2, 'rf': 0.3, 'arf': 0.4, 'nn': 0.7}, True, "SHORT прибыль, все угадали кроме NN"),
        ]

        print("\nСимуляция 3 позиций:")
        for i, (direction, ml_preds, price_went_up, desc) in enumerate(positions, 1):
            print(f"\n  Позиция {i}: {desc}")
            print(f"    Предсказания: {ml_preds}")
            self.simulate_ml_expert_evaluation(direction, ml_preds, price_went_up)

        print("\n\nИтоговая статистика:")
        print("  Expert   | Wins | Losses | Total | WinRate")
        print("  " + "-"*50)
        for expert in ['xgb', 'rf', 'arf', 'nn']:
            stats = self.expert_stats[expert]
            total = stats['total']
            wins = stats['wins']
            wr = wins / total * 100 if total > 0 else 0
            print(f"  {expert.upper():8s} | {wins:4d} | {stats['losses']:6d} | {total:5d} | {wr:6.1f}%")

        print("\nПроверка:")
        # Все эксперты должны иметь разные винрейты
        self.assert_equal(self.expert_stats['xgb']['wins'], 3, "XGB все угадал")
        self.assert_equal(self.expert_stats['rf']['wins'], 3, "RF все угадал")
        self.assert_equal(self.expert_stats['arf']['wins'], 3, "ARF все угадал")
        self.assert_equal(self.expert_stats['nn']['wins'], 0, "NN все ошибся")

    def test_case_8_edge_case_p50(self):
        """
        Тест 8: Граничные случаи (p = 0.5)

        Сценарий:
        - Предсказание ровно 0.5 (нейтральное)
        - Проверяем логику <= и >
        """
        print("\n" + "="*80)
        print("ТЕСТ 8: Граничные случаи (p = 0.5)")
        print("="*80)

        self.reset_stats()

        print("\nВходные данные:")
        print(f"  Направление: LONG")
        print(f"  Цена: выросла")
        print(f"  XGB предсказание: 0.5 (нейтральное)")

        # p_up = 0.5, LONG, цена выросла
        # Логика: (p_up > 0.5 and price_went_up) -> (0.5 > 0.5 and True) -> False
        # Логика: (p_up <= 0.5 and not price_went_up) -> (0.5 <= 0.5 and False) -> False
        # Итого: LOSS

        ml_preds = {'xgb': 0.5}
        self.simulate_ml_expert_evaluation('LONG', ml_preds, price_went_up=True)

        print("\nРезультаты (p=0.5 считается как падение):")
        self.assert_equal(self.expert_stats['xgb']['wins'], 0, "XGB wins (0.5 не считается ростом)")
        self.assert_equal(self.expert_stats['xgb']['losses'], 1, "XGB losses")

    def run_all_tests(self):
        """Запуск всех тестов"""
        print("\n" + "="*80)
        print("ТЕСТИРОВАНИЕ ЛОГИКИ РАСЧЕТА ВИНРЕЙТА")
        print("="*80)

        tests = [
            self.test_case_1_long_position_profit,
            self.test_case_2_short_position_profit,
            self.test_case_3_long_position_loss,
            self.test_case_4_base_logic,
            self.test_case_5_meta_shadow_mode,
            self.test_case_6_meta_active_mode,
            self.test_case_7_mixed_predictions,
            self.test_case_8_edge_case_p50,
        ]

        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"\n  ❌ ТЕСТ УПАЛ С ОШИБКОЙ: {e}")
                import traceback
                traceback.print_exc()
                self.tests_failed += 1

        # Итоговая сводка
        print("\n" + "="*80)
        print("ИТОГОВАЯ СВОДКА")
        print("="*80)
        print(f"  ✅ Тестов пройдено: {self.tests_passed}")
        print(f"  ❌ Тестов провалено: {self.tests_failed}")
        print(f"  📊 Всего тестов: {self.tests_passed + self.tests_failed}")

        if self.tests_failed == 0:
            print("\n  🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
            return 0
        else:
            print(f"\n  ⚠️  ЕСТЬ ПРОВАЛЕННЫЕ ТЕСТЫ: {self.tests_failed}")
            return 1


if __name__ == '__main__':
    tester = TestWinRateLogic()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)
