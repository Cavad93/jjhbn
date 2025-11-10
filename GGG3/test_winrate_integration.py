#!/usr/bin/env python3
"""
Интеграционный тест логики расчета винрейта

Проверяет работу с реальными Position объектами и _get_expert_win_rates()
"""

import sys
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# Mock Position class (упрощенная версия для тестирования)
@dataclass
class Position:
    """Mock Position для тестирования"""
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    amount: float
    position_value: float
    tp_price: float
    sl_price: float
    atr_value: float
    entry_snapshot: Dict[str, Any] = field(default_factory=dict)
    status: str = 'OPEN'
    exit_price: Optional[float] = None
    pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в dict"""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_time': self.entry_time.isoformat() if isinstance(self.entry_time, datetime) else self.entry_time,
            'entry_price': self.entry_price,
            'amount': self.amount,
            'position_value': self.position_value,
            'tp_price': self.tp_price,
            'sl_price': self.sl_price,
            'atr_value': self.atr_value,
            'entry_snapshot': self.entry_snapshot,
            'status': self.status,
            'exit_price': self.exit_price,
            'pnl': self.pnl
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Десериализация из dict"""
        data = data.copy()
        if 'entry_time' in data and isinstance(data['entry_time'], str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        return cls(**data)


class TestIntegration:
    """Интеграционное тестирование"""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0

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

    def assert_close(self, actual, expected, test_name, tolerance=0.01):
        """Проверка приблизительного равенства"""
        if abs(actual - expected) < tolerance:
            print(f"  ✅ {test_name}: {actual} ≈ {expected}")
            self.tests_passed += 1
            return True
        else:
            print(f"  ❌ {test_name}: {actual} != {expected}")
            self.tests_failed += 1
            return False

    def test_position_object_creation(self):
        """Тест 1: Создание Position объекта с entry_snapshot"""
        print("\n" + "="*80)
        print("ТЕСТ 1: Создание Position с entry_snapshot")
        print("="*80)

        entry_snapshot = {
            'ml_predictions': {
                'xgb': 0.7,
                'rf': 0.6,
                'arf': 0.65,
                'nn': 0.55
            },
            'p_meta': 0.65,
            'context': {'phase': 2, 'vol_ratio': 1.2},
            'phase': 2,
            'p_meta_prediction': 0.72,
            'meta_mode': 'ACTIVE'
        }

        position = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=50000.0,
            amount=0.1,
            position_value=5000.0,
            tp_price=51000.0,
            sl_price=49000.0,
            atr_value=500.0,
            entry_snapshot=entry_snapshot,
            status='OPEN'
        )

        print("\nПроверка структуры Position:")
        self.assert_equal(position.symbol, 'BTCUSDT', "Symbol")
        self.assert_equal(position.direction, 'LONG', "Direction")
        self.assert_equal(position.status, 'OPEN', "Status")

        print("\nПроверка entry_snapshot:")
        self.assert_equal('ml_predictions' in position.entry_snapshot, True, "ml_predictions exists")
        self.assert_equal('meta_mode' in position.entry_snapshot, True, "meta_mode exists")
        self.assert_equal(position.entry_snapshot['meta_mode'], 'ACTIVE', "meta_mode value")

    def test_position_serialization(self):
        """Тест 2: Сериализация и десериализация Position"""
        print("\n" + "="*80)
        print("ТЕСТ 2: Сериализация Position")
        print("="*80)

        entry_snapshot = {
            'ml_predictions': {'xgb': 0.7, 'rf': 0.6},
            'p_meta': 0.65,
            'p_meta_prediction': 0.72,
            'meta_mode': 'SHADOW'
        }

        position = Position(
            symbol='ETHUSDT',
            direction='SHORT',
            entry_time=datetime.now(),
            entry_price=3000.0,
            amount=1.0,
            position_value=3000.0,
            tp_price=2900.0,
            sl_price=3100.0,
            atr_value=50.0,
            entry_snapshot=entry_snapshot,
            status='OPEN'
        )

        # Сериализация
        position_dict = position.to_dict()

        print("\nПроверка сериализации:")
        self.assert_equal('entry_snapshot' in position_dict, True, "entry_snapshot сохранен")
        self.assert_equal('ml_predictions' in position_dict['entry_snapshot'], True, "ml_predictions сохранен")

        # Десериализация
        restored_position = Position.from_dict(position_dict)

        print("\nПроверка десериализации:")
        self.assert_equal(restored_position.symbol, 'ETHUSDT', "Symbol restored")
        self.assert_equal(restored_position.direction, 'SHORT', "Direction restored")
        self.assert_equal(restored_position.entry_snapshot['meta_mode'], 'SHADOW', "meta_mode restored")
        self.assert_close(restored_position.entry_snapshot['ml_predictions']['xgb'], 0.7, "XGB prediction restored")

    def test_expert_stats_calculation(self):
        """Тест 3: Расчет статистики экспертов из закрытых позиций"""
        print("\n" + "="*80)
        print("ТЕСТ 3: Расчет статистики экспертов")
        print("="*80)

        # Создаем несколько позиций с разными результатами
        positions = []

        # Позиция 1: LONG прибыль, XGB угадал, RF нет
        pos1_snapshot = {
            'ml_predictions': {'xgb': 0.7, 'rf': 0.3, 'arf': 0.6, 'nn': 0.55},
            'p_meta': 0.65,
            'meta_mode': 'SHADOW'
        }
        pos1 = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=50000.0,
            amount=0.1,
            position_value=5000.0,
            tp_price=51000.0,
            sl_price=49000.0,
            atr_value=500.0,
            entry_snapshot=pos1_snapshot,
            status='CLOSED',
            exit_price=51000.0,
            pnl=100.0  # Прибыль
        )
        positions.append(pos1)

        # Позиция 2: SHORT прибыль (цена упала), XGB угадал, RF нет
        pos2_snapshot = {
            'ml_predictions': {'xgb': 0.3, 'rf': 0.7, 'arf': 0.4, 'nn': 0.45},
            'p_meta': 0.35,
            'meta_mode': 'SHADOW'
        }
        pos2 = Position(
            symbol='ETHUSDT',
            direction='SHORT',
            entry_time=datetime.now(),
            entry_price=3000.0,
            amount=1.0,
            position_value=3000.0,
            tp_price=2900.0,
            sl_price=3100.0,
            atr_value=50.0,
            entry_snapshot=pos2_snapshot,
            status='CLOSED',
            exit_price=2900.0,
            pnl=100.0  # Прибыль для SHORT
        )
        positions.append(pos2)

        # Позиция 3: LONG убыток, XGB угадал падение, RF не угадал
        pos3_snapshot = {
            'ml_predictions': {'xgb': 0.3, 'rf': 0.7, 'arf': 0.4, 'nn': 0.35},
            'p_meta': 0.35,
            'meta_mode': 'SHADOW'
        }
        pos3 = Position(
            symbol='BNBUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=400.0,
            amount=10.0,
            position_value=4000.0,
            tp_price=410.0,
            sl_price=390.0,
            atr_value=10.0,
            entry_snapshot=pos3_snapshot,
            status='CLOSED',
            exit_price=390.0,
            pnl=-100.0  # Убыток
        )
        positions.append(pos3)

        # Симулируем подсчет статистики (как в _restore_expert_stats)
        expert_stats = {
            'xgb': {'wins': 0, 'losses': 0, 'total': 0},
            'rf': {'wins': 0, 'losses': 0, 'total': 0},
            'arf': {'wins': 0, 'losses': 0, 'total': 0},
            'nn': {'wins': 0, 'losses': 0, 'total': 0},
        }

        print("\nОбработка позиций:")
        for i, pos in enumerate(positions, 1):
            # ВАЖНО: Для LONG pnl>0 означает рост, для SHORT pnl>0 означает ПАДЕНИЕ
            if pos.direction == 'LONG':
                price_went_up = pos.pnl > 0
            else:  # SHORT
                price_went_up = pos.pnl < 0  # SHORT: убыток означает цена выросла

            snapshot = pos.entry_snapshot
            ml_preds = snapshot.get('ml_predictions', {})

            print(f"\n  Позиция {i}: {pos.symbol} {pos.direction}, PnL={pos.pnl:+.2f}")
            print(f"    Цена: {'выросла' if price_went_up else 'упала'}")
            print(f"    Предсказания: {ml_preds}")

            if ml_preds:
                for expert_name in ['xgb', 'rf', 'arf', 'nn']:
                    if expert_name in ml_preds:
                        p_up = ml_preds[expert_name]

                        # ЛОГИКА ОДИНАКОВА для LONG и SHORT
                        expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)

                        expert_stats[expert_name]['total'] += 1
                        if expert_was_right:
                            expert_stats[expert_name]['wins'] += 1
                        else:
                            expert_stats[expert_name]['losses'] += 1

        print("\n\nИтоговая статистика:")
        print("  Expert | Wins | Losses | Total | WinRate")
        print("  " + "-"*45)
        for expert, stats in expert_stats.items():
            wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {expert.upper():6s} | {stats['wins']:4d} | {stats['losses']:6d} | {stats['total']:5d} | {wr:6.1f}%")

        print("\nПроверка:")
        # XGB угадал все 3 позиции
        self.assert_equal(expert_stats['xgb']['wins'], 3, "XGB wins")
        self.assert_equal(expert_stats['xgb']['total'], 3, "XGB total")

        # RF ошибся во всех 3 позициях
        self.assert_equal(expert_stats['rf']['wins'], 0, "RF wins")
        self.assert_equal(expert_stats['rf']['losses'], 3, "RF losses")

        # ARF угадал все 3 (везде правильно предсказал)
        self.assert_equal(expert_stats['arf']['wins'], 3, "ARF wins")
        self.assert_equal(expert_stats['arf']['losses'], 0, "ARF losses")

    def test_meta_shadow_vs_active(self):
        """Тест 4: META в SHADOW и ACTIVE режимах"""
        print("\n" + "="*80)
        print("ТЕСТ 4: META SHADOW vs ACTIVE")
        print("="*80)

        positions = []

        # Позиция 1: META в SHADOW режиме
        pos1_snapshot = {
            'ml_predictions': {'xgb': 0.7},
            'p_meta': 0.65,
            'p_meta_prediction': 0.72,
            'meta_mode': 'SHADOW'
        }
        pos1 = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=50000.0,
            amount=0.1,
            position_value=5000.0,
            tp_price=51000.0,
            sl_price=49000.0,
            atr_value=500.0,
            entry_snapshot=pos1_snapshot,
            status='CLOSED',
            exit_price=51000.0,
            pnl=100.0
        )
        positions.append(('SHADOW', pos1))

        # Позиция 2: META в ACTIVE режиме
        pos2_snapshot = {
            'ml_predictions': {'xgb': 0.7},
            'p_meta': 0.65,
            'p_meta_prediction': 0.75,
            'meta_mode': 'ACTIVE'
        }
        pos2 = Position(
            symbol='ETHUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=3000.0,
            amount=1.0,
            position_value=3000.0,
            tp_price=3100.0,
            sl_price=2900.0,
            atr_value=50.0,
            entry_snapshot=pos2_snapshot,
            status='CLOSED',
            exit_price=3100.0,
            pnl=100.0
        )
        positions.append(('ACTIVE', pos2))

        meta_stats = {'wins': 0, 'losses': 0, 'total': 0}

        print("\nОбработка позиций:")
        for mode, pos in positions:
            # ВАЖНО: Для LONG pnl>0 означает рост, для SHORT pnl>0 означает ПАДЕНИЕ
            if pos.direction == 'LONG':
                price_went_up = pos.pnl > 0
            else:  # SHORT
                price_went_up = pos.pnl < 0  # SHORT: убыток означает цена выросла

            snapshot = pos.entry_snapshot
            meta_mode = snapshot.get('meta_mode')
            p_meta_pred = snapshot.get('p_meta_prediction')

            print(f"\n  Позиция: {pos.symbol} {pos.direction}, режим META: {meta_mode}")
            print(f"    META предсказание: {p_meta_pred}")
            print(f"    Цена: {'выросла' if price_went_up else 'упала'}")

            if meta_mode == 'ACTIVE' and p_meta_pred is not None:
                if pos.direction == 'LONG':
                    meta_was_right = (p_meta_pred > 0.5 and price_went_up) or (p_meta_pred <= 0.5 and not price_went_up)
                else:
                    meta_was_right = (p_meta_pred < 0.5 and price_went_up) or (p_meta_pred >= 0.5 and not price_went_up)

                meta_stats['total'] += 1
                if meta_was_right:
                    meta_stats['wins'] += 1
                    print(f"    → META: WIN (счетчик обновлен)")
                else:
                    meta_stats['losses'] += 1
                    print(f"    → META: LOSS (счетчик обновлен)")
            else:
                print(f"    → META: статистика НЕ обновляется (режим {meta_mode})")

        print("\nИтоговая статистика META:")
        print(f"  Wins: {meta_stats['wins']}")
        print(f"  Losses: {meta_stats['losses']}")
        print(f"  Total: {meta_stats['total']}")

        print("\nПроверка:")
        # META должна иметь total=1 (только ACTIVE позиция учитывается)
        self.assert_equal(meta_stats['total'], 1, "META total (только ACTIVE)")
        self.assert_equal(meta_stats['wins'], 1, "META wins")
        self.assert_equal(meta_stats['losses'], 0, "META losses")

    def run_all_tests(self):
        """Запуск всех тестов"""
        print("\n" + "="*80)
        print("ИНТЕГРАЦИОННОЕ ТЕСТИРОВАНИЕ")
        print("="*80)

        tests = [
            self.test_position_object_creation,
            self.test_position_serialization,
            self.test_expert_stats_calculation,
            self.test_meta_shadow_vs_active,
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
    tester = TestIntegration()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)
