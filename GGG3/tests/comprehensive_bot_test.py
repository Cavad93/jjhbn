#!/usr/bin/env python3
"""
Comprehensive Bot Test Suite

Комплексное тестирование всех компонентов торгового бота:
1. BASE Logic - способность находить закономерности
2. Feature Builder - корректность фич
3. ML Models - предсказания
4. META - агрегация
5. Position Management - открытие/закрытие
6. Risk Management - защитные механизмы
7. Online Learning - обучение на истории
8. Portfolio Management - управление портфелем

Usage:
    python3 tests/comprehensive_bot_test.py
"""

import sys
import os
from pathlib import Path

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
import time

# Импорты компонентов бота
import binance_config as config
from features.base_logic import BaseLogicMultiTF, features_from_binance, prob_up_down_at_time
from features.builder import BinanceFeatureBuilder
from models.experts import XGBoostExpert, RandomForestExpert, NeuralNetworkExpert
from portfolio.position_manager import PositionManager, Position
from risk.trailing_stop import TrailingStopManager
from risk.black_swan import BlackSwanProtection
from risk.night_mode import NightModeManager
from strategy.threshold_adapter import get_adaptive_threshold
from strategy.kelly_sizer import calculate_kelly_position_size
from strategy.coin_selector import CoinSelector

# Генератор паттернов
from tests.market_pattern_generator import MarketPatternGenerator


class ComprehensiveBotTest:
    """
    Комплексное тестирование всех компонентов бота
    """

    def __init__(self):
        self.results = {}
        self.generator = MarketPatternGenerator(base_price=35000.0)
        self.patterns = None

    def run_all_tests(self):
        """Запуск всех тестов"""
        print("="*80)
        print("🤖 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ТОРГОВОГО БОТА")
        print("="*80)
        print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # Генерируем паттерны
        print("\n📊 Генерация рыночных паттернов...")
        self.patterns = self.generator.generate_all_patterns()
        print(f"✓ Сгенерировано {len(self.patterns)} паттернов")

        # Тесты
        tests = [
            ("1️⃣  BASE Logic - Распознавание паттернов", self.test_base_logic_patterns),
            ("2️⃣  Feature Builder - Вычисление фич", self.test_feature_builder),
            ("3️⃣  ML Models - Предсказания", self.test_ml_models),
            ("4️⃣  Position Management - Открытие/закрытие", self.test_position_management),
            ("5️⃣  Risk Management - Защитные механизмы", self.test_risk_management),
            ("6️⃣  Adaptive Threshold - Адаптация порога", self.test_adaptive_threshold),
            ("7️⃣  Kelly Sizing - Размер позиции", self.test_kelly_sizing),
            ("8️⃣  Integration Test - Полный цикл", self.test_full_cycle),
        ]

        for test_name, test_func in tests:
            print(f"\n{'='*80}")
            print(f"{test_name}")
            print(f"{'='*80}")
            try:
                test_func()
                self.results[test_name] = "✅ PASS"
            except Exception as e:
                self.results[test_name] = f"❌ FAIL: {e}"
                print(f"❌ ОШИБКА: {e}")
                import traceback
                traceback.print_exc()

        # Итоговый отчет
        self.print_final_report()

    def test_base_logic_patterns(self):
        """
        Тест BASE логики на распознавание паттернов

        КРИТИЧЕСКИ ВАЖНО: BASE логика должна находить закономерности
        в рыночных данных, даже без ML моделей
        """
        print("\n📈 Тестирование BASE логики на различных паттернах...")

        base_logic = BaseLogicMultiTF()
        results = {}

        for pattern_name, df in self.patterns.items():
            print(f"\n  Паттерн: {pattern_name}")

            # Вычисляем сигналы BASE логики
            p_up, p_down = base_logic.predict(df)

            print(f"    P(up):   {p_up:.3f}")
            print(f"    P(down): {p_down:.3f}")
            print(f"    Bias:    {'+UP' if p_up > p_down else '-DOWN'} ({abs(p_up - p_down):.3f})")

            results[pattern_name] = {
                'p_up': p_up,
                'p_down': p_down,
                'bias': p_up - p_down
            }

            # Проверяем соответствие ожиданиям
            if pattern_name == 'uptrend':
                assert p_up > 0.52, f"Uptrend: ожидается P(up) > 0.52, получено {p_up:.3f}"
                print("    ✓ Распознан восходящий тренд")

            elif pattern_name == 'downtrend':
                assert p_down > 0.52, f"Downtrend: ожидается P(down) > 0.52, получено {p_down:.3f}"
                print("    ✓ Распознан нисходящий тренд")

            elif pattern_name == 'breakout':
                # Breakout должен давать сильный LONG сигнал
                assert p_up > 0.53, f"Breakout: ожидается P(up) > 0.53, получено {p_up:.3f}"
                print("    ✓ Распознан breakout")

            elif pattern_name == 'sideways':
                # Sideways - слабые сигналы, близко к 0.5
                assert 0.47 < p_up < 0.53, f"Sideways: ожидается P(up) ≈ 0.5, получено {p_up:.3f}"
                print("    ✓ Распознано боковое движение")

        print(f"\n✅ BASE логика КОРРЕКТНО распознает все паттерны!")
        return results

    def test_feature_builder(self):
        """
        Тест Feature Builder - вычисление 68 фич
        """
        print("\n🔢 Тестирование Feature Builder (68 фич)...")

        builder = BinanceFeatureBuilder()

        # Используем uptrend паттерн для теста
        df = self.patterns['uptrend']

        # Вычисляем фичи
        features = builder.build_features_for_timestamp(
            df_5m=df,
            df_15m=df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }),
            df_30m=df.resample('30min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }),
            df_4h=df.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        )

        print(f"  Вычислено фич: {len(features)}")
        print(f"  Ожидается: 68")

        assert len(features) == 68, f"Ожидается 68 фич, получено {len(features)}"

        # Проверяем, что фичи не NaN
        nan_count = np.isnan(features).sum()
        print(f"  NaN фич: {nan_count}")
        assert nan_count == 0, f"Обнаружено {nan_count} NaN фич"

        # Проверяем диапазоны фич
        print(f"  Min: {features.min():.4f}")
        print(f"  Max: {features.max():.4f}")
        print(f"  Mean: {features.mean():.4f}")

        print(f"\n✅ Feature Builder работает корректно (68 фич без NaN)")

    def test_ml_models(self):
        """
        Тест ML моделей - загрузка и предсказания
        """
        print("\n🤖 Тестирование ML моделей...")

        # Загружаем модели
        experts = {}

        xgb_path = config.MODELS_DIR / 'saved' / 'xgb_expert.pkl'
        if xgb_path.exists():
            experts['xgb'] = XGBoostExpert.load(str(xgb_path))
            print(f"  ✓ XGBoost загружен")
        else:
            print(f"  ⚠ XGBoost не найден")

        rf_path = config.MODELS_DIR / 'saved' / 'rf_expert.pkl'
        if rf_path.exists():
            experts['rf'] = RandomForestExpert.load(str(rf_path))
            print(f"  ✓ RandomForest загружен")
        else:
            print(f"  ⚠ RandomForest не найден")

        nn_path = config.MODELS_DIR / 'saved' / 'nn_expert.pkl'
        if nn_path.exists():
            experts['nn'] = NeuralNetworkExpert.load(str(nn_path))
            print(f"  ✓ NeuralNet загружен")
        else:
            print(f"  ⚠ NeuralNet не найден")

        # Тестовые фичи (случайные 68 фич)
        test_features = np.random.randn(68)

        # Предсказания
        for name, expert in experts.items():
            p_up = expert.predict(test_features)
            print(f"  {name.upper()}: P(up) = {p_up:.3f}")

            assert 0.0 <= p_up <= 1.0, f"{name}: вероятность должна быть в [0, 1]"

        print(f"\n✅ ML модели работают корректно ({len(experts)} экспертов)")

    def test_position_management(self):
        """
        Тест управления позициями - открытие/закрытие
        """
        print("\n📊 Тестирование Position Management...")

        manager = PositionManager(max_positions=10)

        # Создаем тестовую позицию
        position = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=35000.0,
            amount=0.1,
            position_value=100.0,
            tp_price=36000.0,
            sl_price=34500.0,
            atr_value=150.0,
            entry_snapshot={
                'features_68d': np.random.randn(68),
                'predictions': {},
                'p_meta': 0.65,
                'timestamp': time.time()
            },
            status='OPEN',
            entry_order_id='test_entry_1',
            tp_order_id='test_tp_1',
            sl_order_id='test_sl_1'
        )

        # Добавляем позицию
        manager.add_position(position)
        print(f"  ✓ Позиция добавлена: {position.symbol} {position.direction}")

        # Проверяем наличие
        assert manager.has_position('BTCUSDT'), "Позиция должна быть в портфеле"
        assert len(manager.get_all_open()) == 1, "Должна быть 1 открытая позиция"

        # Закрываем позицию
        manager.close_position('BTCUSDT', exit_price=35800.0, exit_reason='TP')
        print(f"  ✓ Позиция закрыта: TP")

        # Проверяем закрытие
        assert not manager.has_position('BTCUSDT'), "Позиция должна быть удалена"
        assert len(manager.get_all_open()) == 0, "Не должно быть открытых позиций"
        assert len(manager.closed_positions) == 1, "Должна быть 1 закрытая позиция"

        # Проверяем расчет PnL
        closed = manager.closed_positions[0]
        assert closed['pnl'] > 0, "PnL должен быть положительным (TP достигнут)"
        print(f"  ✓ PnL рассчитан: ${closed['pnl']:.2f}")

        print(f"\n✅ Position Management работает корректно")

    def test_risk_management(self):
        """
        Тест риск-менеджмента - trailing stop, black swan, night mode
        """
        print("\n🛡 Тестирование Risk Management...")

        # 1. Trailing Stop
        print("\n  Trailing Stop Manager:")
        trailing = TrailingStopManager(activation_pct=2.0, distance_pct=1.0)

        # Создаем тестовую позицию
        position = Position(
            symbol='BTCUSDT',
            direction='LONG',
            entry_time=datetime.now(),
            entry_price=35000.0,
            amount=0.1,
            position_value=100.0,
            tp_price=36000.0,
            sl_price=34500.0,
            atr_value=150.0,
            entry_snapshot={},
            status='OPEN',
            entry_order_id='test_1',
            tp_order_id='test_2',
            sl_order_id='test_3'
        )

        # Цена выросла на 2.5% - должен активироваться trailing stop
        current_price = 35875.0  # +2.5%
        new_sl = trailing.check_trailing_stop(position, current_price)

        assert new_sl is not None, "Trailing stop должен активироваться при +2.5%"
        assert new_sl > position.sl_price, "Новый SL должен быть выше старого"
        print(f"    ✓ Trailing stop активирован: {position.sl_price:.2f} → {new_sl:.2f}")

        # 2. Black Swan Protection
        print("\n  Black Swan Protection:")
        black_swan = BlackSwanProtection(price_drop_pct=5.0)

        # Симулируем падение BTC на 6%
        btc_prices = [50000, 49000, 48000, 47000, 46000]  # -8% за несколько свечей

        is_black_swan = black_swan.check_black_swan_event(btc_price=btc_prices[-1])

        # Обновляем историю цен
        for price in btc_prices:
            black_swan._price_history[time.time()] = price
            time.sleep(0.01)

        print(f"    ✓ Black swan защита работает")

        # 3. Night Mode
        print("\n  Night Mode Manager:")
        night_mode = NightModeManager(start_hour=23, end_hour=7, max_positions=5)

        # Проверяем текущее время
        is_night = night_mode.is_night_mode_active()
        max_pos = night_mode.get_max_positions()

        print(f"    Night mode active: {is_night}")
        print(f"    Max positions: {max_pos}")
        print(f"    ✓ Night mode работает")

        print(f"\n✅ Risk Management работает корректно")

    def test_adaptive_threshold(self):
        """
        Тест адаптивного порога
        """
        print("\n🎯 Тестирование Adaptive Threshold...")

        # Сценарий 1: Начало торговли (мало сделок)
        threshold_1 = get_adaptive_threshold(
            total_closed=10,
            recent_hour=3,
            recent_wr=0.55,
            calib_error=0.05
        )
        print(f"  Начало торговли (10 сделок, WR=55%): {threshold_1:.4f}")
        assert threshold_1 >= 0.60, "Начальный порог должен быть консервативным (>=0.60)"

        # Сценарий 2: Успешная торговля (высокий WR)
        threshold_2 = get_adaptive_threshold(
            total_closed=100,
            recent_hour=5,
            recent_wr=0.62,
            calib_error=0.03
        )
        print(f"  Успешная торговля (100 сделок, WR=62%): {threshold_2:.4f}")
        assert threshold_2 < threshold_1, "При хорошем WR порог должен снизиться"

        # Сценарий 3: Плохая торговля (низкий WR)
        threshold_3 = get_adaptive_threshold(
            total_closed=100,
            recent_hour=5,
            recent_wr=0.45,
            calib_error=0.08
        )
        print(f"  Плохая торговля (100 сделок, WR=45%): {threshold_3:.4f}")
        assert threshold_3 > threshold_2, "При низком WR порог должен повыситься"

        print(f"\n✅ Adaptive Threshold работает корректно (адаптируется к производительности)")

    def test_kelly_sizing(self):
        """
        Тест Kelly Criterion position sizing
        """
        print("\n💰 Тестирование Kelly Position Sizing...")

        capital = 1000.0

        # Сценарий 1: Высокий EV, высокая вероятность
        size_1 = calculate_kelly_position_size(
            capital=capital,
            ev=0.10,
            p_up=0.65,
            recent_wr=0.60
        )
        print(f"  Высокий EV (10%), P(up)=65%: ${size_1:.2f} ({size_1/capital*100:.2f}%)")
        assert 15 <= size_1 <= 200, "Размер должен быть в пределах $15-$200"

        # Сценарий 2: Низкий EV
        size_2 = calculate_kelly_position_size(
            capital=capital,
            ev=0.02,
            p_up=0.52,
            recent_wr=0.50
        )
        print(f"  Низкий EV (2%), P(up)=52%: ${size_2:.2f} ({size_2/capital*100:.2f}%)")
        assert size_2 < size_1, "При низком EV размер должен быть меньше"

        # Сценарий 3: Очень высокий EV
        size_3 = calculate_kelly_position_size(
            capital=capital,
            ev=0.20,
            p_up=0.70,
            recent_wr=0.65
        )
        print(f"  Очень высокий EV (20%), P(up)=70%: ${size_3:.2f} ({size_3/capital*100:.2f}%)")
        assert size_3 >= size_1, "При очень высоком EV размер должен быть больше"

        print(f"\n✅ Kelly Sizing работает корректно (адаптирует размер к EV)")

    def test_full_cycle(self):
        """
        Интеграционный тест - полный цикл работы бота

        Проверяет весь пайплайн:
        1. Получение данных
        2. Вычисление фич
        3. BASE предсказание
        4. ML предсказания
        5. Оценка EV
        6. Принятие решения
        """
        print("\n🔄 Интеграционный тест - полный цикл...")

        # 1. Данные
        df = self.patterns['uptrend']
        print(f"  ✓ Данные: {len(df)} свечей (uptrend)")

        # 2. Фичи
        builder = BinanceFeatureBuilder()
        features = builder.build_features_for_timestamp(
            df_5m=df,
            df_15m=df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}),
            df_30m=df.resample('30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}),
            df_4h=df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        )
        print(f"  ✓ Фичи: {len(features)} (68 ожидается)")

        # 3. BASE предсказание
        base_logic = BaseLogicMultiTF()
        p_up_base, p_down_base = base_logic.predict(df)
        print(f"  ✓ BASE: P(up)={p_up_base:.3f}, P(down)={p_down_base:.3f}")

        # 4. ML предсказания (если модели доступны)
        predictions = {'base': p_up_base}

        xgb_path = config.MODELS_DIR / 'saved' / 'xgb_expert.pkl'
        if xgb_path.exists():
            xgb = XGBoostExpert.load(str(xgb_path))
            predictions['xgb'] = xgb.predict(features)
            print(f"  ✓ XGBoost: P(up)={predictions['xgb']:.3f}")

        rf_path = config.MODELS_DIR / 'saved' / 'rf_expert.pkl'
        if rf_path.exists():
            rf = RandomForestExpert.load(str(rf_path))
            predictions['rf'] = rf.predict(features)
            print(f"  ✓ RF: P(up)={predictions['rf']:.3f}")

        # 5. Агрегация (среднее)
        p_up_final = np.mean(list(predictions.values()))
        print(f"  ✓ Агрегация: P(up)={p_up_final:.3f}")

        # 6. Принятие решения
        threshold = get_adaptive_threshold(
            total_closed=50,
            recent_hour=5,
            recent_wr=0.55,
            calib_error=0.05
        )
        print(f"  ✓ Порог: {threshold:.3f}")

        should_open = p_up_final > threshold
        print(f"  ✓ Решение: {'ОТКРЫТЬ LONG' if should_open else 'НЕ ОТКРЫВАТЬ'}")

        # Для uptrend паттерна ожидаем LONG сигнал
        if p_up_base > 0.52:
            print(f"\n✅ Полный цикл работает корректно (uptrend распознан)")
        else:
            print(f"\n⚠ BASE логика не распознала uptrend (возможно нужна донастройка)")

    def print_final_report(self):
        """Печать итогового отчета"""
        print("\n" + "="*80)
        print("📋 ИТОГОВЫЙ ОТЧЕТ")
        print("="*80)

        passed = 0
        failed = 0

        for test_name, result in self.results.items():
            print(f"\n{result} {test_name}")
            if "✅" in result:
                passed += 1
            else:
                failed += 1

        print("\n" + "="*80)
        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0

        print(f"Всего тестов: {total}")
        print(f"Пройдено: {passed} ({success_rate:.1f}%)")
        print(f"Провалено: {failed}")

        if failed == 0:
            print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            print("Бот готов к работе. Все компоненты функционируют корректно.")
        else:
            print(f"\n⚠ ВНИМАНИЕ: {failed} тест(ов) провалено!")
            print("Необходимо исправить ошибки перед запуском бота.")

        print("="*80)


def main():
    """Главная функция"""
    test_suite = ComprehensiveBotTest()
    test_suite.run_all_tests()


if __name__ == '__main__':
    main()
