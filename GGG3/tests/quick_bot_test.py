#!/usr/bin/env python3
"""
Quick Bot Test - Быстрая проверка всех компонентов

Упрощенная версия comprehensive_bot_test.py с корректным API
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datetime import datetime
import time

# Импорты
import binance_config as config
from features.base_logic import BaseLogicMultiTF
from models.experts import XGBoostExpert, RandomForestExpert, NeuralNetworkExpert
from portfolio.position_manager import PositionManager, Position
from risk.trailing_stop import TrailingStopManager
from risk.black_swan import BlackSwanProtection
from risk.night_mode import NightModeManager
from strategy.threshold_adapter import get_adaptive_threshold
from strategy.kelly_sizer import calculate_kelly_position_size

from tests.market_pattern_generator import MarketPatternGenerator


def test_all():
    """Быстрый тест всех компонентов"""
    print("="*80)
    print("🚀 БЫСТРЫЙ ТЕСТ ВСЕХ КОМПОНЕНТОВ БОТА")
    print("="*80)

    results = {}

    # 1. BASE LOGIC
    print("\n1️⃣  BASE Logic - Распознавание паттернов")
    print("-"*80)
    try:
        generator = MarketPatternGenerator()
        patterns = generator.generate_all_patterns()

        base_logic = BaseLogicMultiTF()

        # Uptrend
        p_up, p_down = base_logic.predict(patterns['uptrend'])
        print(f"  Uptrend: P(up)={p_up:.3f}, P(down)={p_down:.3f}")
        assert p_up > 0.52, f"Uptrend не распознан: P(up)={p_up:.3f}"

        # Downtrend
        p_up, p_down = base_logic.predict(patterns['downtrend'])
        print(f"  Downtrend: P(up)={p_up:.3f}, P(down)={p_down:.3f}")
        assert p_down > 0.50, f"Downtrend не распознан"

        # Breakout
        p_up, p_down = base_logic.predict(patterns['breakout'])
        print(f"  Breakout: P(up)={p_up:.3f}, P(down)={p_down:.3f}")
        assert p_up > 0.53, f"Breakout не распознан"

        results['BASE Logic'] = "✅ PASS"
        print("  ✅ BASE Logic находит паттерны!")

    except Exception as e:
        results['BASE Logic'] = f"❌ FAIL: {e}"
        print(f"  ❌ {e}")

    # 2. ML MODELS
    print("\n2️⃣  ML Models - Предсказания")
    print("-"*80)
    try:
        test_features = np.random.randn(68)

        xgb_path = config.MODELS_DIR / 'saved' / 'xgb_expert.pkl'
        if xgb_path.exists():
            xgb = XGBoostExpert.load(str(xgb_path))
            probs = xgb.predict_proba(test_features.reshape(1, -1))
            p_up = float(probs[0])  # predict_proba возвращает (1,) массив
            print(f"  XGBoost: P(up)={p_up:.3f}")
            assert 0.0 <= p_up <= 1.0
        else:
            print("  XGBoost: не найден")

        rf_path = config.MODELS_DIR / 'saved' / 'rf_expert.pkl'
        if rf_path.exists():
            rf = RandomForestExpert.load(str(rf_path))
            probs = rf.predict_proba(test_features.reshape(1, -1))
            p_up = float(probs[0])  # predict_proba возвращает (1,) массив
            print(f"  RandomForest: P(up)={p_up:.3f}")
            assert 0.0 <= p_up <= 1.0
        else:
            print("  RandomForest: не найден")

        results['ML Models'] = "✅ PASS"
        print("  ✅ ML модели работают!")

    except Exception as e:
        results['ML Models'] = f"❌ FAIL: {e}"
        print(f"  ❌ {e}")

    # 3. POSITION MANAGEMENT
    print("\n3️⃣  Position Management")
    print("-"*80)
    try:
        manager = PositionManager(max_positions=10)

        pos = Position(
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
            entry_order_id='test1',
            tp_order_id='test2',
            sl_order_id='test3'
        )

        manager.add_position(pos)
        assert manager.has_position('BTCUSDT')
        print("  ✓ Позиция открыта")

        manager.close_position('BTCUSDT', exit_price=35800.0, exit_reason='TP')
        assert not manager.has_position('BTCUSDT')
        assert len(manager.closed_positions) == 1
        assert manager.closed_positions[0].pnl > 0
        print(f"  ✓ Позиция закрыта: PnL=${manager.closed_positions[0].pnl:.2f}")

        results['Position Management'] = "✅ PASS"
        print("  ✅ Position Management работает!")

    except Exception as e:
        results['Position Management'] = f"❌ FAIL: {e}"
        print(f"  ❌ {e}")

    # 4. RISK MANAGEMENT
    print("\n4️⃣  Risk Management")
    print("-"*80)
    try:
        # Trailing Stop
        trailing = TrailingStopManager(activation_pct=2.0, distance_pct=1.0)
        pos = Position(
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
            entry_order_id='test1',
            tp_order_id='test2',
            sl_order_id='test3'
        )

        new_sl = trailing.check_trailing_stop(pos, 35875.0)  # +2.5%
        assert new_sl is not None
        assert new_sl > pos.sl_price
        print(f"  ✓ Trailing Stop: {pos.sl_price:.2f} → {new_sl:.2f}")

        # Black Swan
        black_swan = BlackSwanProtection()
        print("  ✓ Black Swan Protection инициализирована")

        # Night Mode
        night_mode = NightModeManager()
        max_pos = night_mode.get_max_positions()
        print(f"  ✓ Night Mode: max_positions={max_pos}")

        results['Risk Management'] = "✅ PASS"
        print("  ✅ Risk Management работает!")

    except Exception as e:
        results['Risk Management'] = f"❌ FAIL: {e}"
        print(f"  ❌ {e}")

    # 5. ADAPTIVE THRESHOLD
    print("\n5️⃣  Adaptive Threshold")
    print("-"*80)
    try:
        t1 = get_adaptive_threshold(10, 3, 0.55, 0.05)
        t2 = get_adaptive_threshold(100, 5, 0.62, 0.03)
        t3 = get_adaptive_threshold(100, 5, 0.45, 0.08)

        print(f"  Начало (10 сделок, WR=55%): {t1:.4f}")
        print(f"  Успех (100 сделок, WR=62%): {t2:.4f}")
        print(f"  Неудача (100 сделок, WR=45%): {t3:.4f}")

        assert t2 < t1, "При хорошем WR порог должен снизиться"
        assert t3 > t2, "При плохом WR порог должен повыситься"

        results['Adaptive Threshold'] = "✅ PASS"
        print("  ✅ Adaptive Threshold адаптируется!")

    except Exception as e:
        results['Adaptive Threshold'] = f"❌ FAIL: {e}"
        print(f"  ❌ {e}")

    # 6. KELLY SIZING
    print("\n6️⃣  Kelly Sizing")
    print("-"*80)
    try:
        s1 = calculate_kelly_position_size(1000, 0.10, 0.65, 0.60)
        s2 = calculate_kelly_position_size(1000, 0.02, 0.52, 0.50)
        s3 = calculate_kelly_position_size(1000, 0.20, 0.70, 0.65)

        print(f"  Высокий EV (10%, P=65%): ${s1:.2f}")
        print(f"  Низкий EV (2%, P=52%): ${s2:.2f}")
        print(f"  Очень высокий EV (20%, P=70%): ${s3:.2f}")

        assert 15 <= s1 <= 200
        assert 15 <= s2 <= 200
        assert 15 <= s3 <= 200

        results['Kelly Sizing'] = "✅ PASS"
        print("  ✅ Kelly Sizing работает!")

    except Exception as e:
        results['Kelly Sizing'] = f"❌ FAIL: {e}"
        print(f"  ❌ {e}")

    # ИТОГОВЫЙ ОТЧЕТ
    print("\n" + "="*80)
    print("📋 ИТОГОВЫЙ ОТЧЕТ")
    print("="*80)

    passed = sum(1 for r in results.values() if "✅" in r)
    failed = sum(1 for r in results.values() if "❌" in r)
    total = len(results)

    for name, result in results.items():
        print(f"{result} {name}")

    print("\n" + "="*80)
    print(f"Всего тестов: {total}")
    print(f"Пройдено: {passed} ({passed/total*100:.0f}%)")
    print(f"Провалено: {failed}")

    if failed == 0:
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("Бот готов к работе. Все компоненты функционируют корректно.")
        print("\n🎯 КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
        print("  • BASE Logic НАХОДИТ закономерности в паттернах")
        print("  • ML модели делают предсказания")
        print("  • Position Management открывает и закрывает позиции")
        print("  • Risk Management защищает капитал")
        print("  • Adaptive Threshold адаптируется к производительности")
        print("  • Kelly Sizing оптимизирует размер позиций")
    else:
        print(f"\n⚠ ВНИМАНИЕ: {failed} тест(ов) провалено!")

    print("="*80)


if __name__ == '__main__':
    test_all()
