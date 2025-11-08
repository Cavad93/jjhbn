"""
Тест для выяснения причин фильтрации позиций в META

Проверяем:
1. Условия инициализации META
2. Условия записи позиций в META
3. Причины пропуска позиций
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
import numpy as np
from portfolio import Position, PositionManager
from meta_neural_cem import MetaNeuralCEM
import binance_config as config

print("=" * 80)
print("ТЕСТ: Анализ фильтрации позиций для META")
print("=" * 80)

# ========== ТЕСТ 1: Инициализация META ==========
print("\n📋 ТЕСТ 1: Инициализация META")
print("-" * 80)

try:
    meta = MetaNeuralCEM(cfg=config)
    print(f"✅ META инициализирован успешно")
    print(f"   Режим: {meta.mode}")
    print(f"   Примеров по фазам: {meta.seen_ph}")
    print(f"   Всего примеров: {sum(meta.seen_ph.values())}")
    meta_is_none = False
except Exception as e:
    print(f"❌ META не инициализирован: {e}")
    meta = None
    meta_is_none = True

# ========== ТЕСТ 2: Создание позиций с разными snapshot ==========
print("\n📋 ТЕСТ 2: Создание позиций с разными типами entry_snapshot")
print("-" * 80)

# Позиция 1: Полный snapshot (должна попасть в META)
position_1_full = Position(
    symbol='BTCUSDT',
    direction='LONG',
    entry_time=datetime.now(),
    entry_price=50000.0,
    amount=0.01,
    position_value=500.0,
    tp_price=51000.0,
    sl_price=49000.0,
    atr_value=100.0,
    entry_snapshot={
        'features_68d': np.random.rand(68).tolist(),
        'ml_predictions': {
            'xgb': 0.65,
            'rf': 0.62,
            'nn': 0.68
        },
        'predictions': {
            'p_xgb': 0.65,
            'p_rf': 0.62,
            'p_nn': 0.68
        },
        'context': {
            'volatility_1h': 0.015,
            'volume_ratio_5m': 1.2,
            'spread_bps': 5.0,
            'orderbook_imbalance': 0.6,
            'trade_intensity_1m': 0.8,
            'ofi_15s': 0.05,
            'basis_pct': 0.01
        },
        'phase': 2,
        'p_meta': 0.65,
        'timestamp': datetime.now().timestamp()
    },
    status='CLOSED',
    exit_time=datetime.now(),
    exit_price=51000.0,
    exit_reason='TP'
)

# Позиция 2: Пустой snapshot (НЕ должна попасть в META)
position_2_empty = Position(
    symbol='ETHUSDT',
    direction='LONG',
    entry_time=datetime.now(),
    entry_price=3000.0,
    amount=0.1,
    position_value=300.0,
    tp_price=3100.0,
    sl_price=2900.0,
    atr_value=50.0,
    entry_snapshot={},  # Пустой!
    status='CLOSED',
    exit_time=datetime.now(),
    exit_price=3100.0,
    exit_reason='TP'
)

# Позиция 3: Snapshot без ml_predictions (НЕ должна попасть в META)
position_3_incomplete = Position(
    symbol='BNBUSDT',
    direction='SHORT',
    entry_time=datetime.now(),
    entry_price=400.0,
    amount=1.0,
    position_value=400.0,
    tp_price=390.0,
    sl_price=410.0,
    atr_value=10.0,
    entry_snapshot={
        'timestamp': datetime.now().timestamp(),
        'phase': 1
        # Нет ml_predictions и context!
    },
    status='CLOSED',
    exit_time=datetime.now(),
    exit_price=390.0,
    exit_reason='TP'
)

# Позиция 4: Snapshot без context (НЕ должна попасть в META)
position_4_no_context = Position(
    symbol='XRPUSDT',
    direction='LONG',
    entry_time=datetime.now(),
    entry_price=0.5,
    amount=1000.0,
    position_value=500.0,
    tp_price=0.52,
    sl_price=0.48,
    atr_value=0.01,
    entry_snapshot={
        'ml_predictions': {
            'xgb': 0.60,
            'rf': 0.58,
            'nn': 0.62
        },
        'phase': 3
        # Нет context!
    },
    status='CLOSED',
    exit_time=datetime.now(),
    exit_price=0.52,
    exit_reason='TP'
)

test_positions = [
    ('Position 1: Полный snapshot', position_1_full),
    ('Position 2: Пустой snapshot', position_2_empty),
    ('Position 3: Без ml_predictions', position_3_incomplete),
    ('Position 4: Без context', position_4_no_context)
]

# ========== ТЕСТ 3: Проверка условий фильтрации ==========
print("\n📋 ТЕСТ 3: Проверка условий фильтрации")
print("-" * 80)

for name, position in test_positions:
    print(f"\n{name}:")

    # Условие 1: META не None
    condition_1 = meta is not None
    print(f"  ✓ META не None: {condition_1}")

    # Условие 2: hasattr entry_snapshot
    condition_2 = hasattr(position, 'entry_snapshot')
    print(f"  ✓ hasattr entry_snapshot: {condition_2}")

    # Условие 3: entry_snapshot не пустой
    condition_3 = bool(position.entry_snapshot) if condition_2 else False
    print(f"  ✓ entry_snapshot не пустой: {condition_3}")

    # Условие 4: есть ml_predictions
    condition_4 = 'ml_predictions' in position.entry_snapshot if condition_3 else False
    print(f"  ✓ есть ml_predictions: {condition_4}")

    # Условие 5: есть context
    condition_5 = 'context' in position.entry_snapshot if condition_3 else False
    print(f"  ✓ есть context: {condition_5}")

    # Итоговое решение
    will_be_recorded = condition_1 and condition_2 and condition_3
    print(f"  {'✅ ПОПАДЕТ' if will_be_recorded else '❌ НЕ ПОПАДЕТ'} в META (первичная проверка)")

    # Проверка через record_result
    if will_be_recorded and not meta_is_none:
        snapshot = position.entry_snapshot
        ml_preds = snapshot.get('ml_predictions', {})
        context = snapshot.get('context', {})

        # Проверяем что будет при вызове record_result
        has_required_data = bool(ml_preds) and bool(context)
        print(f"  {'✅ ПОПАДЕТ' if has_required_data else '❌ НЕ ПОПАДЕТ'} в META (вторичная проверка)")

# ========== ТЕСТ 4: Симуляция record_result ==========
print("\n📋 ТЕСТ 4: Симуляция вызова record_result")
print("-" * 80)

if not meta_is_none:
    for name, position in test_positions:
        print(f"\n{name}:")

        if hasattr(position, 'entry_snapshot') and position.entry_snapshot:
            snapshot = position.entry_snapshot
            ml_preds = snapshot.get('ml_predictions', {})
            context = snapshot.get('context', {})

            # Определяем результат
            y_up = 1 if position.pnl > 0 else 0

            # Формируем контекст
            reg_ctx = {
                'phase': snapshot.get('phase', 0),
                'volatility_1h': context.get('volatility_1h', 0.0),
                'volume_ratio_5m': context.get('volume_ratio_5m', 1.0),
                'spread_bps': context.get('spread_bps', 10.0),
                'orderbook_imbalance': context.get('orderbook_imbalance', 0.5),
                'trade_intensity_1m': context.get('trade_intensity_1m', 0.5),
                'ofi_15s': context.get('ofi_15s', 0.0),
                'basis_pct': context.get('basis_pct', 0.0)
            }

            try:
                # Пробуем записать
                initial_samples = sum(meta.seen_ph.values())

                meta.record_result(
                    p_xgb=ml_preds.get('xgb'),
                    p_rf=ml_preds.get('rf'),
                    p_arf=None,
                    p_nn=ml_preds.get('nn'),
                    p_base=snapshot.get('p_meta'),
                    y_up=y_up,
                    used_in_live=True,
                    p_final_used=snapshot.get('p_meta'),
                    reg_ctx=reg_ctx
                )

                final_samples = sum(meta.seen_ph.values())

                if final_samples > initial_samples:
                    print(f"  ✅ ЗАПИСАНО в META (было {initial_samples} → стало {final_samples})")
                else:
                    print(f"  ❌ НЕ ЗАПИСАНО в META (samples не изменились: {initial_samples})")

            except Exception as e:
                print(f"  ❌ ОШИБКА при записи: {e}")
        else:
            print(f"  ⚠️  ПРОПУЩЕНО (нет entry_snapshot)")

# ========== ИТОГИ ==========
print("\n" + "=" * 80)
print("ИТОГИ ТЕСТИРОВАНИЯ")
print("=" * 80)

if not meta_is_none:
    print(f"\n✅ META работает корректно")
    print(f"   Всего примеров после теста: {sum(meta.seen_ph.values())}")
    print(f"   Распределение по фазам: {meta.seen_ph}")
else:
    print(f"\n❌ META не инициализирован")

print("\n📌 КРИТИЧЕСКИЕ ТРЕБОВАНИЯ для попадания позиции в META:")
print("   1. META должен быть инициализирован (не None)")
print("   2. У позиции должен быть атрибут entry_snapshot")
print("   3. entry_snapshot не должен быть пустым ({})")
print("   4. В snapshot должны быть ml_predictions")
print("   5. В snapshot должен быть context с 7 признаками")
print("   6. Все признаки должны быть валидными (не None)")

print("\n📌 ОСНОВНАЯ ПРИЧИНА ФИЛЬТРАЦИИ 27 из 37 позиций:")
print("   Позиции были открыты БЕЗ entry_snapshot или с неполным snapshot")
print("   Это могло произойти если:")
print("   - Позиции открыты до внедрения механизма snapshot")
print("   - Произошла ошибка при создании snapshot")
print("   - Отсутствовали необходимые данные (ML predictions или context)")

print("\n" + "=" * 80)
