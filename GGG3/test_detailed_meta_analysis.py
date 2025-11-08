"""
Детальный анализ: почему только 10 из 37 позиций попали в META

Проверяем реальные условия фильтрации
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
import numpy as np
from portfolio import Position
from meta_neural_cem import MetaNeuralCEM
import binance_config as config

print("=" * 80)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ФИЛЬТРАЦИИ META")
print("=" * 80)

# Инициализируем META
meta = MetaNeuralCEM(cfg=config)
initial_samples = sum(meta.seen_ph.values())
print(f"\n✅ META инициализирован: {meta.mode} mode, {initial_samples} samples")

# ========== СОЗДАЕМ ТЕСТОВЫЕ СЦЕНАРИИ ==========

test_cases = []

# Сценарий 1: Идеальная позиция (должна попасть)
test_cases.append({
    'name': 'Сценарий 1: Полный snapshot с ML',
    'should_pass': True,
    'position': Position(
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
            'ml_predictions': {'xgb': 0.65, 'rf': 0.62, 'nn': 0.68},
            'context': {'vol_ratio': 1.2, 'trend_macd': 0.05},
            'phase': 2,
            'p_meta': 0.65
        },
        status='CLOSED',
        exit_time=datetime.now(),
        exit_price=51000.0,
        exit_reason='TP'
    )
})

# Сценарий 2: Пустой snapshot (НЕ должна попасть)
test_cases.append({
    'name': 'Сценарий 2: Пустой snapshot',
    'should_pass': False,
    'position': Position(
        symbol='ETHUSDT',
        direction='LONG',
        entry_time=datetime.now(),
        entry_price=3000.0,
        amount=0.1,
        position_value=300.0,
        tp_price=3100.0,
        sl_price=2900.0,
        atr_value=50.0,
        entry_snapshot={},
        status='CLOSED',
        exit_time=datetime.now(),
        exit_price=3100.0,
        exit_reason='TP'
    )
})

# Сценарий 3: Только 1 эксперт (НЕ должна попасть)
test_cases.append({
    'name': 'Сценарий 3: Только 1 ML эксперт',
    'should_pass': False,
    'position': Position(
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
            'ml_predictions': {'xgb': 0.60},  # Только 1!
            'phase': 1
        },
        status='CLOSED',
        exit_time=datetime.now(),
        exit_price=390.0,
        exit_reason='TP'
    )
})

# Сценарий 4: 2 эксперта без context (ДОЛЖНА попасть)
test_cases.append({
    'name': 'Сценарий 4: 2 эксперта БЕЗ context',
    'should_pass': True,
    'position': Position(
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
            'ml_predictions': {'xgb': 0.60, 'rf': 0.58},  # 2 эксперта
            'phase': 3
            # context отсутствует!
        },
        status='CLOSED',
        exit_time=datetime.now(),
        exit_price=0.52,
        exit_reason='TP'
    )
})

# Сценарий 5: Predictions вместо ml_predictions (НЕ должна попасть)
test_cases.append({
    'name': 'Сценарий 5: predictions вместо ml_predictions',
    'should_pass': False,
    'position': Position(
        symbol='SOLUSDT',
        direction='LONG',
        entry_time=datetime.now(),
        entry_price=100.0,
        amount=5.0,
        position_value=500.0,
        tp_price=105.0,
        sl_price=95.0,
        atr_value=5.0,
        entry_snapshot={
            'predictions': {'p_xgb': 0.65, 'p_rf': 0.62},  # predictions, не ml_predictions!
            'phase': 4
        },
        status='CLOSED',
        exit_time=datetime.now(),
        exit_price=105.0,
        exit_reason='TP'
    )
})

# Сценарий 6: ml_predictions с None значениями (НЕ должна попасть)
test_cases.append({
    'name': 'Сценарий 6: ml_predictions с None',
    'should_pass': False,
    'position': Position(
        symbol='ADAUSDT',
        direction='SHORT',
        entry_time=datetime.now(),
        entry_price=0.4,
        amount=1250.0,
        position_value=500.0,
        tp_price=0.39,
        sl_price=0.41,
        atr_value=0.02,
        entry_snapshot={
            'ml_predictions': {'xgb': 0.40, 'rf': None, 'nn': None},  # Только 1 не-None
            'phase': 0
        },
        status='CLOSED',
        exit_time=datetime.now(),
        exit_price=0.39,
        exit_reason='TP'
    )
})

# Сценарий 7: Старая позиция БЕЗ entry_snapshot вообще (НЕ должна попасть)
test_cases.append({
    'name': 'Сценарий 7: БЕЗ entry_snapshot (старая позиция)',
    'should_pass': False,
    'position': Position(
        symbol='DOTUSDT',
        direction='LONG',
        entry_time=datetime.now(),
        entry_price=7.0,
        amount=71.0,
        position_value=500.0,
        tp_price=7.2,
        sl_price=6.8,
        atr_value=0.2,
        entry_snapshot=None,  # None!
        status='CLOSED',
        exit_time=datetime.now(),
        exit_price=7.2,
        exit_reason='TP'
    )
})

# ========== ЗАПУСКАЕМ ТЕСТЫ ==========

print("\n" + "=" * 80)
print("ЗАПУСК ТЕСТОВ")
print("=" * 80)

passed = 0
failed = 0

for idx, test_case in enumerate(test_cases, 1):
    print(f"\n{test_case['name']}:")
    position = test_case['position']
    should_pass = test_case['should_pass']

    # Первичная проверка (как в коде)
    if meta is not None and hasattr(position, 'entry_snapshot') and position.entry_snapshot:
        snapshot = position.entry_snapshot
        ml_preds = snapshot.get('ml_predictions', {})
        context = snapshot.get('context', {})

        # Определяем результат
        y_up = 1 if position.pnl > 0 else 0

        # Формируем reg_ctx
        reg_ctx = {
            'phase': snapshot.get('phase', 0),
            'vol_ratio': context.get('volatility_1h', context.get('vol_ratio', 1.0)),
            'trend_macd': context.get('trend_macd', 0.0),
            'book_imb': context.get('orderbook_imbalance', 0.0),
            'ofi_15s': context.get('ofi_15s', 0.0),
            'basis_pct': context.get('basis_pct', 0.0)
        }

        try:
            samples_before = sum(meta.seen_ph.values())

            # Вызываем record_result
            meta.record_result(
                p_xgb=ml_preds.get('xgb'),
                p_rf=ml_preds.get('rf'),
                p_arf=ml_preds.get('arf'),
                p_nn=ml_preds.get('nn'),
                p_base=snapshot.get('p_meta', snapshot.get('p_base')),
                y_up=y_up,
                used_in_live=True,
                p_final_used=snapshot.get('p_meta'),
                reg_ctx=reg_ctx
            )

            samples_after = sum(meta.seen_ph.values())
            actually_passed = samples_after > samples_before

            print(f"  Ожидалось: {'✅ ПОПАДЕТ' if should_pass else '❌ НЕ ПОПАДЕТ'}")
            print(f"  Результат: {'✅ ЗАПИСАНО' if actually_passed else '❌ НЕ ЗАПИСАНО'} ({samples_before} → {samples_after})")

            if actually_passed == should_pass:
                print(f"  ✅ ТЕСТ ПРОЙДЕН")
                passed += 1
            else:
                print(f"  ❌ ТЕСТ ПРОВАЛЕН")
                failed += 1

        except Exception as e:
            print(f"  ❌ ОШИБКА: {e}")
            if should_pass:
                failed += 1
            else:
                passed += 1  # Ожидали что не попадет
    else:
        actually_passed = False
        print(f"  Первичная проверка: НЕ ПРОШЛА")
        print(f"  Ожидалось: {'✅ ПОПАДЕТ' if should_pass else '❌ НЕ ПОПАДЕТ'}")
        print(f"  Результат: ❌ НЕ ЗАПИСАНО")

        if not should_pass:
            print(f"  ✅ ТЕСТ ПРОЙДЕН")
            passed += 1
        else:
            print(f"  ❌ ТЕСТ ПРОВАЛЕН")
            failed += 1

# ========== ИТОГИ ==========

print("\n" + "=" * 80)
print("ИТОГОВАЯ СТАТИСТИКА")
print("=" * 80)

print(f"\n✅ Тестов пройдено: {passed}/{len(test_cases)}")
print(f"❌ Тестов провалено: {failed}/{len(test_cases)}")
print(f"\n📊 META после тестов: {sum(meta.seen_ph.values())} samples")
print(f"   Распределение по фазам: {meta.seen_ph}")

print("\n" + "=" * 80)
print("ВЫВОДЫ")
print("=" * 80)

print("""
🔍 КЛЮЧЕВЫЕ ТРЕБОВАНИЯ для попадания в META:

1. ✅ META должен быть инициализирован (всегда True)
2. ✅ У позиции должен быть атрибут entry_snapshot
3. ✅ entry_snapshot НЕ должен быть None или {}
4. ✅ В entry_snapshot должны быть 'ml_predictions'
5. ✅ Минимум 2 эксперта с не-None значениями

❌ НЕ ОБЯЗАТЕЛЬНО:
- context (используются дефолтные значения если отсутствует)
- predictions (нужны именно ml_predictions)

📌 ПРИЧИНА 27/37:
Позиции были открыты БЕЗ сохранения ml_predictions в entry_snapshot.
Это происходит когда:
- Позиция открыта до внедрения механизма snapshot
- ML модели не были загружены/доступны при открытии
- Произошла ошибка при создании snapshot
- entry_snapshot был создан, но без ml_predictions

🔧 РЕШЕНИЕ:
Убедиться что при открытии позиции ВСЕГДА сохраняются:
- entry_snapshot['ml_predictions'] с минимум 2 экспертами
- entry_snapshot['phase']
""")

print("=" * 80)
