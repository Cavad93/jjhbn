#!/usr/bin/env python3
"""
Моделирование сценария 16.7% винрейта (1 из 6 сделок)
"""

print("="*80)
print("СЦЕНАРИЙ: Какие сделки дали бы 16.7% винрейт (1 из 6)?")
print("="*80)
print()

# Предположим все эксперты всегда предсказывают РОСТ (p_up > 0.5)
expert_predictions_always_up = {'xgb': 0.65, 'rf': 0.62, 'arf': 0.68, 'nn': 0.60}
expert_stats = {name: {'wins': 0, 'total': 0} for name in expert_predictions_always_up.keys()}

# Сценарий 1: 5 сделок где цена упала + 1 сделка где цена выросла
trades_scenario1 = [
    {'direction': 'LONG', 'pnl': -10, 'desc': 'LONG убыток (цена упала)'},
    {'direction': 'LONG', 'pnl': -15, 'desc': 'LONG убыток (цена упала)'},
    {'direction': 'LONG', 'pnl': -8, 'desc': 'LONG убыток (цена упала)'},
    {'direction': 'LONG', 'pnl': -12, 'desc': 'LONG убыток (цена упала)'},
    {'direction': 'SHORT', 'pnl': 25, 'desc': 'SHORT прибыль (цена упала)'},
    {'direction': 'LONG', 'pnl': 30, 'desc': 'LONG прибыль (цена выросла) ✓'},  # Единственная где эксперты правы
]

print("СЦЕНАРИЙ 1: 5 сделок (цена упала) + 1 сделка (цена выросла)")
print("-"*80)

for i, trade in enumerate(trades_scenario1, 1):
    direction = trade['direction']
    pnl = trade['pnl']

    # Определяем price_went_up
    if direction == 'LONG':
        price_went_up = pnl > 0
    else:  # SHORT
        price_went_up = pnl < 0

    print(f"Сделка {i}: {trade['desc']}")

    # Проверяем всех экспертов
    for expert_name, p_up in expert_predictions_always_up.items():
        expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)
        expert_stats[expert_name]['total'] += 1
        if expert_was_right:
            expert_stats[expert_name]['wins'] += 1

print()
print("РЕЗУЛЬТАТ СЦЕНАРИЯ 1:")
for expert_name, stats in expert_stats.items():
    wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"  {expert_name.upper()}: {wr:.1f}% ({stats['wins']}/{stats['total']} сделок)")

print()
print("="*80)
print()

# Сценарий 2: Все эксперты дают ОДИНАКОВЫЕ предсказания, но иногда ошибаются
expert_stats2 = {name: {'wins': 0, 'total': 0} for name in expert_predictions_always_up.keys()}

# Давайте проверим, могут ли быть разные предсказания для разных сделок
trades_scenario2 = [
    # Сделка 1: эксперты предсказали РОСТ (p_up=0.65), но цена упала (LONG убыток)
    {'direction': 'LONG', 'pnl': -10, 'predictions': {'xgb': 0.65, 'rf': 0.62, 'arf': 0.68, 'nn': 0.60}},
    # Сделка 2: эксперты предсказали РОСТ (p_up=0.70), но цена упала (LONG убыток)
    {'direction': 'LONG', 'pnl': -15, 'predictions': {'xgb': 0.70, 'rf': 0.68, 'arf': 0.72, 'nn': 0.65}},
    # Сделка 3: эксперты предсказали РОСТ (p_up=0.63), но цена упала (LONG убыток)
    {'direction': 'LONG', 'pnl': -8, 'predictions': {'xgb': 0.63, 'rf': 0.61, 'arf': 0.66, 'nn': 0.59}},
    # Сделка 4: эксперты предсказали РОСТ (p_up=0.67), но цена упала (LONG убыток)
    {'direction': 'LONG', 'pnl': -12, 'predictions': {'xgb': 0.67, 'rf': 0.64, 'arf': 0.69, 'nn': 0.62}},
    # Сделка 5: эксперты предсказали РОСТ (p_up=0.68), цена упала, но SHORT прибыль
    {'direction': 'SHORT', 'pnl': 25, 'predictions': {'xgb': 0.68, 'rf': 0.65, 'arf': 0.70, 'nn': 0.63}},
    # Сделка 6: эксперты предсказали РОСТ (p_up=0.72), цена выросла (LONG прибыль) ✓
    {'direction': 'LONG', 'pnl': 30, 'predictions': {'xgb': 0.72, 'rf': 0.69, 'arf': 0.74, 'nn': 0.68}},
]

print("СЦЕНАРИЙ 2: Разные предсказания для каждой сделки (но все > 0.5)")
print("-"*80)

for i, trade in enumerate(trades_scenario2, 1):
    direction = trade['direction']
    pnl = trade['pnl']
    predictions = trade['predictions']

    # Определяем price_went_up
    if direction == 'LONG':
        price_went_up = pnl > 0
    else:  # SHORT
        price_went_up = pnl < 0

    price_direction = "ВЫРОСЛА" if price_went_up else "УПАЛА"
    print(f"\nСделка {i}: {direction} PnL={pnl:+.0f}, Цена {price_direction}")

    # Проверяем всех экспертов
    for expert_name, p_up in predictions.items():
        expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)
        expert_stats2[expert_name]['total'] += 1
        if expert_was_right:
            expert_stats2[expert_name]['wins'] += 1
        result = "✓" if expert_was_right else "✗"
        print(f"  {expert_name.upper()}: p_up={p_up:.2f} → {result}")

print()
print("РЕЗУЛЬТАТ СЦЕНАРИЯ 2:")
for expert_name, stats in expert_stats2.items():
    wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"  {expert_name.upper()}: {wr:.1f}% ({stats['wins']}/{stats['total']} сделок)")

print()
print("="*80)
print("ВЫВОД:")
print("="*80)
print("Если все ML эксперты имеют ОДИНАКОВЫЙ винрейт 16.7% (1 из 6),")
print("это означает что:")
print()
print("1. Все эксперты дают ОДИНАКОВЫЕ или очень ПОХОЖИЕ предсказания")
print("   (все > 0.5 или все < 0.5)")
print()
print("2. Из 6 сделок:")
print("   - 5 сделок: эксперты ошиблись (цена пошла не туда)")
print("   - 1 сделка: эксперты угадали (цена пошла в нужную сторону)")
print()
print("3. Это НЕ ошибка в логике подсчёта винрейта!")
print("   Это означает что модели плохо работают или не обучены.")
