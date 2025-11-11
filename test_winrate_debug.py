#!/usr/bin/env python3
"""
Отладка логики винрейта для ML экспертов
"""

# Симулируем SHORT позицию как в примере
print("="*80)
print("ТЕСТ: SHORT позиция (прибыльная)")
print("="*80)

# ZKUSDT SHORT: Вход $0.0546, Выход $0.0532, PnL +2.50%
entry_price = 0.0546
exit_price = 0.0532
direction = "SHORT"
amount = 100  # для примера
pnl = (entry_price - exit_price) * amount  # SHORT: прибыль когда цена падает
pnl_pct = (1 - exit_price / entry_price) * 100

print(f"Вход: ${entry_price}")
print(f"Выход: ${exit_price}")
print(f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
print()

# Определяем price_went_up (как в коде)
if direction == 'LONG':
    price_went_up = pnl > 0
else:  # SHORT
    price_went_up = pnl < 0  # SHORT: убыток означает цена выросла

print(f"price_went_up = {price_went_up}")
print(f"Цена фактически: {'ВЫРОСЛА' if price_went_up else 'УПАЛА'}")
print()

# Тестируем разные предсказания экспертов
print("-"*80)
print("Проверка логики для разных предсказаний:")
print("-"*80)

test_predictions = [
    ("XGB прогнозирует РОСТ (p_up=0.65)", 0.65),
    ("RF прогнозирует ПАДЕНИЕ (p_up=0.45)", 0.45),
    ("ARF прогнозирует РОСТ (p_up=0.70)", 0.70),
    ("NN прогнозирует НЕЙТРАЛЬНО (p_up=0.50)", 0.50),
]

for name, p_up in test_predictions:
    expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)
    result = "✓ ПРАВ" if expert_was_right else "✗ НЕ ПРАВ"
    print(f"{name:40s} → {result}")
    print(f"  Логика: (p_up={p_up:.2f} > 0.5) and price_went_up={price_went_up} = {p_up > 0.5 and price_went_up}")
    print(f"          (p_up={p_up:.2f} <= 0.5) and not price_went_up={not price_went_up} = {p_up <= 0.5 and not price_went_up}")
    print()

# Теперь тестируем случай, когда ВСЕ эксперты дают p_up > 0.5
print("="*80)
print("ТЕСТ: Что если все эксперты предсказали РОСТ?")
print("="*80)

all_predict_up = {
    'xgb': 0.65,
    'rf': 0.62,
    'arf': 0.68,
    'nn': 0.60
}

wins = 0
total = 0
for expert_name, p_up in all_predict_up.items():
    expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)
    total += 1
    if expert_was_right:
        wins += 1
    result = "✓" if expert_was_right else "✗"
    print(f"{expert_name.upper():4s}: p_up={p_up:.2f} → {result}")

print()
print(f"Общий винрейт если все предсказали рост: {wins}/{total} = {wins/total*100:.1f}%")
print()

# Теперь если бы 1 эксперт предсказал падение
print("="*80)
print("ТЕСТ: Если один эксперт предсказал ПАДЕНИЕ, а остальные РОСТ")
print("="*80)

mixed_predictions = {
    'xgb': 0.65,  # Рост
    'rf': 0.45,   # Падение ✓
    'arf': 0.68,  # Рост
    'nn': 0.60    # Рост
}

wins = 0
total = 0
for expert_name, p_up in mixed_predictions.items():
    expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)
    total += 1
    if expert_was_right:
        wins += 1
    result = "✓" if expert_was_right else "✗"
    print(f"{expert_name.upper():4s}: p_up={p_up:.2f} → {result}")

print()
print(f"Общий винрейт: {wins}/{total} = {wins/total*100:.1f}%")
print()

# Проверяем 6 сделок, где 5 LONG (убыток) и 1 SHORT (прибыль)
print("="*80)
print("СИМУЛЯЦИЯ: 5 LONG (убыток) + 1 SHORT (прибыль) = 6 сделок")
print("="*80)

# Предположим все эксперты всегда предсказывают РОСТ (p_up > 0.5)
expert_predictions_always_up = {'xgb': 0.65, 'rf': 0.62, 'arf': 0.68, 'nn': 0.60}
expert_stats = {name: {'wins': 0, 'total': 0} for name in expert_predictions_always_up.keys()}

trades = [
    # 5 LONG убыточных (цена упала)
    {'direction': 'LONG', 'pnl': -10, 'desc': 'LONG убыток (цена упала)'},
    {'direction': 'LONG', 'pnl': -15, 'desc': 'LONG убыток (цена упала)'},
    {'direction': 'LONG', 'pnl': -8, 'desc': 'LONG убыток (цена упала)'},
    {'direction': 'LONG', 'pnl': -12, 'desc': 'LONG убыток (цена упала)'},
    {'direction': 'LONG', 'pnl': -20, 'desc': 'LONG убыток (цена упала)'},
    # 1 SHORT прибыльный (цена упала)
    {'direction': 'SHORT', 'pnl': 25, 'desc': 'SHORT прибыль (цена упала)'},
]

print()
for i, trade in enumerate(trades, 1):
    direction = trade['direction']
    pnl = trade['pnl']

    # Определяем price_went_up
    if direction == 'LONG':
        price_went_up = pnl > 0
    else:  # SHORT
        price_went_up = pnl < 0

    print(f"Сделка {i}: {trade['desc']}")
    print(f"  price_went_up={price_went_up}")

    # Проверяем всех экспертов
    for expert_name, p_up in expert_predictions_always_up.items():
        expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)
        expert_stats[expert_name]['total'] += 1
        if expert_was_right:
            expert_stats[expert_name]['wins'] += 1
        result = "✓" if expert_was_right else "✗"
        print(f"    {expert_name.upper()}: {result}")
    print()

print("="*80)
print("ИТОГОВАЯ СТАТИСТИКА (если все эксперты всегда предсказывают РОСТ):")
print("="*80)
for expert_name, stats in expert_stats.items():
    wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"{expert_name.upper()}: {wr:.1f}% ({stats['wins']}/{stats['total']} сделок)")

print()
print("ВЫВОД: Если все эксперты всегда предсказывают РОСТ (p_up > 0.5),")
print("       и было 5 LONG убытков + 1 SHORT прибыль, то винрейт = 0%")
print("       Потому что:")
print("       - LONG убыток → цена упала → price_went_up=False → эксперты не правы")
print("       - SHORT прибыль → цена упала → price_went_up=False → эксперты не правы")
