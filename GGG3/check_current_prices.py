#!/usr/bin/env python3
"""
Проверка текущих цен и расчёт P&L для открытых позиций
"""
from paper_trading.paper_exchange import get_binance_price
from datetime import datetime

# Открытые позиции из логов (19:14-19:15 МСК)
positions = [
    {'symbol': 'KSMUSDT', 'direction': 'LONG', 'entry': 13.42, 'amount': 6.706408, 'size': 90.00},
    {'symbol': 'ALICEUSDT', 'direction': 'LONG', 'entry': 0.29, 'amount': 305.291723, 'size': 90.00, 'status': 'CLOSED', 'pnl': 3.66},
    {'symbol': 'ZENUSDT', 'direction': 'SHORT', 'entry': 14.07, 'amount': 1.065947, 'size': 15.00},
    {'symbol': 'ILVUSDT', 'direction': 'LONG', 'entry': 11.64, 'amount': 7.731959, 'size': 90.00},
    {'symbol': 'ICPUSDT', 'direction': 'LONG', 'entry': 9.14, 'amount': 9.850060, 'size': 90.00},
    {'symbol': 'YGGUSDT', 'direction': 'LONG', 'entry': 0.12, 'amount': 743.801653, 'size': 90.00},
    {'symbol': 'SOLUSDT', 'direction': 'SHORT', 'entry': 157.00, 'amount': 0.132590, 'size': 20.82},
    {'symbol': 'XRPUSDT', 'direction': 'SHORT', 'entry': 2.27, 'amount': 9.843188, 'size': 22.30},
    {'symbol': 'BNBUSDT', 'direction': 'LONG', 'entry': 993.09, 'amount': 0.076488, 'size': 75.96},
    {'symbol': 'PENDLEUSDT', 'direction': 'LONG', 'entry': 2.72, 'amount': 26.843242, 'size': 73.12},
]

print("=" * 80)
print(f"Проверка позиций - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

total_pnl = 0
total_invested = 0
open_positions_value = 0

for i, pos in enumerate(positions, 1):
    symbol = pos['symbol']

    # Пропускаем закрытые
    if pos.get('status') == 'CLOSED':
        print(f"{i:2d}. {symbol:<12s} {pos['direction']:<6s} [CLOSED]")
        print(f"    Entry: ${pos['entry']:.4f}")
        print(f"    P&L: +${pos['pnl']:.2f}")
        print()
        total_pnl += pos['pnl']
        continue

    try:
        current_price = get_binance_price(symbol)

        # Расчёт P&L
        if pos['direction'] == 'LONG':
            pnl = (current_price - pos['entry']) * pos['amount']
            pct = ((current_price / pos['entry']) - 1) * 100
        else:  # SHORT
            pnl = (pos['entry'] - current_price) * pos['amount']
            pct = ((pos['entry'] / current_price) - 1) * 100

        current_value = current_price * pos['amount']

        # Форматирование вывода
        pnl_sign = "+" if pnl >= 0 else ""
        pct_sign = "+" if pct >= 0 else ""

        print(f"{i:2d}. {symbol:<12s} {pos['direction']:<6s}")
        print(f"    Entry:   ${pos['entry']:.4f}")
        print(f"    Current: ${current_price:.4f} ({pct_sign}{pct:.2f}%)")
        print(f"    P&L:     {pnl_sign}${pnl:.2f}")
        print()

        total_pnl += pnl
        total_invested += pos['size']
        open_positions_value += current_value

    except Exception as e:
        print(f"{i:2d}. {symbol:<12s} {pos['direction']:<6s} [ERROR]")
        print(f"    Entry: ${pos['entry']:.4f}")
        print(f"    Error: {e}")
        print()

print("=" * 80)
print(f"ИТОГО:")
print(f"  Начальный капитал:        $1,000.00")
print(f"  Реализованная прибыль:    +${positions[1]['pnl']:.2f} (ALICEUSDT)")
print(f"  Нереализованная прибыль:  {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}")
print(f"  Текущий капитал:          ${1000 + total_pnl:.2f}")
print("=" * 80)
