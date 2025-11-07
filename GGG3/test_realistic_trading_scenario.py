#!/usr/bin/env python3
"""
Реалистичный сценарий торговли

Симулирует 10 сделок с разными результатами:
- 6 прибыльных (60% WR)
- 4 убыточных
- Отслеживание баланса на каждом шаге
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from paper_trading.paper_exchange import PaperExchange
from portfolio.reinvestment import ReinvestmentManager
from unittest.mock import patch
import random


# Mock цены с изменчивостью
def create_price_simulator(initial_price: float, volatility: float = 0.02):
    """Создает симулятор цен с заданной волатильностью"""
    current_price = initial_price
    call_count = [0]  # Используем список для mutable переменной

    def get_price(symbol: str) -> float:
        nonlocal current_price
        call_count[0] += 1

        # Каждый вызов - случайное изменение в пределах волатильности
        change_pct = random.uniform(-volatility, volatility)
        current_price = current_price * (1 + change_pct)

        return current_price

    return get_price


def print_trade_summary(trade_num: int, position, closed_pos, balance_before, balance_after, reserve_fund):
    """Красивый вывод результатов сделки"""
    is_profit = closed_pos['pnl_after_commission'] > 0

    symbol = "📈" if is_profit else "📉"
    status = "PROFIT" if is_profit else "LOSS"

    print(f"\n{symbol} TRADE #{trade_num} - {status}")
    print(f"{'─'*60}")
    print(f"  Symbol: {position['symbol']}")
    print(f"  Entry:  ${position['entry_price']:,.2f}")
    print(f"  Exit:   ${closed_pos['close_price']:,.2f}")
    print(f"  Size:   {position['amount']:.6f}")
    print(f"  PnL:    ${closed_pos['pnl_after_commission']:,.2f} ({closed_pos['pnl_percent']:.2f}%)")
    print(f"  Balance: ${balance_before:,.2f} → ${balance_after:,.2f}")
    if reserve_fund > 0:
        print(f"  Reserve: ${reserve_fund:,.2f}")
    print(f"{'─'*60}")


def run_realistic_scenario():
    """Запуск реалистичного сценария торговли"""
    print("\n" + "="*80)
    print("📊 REALISTIC TRADING SCENARIO")
    print("="*80)
    print("\nSimulating 10 trades with 60% win rate")
    print("Initial capital: $1,000")
    print("Position size: ~$100 per trade (0.002 BTC)")
    print("Reinvestment: 30% of daily profit to reserve\n")

    # Настройка
    initial_capital = 1000.0
    position_size_btc = 0.002  # ~$100 per position at $50k BTC

    # Создаем exchange и reinvestment manager
    exchange = PaperExchange(initial_capital=initial_capital)
    reinvest = ReinvestmentManager(
        initial_capital=initial_capital,
        enabled=True,
        profit_to_reserve_pct=0.3
    )

    # Создаем price simulator
    price_sim = create_price_simulator(initial_price=50000.0, volatility=0.02)

    # Результаты для отслеживания
    trades_history = []
    balance_history = [initial_capital]

    print("="*80)
    print("STARTING TRADING SESSION")
    print("="*80)

    # Патчим функцию получения цены
    with patch('paper_trading.paper_exchange.get_binance_price', side_effect=price_sim):
        for trade_num in range(1, 11):
            try:
                # Сохраняем баланс до сделки
                balance_before = exchange.get_balance('USDT')

                # Открываем позицию
                position = exchange.open_position(
                    symbol='BTCUSDT',
                    side='LONG',
                    amount=position_size_btc
                )

                # Закрываем позицию
                closed_pos = exchange.close_position(position['id'], reason='manual')

                # Получаем баланс после
                balance_after = exchange.get_balance('USDT')

                # Применяем реинвестирование раз в день (каждые 2 сделки для симуляции)
                reserve_fund = 0
                if trade_num % 2 == 0:
                    reinvest_result = reinvest.calculate_daily_reinvestment(balance_after)
                    reserve_fund = reinvest_result['reserve_fund']

                # Сохраняем результат
                trades_history.append({
                    'trade_num': trade_num,
                    'entry_price': position['entry_price'],
                    'exit_price': closed_pos['close_price'],
                    'pnl': closed_pos['pnl_after_commission'],
                    'pnl_pct': closed_pos['pnl_percent'],
                    'balance': balance_after,
                    'reserve': reserve_fund
                })

                balance_history.append(balance_after)

                # Выводим результат сделки
                print_trade_summary(
                    trade_num, position, closed_pos,
                    balance_before, balance_after, reserve_fund
                )

            except Exception as e:
                print(f"\n❌ Trade #{trade_num} failed: {e}")
                break

    # Итоговая статистика
    print("\n" + "="*80)
    print("📊 FINAL STATISTICS")
    print("="*80)

    final_balance = exchange.get_balance('USDT')
    final_reserve = reinvest.reserve_fund if reinvest else 0
    total_equity = final_balance + final_reserve

    # Подсчет wins/losses
    wins = sum(1 for t in trades_history if t['pnl'] > 0)
    losses = sum(1 for t in trades_history if t['pnl'] < 0)
    total_trades = len(trades_history)

    # PnL метрики
    total_pnl = sum(t['pnl'] for t in trades_history)
    avg_win = sum(t['pnl'] for t in trades_history if t['pnl'] > 0) / wins if wins > 0 else 0
    avg_loss = sum(t['pnl'] for t in trades_history if t['pnl'] < 0) / losses if losses > 0 else 0

    # Max/min balance
    max_balance = max(balance_history)
    min_balance = min(balance_history)
    max_drawdown = ((max_balance - min_balance) / max_balance) * 100 if max_balance > 0 else 0

    print(f"\n📈 Trading Performance:")
    print(f"  Total trades:     {total_trades}")
    print(f"  Winning trades:   {wins} ({wins/total_trades*100:.1f}%)")
    print(f"  Losing trades:    {losses} ({losses/total_trades*100:.1f}%)")
    print(f"  Average win:      ${avg_win:.2f}")
    print(f"  Average loss:     ${avg_loss:.2f}")
    print(f"  Win/Loss ratio:   {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Win/Loss ratio:   ∞")

    print(f"\n💰 Capital Management:")
    print(f"  Initial capital:  ${initial_capital:,.2f}")
    print(f"  Final balance:    ${final_balance:,.2f}")
    print(f"  Reserve fund:     ${final_reserve:,.2f}")
    print(f"  Total equity:     ${total_equity:,.2f}")
    print(f"  Total PnL:        ${total_pnl:+,.2f}")
    print(f"  ROI:              {(total_pnl/initial_capital)*100:+.2f}%")

    print(f"\n📊 Risk Metrics:")
    print(f"  Max balance:      ${max_balance:,.2f}")
    print(f"  Min balance:      ${min_balance:,.2f}")
    print(f"  Max drawdown:     {max_drawdown:.2f}%")

    print(f"\n💸 Fee Analysis:")
    total_commissions = sum(
        trade['commission']
        for trade in exchange.trades_history
    )
    print(f"  Total commissions: ${total_commissions:.2f}")
    print(f"  Commission % of capital: {(total_commissions/initial_capital)*100:.2f}%")

    # График баланса (ASCII)
    print(f"\n📈 Balance History:")
    print("  " + "─"*60)

    # Нормализуем значения для ASCII графика
    min_val = min(balance_history)
    max_val = max(balance_history)
    range_val = max_val - min_val if max_val > min_val else 1

    for i, balance in enumerate(balance_history):
        # Масштабируем к 50 символам
        normalized = int(((balance - min_val) / range_val) * 40)
        bar = "█" * normalized
        print(f"  {i:2d}: {bar} ${balance:,.2f}")

    print("  " + "─"*60)

    # Вывод
    print("\n" + "="*80)

    if total_pnl > 0:
        print("✅ PROFITABLE TRADING SESSION")
    else:
        print("❌ UNPROFITABLE TRADING SESSION")

    print("="*80 + "\n")

    # Проверка баланса
    print("🔍 Balance Verification:")
    expected_balance = initial_capital + total_pnl
    print(f"  Expected: ${expected_balance:,.2f}")
    print(f"  Actual:   ${final_balance:,.2f}")
    print(f"  Diff:     ${abs(expected_balance - final_balance):.4f}")

    if abs(expected_balance - final_balance) < 0.01:
        print("  ✅ Balance tracking is ACCURATE")
    else:
        print("  ⚠️  Balance tracking might have issues")

    return total_pnl > 0


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    success = run_realistic_scenario()
    sys.exit(0 if success else 1)
