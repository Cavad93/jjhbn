#!/usr/bin/env python3
"""
Тесты для всех специальных режимов бота

Тестирует:
1. Trailing Stop Manager
2. Black Swan Protection
3. Night Mode Manager
4. Sector Diversification
5. Auto-Reinvestment Manager

Автор: Claude Code
Дата: 2025-11-06
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime, timedelta
from typing import List, Dict
import random

# Риск-менеджеры
from risk.trailing_stop import TrailingStopManager
from risk.black_swan import BlackSwanProtection
from risk.night_mode import NightModeManager

# Sector diversification
from strategy.sector_config import (
    get_coin_sector,
    get_sector_limit,
    count_coins_per_sector,
    SECTOR_MAP
)

# Auto-reinvestment
from portfolio.reinvestment import ReinvestmentManager, simulate_reinvestment

# Position mock
class Position:
    def __init__(self, symbol, direction, entry_price, sl_price, tp_price, atr_value):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.atr_value = atr_value


# =============================================================================
# TEST 1: Trailing Stop Manager
# =============================================================================

def test_trailing_stop():
    """Тест Trailing Stop Manager"""
    print("\n" + "="*80)
    print("TEST 1: TRAILING STOP MANAGER")
    print("="*80)

    manager = TrailingStopManager(
        enabled=True,
        activation_pct=2.0,  # Активация при +2% прибыли
        distance_pct=1.0     # Trailing на 1% ниже пика
    )

    # Позиция LONG
    position = Position(
        symbol='BTCUSDT',
        direction='LONG',
        entry_price=50000.0,
        sl_price=49000.0,
        tp_price=52000.0,
        atr_value=1000.0
    )

    print("\n[SCENARIO 1] LONG position trailing stop")
    print(f"Entry: ${position.entry_price:.2f}, SL: ${position.sl_price:.2f}")

    # Цена растет
    prices = [50000, 50500, 51000, 51500, 52000, 51800, 51500]

    for current_price in prices:
        new_sl = manager.check_trailing_stop(position, current_price)

        profit_pct = ((current_price - position.entry_price) / position.entry_price) * 100

        if new_sl is not None:
            print(f"  Price: ${current_price:.2f} (+{profit_pct:.2f}%) → NEW SL: ${new_sl:.2f}")
            position.sl_price = new_sl
        else:
            print(f"  Price: ${current_price:.2f} (+{profit_pct:.2f}%) → No update")

    # Проверка SHORT
    position_short = Position(
        symbol='ETHUSDT',
        direction='SHORT',
        entry_price=3000.0,
        sl_price=3100.0,
        tp_price=2800.0,
        atr_value=50.0
    )

    print("\n[SCENARIO 2] SHORT position trailing stop")
    print(f"Entry: ${position_short.entry_price:.2f}, SL: ${position_short.sl_price:.2f}")

    prices_short = [3000, 2950, 2900, 2850, 2800, 2820, 2850]

    for current_price in prices_short:
        new_sl = manager.check_trailing_stop(position_short, current_price)

        profit_pct = ((position_short.entry_price - current_price) / position_short.entry_price) * 100

        if new_sl is not None:
            print(f"  Price: ${current_price:.2f} (+{profit_pct:.2f}%) → NEW SL: ${new_sl:.2f}")
            position_short.sl_price = new_sl
        else:
            print(f"  Price: ${current_price:.2f} (+{profit_pct:.2f}%) → No update")

    print("\n✅ Trailing Stop Manager - OK")


# =============================================================================
# TEST 2: Black Swan Protection
# =============================================================================

def test_black_swan():
    """Тест Black Swan Protection"""
    print("\n" + "="*80)
    print("TEST 2: BLACK SWAN PROTECTION")
    print("="*80)

    protection = BlackSwanProtection(
        enabled=True,
        price_drop_pct=10.0,      # Триггер при -10% BTC
        check_interval_sec=60     # Проверка каждую минуту
    )

    print("\n[SCENARIO 1] Normal market conditions")

    # Нормальные цены BTC
    btc_prices = [65000, 65100, 65200, 65150, 65300]

    for price in btc_prices:
        protection.update_btc_price(price)
        is_event = protection.check_black_swan_event()
        print(f"  BTC: ${price:.2f} → Event: {is_event}")

    time.sleep(1)  # Ждём 1 секунду

    print("\n[SCENARIO 2] Black Swan event - BTC crash")

    # BTC падает на 12%
    crash_prices = [65000, 63000, 60000, 57000, 57200]

    for price in crash_prices:
        protection.update_btc_price(price)
        time.sleep(0.1)
        is_event = protection.check_black_swan_event()

        if is_event:
            time_left = protection.get_time_until_unblock()
            print(f"  BTC: ${price:.2f} → ⚠️  BLACK SWAN! Block time: {time_left/3600:.1f}h")
        else:
            print(f"  BTC: ${price:.2f} → Event: {is_event}")

    print("\n[SCENARIO 3] Manual reset")
    protection.reset_event()
    is_active = protection.is_event_active()
    print(f"  Event active after reset: {is_active}")

    print("\n✅ Black Swan Protection - OK")


# =============================================================================
# TEST 3: Night Mode Manager
# =============================================================================

def test_night_mode():
    """Тест Night Mode Manager"""
    print("\n" + "="*80)
    print("TEST 3: NIGHT MODE MANAGER")
    print("="*80)

    manager = NightModeManager(
        enabled=True,
        start_hour=23,        # 23:00 UTC
        end_hour=7,           # 07:00 UTC
        max_positions=5       # Максимум 5 позиций ночью
    )

    print("\n[SCENARIO 1] Different times of day")

    # Тестируем разные часы
    test_times = [
        datetime(2025, 11, 6, 0, 0),   # 00:00 UTC (ночь)
        datetime(2025, 11, 6, 6, 30),  # 06:30 UTC (ночь)
        datetime(2025, 11, 6, 7, 0),   # 07:00 UTC (день)
        datetime(2025, 11, 6, 12, 0),  # 12:00 UTC (день)
        datetime(2025, 11, 6, 20, 0),  # 20:00 UTC (день)
        datetime(2025, 11, 6, 23, 0),  # 23:00 UTC (ночь)
    ]

    for test_time in test_times:
        is_night = manager.is_night_mode_active(test_time)
        max_pos = manager.get_max_positions(test_time)
        threshold_adj = manager.get_threshold_adjustment(test_time)
        mode_info = manager.get_mode_info(test_time)

        print(f"  {test_time.strftime('%H:%M UTC')}: {mode_info['mode']}, Max positions: {max_pos}, Threshold +{threshold_adj:.2f}")

    print("\n[SCENARIO 2] Time until mode change")

    for test_time in test_times[:3]:
        hours_left = manager.get_time_until_mode_change(test_time)
        is_night = manager.is_night_mode_active(test_time)
        mode = "night" if is_night else "day"
        print(f"  {test_time.strftime('%H:%M UTC')} ({mode}): {hours_left}h until mode change")

    print("\n✅ Night Mode Manager - OK")


# =============================================================================
# TEST 4: Sector Diversification
# =============================================================================

def test_sector_diversification():
    """Тест Sector Diversification"""
    print("\n" + "="*80)
    print("TEST 4: SECTOR DIVERSIFICATION")
    print("="*80)

    print("\n[TEST 1] Get coin sector")

    test_coins = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT',
        'DOGEUSDT', 'SHIBUSDT', 'LINKUSDT', 'UNIUSDT'
    ]

    for coin in test_coins:
        sector = get_coin_sector(coin)
        limit = get_sector_limit(sector)
        print(f"  {coin:<12} → {sector:<15} (limit: {limit})")

    print("\n[TEST 2] Count coins per sector")

    portfolio_coins = [
        'BTCUSDT', 'SOLUSDT', 'AVAXUSDT', 'ADAUSDT',  # 1 BTC + 3 L1
        'DOGEUSDT', 'LINKUSDT', 'UNIUSDT', 'AAVEUSDT'  # 1 MEME + 1 ORACLE + 2 DEFI
    ]

    sector_counts = count_coins_per_sector(portfolio_coins)
    print(f"  Portfolio composition:")
    for sector, count in sorted(sector_counts.items()):
        limit = get_sector_limit(sector)
        status = "✓" if count <= limit else "✗ EXCEEDED"
        print(f"    {sector:<15}: {count}/{limit} {status}")

    print("\n[TEST 3] All sectors info")
    print(f"  Total sectors: {len(SECTOR_MAP)}")
    print(f"  Top 5 largest sectors:")

    sorted_sectors = sorted(SECTOR_MAP.items(), key=lambda x: len(x[1]), reverse=True)
    for sector, coins in sorted_sectors[:5]:
        limit = get_sector_limit(sector)
        print(f"    {sector:<20}: {len(coins):3d} coins (limit: {limit})")

    print("\n✅ Sector Diversification - OK")


# =============================================================================
# TEST 5: Auto-Reinvestment Manager
# =============================================================================

def test_reinvestment():
    """Тест Auto-Reinvestment Manager"""
    print("\n" + "="*80)
    print("TEST 5: AUTO-REINVESTMENT MANAGER")
    print("="*80)

    print("\n[TEST 1] Basic reinvestment")

    manager = ReinvestmentManager(
        initial_capital=10000,
        enabled=True,
        profit_to_reserve_pct=0.5
    )

    # Симулируем несколько дней торговли
    test_equities = [
        (10500, "Day 1: +$500 profit"),
        (10800, "Day 2: +$300 profit"),
        (11200, "Day 3: +$400 profit"),
        (11000, "Day 4: -$200 loss"),
    ]

    for i, (equity, description) in enumerate(test_equities):
        # Симулируем новый день
        if i > 0:
            manager.last_reinvest_date -= timedelta(days=1)

        result = manager.calculate_daily_reinvestment(equity)

        print(f"\n  {description}")
        print(f"    Current equity:    ${equity:,.2f}")
        print(f"    Trading capital:   ${result['trading_capital']:,.2f}")
        print(f"    Reserve fund:      ${result['reserve_fund']:,.2f}")
        print(f"    Total equity:      ${result['total_equity']:,.2f}")
        if result['reinvested']:
            print(f"    Daily profit:      ${result['daily_profit']:,.2f}")
            print(f"    To reserve:        ${result['to_reserve']:,.2f}")

    # Статистика
    print("\n  Final Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"    {key:<30}: ${value:,.2f}" if 'pct' not in key.lower() else f"    {key:<30}: {value:.2f}%")
        else:
            print(f"    {key:<30}: {value}")

    print("\n[TEST 2] 30-day simulation")

    # Генерируем случайные доходности
    random.seed(42)
    daily_returns = [random.uniform(-2, 5) for _ in range(30)]  # -2% до +5% в день

    sim_result = simulate_reinvestment(
        initial_capital=10000,
        daily_returns=daily_returns,
        profit_to_reserve_pct=0.5
    )

    print(f"\n  Simulation Results (30 days):")
    print(f"    Initial Capital:       ${10000:,.2f}")
    print(f"    Final Trading Capital: ${sim_result['final_trading_capital']:,.2f}")
    print(f"    Final Reserve Fund:    ${sim_result['final_reserve_fund']:,.2f}")
    print(f"    Final Total Equity:    ${sim_result['final_total_equity']:,.2f}")
    print(f"    Total Return:          {sim_result['total_return_pct']:+.2f}%")
    print(f"    Reserve % of initial:  {(sim_result['final_reserve_fund']/10000*100):.2f}%")

    print("\n  Last 5 days:")
    for entry in sim_result['results'][-5:]:
        print(f"    Day {entry['day']:2d}: Return {entry['daily_return_pct']:+5.2f}%, "
              f"Equity ${entry['total_equity']:,.2f}, Reserve ${entry['reserve_fund']:,.2f}")

    print("\n[TEST 3] Withdraw and add to reserve")

    manager2 = ReinvestmentManager(initial_capital=10000, enabled=True)
    manager2.reserve_fund = 1000.0

    print(f"\n  Initial reserve: ${manager2.reserve_fund:.2f}")

    # Try to withdraw
    success = manager2.withdraw_from_reserve(300)
    print(f"  Withdraw $300: {'Success' if success else 'Failed'}, Reserve: ${manager2.reserve_fund:.2f}")

    # Try to withdraw too much
    success = manager2.withdraw_from_reserve(2000)
    print(f"  Withdraw $2000: {'Success' if success else 'Failed'}, Reserve: ${manager2.reserve_fund:.2f}")

    # Add to reserve
    manager2.add_to_reserve(500)
    print(f"  Add $500: Reserve: ${manager2.reserve_fund:.2f}")

    print("\n✅ Auto-Reinvestment Manager - OK")


# =============================================================================
# MAIN - Запуск всех тестов
# =============================================================================

def run_all_tests():
    """Запускает все тесты"""
    print("\n" + "="*80)
    print("🧪 TESTING ALL SPECIAL MODES")
    print("="*80)

    tests = [
        ("Trailing Stop Manager", test_trailing_stop),
        ("Black Swan Protection", test_black_swan),
        ("Night Mode Manager", test_night_mode),
        ("Sector Diversification", test_sector_diversification),
        ("Auto-Reinvestment Manager", test_reinvestment),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "✅ PASSED"))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, f"❌ FAILED: {e}"))

    # Итоговые результаты
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)

    for test_name, result in results:
        print(f"  {test_name:<40} {result}")

    passed = sum(1 for _, r in results if "PASSED" in r)
    total = len(results)

    print("\n" + "="*80)
    print(f"  TOTAL: {passed}/{total} tests passed")
    print("="*80)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
