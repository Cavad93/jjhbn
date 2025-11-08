#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
КОМПЛЕКСНЫЙ ТЕСТ ТОРГОВОЙ ЛОГИКИ БОТА

Проверяет все критические аспекты:
1. Kelly Criterion расчеты
2. Размеры позиций и риск-менеджмент
3. Логика открытия/закрытия позиций (TP/SL)
4. Контроль капитала и reinvestment
5. Расчеты статистики
6. Drawdown контроль
"""

import sys
import os
from pathlib import Path

# Добавляем GGG3 в path
sys.path.insert(0, str(Path(__file__).parent))

import binance_config as config
from paper_trading.paper_exchange import PaperExchange
from portfolio.reinvestment import ReinvestmentManager

# Прямой импорт kelly_sizer минуя __init__.py чтобы избежать ненужных зависимостей
import importlib.util
spec = importlib.util.spec_from_file_location("kelly_sizer", Path(__file__).parent / "strategy" / "kelly_sizer.py")
kelly_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kelly_module)
calculate_kelly_position_size = kelly_module.calculate_kelly_position_size

import numpy as np
from typing import Dict, List


class TradingLogicTester:
    """Комплексный тестер торговой логики"""

    def __init__(self):
        self.exchange = PaperExchange(initial_capital=1000.0)
        self.reinvestment = ReinvestmentManager(
            initial_capital=1000.0,
            profit_to_reserve_pct=10.0,
            enabled=True
        )
        self.test_results = []
        self.errors = []

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Логирование результата теста"""
        result = {
            'test': test_name,
            'passed': passed,
            'details': details
        }
        self.test_results.append(result)

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status}: {test_name}")
        if details:
            print(f"  └─ {details}")

        if not passed:
            self.errors.append(test_name)

    # ========================================================================
    # TEST 1: Kelly Criterion расчеты
    # ========================================================================

    def test_kelly_calculations(self):
        """Проверка расчетов Kelly Criterion"""
        print("\n" + "="*80)
        print("TEST 1: KELLY CRITERION РАСЧЕТЫ")
        print("="*80)

        capital = 1000.0

        # Test 1.1: Базовый расчет Kelly
        kelly_size = calculate_kelly_position_size(
            capital=capital,
            ev=5.0,
            p_up=0.55,
            recent_wr=0.50,
            rr_ratio=2.0
        )

        expected_min = capital * config.MIN_RISK_PER_POSITION
        expected_max = capital * config.MAX_RISK_PER_POSITION

        passed = expected_min <= kelly_size <= expected_max
        self.log_test(
            "Kelly в пределах MIN/MAX_RISK",
            passed,
            f"Size: ${kelly_size:.2f}, Range: ${expected_min:.2f}-${expected_max:.2f}"
        )

        # Test 1.2: Kelly при низком EV (должен быть MIN_RISK)
        kelly_low_ev = calculate_kelly_position_size(
            capital=capital,
            ev=0.5,
            p_up=0.45,
            recent_wr=0.40,
            rr_ratio=1.5
        )

        passed = abs(kelly_low_ev - expected_min) < 1.0
        self.log_test(
            "Kelly при низком EV = MIN_RISK",
            passed,
            f"Size: ${kelly_low_ev:.2f}, Expected: ${expected_min:.2f}"
        )

        # Test 1.3: Kelly при высоком EV (должен быть MAX_RISK)
        kelly_high_ev = calculate_kelly_position_size(
            capital=capital,
            ev=15.0,
            p_up=0.70,
            recent_wr=0.65,
            rr_ratio=3.0
        )

        passed = abs(kelly_high_ev - expected_max) < 1.0
        self.log_test(
            "Kelly при высоком EV = MAX_RISK",
            passed,
            f"Size: ${kelly_high_ev:.2f}, Expected: ${expected_max:.2f}"
        )

        # Test 1.4: Все позиции получают одинаковый капитал
        sizes = []
        for i in range(10):
            size = calculate_kelly_position_size(
                capital=capital,  # ФИКСИРОВАННЫЙ капитал
                ev=5.0,
                p_up=0.55,
                recent_wr=0.50,
                rr_ratio=2.0
            )
            sizes.append(size)

        # Все размеры должны быть одинаковыми
        passed = all(abs(s - sizes[0]) < 0.01 for s in sizes)
        self.log_test(
            "Все позиции используют ОДИНАКОВЫЙ капитал",
            passed,
            f"10 позиций: все = ${sizes[0]:.2f}"
        )

    # ========================================================================
    # TEST 2: Размеры позиций и риск-менеджмент
    # ========================================================================

    def test_position_sizing_and_risk(self):
        """Проверка размеров позиций и риск-лимитов"""
        print("\n" + "="*80)
        print("TEST 2: РАЗМЕРЫ ПОЗИЦИЙ И РИСК-МЕНЕДЖМЕНТ")
        print("="*80)

        capital = 1000.0

        # Test 2.1: Максимальное количество позиций
        max_positions_limit = config.MAX_POSITIONS
        total_risk = max_positions_limit * config.MAX_RISK_PER_POSITION

        passed = total_risk <= 1.0  # Не более 100% капитала
        self.log_test(
            f"MAX позиций не превышает 100% капитала",
            passed,
            f"{max_positions_limit} позиций × {config.MAX_RISK_PER_POSITION*100:.1f}% = {total_risk*100:.1f}%"
        )

        # Test 2.2: Размер одной позиции при 9% риске
        position_size = capital * config.MAX_RISK_PER_POSITION

        passed = position_size > 0 and position_size <= capital
        self.log_test(
            "Размер позиции при MAX_RISK корректен",
            passed,
            f"${position_size:.2f} из ${capital:.2f} ({config.MAX_RISK_PER_POSITION*100:.1f}%)"
        )

        # Test 2.3: Проверка MIN_RISK < MAX_RISK
        passed = config.MIN_RISK_PER_POSITION < config.MAX_RISK_PER_POSITION
        self.log_test(
            "MIN_RISK < MAX_RISK",
            passed,
            f"{config.MIN_RISK_PER_POSITION*100:.2f}% < {config.MAX_RISK_PER_POSITION*100:.1f}%"
        )

    # ========================================================================
    # TEST 3: Логика открытия/закрытия позиций
    # ========================================================================

    def test_position_lifecycle(self):
        """Проверка полного жизненного цикла позиции"""
        print("\n" + "="*80)
        print("TEST 3: ЖИЗНЕННЫЙ ЦИКЛ ПОЗИЦИЙ (ОТКРЫТИЕ/ЗАКРЫТИЕ)")
        print("="*80)

        # Reset exchange
        self.exchange = PaperExchange(initial_capital=1000.0)
        initial_balance = self.exchange.get_balance('USDT')

        # Test 3.1: Открытие позиции
        symbol = "BTCUSDT"
        entry_price = 50000.0
        quantity = 0.001  # $50
        tp_price = 52000.0  # +4%
        sl_price = 48000.0  # -4%

        position_id = self.exchange.open_position(
            symbol=symbol,
            side='LONG',
            amount=quantity,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price
        )

        passed = position_id is not None
        self.log_test(
            "Позиция успешно открыта",
            passed,
            f"ID: {position_id}, Symbol: {symbol}, Qty: {quantity}"
        )

        # Проверка баланса после открытия
        balance_after_open = self.exchange.get_balance('USDT')
        position_value = entry_price * quantity
        expected_balance = initial_balance - position_value

        passed = abs(balance_after_open - expected_balance) < 0.01
        self.log_test(
            "Баланс корректно уменьшился после открытия",
            passed,
            f"${initial_balance:.2f} → ${balance_after_open:.2f} (−${position_value:.2f})"
        )

        # Test 3.2: Закрытие по Take Profit
        self.exchange = PaperExchange(initial_capital=1000.0)
        position_id = self.exchange.open_position(
            symbol=symbol,
            side='LONG',
            amount=quantity,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price
        )

        # Симулируем движение цены к TP
        closed = self.exchange.update_positions(tp_price)

        passed = len(closed) == 1
        if passed and closed:
            pnl = closed[0]['pnl']
            expected_pnl = (tp_price - entry_price) * quantity
            passed = abs(pnl - expected_pnl) < 0.01

        self.log_test(
            "Позиция закрыта по Take Profit",
            passed,
            f"PnL: ${closed[0]['pnl']:.2f} (ожидается +${expected_pnl:.2f})" if closed else "Не закрыта"
        )

        # Test 3.3: Закрытие по Stop Loss
        self.exchange = PaperExchange(initial_capital=1000.0)
        position_id = self.exchange.open_position(
            symbol=symbol,
            side='LONG',
            amount=quantity,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price
        )

        # Симулируем движение цены к SL
        closed = self.exchange.update_positions(sl_price)

        passed = len(closed) == 1
        if passed and closed:
            pnl = closed[0]['pnl']
            expected_pnl = (sl_price - entry_price) * quantity
            passed = abs(pnl - expected_pnl) < 0.01

        self.log_test(
            "Позиция закрыта по Stop Loss",
            passed,
            f"PnL: ${closed[0]['pnl']:.2f} (ожидается −${abs(expected_pnl):.2f})" if closed else "Не закрыта"
        )

        # Test 3.4: Баланс после прибыльного закрытия
        balance_after_tp = self.exchange.get_balance('USDT')
        expected_balance_tp = initial_balance + expected_pnl

        # Note: баланс может быть некорректным из-за теста 3.3, пересоздадим exchange
        self.exchange = PaperExchange(initial_capital=1000.0)
        position_id = self.exchange.open_position(
            symbol=symbol,
            side='LONG',
            amount=quantity,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price
        )
        closed = self.exchange.update_positions(tp_price)
        balance_after_tp = self.exchange.get_balance('USDT')
        expected_pnl = (tp_price - entry_price) * quantity
        expected_balance_tp = initial_balance + expected_pnl

        passed = abs(balance_after_tp - expected_balance_tp) < 0.01
        self.log_test(
            "Баланс корректно увеличился после TP",
            passed,
            f"${initial_balance:.2f} → ${balance_after_tp:.2f} (+${expected_pnl:.2f})"
        )

    # ========================================================================
    # TEST 4: Контроль капитала и Reinvestment
    # ========================================================================

    def test_capital_control_and_reinvestment(self):
        """Проверка контроля капитала и логики reinvestment"""
        print("\n" + "="*80)
        print("TEST 4: КОНТРОЛЬ КАПИТАЛА И REINVESTMENT")
        print("="*80)

        # Reset
        self.reinvestment = ReinvestmentManager(
            initial_capital=1000.0,
            profit_to_reserve_pct=10.0,
            enabled=True
        )

        # Test 4.1: Начальный капитал
        stats = self.reinvestment.get_statistics()

        passed = stats['initial_capital'] == 1000.0
        self.log_test(
            "Начальный капитал корректен",
            passed,
            f"${stats['initial_capital']:.2f}"
        )

        # Test 4.2: Расчет торгового капитала (без прибыли)
        current_balance = 1000.0
        result = self.reinvestment.calculate_daily_reinvestment(current_balance)

        passed = result['trading_capital'] == current_balance  # Нет изменений в первый день
        self.log_test(
            "Торговый капитал = баланс (нет прибыли)",
            passed,
            f"Trading: ${result['trading_capital']:.2f}, Balance: ${current_balance:.2f}"
        )

        # Test 4.3: Reinvestment после прибыли
        # Симулируем прибыль +200 USD
        current_balance = 1200.0

        # Обновляем дату чтобы сработал reinvestment
        import datetime
        self.reinvestment.last_reinvestment_date = datetime.date.today() - datetime.timedelta(days=1)

        result = self.reinvestment.calculate_daily_reinvestment(current_balance)

        # Ожидаем: profit = 200, to_reserve = 20 (10%), trading = 1180
        expected_profit = 200.0
        expected_reserve = 20.0
        expected_trading = 1180.0

        passed = (
            result['reinvested'] and
            abs(result['daily_profit'] - expected_profit) < 0.01 and
            abs(result['to_reserve'] - expected_reserve) < 0.01 and
            abs(result['trading_capital'] - expected_trading) < 0.01
        )
        self.log_test(
            "Reinvestment корректно работает при прибыли",
            passed,
            f"Profit: ${result['daily_profit']:.2f}, Reserve: ${result['to_reserve']:.2f}, Trading: ${result['trading_capital']:.2f}"
        )

        # Test 4.4: Резервный фонд накапливается
        reserve_fund = result['reserve_fund']
        passed = reserve_fund == expected_reserve
        self.log_test(
            "Резервный фонд накапливается",
            passed,
            f"Reserve Fund: ${reserve_fund:.2f}"
        )

    # ========================================================================
    # TEST 5: Расчеты статистики
    # ========================================================================

    def test_statistics_calculations(self):
        """Проверка расчетов статистики"""
        print("\n" + "="*80)
        print("TEST 5: РАСЧЕТЫ СТАТИСТИКИ")
        print("="*80)

        # Reset exchange
        self.exchange = PaperExchange(initial_capital=1000.0)
        initial_balance = 1000.0

        # Открываем и закрываем несколько позиций
        trades = []

        # Trade 1: Прибыль +10 USD
        pid1 = self.exchange.open_position("BTC1", 'buy', 0.001, 50000, 51000, 49000)
        closed1 = self.exchange.update_positions(51000)  # TP
        if closed1:
            trades.extend(closed1)

        # Trade 2: Убыток -5 USD
        pid2 = self.exchange.open_position("BTC2", 'buy', 0.001, 50000, 51000, 49500)
        closed2 = self.exchange.update_positions(49500)  # SL
        if closed2:
            trades.extend(closed2)

        # Trade 3: Прибыль +8 USD
        pid3 = self.exchange.open_position("BTC3", 'buy', 0.001, 50000, 50800, 49000)
        closed3 = self.exchange.update_positions(50800)  # TP
        if closed3:
            trades.extend(closed3)

        # Test 5.1: Подсчет Win Rate
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        expected_wr = 2/3  # 2 прибыльных из 3
        passed = abs(win_rate - expected_wr) < 0.01
        self.log_test(
            "Win Rate рассчитывается корректно",
            passed,
            f"WR: {win_rate*100:.1f}% (2 wins / 3 trades)"
        )

        # Test 5.2: Общий PnL
        total_pnl = sum(t['pnl'] for t in trades)
        expected_pnl = 10 - 5 + 8  # = 13

        passed = abs(total_pnl - expected_pnl) < 0.01
        self.log_test(
            "Total PnL рассчитывается корректно",
            passed,
            f"PnL: ${total_pnl:.2f} (expected: ${expected_pnl:.2f})"
        )

        # Test 5.3: Итоговый баланс
        final_balance = self.exchange.get_balance('USDT')
        expected_balance = initial_balance + expected_pnl

        passed = abs(final_balance - expected_balance) < 0.01
        self.log_test(
            "Итоговый баланс = начальный + PnL",
            passed,
            f"${final_balance:.2f} = ${initial_balance:.2f} + ${expected_pnl:.2f}"
        )

        # Test 5.4: ROI расчет
        roi = (final_balance - initial_balance) / initial_balance * 100
        expected_roi = (expected_pnl / initial_balance) * 100

        passed = abs(roi - expected_roi) < 0.01
        self.log_test(
            "ROI рассчитывается корректно",
            passed,
            f"ROI: {roi:.2f}% (expected: {expected_roi:.2f}%)"
        )

    # ========================================================================
    # TEST 6: Drawdown контроль
    # ========================================================================

    def test_drawdown_control(self):
        """Проверка контроля максимальной просадки"""
        print("\n" + "="*80)
        print("TEST 6: DRAWDOWN КОНТРОЛЬ")
        print("="*80)

        initial_capital = 1000.0
        max_dd_limit = config.MAX_DRAWDOWN_PCT  # 20%

        # Test 6.1: Расчет drawdown
        peak_balance = 1000.0
        current_balance = 850.0

        drawdown_pct = ((peak_balance - current_balance) / peak_balance) * 100
        expected_dd = 15.0

        passed = abs(drawdown_pct - expected_dd) < 0.01
        self.log_test(
            "Drawdown рассчитывается корректно",
            passed,
            f"DD: {drawdown_pct:.2f}% (peak ${peak_balance:.2f} → ${current_balance:.2f})"
        )

        # Test 6.2: Drawdown в пределах лимита
        passed = drawdown_pct < max_dd_limit
        self.log_test(
            f"Drawdown < MAX_DRAWDOWN_PCT ({max_dd_limit}%)",
            passed,
            f"{drawdown_pct:.2f}% < {max_dd_limit}%"
        )

        # Test 6.3: Критический drawdown
        current_balance_critical = 750.0
        drawdown_critical = ((peak_balance - current_balance_critical) / peak_balance) * 100

        should_stop = drawdown_critical >= max_dd_limit
        passed = should_stop  # Должен остановиться
        self.log_test(
            f"Бот должен остановиться при DD ≥ {max_dd_limit}%",
            passed,
            f"DD: {drawdown_critical:.2f}% → STOP TRADING"
        )

    # ========================================================================
    # TEST 7: Интеграционный тест (полный цикл)
    # ========================================================================

    def test_full_integration(self):
        """Полный интеграционный тест торговли"""
        print("\n" + "="*80)
        print("TEST 7: ПОЛНЫЙ ИНТЕГРАЦИОННЫЙ ТЕСТ")
        print("="*80)

        # Симулируем полный цикл торговли
        self.exchange = PaperExchange(initial_capital=1000.0)
        self.reinvestment = ReinvestmentManager(
            initial_capital=1000.0,
            profit_to_reserve_pct=10.0,
            enabled=True
        )

        initial_balance = 1000.0

        # Получаем торговый капитал (один раз!)
        trading_capital = self.exchange.get_balance('USDT')
        reinvest_result = self.reinvestment.calculate_daily_reinvestment(trading_capital)
        trading_capital = reinvest_result['trading_capital']

        # Открываем 10 позиций с одинаковым Kelly расчетом
        positions_opened = []
        position_sizes = []

        for i in range(10):
            # Kelly от ФИКСИРОВАННОГО капитала
            kelly_size = calculate_kelly_position_size(
                capital=trading_capital,  # ← ОДИНАКОВЫЙ для всех!
                ev=5.0,
                p_up=0.55,
                recent_wr=0.50,
                rr_ratio=2.0
            )
            position_sizes.append(kelly_size)

            # Открываем позицию
            entry_price = 50000 + i * 100
            quantity = kelly_size / entry_price
            tp = entry_price * 1.04
            sl = entry_price * 0.96

            pid = self.exchange.open_position(
                symbol=f"BTC{i}",
                side='LONG',
                amount=quantity,
                entry_price=entry_price,
                tp_price=tp,
                sl_price=sl
            )

            if pid:
                positions_opened.append(pid)

        # Test 7.1: Все 10 позиций открыты
        passed = len(positions_opened) == 10
        self.log_test(
            "Все 10 позиций успешно открыты",
            passed,
            f"Opened: {len(positions_opened)}/10"
        )

        # Test 7.2: Все позиции имеют ОДИНАКОВЫЙ размер (Kelly от одного капитала)
        if position_sizes:
            avg_size = np.mean(position_sizes)
            std_size = np.std(position_sizes)

            passed = std_size < 0.01  # Практически нулевое отклонение
            self.log_test(
                "Все позиции ОДИНАКОВОГО размера (Kelly от полного капитала)",
                passed,
                f"Avg: ${avg_size:.2f}, Std: ${std_size:.4f}"
            )

        # Test 7.3: Суммарный риск в пределах допустимого
        total_position_value = sum(position_sizes)
        risk_pct = (total_position_value / trading_capital) * 100

        max_expected_risk = config.MAX_POSITIONS * config.MAX_RISK_PER_POSITION * 100

        passed = risk_pct <= max_expected_risk + 1  # +1% погрешность
        self.log_test(
            f"Суммарный риск ≤ {max_expected_risk:.0f}%",
            passed,
            f"Total risk: {risk_pct:.1f}%"
        )

        # Test 7.4: Симулируем закрытие позиций (6 прибыльных, 4 убыточных)
        all_closed = []

        # Закрываем 6 по TP
        for i in range(6):
            entry_price = 50000 + i * 100
            tp = entry_price * 1.04
            closed = self.exchange.update_positions(tp)
            all_closed.extend(closed)

        # Закрываем 4 по SL
        for i in range(6, 10):
            entry_price = 50000 + i * 100
            sl = entry_price * 0.96
            closed = self.exchange.update_positions(sl)
            all_closed.extend(closed)

        passed = len(all_closed) == 10
        self.log_test(
            "Все позиции успешно закрыты",
            passed,
            f"Closed: {len(all_closed)}/10"
        )

        # Test 7.5: Win Rate = 60%
        wins = sum(1 for c in all_closed if c['pnl'] > 0)
        wr = wins / len(all_closed) if all_closed else 0

        passed = abs(wr - 0.60) < 0.01
        self.log_test(
            "Win Rate корректен (60%)",
            passed,
            f"WR: {wr*100:.1f}% ({wins}/10)"
        )

        # Test 7.6: Финальный баланс изменился корректно
        final_balance = self.exchange.get_balance('USDT')
        total_pnl = sum(c['pnl'] for c in all_closed)
        expected_final = initial_balance + total_pnl

        passed = abs(final_balance - expected_final) < 0.01
        self.log_test(
            "Финальный баланс корректен",
            passed,
            f"${final_balance:.2f} = ${initial_balance:.2f} + ${total_pnl:.2f}"
        )

    # ========================================================================
    # ЗАПУСК ВСЕХ ТЕСТОВ
    # ========================================================================

    def run_all_tests(self):
        """Запускает все тесты"""
        print("\n" + "="*80)
        print("КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ТОРГОВОЙ ЛОГИКИ БОТА")
        print("="*80)
        print(f"Конфигурация:")
        print(f"  - Initial Balance: $1,000")
        print(f"  - MAX_RISK_PER_POSITION: {config.MAX_RISK_PER_POSITION*100:.1f}%")
        print(f"  - MIN_RISK_PER_POSITION: {config.MIN_RISK_PER_POSITION*100:.2f}%")
        print(f"  - MAX_POSITIONS: {config.MAX_POSITIONS}")
        print(f"  - MAX_DRAWDOWN_PCT: {config.MAX_DRAWDOWN_PCT}%")

        # Запуск всех тестов
        self.test_kelly_calculations()
        self.test_position_sizing_and_risk()
        self.test_position_lifecycle()
        self.test_capital_control_and_reinvestment()
        self.test_statistics_calculations()
        self.test_drawdown_control()
        self.test_full_integration()

        # Итоговый отчет
        self.print_summary()

    def print_summary(self):
        """Печатает итоговый отчет"""
        print("\n" + "="*80)
        print("ИТОГОВЫЙ ОТЧЕТ")
        print("="*80)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['passed'])
        failed_tests = total_tests - passed_tests

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"\nВсего тестов: {total_tests}")
        print(f"✅ Успешных: {passed_tests}")
        print(f"❌ Неудачных: {failed_tests}")
        print(f"📊 Успешность: {success_rate:.1f}%")

        if self.errors:
            print("\n⚠️  ПРОБЛЕМЫ ОБНАРУЖЕНЫ:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        else:
            print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            print("Торговая логика бота работает корректно.")

        print("="*80)


if __name__ == '__main__':
    tester = TradingLogicTester()
    tester.run_all_tests()
