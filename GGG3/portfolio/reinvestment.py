#!/usr/bin/env python3
"""
Auto-Reinvestment Manager

Автоматическое управление реинвестированием прибыли.

Логика:
1. Весь рост капитала реинвестируется
2. 50% дневной прибыли → резервный фонд
3. Резерв для защиты от drawdown

Usage:
    from portfolio.reinvestment import ReinvestmentManager

    manager = ReinvestmentManager(initial_capital=10000)

    # Ежедневно
    result = manager.calculate_daily_reinvestment(current_equity=12500)

    print(f"Trading capital: ${result['trading_capital']:.2f}")
    print(f"Reserve fund: ${result['reserve_fund']:.2f}")
    print(f"Total equity: ${result['total_equity']:.2f}")

Автор: Claude Code
Дата: 2025-11-06
"""

from datetime import datetime, date
from typing import Dict, Optional
import logging
import json
from pathlib import Path

try:
    import binance_config as config
except ImportError:
    import sys
    from pathlib import Path
    # Добавляем путь относительно текущего файла (portfolio/ -> GGG3/)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import binance_config as config

logger = logging.getLogger(__name__)


class ReinvestmentManager:
    """
    Управление автоматическим реинвестированием

    Стратегия:
    - 100% прироста капитала реинвестируется
    - 30% дневной прибыли → резервный фонд (настраивается в config)
    - Резерв защищает от drawdown
    """

    def __init__(
        self,
        initial_capital: Optional[float] = None,
        enabled: Optional[bool] = None,
        profit_to_reserve_pct: Optional[float] = None,
        state_file: Optional[str] = None
    ):
        """
        Инициализация ReinvestmentManager

        Args:
            initial_capital: Начальный капитал (default from config)
            enabled: Включить авто-реинвестирование (default from config)
            profit_to_reserve_pct: Процент прибыли в резерв (default from config)
            state_file: Файл для сохранения состояния (default: portfolio/reinvestment_state.json)
        """
        self.initial_capital = initial_capital if initial_capital is not None else config.PAPER_INITIAL_BALANCE
        self.enabled = enabled if enabled is not None else getattr(config, 'AUTO_REINVEST_ENABLED', True)
        self.profit_to_reserve_pct = profit_to_reserve_pct if profit_to_reserve_pct is not None else getattr(config, 'REINVEST_PROFIT_TO_RESERVE_PCT', 0.5)

        # Путь к файлу состояния
        if state_file is None:
            state_file = str(Path(__file__).parent / "reinvestment_state.json")
        self.state_file = state_file

        # Состояние
        self.reserve_fund = 0.0
        self.last_reinvest_date = datetime.now().date()
        self.total_profit_accumulated = 0.0
        self.total_to_reserve_accumulated = 0.0
        self.daily_history = []  # История дневных результатов

        # Загружаем состояние из файла
        self._load_state()

        logger.info(
            f"ReinvestmentManager initialized: "
            f"initial_capital=${self.initial_capital:.2f}, "
            f"enabled={self.enabled}, "
            f"profit_to_reserve_pct={self.profit_to_reserve_pct*100:.0f}%"
        )

    def calculate_daily_reinvestment(self, current_equity: float) -> Dict[str, float]:
        """
        Рассчитывает ежедневное реинвестирование

        Args:
            current_equity: Текущий капитал (equity)

        Returns:
            dict с ключами:
                - 'trading_capital': Капитал для торговли
                - 'reserve_fund': Резервный фонд
                - 'total_equity': Общий капитал
                - 'daily_profit': Дневная прибыль (если был расчёт)
                - 'to_reserve': Сумма добавленная в резерв
                - 'reinvested': True если было реинвестирование
        """
        if not self.enabled:
            return {
                'trading_capital': current_equity,
                'reserve_fund': self.reserve_fund,
                'total_equity': current_equity + self.reserve_fund,
                'daily_profit': 0.0,
                'to_reserve': 0.0,
                'reinvested': False
            }

        today = datetime.now().date()

        # Проверяем, нужно ли делать ежедневный расчёт
        if today > self.last_reinvest_date:
            # Новый день - делаем расчёт

            # Рассчитываем дневную прибыль/убыток
            # Для упрощения считаем как отклонение от initial_capital
            daily_pnl = current_equity - self.initial_capital

            if daily_pnl > 0:
                # ПРИБЫЛЬНЫЙ ДЕНЬ: процент прибыли → резерв
                to_reserve = daily_pnl * self.profit_to_reserve_pct
                self.reserve_fund += to_reserve

                # Остальное реинвестируем (остаётся в trading_capital)
                trading_capital = current_equity - to_reserve

                # Обновляем статистику
                self.total_profit_accumulated += daily_pnl
                self.total_to_reserve_accumulated += to_reserve

                # Сохраняем в историю
                self.daily_history.append({
                    'date': str(today),
                    'daily_profit': daily_pnl,
                    'to_reserve': to_reserve,
                    'from_reserve': 0.0,
                    'reserve_fund': self.reserve_fund,
                    'trading_capital': trading_capital
                })

                logger.info(
                    f"[Reinvestment] Date: {today}, "
                    f"Daily profit: ${daily_pnl:.2f}, "
                    f"To reserve: ${to_reserve:.2f} ({self.profit_to_reserve_pct*100:.0f}%), "
                    f"Trading capital: ${trading_capital:.2f}, "
                    f"Reserve fund: ${self.reserve_fund:.2f}"
                )

                self.last_reinvest_date = today
                self._save_state()

                return {
                    'trading_capital': trading_capital,
                    'reserve_fund': self.reserve_fund,
                    'total_equity': trading_capital + self.reserve_fund,
                    'daily_profit': daily_pnl,
                    'to_reserve': to_reserve,
                    'reinvested': True
                }

            elif daily_pnl < 0:
                # УБЫТОЧНЫЙ ДЕНЬ: покрываем убыток из резерва
                daily_loss = abs(daily_pnl)

                # Покрываем убыток из резерва (но не больше чем есть)
                from_reserve = min(daily_loss, self.reserve_fund)
                self.reserve_fund -= from_reserve

                # Добавляем к торговому капиталу
                trading_capital = current_equity + from_reserve

                # Сохраняем в историю
                self.daily_history.append({
                    'date': str(today),
                    'daily_profit': daily_pnl,
                    'to_reserve': 0.0,
                    'from_reserve': from_reserve,
                    'reserve_fund': self.reserve_fund,
                    'trading_capital': trading_capital
                })

                logger.info(
                    f"[Reinvestment] Date: {today}, "
                    f"Daily loss: ${daily_loss:.2f}, "
                    f"Covered from reserve: ${from_reserve:.2f}, "
                    f"Trading capital: ${trading_capital:.2f}, "
                    f"Reserve fund: ${self.reserve_fund:.2f}"
                )

                self.last_reinvest_date = today
                self._save_state()

                return {
                    'trading_capital': trading_capital,
                    'reserve_fund': self.reserve_fund,
                    'total_equity': trading_capital + self.reserve_fund,
                    'daily_profit': daily_pnl,
                    'to_reserve': 0.0,
                    'from_reserve': from_reserve,
                    'reinvested': True
                }

            else:
                # БЕЗ ИЗМЕНЕНИЙ (daily_pnl == 0)
                self.last_reinvest_date = today
                self._save_state()

                return {
                    'trading_capital': current_equity,
                    'reserve_fund': self.reserve_fund,
                    'total_equity': current_equity + self.reserve_fund,
                    'daily_profit': 0.0,
                    'to_reserve': 0.0,
                    'reinvested': False
                }

        # Тот же день - возвращаем текущие значения
        return {
            'trading_capital': current_equity,
            'reserve_fund': self.reserve_fund,
            'total_equity': current_equity + self.reserve_fund,
            'daily_profit': 0.0,
            'to_reserve': 0.0,
            'reinvested': False
        }

    def get_statistics(self) -> Dict:
        """
        Возвращает статистику реинвестирования

        Returns:
            dict со статистикой
        """
        return {
            'initial_capital': self.initial_capital,
            'reserve_fund': self.reserve_fund,
            'total_profit_accumulated': self.total_profit_accumulated,
            'total_to_reserve_accumulated': self.total_to_reserve_accumulated,
            'reserve_pct_of_initial': (self.reserve_fund / self.initial_capital * 100) if self.initial_capital > 0 else 0,
            'days_tracked': len(self.daily_history),
            'last_reinvest_date': str(self.last_reinvest_date)
        }

    def withdraw_from_reserve(self, amount: float) -> bool:
        """
        Снять средства из резервного фонда

        Args:
            amount: Сумма для снятия

        Returns:
            True если успешно, False если недостаточно средств
        """
        if amount <= 0:
            logger.warning(f"Invalid withdrawal amount: ${amount:.2f}")
            return False

        if amount > self.reserve_fund:
            logger.warning(
                f"Insufficient reserve fund: requested ${amount:.2f}, "
                f"available ${self.reserve_fund:.2f}"
            )
            return False

        self.reserve_fund -= amount
        logger.info(f"Withdrawn ${amount:.2f} from reserve fund. Remaining: ${self.reserve_fund:.2f}")
        self._save_state()

        return True

    def add_to_reserve(self, amount: float):
        """
        Добавить средства в резервный фонд (вручную)

        Args:
            amount: Сумма для добавления
        """
        if amount <= 0:
            logger.warning(f"Invalid deposit amount: ${amount:.2f}")
            return

        self.reserve_fund += amount
        logger.info(f"Added ${amount:.2f} to reserve fund. Total: ${self.reserve_fund:.2f}")
        self._save_state()

    def reset(self, new_initial_capital: Optional[float] = None):
        """
        Сброс состояния (для новой торговой сессии)

        Args:
            new_initial_capital: Новый начальный капитал (опционально)
        """
        if new_initial_capital is not None:
            self.initial_capital = new_initial_capital

        self.reserve_fund = 0.0
        self.last_reinvest_date = datetime.now().date()
        self.total_profit_accumulated = 0.0
        self.total_to_reserve_accumulated = 0.0
        self.daily_history = []

        logger.info(f"ReinvestmentManager reset. New initial capital: ${self.initial_capital:.2f}")
        self._save_state()

    def _save_state(self):
        """Сохраняет состояние в файл"""
        try:
            state = {
                'initial_capital': self.initial_capital,
                'reserve_fund': self.reserve_fund,
                'last_reinvest_date': str(self.last_reinvest_date),
                'total_profit_accumulated': self.total_profit_accumulated,
                'total_to_reserve_accumulated': self.total_to_reserve_accumulated,
                'daily_history': self.daily_history[-30:]  # Сохраняем последние 30 дней
            }

            # Создаём директорию если не существует
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"Reinvestment state saved to {self.state_file}")

        except Exception as e:
            logger.error(f"Failed to save reinvestment state: {e}")

    def _load_state(self):
        """Загружает состояние из файла"""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                self.initial_capital = state.get('initial_capital', self.initial_capital)
                self.reserve_fund = state.get('reserve_fund', 0.0)
                self.last_reinvest_date = datetime.fromisoformat(
                    state.get('last_reinvest_date', str(datetime.now().date()))
                ).date()
                self.total_profit_accumulated = state.get('total_profit_accumulated', 0.0)
                self.total_to_reserve_accumulated = state.get('total_to_reserve_accumulated', 0.0)
                self.daily_history = state.get('daily_history', [])

                logger.info(
                    f"Reinvestment state loaded from {self.state_file}: "
                    f"reserve_fund=${self.reserve_fund:.2f}, "
                    f"total_profit=${self.total_profit_accumulated:.2f}"
                )
            else:
                logger.info("No existing reinvestment state file found, starting fresh")

        except Exception as e:
            logger.error(f"Failed to load reinvestment state: {e}")
            logger.info("Starting with fresh reinvestment state")

    def get_daily_history(self, days: int = 30) -> list:
        """
        Возвращает историю последних N дней

        Args:
            days: Количество дней

        Returns:
            Список записей истории
        """
        return self.daily_history[-days:]

    def print_summary(self, current_balance: Optional[float] = None):
        """
        Выводит краткую сводку

        Args:
            current_balance: Текущий баланс (для отображения актуального состояния)
        """
        stats = self.get_statistics()

        print("\n" + "="*80)
        print("REINVESTMENT SUMMARY")
        print("="*80)
        print(f"Initial Capital:      ${stats['initial_capital']:,.2f}")

        # Если передан текущий баланс - показываем актуальное состояние
        if current_balance is not None:
            net_change = current_balance - stats['initial_capital']
            roi_pct = (net_change / stats['initial_capital'] * 100) if stats['initial_capital'] > 0 else 0
            print(f"Current Balance:      ${current_balance:,.2f}")
            print(f"Net Change:           ${net_change:+,.2f} ({roi_pct:+.2f}%)")

        print(f"Reserve Fund:         ${stats['reserve_fund']:,.2f} ({stats['reserve_pct_of_initial']:.1f}% of initial)")

        # Если есть резерв, показываем total equity
        if current_balance is not None and stats['reserve_fund'] > 0:
            total_equity = current_balance + stats['reserve_fund']
            print(f"Total Equity:         ${total_equity:,.2f} (balance + reserve)")

        print(f"Total Profit:         ${stats['total_profit_accumulated']:,.2f}")
        print(f"Total to Reserve:     ${stats['total_to_reserve_accumulated']:,.2f}")
        print(f"Days Tracked:         {stats['days_tracked']}")
        print(f"Last Reinvest Date:   {stats['last_reinvest_date']}")
        print("="*80)

        if self.daily_history:
            print("\nRecent History (last 7 days):")
            for entry in self.daily_history[-7:]:
                print(
                    f"  {entry['date']}: "
                    f"Profit=${entry['daily_profit']:.2f}, "
                    f"To Reserve=${entry['to_reserve']:.2f}, "
                    f"Reserve=${entry['reserve_fund']:.2f}"
                )

        print()


# =============================================================================
# Вспомогательные функции
# =============================================================================

def simulate_reinvestment(
    initial_capital: float,
    daily_returns: list,
    profit_to_reserve_pct: float = 0.5
) -> Dict:
    """
    Симулирует реинвестирование для списка дневных доходностей

    Args:
        initial_capital: Начальный капитал
        daily_returns: Список дневных доходностей (в процентах)
        profit_to_reserve_pct: Процент прибыли в резерв

    Returns:
        dict с результатами симуляции
    """
    manager = ReinvestmentManager(
        initial_capital=initial_capital,
        enabled=True,
        profit_to_reserve_pct=profit_to_reserve_pct
    )

    trading_capital = initial_capital
    results = []

    for day, daily_return_pct in enumerate(daily_returns, 1):
        # Применяем дневную доходность
        daily_profit = trading_capital * (daily_return_pct / 100)
        current_equity = trading_capital + daily_profit

        # Рассчитываем реинвестирование
        reinvest_result = manager.calculate_daily_reinvestment(current_equity)

        # Обновляем trading_capital для следующего дня
        trading_capital = reinvest_result['trading_capital']

        results.append({
            'day': day,
            'daily_return_pct': daily_return_pct,
            'equity': current_equity,
            'trading_capital': trading_capital,
            'reserve_fund': reinvest_result['reserve_fund'],
            'total_equity': reinvest_result['total_equity']
        })

    return {
        'results': results,
        'final_trading_capital': trading_capital,
        'final_reserve_fund': manager.reserve_fund,
        'final_total_equity': trading_capital + manager.reserve_fund,
        'total_return_pct': ((trading_capital + manager.reserve_fund - initial_capital) / initial_capital) * 100,
        'statistics': manager.get_statistics()
    }


# =============================================================================
# Main для тестирования
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("REINVESTMENT MANAGER TEST")
    print("="*80)

    # Тест 1: Базовый тест
    print("\n[TEST 1] Basic reinvestment")
    manager = ReinvestmentManager(initial_capital=10000, enabled=True, profit_to_reserve_pct=0.5)

    # День 1: прибыль $500
    result = manager.calculate_daily_reinvestment(current_equity=10500)
    print(f"Day 1: Profit=$500, To Reserve=${result['to_reserve']:.2f}, Trading=${result['trading_capital']:.2f}")

    # День 2: прибыль $300
    result = manager.calculate_daily_reinvestment(current_equity=10800)
    print(f"Day 2: Profit=$300, To Reserve=${result['to_reserve']:.2f}, Trading=${result['trading_capital']:.2f}")

    manager.print_summary()

    # Тест 2: Симуляция 30 дней с случайными доходностями
    print("\n[TEST 2] 30-day simulation")
    import random
    random.seed(42)

    daily_returns = [random.uniform(-2, 5) for _ in range(30)]  # -2% до +5% в день

    sim_result = simulate_reinvestment(
        initial_capital=10000,
        daily_returns=daily_returns,
        profit_to_reserve_pct=0.5
    )

    print(f"\nSimulation Results:")
    print(f"  Initial Capital:      ${10000:,.2f}")
    print(f"  Final Trading Capital: ${sim_result['final_trading_capital']:,.2f}")
    print(f"  Final Reserve Fund:    ${sim_result['final_reserve_fund']:,.2f}")
    print(f"  Final Total Equity:    ${sim_result['final_total_equity']:,.2f}")
    print(f"  Total Return:          {sim_result['total_return_pct']:.2f}%")

    print("\n✅ All tests completed!")
