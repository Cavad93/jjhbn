#!/usr/bin/env python3
"""
Performance Tracker

Отслеживает и анализирует производительность торговой стратегии.

Метрики:
- Win Rate (текущий и необходимый)
- Геометрическая прогрессия
- 90% доверительный интервал
- Profit Factor
- Expectancy
- Average Win/Loss
- Max Drawdown
- Consecutive Wins/Losses
"""

import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import csv
import json
import os


@dataclass
class TradeStats:
    """Статистика одной сделки"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    entry_time: datetime
    exit_time: datetime
    duration_hours: float

    def is_win(self) -> bool:
        """Является ли сделка прибыльной"""
        return self.pnl_pct > 0


@dataclass
class PerformanceMetrics:
    """Все метрики производительности"""

    # Базовые метрики
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    required_win_rate: float

    # Геометрическая прогрессия
    geometric_mean_return: float  # Среднее геометрическое доходности
    compound_growth_rate: float   # Совокупный темп роста

    # Доверительный интервал (90%)
    ci_lower: float
    ci_upper: float

    # Качественные метрики
    profit_factor: float
    expectancy: float  # Математическое ожидание прибыли на сделку
    avg_win: float
    avg_loss: float
    avg_win_loss_ratio: float  # Реальный R:R

    # Drawdown
    max_drawdown_pct: float
    current_drawdown_pct: float

    # Серии
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int  # Текущая серия (+ wins, - losses)

    # Капитал
    starting_capital: float
    ending_capital: float
    total_return_pct: float

    # Статус
    all_conditions_met: bool
    conditions: Dict[str, bool] = field(default_factory=dict)

    # Временные метки
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class PerformanceTracker:
    """
    Трекер производительности торговой стратегии

    Поддерживает:
    - Множественные окна анализа (30/50/100/все)
    - Расчет всех метрик
    - Сохранение истории
    """

    def __init__(
        self,
        windows: Dict[str, int] = None,
        history_dir: str = 'data/performance',
        enable_history: bool = True
    ):
        """
        Args:
            windows: Окна анализа {'short': 50, 'medium': 100, 'all': None}
            history_dir: Директория для сохранения истории
            enable_history: Включить сохранение истории
        """
        self.windows = windows or {
            'short': 50,
            'medium': 100,
            'long': 200,
            'all': None  # Все сделки
        }

        self.history_dir = history_dir
        self.enable_history = enable_history

        # История всех сделок (с момента запуска бота)
        self.trades: List[TradeStats] = []

        # Отслеживание капитала для drawdown
        self.capital_history: List[float] = []

        # Создаем директорию для истории
        if self.enable_history:
            os.makedirs(self.history_dir, exist_ok=True)

    def add_trade(self, trade: TradeStats) -> None:
        """Добавляет завершенную сделку"""
        self.trades.append(trade)

        # Сохраняем в JSON (для AI анализа)
        if self.enable_history:
            self._save_trade_to_json(trade)

    def update_capital(self, capital: float) -> None:
        """Обновляет текущий капитал для отслеживания drawdown"""
        self.capital_history.append(capital)

    def get_metrics(self, window_name: str = 'short') -> Optional[PerformanceMetrics]:
        """
        Рассчитывает метрики для указанного окна

        Args:
            window_name: Название окна ('short', 'medium', 'long', 'all')

        Returns:
            PerformanceMetrics или None если недостаточно данных
        """
        window_size = self.windows.get(window_name)

        # Выбираем сделки для анализа
        if window_size is None:
            # Все сделки
            trades = self.trades
        else:
            # Последние N сделок
            trades = self.trades[-window_size:] if len(self.trades) >= window_size else []

        if not trades:
            return None

        # Базовые метрики
        total_trades = len(trades)
        wins = [t for t in trades if t.is_win()]
        losses = [t for t in trades if not t.is_win()]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        # Средние прибыли и убытки
        avg_win = sum(t.pnl_pct for t in wins) / win_count if win_count > 0 else 0.0
        avg_loss = abs(sum(t.pnl_pct for t in losses) / loss_count) if loss_count > 0 else 0.0

        # Необходимый винрейт (на основе среднего R:R)
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        required_win_rate = 1 / (1 + avg_win_loss_ratio) if avg_win_loss_ratio > 0 else 1.0

        # Геометрическая прогрессия
        geometric_mean_return = self._calculate_geometric_mean([t.pnl_pct for t in trades])

        # Compound growth rate (если есть история капитала)
        if len(trades) >= 2:
            # Рассчитываем на основе первой и последней сделки
            initial_capital = 1000.0  # Нормализованный капитал
            final_capital = initial_capital * math.prod([1 + t.pnl_pct / 100 for t in trades])
            compound_growth_rate = (final_capital / initial_capital) ** (1 / total_trades) - 1
            compound_growth_rate *= 100  # В проценты
        else:
            compound_growth_rate = geometric_mean_return

        # 90% доверительный интервал для винрейта
        ci_lower, ci_upper = self._calculate_confidence_interval(win_rate, total_trades, 0.90)

        # Profit Factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy (математическое ожидание)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Max Drawdown
        max_drawdown_pct, current_drawdown_pct = self._calculate_drawdown(trades)

        # Consecutive wins/losses
        max_cons_wins, max_cons_losses, current_streak = self._calculate_streaks(trades)

        # Капитал
        starting_capital = 1000.0  # Нормализованный
        ending_capital = starting_capital * math.prod([1 + t.pnl_pct / 100 for t in trades])
        total_return_pct = ((ending_capital - starting_capital) / starting_capital) * 100

        # Проверка условий
        conditions = {
            'win_rate_above_required': win_rate > required_win_rate,
            'positive_geometric_growth': geometric_mean_return > 0,
            'ci_above_zero': ci_lower > 0
        }
        all_conditions_met = all(conditions.values())

        # Формируем метрики
        metrics = PerformanceMetrics(
            total_trades=total_trades,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            required_win_rate=required_win_rate,
            geometric_mean_return=geometric_mean_return,
            compound_growth_rate=compound_growth_rate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_loss_ratio=avg_win_loss_ratio,
            max_drawdown_pct=max_drawdown_pct,
            current_drawdown_pct=current_drawdown_pct,
            max_consecutive_wins=max_cons_wins,
            max_consecutive_losses=max_cons_losses,
            current_streak=current_streak,
            starting_capital=starting_capital,
            ending_capital=ending_capital,
            total_return_pct=total_return_pct,
            all_conditions_met=all_conditions_met,
            conditions=conditions,
            period_start=trades[0].exit_time,
            period_end=trades[-1].exit_time,
            last_updated=datetime.now()
        )

        # Сохраняем snapshot
        if self.enable_history:
            self._save_metrics_snapshot(window_name, metrics)

        return metrics

    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Возвращает метрики для всех окон"""
        result = {}
        for window_name in self.windows.keys():
            metrics = self.get_metrics(window_name)
            if metrics:
                result[window_name] = metrics
        return result

    def _calculate_geometric_mean(self, returns_pct: List[float]) -> float:
        """
        Рассчитывает среднее геометрическое доходности

        Args:
            returns_pct: Список доходностей в процентах

        Returns:
            Среднее геометрическое в процентах
        """
        if not returns_pct:
            return 0.0

        # Конвертируем проценты в множители (1 + r)
        multipliers = [1 + r / 100 for r in returns_pct]

        # Геометрическое среднее: (∏(1+r_i))^(1/N) - 1
        product = math.prod(multipliers)
        geometric_mean = product ** (1 / len(multipliers)) - 1

        return geometric_mean * 100  # Обратно в проценты

    def _calculate_confidence_interval(
        self,
        win_rate: float,
        n: int,
        confidence: float = 0.90
    ) -> tuple:
        """
        Рассчитывает доверительный интервал для винрейта (метод Вильсона)

        Args:
            win_rate: Текущий винрейт (0-1)
            n: Количество сделок
            confidence: Уровень доверия (0.90 = 90%)

        Returns:
            (lower_bound, upper_bound)
        """
        if n == 0:
            return (0.0, 0.0)

        # Z-score для заданного уровня доверия
        # 90% -> 1.645, 95% -> 1.96, 99% -> 2.576
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.645)

        # Метод Вильсона (более точный для малых выборок)
        p = win_rate
        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denominator
        margin = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denominator

        lower = max(0, centre - margin)
        upper = min(1, centre + margin)

        return (lower, upper)

    def _calculate_drawdown(self, trades: List[TradeStats]) -> tuple:
        """
        Рассчитывает максимальную и текущую просадку

        Returns:
            (max_drawdown_pct, current_drawdown_pct)
        """
        if not trades:
            return (0.0, 0.0)

        # Строим кривую капитала
        capital = 1000.0  # Начальный капитал (нормализованный)
        capital_curve = [capital]

        for trade in trades:
            capital *= (1 + trade.pnl_pct / 100)
            capital_curve.append(capital)

        # Рассчитываем drawdown на каждом шаге
        max_drawdown = 0.0
        peak = capital_curve[0]

        for value in capital_curve:
            if value > peak:
                peak = value
            drawdown = ((peak - value) / peak) * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Текущая просадка
        current_peak = max(capital_curve)
        current_drawdown = ((current_peak - capital_curve[-1]) / current_peak) * 100

        return (max_drawdown, current_drawdown)

    def _calculate_streaks(self, trades: List[TradeStats]) -> tuple:
        """
        Рассчитывает максимальные серии побед/поражений

        Returns:
            (max_consecutive_wins, max_consecutive_losses, current_streak)
        """
        if not trades:
            return (0, 0, 0)

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.is_win():
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        # Текущая серия (положительная для wins, отрицательная для losses)
        current_streak = current_wins if current_wins > 0 else -current_losses

        return (max_wins, max_losses, current_streak)

    def _save_trade_to_json(self, trade: TradeStats) -> None:
        """Сохраняет сделку в JSON файл (для AI анализа)"""
        filepath = os.path.join(self.history_dir, 'trades_history.json')

        trade_data = {
            'symbol': trade.symbol,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'exit_reason': trade.exit_reason,
            'entry_time': trade.entry_time.isoformat(),
            'exit_time': trade.exit_time.isoformat(),
            'duration_hours': trade.duration_hours
        }

        # Читаем существующие данные
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {'trades': []}

        # Добавляем новую сделку
        data['trades'].append(trade_data)

        # Сохраняем
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_metrics_snapshot(self, window_name: str, metrics: PerformanceMetrics) -> None:
        """Сохраняет snapshot метрик в CSV"""
        filepath = os.path.join(self.history_dir, f'performance_snapshots_{window_name}.csv')

        # Проверяем существует ли файл
        file_exists = os.path.isfile(filepath)

        with open(filepath, 'a', newline='') as f:
            fieldnames = [
                'timestamp', 'total_trades', 'win_rate', 'required_win_rate',
                'geometric_mean_return', 'compound_growth_rate',
                'ci_lower', 'ci_upper', 'profit_factor', 'expectancy',
                'avg_win', 'avg_loss', 'avg_win_loss_ratio',
                'max_drawdown_pct', 'current_drawdown_pct',
                'max_consecutive_wins', 'max_consecutive_losses',
                'total_return_pct', 'all_conditions_met'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Записываем заголовок если файл новый
            if not file_exists:
                writer.writeheader()

            # Записываем данные
            writer.writerow({
                'timestamp': metrics.last_updated.isoformat(),
                'total_trades': metrics.total_trades,
                'win_rate': round(metrics.win_rate, 4),
                'required_win_rate': round(metrics.required_win_rate, 4),
                'geometric_mean_return': round(metrics.geometric_mean_return, 4),
                'compound_growth_rate': round(metrics.compound_growth_rate, 4),
                'ci_lower': round(metrics.ci_lower, 4),
                'ci_upper': round(metrics.ci_upper, 4),
                'profit_factor': round(metrics.profit_factor, 4) if metrics.profit_factor != float('inf') else 999.99,
                'expectancy': round(metrics.expectancy, 4),
                'avg_win': round(metrics.avg_win, 4),
                'avg_loss': round(metrics.avg_loss, 4),
                'avg_win_loss_ratio': round(metrics.avg_win_loss_ratio, 4),
                'max_drawdown_pct': round(metrics.max_drawdown_pct, 4),
                'current_drawdown_pct': round(metrics.current_drawdown_pct, 4),
                'max_consecutive_wins': metrics.max_consecutive_wins,
                'max_consecutive_losses': metrics.max_consecutive_losses,
                'total_return_pct': round(metrics.total_return_pct, 4),
                'all_conditions_met': metrics.all_conditions_met
            })
