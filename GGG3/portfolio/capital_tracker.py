"""
Capital Tracker - отслеживание изменений капитала за периоды времени

Сохраняет snapshots капитала и позволяет рассчитывать изменения за:
- 24 часа
- 7 дней
- 30 дней (месяц)
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CapitalTracker:
    """Отслеживает изменения капитала за периоды времени"""

    def __init__(self, state_file: str = 'data/capital_history.json'):
        """
        Инициализация трекера

        Args:
            state_file: Путь к файлу для сохранения истории
        """
        self.state_file = Path(state_file)
        self.history: List[Dict] = []  # История snapshots

        # Создаем директорию если не существует
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Загружаем историю
        self.load_history()

        logger.info(f"CapitalTracker initialized with {len(self.history)} historical snapshots")

    def load_history(self):
        """Загружает историю из файла"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                logger.info(f"Loaded {len(self.history)} capital snapshots")
            except Exception as e:
                logger.error(f"Failed to load capital history: {e}")
                self.history = []
        else:
            logger.info("No existing capital history found, starting fresh")
            self.history = []

    def save_history(self):
        """Сохраняет историю в файл"""
        try:
            data = {
                'history': self.history,
                'last_updated': datetime.utcnow().isoformat() + 'Z'  # UTC время
            }

            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Capital history saved ({len(self.history)} snapshots)")
        except Exception as e:
            logger.error(f"Failed to save capital history: {e}")

    def record_snapshot(
        self,
        free_balance: float,
        unrealized_pnl: float,
        reserve_fund: float,
        num_open_positions: int = 0
    ):
        """
        Записывает snapshot капитала

        ФЬЮЧЕРСЫ: Для фьючерсной торговли баланс НЕ меняется при открытии позиции.
        Общий капитал = баланс + нереализованный PnL + резервный фонд

        Args:
            free_balance: Текущий баланс (не меняется при открытии фьючерсов)
            unrealized_pnl: Нереализованный PnL от всех открытых позиций
            reserve_fund: Резервный фонд
            num_open_positions: Количество открытых позиций
        """
        total_equity = free_balance + unrealized_pnl + reserve_fund

        snapshot = {
            'timestamp': time.time(),
            'datetime': datetime.utcnow().isoformat() + 'Z',  # UTC время с маркером Z
            'free_balance': free_balance,
            'unrealized_pnl': unrealized_pnl,
            'reserve_fund': reserve_fund,
            'total_equity': total_equity,
            'num_open_positions': num_open_positions
        }

        self.history.append(snapshot)

        # Очищаем старые snapshots (храним только за последние 31 день)
        self._cleanup_old_snapshots(days=31)

        # Сохраняем
        self.save_history()

        logger.debug(f"Capital snapshot recorded: ${total_equity:,.2f}")

    def _cleanup_old_snapshots(self, days: int = 31):
        """
        Удаляет snapshots старше указанного количества дней

        Args:
            days: Количество дней для хранения
        """
        cutoff_time = time.time() - (days * 24 * 3600)

        original_count = len(self.history)
        self.history = [
            snap for snap in self.history
            if snap['timestamp'] > cutoff_time
        ]

        removed = original_count - len(self.history)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old snapshots (older than {days} days)")

    def _is_snapshot_suspicious(self, snapshot: Dict, max_jump_pct: float = 50.0) -> bool:
        """
        Проверяет snapshot на подозрительные скачки капитала

        ЗАЩИТА ОТ АРТЕФАКТОВ: Если капитал изменился более чем на X% за короткий период
        без очевидных причин (перезапуск, смена backend, пополнение) -
        помечаем snapshot как подозрительный.

        Args:
            snapshot: Snapshot для проверки
            max_jump_pct: Максимальное допустимое изменение за 1 час (%)

        Returns:
            True если snapshot подозрительный
        """
        # Находим предыдущий snapshot
        idx = self.history.index(snapshot)
        if idx == 0:
            return False  # Первый snapshot не проверяем

        prev_snapshot = self.history[idx - 1]

        # Вычисляем изменение
        time_diff_hours = (snapshot['timestamp'] - prev_snapshot['timestamp']) / 3600
        if time_diff_hours < 0.1:  # Меньше 6 минут - скипаем
            return False

        prev_equity = prev_snapshot['total_equity']
        curr_equity = snapshot['total_equity']

        if prev_equity <= 0:
            return False

        # Процентное изменение
        pct_change = abs((curr_equity / prev_equity - 1) * 100)

        # Нормализуем на время (скачок за 1 час)
        pct_per_hour = pct_change / max(time_diff_hours, 0.1)

        # Если изменение больше max_jump_pct% в час - подозрительно
        if pct_per_hour > max_jump_pct:
            logger.warning(
                f"Suspicious snapshot detected: {pct_change:.1f}% change "
                f"in {time_diff_hours:.1f}h ({pct_per_hour:.1f}%/h) "
                f"from ${prev_equity:.2f} to ${curr_equity:.2f}"
            )
            return True

        return False

    def get_snapshot_at_time(self, hours_ago: float) -> Optional[Dict]:
        """
        Получает ближайший snapshot за указанное время назад

        Args:
            hours_ago: Сколько часов назад

        Returns:
            Snapshot или None если нет данных
        """
        if not self.history:
            return None

        target_time = time.time() - (hours_ago * 3600)

        # Находим ближайший snapshot
        closest_snapshot = None
        min_diff = float('inf')

        for snapshot in self.history:
            diff = abs(snapshot['timestamp'] - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_snapshot = snapshot

        # КРИТИЧНО: Проверяем что snapshot НЕ слишком далеко от нужного времени
        # Если разница больше 25% от запрошенного периода - возвращаем None
        max_allowed_diff = hours_ago * 3600 * 0.25  # 25% от периода

        if closest_snapshot and min_diff <= max_allowed_diff:
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Санity-check на подозрительные скачки
            if self._is_snapshot_suspicious(closest_snapshot):
                logger.debug(
                    f"Snapshot for {hours_ago}h ago rejected (suspicious capital jump)"
                )
                return None

            return closest_snapshot
        else:
            # Snapshot слишком далеко - считаем что данных нет
            logger.debug(f"Snapshot for {hours_ago}h ago not found (closest is {min_diff/3600:.1f}h away)")
            return None

    def get_change_over_period(
        self,
        current_equity: float,
        hours_ago: float
    ) -> Dict:
        """
        Рассчитывает изменение капитала за период

        Args:
            current_equity: Текущий капитал
            hours_ago: Период в часах (24, 168, 720)

        Returns:
            Словарь с изменениями:
            - absolute: Абсолютное изменение ($)
            - percent: Процентное изменение (%)
            - available: Доступны ли данные
        """
        snapshot = self.get_snapshot_at_time(hours_ago)

        if snapshot is None:
            return {
                'absolute': 0.0,
                'percent': 0.0,
                'available': False,
                'reason': 'No historical data'
            }

        old_equity = snapshot['total_equity']
        absolute_change = current_equity - old_equity
        percent_change = ((current_equity / old_equity) - 1) * 100 if old_equity > 0 else 0

        return {
            'absolute': absolute_change,
            'percent': percent_change,
            'available': True,
            'old_equity': old_equity,
            'new_equity': current_equity,
            'snapshot_time': snapshot['datetime']
        }

    def get_all_period_changes(self, current_equity: float) -> Dict:
        """
        Получает изменения за все периоды

        Args:
            current_equity: Текущий капитал

        Returns:
            Словарь с изменениями за 24h, 7d, 30d
        """
        return {
            '24h': self.get_change_over_period(current_equity, hours_ago=24),
            '7d': self.get_change_over_period(current_equity, hours_ago=7*24),
            '30d': self.get_change_over_period(current_equity, hours_ago=30*24)
        }

    def get_latest_snapshot(self) -> Optional[Dict]:
        """Получает последний snapshot"""
        return self.history[-1] if self.history else None

    def get_statistics(self) -> Dict:
        """
        Получает статистику по капиталу

        Returns:
            Словарь со статистикой
        """
        if not self.history:
            return {
                'snapshots_count': 0,
                'first_snapshot': None,
                'last_snapshot': None,
                'max_equity': 0.0,
                'min_equity': 0.0
            }

        equities = [snap['total_equity'] for snap in self.history]

        return {
            'snapshots_count': len(self.history),
            'first_snapshot': self.history[0]['datetime'],
            'last_snapshot': self.history[-1]['datetime'],
            'max_equity': max(equities),
            'min_equity': min(equities),
            'current_equity': equities[-1]
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    tracker = CapitalTracker()

    # Record snapshot
    tracker.record_snapshot(
        free_balance=500.0,
        unrealized_pnl=50.0,  # Нереализованный PnL от открытых позиций
        reserve_fund=50.0,
        num_open_positions=5
    )

    # Get current equity
    latest = tracker.get_latest_snapshot()
    if latest:
        current_equity = latest['total_equity']
        print(f"Current equity: ${current_equity:,.2f}")

        # Get changes over periods
        changes = tracker.get_all_period_changes(current_equity)

        print("\n📊 Period Changes:")
        for period, data in changes.items():
            if data['available']:
                print(f"  {period}: {data['absolute']:+,.2f} ({data['percent']:+.2f}%)")
            else:
                print(f"  {period}: Not available ({data['reason']})")

    # Get statistics
    stats = tracker.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"  Total snapshots: {stats['snapshots_count']}")
    print(f"  Max equity: ${stats['max_equity']:,.2f}")
    print(f"  Min equity: ${stats['min_equity']:,.2f}")
