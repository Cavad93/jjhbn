"""
Position Manager

Управление портфелем из 10 одновременных позиций с полным snapshot для temporal consistency.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Торговая позиция с полным snapshot на момент входа

    КРИТИЧНО: entry_snapshot сохраняет все данные на момент открытия позиции
    для обеспечения temporal consistency при обучении моделей.
    """

    # Основная информация
    symbol: str                         # Торговая пара (например, 'BTCUSDT')
    direction: str                      # 'LONG' или 'SHORT'
    entry_time: datetime                # Время открытия позиции
    entry_price: float                  # Цена входа
    amount: float                       # Количество актива
    position_value: float               # Стоимость позиции (в USDT)

    # Take Profit и Stop Loss
    tp_price: float                     # Цена Take Profit
    sl_price: float                     # Цена Stop Loss
    atr_value: float                    # Значение ATR на момент входа

    # ID ордеров на бирже
    entry_order_id: Optional[str] = None    # ID entry ордера
    tp_order_id: Optional[str] = None       # ID TP ордера
    sl_order_id: Optional[str] = None       # ID SL ордера
    exchange_position_id: Optional[str] = None  # ID позиции в PaperExchange (для фьючерсов)

    # ✅ КРИТИЧНО: Snapshot на момент входа (temporal consistency)
    entry_snapshot: Dict[str, Any] = field(default_factory=dict)
    # Структура entry_snapshot:
    # {
    #   'features_68d': [float, ...],           # 68 признаков на момент входа
    #   'predictions': {                         # Предсказания всех моделей
    #       'p_xgb': float,
    #       'p_rf': float,
    #       'p_arf': float,
    #       'p_nn': float,
    #       'p_base': float
    #   },
    #   'meta_context': {                        # 7 признаков для META
    #       'feature1': float,
    #       ...
    #   },
    #   'p_meta': float,                         # Финальное предсказание META
    #   'ev_long': float,                        # Expected Value для LONG
    #   'ev_short': float,                       # Expected Value для SHORT
    #   'selected_ev': float,                    # Выбранный EV
    #   'timestamp': float                       # Unix timestamp
    # }

    # Информация о закрытии
    exit_time: Optional[datetime] = None    # Время закрытия
    exit_price: Optional[float] = None      # Цена выхода
    exit_reason: Optional[str] = None       # 'TP', 'SL', 'TIMEOUT', 'MANUAL', 'REPLACED'

    # PnL
    pnl: float = 0.0                        # Прибыль/убыток в USDT
    pnl_pct: float = 0.0                    # Прибыль/убыток в %

    # Статус
    status: str = 'OPEN'                    # 'OPEN' или 'CLOSED'

    # Дополнительные метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Валидация после инициализации"""
        # Проверка direction
        if self.direction not in ['LONG', 'SHORT']:
            raise ValueError(f"direction должен быть 'LONG' или 'SHORT', получено: {self.direction}")

        # Проверка status
        if self.status not in ['OPEN', 'CLOSED']:
            raise ValueError(f"status должен быть 'OPEN' или 'CLOSED', получено: {self.status}")

        # Преобразование datetime из строки, если нужно
        if isinstance(self.entry_time, str):
            self.entry_time = datetime.fromisoformat(self.entry_time)

        if isinstance(self.exit_time, str) and self.exit_time:
            self.exit_time = datetime.fromisoformat(self.exit_time)

    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализация позиции в dict для JSON

        Returns:
            Dict с данными позиции
        """
        data = asdict(self)

        # Преобразуем datetime в ISO строки
        if self.entry_time:
            data['entry_time'] = self.entry_time.isoformat()

        if self.exit_time:
            data['exit_time'] = self.exit_time.isoformat()

        # Преобразуем numpy arrays в списки (если есть)
        if 'features_68d' in self.entry_snapshot:
            features = self.entry_snapshot['features_68d']
            if isinstance(features, np.ndarray):
                data['entry_snapshot']['features_68d'] = features.tolist()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """
        Десериализация позиции из dict

        Args:
            data: Dict с данными позиции

        Returns:
            Position объект
        """
        # Создаем копию чтобы не модифицировать оригинал
        data = data.copy()

        # Преобразуем ISO строки в datetime
        if 'entry_time' in data and isinstance(data['entry_time'], str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])

        if 'exit_time' in data and data['exit_time'] and isinstance(data['exit_time'], str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])

        # Преобразуем списки обратно в numpy arrays (если нужно)
        if 'entry_snapshot' in data and 'features_68d' in data['entry_snapshot']:
            features = data['entry_snapshot']['features_68d']
            if isinstance(features, list):
                data['entry_snapshot']['features_68d'] = np.array(features)

        return cls(**data)

    def calculate_pnl(self, current_price: float) -> tuple[float, float]:
        """
        Рассчитывает текущий нереализованный PnL

        Args:
            current_price: Текущая цена

        Returns:
            (pnl_usdt, pnl_percent)
        """
        if self.direction == 'LONG':
            pnl_usdt = (current_price - self.entry_price) * self.amount
            pnl_pct = (current_price / self.entry_price - 1) * 100
        else:  # SHORT
            pnl_usdt = (self.entry_price - current_price) * self.amount
            pnl_pct = (1 - current_price / self.entry_price) * 100

        return pnl_usdt, pnl_pct

    def is_tp_hit(self, current_price: float) -> bool:
        """Проверяет достиг ли цена TP"""
        if self.direction == 'LONG':
            return current_price >= self.tp_price
        else:  # SHORT
            return current_price <= self.tp_price

    def is_sl_hit(self, current_price: float) -> bool:
        """Проверяет достиг ли цена SL"""
        if self.direction == 'LONG':
            return current_price <= self.sl_price
        else:  # SHORT
            return current_price >= self.sl_price

    def get_duration(self) -> float:
        """
        Возвращает длительность позиции в секундах

        Returns:
            Длительность в секундах
        """
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds()
        else:
            return (datetime.now() - self.entry_time).total_seconds()

    def __repr__(self) -> str:
        """Строковое представление позиции"""
        pnl_str = f"{self.pnl:+.2f} USDT ({self.pnl_pct:+.2f}%)" if self.status == 'CLOSED' else "N/A"
        return (
            f"Position({self.symbol}, {self.direction}, "
            f"entry={self.entry_price:.2f}, "
            f"TP={self.tp_price:.2f}, SL={self.sl_price:.2f}, "
            f"status={self.status}, PnL={pnl_str})"
        )


class PositionManager:
    """
    Менеджер для управления портфелем из максимум 10 одновременных позиций

    Основные функции:
    - Добавление/закрытие позиций
    - Проверка возможности открытия новой позиции
    - Расчет статистики и PnL
    - Сохранение/загрузка состояния в JSON
    """

    def __init__(
        self,
        max_positions: int = 10,
        data_dir: str = None
    ):
        """
        Инициализация менеджера позиций

        Args:
            max_positions: Максимум одновременных позиций
            data_dir: Директория для сохранения данных (по умолчанию: относительный путь)
        """
        self.max_positions = max_positions

        # Используем относительный путь если не указан явно
        if data_dir is None:
            # Путь относительно текущей директории проекта
            from pathlib import Path
            project_root = Path(__file__).parent.parent  # GGG3/
            self.data_dir = str(project_root / 'data' / 'positions')
        else:
            self.data_dir = data_dir

        # Открытые позиции: {symbol: Position}
        self.positions: Dict[str, Position] = {}

        # Закрытые позиции (история)
        self.closed_positions: List[Position] = []

        # Создаем директорию если не существует
        os.makedirs(self.data_dir, exist_ok=True)

        logger.info(f"PositionManager initialized: max_positions={max_positions}")

    def can_open_position(self, symbol: str) -> tuple[bool, str]:
        """
        Проверяет можно ли открыть позицию

        Правила:
        1. Максимум max_positions одновременных позиций
        2. Не более 1 позиции на монету

        Args:
            symbol: Торговая пара

        Returns:
            (can_open: bool, reason: str)
        """
        # Правило 1: Проверка максимума позиций
        if len(self.positions) >= self.max_positions:
            return False, f"Достигнут максимум позиций ({self.max_positions})"

        # Правило 2: Проверка дублирования символа
        if symbol in self.positions:
            return False, f"Позиция по {symbol} уже открыта"

        return True, "OK"

    def add_position(self, position: Position) -> None:
        """
        Добавляет открытую позицию в портфель

        Args:
            position: Объект Position

        Raises:
            ValueError: Если позицию нельзя открыть
        """
        # Проверяем возможность открытия
        can_open, reason = self.can_open_position(position.symbol)
        if not can_open:
            raise ValueError(f"Невозможно открыть позицию: {reason}")

        # Валидация позиции
        if position.status != 'OPEN':
            raise ValueError(f"Можно добавить только OPEN позиции, получено: {position.status}")

        # Добавляем в словарь
        self.positions[position.symbol] = position

        logger.info(
            f"Позиция добавлена: {position.symbol} {position.direction} "
            f"@ {position.entry_price:.2f}, "
            f"TP={position.tp_price:.2f}, SL={position.sl_price:.2f}"
        )

        # Сохраняем состояние
        self.save_to_file()

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str
    ) -> Position:
        """
        Закрывает позицию и рассчитывает PnL

        Args:
            symbol: Торговая пара
            exit_price: Цена выхода
            exit_reason: Причина закрытия ('TP', 'SL', 'TIMEOUT', 'MANUAL', 'REPLACED')

        Returns:
            Закрытая позиция

        Raises:
            ValueError: Если позиция не найдена
        """
        if symbol not in self.positions:
            raise ValueError(f"Позиция {symbol} не найдена")

        # Получаем позицию
        position = self.positions[symbol]

        # Обновляем информацию о закрытии
        position.exit_time = datetime.now()
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.status = 'CLOSED'

        # Рассчитываем PnL
        if position.direction == 'LONG':
            position.pnl = (exit_price - position.entry_price) * position.amount
            position.pnl_pct = (exit_price / position.entry_price - 1) * 100
        else:  # SHORT
            position.pnl = (position.entry_price - exit_price) * position.amount
            position.pnl_pct = (1 - exit_price / position.entry_price) * 100

        logger.info(
            f"Позиция закрыта: {symbol} {position.direction} "
            f"entry={position.entry_price:.2f}, exit={exit_price:.2f}, "
            f"PnL={position.pnl:+.2f} USDT ({position.pnl_pct:+.2f}%), "
            f"reason={exit_reason}"
        )

        # Перемещаем в закрытые
        self.closed_positions.append(position)
        del self.positions[symbol]

        # Сохраняем состояние
        self.save_to_file()

        return position

    def get_all_open(self) -> List[Position]:
        """
        Возвращает все открытые позиции

        Returns:
            Список открытых позиций
        """
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Получает позицию по символу

        Args:
            symbol: Торговая пара

        Returns:
            Position или None если не найдена
        """
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """
        Проверяет есть ли открытая позиция по символу

        Args:
            symbol: Торговая пара

        Returns:
            True если позиция открыта
        """
        return symbol in self.positions

    def get_total_exposure(self) -> float:
        """
        Рассчитывает общую стоимость всех открытых позиций

        Returns:
            Общая стоимость в USDT
        """
        return sum(p.position_value for p in self.positions.values())

    def get_open_count(self) -> int:
        """Количество открытых позиций"""
        return len(self.positions)

    def get_available_slots(self) -> int:
        """Количество доступных слотов для новых позиций"""
        return self.max_positions - len(self.positions)

    def get_pnl_summary(self) -> Dict[str, Any]:
        """
        Рассчитывает статистику по PnL

        Returns:
            Dict с метриками
        """
        if not self.closed_positions:
            return {
                'total_pnl': 0.0,
                'total_pnl_pct': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0
            }

        # Основные метрики
        total_pnl = sum(p.pnl for p in self.closed_positions)
        total_trades = len(self.closed_positions)

        # Разделение на wins и losses
        wins = [p for p in self.closed_positions if p.pnl > 0]
        losses = [p for p in self.closed_positions if p.pnl <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        # Средние значения
        avg_win = sum(p.pnl for p in wins) / win_count if win_count > 0 else 0
        avg_loss = sum(p.pnl for p in losses) / loss_count if loss_count > 0 else 0

        # Profit factor
        total_wins = sum(p.pnl for p in wins)
        total_losses = abs(sum(p.pnl for p in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Max win/loss
        max_win = max((p.pnl for p in wins), default=0)
        max_loss = min((p.pnl for p in losses), default=0)

        # Средний PnL в процентах
        avg_pnl_pct = sum(p.pnl_pct for p in self.closed_positions) / total_trades

        return {
            'total_pnl': total_pnl,
            'total_pnl_pct': avg_pnl_pct,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'wins': win_count,
            'losses': loss_count,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max_win,
            'max_loss': max_loss
        }

    def get_positions_by_direction(self) -> Dict[str, int]:
        """
        Подсчитывает количество позиций по направлениям

        Returns:
            {'LONG': count, 'SHORT': count}
        """
        long_count = sum(1 for p in self.positions.values() if p.direction == 'LONG')
        short_count = sum(1 for p in self.positions.values() if p.direction == 'SHORT')

        return {
            'LONG': long_count,
            'SHORT': short_count
        }

    def get_recent_closed(self, limit: int = 10) -> List[Position]:
        """
        Возвращает последние закрытые позиции

        Args:
            limit: Максимальное количество позиций

        Returns:
            Список позиций
        """
        return self.closed_positions[-limit:]

    def save_to_file(self, filename: str = 'positions.json') -> None:
        """
        Сохраняет позиции в JSON файл

        Args:
            filename: Имя файла
        """
        filepath = os.path.join(self.data_dir, filename)

        data = {
            'open_positions': [p.to_dict() for p in self.positions.values()],
            'closed_positions': [p.to_dict() for p in self.closed_positions[-100:]],  # Последние 100
            'metadata': {
                'max_positions': self.max_positions,
                'total_open': len(self.positions),
                'total_closed': len(self.closed_positions),
                'last_updated': datetime.now().isoformat()
            }
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Позиции сохранены в {filepath}")
        except Exception as e:
            logger.error(f"Ошибка сохранения позиций: {e}")

    def load_from_file(self, filename: str = 'positions.json') -> bool:
        """
        Загружает позиции из JSON файла

        Args:
            filename: Имя файла

        Returns:
            True если загрузка успешна
        """
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            logger.warning(f"Файл {filepath} не найден")
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Восстанавливаем открытые позиции
            self.positions = {}
            for p_dict in data.get('open_positions', []):
                position = Position.from_dict(p_dict)
                self.positions[position.symbol] = position

            # Восстанавливаем закрытые позиции
            self.closed_positions = []
            for p_dict in data.get('closed_positions', []):
                position = Position.from_dict(p_dict)
                self.closed_positions.append(position)

            logger.info(
                f"Позиции загружены: {len(self.positions)} открытых, "
                f"{len(self.closed_positions)} закрытых"
            )
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки позиций: {e}")
            return False

    def clear_all(self) -> None:
        """Очищает все позиции (для тестирования)"""
        self.positions = {}
        self.closed_positions = []
        logger.warning("Все позиции очищены")

    def __repr__(self) -> str:
        """Строковое представление менеджера"""
        return (
            f"PositionManager("
            f"open={len(self.positions)}/{self.max_positions}, "
            f"closed={len(self.closed_positions)})"
        )
