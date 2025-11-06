#!/usr/bin/env python3
"""
Асинхронный менеджер PostgreSQL для OHLCV данных

Использует asyncpg для высокопроизводительной работы с PostgreSQL.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import os

import pandas as pd
import asyncpg
from asyncpg import Pool, Connection

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Асинхронный менеджер базы данных для OHLCV данных

    Особенности:
    - Connection pooling для высокой производительности
    - Batch inserts для быстрой загрузки
    - Автоматическое создание индексов
    - UPSERT для обновления данных
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'binance_bot',
        user: str = 'bot_user',
        password: str = 'bot_password',
        min_pool_size: int = 10,
        max_pool_size: int = 20
    ):
        """
        Инициализация менеджера БД

        Args:
            host: Хост PostgreSQL
            port: Порт PostgreSQL
            database: Имя базы данных
            user: Пользователь
            password: Пароль
            min_pool_size: Минимальный размер connection pool
            max_pool_size: Максимальный размер connection pool
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size

        self.pool: Optional[Pool] = None

    async def connect(self):
        """Создание connection pool"""
        if self.pool is not None:
            logger.warning("Connection pool уже создан")
            return

        logger.info(f"Подключение к PostgreSQL: {self.host}:{self.port}/{self.database}")

        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                command_timeout=60
            )

            logger.info("✅ Connection pool создан")

        except Exception as e:
            logger.error(f"❌ Ошибка подключения к БД: {e}")
            raise

    async def disconnect(self):
        """Закрытие connection pool"""
        if self.pool is None:
            return

        logger.info("Закрытие connection pool...")
        await self.pool.close()
        self.pool = None
        logger.info("✅ Connection pool закрыт")

    async def execute_script(self, script_path: str):
        """
        Выполнение SQL скрипта (для создания схемы)

        Args:
            script_path: Путь к SQL файлу
        """
        logger.info(f"Выполнение SQL скрипта: {script_path}")

        with open(script_path, 'r', encoding='utf-8') as f:
            sql = f.read()

        async with self.pool.acquire() as conn:
            await conn.execute(sql)

        logger.info("✅ SQL скрипт выполнен")

    # ========================================================================
    # OHLCV ДАННЫЕ
    # ========================================================================

    async def insert_ohlcv_batch(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame
    ) -> int:
        """
        Вставка OHLCV данных пакетом (UPSERT)

        Args:
            symbol: Символ (BTCUSDT)
            timeframe: Таймфрейм (5m, 15m, etc.)
            df: DataFrame с колонками: timestamp, open, high, low, close, volume

        Returns:
            int: Количество вставленных строк
        """
        if len(df) == 0:
            return 0

        # Подготовка данных
        records = [
            (
                symbol,
                timeframe,
                row['timestamp'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                row.get('quote_volume'),
                row.get('trades_count')
            )
            for _, row in df.iterrows()
        ]

        # UPSERT запрос
        query = """
            INSERT INTO ohlcv_data (
                symbol, timeframe, timestamp,
                open, high, low, close, volume,
                quote_volume, trades_count
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (symbol, timeframe, timestamp)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                quote_volume = EXCLUDED.quote_volume,
                trades_count = EXCLUDED.trades_count,
                updated_at = NOW()
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(query, records)

        logger.debug(f"✓ {symbol} {timeframe}: вставлено {len(records)} свечей")

        return len(records)

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Получение OHLCV данных

        Args:
            symbol: Символ
            timeframe: Таймфрейм
            limit: Количество свечей

        Returns:
            DataFrame с OHLCV данными
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, timeframe, limit)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Сортируем по возрастанию времени
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    async def get_last_timestamp(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[datetime]:
        """
        Получение последней временной метки для символа и таймфрейма

        Returns:
            datetime или None если данных нет
        """
        query = """
            SELECT MAX(timestamp) as last_ts
            FROM ohlcv_data
            WHERE symbol = $1 AND timeframe = $2
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, symbol, timeframe)

        return row['last_ts'] if row else None

    async def delete_ohlcv(
        self,
        symbol: str,
        timeframe: Optional[str] = None
    ) -> int:
        """
        Удаление OHLCV данных для символа

        Args:
            symbol: Символ
            timeframe: Таймфрейм (если None - удаляет все таймфреймы)

        Returns:
            int: Количество удалённых строк
        """
        if timeframe:
            query = "DELETE FROM ohlcv_data WHERE symbol = $1 AND timeframe = $2"
            args = [symbol, timeframe]
        else:
            query = "DELETE FROM ohlcv_data WHERE symbol = $1"
            args = [symbol]

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *args)

        # Парсим результат "DELETE N"
        rows_deleted = int(result.split()[-1])

        logger.info(f"✓ Удалено {rows_deleted} строк для {symbol} {timeframe or 'all'}")

        return rows_deleted

    # ========================================================================
    # АКТИВНЫЕ СИМВОЛЫ (ТОП-10)
    # ========================================================================

    async def update_active_symbols(
        self,
        top_symbols: List[Dict[str, Any]]
    ):
        """
        Обновление списка активных символов (топ-10 по EV)

        Args:
            top_symbols: Список словарей с ключами:
                - symbol: str
                - ev_score: float
                - rank: int (1-10)
                - volume_24h: float
        """
        # Деактивируем все текущие символы
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE active_symbols SET is_active = FALSE")

            # Обновляем или вставляем новые символы
            for sym in top_symbols:
                await conn.execute("""
                    INSERT INTO active_symbols (symbol, ev_score, rank, volume_24h, is_active)
                    VALUES ($1, $2, $3, $4, TRUE)
                    ON CONFLICT (symbol)
                    DO UPDATE SET
                        ev_score = EXCLUDED.ev_score,
                        rank = EXCLUDED.rank,
                        volume_24h = EXCLUDED.volume_24h,
                        is_active = TRUE,
                        updated_at = NOW()
                """, sym['symbol'], sym['ev_score'], sym['rank'], sym['volume_24h'])

        logger.info(f"✅ Обновлено {len(top_symbols)} активных символов")

    async def get_active_symbols(self) -> List[str]:
        """
        Получение списка активных символов

        Returns:
            List[str]: Список символов
        """
        query = """
            SELECT symbol
            FROM active_symbols
            WHERE is_active = TRUE
            ORDER BY rank
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)

        return [row['symbol'] for row in rows]

    async def update_position_time(
        self,
        symbol: str
    ):
        """
        Обновление времени последней позиции для символа

        Args:
            symbol: Символ
        """
        query = """
            UPDATE active_symbols
            SET
                last_position_at = NOW(),
                positions_count = positions_count + 1
            WHERE symbol = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, symbol)

        logger.debug(f"✓ Обновлено время последней позиции для {symbol}")

    # ========================================================================
    # ОЧИСТКА СТАРЫХ ДАННЫХ
    # ========================================================================

    async def cleanup_inactive_symbols(
        self,
        days_threshold: int = 7
    ) -> List[str]:
        """
        Удаление данных для неактивных символов

        Критерии удаления:
        1. Символ не в топ-10 (is_active = FALSE)
        2. Прошло более days_threshold дней с последней позиции

        Args:
            days_threshold: Порог в днях (по умолчанию 7)

        Returns:
            List[str]: Список удалённых символов
        """
        logger.info(f"Поиск неактивных символов (порог: {days_threshold} дней)...")

        # Получаем список символов для удаления
        query = """
            SELECT symbol
            FROM active_symbols
            WHERE is_active = FALSE
              AND (
                  last_position_at IS NULL
                  OR last_position_at < NOW() - INTERVAL '%s days'
              )
        """ % days_threshold

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)

        symbols_to_delete = [row['symbol'] for row in rows]

        if not symbols_to_delete:
            logger.info("✅ Нет символов для удаления")
            return []

        logger.info(f"Найдено символов для удаления: {len(symbols_to_delete)}")

        # Удаляем данные для каждого символа
        deleted_symbols = []

        for symbol in symbols_to_delete:
            try:
                rows_deleted = await self._cleanup_symbol(
                    symbol,
                    reason=f'not_in_top10_{days_threshold}days'
                )

                deleted_symbols.append(symbol)
                logger.info(f"✓ {symbol}: удалено {rows_deleted} строк")

            except Exception as e:
                logger.error(f"✗ {symbol}: ошибка удаления - {e}")

        logger.info(f"✅ Удалено данных для {len(deleted_symbols)} символов")

        return deleted_symbols

    async def _cleanup_symbol(
        self,
        symbol: str,
        reason: str = 'manual'
    ) -> int:
        """
        Удаление всех данных для символа (внутренняя функция)

        Args:
            symbol: Символ
            reason: Причина удаления

        Returns:
            int: Количество удалённых строк OHLCV
        """
        async with self.pool.acquire() as conn:
            # Используем SQL функцию cleanup_symbol_data
            rows_deleted = await conn.fetchval(
                "SELECT cleanup_symbol_data($1, $2)",
                symbol,
                reason
            )

        return rows_deleted

    # ========================================================================
    # СТАТИСТИКА
    # ========================================================================

    async def refresh_stats(self):
        """Обновление статистики по всем данным"""
        logger.info("Обновление статистики...")

        async with self.pool.acquire() as conn:
            await conn.execute("SELECT refresh_data_stats()")

        logger.info("✅ Статистика обновлена")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Получение общей статистики

        Returns:
            Dict со статистикой
        """
        queries = {
            'total_rows': "SELECT COUNT(*) FROM ohlcv_data",
            'total_symbols': "SELECT COUNT(DISTINCT symbol) FROM ohlcv_data",
            'active_symbols': "SELECT COUNT(*) FROM active_symbols WHERE is_active = TRUE",
            'total_size_mb': """
                SELECT ROUND(pg_total_relation_size('ohlcv_data') / 1024.0 / 1024.0, 2)
            """
        }

        stats = {}

        async with self.pool.acquire() as conn:
            for key, query in queries.items():
                stats[key] = await conn.fetchval(query)

        return stats

    async def get_symbol_stats(self, symbol: str) -> Dict[str, Any]:
        """
        Получение статистики для конкретного символа

        Args:
            symbol: Символ

        Returns:
            Dict со статистикой
        """
        query = """
            SELECT
                timeframe,
                rows_count,
                first_timestamp,
                last_timestamp,
                data_size_mb,
                last_update
            FROM data_stats
            WHERE symbol = $1
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol)

        stats = {
            'symbol': symbol,
            'timeframes': {}
        }

        for row in rows:
            stats['timeframes'][row['timeframe']] = {
                'rows_count': row['rows_count'],
                'first_timestamp': row['first_timestamp'],
                'last_timestamp': row['last_timestamp'],
                'data_size_mb': float(row['data_size_mb']) if row['data_size_mb'] else 0,
                'last_update': row['last_update']
            }

        return stats

    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    # ========================================================================

    async def __aenter__(self):
        """Context manager: вход"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager: выход"""
        await self.disconnect()


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

async def create_database_from_env() -> DatabaseManager:
    """
    Создание DatabaseManager из переменных окружения

    Переменные:
    - DB_HOST (default: localhost)
    - DB_PORT (default: 5432)
    - DB_NAME (default: binance_bot)
    - DB_USER (default: bot_user)
    - DB_PASSWORD (default: bot_password)
    """
    db = DatabaseManager(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432')),
        database=os.getenv('DB_NAME', 'binance_bot'),
        user=os.getenv('DB_USER', 'bot_user'),
        password=os.getenv('DB_PASSWORD', 'bot_password')
    )

    await db.connect()

    return db


# ============================================================================
# MAIN (для тестирования)
# ============================================================================

if __name__ == '__main__':
    async def test():
        """Тест подключения и базовых операций"""
        logging.basicConfig(level=logging.INFO)

        # Создание БД менеджера
        db = DatabaseManager(
            host='localhost',
            port=5432,
            database='binance_bot',
            user='bot_user',
            password='bot_password'
        )

        async with db:
            # Тест подключения
            logger.info("✓ Подключение успешно")

            # Тест статистики
            stats = await db.get_stats()
            logger.info(f"Статистика: {stats}")

            # Тест получения активных символов
            active = await db.get_active_symbols()
            logger.info(f"Активные символы: {active}")

    # Запуск теста
    asyncio.run(test())
