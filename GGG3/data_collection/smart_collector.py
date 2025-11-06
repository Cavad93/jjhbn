#!/usr/bin/env python3
"""
Умный сборщик данных - загружает только топ-10 монет по EV

Особенности:
- Загрузка только для топ-10 монет (не всех 300-400)
- Автоматическое удаление данных для монет вне топ-10
- Асинхронное выполнение
- Интеграция с PostgreSQL
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

# Добавляем родительскую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from binance_api.client import BinanceClient
from binance_api.exceptions import BinanceAPIError
from data_collection.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class SmartCollector:
    """
    Умный сборщик данных для топ-10 монет

    Workflow:
    1. Получить все USDT пары с Binance
    2. Отфильтровать по объёму (> 1M USDT)
    3. Рассчитать EV для каждой пары
    4. Выбрать топ-10 по EV
    5. Загрузить/обновить данные только для топ-10
    6. Удалить данные для монет вне топ-10 (если прошло 7+ дней)
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        binance_client: BinanceClient,
        timeframes: Optional[List[str]] = None,
        limit_per_timeframe: int = 500
    ):
        """
        Args:
            db_manager: Менеджер базы данных
            binance_client: Клиент Binance API
            timeframes: Список таймфреймов (default: ['5m', '15m', '30m', '4h'])
            limit_per_timeframe: Количество свечей для загрузки
        """
        self.db = db_manager
        self.client = binance_client

        self.timeframes = timeframes or ['5m', '15m', '30m', '4h']
        self.limit = limit_per_timeframe

    # ========================================================================
    # ФИЛЬТРАЦИЯ И РАНЖИРОВАНИЕ
    # ========================================================================

    async def get_top_symbols_by_ev(
        self,
        min_volume_24h: float = 1_000_000,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Получение топ-N символов по EV (Expected Value)

        ВАЖНО: Эта функция должна вызывать вашу ML модель для расчёта EV.
        Сейчас используется заглушка с объёмом торгов.

        Args:
            min_volume_24h: Минимальный объём торгов
            top_n: Количество топ монет (обычно 10)

        Returns:
            List[Dict]: Топ-N символов с EV scores
                [
                    {'symbol': 'BTCUSDT', 'ev_score': 0.045, 'rank': 1, 'volume_24h': 25000000000},
                    ...
                ]
        """
        logger.info(f"Получение топ-{top_n} символов по EV...")

        # Получаем все USDT пары
        all_pairs = self.client.get_all_usdt_pairs(min_volume_24h=min_volume_24h)

        logger.info(f"Найдено {len(all_pairs)} пар с объёмом > {min_volume_24h:,.0f}")

        # TODO: ИНТЕГРАЦИЯ С ML МОДЕЛЬЮ
        # Здесь должен быть вызов вашей модели для расчёта EV для каждой пары
        #
        # Пример:
        # ev_scores = []
        # for symbol in all_pairs:
        #     features = await calculate_features(symbol)
        #     predictions = model.predict(features)
        #     ev = calculate_ev(predictions, tp_sl_params)
        #     ev_scores.append({'symbol': symbol, 'ev_score': ev})
        #
        # Сейчас используем заглушку: сортируем по объёму торгов

        # ЗАГЛУШКА: Получаем тикеры 24h для сортировки по объёму
        logger.info("Получение данных о объёмах торгов (временная заглушка для EV)...")

        try:
            tickers = self.client.client.futures_ticker()

            # Создаём словарь symbol -> volume_24h
            volume_dict = {
                t['symbol']: float(t['quoteVolume'])
                for t in tickers
                if t['symbol'] in all_pairs
            }

            # Сортируем по объёму (заглушка для EV)
            sorted_pairs = sorted(
                volume_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]

            # Формируем результат
            top_symbols = []
            for rank, (symbol, volume) in enumerate(sorted_pairs, start=1):
                # ЗАГЛУШКА: ev_score = normalized volume
                # В реальности здесь должна быть ML модель
                max_volume = sorted_pairs[0][1]
                ev_score = volume / max_volume * 0.05  # Нормализация к 0-0.05

                top_symbols.append({
                    'symbol': symbol,
                    'ev_score': round(ev_score, 6),
                    'rank': rank,
                    'volume_24h': volume
                })

            logger.info(f"✅ Получено топ-{len(top_symbols)} символов:")
            for sym in top_symbols:
                logger.info(f"  {sym['rank']}. {sym['symbol']}: EV={sym['ev_score']:.4f}, "
                           f"Volume=${sym['volume_24h']:,.0f}")

            return top_symbols

        except Exception as e:
            logger.error(f"Ошибка при получении топ символов: {e}")
            raise

    # ========================================================================
    # ЗАГРУЗКА ДАННЫХ
    # ========================================================================

    async def download_symbol_data(
        self,
        symbol: str,
        timeframe: str
    ) -> int:
        """
        Загрузка данных для символа и таймфрейма

        Args:
            symbol: Символ (BTCUSDT)
            timeframe: Таймфрейм (5m, 15m, etc.)

        Returns:
            int: Количество загруженных свечей
        """
        try:
            # Получаем данные через Binance API
            df = self.client.get_ohlcv(symbol, timeframe, limit=self.limit)

            if df is None or len(df) == 0:
                logger.warning(f"⚠️  {symbol} {timeframe}: нет данных")
                return 0

            # Вставляем в БД
            rows_inserted = await self.db.insert_ohlcv_batch(symbol, timeframe, df)

            return rows_inserted

        except BinanceAPIError as e:
            logger.error(f"✗ {symbol} {timeframe}: Binance API ошибка - {e}")
            return 0

        except Exception as e:
            logger.error(f"✗ {symbol} {timeframe}: {e}")
            return 0

    async def update_symbol_data(
        self,
        symbol: str,
        timeframe: str
    ) -> int:
        """
        Обновление данных для символа (дозагрузка новых свечей)

        Args:
            symbol: Символ
            timeframe: Таймфрейм

        Returns:
            int: Количество добавленных свечей
        """
        try:
            # Получаем последнюю временную метку из БД
            last_timestamp = await self.db.get_last_timestamp(symbol, timeframe)

            # Загружаем последние свечи через API
            df = self.client.get_ohlcv(symbol, timeframe, limit=self.limit)

            if df is None or len(df) == 0:
                logger.debug(f"{symbol} {timeframe}: нет новых данных от API")
                return 0

            # Фильтруем только новые свечи
            if last_timestamp:
                df = df[df['timestamp'] > last_timestamp]

            if len(df) == 0:
                logger.debug(f"{symbol} {timeframe}: нет новых свечей")
                return 0

            # Вставляем в БД
            rows_inserted = await self.db.insert_ohlcv_batch(symbol, timeframe, df)

            logger.info(f"✓ {symbol} {timeframe}: добавлено {rows_inserted} свечей")

            return rows_inserted

        except Exception as e:
            logger.error(f"✗ {symbol} {timeframe}: ошибка обновления - {e}")
            return 0

    # ========================================================================
    # МАССОВЫЕ ОПЕРАЦИИ
    # ========================================================================

    async def collect_top_symbols_data(
        self,
        top_n: int = 10,
        min_volume_24h: float = 1_000_000
    ) -> Dict[str, Any]:
        """
        Загрузка данных для топ-N символов

        Workflow:
        1. Получить топ-N по EV
        2. Обновить список активных символов в БД
        3. Загрузить/обновить данные для каждого символа
        4. Обновить статистику

        Args:
            top_n: Количество топ монет (обычно 10)
            min_volume_24h: Минимальный объём торгов

        Returns:
            Dict: Результаты загрузки
        """
        logger.info("=" * 80)
        logger.info("ЗАГРУЗКА ДАННЫХ ДЛЯ ТОП-10 МОНЕТ")
        logger.info("=" * 80)

        # Этап 1: Получение топ-N символов
        logger.info(f"\nЭтап 1: Получение топ-{top_n} символов по EV")
        top_symbols = await self.get_top_symbols_by_ev(min_volume_24h, top_n)

        if not top_symbols:
            logger.error("❌ Не удалось получить топ символы")
            return {'error': 'No top symbols found'}

        # Этап 2: Обновление списка активных символов в БД
        logger.info(f"\nЭтап 2: Обновление активных символов в БД")
        await self.db.update_active_symbols(top_symbols)

        # Этап 3: Загрузка данных
        logger.info(f"\nЭтап 3: Загрузка/обновление данных")
        logger.info(f"Символов: {len(top_symbols)}, Таймфреймов: {len(self.timeframes)}")

        symbols = [s['symbol'] for s in top_symbols]
        total_inserted = 0
        success_count = 0
        failed_symbols = []

        for symbol in symbols:
            try:
                symbol_inserted = 0

                for timeframe in self.timeframes:
                    # Проверяем, есть ли уже данные
                    last_ts = await self.db.get_last_timestamp(symbol, timeframe)

                    if last_ts:
                        # Обновление
                        rows = await self.update_symbol_data(symbol, timeframe)
                    else:
                        # Первичная загрузка
                        rows = await self.download_symbol_data(symbol, timeframe)

                    symbol_inserted += rows

                    # Небольшая пауза для rate limiting
                    await asyncio.sleep(0.1)

                total_inserted += symbol_inserted
                success_count += 1

                logger.info(f"✓ {symbol}: загружено {symbol_inserted} свечей")

                # Пауза между символами
                await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"✗ {symbol}: ошибка - {e}")
                failed_symbols.append(symbol)

        # Этап 4: Обновление статистики
        logger.info(f"\nЭтап 4: Обновление статистики")
        await self.db.refresh_stats()

        # Итоги
        logger.info("\n" + "=" * 80)
        logger.info("ИТОГИ ЗАГРУЗКИ")
        logger.info("=" * 80)
        logger.info(f"✅ Успешно: {success_count}/{len(symbols)} символов")
        logger.info(f"📊 Всего загружено: {total_inserted} свечей")

        if failed_symbols:
            logger.warning(f"❌ Не удалось: {failed_symbols}")

        return {
            'top_symbols': top_symbols,
            'success_count': success_count,
            'failed_symbols': failed_symbols,
            'total_inserted': total_inserted,
            'timestamp': datetime.now().isoformat()
        }

    async def cleanup_old_data(
        self,
        days_threshold: int = 7
    ) -> List[str]:
        """
        Асинхронная очистка старых данных

        Удаляет данные для монет, которые:
        1. Не в топ-10 (is_active = FALSE)
        2. Прошло более days_threshold дней с последней позиции

        Args:
            days_threshold: Порог в днях (по умолчанию 7)

        Returns:
            List[str]: Список удалённых символов
        """
        logger.info("=" * 80)
        logger.info("ОЧИСТКА СТАРЫХ ДАННЫХ")
        logger.info("=" * 80)
        logger.info(f"Порог: {days_threshold} дней без позиций")

        deleted_symbols = await self.db.cleanup_inactive_symbols(days_threshold)

        logger.info("=" * 80)
        logger.info(f"✅ Очистка завершена: удалено {len(deleted_symbols)} символов")
        logger.info("=" * 80)

        return deleted_symbols

    # ========================================================================
    # АВТОМАТИЧЕСКИЙ РЕЖИМ
    # ========================================================================

    async def auto_update_loop(
        self,
        interval_minutes: int = 60,
        cleanup_days: int = 7
    ):
        """
        Автоматический режим обновления данных

        Args:
            interval_minutes: Интервал обновления в минутах (default: 60)
            cleanup_days: Порог для очистки в днях (default: 7)
        """
        logger.info("=" * 80)
        logger.info("АВТОМАТИЧЕСКИЙ РЕЖИМ ОБНОВЛЕНИЯ")
        logger.info("=" * 80)
        logger.info(f"Интервал обновления: {interval_minutes} минут")
        logger.info(f"Очистка: каждые 24 часа (порог {cleanup_days} дней)")

        iteration = 0

        while True:
            try:
                iteration += 1
                logger.info(f"\n{'=' * 80}")
                logger.info(f"ИТЕРАЦИЯ #{iteration} - {datetime.now()}")
                logger.info(f"{'=' * 80}")

                # Обновление данных для топ-10
                await self.collect_top_symbols_data()

                # Очистка старых данных (раз в 24 часа)
                if iteration % (1440 // interval_minutes) == 0:
                    await self.cleanup_old_data(cleanup_days)

                # Статистика
                stats = await self.db.get_stats()
                logger.info(f"\nТекущая статистика:")
                logger.info(f"  - Всего строк: {stats['total_rows']:,}")
                logger.info(f"  - Всего символов: {stats['total_symbols']}")
                logger.info(f"  - Активных символов: {stats['active_symbols']}")
                logger.info(f"  - Размер БД: {stats['total_size_mb']} MB")

                # Ожидание до следующей итерации
                logger.info(f"\n⏳ Ожидание {interval_minutes} минут до следующего обновления...")
                await asyncio.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("\n⚠️  Получен сигнал прерывания. Остановка...")
                break

            except Exception as e:
                logger.error(f"❌ Ошибка в auto_update_loop: {e}")
                logger.info("⏳ Ожидание 5 минут перед повтором...")
                await asyncio.sleep(300)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Smart Data Collector')
    parser.add_argument('--mode', choices=['once', 'auto', 'cleanup'], default='once',
                       help='Режим работы')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Количество топ монет')
    parser.add_argument('--interval', type=int, default=60,
                       help='Интервал обновления (минуты)')
    parser.add_argument('--cleanup-days', type=int, default=7,
                       help='Порог для очистки (дни)')

    args = parser.parse_args()

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def main():
        # Создание Binance клиента
        binance = BinanceClient(
            api_key="",
            secret_key="",
            testnet=False
        )

        # Создание DB менеджера
        db = DatabaseManager(
            host='localhost',
            port=5432,
            database='binance_bot',
            user='bot_user',
            password='bot_password'
        )

        async with db:
            # Создание умного сборщика
            collector = SmartCollector(
                db_manager=db,
                binance_client=binance
            )

            if args.mode == 'once':
                # Одноразовое обновление
                result = await collector.collect_top_symbols_data(top_n=args.top_n)
                print(f"\n✅ Результат: {result}")

            elif args.mode == 'auto':
                # Автоматический режим
                await collector.auto_update_loop(
                    interval_minutes=args.interval,
                    cleanup_days=args.cleanup_days
                )

            elif args.mode == 'cleanup':
                # Только очистка
                deleted = await collector.cleanup_old_data(days_threshold=args.cleanup_days)
                print(f"\n✅ Удалено символов: {len(deleted)}")

    # Запуск
    asyncio.run(main())
