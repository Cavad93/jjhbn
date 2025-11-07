"""
Coin Selector для Binance Futures Bot

Отбирает топ-10 торговых возможностей (LONG или SHORT) на основе Expected Value.

Процесс:
1. Скрининг ~400 монет
2. Применение фильтров (ликвидность, blacklist, etc)
3. Расчёт 68 фич для каждой монеты
4. Получение предсказаний от ML моделей
5. Расчёт EV для LONG и SHORT
6. Выбор направления с максимальным EV
7. Сортировка по EV и выбор топ-10
8. Применение sector diversification

Автор: Claude Code
Дата: 2025-11-06
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from .sector_config import get_coin_sector, get_sector_limit, count_coins_per_sector
    from .blacklist import is_blacklisted, is_high_risk, get_risk_level, get_max_allocation
except ImportError:
    import sys
    sys.path.append('/home/user/jjhbn/GGG3')
    from strategy.sector_config import get_coin_sector, get_sector_limit, count_coins_per_sector
    from strategy.blacklist import is_blacklisted, is_high_risk, get_risk_level, get_max_allocation


class CoinSelector:
    """
    Селектор монет для торговли

    Отбирает топ-N возможностей (LONG или SHORT) на основе Expected Value
    """

    def __init__(
        self,
        min_volume_24h: float = 5_000_000,      # $5M минимум
        max_spread_pct: float = 0.002,          # 0.2% максимум
        min_atr_pct: float = 0.02,              # 2% минимум
        max_atr_pct: float = 0.15,              # 15% максимум
        min_market_cap: float = 50_000_000,     # $50M минимум
        min_listing_age_days: int = 30,         # 30 дней минимум
        min_volume_vs_btc: float = 0.001,       # 0.1% от BTC минимум
        min_ev_threshold: float = 0.02,         # 2% минимум EV
        sector_diversification: bool = True,
        verbose: bool = False
    ):
        """
        Инициализация CoinSelector

        Args:
            min_volume_24h: Минимальный объём 24h (USDT)
            max_spread_pct: Максимальный спред (%)
            min_atr_pct: Минимальный ATR (% от цены)
            max_atr_pct: Максимальный ATR (% от цены)
            min_market_cap: Минимальная капитализация (USDT)
            min_listing_age_days: Минимальный возраст листинга (дни)
            min_volume_vs_btc: Минимальный объём относительно BTC
            min_ev_threshold: Минимальный EV для входа
            sector_diversification: Применять ли sector diversification
            verbose: Выводить ли подробные логи
        """
        self.min_volume_24h = min_volume_24h
        self.max_spread_pct = max_spread_pct
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct
        self.min_market_cap = min_market_cap
        self.min_listing_age_days = min_listing_age_days
        self.min_volume_vs_btc = min_volume_vs_btc
        self.min_ev_threshold = min_ev_threshold
        self.sector_diversification = sector_diversification
        self.verbose = verbose

        # Статистика фильтрации
        self.filter_stats = {
            'total_pairs': 0,
            'blacklisted': 0,
            'low_volume': 0,
            'high_spread': 0,
            'atr_out_of_range': 0,
            'low_market_cap': 0,
            'new_listing': 0,
            'low_volume_vs_btc': 0,
            'passed_filters': 0
        }

    def calculate_ev_bidirectional(self, p_up: float) -> Tuple[float, float]:
        """
        Рассчитывает Expected Value для LONG и SHORT

        При R/R = 1.5:1 (TP = 1.2×ATR, SL = 0.8×ATR):
        - Reward = 1.5R
        - Risk = 1.0R

        EV_long = P(TP) × Reward - P(SL) × Risk
                = p_up × 1.5 - (1 - p_up) × 1.0
                = p_up × 2.5 - 1.0

        EV_short = P(TP) × Reward - P(SL) × Risk
                 = (1 - p_up) × 1.5 - p_up × 1.0
                 = 1.5 - p_up × 2.5

        Break-even:
        - EV_long > 0 требует p_up > 0.40 (40%)
        - EV_short > 0 требует p_up < 0.60 (60%)

        Args:
            p_up: Вероятность роста цены (0-1)

        Returns:
            Tuple[ev_long, ev_short]

        Examples:
            >>> selector = CoinSelector()
            >>> ev_long, ev_short = selector.calculate_ev_bidirectional(0.67)
            >>> print(f"LONG EV: {ev_long:.4f}")  # 0.0675 (+6.75%)
            >>> print(f"SHORT EV: {ev_short:.4f}")  # -0.1750 (-17.5%)
        """
        ev_long = p_up * 2.5 - 1.0
        ev_short = 1.5 - p_up * 2.5

        return ev_long, ev_short

    def apply_filters(
        self,
        symbol: str,
        ticker_data: Optional[dict] = None,
        market_data: Optional[dict] = None
    ) -> Tuple[bool, str]:
        """
        Применяет фильтры ликвидности и blacklist

        Фильтры:
        1. Blacklist (скам, delisted, etc)
        2. Объём 24h > min_volume_24h
        3. Спред < max_spread_pct
        4. ATR в пределах [min_atr_pct, max_atr_pct]
        5. Market cap > min_market_cap
        6. Листинг > min_listing_age_days дней назад
        7. Объём > min_volume_vs_btc от BTC
        8. Не в высокорисковых категориях (опционально)

        Args:
            symbol: Символ монеты (например, 'BTCUSDT')
            ticker_data: Данные тикера с биржи (опционально)
            market_data: Рыночные данные (опционально)

        Returns:
            Tuple[passed: bool, reason: str]

        Examples:
            >>> selector = CoinSelector()
            >>> passed, reason = selector.apply_filters('BTCUSDT')
            >>> if passed:
            >>>     print(f"{symbol} прошёл фильтры")
            >>> else:
            >>>     print(f"{symbol} отфильтрован: {reason}")
        """
        # Фильтр 1: Blacklist
        if is_blacklisted(symbol):
            self.filter_stats['blacklisted'] += 1
            return False, f"В blacklist"

        # Фильтр 2: Объём 24h
        if ticker_data:
            volume_24h = float(ticker_data.get('quoteVolume', 0))
            if volume_24h < self.min_volume_24h:
                self.filter_stats['low_volume'] += 1
                return False, f"Низкий объём: ${volume_24h/1e6:.1f}M < ${self.min_volume_24h/1e6:.1f}M"

        # Фильтр 3: Спред
        if ticker_data:
            bid_price = float(ticker_data.get('bidPrice', 0))
            ask_price = float(ticker_data.get('askPrice', 0))

            if bid_price > 0 and ask_price > 0:
                spread_pct = (ask_price - bid_price) / bid_price
                if spread_pct > self.max_spread_pct:
                    self.filter_stats['high_spread'] += 1
                    return False, f"Высокий спред: {spread_pct*100:.2f}% > {self.max_spread_pct*100:.2f}%"

        # Фильтр 4: ATR
        if market_data and 'atr_pct' in market_data:
            atr_pct = market_data['atr_pct']
            if atr_pct < self.min_atr_pct or atr_pct > self.max_atr_pct:
                self.filter_stats['atr_out_of_range'] += 1
                return False, f"ATR вне допустимого диапазона: {atr_pct*100:.1f}%"

        # Фильтр 5: Market cap (симулируется через объём × множитель)
        # В реальности нужны данные от CoinGecko/CoinMarketCap API
        if ticker_data:
            volume_24h = float(ticker_data.get('quoteVolume', 0))
            # Примерная оценка: market cap ~ 50× дневной объём
            estimated_market_cap = volume_24h * 50

            if estimated_market_cap < self.min_market_cap:
                self.filter_stats['low_market_cap'] += 1
                return False, f"Низкая капитализация (оценка): ${estimated_market_cap/1e6:.1f}M"

        # Фильтр 6: Возраст листинга (в реальности нужен API)
        # Здесь пропускаем, так как нужна дополнительная информация

        # Фильтр 7: Объём относительно BTC (в реальности нужен BTC объём)
        # Здесь пропускаем, так как нужна дополнительная информация

        # Все фильтры пройдены
        self.filter_stats['passed_filters'] += 1
        return True, "Passed all filters"

    def _apply_sector_limits(self, opportunities: List[dict]) -> List[dict]:
        """
        Применяет ограничения по секторам (diversification)

        Ограничивает количество монет из одного сектора в портфеле.

        Args:
            opportunities: Список возможностей, отсортированный по EV

        Returns:
            Отфильтрованный список с учётом sector limits
        """
        if not self.sector_diversification:
            return opportunities

        sector_counts: Dict[str, int] = {}
        filtered: List[dict] = []

        for opp in opportunities:
            symbol = opp['symbol']
            sector = get_coin_sector(symbol)
            count = sector_counts.get(sector, 0)
            limit = get_sector_limit(sector)

            if count < limit:
                filtered.append(opp)
                sector_counts[sector] = count + 1

                if self.verbose:
                    print(f"  ✓ {symbol:15s} ({sector:15s}) - {count+1}/{limit} в секторе")
            else:
                if self.verbose:
                    print(f"  ✗ {symbol:15s} ({sector:15s}) - Превышен лимит {limit} в секторе")

        return filtered

    def select_top_coins(
        self,
        all_pairs: List[str],
        get_ticker_func: callable,
        get_ohlcv_func: callable,
        feature_builder,
        calculate_atr_func: callable,
        calculate_phase_func: callable,
        get_predictions_func: callable = None,
        top_n: int = 10
    ) -> List[dict]:
        """
        Отбирает топ-N торговых возможностей

        Процесс:
        1. Для каждой монеты:
           - Применить фильтры
           - Получить 68 фич (multi-TF)
           - Получить p_meta (вероятность роста)
           - Рассчитать EV_long и EV_short
           - Выбрать направление с max(EV) > threshold
        2. Сортировать по EV (descending)
        3. Применить sector diversification
        4. Вернуть топ-N

        Args:
            all_pairs: Список всех пар для скрининга
            get_ticker_func: Функция для получения ticker данных
            get_ohlcv_func: Функция для получения OHLCV данных
            feature_builder: BinanceFeatureBuilder instance
            calculate_atr_func: Функция для расчёта ATR
            calculate_phase_func: Функция для определения фазы рынка
            get_predictions_func: Функция для получения предсказаний (опционально)
            top_n: Количество возможностей для возврата

        Returns:
            Список словарей с торговыми возможностями:
            [
                {
                    'symbol': 'BTCUSDT',
                    'direction': 'LONG',
                    'p_up': 0.67,
                    'ev': 0.0675,
                    'atr': 0.032,
                    'atr_value': 1500.0,
                    'phase': 2,
                    'entry_price': 50000.0,
                    'tp_price': 51800.0,
                    'sl_price': 48800.0,
                    'sector': 'BTC',
                    'risk_level': 'LOW',
                    'timestamp': '2025-11-06T12:00:00'
                },
                ...
            ]

        Examples:
            >>> from binance_api import BinanceClient
            >>> from features import BinanceFeatureBuilder, calculate_atr, detect_market_phase
            >>>
            >>> client = BinanceClient(...)
            >>> builder = BinanceFeatureBuilder()
            >>> selector = CoinSelector(min_ev_threshold=0.02, verbose=True)
            >>>
            >>> all_pairs = client.get_all_usdt_pairs()
            >>>
            >>> opportunities = selector.select_top_coins(
            ...     all_pairs=all_pairs,
            ...     get_ticker_func=lambda s: client.get_ticker(s),
            ...     get_ohlcv_func=lambda s, tf, lim: client.get_ohlcv(s, tf, lim),
            ...     feature_builder=builder,
            ...     calculate_atr_func=calculate_atr,
            ...     calculate_phase_func=detect_market_phase,
            ...     top_n=10
            ... )
            >>>
            >>> for opp in opportunities:
            >>>     print(f"{opp['symbol']:15s} {opp['direction']:5s} EV={opp['ev']:+.4f}")
        """
        if self.verbose:
            print("="*80)
            print("СКРИНИНГ МОНЕТ ДЛЯ ТОРГОВЫХ ВОЗМОЖНОСТЕЙ")
            print("="*80)
            print(f"Всего пар для скрининга: {len(all_pairs)}")
            print(f"Минимальный EV: {self.min_ev_threshold*100:.1f}%")
            print(f"Целевое количество: {top_n}")
            print()

        # Сброс статистики
        for key in self.filter_stats:
            self.filter_stats[key] = 0

        self.filter_stats['total_pairs'] = len(all_pairs)

        opportunities: List[dict] = []

        for symbol in all_pairs:
            try:
                # Получаем ticker данные
                ticker_data = get_ticker_func(symbol)

                # Получаем OHLCV для всех таймфреймов
                df_5m = get_ohlcv_func(symbol, '5m', 200)
                df_15m = get_ohlcv_func(symbol, '15m', 200)
                df_30m = get_ohlcv_func(symbol, '30m', 200)
                df_4h = get_ohlcv_func(symbol, '4h', 100)

                # Проверяем достаточно ли данных
                if (len(df_5m) < 100 or len(df_15m) < 100 or
                    len(df_30m) < 100 or len(df_4h) < 50):
                    continue

                # Рассчитываем ATR
                atr_series = calculate_atr_func(df_4h, period=10)
                atr_value = atr_series.iloc[-1]
                entry_price = df_4h['close'].iloc[-1]
                atr_pct = (atr_value / entry_price) * 100

                # Применяем фильтры
                market_data = {'atr_pct': atr_pct / 100}  # В долях
                passed, reason = self.apply_filters(symbol, ticker_data, market_data)

                if not passed:
                    if self.verbose:
                        print(f"✗ {symbol:15s}: {reason}")
                    continue

                # Рассчитываем 68 фич
                features = feature_builder.build_extended_features(
                    df_5m=df_5m,
                    df_15m=df_15m,
                    df_30m=df_30m,
                    df_4h=df_4h,
                    current_time=df_4h['timestamp'].iloc[-1]
                )

                # Определяем фазу рынка
                phase = calculate_phase_func(df_4h)

                # Получаем предсказание p_up
                if get_predictions_func:
                    p_up = get_predictions_func(features, phase, symbol, df_4h)  # Добавлен df_4h
                else:
                    # Если нет функции предсказаний, используем dummy значение
                    # В реальности здесь должны быть predictions от ML моделей
                    p_up = 0.5  # Нейтральное значение

                # Рассчитываем EV для обоих направлений
                ev_long, ev_short = self.calculate_ev_bidirectional(p_up)

                # Выбираем лучшее направление
                if ev_long > self.min_ev_threshold and ev_long > ev_short:
                    direction = 'LONG'
                    ev = ev_long
                    tp_price = entry_price + 1.2 * atr_value
                    sl_price = entry_price - 0.8 * atr_value

                elif ev_short > self.min_ev_threshold and ev_short > ev_long:
                    direction = 'SHORT'
                    ev = ev_short
                    tp_price = entry_price - 1.2 * atr_value
                    sl_price = entry_price + 0.8 * atr_value

                else:
                    # Нет подходящего направления с достаточным EV
                    continue

                # Получаем sector и risk level
                sector = get_coin_sector(symbol)
                risk_level = get_risk_level(symbol)

                # Формируем opportunity
                opportunity = {
                    'symbol': symbol,
                    'direction': direction,
                    'p_up': p_up,
                    'ev': ev,
                    'atr': atr_pct / 100,  # В долях
                    'atr_value': atr_value,
                    'phase': phase,
                    'entry_price': entry_price,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'sector': sector,
                    'risk_level': risk_level,
                    'timestamp': datetime.now().isoformat()
                }

                opportunities.append(opportunity)

                if self.verbose:
                    print(f"✓ {symbol:15s} {direction:5s} EV={ev:+.4f} ({ev*100:+.1f}%) p_up={p_up:.3f} ATR={atr_pct:.1f}%")

            except Exception as e:
                if self.verbose:
                    print(f"⚠ {symbol:15s}: Ошибка - {str(e)[:50]}")
                continue

        if self.verbose:
            print()
            print(f"Найдено возможностей до sector filter: {len(opportunities)}")

        # Сортируем по EV (descending)
        opportunities.sort(key=lambda x: x['ev'], reverse=True)

        # Применяем sector diversification
        if self.sector_diversification:
            if self.verbose:
                print("\nПрименение sector diversification:")
            opportunities = self._apply_sector_limits(opportunities)

        # Возвращаем топ-N
        result = opportunities[:top_n]

        if self.verbose:
            print()
            print("="*80)
            print("РЕЗУЛЬТАТЫ СКРИНИНГА")
            print("="*80)
            print(f"Всего пар проверено: {self.filter_stats['total_pairs']}")
            print(f"Blacklisted: {self.filter_stats['blacklisted']}")
            print(f"Низкий объём: {self.filter_stats['low_volume']}")
            print(f"Высокий спред: {self.filter_stats['high_spread']}")
            print(f"ATR вне диапазона: {self.filter_stats['atr_out_of_range']}")
            print(f"Низкая капитализация: {self.filter_stats['low_market_cap']}")
            print(f"Прошли фильтры: {self.filter_stats['passed_filters']}")
            print(f"Итоговое количество: {len(result)}")
            print()

            if result:
                print("ТОП ВОЗМОЖНОСТИ:")
                for i, opp in enumerate(result, 1):
                    print(f"{i:2d}. {opp['symbol']:15s} {opp['direction']:5s} "
                          f"EV={opp['ev']:+.4f} ({opp['ev']*100:+.1f}%) "
                          f"p_up={opp['p_up']:.3f} "
                          f"ATR={opp['atr']*100:.1f}% "
                          f"Sector={opp['sector']:10s}")
            print("="*80)

        return result

    def get_filter_stats(self) -> dict:
        """
        Возвращает статистику фильтрации

        Returns:
            Словарь со статистикой
        """
        return self.filter_stats.copy()


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def create_mock_predictions_func(p_up_mean: float = 0.5, p_up_std: float = 0.1):
    """
    Создаёт mock функцию для предсказаний (для тестирования)

    Args:
        p_up_mean: Среднее значение p_up
        p_up_std: Стандартное отклонение p_up

    Returns:
        Функция для получения предсказаний
    """
    def mock_predictions(features, phase, symbol):
        # Генерируем случайное значение p_up
        np.random.seed(hash(symbol) % (2**32))
        p_up = np.clip(np.random.normal(p_up_mean, p_up_std), 0.1, 0.9)
        return p_up

    return mock_predictions


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ТЕСТ COIN SELECTOR")
    print("="*80)

    # Тест 1: Расчёт EV
    print("\n1️⃣ Тест расчёта EV:")
    selector = CoinSelector()

    test_probabilities = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    print(f"\n{'p_up':>6s} | {'EV_LONG':>10s} | {'EV_SHORT':>10s} | {'Рекомендация':>15s}")
    print("-" * 55)

    for p_up in test_probabilities:
        ev_long, ev_short = selector.calculate_ev_bidirectional(p_up)

        if ev_long > 0.02 and ev_long > ev_short:
            recommendation = "LONG"
        elif ev_short > 0.02 and ev_short > ev_long:
            recommendation = "SHORT"
        else:
            recommendation = "SKIP"

        print(f"{p_up:6.2f} | {ev_long:+10.4f} | {ev_short:+10.4f} | {recommendation:>15s}")

    # Тест 2: Фильтры
    print("\n\n2️⃣ Тест фильтров:")
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'USTCUSDT', 'SOLUSDT']

    for symbol in test_symbols:
        ticker_data = {'quoteVolume': 10_000_000}  # $10M
        passed, reason = selector.apply_filters(symbol, ticker_data)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {symbol:15s}: {status:10s} - {reason}")

    # Тест 3: Sector diversification
    print("\n\n3️⃣ Тест sector diversification:")

    mock_opportunities = [
        {'symbol': 'BTCUSDT', 'ev': 0.08},
        {'symbol': 'ETHUSDT', 'ev': 0.07},
        {'symbol': 'SOLUSDT', 'ev': 0.06},  # L1
        {'symbol': 'AVAXUSDT', 'ev': 0.055},  # L1
        {'symbol': 'ADAUSDT', 'ev': 0.05},  # L1
        {'symbol': 'DOTUSDT', 'ev': 0.048},  # L1 - должен быть отфильтрован (лимит 3)
        {'symbol': 'DOGEUSDT', 'ev': 0.045},  # MEME
        {'symbol': 'UNIUSDT', 'ev': 0.04},  # DEFI
    ]

    print(f"До фильтрации: {len(mock_opportunities)} возможностей")
    filtered = selector._apply_sector_limits(mock_opportunities)
    print(f"После фильтрации: {len(filtered)} возможностей")

    print("\nРезультат:")
    for i, opp in enumerate(filtered, 1):
        sector = get_coin_sector(opp['symbol'])
        print(f"  {i}. {opp['symbol']:15s} (Sector: {sector:10s}) EV={opp['ev']:.4f}")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*80)
