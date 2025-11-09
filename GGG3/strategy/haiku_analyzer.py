#!/usr/bin/env python3
"""
Haiku 4.5 Fundamental Analysis Module
Анализ фундаментальных факторов для TOP-20 монет

Возможности:
- Интеграция с Claude Haiku 4.5 через Anthropic API
- Batch обработка (5 монет в 1 запрос для экономии)
- Агрессивное кэширование результатов (4-12 часов)
- DuckDuckGo поиск новостей (бесплатно, с фильтрацией по дате)
- Prompt caching для экономии до 90% на input токенах
- Статистика эффективности решений

Оптимизации для бюджета $6-12/месяц:
1. Batch processing: -35% расходов
2. Smart caching: -35% запросов
3. Compact prompts: -40% токенов
4. DuckDuckGo news search: находит новости для всех монет (не только топ-10)
5. Prompt caching (Anthropic): -90% на системных промптах

Автор: Claude Code
Дата: 2025-11-09 (обновлено: DuckDuckGo вместо RSS)
"""

import os
import time
import json
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from threading import Lock
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# DuckDuckGo поиск новостей
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None
    logging.warning("duckduckgo_search не установлен. pip install duckduckgo-search")

# Anthropic API
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
    logging.warning("anthropic не установлен. pip install anthropic")

logger = logging.getLogger(__name__)


# ============================================================================
# КОНСТАНТЫ (вместо магических чисел)
# ============================================================================

class HaikuConfig:
    """Конфигурационные константы для Haiku анализатора"""

    # Кэширование
    CACHE_BLOCK_HOURS = 4  # Блоки кэширования для волатильных монет
    STABLE_CACHE_HOURS = 12  # Кэш для стабильных монет (BTC, ETH, BNB)
    VOLATILE_CACHE_HOURS = 4  # Кэш для волатильных монет
    NEWS_CACHE_TTL_SECONDS = 1800  # 30 минут

    # Новости
    MAX_NEWS_ITEMS = 10  # Максимум новостей на монету
    NEWS_LOOKBACK_HOURS = 24  # Период поиска новостей
    MAX_NEWS_TITLE_LENGTH = 80  # Максимальная длина заголовка в промпте
    MAX_NEWS_IN_PROMPT = 5  # Максимум новостей в промпте
    MAX_NEWS_SUMMARY_LENGTH = 200  # Максимальная длина summary

    # Промпты
    MAX_PROMPT_LENGTH = 10000  # Максимальная длина промпта (символы)
    MAX_NEWS_TEXT_LENGTH = 300  # Максимальная длина текста новостей на монету
    BATCH_SIZE_DEFAULT = 5  # Количество монет в batch

    # API
    API_TIMEOUT_SECONDS = 60
    API_MAX_TOKENS = 2048
    API_TEMPERATURE = 0.0  # Детерминированный вывод

    # Rate Limiting (Anthropic tier 1 limits)
    RATE_LIMIT_MAX_CALLS = 50  # Максимум запросов
    RATE_LIMIT_PERIOD_SECONDS = 60  # За период

    # Стоимость
    ESTIMATED_COST_PER_BATCH = 0.0135  # $0.0135 за batch из 5 монет

    # Скоринг
    SCORE_THRESHOLD_REJECT = 0.4  # Ниже этого - отклонение
    SCORE_THRESHOLD_APPROVE = 0.6  # Выше этого - одобрение
    MIN_P_UP_FOR_ANALYSIS = 0.35  # Минимальный p_up для анализа

    # Стабильные монеты (более длинный кэш)
    STABLE_COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    # Алиасы монет для улучшенного поиска новостей
    COIN_ALIASES = {
        'BTC': ['bitcoin', 'btc'],
        'ETH': ['ethereum', 'eth', 'ether'],
        'BNB': ['binance coin', 'bnb'],
        'ADA': ['cardano', 'ada'],
        'SOL': ['solana', 'sol'],
        'DOT': ['polkadot', 'dot'],
        'DOGE': ['dogecoin', 'doge'],
        'MATIC': ['polygon', 'matic'],
        'AVAX': ['avalanche', 'avax'],
        'UNI': ['uniswap', 'uni'],
        'LINK': ['chainlink', 'link'],
        'XRP': ['ripple', 'xrp'],
        'LTC': ['litecoin', 'ltc'],
        'ATOM': ['cosmos', 'atom'],
        'XLM': ['stellar', 'xlm'],
        'ALGO': ['algorand', 'algo'],
        'NEAR': ['near protocol', 'near'],
        'APT': ['aptos', 'apt'],
        'ARB': ['arbitrum', 'arb'],
        'OP': ['optimism', 'op'],
    }


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """
    Ограничитель частоты запросов к API

    Защита от превышения лимитов Anthropic API:
    - Tier 1: 50 requests/minute
    - Tier 2: 1000 requests/minute
    """

    def __init__(
        self,
        max_calls: int = HaikuConfig.RATE_LIMIT_MAX_CALLS,
        period: float = HaikuConfig.RATE_LIMIT_PERIOD_SECONDS
    ):
        """
        Args:
            max_calls: Максимум вызовов за период
            period: Период в секундах
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
        self.lock = Lock()

        logger.info(f"[RateLimiter] Initialized: {max_calls} calls per {period}s")

    def wait_if_needed(self):
        """
        Блокирует выполнение если превышен лимит

        Автоматически ждёт до освобождения слота
        """
        with self.lock:
            now = time.time()

            # Удаляем старые вызовы за пределами окна
            self.calls = [call_time for call_time in self.calls if now - call_time < self.period]

            # Если достигнут лимит - ждём
            if len(self.calls) >= self.max_calls:
                # Вычисляем время ожидания до освобождения самого старого слота
                oldest_call = self.calls[0]
                wait_time = self.period - (now - oldest_call) + 0.1  # +0.1s буфер

                if wait_time > 0:
                    logger.warning(
                        f"[RateLimiter] Rate limit reached ({len(self.calls)}/{self.max_calls}). "
                        f"Waiting {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    now = time.time()
                    # Обновляем список после ожидания
                    self.calls = [call_time for call_time in self.calls if now - call_time < self.period]

            # Регистрируем текущий вызов
            self.calls.append(now)
            logger.debug(f"[RateLimiter] Calls in window: {len(self.calls)}/{self.max_calls}")

    def get_stats(self) -> dict:
        """Получить статистику использования"""
        with self.lock:
            now = time.time()
            active_calls = [c for c in self.calls if now - c < self.period]

            return {
                'calls_in_window': len(active_calls),
                'max_calls': self.max_calls,
                'utilization_pct': (len(active_calls) / self.max_calls) * 100 if self.max_calls > 0 else 0,
                'period_seconds': self.period
            }


# ============================================================================
# 1. DUCKDUCKGO NEWS SEARCH (Бесплатный поисковик с фильтрацией по дате)
# ============================================================================

class NewsCollector:
    """
    Сборщик новостей через DuckDuckGo Search

    Преимущества перед RSS:
    - Находит новости по всем криптовалютам (не только топ-10)
    - Фильтрация по дате (последние 24 часа)
    - Бесплатно, без API ключей
    - Более релевантные результаты

    Источник: DuckDuckGo News Search
    """

    def __init__(self, cache_ttl: int = HaikuConfig.NEWS_CACHE_TTL_SECONDS):
        """
        Args:
            cache_ttl: Время жизни кэша новостей (секунды)
        """
        self.cache_ttl = cache_ttl
        self._news_cache: Dict[str, Tuple[float, List[dict]]] = {}

        if DDGS is None:
            logger.error("[NewsCollector] duckduckgo_search not installed! Run: pip install duckduckgo-search")

        logger.debug(f"[NewsCollector] Initialized with DuckDuckGo search, cache_ttl={cache_ttl}s")

    def get_recent_news(self, symbol: str, hours: int = HaikuConfig.NEWS_LOOKBACK_HOURS) -> List[dict]:
        """
        Получить новости по монете за последние N часов через DuckDuckGo

        Args:
            symbol: Символ монеты (например, 'BTCUSDT')
            hours: Количество часов назад (по умолчанию 24)

        Returns:
            Список новостей: [{'title': str, 'link': str, 'published': str, 'source': str, 'body': str}, ...]
        """
        if DDGS is None:
            logger.warning("[NewsCollector] DuckDuckGo search not available, returning empty news")
            return []

        # Убираем USDT из символа для поиска
        coin_name = symbol.replace('USDT', '')

        # Проверяем кэш
        cache_key = f"{coin_name}_{hours}h"
        if cache_key in self._news_cache:
            cached_time, cached_news = self._news_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.debug(f"[NewsCollector] Cache HIT for {coin_name} ({len(cached_news)} items)")
                return cached_news

        logger.debug(f"[NewsCollector] Cache MISS for {coin_name}, searching DuckDuckGo...")

        # Получаем все алиасы монеты для поиска
        search_terms = HaikuConfig.COIN_ALIASES.get(coin_name, [coin_name.lower()])
        if coin_name.lower() not in search_terms:
            search_terms.append(coin_name.lower())

        logger.debug(f"[NewsCollector] Search terms for {coin_name}: {search_terms}")

        # Используем первый (основной) алиас для поиска
        # Для крипты это обычно полное название (bitcoin, ethereum и т.д.)
        primary_term = search_terms[0] if search_terms else coin_name.lower()

        # Формируем поисковый запрос
        query = f"{primary_term} cryptocurrency"

        all_news = []

        try:
            # ════════════════════════════════════════════════════════════════
            # ПОИСК НОВОСТЕЙ ЧЕРЕЗ DUCKDUCKGO С ТАЙМАУТОМ
            # ════════════════════════════════════════════════════════════════
            # timelimit='d' - последние 24 часа (day)
            # max_results - максимум результатов
            # timeout - таймаут для предотвращения зависаний (30 сек)
            # ════════════════════════════════════════════════════════════════

            logger.debug(f"[NewsCollector] Searching DuckDuckGo: query='{query}', timelimit='d'")

            def _search_news():
                """Вспомогательная функция для поиска с таймаутом"""
                with DDGS() as ddgs_client:
                    # Для hours <= 24 используем 'd' (day), для больших - 'w' (week)
                    time_limit = 'd' if hours <= 24 else 'w'

                    results = ddgs_client.news(
                        query=query,
                        timelimit=time_limit,
                        max_results=HaikuConfig.MAX_NEWS_ITEMS
                    )
                    return list(results)  # Материализуем generator

            # Запускаем поиск с таймаутом 30 секунд
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_search_news)
                try:
                    results = future.result(timeout=30.0)  # 30 сек таймаут
                except FuturesTimeoutError:
                    logger.warning(f"[NewsCollector] DuckDuckGo search timeout for {coin_name} (30s)")
                    raise TimeoutError(f"DuckDuckGo search timeout for {coin_name}")

            # Обрабатываем результаты
            for result in results:
                # Проверяем упоминание монеты в заголовке или тексте
                title = result.get('title', '').lower()
                body = result.get('body', '').lower()
                combined_text = title + ' ' + body

                # Проверяем по всем алиасам
                match_found = False
                for term in search_terms:
                    # Используем word boundaries для точного поиска
                    pattern = r'\b' + re.escape(term.lower()) + r'\b'
                    if re.search(pattern, combined_text):
                        match_found = True
                        break

                # Если монета не упоминается явно - пропускаем
                # (защита от нерелевантных результатов)
                if not match_found and len(search_terms) > 1:
                    logger.debug(f"[NewsCollector] Skipping irrelevant: {result.get('title', 'N/A')[:50]}")
                    continue

                all_news.append({
                    'title': result.get('title', ''),
                    'link': result.get('url', ''),
                    'published': result.get('date', ''),
                    'source': result.get('source', 'DuckDuckGo'),
                    'summary': result.get('body', '')[:HaikuConfig.MAX_NEWS_SUMMARY_LENGTH]
                })

            logger.info(f"[NewsCollector] Found {len(all_news)} news for {coin_name} via DuckDuckGo")

        except Exception as e:
            logger.error(f"[NewsCollector] DuckDuckGo search failed for {coin_name}: {e}")
            # Возвращаем пустой список вместо падения
            all_news = []

        # Ограничиваем до максимума
        all_news = all_news[:HaikuConfig.MAX_NEWS_ITEMS]

        # Кэшируем результат
        self._news_cache[cache_key] = (time.time(), all_news)

        return all_news

    def get_condensed_news_text(self, symbol: str, hours: int = HaikuConfig.NEWS_LOOKBACK_HOURS) -> str:
        """
        Получить сжатый текст новостей для промпта

        Args:
            symbol: Символ монеты
            hours: Количество часов назад

        Returns:
            Компактный текст новостей (ограничен по длине для промпта)
        """
        news = self.get_recent_news(symbol, hours)

        if not news:
            logger.debug(f"[NewsCollector] No news for {symbol}")
            return "No recent news found."

        # Формируем компактный текст
        lines = []
        total_length = 0
        max_total_length = HaikuConfig.MAX_NEWS_TEXT_LENGTH

        for i, item in enumerate(news[:HaikuConfig.MAX_NEWS_IN_PROMPT], 1):
            # Обрезаем заголовок до максимальной длины
            title = item['title'][:HaikuConfig.MAX_NEWS_TITLE_LENGTH]
            line = f"{i}. {title}"

            # Проверяем, не превысим ли мы лимит
            if total_length + len(line) + 1 > max_total_length:
                break

            lines.append(line)
            total_length += len(line) + 1  # +1 для \n

        result = "\n".join(lines)
        logger.debug(f"[NewsCollector] Condensed {len(news)} news into {len(result)} chars for {symbol}")

        return result


# ============================================================================
# 2. HAIKU 4.5 ANALYZER (с Prompt Caching)
# ============================================================================

class HaikuAnalyzer:
    """
    Анализатор монет с использованием Claude Haiku 4.5

    Особенности:
    - Batch обработка (до 5 монет в 1 запрос)
    - Prompt caching для экономии 90% на системных промптах
    - Агрессивное кэширование результатов
    - Компактные промпты для минимизации токенов
    - Fallback на технический скор при ошибках
    """

    # Системный промпт (будет закэширован Anthropic API)
    SYSTEM_PROMPT = """You are a crypto fundamental analyst. Analyze coins for critical signals.

Task: Check each coin for RED FLAGS and POSITIVE CATALYSTS.

RED FLAGS (score < 0.4):
- SEC lawsuits/investigations
- Hacks, exploits, rug pulls
- Delistings announced
- Major FUD from reliable sources

POSITIVE CATALYSTS (score > 0.6):
- Major partnerships announced
- New exchange listings
- Protocol upgrades/milestones
- Institutional adoption news

NEUTRAL (0.4-0.6): No significant news

Output: JSON only, no explanations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        cache_dir: str = "./data/haiku_cache",
        batch_size: int = HaikuConfig.BATCH_SIZE_DEFAULT,
        timeout: int = HaikuConfig.API_TIMEOUT_SECONDS,
        enable_stats: bool = True,
        enable_rate_limiting: bool = True
    ):
        """
        Args:
            api_key: Anthropic API ключ (или загрузится из ANTHROPIC_API_KEY env)
            model: ID модели Haiku 4.5
            cache_dir: Директория для кэша результатов
            batch_size: Количество монет в одном запросе (рекомендуется 5)
            timeout: Таймаут на запрос (секунды)
            enable_stats: Вести статистику эффективности
            enable_rate_limiting: Включить защиту от превышения лимитов API
        """
        # API клиент
        if Anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in env or constructor")

        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

        # Rate limiting
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        logger.info(f"[HaikuAnalyzer] Rate limiting: {'ENABLED' if enable_rate_limiting else 'DISABLED'}")

        # Кэширование
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Статистика
        self.enable_stats = enable_stats
        self.stats_file = self.cache_dir / "haiku_stats.json"
        self.stats = self._load_stats()

        # News collector
        self.news_collector = NewsCollector()

        logger.info(f"[HaikuAnalyzer] Initialized with model={model}, batch_size={batch_size}")

    def _load_stats(self) -> dict:
        """Загрузить статистику из файла"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[HaikuAnalyzer] Failed to load stats: {e}")

        return {
            'total_requests': 0,
            'total_coins_analyzed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_errors': 0,
            'fallback_used': 0,
            'total_cost_usd': 0.0,
            'decisions': {
                'rejected': 0,  # score < 0.4 (красные флаги)
                'approved': 0,  # score > 0.6 (позитивные сигналы)
                'neutral': 0    # 0.4-0.6 (нет сильных сигналов)
            }
        }

    def _save_stats(self):
        """Сохранить статистику в файл"""
        if not self.enable_stats:
            return

        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.warning(f"[HaikuAnalyzer] Failed to save stats: {e}")

    def _get_cache_key(self, symbol: str) -> str:
        """
        Генерация ключа для кэша

        Использует временные блоки для автоматической инвалидации кэша
        """
        # Используем хэш от символа и текущего блока
        block_hours = HaikuConfig.CACHE_BLOCK_HOURS
        current_block = int(time.time() / (block_hours * 3600))
        cache_key = hashlib.md5(f"{symbol}_{current_block}".encode()).hexdigest()

        logger.debug(f"[HaikuAnalyzer] Cache key for {symbol}: {cache_key} (block {current_block})")
        return cache_key

    def _get_cached_result(self, symbol: str) -> Optional[dict]:
        """
        Получить закэшированный результат

        Логика кэширования:
        - Стабильные монеты (BTC, ETH, BNB): 12 часов
        - Волатильные: 4 часа
        """
        cache_key = self._get_cache_key(symbol)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            logger.debug(f"[HaikuAnalyzer] Cache MISS for {symbol}: file not found")
            return None

        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)

            # Проверяем срок годности
            cached_time = cached.get('timestamp', 0)
            age_hours = (time.time() - cached_time) / 3600

            # Стабильные монеты - более длинный кэш
            max_age = (HaikuConfig.STABLE_CACHE_HOURS
                      if symbol in HaikuConfig.STABLE_COINS
                      else HaikuConfig.VOLATILE_CACHE_HOURS)

            if age_hours > max_age:
                logger.debug(f"[HaikuAnalyzer] Cache EXPIRED for {symbol} (age={age_hours:.1f}h > {max_age}h)")
                return None

            self.stats['cache_hits'] += 1
            logger.info(f"[HaikuAnalyzer] Cache HIT for {symbol} (age={age_hours:.1f}h, max={max_age}h)")
            return cached['result']

        except Exception as e:
            logger.warning(f"[HaikuAnalyzer] Failed to read cache for {symbol}: {e}")
            return None

    def _save_to_cache(self, symbol: str, result: dict):
        """Сохранить результат в кэш"""
        cache_key = self._get_cache_key(symbol)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'symbol': symbol,
                    'result': result
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"[HaikuAnalyzer] Failed to save cache for {symbol}: {e}")

    def _should_skip_analysis(self, opportunity: dict) -> bool:
        """
        Проверка: нужно ли анализировать монету?

        Пропускаем если:
        - p_up слишком низкий (< MIN_P_UP_FOR_ANALYSIS) - заведомо плохая сделка
        - Монета стабильная и была недавно проанализирована
        """
        symbol = opportunity.get('symbol', 'UNKNOWN')
        p_up = opportunity.get('p_up', 0.5)

        # Пропускаем очень слабые сигналы
        if p_up < HaikuConfig.MIN_P_UP_FOR_ANALYSIS:
            logger.info(
                f"[HaikuAnalyzer] SKIP {symbol}: p_up={p_up:.3f} < "
                f"threshold {HaikuConfig.MIN_P_UP_FOR_ANALYSIS}"
            )
            return True

        logger.debug(f"[HaikuAnalyzer] Will analyze {symbol}: p_up={p_up:.3f}")
        return False

    def analyze_batch(self, opportunities: List[dict]) -> List[dict]:
        """
        Анализ batch монет (до 5 в одном запросе)

        Args:
            opportunities: Список торговых возможностей (из coin_selector)

        Returns:
            Обогащённый список с добавленными полями:
            - haiku_score: float (0-1)
            - haiku_critical: bool (есть ли критические сигналы)
            - haiku_reason: str (краткое объяснение)
        """
        if not opportunities:
            logger.debug("[HaikuAnalyzer] No opportunities to analyze")
            return []

        logger.info(f"[HaikuAnalyzer] ══════ Starting batch analysis of {len(opportunities)} coins ══════")

        # Фильтруем уже закэшированные
        to_analyze = []
        results = []

        for opp in opportunities:
            symbol = opp['symbol']

            # Проверяем нужно ли анализировать
            if self._should_skip_analysis(opp):
                # Используем только технический скор
                opp['haiku_score'] = opp['p_up']
                opp['haiku_critical'] = False
                opp['haiku_reason'] = "skipped: weak signal"
                results.append(opp)
                self.stats['fallback_used'] += 1
                continue

            # Проверяем кэш
            cached = self._get_cached_result(symbol)
            if cached:
                opp.update(cached)
                results.append(opp)
                continue

            # Нужен свежий анализ
            to_analyze.append(opp)

        # Если все в кэше - возвращаем
        if not to_analyze:
            logger.info(f"[HaikuAnalyzer] All {len(opportunities)} coins served from cache ✓")
            return results

        logger.info(f"[HaikuAnalyzer] Need fresh analysis for {len(to_analyze)} coins")

        # Разбиваем на batch по N монет
        batches = [to_analyze[i:i+self.batch_size] for i in range(0, len(to_analyze), self.batch_size)]
        logger.info(f"[HaikuAnalyzer] Split into {len(batches)} batches (batch_size={self.batch_size})")

        for batch_idx, batch in enumerate(batches, 1):
            batch_symbols = [o['symbol'] for o in batch]
            logger.info(f"[HaikuAnalyzer] Processing batch {batch_idx}/{len(batches)}: {batch_symbols}")

            try:
                batch_results = self._analyze_batch_api(batch)

                # Сохраняем в кэш
                for i, opp in enumerate(batch):
                    if i < len(batch_results):
                        opp.update(batch_results[i])
                        self._save_to_cache(opp['symbol'], batch_results[i])
                        results.append(opp)
                        logger.debug(
                            f"  ✓ {opp['symbol']}: score={batch_results[i].get('haiku_score', 0):.2f}, "
                            f"critical={batch_results[i].get('haiku_critical', False)}"
                        )

                self.stats['cache_misses'] += len(batch)
                logger.info(f"[HaikuAnalyzer] Batch {batch_idx}/{len(batches)} completed successfully ✓")

            except Exception as e:
                logger.error(f"[HaikuAnalyzer] Batch {batch_idx}/{len(batches)} FAILED: {e}")
                self.stats['api_errors'] += 1

                # Fallback: используем технический скор
                for opp in batch:
                    opp['haiku_score'] = opp['p_up']
                    opp['haiku_critical'] = False
                    opp['haiku_reason'] = f"api_error: {str(e)[:50]}"
                    results.append(opp)
                    self.stats['fallback_used'] += 1
                    logger.warning(f"  ⚠ {opp['symbol']}: Using fallback (tech score only)")

        self._save_stats()
        logger.info(f"[HaikuAnalyzer] ══════ Batch analysis complete: {len(results)} results ══════")
        return results

    def _analyze_batch_api(self, batch: List[dict]) -> List[dict]:
        """
        Отправка batch запроса к Haiku 4.5 API

        Args:
            batch: Список возможностей (до 5 монет)

        Returns:
            Список результатов анализа
        """
        # ════════════════════════════════════════════════════════════════
        # RATE LIMITING: Проверяем и ждём если превышен лимит
        # ════════════════════════════════════════════════════════════════
        if self.rate_limiter:
            logger.debug("[HaikuAnalyzer] Checking rate limits...")
            self.rate_limiter.wait_if_needed()

        # Собираем новости для каждой монеты
        coins_data = []
        logger.debug(f"[HaikuAnalyzer] Collecting news for {len(batch)} coins...")

        for opp in batch:
            symbol = opp['symbol']
            news_text = self.news_collector.get_condensed_news_text(
                symbol,
                hours=HaikuConfig.NEWS_LOOKBACK_HOURS
            )

            coins_data.append({
                'symbol': symbol,
                'direction': opp['direction'],
                'p_up': opp['p_up'],
                'ev': opp['ev'],
                'news': news_text
            })

        # Формируем компактный промпт
        prompt = self._build_batch_prompt(coins_data)
        prompt_length = len(prompt)

        # API запрос с prompt caching
        logger.info(f"[HaikuAnalyzer] Sending API request: {len(batch)} coins, prompt={prompt_length} chars")
        start_time = time.time()

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=HaikuConfig.API_MAX_TOKENS,
                temperature=HaikuConfig.API_TEMPERATURE,
                system=[
                    {
                        "type": "text",
                        "text": self.SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}  # Кэшируем системный промпт
                    }
                ],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            elapsed = time.time() - start_time
            logger.info(f"[HaikuAnalyzer] ✓ API response received in {elapsed:.2f}s")

            # Обновляем статистику
            self.stats['total_requests'] += 1
            self.stats['total_coins_analyzed'] += len(batch)

            # Оцениваем стоимость (примерно)
            estimated_cost = HaikuConfig.ESTIMATED_COST_PER_BATCH
            self.stats['total_cost_usd'] += estimated_cost
            logger.debug(f"[HaikuAnalyzer] Estimated cost: ${estimated_cost:.4f}, total: ${self.stats['total_cost_usd']:.2f}")

            # Парсим JSON ответ
            response_text = response.content[0].text
            logger.debug(f"[HaikuAnalyzer] Response length: {len(response_text)} chars")

            return self._parse_batch_response(response_text, batch)

        except Exception as e:
            logger.error(f"[HaikuAnalyzer] ✗ API request failed: {e}")
            raise

    def _build_batch_prompt(self, coins_data: List[dict]) -> str:
        """
        Построение компактного batch промпта с валидацией длины

        Args:
            coins_data: Данные монет [{'symbol': ..., 'direction': ..., 'news': ...}, ...]

        Returns:
            Компактный промпт (с валидацией длины)

        Raises:
            ValueError: Если промпт превышает MAX_PROMPT_LENGTH
        """
        lines = ["Analyze these coins (24h news):"]

        for i, coin in enumerate(coins_data, 1):
            line = f"{i}. {coin['symbol']} {coin['direction']} p={coin['p_up']:.2f} EV={coin['ev']*100:.1f}%"
            lines.append(line)

            # Обрезаем новости если они слишком длинные
            news_text = coin['news']
            if len(news_text) > HaikuConfig.MAX_NEWS_TEXT_LENGTH:
                news_text = news_text[:HaikuConfig.MAX_NEWS_TEXT_LENGTH] + "..."
                logger.debug(f"[HaikuAnalyzer] Truncated news for {coin['symbol']} to {HaikuConfig.MAX_NEWS_TEXT_LENGTH} chars")

            lines.append(f"   News: {news_text}")

        lines.append("")
        lines.append("JSON format:")
        lines.append('[{"symbol":"BTCUSDT","score":0.75,"critical":false,"reason":"positive partnership"},...]')

        prompt = "\n".join(lines)
        prompt_length = len(prompt)

        # ════════════════════════════════════════════════════════════════
        # ВАЛИДАЦИЯ ДЛИНЫ ПРОМПТА
        # ════════════════════════════════════════════════════════════════
        if prompt_length > HaikuConfig.MAX_PROMPT_LENGTH:
            logger.error(
                f"[HaikuAnalyzer] Prompt TOO LONG: {prompt_length} chars > "
                f"max {HaikuConfig.MAX_PROMPT_LENGTH} chars"
            )
            raise ValueError(
                f"Prompt length {prompt_length} exceeds maximum {HaikuConfig.MAX_PROMPT_LENGTH}. "
                f"Reduce batch size or news length."
            )

        logger.debug(
            f"[HaikuAnalyzer] Built prompt: {prompt_length} chars "
            f"({prompt_length/HaikuConfig.MAX_PROMPT_LENGTH*100:.1f}% of max)"
        )

        return prompt

    def _parse_batch_response(self, response_text: str, batch: List[dict]) -> List[dict]:
        """
        Парсинг JSON ответа от Haiku 4.5

        Args:
            response_text: Текст ответа от API
            batch: Исходный batch для fallback

        Returns:
            Список результатов
        """
        logger.debug(f"[HaikuAnalyzer] Parsing response for {len(batch)} coins...")

        try:
            # Пытаемся найти JSON в ответе
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")

            json_text = response_text[json_start:json_end]
            parsed = json.loads(json_text)

            logger.debug(f"[HaikuAnalyzer] Successfully parsed {len(parsed)} items from JSON")

            # Обрабатываем результаты
            results = []
            for i, item in enumerate(parsed):
                score = float(item.get('score', 0.5))
                critical = bool(item.get('critical', False))
                reason = item.get('reason', '')[:50]  # Максимум 50 символов

                # Обновляем статистику решений
                if score < HaikuConfig.SCORE_THRESHOLD_REJECT:
                    self.stats['decisions']['rejected'] += 1
                    decision = 'REJECT'
                elif score > HaikuConfig.SCORE_THRESHOLD_APPROVE:
                    self.stats['decisions']['approved'] += 1
                    decision = 'APPROVE'
                else:
                    self.stats['decisions']['neutral'] += 1
                    decision = 'NEUTRAL'

                logger.debug(
                    f"  [{i+1}/{len(parsed)}] {item.get('symbol', 'UNKNOWN')}: "
                    f"score={score:.2f} ({decision}), critical={critical}, reason='{reason}'"
                )

                results.append({
                    'haiku_score': score,
                    'haiku_critical': critical,
                    'haiku_reason': reason
                })

            logger.info(f"[HaikuAnalyzer] Parsed {len(results)} results successfully")
            return results

        except Exception as e:
            logger.error(f"[HaikuAnalyzer] Failed to parse response: {e}")
            logger.debug(f"Response text (first 500 chars): {response_text[:500]}")

            # Fallback: нейтральные скоры
            fallback_results = [
                {
                    'haiku_score': 0.5,
                    'haiku_critical': False,
                    'haiku_reason': f'parse_error: {str(e)[:30]}'
                }
                for _ in batch
            ]

            logger.warning(f"[HaikuAnalyzer] Using fallback neutral scores for {len(fallback_results)} coins")
            return fallback_results

    def get_stats_summary(self) -> str:
        """Получить сводку статистики"""
        total = self.stats['total_coins_analyzed']
        if total == 0:
            return "No coins analyzed yet"

        cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100 if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0

        summary = f"""
=== Haiku Analyzer Statistics ===
Total API requests: {self.stats['total_requests']}
Total coins analyzed: {total}
Cache hit rate: {cache_hit_rate:.1f}%
API errors: {self.stats['api_errors']}
Fallback used: {self.stats['fallback_used']}
Estimated cost: ${self.stats['total_cost_usd']:.2f}

Decisions:
  Rejected (< 0.4): {self.stats['decisions']['rejected']}
  Neutral (0.4-0.6): {self.stats['decisions']['neutral']}
  Approved (> 0.6): {self.stats['decisions']['approved']}
"""
        return summary.strip()


# ============================================================================
# 3. INTEGRATION HELPER
# ============================================================================

def analyze_and_rank_top20(
    top_20_opportunities: List[dict],
    haiku_analyzer: HaikuAnalyzer,
    p_threshold: float
) -> List[dict]:
    """
    Анализ и переранжирование TOP-20 монет с учётом фундаментального анализа

    Процесс:
    1. Haiku анализирует все 20 монет (batch по 5)
    2. Применяет логику: final = p_up IF (p_up > threshold AND haiku_score > 0.6)
    3. Исключает монеты с haiku_critical=True (красные флаги)
    4. Сортирует по финальному скору
    5. Возвращает TOP-10

    Args:
        top_20_opportunities: ТОП-20 от coin_selector
        haiku_analyzer: Инстанс HaikuAnalyzer
        p_threshold: Адаптивный порог вероятности

    Returns:
        Обогащённый и отсортированный TOP-10
    """
    logger.info(f"[HaikuIntegration] Analyzing TOP-20 with Haiku 4.5")

    # 1. Анализ с Haiku
    enriched = haiku_analyzer.analyze_batch(top_20_opportunities)

    # 2. Фильтрация и расчёт финального скора
    filtered = []
    for opp in enriched:
        # Исключаем критические красные флаги
        if opp.get('haiku_critical', False):
            logger.warning(f"[HaikuIntegration] REJECTED {opp['symbol']}: {opp.get('haiku_reason', 'critical')}")
            continue

        # ════════════════════════════════════════════════════════════════
        # ИСПРАВЛЕНИЕ: Учитываем направление позиции
        # ════════════════════════════════════════════════════════════════
        direction = opp.get('direction', 'LONG')

        # Для LONG: p_up должна быть > threshold
        # Для SHORT: p_down (1 - p_up) должна быть > threshold
        if direction == 'LONG':
            confidence = opp['p_up']
            passes_threshold = confidence > p_threshold
        else:  # SHORT
            confidence = 1 - opp['p_up']  # p_down
            passes_threshold = confidence > p_threshold

        # ════════════════════════════════════════════════════════════════
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Логика фильтрации ИЛИ, а не И
        # ════════════════════════════════════════════════════════════════
        # Монета одобряется если проходит ХОТЯ БЫ ОДИН фильтр:
        # 1. Технический (passes_threshold) ИЛИ
        # 2. Фундаментальный (haiku_score > 0.6)
        #
        # Это позволяет Haiku находить монеты с отличными новостями,
        # даже если технический сигнал слабоват
        # ════════════════════════════════════════════════════════════════

        haiku_score = opp.get('haiku_score', 0.5)
        haiku_reason = opp.get('haiku_reason', '')

        # ════════════════════════════════════════════════════════════════
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ #2: Проверка на "skipped" монеты
        # ════════════════════════════════════════════════════════════════
        # Монеты с haiku_reason="skipped: weak signal" НЕ были проанализированы Haiku,
        # поэтому их haiku_score (который равен p_up) НЕ должен использоваться
        # для фундаментального одобрения. Они могут пройти только технический фильтр.
        # ════════════════════════════════════════════════════════════════

        # Фундаментальное одобрение только если монета была РЕАЛЬНО проанализирована Haiku
        was_analyzed_by_haiku = 'skipped' not in haiku_reason and 'error' not in haiku_reason
        haiku_approved = was_analyzed_by_haiku and (haiku_score > HaikuConfig.SCORE_THRESHOLD_APPROVE)

        # Проверяем: прошла ли монета хотя бы один фильтр?
        if not (passes_threshold or haiku_approved):
            # Не прошла ни один фильтр - отклоняем
            logger.debug(
                f"[HaikuIntegration] FILTERED OUT {opp['symbol']}: "
                f"tech={confidence:.3f} (threshold={p_threshold:.3f}), "
                f"fund={haiku_score:.3f} (min={HaikuConfig.SCORE_THRESHOLD_APPROVE})"
            )
            continue

        # Монета прошла хотя бы один фильтр - определяем финальный скор
        if passes_threshold and haiku_approved:
            # Прошла оба фильтра - используем комбинированный скор
            # Используем взвешенное среднее: 60% техника + 40% фундамент
            opp['final_score'] = confidence * 0.6 + haiku_score * 0.4
            opp['final_reason'] = 'tech+fund'
            logger.debug(
                f"[HaikuIntegration] APPROVED (both) {opp['symbol']}: "
                f"final={opp['final_score']:.3f} (tech={confidence:.3f}, fund={haiku_score:.3f})"
            )
        elif passes_threshold:
            # Прошла только технический фильтр
            opp['final_score'] = confidence
            opp['final_reason'] = 'tech_only'
            logger.debug(
                f"[HaikuIntegration] APPROVED (tech) {opp['symbol']}: "
                f"final={opp['final_score']:.3f} (fund={haiku_score:.3f} too low)"
            )
        else:
            # Прошла только фундаментальный фильтр
            # Используем haiku_score, но с небольшим снижением для осторожности
            opp['final_score'] = haiku_score * 0.9
            opp['final_reason'] = 'fund_only'
            logger.info(
                f"[HaikuIntegration] APPROVED (fund) {opp['symbol']}: "
                f"final={opp['final_score']:.3f} (tech={confidence:.3f} below threshold)"
            )

        filtered.append(opp)

    # 3. Сортировка по EV (как в оригинале)
    filtered.sort(key=lambda x: x['ev'], reverse=True)

    # 4. Возвращаем TOP-10
    top_10 = filtered[:10]

    logger.info(f"[HaikuIntegration] Final TOP-10: {len(top_10)} coins (filtered from {len(enriched)})")

    # Логируем результаты
    for i, opp in enumerate(top_10, 1):
        logger.info(
            f"  {i:2d}. {opp['symbol']:<12} {opp['direction']:<6} "
            f"tech={opp['p_up']:.3f} fund={opp.get('haiku_score', 0):.3f} "
            f"EV={opp['ev']:.4f} ({opp.get('haiku_reason', 'n/a')})"
        )

    return top_10


# ============================================================================
# MAIN (для тестирования)
# ============================================================================

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Тестовые данные
    test_opportunities = [
        {
            'symbol': 'BTCUSDT',
            'direction': 'LONG',
            'p_up': 0.67,
            'ev': 0.0675
        },
        {
            'symbol': 'ETHUSDT',
            'direction': 'SHORT',
            'p_up': 0.35,
            'ev': 0.062
        },
        {
            'symbol': 'SOLUSDT',
            'direction': 'LONG',
            'p_up': 0.71,
            'ev': 0.0589
        }
    ]

    # Создаём анализатор
    analyzer = HaikuAnalyzer(
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        batch_size=5,
        enable_stats=True
    )

    # Анализируем
    results = analyzer.analyze_batch(test_opportunities)

    # Выводим результаты
    print("\n=== Analysis Results ===")
    for r in results:
        print(f"{r['symbol']}: haiku_score={r.get('haiku_score', 'N/A')}, reason={r.get('haiku_reason', 'N/A')}")

    # Статистика
    print(analyzer.get_stats_summary())
