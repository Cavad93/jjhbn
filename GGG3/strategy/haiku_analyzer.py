#!/usr/bin/env python3
"""
Haiku 4.5 Fundamental Analysis Module
Анализ фундаментальных факторов для TOP-20 монет

Возможности:
- Интеграция с Claude Haiku 4.5 через Anthropic API
- Batch обработка (5 монет в 1 запрос для экономии)
- Агрессивное кэширование результатов (4-12 часов)
- RSS парсинг новостей (бесплатные источники)
- Prompt caching для экономии до 90% на input токенах
- Статистика эффективности решений

Оптимизации для бюджета $6-12/месяц:
1. Batch processing: -35% расходов
2. Smart caching: -35% запросов
3. Compact prompts: -40% токенов
4. RSS instead of search: -10% токенов
5. Prompt caching (Anthropic): -90% на системных промптах

Автор: Claude Code
Дата: 2025-11-08
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import hashlib

# RSS парсинг
try:
    import feedparser
except ImportError:
    feedparser = None
    logging.warning("feedparser не установлен. RSS парсинг отключён. pip install feedparser")

# Anthropic API
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
    logging.warning("anthropic не установлен. pip install anthropic")

# Для веб-запросов (fallback если нет RSS)
import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)


# ============================================================================
# 1. RSS NEWS PARSER (Бесплатные источники)
# ============================================================================

class NewsCollector:
    """
    Сборщик новостей из бесплатных RSS источников

    Источники:
    - CoinDesk RSS
    - CoinTelegraph RSS
    - Binance News
    - Reddit r/cryptocurrency (через RSS)
    """

    RSS_FEEDS = {
        'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'cointelegraph': 'https://cointelegraph.com/rss',
        'binance': 'https://www.binance.com/en/feed/index/rss',
        'reddit_crypto': 'https://www.reddit.com/r/cryptocurrency/.rss',
    }

    def __init__(self, cache_ttl: int = 1800):
        """
        Args:
            cache_ttl: Время жизни кэша новостей (секунды), по умолчанию 30 минут
        """
        self.cache_ttl = cache_ttl
        self._news_cache: Dict[str, Tuple[float, List[dict]]] = {}

    def get_recent_news(self, symbol: str, hours: int = 24) -> List[dict]:
        """
        Получить новости по монете за последние N часов

        Args:
            symbol: Символ монеты (например, 'BTCUSDT')
            hours: Количество часов назад

        Returns:
            Список новостей: [{'title': str, 'link': str, 'published': str, 'source': str}, ...]
        """
        # Убираем USDT из символа для поиска
        coin_name = symbol.replace('USDT', '')

        # Проверяем кэш
        cache_key = f"{coin_name}_{hours}h"
        if cache_key in self._news_cache:
            cached_time, cached_news = self._news_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.debug(f"[NewsCollector] Using cached news for {coin_name}")
                return cached_news

        # Собираем новости из всех источников
        all_news = []
        cutoff_time = datetime.now() - timedelta(hours=hours)

        if feedparser is None:
            logger.warning("[NewsCollector] feedparser not installed, returning empty news")
            return []

        for source_name, feed_url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries[:50]:  # Берём последние 50 записей
                    # Проверяем наличие упоминания монеты
                    title = entry.get('title', '').lower()
                    summary = entry.get('summary', '').lower()

                    if coin_name.lower() not in title and coin_name.lower() not in summary:
                        continue

                    # Парсим дату публикации
                    published = entry.get('published_parsed')
                    if published:
                        pub_date = datetime(*published[:6])
                        if pub_date < cutoff_time:
                            continue

                    all_news.append({
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': source_name,
                        'summary': entry.get('summary', '')[:200]  # Первые 200 символов
                    })

            except Exception as e:
                logger.warning(f"[NewsCollector] Failed to parse {source_name}: {e}")
                continue

        # Сортируем по дате (новые первыми)
        all_news.sort(key=lambda x: x['published'], reverse=True)

        # Ограничиваем до 10 самых свежих
        all_news = all_news[:10]

        # Кэшируем результат
        self._news_cache[cache_key] = (time.time(), all_news)

        logger.info(f"[NewsCollector] Found {len(all_news)} news for {coin_name}")
        return all_news

    def get_condensed_news_text(self, symbol: str, hours: int = 24) -> str:
        """
        Получить сжатый текст новостей для промпта

        Args:
            symbol: Символ монеты
            hours: Количество часов назад

        Returns:
            Компактный текст новостей (максимум 500 символов)
        """
        news = self.get_recent_news(symbol, hours)

        if not news:
            return "No recent news found."

        # Формируем компактный текст
        lines = []
        for i, item in enumerate(news[:5], 1):  # Максимум 5 новостей
            line = f"{i}. {item['title'][:80]}"  # Обрезаем до 80 символов
            lines.append(line)

        return "\n".join(lines)


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
        batch_size: int = 5,
        timeout: int = 60,
        enable_stats: bool = True
    ):
        """
        Args:
            api_key: Anthropic API ключ (или загрузится из ANTHROPIC_API_KEY env)
            model: ID модели Haiku 4.5
            cache_dir: Директория для кэша результатов
            batch_size: Количество монет в одном запросе (рекомендуется 5)
            timeout: Таймаут на запрос (секунды)
            enable_stats: Вести статистику эффективности
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
        """Генерация ключа для кэша"""
        # Используем хэш от символа и текущего 4-часового блока
        block_4h = int(time.time() / (4 * 3600))
        return hashlib.md5(f"{symbol}_{block_4h}".encode()).hexdigest()

    def _get_cached_result(self, symbol: str) -> Optional[dict]:
        """
        Получить закэшированный результат

        Логика кэширования:
        - Стабильные монеты (BTC, ETH, BNB): 12 часов
        - Волатильные: 4 часа (по условию задачи)
        """
        cache_key = self._get_cache_key(symbol)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)

            # Проверяем срок годности
            cached_time = cached.get('timestamp', 0)
            age_hours = (time.time() - cached_time) / 3600

            # Стабильные монеты - кэш на 12 часов
            stable_coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            max_age = 12 if symbol in stable_coins else 4

            if age_hours > max_age:
                logger.debug(f"[HaikuAnalyzer] Cache expired for {symbol} (age={age_hours:.1f}h)")
                return None

            self.stats['cache_hits'] += 1
            logger.info(f"[HaikuAnalyzer] Cache HIT for {symbol} (age={age_hours:.1f}h)")
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
        - p_up слишком низкий (< 0.35) - заведомо плохая сделка
        - Монета стабильная и была недавно проанализирована
        """
        # Пропускаем очень слабые сигналы
        if opportunity.get('p_up', 0.5) < 0.35:
            logger.debug(f"[HaikuAnalyzer] Skipping {opportunity['symbol']}: p_up too low")
            return True

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
            return []

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
            logger.info(f"[HaikuAnalyzer] All {len(opportunities)} coins from cache")
            return results

        # Разбиваем на batch по N монет
        batches = [to_analyze[i:i+self.batch_size] for i in range(0, len(to_analyze), self.batch_size)]

        for batch in batches:
            try:
                batch_results = self._analyze_batch_api(batch)

                # Сохраняем в кэш
                for i, opp in enumerate(batch):
                    if i < len(batch_results):
                        opp.update(batch_results[i])
                        self._save_to_cache(opp['symbol'], batch_results[i])
                        results.append(opp)

                self.stats['cache_misses'] += len(batch)

            except Exception as e:
                logger.error(f"[HaikuAnalyzer] Batch API failed: {e}")
                self.stats['api_errors'] += 1

                # Fallback: используем технический скор
                for opp in batch:
                    opp['haiku_score'] = opp['p_up']
                    opp['haiku_critical'] = False
                    opp['haiku_reason'] = f"api_error: {str(e)[:50]}"
                    results.append(opp)
                    self.stats['fallback_used'] += 1

        self._save_stats()
        return results

    def _analyze_batch_api(self, batch: List[dict]) -> List[dict]:
        """
        Отправка batch запроса к Haiku 4.5 API

        Args:
            batch: Список возможностей (до 5 монет)

        Returns:
            Список результатов анализа
        """
        # Собираем новости для каждой монеты
        coins_data = []
        for opp in batch:
            symbol = opp['symbol']
            news_text = self.news_collector.get_condensed_news_text(symbol, hours=24)

            coins_data.append({
                'symbol': symbol,
                'direction': opp['direction'],
                'p_up': opp['p_up'],
                'ev': opp['ev'],
                'news': news_text
            })

        # Формируем компактный промпт
        prompt = self._build_batch_prompt(coins_data)

        # API запрос с prompt caching
        logger.info(f"[HaikuAnalyzer] Sending batch request for {len(batch)} coins")
        start_time = time.time()

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.0,  # Детерминированный вывод
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
            logger.info(f"[HaikuAnalyzer] API response received in {elapsed:.2f}s")

            # Обновляем статистику
            self.stats['total_requests'] += 1
            self.stats['total_coins_analyzed'] += len(batch)

            # Оцениваем стоимость (примерно)
            # Input: ~3.5K tokens × $1/M = $0.0035
            # Output: ~2K tokens × $5/M = $0.01
            estimated_cost = 0.0135  # $0.0135 за batch из 5 монет
            self.stats['total_cost_usd'] += estimated_cost

            # Парсим JSON ответ
            response_text = response.content[0].text
            return self._parse_batch_response(response_text, batch)

        except Exception as e:
            logger.error(f"[HaikuAnalyzer] API request failed: {e}")
            raise

    def _build_batch_prompt(self, coins_data: List[dict]) -> str:
        """
        Построение компактного batch промпта

        Args:
            coins_data: Данные монет [{'symbol': ..., 'direction': ..., 'news': ...}, ...]

        Returns:
            Компактный промпт
        """
        lines = ["Analyze these coins (24h news):"]

        for i, coin in enumerate(coins_data, 1):
            line = f"{i}. {coin['symbol']} {coin['direction']} p={coin['p_up']:.2f} EV={coin['ev']*100:.1f}%"
            lines.append(line)
            lines.append(f"   News: {coin['news']}")

        lines.append("")
        lines.append("JSON format:")
        lines.append('[{"symbol":"BTCUSDT","score":0.75,"critical":false,"reason":"positive partnership"},...]')

        return "\n".join(lines)

    def _parse_batch_response(self, response_text: str, batch: List[dict]) -> List[dict]:
        """
        Парсинг JSON ответа от Haiku 4.5

        Args:
            response_text: Текст ответа от API
            batch: Исходный batch для fallback

        Returns:
            Список результатов
        """
        try:
            # Пытаемся найти JSON в ответе
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")

            json_text = response_text[json_start:json_end]
            parsed = json.loads(json_text)

            # Обрабатываем результаты
            results = []
            for item in parsed:
                score = float(item.get('score', 0.5))
                critical = bool(item.get('critical', False))
                reason = item.get('reason', '')[:50]  # Максимум 50 символов

                # Обновляем статистику решений
                if score < 0.4:
                    self.stats['decisions']['rejected'] += 1
                elif score > 0.6:
                    self.stats['decisions']['approved'] += 1
                else:
                    self.stats['decisions']['neutral'] += 1

                results.append({
                    'haiku_score': score,
                    'haiku_critical': critical,
                    'haiku_reason': reason
                })

            return results

        except Exception as e:
            logger.error(f"[HaikuAnalyzer] Failed to parse response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")

            # Fallback: нейтральные скоры
            return [
                {
                    'haiku_score': 0.5,
                    'haiku_critical': False,
                    'haiku_reason': f'parse_error: {str(e)[:30]}'
                }
                for _ in batch
            ]

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

        # Применяем логику: final = p_up IF (p_up > threshold AND haiku_score > 0.6)
        if opp['p_up'] > p_threshold and opp.get('haiku_score', 0.5) > 0.6:
            # Монета прошла оба фильтра
            opp['final_score'] = opp['p_up']
            opp['final_reason'] = 'tech+fund'
        elif opp['p_up'] > p_threshold:
            # Только технический скор
            opp['final_score'] = opp['p_up']
            opp['final_reason'] = 'tech_only'
        else:
            # Не прошла порог
            continue

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
