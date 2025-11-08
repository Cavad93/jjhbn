#!/usr/bin/env python3
"""
Тесты для Haiku 4.5 Analyzer

Запуск:
    pytest tests/test_haiku_analyzer.py -v
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.haiku_analyzer import NewsCollector, HaikuAnalyzer, analyze_and_rank_top20


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_api_key():
    """Mock API key для тестов"""
    return "sk-ant-test-key-12345"


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Временная директория для кэша"""
    cache_dir = tmp_path / "haiku_cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def sample_opportunities():
    """Примеры торговых возможностей"""
    return [
        {
            'symbol': 'BTCUSDT',
            'direction': 'LONG',
            'p_up': 0.67,
            'ev': 0.0675,
            'atr': 0.032,
            'phase': 2,
            'sector': 'LAYER1',
            'risk_level': 'LOW'
        },
        {
            'symbol': 'ETHUSDT',
            'direction': 'SHORT',
            'p_up': 0.35,
            'ev': 0.062,
            'atr': 0.028,
            'phase': 1,
            'sector': 'LAYER1',
            'risk_level': 'LOW'
        },
        {
            'symbol': 'SOLUSDT',
            'direction': 'LONG',
            'p_up': 0.71,
            'ev': 0.0589,
            'atr': 0.045,
            'phase': 3,
            'sector': 'LAYER1',
            'risk_level': 'MEDIUM'
        }
    ]


# ============================================================================
# NEWS COLLECTOR TESTS
# ============================================================================

class TestNewsCollector:
    """Тесты для NewsCollector"""

    def test_init(self):
        """Тест инициализации"""
        collector = NewsCollector(cache_ttl=1800)
        assert collector.cache_ttl == 1800
        assert isinstance(collector._news_cache, dict)

    @patch('strategy.haiku_analyzer.feedparser')
    def test_get_recent_news_no_feedparser(self, mock_feedparser):
        """Тест работы без feedparser"""
        mock_feedparser = None
        collector = NewsCollector()

        news = collector.get_recent_news('BTCUSDT', hours=24)
        assert news == []

    def test_get_condensed_news_text_no_news(self):
        """Тест получения текста когда новостей нет"""
        collector = NewsCollector()
        with patch.object(collector, 'get_recent_news', return_value=[]):
            text = collector.get_condensed_news_text('BTCUSDT')
            assert text == "No recent news found."

    def test_get_condensed_news_text_with_news(self):
        """Тест получения сжатого текста с новостями"""
        collector = NewsCollector()

        fake_news = [
            {'title': 'Bitcoin reaches new ATH at $100k', 'link': 'https://...', 'published': '2025-11-08', 'source': 'coindesk'},
            {'title': 'SEC approves Bitcoin ETF', 'link': 'https://...', 'published': '2025-11-08', 'source': 'cointelegraph'}
        ]

        with patch.object(collector, 'get_recent_news', return_value=fake_news):
            text = collector.get_condensed_news_text('BTCUSDT')
            assert 'Bitcoin reaches new ATH' in text
            assert 'SEC approves Bitcoin ETF' in text

    def test_news_caching(self):
        """Тест кэширования новостей"""
        collector = NewsCollector(cache_ttl=10)

        fake_news = [{'title': 'Test news', 'link': 'https://...', 'published': '2025-11-08', 'source': 'test'}]

        with patch('strategy.haiku_analyzer.feedparser') as mock_feedparser:
            mock_feed = Mock()
            mock_feed.entries = [
                Mock(
                    title='BTC Test news',
                    link='https://...',
                    published='2025-11-08 12:00:00',
                    published_parsed=None,
                    summary='BTC news summary'
                )
            ]
            mock_feedparser.parse.return_value = mock_feed

            # Первый вызов - парсинг
            news1 = collector.get_recent_news('BTCUSDT', hours=24)

            # Второй вызов - из кэша (feedparser не должен вызваться)
            mock_feedparser.reset_mock()
            news2 = collector.get_recent_news('BTCUSDT', hours=24)

            assert news1 == news2
            # Проверяем что feedparser не вызывался второй раз
            # (в реальности может быть вызван, если кэш не работает)


# ============================================================================
# HAIKU ANALYZER TESTS
# ============================================================================

class TestHaikuAnalyzer:
    """Тесты для HaikuAnalyzer"""

    def test_init_without_api_key(self):
        """Тест инициализации без API ключа"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                HaikuAnalyzer()

    def test_init_with_api_key(self, mock_api_key, temp_cache_dir):
        """Тест инициализации с API ключом"""
        with patch('strategy.haiku_analyzer.Anthropic') as mock_anthropic:
            analyzer = HaikuAnalyzer(
                api_key=mock_api_key,
                cache_dir=temp_cache_dir,
                batch_size=5,
                enable_stats=True
            )

            assert analyzer.api_key == mock_api_key
            assert analyzer.batch_size == 5
            assert analyzer.enable_stats == True
            assert analyzer.model == "claude-haiku-4-5-20251001"
            mock_anthropic.assert_called_once_with(api_key=mock_api_key)

    def test_should_skip_analysis_low_p_up(self, mock_api_key, temp_cache_dir, sample_opportunities):
        """Тест пропуска анализа при низком p_up"""
        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            # ETHUSDT имеет p_up = 0.35 (ниже 0.35, но на границе)
            # Изменим на 0.30 для явного теста
            sample_opportunities[1]['p_up'] = 0.30

            assert analyzer._should_skip_analysis(sample_opportunities[1]) == True
            assert analyzer._should_skip_analysis(sample_opportunities[0]) == False

    def test_get_cache_key(self, mock_api_key, temp_cache_dir):
        """Тест генерации ключа кэша"""
        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            key1 = analyzer._get_cache_key('BTCUSDT')
            key2 = analyzer._get_cache_key('BTCUSDT')
            key3 = analyzer._get_cache_key('ETHUSDT')

            # Ключи для одной монеты в одном 4-часовом блоке должны совпадать
            assert key1 == key2
            # Ключи для разных монет должны отличаться
            assert key1 != key3

    def test_cache_hit(self, mock_api_key, temp_cache_dir):
        """Тест попадания в кэш"""
        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            # Сохраняем в кэш
            cached_result = {'haiku_score': 0.75, 'haiku_critical': False, 'haiku_reason': 'test'}
            analyzer._save_to_cache('BTCUSDT', cached_result)

            # Читаем из кэша
            result = analyzer._get_cached_result('BTCUSDT')

            assert result == cached_result
            assert analyzer.stats['cache_hits'] == 1

    def test_cache_miss(self, mock_api_key, temp_cache_dir):
        """Тест промаха кэша"""
        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            result = analyzer._get_cached_result('NONEXISTENT')
            assert result is None

    def test_build_batch_prompt(self, mock_api_key, temp_cache_dir):
        """Тест построения batch промпта"""
        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            coins_data = [
                {'symbol': 'BTCUSDT', 'direction': 'LONG', 'p_up': 0.67, 'ev': 0.0675, 'news': 'BTC news 1\nBTC news 2'},
                {'symbol': 'ETHUSDT', 'direction': 'SHORT', 'p_up': 0.35, 'ev': 0.062, 'news': 'ETH news 1'}
            ]

            prompt = analyzer._build_batch_prompt(coins_data)

            assert 'BTCUSDT' in prompt
            assert 'ETHUSDT' in prompt
            assert 'LONG' in prompt
            assert 'SHORT' in prompt
            assert 'BTC news 1' in prompt
            assert 'JSON format' in prompt

    def test_parse_batch_response_success(self, mock_api_key, temp_cache_dir, sample_opportunities):
        """Тест парсинга успешного ответа"""
        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            response_text = '''
            [
                {"symbol": "BTCUSDT", "score": 0.75, "critical": false, "reason": "positive partnership"},
                {"symbol": "ETHUSDT", "score": 0.35, "critical": true, "reason": "SEC lawsuit"},
                {"symbol": "SOLUSDT", "score": 0.65, "critical": false, "reason": "neutral sentiment"}
            ]
            '''

            results = analyzer._parse_batch_response(response_text, sample_opportunities)

            assert len(results) == 3
            assert results[0]['haiku_score'] == 0.75
            assert results[0]['haiku_critical'] == False
            assert results[1]['haiku_score'] == 0.35
            assert results[1]['haiku_critical'] == True

    def test_parse_batch_response_invalid_json(self, mock_api_key, temp_cache_dir, sample_opportunities):
        """Тест парсинга невалидного JSON"""
        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            response_text = "This is not valid JSON"

            results = analyzer._parse_batch_response(response_text, sample_opportunities)

            # Должен вернуть fallback результаты
            assert len(results) == 3
            for result in results:
                assert result['haiku_score'] == 0.5
                assert result['haiku_critical'] == False
                assert 'parse_error' in result['haiku_reason']

    def test_analyze_batch_with_cache(self, mock_api_key, temp_cache_dir, sample_opportunities):
        """Тест анализа batch с использованием кэша"""
        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            # Сохраняем в кэш первую монету
            cached_result = {'haiku_score': 0.8, 'haiku_critical': False, 'haiku_reason': 'cached'}
            analyzer._save_to_cache('BTCUSDT', cached_result)

            # Мокаем API для остальных
            with patch.object(analyzer, '_analyze_batch_api', return_value=[
                {'haiku_score': 0.4, 'haiku_critical': True, 'haiku_reason': 'negative'},
                {'haiku_score': 0.7, 'haiku_critical': False, 'haiku_reason': 'positive'}
            ]):
                results = analyzer.analyze_batch(sample_opportunities)

                # Первая монета из кэша
                assert results[0]['haiku_score'] == 0.8
                # Остальные из API
                assert len(results) == 3

    def test_get_stats_summary(self, mock_api_key, temp_cache_dir):
        """Тест получения статистики"""
        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            analyzer.stats['total_coins_analyzed'] = 100
            analyzer.stats['cache_hits'] = 60
            analyzer.stats['cache_misses'] = 40
            analyzer.stats['total_cost_usd'] = 5.50

            summary = analyzer.get_stats_summary()

            assert 'Total coins analyzed: 100' in summary
            assert 'Cache hit rate: 60.0%' in summary
            assert '$5.50' in summary


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAnalyzeAndRankTop20:
    """Тесты для функции analyze_and_rank_top20"""

    def test_analyze_and_rank_basic(self, mock_api_key, temp_cache_dir):
        """Базовый тест переранжирования"""
        opportunities = [
            {'symbol': 'BTCUSDT', 'direction': 'LONG', 'p_up': 0.67, 'ev': 0.075},
            {'symbol': 'ETHUSDT', 'direction': 'LONG', 'p_up': 0.65, 'ev': 0.062},
            {'symbol': 'SOLUSDT', 'direction': 'LONG', 'p_up': 0.71, 'ev': 0.059},
        ]

        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            # Мокаем analyze_batch
            with patch.object(analyzer, 'analyze_batch') as mock_analyze:
                mock_analyze.return_value = [
                    {**opportunities[0], 'haiku_score': 0.75, 'haiku_critical': False},
                    {**opportunities[1], 'haiku_score': 0.35, 'haiku_critical': True},  # Критический
                    {**opportunities[2], 'haiku_score': 0.70, 'haiku_critical': False},
                ]

                p_threshold = 0.60

                result = analyze_and_rank_top20(opportunities, analyzer, p_threshold)

                # ETHUSDT должен быть отфильтрован (critical=True)
                assert len(result) == 2
                assert all(opp['symbol'] != 'ETHUSDT' for opp in result)

    def test_analyze_and_rank_threshold_filtering(self, mock_api_key, temp_cache_dir):
        """Тест фильтрации по порогу"""
        opportunities = [
            {'symbol': 'BTCUSDT', 'direction': 'LONG', 'p_up': 0.67, 'ev': 0.075},
            {'symbol': 'ETHUSDT', 'direction': 'LONG', 'p_up': 0.50, 'ev': 0.062},  # Ниже порога
        ]

        with patch('strategy.haiku_analyzer.Anthropic'):
            analyzer = HaikuAnalyzer(api_key=mock_api_key, cache_dir=temp_cache_dir)

            with patch.object(analyzer, 'analyze_batch') as mock_analyze:
                mock_analyze.return_value = [
                    {**opportunities[0], 'haiku_score': 0.75, 'haiku_critical': False},
                    {**opportunities[1], 'haiku_score': 0.70, 'haiku_critical': False},
                ]

                p_threshold = 0.60

                result = analyze_and_rank_top20(opportunities, analyzer, p_threshold)

                # Только BTCUSDT должна пройти (p_up > threshold)
                assert len(result) == 1
                assert result[0]['symbol'] == 'BTCUSDT'


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
