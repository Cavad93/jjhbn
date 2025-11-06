"""
Blacklist монет для Binance Futures Bot

Исключает проблемные, скам и высокорисковые монеты из торговли.

Автор: Claude Code
Дата: 2025-11-06
"""

from typing import List, Set

# Blacklist монет (не торговать!)
BLACKLIST_SYMBOLS: Set[str] = {
    # Delisted or suspended
    'USTCUSDT',      # TerraUSD Classic (крах)
    'LUNAUSDT',      # Terra Luna Classic (крах)
    'FTUSDT',        # FTX Token (банкротство)
    'CELSIUSUSDT',   # Celsius (банкротство)
    'VOYAGERUSDT',   # Voyager (банкротство)

    # Known scams or high risk
    'TITANUSDT',     # Iron Finance Titan (rug pull)
    'SQUIDUSDT',     # Squid Game Token (скам)
    'SAFEMOONUSDT',  # SafeMoon (rug pull подозрение)
    'SHIBAINU2USDT', # Фейковые форки

    # Extremely low liquidity or delisted
    'BTGUSDT',       # Bitcoin Gold (низкая ликвидность)
    'BCDUSDT',       # Bitcoin Diamond
    'BSVUSDT',       # Bitcoin SV (проблемы с листингами)

    # Problematic forks
    'ETHWUSDT',      # ETH PoW fork (низкий интерес)
    'ETCUSDT',       # Ethereum Classic (низкая активность)

    # Highly manipulated
    'IQUSDT',        # Everipedia IQ (подозрение в манипуляциях)
    'KEYUSDT',       # SelfKey (низкая ликвидность)

    # Old/deprecated projects
    'QTUMUSDT',      # Qtum (низкая активность)
    'LSKUSDT',       # Lisk (низкая активность)
    'STRATUSDT',     # Stratis (низкая активность)
    'ARKUSDT',       # Ark (низкая активность)

    # Privacy coins with regulatory issues
    'XMRUSDT',       # Monero (проблемы с регуляцией)
    'ZECUSDT',       # Zcash (проблемы с листингами)
    'DASHUSDT',      # Dash (низкая ликвидность)

    # Stablecoins (не торговать на futures)
    'USDCUSDT',
    'BUSDUSDT',
    'TUSDUSDT',
    'DAIUSDT',
    'USDPUSDT',
    'PAXUSDT',
    'GUSDUSDT'
}


# Категории рискованных монет (торговать с осторожностью)
HIGH_RISK_CATEGORIES = {
    'NEW_LISTING': {
        # Новые листинги (< 30 дней) - добавляются автоматически
        'description': 'Монеты листингом менее 30 дней назад',
        'risk_level': 'HIGH'
    },

    'LOW_MARKET_CAP': {
        # Market cap < $50M
        'description': 'Market cap < $50M',
        'risk_level': 'HIGH'
    },

    'MEME_COINS': {
        # Мемкоины (высокая волатильность)
        'symbols': [
            'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT',
            'BONKUSDT', 'WIFUSDT', 'BABYDOGEUSDT'
        ],
        'description': 'Мемкоины с высокой волатильностью',
        'risk_level': 'MEDIUM',
        'max_allocation': 0.005  # Максимум 0.5% капитала
    },

    'FAN_TOKENS': {
        # Fan tokens (низкая ликвидность вне событий)
        'symbols': [
            'SANTOSUSDT', 'POROUSDT', 'CITYUSDT', 'BARUSDT',
            'PSGUSDT', 'JUVUSDT', 'ACMUSDT', 'ATMUSDT',
            'ASRUSDT', 'OGUSDT'
        ],
        'description': 'Fan tokens с низкой ликвидностью',
        'risk_level': 'MEDIUM'
    },

    'NFT_RELATED': {
        # NFT проекты (высокая волатильность)
        'symbols': [
            'AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'ENJUSDT',
            'FLOWUSDT', 'APEUS DT', 'LOOKSUSDT'
        ],
        'description': 'NFT проекты с высокой волатильностью',
        'risk_level': 'MEDIUM'
    }
}


def is_blacklisted(symbol: str) -> bool:
    """
    Проверяет находится ли монета в blacklist

    Args:
        symbol: Символ монеты (например, 'BTCUSDT')

    Returns:
        True если в blacklist, иначе False
    """
    return symbol in BLACKLIST_SYMBOLS


def is_high_risk(symbol: str, category: str = None) -> bool:
    """
    Проверяет является ли монета высокорисковой

    Args:
        symbol: Символ монеты
        category: Конкретная категория риска (опционально)

    Returns:
        True если высокорисковая, иначе False
    """
    if category:
        cat_data = HIGH_RISK_CATEGORIES.get(category, {})
        symbols = cat_data.get('symbols', [])
        return symbol in symbols

    # Проверяем все категории
    for cat_name, cat_data in HIGH_RISK_CATEGORIES.items():
        if cat_name in ['NEW_LISTING', 'LOW_MARKET_CAP']:
            continue  # Эти проверяются динамически

        symbols = cat_data.get('symbols', [])
        if symbol in symbols:
            return True

    return False


def get_risk_level(symbol: str) -> str:
    """
    Определяет уровень риска монеты

    Args:
        symbol: Символ монеты

    Returns:
        'BLACKLIST', 'HIGH', 'MEDIUM', или 'LOW'
    """
    if is_blacklisted(symbol):
        return 'BLACKLIST'

    for cat_name, cat_data in HIGH_RISK_CATEGORIES.items():
        if cat_name in ['NEW_LISTING', 'LOW_MARKET_CAP']:
            continue

        symbols = cat_data.get('symbols', [])
        if symbol in symbols:
            return cat_data.get('risk_level', 'MEDIUM')

    return 'LOW'


def get_max_allocation(symbol: str, default_allocation: float = 0.025) -> float:
    """
    Возвращает максимальную аллокацию капитала для монеты

    Args:
        symbol: Символ монеты
        default_allocation: Аллокация по умолчанию (2.5%)

    Returns:
        Максимальная доля капитала (0.005 - 0.025)
    """
    for cat_name, cat_data in HIGH_RISK_CATEGORIES.items():
        symbols = cat_data.get('symbols', [])
        if symbol in symbols:
            return cat_data.get('max_allocation', default_allocation)

    return default_allocation


def get_blacklist_reason(symbol: str) -> str:
    """
    Возвращает причину нахождения монеты в blacklist

    Args:
        symbol: Символ монеты

    Returns:
        Строка с причиной или пустая строка
    """
    reasons = {
        'USTCUSDT': 'TerraUSD Classic крах (May 2022)',
        'LUNAUSDT': 'Terra Luna Classic крах (May 2022)',
        'FTUSDT': 'FTX банкротство (Nov 2022)',
        'CELSIUSUSDT': 'Celsius банкротство (2022)',
        'TITANUSDT': 'Iron Finance rug pull',
        'SQUIDUSDT': 'Squid Game Token скам',
        'BTGUSDT': 'Bitcoin Gold низкая ликвидность',
        'BSVUSDT': 'Bitcoin SV проблемы с листингами',
        'XMRUSDT': 'Monero регуляторные проблемы',
        'USDCUSDT': 'Стейблкоин (не торговать на futures)',
        'BUSDUSDT': 'Стейблкоин (не торговать на futures)',
        'TUSDUSDT': 'Стейблкоин (не торговать на futures)',
        'DAIUSDT': 'Стейблкоин (не торговать на futures)'
    }

    return reasons.get(symbol, 'Неизвестная причина')


def filter_blacklisted(symbols: List[str]) -> List[str]:
    """
    Фильтрует список монет, удаляя blacklisted

    Args:
        symbols: Список символов монет

    Returns:
        Отфильтрованный список
    """
    return [s for s in symbols if not is_blacklisted(s)]


def get_blacklist_summary() -> dict:
    """
    Возвращает сводку по blacklist

    Returns:
        Словарь со статистикой
    """
    return {
        'total_blacklisted': len(BLACKLIST_SYMBOLS),
        'blacklist': list(BLACKLIST_SYMBOLS),
        'high_risk_categories': len(HIGH_RISK_CATEGORIES),
        'categories': list(HIGH_RISK_CATEGORIES.keys())
    }


if __name__ == "__main__":
    print("="*80)
    print("BLACKLIST МОНЕТ")
    print("="*80)

    summary = get_blacklist_summary()
    print(f"\nВсего в blacklist: {summary['total_blacklisted']}")

    print("\nBlacklisted монеты:")
    for symbol in sorted(BLACKLIST_SYMBOLS):
        reason = get_blacklist_reason(symbol)
        print(f"  ❌ {symbol:20s} - {reason}")

    print(f"\n\nКатегорий высокого риска: {summary['high_risk_categories']}")
    for cat_name, cat_data in HIGH_RISK_CATEGORIES.items():
        print(f"\n{cat_name}:")
        print(f"  Описание: {cat_data['description']}")
        print(f"  Уровень риска: {cat_data['risk_level']}")
        if 'symbols' in cat_data:
            print(f"  Монет: {len(cat_data['symbols'])}")
            if 'max_allocation' in cat_data:
                print(f"  Макс. аллокация: {cat_data['max_allocation']*100}%")

    print("\n\nПримеры проверки риска:")
    test_symbols = ['BTCUSDT', 'DOGEUSDT', 'USTCUSDT', 'SANTOSUSDT']
    for symbol in test_symbols:
        risk = get_risk_level(symbol)
        allocation = get_max_allocation(symbol) * 100
        blacklisted = is_blacklisted(symbol)

        status = "❌ BLACKLIST" if blacklisted else f"✅ {risk} RISK"
        print(f"  {symbol:20s}: {status:20s} (макс. {allocation:.1f}%)")
