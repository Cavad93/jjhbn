"""
Конфигурация секторов для диверсификации портфеля

Группирует криптовалюты по категориям для ограничения концентрации в одном секторе.

Автор: Claude Code
Дата: 2025-11-06
"""

from typing import Dict, List

# Mapping монет по секторам
SECTOR_MAP: Dict[str, List[str]] = {
    'BTC': [
        'BTCUSDT'
    ],

    'ETH': [
        'ETHUSDT'
    ],

    'L1': [  # Layer 1 blockchains
        'SOLUSDT', 'AVAXUSDT', 'ADAUSDT', 'DOTUSDT', 'ATOMUSDT',
        'NEARUSDT', 'APTUSDT', 'SUIUSDT', 'ALGOUSDT', 'ICPUSDT',
        'FTMUSDT', 'INJUSDT', 'KASUSDT', 'ROSEUSDT', 'FLOWUSDT'
    ],

    'L2': [  # Layer 2 solutions
        'MATICUSDT', 'ARBUSDT', 'OPUSDT', 'IMXUSDT', 'STRKUSDT',
        'METISUSDT', 'MANTAUSDT', 'BLURUSDT'
    ],

    'DEFI': [  # Decentralized Finance
        'UNIUSDT', 'AAVEUSDT', 'CRVUSDT', 'MKRUSDT', 'SNXUSDT',
        'COMPUSDT', 'SUSHIUSDT', 'YFIUSDT', 'BALUSDT', 'LLDOUSDT',
        'CAKEUSDT', 'GMXUSDT', 'DYDXUSDT', 'PENDLEUSDT'
    ],

    'EXCHANGE': [  # Exchange tokens
        'BNBUSDT', 'FTUSDT', 'OKBUSDT', 'KCSUSDT', 'HTUSDT',
        'BTTCUSDT'
    ],

    'ORACLE': [  # Oracle solutions
        'LINKUSDT', 'BANDUSDT', 'APIUSDT', 'TRBUS DT'
    ],

    'GAMING': [  # Gaming & Metaverse
        'AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'GALAUSDT',
        'RONUSDT', 'IMXUSDT', 'MAGICUSDT', 'BLURUSDT', 'PIXELUSDT',
        'YGGUSDT', 'ILVUSDT', 'BEAMUSDT'
    ],

    'AI': [  # Artificial Intelligence
        'FETUSDT', 'AGIXUSDT', 'RENDERUSDT', 'OCEANUSDT', 'THETAUSDT',
        'NMRUSDT', 'GRTUSDT', 'ARKMUSDT', 'PHBUSDT'
    ],

    'STORAGE': [  # Storage & Data
        'FILUSDT', 'ARUSDT', 'STORJUSDT', 'SCUSDT', 'BLZUSDT'
    ],

    'PRIVACY': [  # Privacy coins
        'XMRUSDT', 'ZECUSDT', 'DASHUSDT', 'SCRTUSDT'
    ],

    'INTEROP': [  # Interoperability
        'DOTUSDT', 'ATOMUSDT', 'AXLUSDT', 'CELRUSDT', 'QNTUSDT'
    ],

    'NFT': [  # NFT platforms
        'FLOUSDT', 'THETAUSDT', 'CHZUSDT', 'APEUSDT', 'LOOKSUSDT'
    ],

    'MEME': [  # Meme coins (высокий риск)
        'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT',
        'WI FUSDT', 'BABYDOGEUSDT', '1000RATSUSDT', 'ORDIUSDT'
    ],

    'STABLECOIN_RELATED': [  # Стейблкоины и связанные проекты
        'USTCUSDT', 'LUNAUSDT', 'LUNA2USDT', 'FXSUSDT', 'MANUS DT'
    ],

    'PAYMENT': [  # Payment systems
        'XLMUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'TRXUSDT',
        'ADAUSDT', 'XTZUSDT'
    ],

    'INFRASTRUCTURE': [  # Infrastructure
        'GRTUSDT', 'HNTUSDT', 'ANKRUSDT', 'ORNNUSDT', 'LRCUSDT',
        'SKLUSDT', 'POWRUSDT', 'CTKUSDT'
    ],

    'SOCIAL': [  # Social & Content
        'MASKUSDT', 'GALAUSDT', 'SANTOSUSDT', 'POROUSDT', 'CITYUSDT',
        'BARUSDT', 'PSGUSDT', 'JUVUSDT', 'ACMUSDT', 'ATMUSDT',
        'ASRUSDT', 'OGUSDT'
    ],

    'GOVERNANCE': [  # Governance tokens
        'MKRUSDT', 'COMPUSDT', 'AAVEUSDT', 'UNIUSDT', 'SUSHIUSDT',
        'SNXUSDT', 'BALUSDT', 'CRVUSDT'
    ],

    'WEB3': [  # Web3 & Decentralized Internet
        'ENSUSDT', 'LDOUSDT', 'STXUSDT', 'ICXUSDT', 'ONTUSDT',
        'HBARUSDT', 'ZILUSDT', 'IOTXUSDT'
    ],

    'OTHER': [  # Прочие (низкоприоритетные)
        'VETUSDT', 'NEOUSDT', 'XEMUSDT', 'WAVESUSDT', 'ZILUSDT',
        'RVNUSDT', 'BTGUSDT', 'BATUSDT', 'ZRXUSDT', 'RVNUSDT'
    ]
}


# Инвертированный mapping (symbol -> sector)
SYMBOL_TO_SECTOR: Dict[str, str] = {}
for sector, symbols in SECTOR_MAP.items():
    for symbol in symbols:
        SYMBOL_TO_SECTOR[symbol] = sector


def get_coin_sector(symbol: str) -> str:
    """
    Определяет сектор монеты

    Args:
        symbol: Символ монеты (например, 'BTCUSDT')

    Returns:
        Название сектора или 'UNKNOWN'
    """
    return SYMBOL_TO_SECTOR.get(symbol, 'UNKNOWN')


def get_sector_coins(sector: str) -> List[str]:
    """
    Возвращает список монет в секторе

    Args:
        sector: Название сектора

    Returns:
        Список символов монет
    """
    return SECTOR_MAP.get(sector, [])


def get_all_sectors() -> List[str]:
    """
    Возвращает список всех секторов

    Returns:
        Список названий секторов
    """
    return list(SECTOR_MAP.keys())


def count_coins_per_sector(symbols: List[str]) -> Dict[str, int]:
    """
    Подсчитывает количество монет в каждом секторе

    Args:
        symbols: Список символов монет

    Returns:
        Словарь {сектор: количество}
    """
    sector_counts = {}
    for symbol in symbols:
        sector = get_coin_sector(symbol)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    return sector_counts


# Конфигурация ограничений
MAX_COINS_PER_SECTOR = {
    'BTC': 1,      # Только BTC
    'ETH': 1,      # Только ETH
    'L1': 3,       # Максимум 3 L1
    'L2': 2,       # Максимум 2 L2
    'DEFI': 3,     # Максимум 3 DeFi
    'MEME': 1,     # Максимум 1 мемкоин (высокий риск!)
    'AI': 2,       # Максимум 2 AI
    'GAMING': 2,   # Максимум 2 Gaming
    'DEFAULT': 3   # Для остальных секторов
}


def get_sector_limit(sector: str) -> int:
    """
    Возвращает лимит монет для сектора

    Args:
        sector: Название сектора

    Returns:
        Максимальное количество монет
    """
    return MAX_COINS_PER_SECTOR.get(sector, MAX_COINS_PER_SECTOR['DEFAULT'])


if __name__ == "__main__":
    print("="*80)
    print("КОНФИГУРАЦИЯ СЕКТОРОВ")
    print("="*80)

    print(f"\nВсего секторов: {len(SECTOR_MAP)}")
    print(f"Всего монет в mapping: {len(SYMBOL_TO_SECTOR)}")

    print("\nСектора и количество монет:")
    for sector, coins in sorted(SECTOR_MAP.items(), key=lambda x: len(x[1]), reverse=True):
        limit = get_sector_limit(sector)
        print(f"  {sector:20s}: {len(coins):3d} монет (лимит: {limit})")

    print("\nПримеры:")
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'LINKUSDT']
    for symbol in test_symbols:
        sector = get_coin_sector(symbol)
        limit = get_sector_limit(sector)
        print(f"  {symbol:15s} → {sector:15s} (лимит: {limit})")
