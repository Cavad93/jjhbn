"""
Data collection module for Binance historical data

Модуль для сбора и обновления исторических данных с Binance

Две системы:
1. collect_binance_data - старая система с Parquet файлами (все 300+ пар)
2. smart_collector + db_manager - новая система с PostgreSQL (только топ-10)

Рекомендуется использовать новую систему (smart_collector).
"""

# Старая система (Parquet файлы)
from .collect_binance_data import (
    get_all_tradeable_pairs,
    download_ohlcv_data,
    collect_all_data,
    update_all_data,
    create_data_index
)

# Новая система (PostgreSQL + топ-10)
from .db_manager import DatabaseManager
from .smart_collector import SmartCollector

__all__ = [
    # Старая система
    'get_all_tradeable_pairs',
    'download_ohlcv_data',
    'collect_all_data',
    'update_all_data',
    'create_data_index',
    # Новая система
    'DatabaseManager',
    'SmartCollector',
]
