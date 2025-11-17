#!/usr/bin/env python3
"""
Paper Trading Bot - Main Entry Point

ПОЛНАЯ копия реального бота, но с PaperExchange вместо Binance.
Использует РЕАЛЬНЫЕ цены Binance mainnet для симуляции.

Функционал:
- Загрузка ML моделей (XGB, RF, ARF, NN) + META нейросеть
- EV Filter для отбора топ-10 монет из ~400
- Торговый цикл: 4h для управления позициями, 5m для проверки TP/SL
- LONG/SHORT bidirectional trading
- Snapshot pattern для temporal consistency
- Online learning при закрытии позиций
- Логирование всех операций
- Сохранение истории в CSV

Usage:
    python3 paper_bot_main.py --initial-capital 1000 --max-positions 10
"""

import argparse
import time
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import pickle
import json
import csv
import logging
from pathlib import Path

# Добавляем родительскую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импортируем наш PaperExchange
from paper_trading.paper_exchange import PaperExchange, get_binance_price, get_binance_kline

# Импорт обученных моделей экспертов
try:
    from models.experts import XGBoostExpert, RandomForestExpert, AdaptiveRFExpert, NeuralNetworkExpert
    HAVE_EXPERTS = True
except ImportError as e:
    print(f"WARNING: Failed to import experts: {e}")
    XGBoostExpert = RandomForestExpert = AdaptiveRFExpert = NeuralNetworkExpert = None
    HAVE_EXPERTS = False

# Импорт feature builder
try:
    from features.builder import BinanceFeatureBuilder
    HAVE_FEATURE_BUILDER = True
except ImportError as e:
    print(f"WARNING: Failed to import feature builder: {e}")
    BinanceFeatureBuilder = None
    HAVE_FEATURE_BUILDER = False


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Конфигурация бота"""

    # Trading parameters
    MAX_POSITIONS = 10
    INITIAL_CAPITAL = 1000.0
    RISK_PER_TRADE = 0.01  # 1% риска на сделку (Kelly)

    # ATR parameters (для TP/SL)
    ATR_PERIOD = 10
    ATR_TIMEFRAME = '4h'
    TP_MULTIPLIER = 1.2  # TP = entry ± 1.2 * ATR
    SL_MULTIPLIER = 0.8  # SL = entry ∓ 0.8 * ATR

    # Threshold parameters
    BASE_THRESHOLD = 0.51
    MIN_THRESHOLD = 0.45
    MAX_THRESHOLD = 0.70

    # EV Filter
    MIN_EV = 0.02  # Минимальный EV для входа
    TOP_N_COINS = 10  # Топ-10 по EV

    # Timings
    PORTFOLIO_UPDATE_INTERVAL = 4 * 3600  # 4 часа в секундах
    CHECK_ORDERS_INTERVAL = 5 * 60  # 5 минут в секундах

    # Blacklist (мемкоины, новые листинги, low liquidity)
    BLACKLIST = [
        'BTCDOMUSDT', 'DEFIUSDT', 'USDCUSDT', 'BUSDUSDT', 'TUSDUSDT',
        # Добавьте сюда проблемные монеты
    ]

    # Paths
    MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'  # GGG3/models/saved/
    LOGS_DIR = Path(__file__).parent / 'logs'
    DATA_DIR = Path(__file__).parent / 'data'

    # Files
    STATE_FILE = DATA_DIR / 'paper_bot_state.json'
    TRADES_CSV = DATA_DIR / 'paper_trades.csv'
    PORTFOLIO_CSV = DATA_DIR / 'paper_portfolio.csv'

    @classmethod
    def create_dirs(cls):
        """Создает необходимые директории"""
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level=logging.INFO):
    """Настройка логирования"""
    Config.create_dirs()

    log_file = Config.LOGS_DIR / f'paper_bot_{datetime.now().strftime("%Y%m%d")}.log'

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# MODELS LOADING
# ============================================================================

class ModelsManager:
    """Менеджер ML моделей"""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models = {}

    def load_models(self) -> Dict:
        """
        Загружает все модели экспертов + META

        Returns:
            dict: {'xgb': model, 'rf': model, 'arf': model, 'nn': model, 'meta': model}
        """
        logger.info("=" * 80)
        logger.info("LOADING ML MODELS")
        logger.info("=" * 80)

        if not HAVE_EXPERTS:
            logger.error("Expert models not available - missing imports!")
            return self.models

        # XGBoost Expert
        self.models['xgb'] = self._load_expert('xgb_expert.pkl', 'XGBoost', XGBoostExpert)

        # Random Forest Expert
        self.models['rf'] = self._load_expert('rf_expert.pkl', 'RandomForest', RandomForestExpert)

        # Adaptive Random Forest Expert
        self.models['arf'] = self._load_expert('arf_expert.pkl', 'AdaptiveRF', AdaptiveRFExpert)

        # Neural Network Expert
        self.models['nn'] = self._load_expert('nn_expert.pkl', 'NeuralNet', NeuralNetworkExpert)

        # META model
        meta_path = self.models_dir / 'meta_model.pkl'
        if meta_path.exists():
            try:
                # Загружаем META через train_meta модуль
                sys.path.insert(0, str(self.models_dir.parent))
                from models.train_meta import SimpleMetaNetwork

                self.models['meta'] = SimpleMetaNetwork.load(str(meta_path))
                logger.info(f"✓ [META] Loaded from {meta_path.name}")
                logger.info(f"  Trained: {self.models['meta'].is_trained}, "
                           f"Accuracy: {self.models['meta'].train_accuracy:.3f}")
            except Exception as e:
                logger.error(f"✗ [META] Failed to load: {e}")
                self.models['meta'] = None
        else:
            logger.warning("[META] Not found - using average of experts")
            self.models['meta'] = None

        loaded = sum(1 for v in self.models.values() if v is not None)
        logger.info(f"Models loaded: {loaded}/{len(self.models)}")
        logger.info("=" * 80)

        return self.models

    def _load_expert(self, filename: str, model_name: str, expert_class):
        """Загружает эксперта используя метод .load()"""
        path = self.models_dir / filename

        if not path.exists():
            logger.warning(f"[{model_name}] Model file not found: {path}")
            return None

        try:
            # Используем метод load класса эксперта
            expert = expert_class.load(str(path))
            logger.info(f"✓ [{model_name}] Loaded from {path.name}")
            logger.info(f"  Trained: {expert.is_trained}, Samples: {expert.train_samples}")
            return expert
        except Exception as e:
            logger.error(f"✗ [{model_name}] Failed to load: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# FEATURES CALCULATION
# ============================================================================

class FeaturesCalculator:
    """Расчет фич для экспертов"""

    def __init__(self):
        """Инициализация с feature builder"""
        self.feature_builder = BinanceFeatureBuilder() if HAVE_FEATURE_BUILDER else None

    def calculate_features(self, symbol: str) -> Optional[np.ndarray]:
        """
        Рассчитывает 68D вектор фич для символа

        Args:
            symbol: торговая пара (например, 'BTCUSDT')

        Returns:
            np.ndarray: вектор фич 68D или None при ошибке
        """
        if not HAVE_FEATURE_BUILDER or self.feature_builder is None:
            logger.error("BinanceFeatureBuilder not available!")
            return self._calculate_features_fallback(symbol)

        try:
            # Получаем данные для всех 4 таймфреймов
            logger.debug(f"Fetching multi-timeframe data for {symbol}...")

            df_5m = self._get_klines_df(symbol, '5m', limit=200)
            df_15m = self._get_klines_df(symbol, '15m', limit=200)
            df_30m = self._get_klines_df(symbol, '30m', limit=200)
            df_4h = self._get_klines_df(symbol, '4h', limit=100)

            # Проверяем что данные получены
            if any(df is None for df in [df_5m, df_15m, df_30m, df_4h]):
                logger.error(f"Failed to fetch data for {symbol}")
                return None

            # Рассчитываем фичи через BinanceFeatureBuilder
            features = self.feature_builder.build_extended_features(
                df_5m=df_5m,
                df_15m=df_15m,
                df_30m=df_30m,
                df_4h=df_4h,
                current_time=pd.Timestamp.now()
            )

            return features

        except Exception as e:
            logger.error(f"Failed to calculate features for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_klines_df(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Получает OHLCV данные для символа

        Args:
            symbol: торговая пара
            timeframe: '5m', '15m', '30m', '4h'
            limit: количество свечей

        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume
        """
        try:
            # Для упрощения используем только последнюю свечу
            # В production нужно получать через WebSocket или batch API
            klines = []

            # Получаем limit свечей (упрощенно - в production лучше batch request)
            for _ in range(min(limit, 20)):  # Ограничиваем для demo
                kline = get_binance_kline(symbol, timeframe, 1)
                if kline:
                    klines.append(kline)
                time.sleep(0.05)  # Rate limit

            if not klines:
                return None

            # Создаем DataFrame
            df = pd.DataFrame(klines)

            # Проверяем наличие всех необходимых колонок
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in klines data")
                return None

            # Конвертируем timestamp в datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            return df[required_cols]

        except Exception as e:
            logger.error(f"Failed to get klines for {symbol} {timeframe}: {e}")
            return None

    def _calculate_features_fallback(self, symbol: str) -> Optional[np.ndarray]:
        """Fallback расчет фич (упрощенный)"""
        try:
            # Простой вектор из zeros (baseline)
            logger.warning(f"Using fallback features for {symbol}")
            return np.zeros(68, dtype=np.float64)
        except Exception as e:
            logger.error(f"Fallback features failed: {e}")
            return None


# ============================================================================
# EV FILTER
# ============================================================================

class EVFilter:
    """Expected Value фильтр для отбора монет"""

    def __init__(self, models: Dict, config: Config):
        self.models = models
        self.config = config

    def calculate_ev(self, p_up: float) -> Tuple[float, float]:
        """
        Рассчитывает EV для LONG и SHORT

        Args:
            p_up: вероятность роста (0-1)

        Returns:
            (ev_long, ev_short)
        """
        # R/R = 1.5:1
        # EV_long = p_up * 1.5 - (1 - p_up) * 1.0 = 2.5 * p_up - 1.0
        # EV_short = (1 - p_up) * 1.5 - p_up * 1.0 = 1.5 - 2.5 * p_up

        ev_long = 2.5 * p_up - 1.0
        ev_short = 1.5 - 2.5 * p_up

        return ev_long, ev_short

    def get_top_coins(self, universe: List[str]) -> List[Dict]:
        """
        Отбирает топ-N монет по EV

        Args:
            universe: список всех монет для проверки

        Returns:
            List[Dict]: топ монет с данными {'symbol', 'ev', 'direction', 'p_up', 'predictions'}
        """
        logger.info(f"[EV Filter] Scanning {len(universe)} coins...")

        results = []
        features_calc = FeaturesCalculator()

        for symbol in universe:
            if symbol in self.config.BLACKLIST:
                continue

            try:
                # Рассчитываем фичи (68D)
                features = features_calc.calculate_features(symbol)
                if features is None:
                    continue

                # Получаем предсказания от всех 4 экспертов
                predictions = self._get_expert_predictions(features)
                if predictions is None:
                    continue

                # Финальное предсказание
                if self.models.get('meta') is not None:
                    # Используем META для комбинирования экспертов
                    expert_probs = np.array([
                        predictions['p_xgb'],
                        predictions['p_rf'],
                        predictions['p_arf'],
                        predictions['p_nn']
                    ]).reshape(1, -1)

                    p_up_array = self.models['meta'].predict(expert_probs)
                    p_up = float(p_up_array[0])
                    predictions['p_meta'] = p_up
                else:
                    # SHADOW mode: используем среднее экспертов
                    p_up = predictions['p_mean']
                    predictions['p_meta'] = p_up

                # Рассчитываем EV
                ev_long, ev_short = self.calculate_ev(p_up)

                # Выбираем лучшее направление
                if ev_long > ev_short and ev_long > self.config.MIN_EV:
                    results.append({
                        'symbol': symbol,
                        'ev': ev_long,
                        'direction': 'LONG',
                        'p_up': p_up,
                        'predictions': predictions
                    })
                elif ev_short > ev_long and ev_short > self.config.MIN_EV:
                    results.append({
                        'symbol': symbol,
                        'ev': ev_short,
                        'direction': 'SHORT',
                        'p_up': p_up,
                        'predictions': predictions
                    })

            except Exception as e:
                logger.debug(f"[EV Filter] Error processing {symbol}: {e}")
                continue

        # Сортируем по EV (убыванию)
        results.sort(key=lambda x: x['ev'], reverse=True)

        # Берем топ-N
        top_coins = results[:self.config.TOP_N_COINS]

        logger.info(f"[EV Filter] Selected {len(top_coins)} coins:")
        for coin in top_coins:
            preds = coin['predictions']
            logger.info(f"  {coin['symbol']:12s} {coin['direction']:5s} EV={coin['ev']:+.4f} "
                       f"p_up={coin['p_up']:.3f} [XGB:{preds['p_xgb']:.2f} RF:{preds['p_rf']:.2f} "
                       f"NN:{preds['p_nn']:.2f}]")

        return top_coins

    def _get_expert_predictions(self, features: np.ndarray) -> Optional[Dict]:
        """
        Получает предсказания от всех экспертов

        Args:
            features: 68D вектор фич

        Returns:
            Dict с предсказаниями {'p_xgb', 'p_rf', 'p_arf', 'p_nn', 'p_mean'}
        """
        try:
            predictions = {}
            reg_ctx = {'phase': 0}  # Можно добавить определение фазы

            # XGBoost
            if self.models.get('xgb') is not None:
                p_xgb, meta = self.models['xgb'].proba_up(features, reg_ctx)
                predictions['p_xgb'] = p_xgb
            else:
                predictions['p_xgb'] = 0.5

            # Random Forest
            if self.models.get('rf') is not None:
                p_rf, meta = self.models['rf'].proba_up(features, reg_ctx)
                predictions['p_rf'] = p_rf
            else:
                predictions['p_rf'] = 0.5

            # Adaptive Random Forest
            if self.models.get('arf') is not None:
                p_arf, meta = self.models['arf'].proba_up(features, reg_ctx)
                predictions['p_arf'] = p_arf
            else:
                predictions['p_arf'] = 0.5

            # Neural Network
            if self.models.get('nn') is not None:
                p_nn, meta = self.models['nn'].proba_up(features, reg_ctx)
                predictions['p_nn'] = p_nn
            else:
                predictions['p_nn'] = 0.5

            # Среднее предсказание
            probs = [predictions['p_xgb'], predictions['p_rf'], predictions['p_arf'], predictions['p_nn']]
            predictions['p_mean'] = np.mean(probs)

            return predictions

        except Exception as e:
            logger.error(f"Failed to get expert predictions: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# ATR CALCULATOR
# ============================================================================

class ATRCalculator:
    """Расчет ATR для определения TP/SL"""

    @staticmethod
    def calculate_atr(symbol: str, period: int = 10, timeframe: str = '4h') -> float:
        """
        Рассчитывает ATR (Average True Range)

        Args:
            symbol: торговая пара
            period: период для ATR
            timeframe: таймфрейм

        Returns:
            float: значение ATR
        """
        try:
            # Получаем исторические свечи
            klines = []
            for i in range(period + 1):
                kline = get_binance_kline(symbol, timeframe, 1)
                klines.append(kline)
                time.sleep(0.1)

            # Рассчитываем True Range
            trs = []
            for i in range(1, len(klines)):
                high = klines[i]['high']
                low = klines[i]['low']
                prev_close = klines[i-1]['close']

                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                trs.append(tr)

            # ATR = среднее TR
            atr = np.mean(trs) if trs else klines[-1]['close'] * 0.02  # Fallback 2%

            return atr

        except Exception as e:
            logger.error(f"Failed to calculate ATR for {symbol}: {e}")
            # Fallback: 2% от цены
            try:
                price = get_binance_price(symbol)
                return price * 0.02
            except:
                return 100.0  # Hard fallback


# ============================================================================
# PAPER TRADING BOT
# ============================================================================

class PaperTradingBot:
    """Главный класс paper trading бота"""

    def __init__(self, initial_capital: float, max_positions: int):
        self.config = Config()
        self.config.INITIAL_CAPITAL = initial_capital
        self.config.MAX_POSITIONS = max_positions

        # Paper Exchange - загружаем сохраненное состояние или создаем новую
        Config.create_dirs()  # Убеждаемся что директории существуют

        state_file = self.config.DATA_DIR / 'paper_exchange_state.json'

        if state_file.exists():
            try:
                logger.info(f"Loading saved exchange state from {state_file}...")
                self.exchange = PaperExchange.load_state(str(state_file))
                logger.info(f"✓ Exchange state loaded successfully!")
                logger.info(f"  Balance: {self.exchange.get_balance():.2f} USDT")
                logger.info(f"  Equity: {self.exchange.get_equity():.2f} USDT")
                logger.info(f"  Open positions: {len(self.exchange.get_open_positions())}")
            except Exception as e:
                logger.warning(f"Failed to load saved state: {e}")
                # 👉 Авто-восстановление из capital_history.json (берём total_equity)
                self.exchange = self._recover_exchange_from_capital_history(prefer_total_equity=True)
                if self.exchange is None:
                    logger.info(f"Creating new exchange with initial capital {initial_capital} USDT...")
                    self.exchange = PaperExchange(initial_capital=initial_capital)
        else:
            # Нет файла состояния — пробуем восстановить из capital_history.json
            self.exchange = self._recover_exchange_from_capital_history(prefer_total_equity=True)
            if self.exchange is None:
                logger.info(f"No saved state found. Creating new exchange with initial capital {initial_capital} USDT...")
                self.exchange = PaperExchange(initial_capital=initial_capital)

        # Models
        self.models_manager = ModelsManager(self.config.MODELS_DIR)
        self.models = self.models_manager.load_models()

        # Components
        self.ev_filter = EVFilter(self.models, self.config)
        self.features_calc = FeaturesCalculator()
        self.atr_calc = ATRCalculator()

        # State
        self.active_symbols = []  # Список топ-10 символов
        self.last_portfolio_update = 0
        self.last_orders_check = 0
        self.total_trades = 0

        # Statistics
        self.stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'start_time': time.time()
        }

        logger.info("=" * 80)
        logger.info("PAPER TRADING BOT INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Initial Capital: {initial_capital} USDT")
        logger.info(f"Max Positions: {max_positions}")
        logger.info(f"Models Loaded: {sum(1 for v in self.models.values() if v is not None)}")
        logger.info("=" * 80)

    def get_binance_universe(self) -> List[str]:
        """
        Получает список всех USDT торговых пар с Binance

        Returns:
            List[str]: список символов типа ['BTCUSDT', 'ETHUSDT', ...]
        """
        # Для demo используем топовые монеты
        # В production нужно получать через API /api/v3/exchangeInfo

        universe = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
            'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'LTCUSDT', 'LINKUSDT',
            'MATICUSDT', 'SOLUSDT', 'AVAXUSDT', 'ATOMUSDT', 'ETCUSDT',
            'XLMUSDT', 'VETUSDT', 'FILUSDT', 'TRXUSDT', 'EOSUSDT',
        ]

        return [s for s in universe if s not in self.config.BLACKLIST]

    def update_portfolio(self):
        """Обновление портфеля: отбор топ-10 монет и управление позициями"""
        logger.info("\n" + "=" * 80)
        logger.info("PORTFOLIO UPDATE")
        logger.info("=" * 80)

        # Получаем universe
        universe = self.get_binance_universe()

        # EV Filter: отбираем топ-10
        top_coins = self.ev_filter.get_top_coins(universe)

        # Обновляем список активных символов
        self.active_symbols = [c['symbol'] for c in top_coins]

        # Проверяем текущие позиции
        open_positions = self.exchange.get_open_positions()

        logger.info(f"\nCurrent positions: {len(open_positions)}/{self.config.MAX_POSITIONS}")

        # Закрываем позиции по символам вне топ-10
        for position in open_positions:
            if position['symbol'] not in self.active_symbols:
                logger.info(f"Closing position {position['symbol']} - not in top-10")
                self.exchange.close_position(position['id'], reason='rebalance')

        # Открываем новые позиции
        open_positions = self.exchange.get_open_positions()
        symbols_with_positions = {p['symbol'] for p in open_positions}

        for coin_data in top_coins:
            if len(open_positions) >= self.config.MAX_POSITIONS:
                break

            symbol = coin_data['symbol']

            if symbol in symbols_with_positions:
                continue

            # Открываем новую позицию
            try:
                self.open_position(
                    symbol=symbol,
                    direction=coin_data['direction'],
                    p_up=coin_data['p_up'],
                    ev=coin_data['ev']
                )
                open_positions = self.exchange.get_open_positions()
                symbols_with_positions.add(symbol)
            except Exception as e:
                logger.error(f"Failed to open position {symbol}: {e}")

        self.last_portfolio_update = time.time()
        logger.info("=" * 80 + "\n")

    def open_position(self, symbol: str, direction: str, p_up: float, ev: float):
        """
        Открывает новую позицию

        Args:
            symbol: торговая пара
            direction: 'LONG' или 'SHORT'
            p_up: вероятность роста
            ev: Expected Value
        """
        logger.info(f"\n[OPENING POSITION] {symbol} {direction}")

        try:
            # Получаем текущую цену
            current_price = get_binance_price(symbol)

            # Рассчитываем ATR
            atr = self.atr_calc.calculate_atr(symbol, self.config.ATR_PERIOD, self.config.ATR_TIMEFRAME)

            # Рассчитываем TP/SL
            if direction == 'LONG':
                tp_price = current_price + self.config.TP_MULTIPLIER * atr
                sl_price = current_price - self.config.SL_MULTIPLIER * atr
            else:  # SHORT
                tp_price = current_price - self.config.TP_MULTIPLIER * atr
                sl_price = current_price + self.config.SL_MULTIPLIER * atr

            # Kelly criterion: размер позиции
            equity = self.exchange.get_equity()
            risk_amount = equity * self.config.RISK_PER_TRADE
            position_size = risk_amount / (self.config.SL_MULTIPLIER * atr)

            # Ограничиваем размер позиции
            max_position_value = equity * 0.15  # Макс 15% от капитала
            position_size = min(position_size, max_position_value / current_price)

            logger.info(f"  Price: {current_price:.2f}")
            logger.info(f"  ATR: {atr:.2f}")
            logger.info(f"  TP: {tp_price:.2f}")
            logger.info(f"  SL: {sl_price:.2f}")
            logger.info(f"  Size: {position_size:.6f}")
            logger.info(f"  EV: {ev:+.4f}")

            # Рассчитываем фичи для snapshot
            features = self.features_calc.calculate_features(symbol)

            # Создаем snapshot (temporal consistency)
            entry_snapshot = {
                'features': features.tolist() if features is not None else [],
                'predictions': {'p_up': p_up},
                'ev': ev,
                'direction': direction,
                'timestamp': time.time()
            }

            # Открываем позицию
            position = self.exchange.open_position(
                symbol=symbol,
                side=direction,
                amount=position_size,
                tp_price=tp_price,
                sl_price=sl_price,
                metadata={
                    'entry_snapshot': entry_snapshot,
                    'atr_value': atr  # ✅ Сохраняем ATR для trailing stop
                }
            )

            logger.info(f"  ✓ Position opened: {position['id']}")

            # Сохраняем в CSV
            self.log_trade_to_csv({
                'timestamp': datetime.now().isoformat(),
                'action': 'OPEN',
                'symbol': symbol,
                'direction': direction,
                'price': current_price,
                'size': position_size,
                'tp': tp_price,
                'sl': sl_price,
                'ev': ev,
                'position_id': position['id']
            })

        except Exception as e:
            logger.error(f"Failed to open position {symbol}: {e}")
            raise

    def check_orders(self):
        """Проверяет и исполняет TP/SL ордера"""
        executed = self.exchange.check_orders()

        if executed:
            logger.info(f"\n[ORDERS CHECK] {len(executed)} order(s) executed")

            for exec_info in executed:
                position = exec_info['position']
                order = exec_info['order']

                logger.info(f"  {position['symbol']} {order['type']} @ {exec_info['execution_price']:.2f}")
                logger.info(f"  PnL: {position['pnl_after_commission']:+.2f} USDT ({position['pnl_percent']:+.2f}%)")

                # Статистика
                self.stats['trades'] += 1
                self.stats['total_pnl'] += position['pnl_after_commission']

                if position['pnl_after_commission'] > 0:
                    self.stats['wins'] += 1
                else:
                    self.stats['losses'] += 1

                # Online learning (здесь нужно реализовать обучение на snapshot)
                self.update_models_online(position)

                # Логируем в CSV
                self.log_trade_to_csv({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'CLOSE',
                    'symbol': position['symbol'],
                    'direction': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': position['close_price'],
                    'pnl': position['pnl_after_commission'],
                    'pnl_percent': position['pnl_percent'],
                    'reason': position['close_reason'],
                    'position_id': position['id']
                })

        self.last_orders_check = time.time()

    def update_models_online(self, position: Dict):
        """
        Обновляет модели на основе закрытой позиции (online learning)

        Args:
            position: данные закрытой позиции с entry_snapshot
        """
        try:
            # Получаем snapshot
            snapshot = position.get('metadata', {}).get('entry_snapshot', {})
            if not snapshot:
                logger.debug("No entry snapshot for online learning")
                return

            # Определяем outcome: 1 если TP, 0 если SL
            outcome = 1 if position['close_reason'] == 'tp' else 0

            # Получаем фичи из snapshot (temporal consistency!)
            features = np.array(snapshot.get('features', []))

            if len(features) == 0 or len(features) != 68:
                logger.debug(f"Invalid features in snapshot: {len(features)}")
                return

            # Обучаем все эксперты online
            logger.info(f"[Online Learning] {position['symbol']} outcome={outcome} ({position['close_reason']})")

            features_2d = features.reshape(1, -1)
            y = np.array([outcome])

            # XGBoost - добавляем новые деревья
            if self.models.get('xgb') is not None and HAVE_EXPERTS:
                try:
                    self.models['xgb'].partial_fit(features_2d, y, n_new_trees=5)
                    logger.debug("  ✓ XGBoost updated")
                except Exception as e:
                    logger.error(f"  ✗ XGBoost update failed: {e}")

            # Random Forest - добавляем новые деревья
            if self.models.get('rf') is not None and HAVE_EXPERTS:
                try:
                    self.models['rf'].partial_fit(features_2d, y, n_new_trees=5)
                    logger.debug("  ✓ RandomForest updated")
                except Exception as e:
                    logger.error(f"  ✗ RandomForest update failed: {e}")

            # Adaptive RF - online learning
            if self.models.get('arf') is not None and HAVE_EXPERTS:
                try:
                    self.models['arf'].partial_fit(features_2d, y)
                    logger.debug("  ✓ AdaptiveRF updated")
                except Exception as e:
                    logger.error(f"  ✗ AdaptiveRF update failed: {e}")

            # Neural Network - fine-tuning
            if self.models.get('nn') is not None and HAVE_EXPERTS:
                try:
                    self.models['nn'].partial_fit(features_2d, y, n_epochs=5)
                    logger.debug("  ✓ NeuralNet updated")
                except Exception as e:
                    logger.error(f"  ✗ NeuralNet update failed: {e}")

            # TODO: Online learning для META нейросети

            logger.info(f"[Online Learning] Models updated successfully")

        except Exception as e:
            logger.error(f"Failed to update models online: {e}")
            import traceback
            traceback.print_exc()

    def log_trade_to_csv(self, trade_data: Dict):
        """Сохраняет данные сделки в CSV"""
        Config.create_dirs()

        file_exists = self.config.TRADES_CSV.exists()

        with open(self.config.TRADES_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trade_data.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(trade_data)

    def print_status(self):
        """Выводит текущий статус бота"""
        equity = self.exchange.get_equity()
        balance = self.exchange.get_balance()
        open_positions = self.exchange.get_open_positions()

        stats = self.exchange.get_statistics()

        logger.info("\n" + "=" * 80)
        logger.info("BOT STATUS")
        logger.info("=" * 80)
        logger.info(f"Balance: {balance:.2f} USDT")
        logger.info(f"Equity: {equity:.2f} USDT")
        logger.info(f"Open Positions: {len(open_positions)}/{self.config.MAX_POSITIONS}")
        logger.info(f"Total Trades: {stats['total_trades']}")
        logger.info(f"Win Rate: {stats['winrate']:.2f}%")
        logger.info(f"Total PnL: {stats['total_pnl']:+.2f} USDT")
        logger.info(f"ROI: {stats['roi']:+.2f}%")
        logger.info("=" * 80 + "\n")

    def save_state(self):
        """Сохраняет состояние бота"""
        state = {
            'timestamp': time.time(),
            'stats': self.stats,
            'active_symbols': self.active_symbols
        }

        Config.create_dirs()

        with open(self.config.STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

        # Сохраняем состояние exchange
        self.exchange.save_state(str(self.config.DATA_DIR / 'paper_exchange_state.json'))

    def _recover_exchange_from_capital_history(self, prefer_total_equity: bool = True) -> Optional[PaperExchange]:
        """Пытается восстановить PaperExchange из data/capital_history.json.
        Возвращает PaperExchange или None, если восстановление невозможно.
        """
        try:
            # Кандидаты путей: <repo>/data/capital_history.json и <repo>/paper_trading/data/capital_history.json
            root_data = Path(__file__).parent.parent / 'data' / 'capital_history.json'
            local_data = self.config.DATA_DIR / 'capital_history.json'
            src = next((p for p in (root_data, local_data) if p.exists()), None)
            if not src:
                logger.warning("No capital_history.json found for recovery")
                return None

            with open(src, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            history = payload.get('history') or []
            if not history:
                logger.warning("capital_history.json has no 'history'")
                return None

            last = history[-1]
            if prefer_total_equity:
                equity = last.get('total_equity')
                if equity is None:
                    fb = float(last.get('free_balance', 0.0))
                    upnl = float(last.get('unrealized_pnl', 0.0))
                    reserve = float(last.get('reserve_fund', 0.0))
                    equity = fb + upnl + reserve
            else:
                equity = last.get('free_balance')

            if equity is None:
                logger.warning("capital_history.json lacks usable equity values")
                return None

            ex = PaperExchange(initial_capital=float(equity))
            ex.balance = float(equity)  # считаем, что позиции потеряны → всё в кэше
            # сразу сохраняем восстановленное состояние, чтобы следующий старт его увидел
            state_file = self.config.DATA_DIR / 'paper_exchange_state.json'
            ex.save_state(str(state_file))
            logger.info(f"Recovered PaperExchange from capital_history.json with balance={ex.balance:.2f}")
            return ex
        except Exception as e:
            logger.error(f"Recovery from capital_history.json failed: {e}")
            return None

    def run(self):
        """Главный цикл бота"""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING PAPER TRADING BOT")
        logger.info("=" * 80)
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80 + "\n")

        try:
            # Начальное обновление портфеля
            self.update_portfolio()

            while True:
                current_time = time.time()

                # Проверка TP/SL каждые 5 минут
                if current_time - self.last_orders_check >= self.config.CHECK_ORDERS_INTERVAL:
                    self.check_orders()
                    self.print_status()

                # Обновление портфеля каждые 4 часа
                if current_time - self.last_portfolio_update >= self.config.PORTFOLIO_UPDATE_INTERVAL:
                    self.update_portfolio()

                # Сохранение состояния
                self.save_state()

                # Sleep
                time.sleep(60)  # Проверяем каждую минуту

        except KeyboardInterrupt:
            logger.info("\n\nStopping bot...")
            self.save_state()
            self.print_status()
            logger.info("Bot stopped.")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
            self.save_state()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(description='Paper Trading Bot')

    parser.add_argument('--initial-capital', type=float, default=1000.0,
                       help='Initial capital in USDT (default: 1000)')

    parser.add_argument('--max-positions', type=int, default=10,
                       help='Maximum number of positions (default: 10)')

    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    global logger
    logger = setup_logging(log_level)

    # Create and run bot
    bot = PaperTradingBot(
        initial_capital=args.initial_capital,
        max_positions=args.max_positions
    )

    bot.run()


if __name__ == '__main__':
    main()
