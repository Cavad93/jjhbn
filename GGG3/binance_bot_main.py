#!/usr/bin/env python3
"""
Binance Trading Bot - Главный торговый бот для Binance Futures

Управляет:
- 10 одновременными позициями
- EV-based coin selection
- Snapshot pattern для temporal consistency
- Online learning
- Risk management

Usage:
    # Paper trading (по умолчанию)
    python3 binance_bot_main.py --paper

    # Live trading (ОСТОРОЖНО!)
    python3 binance_bot_main.py --live
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging
import pickle
import numpy as np
import pandas as pd
import signal

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent))

# Конфиг
import binance_config as config

# Binance API
from binance_api.client import BinanceClient

# Portfolio
from portfolio.position_manager import PositionManager, Position
from portfolio.reinvestment import ReinvestmentManager
from portfolio.capital_tracker import CapitalTracker

# Strategy
from strategy.coin_selector import CoinSelector
from strategy.threshold_adapter import get_adaptive_threshold
from strategy.kelly_sizer import calculate_kelly_position_size
from strategy.haiku_analyzer import HaikuAnalyzer, analyze_and_rank_top20

# Features
from features.builder import BinanceFeatureBuilder
from features.base_logic import BaseLogicMultiTF
from features.target_calculator import calculate_atr, detect_market_phase

# Models
from models.experts import XGBoostExpert, RandomForestExpert, NeuralNetworkExpert
from meta_neural_cem import MetaNeuralCEM

# Risk management
from risk.trailing_stop import TrailingStopManager
from risk.black_swan import BlackSwanProtection
from risk.night_mode import NightModeManager

# Paper trading
from paper_trading.paper_exchange import PaperExchange

# Notifications
from notifications.telegram_notifier import TelegramNotifier

# Logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# MAIN BOT CLASS
# ============================================================================

class BinanceTradingBot:
    """
    Главный торговый бот для Binance Futures

    Управляет:
    - 10 одновременными позициями
    - EV-based coin selection
    - Snapshot pattern для temporal consistency
    - Online learning
    - Risk management
    """

    def __init__(self, paper_mode: bool = True):
        """
        Инициализация бота

        Args:
            paper_mode: True для paper trading, False для live trading
        """
        print("="*80)
        print("🤖 INITIALIZING BINANCE TRADING BOT")
        print("="*80)

        self.paper_mode = paper_mode

        # Binance client / Paper Exchange
        if paper_mode:
            # Попытка загрузить существующее состояние Paper Exchange
            state_file = 'data/paper_exchange_state.json'
            if os.path.exists(state_file):
                try:
                    self.exchange = PaperExchange.load_state(state_file)
                    balance = self.exchange.balance
                    print(f"  Mode: 📄 PAPER TRADING (Loaded: ${balance:.2f} USDT)")
                except Exception as e:
                    print(f"  ⚠️  Failed to load Paper Exchange state: {e}")
                    self.exchange = PaperExchange(initial_capital=config.PAPER_INITIAL_BALANCE)
                    print(f"  Mode: 📄 PAPER TRADING (New: ${config.PAPER_INITIAL_BALANCE})")
            else:
                self.exchange = PaperExchange(initial_capital=config.PAPER_INITIAL_BALANCE)
                print(f"  Mode: 📄 PAPER TRADING (New: ${config.PAPER_INITIAL_BALANCE})")
        else:
            self.exchange = BinanceClient(
                api_key=config.BINANCE_API_KEY,
                secret_key=config.BINANCE_SECRET_KEY,
                testnet=config.USE_TESTNET
            )
            print(f"  Mode: ⚠️  LIVE TRADING {'(TESTNET)' if config.USE_TESTNET else '(REAL)'}")

        # Portfolio manager
        self.position_manager = PositionManager(max_positions=config.MAX_POSITIONS)
        self.position_manager.load_from_file()
        print(f"  Loaded {len(self.position_manager.get_all_open())} open positions")

        # Coin selector
        self.coin_selector = CoinSelector()

        # Feature builder
        self.feature_builder = BinanceFeatureBuilder()

        # Load models
        print("\n  Loading ML models...")
        self.experts = self._load_experts()
        self.meta = self._load_meta()

        # BASE logic
        self.base_logic = BaseLogicMultiTF()

        # BASE weight calibrator
        from features.weight_calibrator import BaseWeightCalibrator
        self.base_weight_calibrator = BaseWeightCalibrator(
            min_samples=50,  # Калибровать каждые 50 сделок
            regularization=1.0,
            smoothing_alpha=0.3  # 30% EMA с предыдущими весами
        )
        self.last_calibration_at = 0  # Счётчик для калибровки

        # Risk managers
        print("\n  Initializing risk managers...")
        self.trailing_stop = TrailingStopManager()
        self.black_swan = BlackSwanProtection()
        self.night_mode = NightModeManager()

        # Reinvestment manager (только для paper trading)
        if paper_mode:
            self.reinvestment = ReinvestmentManager(
                initial_capital=config.PAPER_INITIAL_BALANCE,
                enabled=getattr(config, 'AUTO_REINVEST_ENABLED', True)
            )
            print(f"  Reinvestment: Enabled (50% profit → reserve)")

            # Capital tracker (отслеживание изменений за периоды)
            self.capital_tracker = CapitalTracker()
            print(f"  Capital tracker: Enabled")
        else:
            self.reinvestment = None
            self.capital_tracker = None

        # Telegram notifier
        self.telegram = TelegramNotifier(
            bot_token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
            enabled=config.TELEGRAM_ALERTS_ENABLED
        )

        # Haiku 4.5 Fundamental Analyzer
        if config.HAIKU_ANALYSIS_ENABLED and config.ANTHROPIC_API_KEY:
            try:
                self.haiku_analyzer = HaikuAnalyzer(
                    api_key=config.ANTHROPIC_API_KEY,
                    model=config.HAIKU_MODEL_ID,
                    cache_dir=str(config.HAIKU_CACHE_DIR),
                    batch_size=config.HAIKU_BATCH_SIZE,
                    timeout=config.HAIKU_TIMEOUT,
                    enable_stats=config.HAIKU_ENABLE_STATS
                )
                print(f"  Haiku 4.5 Analyzer: Enabled (batch_size={config.HAIKU_BATCH_SIZE})")
            except Exception as e:
                logger.warning(f"Failed to initialize Haiku Analyzer: {e}")
                self.haiku_analyzer = None
                print(f"  Haiku 4.5 Analyzer: Disabled (initialization failed)")
        else:
            self.haiku_analyzer = None
            if not config.HAIKU_ANALYSIS_ENABLED:
                print(f"  Haiku 4.5 Analyzer: Disabled (config)")
            else:
                print(f"  Haiku 4.5 Analyzer: Disabled (no API key)")

        # Statistics
        self.total_closed_trades = len(self.position_manager.closed_positions)
        self.stats = {
            'win_rate_last_100': self._calculate_win_rate(),
            'calibration_error': 0.05,
            'recent_hour_trades': 0
        }

        # Track last trade time for adaptive threshold
        self.last_trade_time = None

        print("\n✅ Bot initialized successfully\n")
        print("="*80)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Send bot started notification
        if config.TELEGRAM_ALERT_TYPES.get('bot_started', True):
            self.telegram.notify_bot_started(
                paper_mode=self.paper_mode,
                initial_capital=config.PAPER_INITIAL_BALANCE if self.paper_mode else 0.0,
                max_positions=config.MAX_POSITIONS
            )

    def _load_experts(self) -> Dict:
        """Загружает ML экспертов"""
        experts = {}

        # XGBoost
        xgb_path = config.MODELS_DIR / 'saved' / 'xgb_expert.pkl'
        if xgb_path.exists():
            try:
                experts['xgb'] = XGBoostExpert.load(str(xgb_path))
                print(f"    ✓ XGBoost")
            except Exception as e:
                print(f"    ⚠ XGBoost failed: {e}")
                experts['xgb'] = None
        else:
            experts['xgb'] = None

        # Random Forest
        rf_path = config.MODELS_DIR / 'saved' / 'rf_expert.pkl'
        if rf_path.exists():
            try:
                experts['rf'] = RandomForestExpert.load(str(rf_path))
                print(f"    ✓ RandomForest")
            except Exception as e:
                print(f"    ⚠ RandomForest failed: {e}")
                experts['rf'] = None
        else:
            experts['rf'] = None

        # ARF (fallback)
        experts['arf'] = None
        print(f"    ⚠ ARF (using fallback)")

        # Neural Network
        nn_path = config.MODELS_DIR / 'saved' / 'nn_expert.pkl'
        if nn_path.exists():
            try:
                experts['nn'] = NeuralNetworkExpert.load(str(nn_path))
                print(f"    ✓ NeuralNet")
            except Exception as e:
                print(f"    ⚠ NeuralNet failed: {e}")
                experts['nn'] = None
        else:
            experts['nn'] = None

        return experts

    def _load_meta(self):
        """Загружает META модель"""
        try:
            # Создаем экземпляр MetaNeuralCEM
            meta = MetaNeuralCEM(cfg=config)
            print(f"    ✓ META ({meta.mode} mode, {sum(meta.seen_ph.values())} samples)")
            return meta
        except Exception as e:
            print(f"    ⚠ META init failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_win_rate(self) -> float:
        """Рассчитывает win rate за последние 100 сделок"""
        closed = self.position_manager.closed_positions[-100:]
        if not closed:
            return 0.5

        wins = sum(1 for p in closed if p.pnl > 0)
        return wins / len(closed)

    def _maybe_calibrate_base_weights(self):
        """
        Проверяет нужна ли калибровка BASE весов и выполняет её

        Калибрует каждые N закрытых позиций (N = min_samples калибратора)
        """
        n_closed = len(self.position_manager.closed_positions)

        # Проверяем нужна ли калибровка
        if not self.base_weight_calibrator.should_calibrate(n_closed, self.last_calibration_at):
            return

        try:
            logger.info("[BaseWeightCalibrator] Начинаем калибровку весов...")

            # Текущие веса
            old_weights = self.base_logic.w_dyn
            if old_weights is None:
                old_weights = np.array([0.35, 0.20, 0.20, 0.25])  # Дефолтные

            # Калибруем
            result = self.base_weight_calibrator.calibrate(
                closed_positions=self.position_manager.closed_positions,
                old_weights=old_weights
            )

            if result is not None:
                # Применяем новые веса
                self.base_logic.set_weights(result.weights)

                # Обновляем счётчик
                self.last_calibration_at = n_closed

                logger.info(
                    f"[BaseWeightCalibrator] Веса обновлены: "
                    f"M={result.weights[0]:.3f}, S={result.weights[1]:.3f}, "
                    f"B={result.weights[2]:.3f}, R={result.weights[3]:.3f} "
                    f"(improvement={result.improvement:+.1f}%)"
                )

                # Отправляем уведомление в Telegram
                if self.telegram_notifier and hasattr(self.telegram_notifier, 'send_calibration_update'):
                    self.telegram_notifier.send_calibration_update({
                        'old_weights': result.old_weights.tolist(),
                        'new_weights': result.weights.tolist(),
                        'n_samples': result.n_samples,
                        'improvement': result.improvement,
                        'win_rate': result.win_rate
                    })
            else:
                logger.info("[BaseWeightCalibrator] Калибровка пропущена (недостаточно данных)")

        except Exception as e:
            logger.error(f"[BaseWeightCalibrator] Ошибка калибровки: {e}", exc_info=True)

    def _get_predictions(self, features: np.ndarray, phase: int, symbol: str, df_4h: pd.DataFrame = None) -> tuple:
        """
        Получает предсказание вероятности роста от БАЗОВОЙ ЛОГИКИ

        ВАЖНО: Используется ТОЛЬКО базовая логика (БЕЗ ML)!
        ML эксперты собираются отдельно для обучения META в shadow режиме.

        Args:
            features: 68D вектор фич (не используется для базовой логики)
            phase: Фаза рынка (0-5)
            symbol: Символ монеты
            df_4h: DataFrame 4h для базовой логики

        Returns:
            (p_up, base_signals): Вероятность роста (0-1) и словарь BASE сигналов {M, S, B, R}
        """
        try:
            # ===== ИСПОЛЬЗУЕМ ТОЛЬКО БАЗОВУЮ ЛОГИКУ =====
            # BaseLogicMultiTF работает на основе правил и индикаторов БЕЗ ML

            if df_4h is not None and len(df_4h) >= 50:
                # Используем базовую логику с 4h данными
                p_up, p_down = self.base_logic.predict(df_4h)

                # Извлекаем BASE сигналы для калибровки весов
                from features.base_logic import features_from_binance, _index_pad
                feats = features_from_binance(df_4h)
                timestamp = df_4h.index[-1]
                i = _index_pad(feats["M_up"], timestamp)

                if i is not None:
                    # Вычисляем разности сигналов (как в prob_up_down_at_time)
                    M = float(feats["M_up"].iloc[i] - feats["M_dn"].iloc[i])
                    S = float(feats["S_up"].iloc[i] - feats["S_dn"].iloc[i])
                    B = float(feats["B_up"].iloc[i] - feats["B_dn"].iloc[i])
                    R = float(feats["R_up"].iloc[i] - feats["R_dn"].iloc[i])

                    base_signals = {'M': M, 'S': S, 'B': B, 'R': R}
                else:
                    base_signals = {'M': 0.0, 'S': 0.0, 'B': 0.0, 'R': 0.0}

                # Проверяем валидность
                if not np.isfinite(p_up) or p_up < 0 or p_up > 1:
                    logger.warning(f"{symbol}: Invalid p_up from base_logic: {p_up}, using 0.5")
                    p_up = 0.5

                return float(p_up), base_signals
            else:
                # Если нет df_4h, используем нейтральное значение
                logger.warning(f"{symbol}: No df_4h provided, using neutral p_up=0.5")
                return 0.5, {'M': 0.0, 'S': 0.0, 'B': 0.0, 'R': 0.0}

        except Exception as e:
            logger.error(f"Error getting base_logic prediction for {symbol}: {e}")
            # Fallback - нейтральное значение
            return 0.5, {'M': 0.0, 'S': 0.0, 'B': 0.0, 'R': 0.0}

    def _collect_ml_predictions_for_meta(self, features: np.ndarray, phase: int, symbol: str) -> Dict[str, float]:
        """
        Собирает предсказания от ML экспертов для передачи в META

        ВАЖНО: Эти предсказания НЕ используются для торговли!
        Они передаются в META для обучения ВСЕГДА (в режимах SHADOW и ACTIVE).

        Args:
            features: 68D вектор фич
            phase: Фаза рынка (0-5)
            symbol: Символ монеты

        Returns:
            predictions: Dict с предсказаниями экспертов {'xgb': 0.62, 'rf': 0.58, ...}
        """
        predictions = {}

        try:
            # XGBoost
            if self.experts.get('xgb') is not None:
                try:
                    p_xgb = self.experts['xgb'].predict_proba(features.reshape(1, -1))[0]
                    if np.isfinite(p_xgb) and 0 <= p_xgb <= 1:
                        predictions['xgb'] = float(p_xgb)
                except Exception as e:
                    logger.debug(f"{symbol}: XGBoost prediction failed: {e}")

            # Random Forest
            if self.experts.get('rf') is not None:
                try:
                    p_rf = self.experts['rf'].predict_proba(features.reshape(1, -1))[0]
                    if np.isfinite(p_rf) and 0 <= p_rf <= 1:
                        predictions['rf'] = float(p_rf)
                except Exception as e:
                    logger.debug(f"{symbol}: RandomForest prediction failed: {e}")

            # Neural Network
            if self.experts.get('nn') is not None:
                try:
                    p_nn = self.experts['nn'].predict_proba(features.reshape(1, -1))[0]
                    if np.isfinite(p_nn) and 0 <= p_nn <= 1:
                        predictions['nn'] = float(p_nn)
                except Exception as e:
                    logger.debug(f"{symbol}: NeuralNet prediction failed: {e}")

        except Exception as e:
            logger.error(f"Error collecting ML predictions for {symbol}: {e}")

        return predictions

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        """Главный торговый цикл"""

        print(f"\n{'='*80}")
        print(f"🚀 STARTING BINANCE TRADING BOT")
        print(f"{'='*80}\n")

        last_4h_check = 0
        last_5m_check = 0

        while True:
            try:
                current_time = time.time()

                # ═══════════════════════════════════════════════════
                # КАЖДЫЕ 4 ЧАСА: Обновление портфеля
                # ═══════════════════════════════════════════════════

                if current_time - last_4h_check >= 4 * 3600:
                    print(f"\n{'='*80}")
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 4H CYCLE: Portfolio rebalancing")
                    print(f"{'='*80}\n")

                    self.rebalance_portfolio()

                    # Выводим статистику Haiku после ребалансировки
                    if self.haiku_analyzer is not None:
                        print(self.haiku_analyzer.get_stats_summary())

                    last_4h_check = current_time

                # ═══════════════════════════════════════════════════
                # КАЖДЫЕ 5 МИНУТ: Проверка позиций
                # ═══════════════════════════════════════════════════

                if current_time - last_5m_check >= 300:
                    self.check_positions()
                    last_5m_check = current_time

                # Пауза
                time.sleep(60)

            except KeyboardInterrupt:
                print("\n\n⚠️  Stopping bot...")
                self.shutdown()
                break

            except Exception as e:
                logger.error(f"ERROR in main loop: {e}", exc_info=True)
                print(f"\n❌ ERROR: {e}")
                time.sleep(300)

    # ========================================================================
    # PORTFOLIO REBALANCING
    # ========================================================================

    def rebalance_portfolio(self):
        """
        Ребалансировка портфеля каждые 4 часа

        1. Проверяем защитные режимы
        2. Обновляем топ-10 монет
        3. Закрываем позиции не в топ-10
        4. Открываем новые позиции
        """

        # Защитные режимы
        is_black_swan = self.black_swan.check_black_swan_event()
        if is_black_swan:
            print("⚠️  BLACK SWAN EVENT: Closing all positions and skipping new entries")

            # Закрываем все открытые позиции
            open_positions = list(self.position_manager.get_all_open())
            positions_closed = len(open_positions)

            for position in open_positions:
                self.close_position_manual(position)

            # Send Telegram notification
            if config.TELEGRAM_ALERT_TYPES.get('black_swan', True):
                # Получаем реальный процент падения BTC из black_swan модуля
                btc_drop = self.black_swan.get_last_price_drop_pct()
                self.telegram.notify_black_swan_triggered(
                    btc_drop=btc_drop,
                    positions_closed=positions_closed
                )

            return

        # Night mode
        is_night_mode = self.night_mode.is_night_mode_active()
        if is_night_mode:
            print("🌙 NIGHT MODE: Reducing activity")

            # Send Telegram notification (только при первой проверке в night mode)
            # Используем hasattr для отслеживания состояния
            if not hasattr(self, '_night_mode_notified') or not self._night_mode_notified:
                if config.TELEGRAM_ALERT_TYPES.get('night_mode', True):
                    max_pos = self.night_mode.get_max_positions()
                    self.telegram.notify_night_mode_activated(max_positions=max_pos)
                    self._night_mode_notified = True
        else:
            # Деактивация night mode
            if hasattr(self, '_night_mode_notified') and self._night_mode_notified:
                if config.TELEGRAM_ALERT_TYPES.get('night_mode', True):
                    self.telegram.notify_night_mode_deactivated()
                    self._night_mode_notified = False

        # Получаем все торгуемые пары
        print("Fetching all USDT pairs...")
        all_pairs = self.exchange.get_all_usdt_pairs()
        print(f"  Found {len(all_pairs)} pairs")

        # Отбираем топ-10
        print("\nSelecting top-10 opportunities...")

        # Adaptive threshold
        p_threshold = get_adaptive_threshold(
            total_closed=self.total_closed_trades,
            recent_hour=self.stats['recent_hour_trades'],
            recent_wr=self.stats['win_rate_last_100'],
            calib_error=self.stats['calibration_error'],
            last_trade_time=self.last_trade_time
        )

        print(f"  Adaptive threshold: {p_threshold:.4f}")

        # ═══════════════════════════════════════════════════════════════════════
        # ОПТИМИЗАЦИЯ: Один проход для топ-20, экономия ~30-60 минут!
        # ═══════════════════════════════════════════════════════════════════════
        # Сканируем рынок ОДИН РАЗ для получения топ-20 возможностей
        # Затем используем:
        #   - Все 20 для решений о закрытии существующих позиций
        #   - Только первые 10 для открытия новых позиций
        # ═══════════════════════════════════════════════════════════════════════
        top_20_opportunities = self.coin_selector.select_top_coins(
            all_pairs=all_pairs,
            get_ticker_func=lambda s: self.exchange.get_ticker(s),
            get_ohlcv_func=lambda s, tf, lim: self.exchange.get_ohlcv(s, tf, lim),
            feature_builder=self.feature_builder,
            calculate_atr_func=calculate_atr,
            calculate_phase_func=detect_market_phase,
            get_predictions_func=self._get_predictions,  # Базовая логика для торговли
            collect_ml_predictions_func=self._collect_ml_predictions_for_meta,  # ML для META
            top_n=20,  # Получаем топ-20 за один проход
            exchange=self.exchange  # Для получения funding rate
        )

        print(f"\n  Top-20 opportunities (for closure decisions):")
        for i, opp in enumerate(top_20_opportunities, 1):
            marker = "✓" if i <= 10 else " "  # Отмечаем первые 10
            print(f"    {marker} {i:2d}. {opp['symbol']:<12} {opp['direction']:<6} "
                  f"p_up={opp['p_up']:.3f}  EV={opp['ev']:.4f}")

        # ═══════════════════════════════════════════════════════════════════════
        # HAIKU 4.5: Фундаментальный анализ TOP-20 для формирования финального TOP-10
        # ═══════════════════════════════════════════════════════════════════════
        if self.haiku_analyzer is not None:
            print(f"\n🤖 Running Haiku 4.5 fundamental analysis on TOP-20...")
            try:
                # Анализ и переранжирование с учётом фундаментальных факторов
                top_10_enriched = analyze_and_rank_top20(
                    top_20_opportunities=top_20_opportunities,
                    haiku_analyzer=self.haiku_analyzer,
                    p_threshold=p_threshold
                )

                # Заменяем TOP-20 на обогащённый TOP-10
                # (для закрытия позиций всё ещё используем все 20)
                print(f"\n  Haiku 4.5 analysis completed: {len(top_10_enriched)} coins approved")

                # Логируем решения Haiku
                for i, opp in enumerate(top_10_enriched, 1):
                    haiku_score = opp.get('haiku_score', 'N/A')
                    haiku_reason = opp.get('haiku_reason', 'N/A')
                    print(f"    {i:2d}. {opp['symbol']:<12} {opp['direction']:<6} "
                          f"tech={opp['p_up']:.3f} fund={haiku_score:.3f if isinstance(haiku_score, float) else haiku_score} "
                          f"EV={opp['ev']:.4f} ({haiku_reason})")

                # Используем обогащённый список для открытия позиций
                top_10_for_opening = top_10_enriched

            except Exception as e:
                logger.error(f"Haiku analysis failed: {e}")
                print(f"  ⚠️  Haiku analysis failed: {e}")
                print(f"  Fallback: using technical analysis only")
                # Fallback: используем первые 10 из TOP-20
                top_10_for_opening = top_20_opportunities[:10]
        else:
            # Haiku отключён - используем первые 10 из TOP-20
            top_10_for_opening = top_20_opportunities[:10]

        print(f"\n  Checking if existing positions should be closed (using top-20)...")

        # Счетчик закрытых позиций
        positions_before_close = len(self.position_manager.get_all_open())
        closed_count = 0

        # Получаем символы из топ-20 для проверки закрытия позиций
        top_20_symbols = {opp['symbol'] for opp in top_20_opportunities}

        for position in list(self.position_manager.get_all_open()):
            if position.symbol not in top_20_symbols:
                print(f"\n  Closing {position.symbol}: not in top-20 anymore")
                self.close_position_manual(position)
                closed_count += 1

        # ═══════════════════════════════════════════════════════════════════════
        # КРИТИЧНО: Вычисляем торговый капитал ОДИН РАЗ для всех позиций
        # Kelly будет рассчитываться от ПОЛНОГО баланса, а не от остатка
        # ═══════════════════════════════════════════════════════════════════════
        initial_trading_capital = self.exchange.get_balance('USDT')

        # Apply reinvestment if enabled (paper trading only)
        if self.paper_mode and self.reinvestment and self.reinvestment.enabled:
            reinvest_result = self.reinvestment.calculate_daily_reinvestment(initial_trading_capital)
            initial_trading_capital = reinvest_result['trading_capital']

            # Log reserve fund info
            if reinvest_result['reinvested']:
                print(f"\n  [Reinvest] Daily profit: ${reinvest_result['daily_profit']:.2f}")
                print(f"  [Reinvest] To reserve: ${reinvest_result['to_reserve']:.2f}")
                print(f"  [Reinvest] Trading capital: ${initial_trading_capital:.2f}")
                print(f"  [Reinvest] Reserve fund: ${reinvest_result['reserve_fund']:.2f}")

        # Открываем новые позиции (используем обогащённый Haiku топ-10)
        print(f"\n  Opening new positions (using top-10 from Haiku analysis)...")
        positions_before_open = len(self.position_manager.get_all_open())
        opened_count = 0

        for opp in top_10_for_opening:
            if self.position_manager.has_position(opp['symbol']):
                continue  # Уже открыта

            # Проверяем порог
            if opp['direction'] == 'LONG':
                if opp['p_up'] <= p_threshold:
                    continue
            else:  # SHORT
                if opp['p_up'] >= (1 - p_threshold):
                    continue

            # Проверяем лимит позиций (ночью меньше)
            current_positions = len(self.position_manager.get_all_open())
            max_positions = self.night_mode.get_max_positions()

            if current_positions >= max_positions:
                print(f"\n  Max positions reached ({current_positions}/{max_positions})")
                break

            # Открываем позицию с фиксированным торговым капиталом
            success = self.open_position(opp, initial_trading_capital=initial_trading_capital)
            if success:
                opened_count += 1

        final_positions = len(self.position_manager.get_all_open())
        print(f"\n  Portfolio status: {final_positions}/{config.MAX_POSITIONS} positions")
        print(f"  Rebalance summary: closed {closed_count}, opened {opened_count}")

        # Send Telegram notification
        if config.TELEGRAM_ALERT_TYPES.get('rebalance', True):
            stats = {
                'closed_positions': closed_count,
                'opened_positions': opened_count,
                'total_positions': final_positions,
                'top_opportunities': top_20_opportunities[:5]  # Top 5
            }
            self.telegram.notify_rebalance_stats(stats)

    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================

    def open_position(self, opportunity: dict, initial_trading_capital: float = None):
        """
        Открывает позицию с полным snapshot

        КРИТИЧНО: Сохраняет ВСЕ данные на момент входа (t0)

        Args:
            opportunity: Торговая возможность с данными
            initial_trading_capital: Начальный торговый капитал для Kelly (фиксированный)
                                    Если None, вычисляется из текущего баланса
        """

        symbol = opportunity['symbol']
        direction = opportunity['direction']

        print(f"\n{'─'*60}")
        print(f"[ENTRY] {symbol} {direction}")

        try:
            # Получаем текущую цену
            current_price = self.exchange.get_current_price(symbol)

            # ATR для расчета TP/SL
            df_4h = self.exchange.get_ohlcv(symbol, '4h', 100)

            from indicators import atr as calc_atr
            atr_values = calc_atr(df_4h['high'], df_4h['low'], df_4h['close'], period=config.ATR_PERIOD)
            atr = atr_values[-1]

            # Рассчитываем ATR в процентах для адаптивных множителей
            atr_pct = (atr / current_price) * 100

            # Получаем адаптивные множители TP/SL на основе волатильности
            tp_mult, sl_mult = config.get_adaptive_tp_sl_multipliers(atr_pct)

            # Рассчитываем R:R ratio для Kelly Criterion
            rr_ratio = tp_mult / sl_mult

            # TP/SL уровни с адаптивными множителями
            if direction == 'LONG':
                tp_price = current_price + tp_mult * atr
                sl_price = current_price - sl_mult * atr
            else:  # SHORT
                tp_price = current_price - tp_mult * atr
                sl_price = current_price + sl_mult * atr

            # ═══════════════════════════════════════════════════════════════
            # Размер позиции (Kelly)
            # КРИТИЧНО: Используем ФИКСИРОВАННЫЙ торговый капитал для всех позиций
            # Это гарантирует что Kelly рассчитывается от полного баланса,
            # а не от остатка после предыдущих покупок
            # ═══════════════════════════════════════════════════════════════
            if initial_trading_capital is not None:
                # Используем переданный фиксированный капитал
                capital = initial_trading_capital
            else:
                # Fallback: вычисляем из текущего баланса (старое поведение)
                capital = self.exchange.get_balance('USDT')

                # Apply reinvestment if enabled (paper trading only)
                if self.paper_mode and self.reinvestment and self.reinvestment.enabled:
                    reinvest_result = self.reinvestment.calculate_daily_reinvestment(capital)
                    capital = reinvest_result['trading_capital']

            position_size_usd = calculate_kelly_position_size(
                capital=capital,
                ev=opportunity['ev'],
                p_up=opportunity['p_up'],
                recent_wr=self.stats['win_rate_last_100'],
                rr_ratio=rr_ratio
            )

            amount = position_size_usd / current_price

            # ✅ КРИТИЧНО: Создаем snapshot на момент входа (t0)
            entry_snapshot = {
                'features_68d': opportunity.get('features', np.zeros(68)),
                'predictions': opportunity.get('predictions', {}),
                'ml_predictions': opportunity.get('ml_predictions', {}),  # ML предсказания для META
                'meta_context': opportunity.get('context', {}),  # ✅ СОХРАНЯЕМ РЕАЛЬНЫЕ КОНТЕКСТНЫЕ ФИЧИ
                'p_meta': opportunity['p_up'],
                'phase': opportunity.get('phase', 0),  # Фаза рынка
                'timestamp': time.time(),
                'context': opportunity.get('context', {}),  # ✅ Дублируем для совместимости
                'base_signals': opportunity.get('base_signals', {})  # ✅ BASE сигналы [M, S, B, R] для калибровки
            }

            # Создаем Position объект
            position = Position(
                symbol=symbol,
                direction=direction,
                entry_time=datetime.now(),
                entry_price=current_price,
                amount=amount,
                position_value=position_size_usd,
                tp_price=tp_price,
                sl_price=sl_price,
                atr_value=atr,
                entry_snapshot=entry_snapshot,  # ✅ SNAPSHOT SAVED
                status='OPEN',
                entry_order_id=None,
                tp_order_id=None,
                sl_order_id=None
            )

            # Открываем на бирже
            # Entry order
            entry_order = self.exchange.create_market_order(
                symbol=symbol,
                side='BUY' if direction == 'LONG' else 'SELL',
                amount=amount
            )
            position.entry_order_id = entry_order['id']

            # TP order
            tp_order = self.exchange.create_limit_order(
                symbol=symbol,
                side='SELL' if direction == 'LONG' else 'BUY',
                amount=amount,
                price=tp_price
            )
            position.tp_order_id = tp_order['id']

            # SL order
            sl_order = self.exchange.create_stop_market_order(
                symbol=symbol,
                side='SELL' if direction == 'LONG' else 'BUY',
                amount=amount,
                stop_price=sl_price
            )
            position.sl_order_id = sl_order['id']

            # Добавляем в портфель
            self.position_manager.add_position(position)

            print(f"  Entry:  ${current_price:.2f}")
            print(f"  TP:     ${tp_price:.2f} ({((tp_price/current_price-1)*100):+.2f}%)")
            print(f"  SL:     ${sl_price:.2f} ({((sl_price/current_price-1)*100):+.2f}%)")
            print(f"  Size:   ${position_size_usd:.2f}")
            print(f"  ✅ Position opened")

            # Send Telegram notification
            if config.TELEGRAM_ALERT_TYPES.get('position_opened', True):
                position_data = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'position_size': position_size_usd,
                    'leverage': config.LEVERAGE,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'p_up': opportunity['p_up'],
                    'ev': opportunity['ev'],
                    'predictions': opportunity.get('predictions', {})
                }

                # ПРАВИЛЬНЫЙ расчет капитала ПОСЛЕ открытия позиции
                if self.paper_mode:
                    # Получаем текущий свободный баланс (после открытия позиции)
                    current_free_balance = self.exchange.get_balance('USDT')

                    # Рассчитываем стоимость всех открытых позиций
                    locked_in_positions = sum(
                        pos.position_value
                        for pos in self.position_manager.get_all_open()
                    )

                    # Резервный фонд
                    current_reserve = self.reinvestment.reserve_fund if self.reinvestment else 0.0

                    # Общий капитал = Свободный + Заблокированный + Резерв
                    actual_total_equity = current_free_balance + locked_in_positions + current_reserve

                    # Записываем snapshot
                    if self.capital_tracker:
                        self.capital_tracker.record_snapshot(
                            free_balance=current_free_balance,
                            locked_in_positions=locked_in_positions,
                            reserve_fund=current_reserve,
                            num_open_positions=len(self.position_manager.get_all_open())
                        )

                        # Получаем изменения за периоды
                        period_changes = self.capital_tracker.get_all_period_changes(actual_total_equity)
                        position_data['period_changes'] = period_changes

                    # Обновляем данные для уведомления
                    position_data['reserve_fund'] = current_reserve
                    position_data['trading_capital'] = current_free_balance
                    position_data['locked_in_positions'] = locked_in_positions
                    position_data['total_equity'] = actual_total_equity

                self.telegram.notify_position_opened(position_data)

            return True  # ✅ Позиция успешно открыта

        except Exception as e:
            logger.error(f"Error opening position {symbol}: {e}", exc_info=True)
            print(f"  ❌ Error: {e}")
            return False  # ❌ Ошибка при открытии

    def check_positions(self):
        """
        Проверяет открытые позиции каждые 5 минут

        1. Проверяет TP/SL исполнение
        2. Проверяет trailing stop
        3. Проверяет timeout
        4. Обучает модели на закрытых позициях
        """

        for position in list(self.position_manager.get_all_open()):

            try:
                # Получаем текущую цену
                current_price = self.exchange.get_current_price(position.symbol)

                # Проверяем исполнение TP/SL
                if self.paper_mode:
                    # Paper mode: проверяем достижение уровней вручную
                    if position.direction == 'LONG':
                        if current_price >= position.tp_price:
                            # Закрываем позицию на бирже (КРИТИЧНО: возвращает деньги на баланс!)
                            self.exchange.create_market_order(
                                symbol=position.symbol,
                                side='SELL',
                                amount=position.amount
                            )
                            self.close_position(position, 'TP', position.tp_price)
                            continue
                        elif current_price <= position.sl_price:
                            # Закрываем позицию на бирже (КРИТИЧНО: возвращает деньги на баланс!)
                            self.exchange.create_market_order(
                                symbol=position.symbol,
                                side='SELL',
                                amount=position.amount
                            )
                            self.close_position(position, 'SL', position.sl_price)
                            continue
                    else:  # SHORT
                        if current_price <= position.tp_price:
                            # Закрываем позицию на бирже (КРИТИЧНО: возвращает деньги на баланс!)
                            self.exchange.create_market_order(
                                symbol=position.symbol,
                                side='BUY',
                                amount=position.amount
                            )
                            self.close_position(position, 'TP', position.tp_price)
                            continue
                        elif current_price >= position.sl_price:
                            # Закрываем позицию на бирже (КРИТИЧНО: возвращает деньги на баланс!)
                            self.exchange.create_market_order(
                                symbol=position.symbol,
                                side='BUY',
                                amount=position.amount
                            )
                            self.close_position(position, 'SL', position.sl_price)
                            continue

                # Проверяем trailing stop
                new_sl = self.trailing_stop.check_trailing_stop(position, current_price)
                if new_sl is not None:
                    print(f"[{position.symbol}] Trailing stop: SL {position.sl_price:.2f} → {new_sl:.2f}")

                    # Рассчитываем прибыль
                    if position.direction == 'LONG':
                        profit_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    else:
                        profit_pct = ((position.entry_price - current_price) / position.entry_price) * 100

                    # Обновляем SL
                    self.exchange.cancel_order(position.symbol, position.sl_order_id)
                    sl_order = self.exchange.create_stop_market_order(
                        symbol=position.symbol,
                        side='SELL' if position.direction == 'LONG' else 'BUY',
                        amount=position.amount,
                        stop_price=new_sl
                    )
                    position.sl_order_id = sl_order['id']
                    position.sl_price = new_sl

                    # Send Telegram notification
                    if config.TELEGRAM_ALERT_TYPES.get('trailing_stop', True):
                        self.telegram.notify_trailing_stop_activated(
                            symbol=position.symbol,
                            direction=position.direction,
                            current_price=current_price,
                            new_sl=new_sl,
                            profit_pct=profit_pct
                        )

                # Проверяем timeout
                duration_hours = (datetime.now() - position.entry_time).total_seconds() / 3600
                if duration_hours >= config.MAX_POSITION_HOLD_TIME_HOURS:
                    print(f"[{position.symbol}] TIMEOUT: {duration_hours:.1f}h")
                    self.close_position_manual(position)

            except Exception as e:
                logger.error(f"Error checking position {position.symbol}: {e}", exc_info=True)

    def close_position(self, position: Position, exit_reason: str, exit_price: float):
        """
        Закрывает позицию и обучает модели

        КРИТИЧНО: Использует СОХРАНЕННЫЕ фичи из entry_snapshot (t0)
        """

        print(f"\n[EXIT] {position.symbol} {position.direction} - {exit_reason}")

        # Обновляем Position
        position.exit_time = datetime.now()
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.status = 'CLOSED'

        # Рассчитываем PnL
        if position.direction == 'LONG':
            position.pnl = (exit_price - position.entry_price) * position.amount
            position.pnl_pct = (exit_price / position.entry_price - 1) * 100
        else:  # SHORT
            position.pnl = (position.entry_price - exit_price) * position.amount
            position.pnl_pct = (1 - exit_price / position.entry_price) * 100

        duration = (position.exit_time - position.entry_time).total_seconds() / 3600

        print(f"  Duration: {duration:.1f}h")
        print(f"  PnL: ${position.pnl:.2f} ({position.pnl_pct:+.2f}%)")

        # ✅ КРИТИЧНО: Online learning с snapshot
        # Передаем результат в META для обучения (ВСЕГДА: в SHADOW и ACTIVE)
        try:
            if self.meta is not None and hasattr(position, 'entry_snapshot') and position.entry_snapshot:
                snapshot = position.entry_snapshot
                ml_preds = snapshot.get('ml_predictions', {})

                # Определяем фактический результат
                y_up = 1 if position.pnl > 0 else 0

                # ✅ Формируем полный контекст из snapshot (реальные значения из t0)
                context = snapshot.get('context', {})
                reg_ctx = {
                    'phase': snapshot.get('phase', 0),
                    'vol_ratio': context.get('vol_ratio', 1.0),
                    'trend_macd': context.get('trend_macd', 0.0),
                    'jump_detected': context.get('jump_detected', False),
                    'funding_sign': context.get('funding_sign', 0.0),
                    'book_imb': context.get('book_imb', 0.0),
                    'ofi_15s': context.get('ofi_15s', 0.0),
                    'basis_pct': context.get('basis_pct', 0.0)
                }

                # Передаем в META для обучения с РЕАЛЬНЫМ КОНТЕКСТОМ
                self.meta.record_result(
                    p_xgb=ml_preds.get('xgb'),
                    p_rf=ml_preds.get('rf'),
                    p_arf=None,  # У нас нет ARF
                    p_nn=ml_preds.get('nn'),
                    p_base=snapshot.get('p_meta'),  # Предсказание базовой логики
                    y_up=y_up,
                    used_in_live=True,
                    p_final_used=snapshot.get('p_meta'),
                    reg_ctx=reg_ctx  # ✅ ПЕРЕДАЕМ РЕАЛЬНЫЙ КОНТЕКСТ из t0
                )
                logger.debug(f"META.record_result() called for {position.symbol}: y_up={y_up}, context={reg_ctx}")
        except Exception as e:
            logger.error(f"Error recording result to META: {e}")

        # Отменяем TP/SL ордера в exchange (для paper trading, если они существуют)
        if self.paper_mode and hasattr(self.exchange, 'orders'):
            try:
                if position.tp_order_id and position.tp_order_id in self.exchange.orders:
                    self.exchange.cancel_order(position.symbol, position.tp_order_id)
                if position.sl_order_id and position.sl_order_id in self.exchange.orders:
                    self.exchange.cancel_order(position.symbol, position.sl_order_id)
            except Exception as e:
                logger.warning(f"Failed to cancel TP/SL orders for {position.symbol}: {e}")

        # ═══════════════════════════════════════════════════════════════════════
        # КРИТИЧНО: Закрываем позицию в exchange для возврата баланса
        # Без этого баланс "застревает" в закрытых позициях!
        # ═══════════════════════════════════════════════════════════════════════
        if self.paper_mode:
            try:
                # Создаем market order для закрытия позиции
                close_side = 'SELL' if position.direction == 'LONG' else 'BUY'
                self.exchange.create_market_order(
                    symbol=position.symbol,
                    side=close_side,
                    amount=position.amount
                )
            except Exception as e:
                logger.error(f"Failed to close position in exchange for {position.symbol}: {e}")

        # Закрываем позицию в manager
        self.position_manager.close_position(position.symbol, exit_price, exit_reason)

        # Обновляем статистику
        self.total_closed_trades += 1
        self.stats['win_rate_last_100'] = self._calculate_win_rate()
        self.last_trade_time = time.time()  # Track time for adaptive threshold

        # Проверяем нужна ли калибровка BASE весов
        self._maybe_calibrate_base_weights()

        # Сбрасываем trailing stop
        self.trailing_stop.reset_position(position.symbol)

        # Send Telegram notification
        if config.TELEGRAM_ALERT_TYPES.get('position_closed', True):
            position_data = {
                'symbol': position.symbol,
                'direction': position.direction,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'position_size': position.position_value,
                'pnl': position.pnl,
                'pnl_percent': position.pnl_pct,
                'exit_reason': exit_reason,
                'holding_time': f"{duration:.1f}h"
            }
            self.telegram.notify_position_closed(position_data)

    def close_position_manual(self, position: Position):
        """Ручное закрытие позиции (для таймаута или замены)"""
        try:
            current_price = self.exchange.get_current_price(position.symbol)

            # Отменяем TP/SL ордера (если они существуют)
            if self.paper_mode and hasattr(self.exchange, 'orders'):
                if position.tp_order_id and position.tp_order_id in self.exchange.orders:
                    self.exchange.cancel_order(position.symbol, position.tp_order_id)
                if position.sl_order_id and position.sl_order_id in self.exchange.orders:
                    self.exchange.cancel_order(position.symbol, position.sl_order_id)

            # Закрываем по рынку
            self.exchange.create_market_order(
                symbol=position.symbol,
                side='SELL' if position.direction == 'LONG' else 'BUY',
                amount=position.amount
            )

            self.close_position(position, 'MANUAL', current_price)

        except Exception as e:
            logger.error(f"Error closing position manually {position.symbol}: {e}", exc_info=True)

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful shutdown"""
        signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
        print(f"\n\n⚠️  Received {signal_name} signal, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        """Корректное завершение бота с закрытием всех активных позиций"""
        print("\n\n" + "="*80)
        print("🛑 GRACEFUL SHUTDOWN - Closing all positions...")
        print("="*80 + "\n")

        # ═══════════════════════════════════════════════════════════
        # ШАГ 1: Закрываем все открытые позиции
        # ═══════════════════════════════════════════════════════════

        open_positions = self.position_manager.get_all_open()

        if open_positions:
            print(f"📊 Found {len(open_positions)} open positions to close\n")

            for position in open_positions:
                try:
                    # Получаем текущую цену
                    current_price = self.exchange.get_current_price(position.symbol)

                    # Закрываем позицию через общий метод (который теперь корректно возвращает баланс)
                    self.close_position(position, 'SHUTDOWN', current_price)

                    print(f"✅ Closed {position.symbol}: Entry={position.entry_price:.6f}, Exit={current_price:.6f}")

                except Exception as e:
                    logger.error(f"❌ Failed to close {position.symbol}: {e}")
                    print(f"❌ Error closing {position.symbol}: {e}")
        else:
            print("ℹ️  No open positions to close\n")

        # ═══════════════════════════════════════════════════════════
        # ИТОГИ: Показываем финальный баланс
        # ═══════════════════════════════════════════════════════════

        if self.paper_mode:
            print("\n" + "="*80)
            print("💰 FINAL BALANCE")
            print("="*80)

            current_balance = self.exchange.get_balance('USDT')
            initial_balance = self.exchange.initial_capital

            # Подсчитываем общий PnL из закрытых позиций
            total_pnl = sum(
                pos.pnl
                for pos in self.position_manager.closed_positions
            )

            # Статистика по позициям
            total_closed = len(self.position_manager.closed_positions)
            if total_closed > 0:
                wins = sum(1 for pos in self.position_manager.closed_positions if pos.pnl > 0)
                losses = total_closed - wins
                win_rate = (wins / total_closed) * 100 if total_closed > 0 else 0

                print(f"\n📊 Trading Statistics:")
                print(f"  Total trades:     {total_closed}")
                print(f"  Winning trades:   {wins} ({win_rate:.1f}%)")
                print(f"  Losing trades:    {losses}")
                print(f"  Total PnL:        ${total_pnl:+,.2f}")

            print(f"\n💵 Account Balance:")
            print(f"  Initial balance:  ${initial_balance:,.2f}")
            print(f"  Current balance:  ${current_balance:,.2f}")
            print(f"  Net change:       ${current_balance - initial_balance:+,.2f}")

            # Если есть резервный фонд
            if self.reinvestment and self.reinvestment.reserve_fund > 0:
                reserve = self.reinvestment.reserve_fund
                total_equity = current_balance + reserve
                print(f"\n  Reserve fund:     ${reserve:,.2f}")
                print(f"  Total equity:     ${total_equity:,.2f}")
                print(f"  Total change:     ${total_equity - initial_balance:+,.2f}")

            # ROI
            roi = ((current_balance - initial_balance) / initial_balance) * 100
            print(f"\n  ROI:              {roi:+.2f}%")
            print("="*80)

        # ═══════════════════════════════════════════════════════════
        # ШАГ 2: Отменяем все оставшиеся активные ордера
        # ═══════════════════════════════════════════════════════════

        if self.paper_mode and hasattr(self.exchange, 'orders'):
            remaining_orders = list(self.exchange.orders.values())
            if remaining_orders:
                print(f"\n📋 Cancelling {len(remaining_orders)} remaining orders...")
                for order in remaining_orders:
                    try:
                        self.exchange.cancel_order(order['symbol'], order['id'])
                        print(f"✅ Cancelled order {order['id']} for {order['symbol']}")
                    except Exception as e:
                        logger.warning(f"Failed to cancel order {order['id']}: {e}")

        # ═══════════════════════════════════════════════════════════
        # ШАГ 3: Сохраняем состояние
        # ═══════════════════════════════════════════════════════════

        print("\n💾 Saving state...")
        self.position_manager.save_to_file()
        print("✅ Positions saved")

        # Сохранение баланса Paper Exchange
        if self.paper_mode:
            try:
                self.exchange.save_state('data/paper_exchange_state.json')
                print("✅ Paper Exchange balance saved")
            except Exception as e:
                print(f"⚠️  Failed to save Paper Exchange state: {e}")

        # Print reinvestment summary if enabled
        if self.paper_mode and self.reinvestment:
            print("\n" + "="*80)
            print("📊 REINVESTMENT SUMMARY")
            print("="*80)
            # Передаём текущий баланс для отображения актуального состояния
            current_balance = self.exchange.get_balance('USDT')
            self.reinvestment.print_summary(current_balance=current_balance)

        # Send bot stopped notification
        if config.TELEGRAM_ALERT_TYPES.get('bot_stopped', True):
            self.telegram.notify_bot_stopped(reason="Manual shutdown")

        print("\n" + "="*80)
        print("✅ BOT STOPPED SUCCESSFULLY")
        print("="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Binance Trading Bot')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode (default)')
    parser.add_argument('--live', action='store_true', help='Live trading mode (CAUTION!)')
    args = parser.parse_args()

    if args.live:
        print("\n⚠️  WARNING: LIVE TRADING MODE")
        print("⚠️  This will use REAL money!")
        response = input("Are you sure? Type 'YES' to continue: ")
        if response != 'YES':
            print("Aborted.")
            sys.exit(0)
        bot = BinanceTradingBot(paper_mode=False)
    else:
        bot = BinanceTradingBot(paper_mode=True)

    bot.run()
