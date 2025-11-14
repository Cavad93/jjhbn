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
import io
import threading

# ════════════════════════════════════════════════════════════════════════════════
# ЗАЩИТА ОТ БЛОКИРУЮЩЕГО ВВОДА
# ════════════════════════════════════════════════════════════════════════════════
# Перенаправляем stdin на пустой поток, чтобы любые input() или блокирующие
# чтения (в т.ч. в сторонних библиотеках) сразу получали EOF вместо зависания.
# Это предотвращает ситуации, когда бот зависает в ожидании ввода от пользователя.
# ════════════════════════════════════════════════════════════════════════════════
sys.stdin = io.StringIO('')
print("[STARTUP] stdin redirected to empty stream (protection against blocking input)", flush=True)

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
from models.experts import XGBoostExpert, RandomForestExpert, AdaptiveRFExpert, NeuralNetworkExpert
from simplified_ensemble_meta import SimplifiedEnsembleMETA
from models.transfer_learning import TransferLearningManager, PretrainingConfig
from models.diversity_metric import DiversityMonitor, AlertType

# Risk management
from risk.trailing_stop import TrailingStopManager
from risk.black_swan import BlackSwanProtection
from risk.night_mode import NightModeManager

# Paper trading
from paper_trading.paper_exchange import PaperExchange

# Notifications
from notifications.telegram_notifier import TelegramNotifier
from notifications.ai_assistant import AITradingAssistant, BotContextCollector

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

        # Diversity Monitor (для мониторинга разнообразия экспертов)
        self.diversity_monitor = DiversityMonitor(
            window_size=100,
            low_diversity_threshold=0.05,
            high_disagreement_threshold=0.20,
            high_correlation_threshold=0.95
        )
        print("    ✓ Diversity Monitor initialized (window=100)")

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
            enabled=config.TELEGRAM_ALERTS_ENABLED,
            group_chat_id=config.TELEGRAM_GROUP_CHAT_ID,
            allowed_user_id=config.TELEGRAM_ALLOWED_USER_ID
        )

        # AI Trading Assistant (Claude Sonnet 4.5)
        if config.TELEGRAM_ALERTS_ENABLED and config.ANTHROPIC_API_KEY:
            try:
                # Context collector для сбора данных бота
                self.bot_context_collector = BotContextCollector(
                    position_manager=self.position_manager,
                    capital_tracker=self.capital_tracker if paper_mode else None,
                    config=config,
                    exchange=self.exchange  # ✅ Передаём exchange для получения текущих цен
                )

                # AI Assistant
                self.ai_assistant = AITradingAssistant(
                    api_key=config.ANTHROPIC_API_KEY,
                    context_collector=self.bot_context_collector
                )

                # Связываем с Telegram
                self.telegram.set_ai_assistant(self.ai_assistant)

                print(f"  AI Assistant: Enabled (Claude Sonnet 4.5)")
            except Exception as e:
                logger.warning(f"Failed to initialize AI Assistant: {e}")
                self.ai_assistant = None
                self.bot_context_collector = None
                print(f"  AI Assistant: Disabled (initialization failed)")
        else:
            self.ai_assistant = None
            self.bot_context_collector = None
            if not config.TELEGRAM_ALERTS_ENABLED:
                print(f"  AI Assistant: Disabled (Telegram disabled)")
            else:
                print(f"  AI Assistant: Disabled (no API key)")

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

        # ════════════════════════════════════════════════════════════════════════
        # EXPERT WIN RATE TRACKING
        # ════════════════════════════════════════════════════════════════════════
        # Отслеживание винрейтов по каждому эксперту и базовой логике
        self.expert_stats = {
            'base': {'wins': 0, 'losses': 0, 'total': 0},  # Базовая логика
            'xgb': {'wins': 0, 'losses': 0, 'total': 0},   # XGBoost
            'rf': {'wins': 0, 'losses': 0, 'total': 0},    # RandomForest
            'arf': {'wins': 0, 'losses': 0, 'total': 0},   # AdaptiveRF
            'nn': {'wins': 0, 'losses': 0, 'total': 0},    # NeuralNet
            'meta': {'wins': 0, 'losses': 0, 'total': 0}   # META
        }
        # Восстанавливаем статистику из истории закрытых позиций
        self._restore_expert_stats()

        # Track last trade time for adaptive threshold
        self.last_trade_time = None

        # ════════════════════════════════════════════════════════════════════════
        # WATCHDOG: Защита от зависаний
        # ════════════════════════════════════════════════════════════════════════
        self._last_activity = time.time()
        self._watchdog_enabled = True
        self._watchdog_thread = threading.Thread(target=self._watchdog_monitor, daemon=True)
        self._activity_lock = threading.Lock()

        print("\n✅ Bot initialized successfully\n")
        print("="*80)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Запускаем watchdog thread
        self._watchdog_thread.start()
        print("[WATCHDOG] Started (timeout: 600s)", flush=True)

        # Send bot started notification
        if config.TELEGRAM_ALERT_TYPES.get('bot_started', True):
            self.telegram.notify_bot_started(
                paper_mode=self.paper_mode,
                initial_capital=config.PAPER_INITIAL_BALANCE if self.paper_mode else 0.0,
                max_positions=config.MAX_POSITIONS
            )

        # ✅ Регистрация обработчиков команд Telegram
        self.telegram.register_command_handler('status', self._handle_status_command)
        self.telegram.register_command_handler('ai', self._handle_ai_command)
        self.telegram.register_command_handler('closeall', self._handle_closeall_command)

        # Устанавливаем обработчик закрытия всех позиций
        self.telegram.closeall_handler = self.close_all_positions

        self.telegram.start_command_listener()
        logger.info("Telegram command listener started")

    def _update_activity(self):
        """Обновляет timestamp последней активности для watchdog"""
        with self._activity_lock:
            self._last_activity = time.time()

    def _watchdog_monitor(self):
        """
        Watchdog thread для мониторинга зависаний

        Если бот не проявляет активность более 10 минут (600 сек),
        выводит предупреждение и может предпринять действия.
        """
        TIMEOUT_SECONDS = 600  # 10 минут
        CHECK_INTERVAL = 30    # Проверка каждые 30 секунд

        logger.info("[WATCHDOG] Monitoring thread started")

        while self._watchdog_enabled:
            try:
                time.sleep(CHECK_INTERVAL)

                with self._activity_lock:
                    idle_time = time.time() - self._last_activity

                if idle_time > TIMEOUT_SECONDS:
                    # Бот завис!
                    logger.error(f"[WATCHDOG] ⚠️ BOT APPEARS FROZEN! No activity for {idle_time:.0f}s")
                    print(f"\n{'='*80}", flush=True)
                    print(f"[WATCHDOG] ⚠️ WARNING: Bot appears frozen!", flush=True)
                    print(f"[WATCHDOG] No activity detected for {idle_time:.0f} seconds", flush=True)
                    print(f"[WATCHDOG] Last activity: {datetime.fromtimestamp(self._last_activity)}", flush=True)
                    print(f"{'='*80}\n", flush=True)

                    # Обновляем активность чтобы не спамить
                    self._update_activity()

                    # Можно добавить более агрессивные действия:
                    # - Отправка Telegram уведомления
                    # - Принудительный перезапуск операции
                    # - Завершение с ошибкой

            except Exception as e:
                logger.error(f"[WATCHDOG] Error in watchdog thread: {e}")

        logger.info("[WATCHDOG] Monitoring thread stopped")

    def _load_experts(self) -> Dict:
        """Загружает ML экспертов (с поддержкой Transfer Learning)"""
        experts = {}

        # ========== PURE ONLINE LEARNING: Создаем все эксперты с нуля ==========
        print("    🎯 PURE ONLINE LEARNING MODE")
        print("    ⚠️  All experts start from scratch and learn during live trading!")

        # XGBoost - online learning с partial_fit
        xgb_path = config.MODELS_DIR / 'saved' / 'xgb_expert_online.pkl'
        if xgb_path.exists():
            try:
                experts['xgb'] = XGBoostExpert.load(str(xgb_path))
                print(f"    ✓ XGBoost (online, {experts['xgb'].train_samples} samples)")
            except Exception as e:
                print(f"    ⚠ XGBoost load failed: {e}, creating new")
                experts['xgb'] = XGBoostExpert(n_estimators=10, max_depth=5, learning_rate=0.1, random_state=42)
                print(f"    ✓ XGBoost (new, online learning)")
        else:
            experts['xgb'] = XGBoostExpert(n_estimators=10, max_depth=5, learning_rate=0.1, random_state=42)
            print(f"    ✓ XGBoost (new, online learning)")

        # Random Forest - заменяем на второй ARF (sklearn RF не поддерживает online)
        rf_path = config.MODELS_DIR / 'saved' / 'rf_expert_online.pkl'
        if rf_path.exists():
            try:
                experts['rf'] = AdaptiveRFExpert.load(str(rf_path))
                print(f"    ✓ RandomForest/ARF (online, {experts['rf'].train_samples} samples)")
            except Exception as e:
                print(f"    ⚠ RandomForest/ARF load failed: {e}, creating new")
                experts['rf'] = AdaptiveRFExpert(n_models=15, random_state=123)
                print(f"    ✓ RandomForest/ARF (new, online learning)")
        else:
            experts['rf'] = AdaptiveRFExpert(n_models=15, random_state=123)
            print(f"    ✓ RandomForest/ARF (new, online learning)")

        # Neural Network - online learning с partial_fit
        nn_path = config.MODELS_DIR / 'saved' / 'nn_expert_online.pkl'
        if nn_path.exists():
            try:
                experts['nn'] = NeuralNetworkExpert.load(str(nn_path))
                print(f"    ✓ NeuralNet (online, {experts['nn'].train_epochs} epochs)")
            except Exception as e:
                print(f"    ⚠ NeuralNet load failed: {e}, creating new")
                experts['nn'] = NeuralNetworkExpert(input_dim=68, dropout=0.2, learning_rate=0.001)
                print(f"    ✓ NeuralNet (new, online learning)")
        else:
            experts['nn'] = NeuralNetworkExpert(input_dim=68, dropout=0.2, learning_rate=0.001)
            print(f"    ✓ NeuralNet (new, online learning)")

        # Adaptive RF - третий эксперт с другими параметрами
        arf_path = config.MODELS_DIR / 'saved' / 'arf_expert_online.pkl'
        if arf_path.exists():
            try:
                experts['arf'] = AdaptiveRFExpert.load(str(arf_path))
                print(f"    ✓ AdaptiveRF (online, {experts['arf'].train_samples} samples)")
            except Exception as e:
                print(f"    ⚠ AdaptiveRF load failed: {e}, creating new")
                experts['arf'] = AdaptiveRFExpert(n_models=10, random_state=42)
                print(f"    ✓ AdaptiveRF (new, online learning)")
        else:
            experts['arf'] = AdaptiveRFExpert(n_models=10, random_state=42)
            print(f"    ✓ AdaptiveRF (new, online learning)")

        return experts

    def _load_meta(self):
        """Загружает META модель"""
        try:
            # Создаем экземпляр SimplifiedEnsembleMETA
            meta = SimplifiedEnsembleMETA(cfg=config)
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

    def _restore_expert_stats(self):
        """
        Восстанавливает статистику экспертов из истории закрытых позиций

        Анализирует predictions сохраненные в позициях и обновляет винрейты.
        """
        logger.info("[ExpertStats] Restoring expert statistics from closed positions...")

        for pos in self.position_manager.closed_positions:
            # ВАЖНО: Для LONG pnl>0 означает рост, для SHORT pnl>0 означает ПАДЕНИЕ
            if pos.direction == 'LONG':
                price_went_up = pos.pnl > 0
            else:  # SHORT
                price_went_up = pos.pnl < 0  # SHORT: убыток означает цена выросла

            # Получаем entry_snapshot если есть
            snapshot = getattr(pos, 'entry_snapshot', {}) if hasattr(pos, 'entry_snapshot') else {}
            ml_preds = snapshot.get('ml_predictions', {})

            # ✅ ИСПРАВЛЕНИЕ: Обновляем статистику ML экспертов с учетом их предсказаний
            # ЛОГИКА ОДИНАКОВА для LONG и SHORT, т.к. price_went_up уже учитывает направление
            if ml_preds:
                for expert_name in ['xgb', 'rf', 'arf', 'nn']:
                    if expert_name in ml_preds and expert_name in self.expert_stats:
                        p_up = ml_preds[expert_name]

                        # Эксперт прав если:
                        # - предсказал рост (p_up > 0.5) и цена выросла (price_went_up = True)
                        # - предсказал падение (p_up <= 0.5) и цена упала (price_went_up = False)
                        expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)

                        self.expert_stats[expert_name]['total'] += 1
                        if expert_was_right:
                            self.expert_stats[expert_name]['wins'] += 1
                        else:
                            self.expert_stats[expert_name]['losses'] += 1

            # Базовая логика (основано на финальном решении)
            # ВАЖНО: p_meta в snapshot - это предсказание от BASE логики (ужасное название!)
            p_base = snapshot.get('p_meta', 0.5) if snapshot else 0.5
            if 'base' in self.expert_stats:
                # ✅ ИСПРАВЛЕНИЕ: Логика одинакова для LONG и SHORT, т.к. price_went_up уже учитывает направление
                base_was_right = (p_base > 0.5 and price_went_up) or (p_base <= 0.5 and not price_went_up)

                self.expert_stats['base']['total'] += 1
                if base_was_right:
                    self.expert_stats['base']['wins'] += 1
                else:
                    self.expert_stats['base']['losses'] += 1

            # ✅ META: Восстанавливаем статистику только для позиций в ACTIVE режиме
            if snapshot:
                meta_mode = snapshot.get('meta_mode')
                p_meta_pred = snapshot.get('p_meta_prediction')

                if meta_mode == 'ACTIVE' and p_meta_pred is not None:
                    if 'meta' in self.expert_stats:
                        # ✅ ИСПРАВЛЕНИЕ: Логика одинакова для LONG и SHORT, т.к. price_went_up уже учитывает направление
                        meta_was_right = (p_meta_pred > 0.5 and price_went_up) or (p_meta_pred <= 0.5 and not price_went_up)

                        self.expert_stats['meta']['total'] += 1
                        if meta_was_right:
                            self.expert_stats['meta']['wins'] += 1
                        else:
                            self.expert_stats['meta']['losses'] += 1

        # Логируем восстановленную статистику
        for expert_name, stats in self.expert_stats.items():
            if stats['total'] > 0:
                wr = stats['wins'] / stats['total'] * 100
                logger.info(f"[ExpertStats] {expert_name.upper()}: {wr:.1f}% WR ({stats['wins']}/{stats['total']})")
            else:
                logger.info(f"[ExpertStats] {expert_name.upper()}: No data yet")

    def _get_expert_win_rates(self) -> Dict[str, Dict]:
        """
        Получает текущие винрейты всех экспертов

        Returns:
            Dict с полной статистикой:
            {
                'base': {'win_rate': 0.53, 'wins': 10, 'losses': 9, 'total': 19},
                'xgb': {'win_rate': 0.55, 'wins': 11, 'losses': 9, 'total': 20},
                ...
            }
        """
        win_rates = {}
        for expert_name, stats in self.expert_stats.items():
            if stats['total'] > 0:
                win_rate = stats['wins'] / stats['total']
            else:
                win_rate = 0.0  # Нет данных

            # ✅ Возвращаем полную структуру (не только float!)
            win_rates[expert_name] = {
                'win_rate': win_rate,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'total': stats['total']
            }
        return win_rates

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

            # Adaptive Random Forest (онлайн обучение)
            if self.experts.get('arf') is not None:
                try:
                    p_arf = self.experts['arf'].predict_proba(features.reshape(1, -1))[0]
                    if np.isfinite(p_arf) and 0 <= p_arf <= 1:
                        predictions['arf'] = float(p_arf)
                except Exception as e:
                    logger.debug(f"{symbol}: AdaptiveRF prediction failed: {e}")

            # Neural Network
            if self.experts.get('nn') is not None:
                try:
                    p_nn = self.experts['nn'].predict_proba(features.reshape(1, -1))[0]
                    if np.isfinite(p_nn) and 0 <= p_nn <= 1:
                        predictions['nn'] = float(p_nn)
                except Exception as e:
                    logger.debug(f"{symbol}: NeuralNet prediction failed: {e}")

            # ========== DIVERSITY MONITORING ==========
            # Добавляем предсказания в Diversity Monitor
            if predictions:
                self.diversity_monitor.add_predictions(
                    p_xgb=predictions.get('xgb'),
                    p_rf=predictions.get('rf'),
                    p_arf=predictions.get('arf'),
                    p_nn=predictions.get('nn')
                )

                # Проверяем diversity метрики каждые 20 предсказаний
                if self.diversity_monitor.n_samples % 20 == 0:
                    metrics = self.diversity_monitor.get_metrics()
                    if metrics:
                        # Логируем метрики
                        logger.info(f"[Diversity] {metrics.alert.value}: {metrics.alert_message}")
                        logger.debug(f"  Disagreement={metrics.disagreement:.4f}, "
                                   f"StdDev={metrics.std_dev:.4f}, "
                                   f"MaxCorr={metrics.max_correlation:.4f}")

                        # ========== РЕАКЦИЯ НА АЛЕРТЫ ==========
                        # LOW_DIVERSITY: Эксперты слишком похожи → риск overfitting
                        if metrics.alert == AlertType.LOW_DIVERSITY:
                            logger.warning(f"⚠️ [Diversity] LOW_DIVERSITY detected! "
                                         f"Disagreement={metrics.disagreement:.4f} < 0.05")
                            # Откатываем META в SHADOW режим для защиты
                            if self.meta and self.meta.mode == "ACTIVE":
                                self.meta.mode = "SHADOW"
                                logger.info("→ META: ACTIVE → SHADOW (low diversity protection)")

                        # HIGH_DISAGREEMENT: Высокая uncertainty
                        elif metrics.alert == AlertType.HIGH_DISAGREEMENT:
                            logger.warning(f"⚠️ [Diversity] HIGH_DISAGREEMENT detected! "
                                         f"Disagreement={metrics.disagreement:.4f} > 0.20")
                            # Можно снизить position size, но это делается в стратегии

                        # HIGH_CORRELATION: Избыточность экспертов
                        elif metrics.alert == AlertType.HIGH_CORRELATION:
                            logger.warning(f"⚠️ [Diversity] HIGH_CORRELATION detected! "
                                         f"MaxCorr={metrics.max_correlation:.4f} > 0.95")

        except Exception as e:
            logger.error(f"Error collecting ML predictions for {symbol}: {e}")

        return predictions

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        """Главный торговый цикл"""

        print(f"\n{'='*80}", flush=True)
        print(f"🚀 STARTING BINANCE TRADING BOT", flush=True)
        print(f"{'='*80}\n", flush=True)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

        last_4h_check = 0
        last_5m_check = 0

        # ✅ ЗАЩИТА ОТ INFINITE LOOP
        MAX_CONSECUTIVE_ERRORS = 10
        consecutive_errors = 0

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

                # ✅ Сброс счётчика ошибок при успехе
                consecutive_errors = 0

            except KeyboardInterrupt:
                print("\n\n⚠️  Stopping bot...")
                self.shutdown()
                break

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"ERROR in main loop ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}", exc_info=True)
                print(f"\n❌ ERROR ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")

                # ✅ КРИТИЧЕСКАЯ ЗАЩИТА: Останавливаем бота при слишком многих ошибках
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical(f"⚠️  TOO MANY CONSECUTIVE ERRORS ({consecutive_errors}) - STOPPING BOT FOR SAFETY")
                    print(f"\n{'='*80}")
                    print(f"🛑 CRITICAL: {consecutive_errors} consecutive errors detected!")
                    print(f"🛑 Bot stopped to prevent infinite error loop")
                    print(f"🛑 Please check logs and fix the issue before restarting")
                    print(f"{'='*80}\n")

                    # Отправляем Telegram уведомление если доступно
                    if hasattr(self, 'telegram') and self.telegram:
                        try:
                            self.telegram.send_message(
                                f"🛑 BOT STOPPED: {consecutive_errors} consecutive errors\n"
                                f"Last error: {str(e)[:200]}\n"
                                f"Check logs immediately!"
                            )
                        except:
                            pass

                    self.shutdown()
                    break

                # Exponential backoff для ретраев
                sleep_time = min(300, 30 * (2 ** (consecutive_errors - 1)))
                print(f"⏳ Waiting {sleep_time}s before retry...")
                time.sleep(sleep_time)

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
        # Обновляем активность в начале ребалансировки
        self._update_activity()

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
        print("Fetching all USDT pairs...", flush=True)
        all_pairs = self.exchange.get_all_usdt_pairs()
        print(f"  Found {len(all_pairs)} pairs", flush=True)

        # Отбираем топ-10
        print("\nSelecting top-10 opportunities...", flush=True)

        # Adaptive threshold
        p_threshold = get_adaptive_threshold(
            total_closed=self.total_closed_trades,
            recent_hour=self.stats['recent_hour_trades'],
            recent_wr=self.stats['win_rate_last_100'],
            calib_error=self.stats['calibration_error'],
            last_trade_time=self.last_trade_time
        )

        print(f"  Adaptive threshold: {p_threshold:.4f}", flush=True)

        # ═══════════════════════════════════════════════════════════════════════
        # ОПТИМИЗАЦИЯ: Один проход для топ-20, экономия ~30-60 минут!
        # ═══════════════════════════════════════════════════════════════════════
        # ✅ ОПТИМИЗАЦИЯ: Предзагрузка OHLCV данных параллельно (ускорение 8-10x)
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n📥 Fetching OHLCV data in parallel for {len(all_pairs)} pairs...", flush=True)
        ohlcv_cache = {}

        # Подготовка batch запросов
        batch_requests = []
        for symbol in all_pairs:
            batch_requests.append((symbol, '5m', 200))
            batch_requests.append((symbol, '15m', 200))
            batch_requests.append((symbol, '30m', 200))
            batch_requests.append((symbol, '4h', 100))

        # Параллельное получение данных (8-10x ускорение)
        import time as time_module
        start_time = time_module.time()

        if hasattr(self.exchange, 'get_ohlcv_batch'):
            # Используем batch метод если доступен
            ohlcv_cache = self.exchange.get_ohlcv_batch(batch_requests, max_workers=10)
            elapsed = time_module.time() - start_time
            print(f"✅ OHLCV data fetched in {elapsed:.1f}s (parallel mode)", flush=True)
        else:
            # Fallback на синхронный режим
            print(f"⚠️  Batch mode unavailable, using sequential fetch", flush=True)

        # Создаём обёрнутую функцию get_ohlcv с кешем
        def get_ohlcv_cached(symbol, timeframe, limit):
            """Обёртка которая использует предзагруженные данные"""
            if symbol in ohlcv_cache and timeframe in ohlcv_cache[symbol]:
                return ohlcv_cache[symbol][timeframe]
            else:
                # Fallback на прямой вызов если нет в кеше
                return self.exchange.get_ohlcv(symbol, timeframe, limit)

        # ═══════════════════════════════════════════════════════════════════════
        # Сканируем рынок ОДИН РАЗ для получения топ-20 возможностей
        # Затем используем:
        #   - Все 20 для решений о закрытии существующих позиций
        #   - Только первые 10 для открытия новых позиций
        # ═══════════════════════════════════════════════════════════════════════
        top_20_opportunities = self.coin_selector.select_top_coins(
            all_pairs=all_pairs,
            get_ticker_func=lambda s: self.exchange.get_ticker(s),
            get_ohlcv_func=get_ohlcv_cached,  # ✅ Используем кешированную версию
            feature_builder=self.feature_builder,
            calculate_atr_func=calculate_atr,
            calculate_phase_func=detect_market_phase,
            get_predictions_func=self._get_predictions,  # Базовая логика для торговли
            collect_ml_predictions_func=self._collect_ml_predictions_for_meta,  # ML для META
            top_n=20,  # Получаем топ-20 за один проход
            exchange=self.exchange,  # Для получения funding rate
            meta_model=self.meta  # ✅ Передаем META для ACTIVE режима
        )

        print(f"\n  Top-20 opportunities (for closure decisions):", flush=True)
        for i, opp in enumerate(top_20_opportunities, 1):
            marker = "✓" if i <= 10 else " "  # Отмечаем первые 10
            print(f"    {marker} {i:2d}. {opp['symbol']:<12} {opp['direction']:<6} "
                  f"p_up={opp['p_up']:.3f}  EV={opp['ev']:.4f}", flush=True)

        # Обновляем активность после скрининга
        self._update_activity()

        # ═══════════════════════════════════════════════════════════════════════
        # HAIKU 4.5: Фундаментальный анализ TOP-20 для формирования финального TOP-10
        # ═══════════════════════════════════════════════════════════════════════
        if self.haiku_analyzer is not None:
            print(f"\n🤖 Running Haiku 4.5 fundamental analysis on TOP-20...", flush=True)
            self._update_activity()  # Перед запуском Haiku
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
                    haiku_score = opp.get('haiku_score', 0.5)
                    haiku_reason = opp.get('haiku_reason', 'N/A')
                    # Форматируем haiku_score правильно
                    fund_str = f"{haiku_score:.3f}" if isinstance(haiku_score, (int, float)) else str(haiku_score)
                    print(f"    {i:2d}. {opp['symbol']:<12} {opp['direction']:<6} "
                          f"tech={opp['p_up']:.3f} fund={fund_str} "
                          f"EV={opp['ev']:.4f} ({haiku_reason})")

                # Используем обогащённый список для открытия позиций
                top_10_for_opening = top_10_enriched

                # Обновляем активность после Haiku
                self._update_activity()

            except Exception as e:
                logger.error(f"Haiku analysis failed: {e}")
                print(f"  ⚠️  Haiku analysis failed: {e}")
                print(f"  Fallback: using technical analysis only")
                # Fallback: используем первые 10 из TOP-20
                top_10_for_opening = top_20_opportunities[:10]
                self._update_activity()
        else:
            # Haiku отключён - используем первые 10 из TOP-20
            top_10_for_opening = top_20_opportunities[:10]
            self._update_activity()

        print(f"\n  Checking if existing positions should be closed (using top-20)...", flush=True)

        # ═══════════════════════════════════════════════════════════════════════
        # СТРАТЕГИЯ: Позиции закрываются ТОЛЬКО по TP/SL
        # Не закрываем позиции принудительно при выпадении из TOP-20
        # Trailing stop и TP/SL ордера управляют выходами
        # ═══════════════════════════════════════════════════════════════════════
        positions_before_close = len(self.position_manager.get_all_open())
        closed_count = 0
        print(f"  Strategy: Positions close on TP/SL only - no manual closure during rebalance", flush=True)

        # ═══════════════════════════════════════════════════════════════════════
        # КРИТИЧНО: Вычисляем торговый капитал ОДИН РАЗ для всех позиций
        # Kelly будет рассчитываться от ПОЛНОГО баланса, а не от остатка
        # ═══════════════════════════════════════════════════════════════════════
        # ФЬЮЧЕРСЫ: Для reinvestment используем Equity (не Balance!)
        # Balance не меняется в фьючерсах, поэтому нужен Equity
        if self.paper_mode:
            current_equity = self.exchange.get_equity()
        else:
            current_equity = self.exchange.get_balance('USDT')

        # Apply reinvestment if enabled (paper trading only)
        if self.paper_mode and self.reinvestment and self.reinvestment.enabled:
            reinvest_result = self.reinvestment.calculate_daily_reinvestment(current_equity)
            initial_trading_capital = reinvest_result['trading_capital']

            # Log reserve fund info
            if reinvest_result['reinvested']:
                print(f"\n  [Reinvest] Daily profit: ${reinvest_result['daily_profit']:.2f}")
                print(f"  [Reinvest] To reserve: ${reinvest_result['to_reserve']:.2f}")
                print(f"  [Reinvest] Trading capital: ${initial_trading_capital:.2f}")
                print(f"  [Reinvest] Reserve fund: ${reinvest_result['reserve_fund']:.2f}")
        else:
            initial_trading_capital = current_equity

        # Открываем новые позиции (используем обогащённый Haiku топ-10)
        print(f"\n  Opening new positions (using top-10 from Haiku analysis)...", flush=True)
        positions_before_open = len(self.position_manager.get_all_open())
        opened_count = 0

        for opp in top_10_for_opening:
            if self.position_manager.has_position(opp['symbol']):
                continue  # Уже открыта

            # Проверяем порог
            # ✅ ИСПРАВЛЕНО: p_up теперь уже инвертирован для SHORT в coin_selector
            # Поэтому проверка одинакова для обоих направлений
            if opp['p_up'] <= p_threshold:
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
                # Fallback: вычисляем из текущего Equity (для фьючерсов)
                # Balance не меняется в фьючерсах, поэтому используем Equity
                if self.paper_mode:
                    # ФЬЮЧЕРСЫ: Используем Equity, но НЕ превышаем свободную маржу
                    # Это важно если есть открытые позиции с unrealized PnL
                    equity = self.exchange.get_equity()
                    free_margin = self.exchange.get_free_margin(leverage=config.LEVERAGE)

                    # Используем минимум из equity и free_margin
                    # чтобы не превысить доступные средства
                    capital = min(equity, free_margin + equity * 0.1)  # +10% буфер от equity
                else:
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

            # ═══════════════════════════════════════════════════════════════
            # ФЬЮЧЕРСЫ: Проверяем достаточность маржи ПЕРЕД открытием
            # ═══════════════════════════════════════════════════════════════
            if self.paper_mode:
                free_margin = self.exchange.get_free_margin(leverage=config.LEVERAGE)

                if position_size_usd > free_margin:
                    logger.warning(
                        f"Insufficient margin for {symbol}: "
                        f"need ${position_size_usd:.2f}, have ${free_margin:.2f}. "
                        f"Reducing position size to ${free_margin * 0.95:.2f} (95% of free margin)"
                    )
                    position_size_usd = free_margin * 0.95  # 95% для безопасности

                    # Если даже 95% слишком мало, пропускаем
                    if position_size_usd < config.MIN_POSITION_SIZE_USDT:
                        logger.warning(f"Position size too small after margin check, skipping {symbol}")
                        return False

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
                'base_signals': opportunity.get('base_signals', {}),  # ✅ BASE сигналы [M, S, B, R] для калибровки
                'p_meta_prediction': opportunity.get('p_meta_prediction'),  # ✅ Предсказание META (если ACTIVE)
                'meta_mode': self.meta.mode if self.meta is not None else None  # ✅ Режим META на момент входа
            }

            # Открываем позицию на бирже (фьючерсная логика)
            # ВАЖНО: exchange.open_position() сам создаст ордера и вернет position_data
            exchange_position = self.exchange.open_position(
                symbol=symbol,
                side=direction,  # 'LONG' или 'SHORT'
                amount=amount,
                entry_price=current_price,
                tp_price=tp_price,
                sl_price=sl_price,
                metadata={'entry_snapshot': entry_snapshot},
                leverage=config.LEVERAGE  # Передаем плечо для проверки маржи
            )

            # Создаем Position объект с данными из биржи
            position = Position(
                symbol=symbol,
                direction=direction,
                entry_time=datetime.now(),
                entry_price=exchange_position['entry_price'],
                amount=amount,
                position_value=position_size_usd,
                tp_price=tp_price,
                sl_price=sl_price,
                atr_value=atr,
                entry_snapshot=entry_snapshot,  # ✅ SNAPSHOT SAVED
                status='OPEN',
                entry_order_id=exchange_position['entry_order_id'],
                tp_order_id=exchange_position.get('tp_order_id'),
                sl_order_id=exchange_position.get('sl_order_id'),
                exchange_position_id=exchange_position['id']  # ✅ ID позиции в PaperExchange
            )

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
                    'atr': opportunity.get('atr', 0.0),  # ATR в долях для расчета EV%
                    'predictions': opportunity.get('predictions', {})
                }

                # ПРАВИЛЬНЫЙ расчет капитала ПОСЛЕ открытия позиции
                if self.paper_mode:
                    # Для детализации в уведомлениях и логах
                    current_free_balance = self.exchange.get_balance('USDT')

                    # ФЬЮЧЕРСЫ: Вычисляем нереализованный PnL всех открытых позиций
                    total_unrealized_pnl = 0.0
                    for pos in self.position_manager.get_all_open():
                        try:
                            ticker = self.exchange.get_ticker(pos.symbol)
                            current_price = ticker['lastPrice']
                            pnl_usdt, _ = pos.calculate_pnl(current_price)
                            total_unrealized_pnl += pnl_usdt
                        except Exception as e:
                            logger.warning(f"Failed to calculate PnL for {pos.symbol}: {e}")
                            continue

                    # Резервный фонд
                    current_reserve = self.reinvestment.reserve_fund if self.reinvestment else 0.0

                    # Общий капитал = баланс + unrealized PnL + резерв (правильная формула для фьючерсов)
                    actual_total_equity = current_free_balance + total_unrealized_pnl + current_reserve

                    # Используемая маржа (для информации в уведомлениях)
                    locked_in_positions = self.exchange.get_used_margin(leverage=config.LEVERAGE)

                    # Записываем snapshot с правильными данными для фьючерсов
                    if self.capital_tracker:
                        self.capital_tracker.record_snapshot(
                            free_balance=current_free_balance,
                            unrealized_pnl=total_unrealized_pnl,
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
                            # close_position() сам вызовет exchange.create_market_order()
                            self.close_position(position, 'TP', position.tp_price)
                            continue
                        elif current_price <= position.sl_price:
                            # close_position() сам вызовет exchange.create_market_order()
                            self.close_position(position, 'SL', position.sl_price)
                            continue
                    else:  # SHORT
                        if current_price <= position.tp_price:
                            # close_position() сам вызовет exchange.create_market_order()
                            self.close_position(position, 'TP', position.tp_price)
                            continue
                        elif current_price >= position.sl_price:
                            # close_position() сам вызовет exchange.create_market_order()
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

                    # ФЬЮЧЕРСЫ: Также обновляем sl_order_id в позиции PaperExchange
                    if position.exchange_position_id and position.exchange_position_id in self.exchange.positions:
                        self.exchange.positions[position.exchange_position_id]['sl_order_id'] = sl_order['id']

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

        # HEARTBEAT: Записываем snapshot капитала каждые 5 минут
        # Это решает проблему "дырявых" snapshot'ов когда позиции не открываются часами
        if self.paper_mode and self.capital_tracker:
            try:
                current_free_balance = self.exchange.get_balance('USDT')
                current_reserve = self.reinvestment.reserve_fund if self.reinvestment else 0.0

                # ФЬЮЧЕРСЫ: Вычисляем нереализованный PnL всех открытых позиций
                total_unrealized_pnl = 0.0
                for pos in self.position_manager.get_all_open():
                    try:
                        ticker = self.exchange.get_ticker(pos.symbol)
                        current_price = ticker['lastPrice']
                        pnl_usdt, _ = pos.calculate_pnl(current_price)
                        total_unrealized_pnl += pnl_usdt
                    except Exception as e:
                        logger.warning(f"Failed to calculate PnL for {pos.symbol}: {e}")
                        continue

                # Общий капитал = баланс + unrealized PnL + резерв
                current_equity = current_free_balance + total_unrealized_pnl + current_reserve

                self.capital_tracker.record_snapshot(
                    free_balance=current_free_balance,
                    unrealized_pnl=total_unrealized_pnl,
                    reserve_fund=current_reserve,
                    num_open_positions=len(self.position_manager.get_all_open())
                )
                logger.debug(f"Heartbeat snapshot recorded: equity=${current_equity:.2f}")
            except Exception as e:
                logger.warning(f"Failed to record heartbeat snapshot: {e}")

    def close_position(self, position: Position, exit_reason: str, exit_price: float):
        """
        Закрывает позицию и обучает модели

        КРИТИЧНО: Использует СОХРАНЕННЫЕ фичи из entry_snapshot (t0)
        """

        # ════════════════════════════════════════════════════════════════
        # ЗАЩИТА: Проверяем, что позиция еще не закрыта
        # ════════════════════════════════════════════════════════════════
        if position.status == 'CLOSED':
            logger.warning(f"Position {position.symbol} is already CLOSED, skipping duplicate close")
            return

        logger.info(f"[close_position] Called for {position.symbol} {position.direction}, "
                   f"reason={exit_reason}, exchange_position_id={position.exchange_position_id}")

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

                # Определяем фактический результат (цена выросла или упала)
                # ВАЖНО: Для LONG pnl>0 означает рост, для SHORT pnl>0 означает ПАДЕНИЕ
                if position.direction == 'LONG':
                    price_went_up = position.pnl > 0
                else:  # SHORT
                    price_went_up = position.pnl < 0  # SHORT: убыток означает цена выросла

                # ✅ ИСПРАВЛЕНИЕ: Обновляем статистику экспертов с учетом их ПРЕДСКАЗАНИЙ
                # Каждый эксперт дает свое p_up, нужно проверять правильность его прогноза
                # ЛОГИКА ОДИНАКОВА для LONG и SHORT, т.к. price_went_up уже учитывает направление
                for expert_name in ['xgb', 'rf', 'arf', 'nn']:
                    if expert_name in ml_preds and expert_name in self.expert_stats:
                        p_up = ml_preds[expert_name]  # Предсказание эксперта (0-1)

                        # Эксперт прав если:
                        # - предсказал рост (p_up > 0.5) и цена выросла (price_went_up = True)
                        # - предсказал падение (p_up <= 0.5) и цена упала (price_went_up = False)
                        expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)

                        self.expert_stats[expert_name]['total'] += 1
                        if expert_was_right:
                            self.expert_stats[expert_name]['wins'] += 1
                        else:
                            self.expert_stats[expert_name]['losses'] += 1

                # Обновляем статистику BASE логики (основано на финальном решении)
                # ВАЖНО: p_meta в snapshot - это предсказание от BASE логики (ужасное название!)
                p_base = snapshot.get('p_meta', 0.5)  # Финальное предсказание базовой логики
                if 'base' in self.expert_stats:
                    # ✅ ИСПРАВЛЕНИЕ: Логика одинакова для LONG и SHORT, т.к. price_went_up уже учитывает направление
                    base_was_right = (p_base > 0.5 and price_went_up) or (p_base <= 0.5 and not price_went_up)

                    self.expert_stats['base']['total'] += 1
                    if base_was_right:
                        self.expert_stats['base']['wins'] += 1
                    else:
                        self.expert_stats['base']['losses'] += 1

                # ✅ META: Оцениваем только если была в ACTIVE режиме при открытии позиции
                meta_mode = snapshot.get('meta_mode')
                p_meta_pred = snapshot.get('p_meta_prediction')

                if meta_mode == 'ACTIVE' and p_meta_pred is not None:
                    # META принимала решение - оцениваем её предсказание
                    if 'meta' in self.expert_stats:
                        # ✅ ИСПРАВЛЕНИЕ: Логика одинакова для LONG и SHORT, т.к. price_went_up уже учитывает направление
                        meta_was_right = (p_meta_pred > 0.5 and price_went_up) or (p_meta_pred <= 0.5 and not price_went_up)

                        self.expert_stats['meta']['total'] += 1
                        if meta_was_right:
                            self.expert_stats['meta']['wins'] += 1
                        else:
                            self.expert_stats['meta']['losses'] += 1
                # В SHADOW режиме не обновляем статистику META

                # Для META обучения используем старую логику (y_up основан на PnL)
                y_up = 1 if price_went_up else 0

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
                    p_arf=ml_preds.get('arf'),  # ✅ Включен ARF
                    p_nn=ml_preds.get('nn'),
                    p_base=snapshot.get('p_meta'),  # Предсказание базовой логики
                    y_up=y_up,
                    used_in_live=True,
                    p_final_used=snapshot.get('p_meta'),
                    reg_ctx=reg_ctx  # ✅ ПЕРЕДАЕМ РЕАЛЬНЫЙ КОНТЕКСТ из t0
                )
                logger.debug(f"META.record_result() called for {position.symbol}: y_up={y_up}, context={reg_ctx}")

                # ✅ ONLINE LEARNING: Train all experts with real outcome
                features = snapshot.get('features')
                if features is not None and hasattr(features, 'reshape'):
                    features_2d = features.reshape(1, -1) if features.ndim == 1 else features
                    y_outcome = np.array([y_up])

                    # XGBoost online learning (с ограничением деревьев)
                    if 'xgb' in self.experts and self.experts['xgb'] is not None:
                        try:
                            self.experts['xgb'].partial_fit(features_2d, y_outcome, n_new_trees=2, max_total_trees=200)
                            logger.debug(f"XGBoost trained on {position.symbol}: y_up={y_up}, trees={self.experts['xgb'].model.num_boosted_rounds()}")
                        except Exception as e:
                            logger.error(f"Error training XGBoost: {e}")

                    # RandomForest/ARF online learning
                    if 'rf' in self.experts and self.experts['rf'] is not None:
                        try:
                            self.experts['rf'].partial_fit(features_2d, y_outcome)
                            logger.debug(f"RandomForest/ARF trained on {position.symbol}: y_up={y_up}, total_samples={self.experts['rf'].train_samples}")
                        except Exception as e:
                            logger.error(f"Error training RandomForest/ARF: {e}")

                    # AdaptiveRF online learning
                    if 'arf' in self.experts and self.experts['arf'] is not None:
                        try:
                            self.experts['arf'].partial_fit(features_2d, y_outcome)
                            logger.debug(f"AdaptiveRF trained on {position.symbol}: y_up={y_up}, total_samples={self.experts['arf'].train_samples}")
                        except Exception as e:
                            logger.error(f"Error training AdaptiveRF: {e}")

                    # NeuralNetwork online learning
                    if 'nn' in self.experts and self.experts['nn'] is not None:
                        try:
                            self.experts['nn'].partial_fit(features_2d, y_outcome, n_epochs=1)
                            logger.debug(f"NeuralNetwork trained on {position.symbol}: y_up={y_up}, epochs={self.experts['nn'].train_epochs}")
                        except Exception as e:
                            logger.error(f"Error training NeuralNetwork: {e}")

                    # Периодическое сохранение всех моделей (каждые 5 сделок)
                    if self.total_closed_trades % 5 == 0:
                        try:
                            if self.experts.get('xgb'):
                                xgb_path = config.MODELS_DIR / 'saved' / 'xgb_expert_online.pkl'
                                self.experts['xgb'].save(str(xgb_path))
                                logger.info(f"[XGBoost] Saved model")
                        except Exception as e:
                            logger.error(f"Error saving XGBoost: {e}")

                        try:
                            if self.experts.get('rf'):
                                rf_path = config.MODELS_DIR / 'saved' / 'rf_expert_online.pkl'
                                self.experts['rf'].save(str(rf_path))
                                logger.info(f"[RandomForest/ARF] Saved model")
                        except Exception as e:
                            logger.error(f"Error saving RandomForest/ARF: {e}")

                        try:
                            if self.experts.get('arf'):
                                arf_path = config.MODELS_DIR / 'saved' / 'arf_expert_online.pkl'
                                self.experts['arf'].save(str(arf_path))
                                logger.info(f"[AdaptiveRF] Saved model")
                        except Exception as e:
                            logger.error(f"Error saving AdaptiveRF: {e}")

                        try:
                            if self.experts.get('nn'):
                                nn_path = config.MODELS_DIR / 'saved' / 'nn_expert_online.pkl'
                                self.experts['nn'].save(str(nn_path))
                                logger.info(f"[NeuralNetwork] Saved model")
                        except Exception as e:
                            logger.error(f"Error saving NeuralNetwork: {e}")
                else:
                    logger.warning(f"Expert training skipped: features not found in snapshot for {position.symbol}")

        except Exception as e:
            logger.error(f"Error recording result to META: {e}")

        # ═══════════════════════════════════════════════════════════════════════
        # КРИТИЧНО: Закрываем позицию в exchange (ФЬЮЧЕРСНАЯ ЛОГИКА)
        # exchange.close_position() сам создаст market order И добавит PnL в баланс
        # ═══════════════════════════════════════════════════════════════════════
        if self.paper_mode and position.exchange_position_id:
            try:
                # ЗАЩИТА: Проверяем, что позиция все еще существует в exchange
                if position.exchange_position_id not in self.exchange.positions:
                    logger.warning(f"Position {position.symbol} ({position.exchange_position_id}) "
                                 f"already closed in exchange, skipping")
                else:
                    # Закрываем позицию в PaperExchange (фьючерсная логика)
                    # Метод close_position() сам:
                    # 1. Создаст closing market order
                    # 2. Рассчитает PnL
                    # 3. Добавит PnL в balance
                    closed_exchange_position = self.exchange.close_position(
                        position_id=position.exchange_position_id,
                        reason=exit_reason.lower()
                    )
                    logger.debug(f"Closed position in PaperExchange: {closed_exchange_position.get('pnl_after_commission', 0):.2f} USDT")
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
            # Получаем win rates экспертов
            win_rates = self._get_expert_win_rates()

            position_data = {
                'symbol': position.symbol,
                'direction': position.direction,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'position_size': position.position_value,
                'pnl': position.pnl,
                'pnl_percent': position.pnl_pct,
                'exit_reason': exit_reason,
                'holding_time': f"{duration:.1f}h",
                'win_rates': win_rates  # ✅ Добавляем win rates экспертов
            }
            self.telegram.notify_position_closed(position_data)

    def close_position_manual(self, position: Position):
        """Ручное закрытие позиции (для таймаута или замены)"""
        try:
            current_price = self.exchange.get_current_price(position.symbol)

            # close_position() сам вызовет exchange.close_position(), который отменит TP/SL ордера
            self.close_position(position, 'MANUAL', current_price)

        except Exception as e:
            logger.error(f"Error closing position manually {position.symbol}: {e}", exc_info=True)

    # ========================================================================
    # TELEGRAM COMMAND HANDLERS
    # ========================================================================

    def _handle_status_command(self) -> str:
        """
        Обработчик команды /status

        Собирает и возвращает детальную статистику по работе бота:
        - Открытые позиции с нереализованным PnL
        - Баланс и капитал
        - Торговую статистику

        Returns:
            str: Сообщение для отправки в Telegram (уже не нужно, метод вызывает notify_status_report)
        """
        try:
            # Собираем данные об открытых позициях с нереализованным PnL
            open_positions = self.position_manager.get_all_open()
            open_positions_data = []
            total_unrealized_pnl = 0.0
            total_position_value = 0.0

            for pos in open_positions:
                try:
                    # Получаем текущую цену
                    ticker = self.exchange.get_ticker(pos.symbol)
                    current_price = ticker['lastPrice']  # Правильный ключ!

                    # Вычисляем нереализованный PnL
                    pnl_usdt, pnl_pct = pos.calculate_pnl(current_price)

                    # Форматируем время удержания
                    duration_sec = pos.get_duration()
                    hours = int(duration_sec / 3600)
                    minutes = int((duration_sec % 3600) / 60)
                    duration_str = f"{hours}h {minutes}m"

                    open_positions_data.append({
                        'symbol': pos.symbol,
                        'direction': pos.direction,
                        'entry_price': pos.entry_price,
                        'current_price': current_price,
                        'pnl_usdt': pnl_usdt,
                        'pnl_pct': pnl_pct,
                        'position_value': pos.position_value,
                        'tp_price': pos.tp_price,
                        'sl_price': pos.sl_price,
                        'duration': duration_str
                    })

                    total_unrealized_pnl += pnl_usdt
                    total_position_value += pos.position_value

                except Exception as e:
                    logger.error(f"Error calculating PnL for {pos.symbol}: {e}")
                    continue

            # Процент нереализованного PnL
            total_unrealized_pnl_pct = (total_unrealized_pnl / total_position_value * 100) if total_position_value > 0 else 0.0

            # Получаем баланс и капитал
            try:
                current_free_balance = self.exchange.get_balance()
                locked_in_positions = self.exchange.get_used_margin(leverage=config.LEVERAGE)

                # Резервный фонд (если есть)
                if hasattr(self, 'reserve_fund') and self.reserve_fund:
                    current_reserve = self.reserve_fund.get_current_reserve()
                else:
                    current_reserve = 0.0

                # ФЬЮЧЕРСЫ: Правильная формула для фьючерсной торговли
                # При открытии позиции баланс НЕ меняется, только блокируется маржа
                # Капитал = баланс + нереализованный PnL + резерв
                total_equity = current_free_balance + total_unrealized_pnl + current_reserve

            except Exception as e:
                logger.error(f"Error getting balance: {e}")
                current_free_balance = 0.0
                locked_in_positions = 0.0
                current_reserve = 0.0
                total_equity = 0.0

            # Изменения за периоды (если есть capital_tracker)
            period_changes = {}
            if hasattr(self, 'capital_tracker') and self.capital_tracker:
                try:
                    changes_1h = self.capital_tracker.get_change_over_period(total_equity, hours_ago=1)
                    changes_24h = self.capital_tracker.get_change_over_period(total_equity, hours_ago=24)
                    changes_7d = self.capital_tracker.get_change_over_period(total_equity, hours_ago=7*24)
                    changes_30d = self.capital_tracker.get_change_over_period(total_equity, hours_ago=30*24)

                    period_changes = {
                        '1ч': changes_1h,
                        '24ч': changes_24h,
                        '7д': changes_7d,
                        '30д': changes_30d
                    }
                except Exception as e:
                    logger.error(f"Error getting period changes: {e}")

            # Торговая статистика
            pnl_summary = self.position_manager.get_pnl_summary()

            total_trades = pnl_summary.get('total_trades', 0)
            win_rate = pnl_summary.get('win_rate', 0.0)
            total_realized_pnl = pnl_summary.get('total_pnl', 0.0)
            avg_win = pnl_summary.get('avg_win', 0.0)
            avg_loss = pnl_summary.get('avg_loss', 0.0)
            best_trade = pnl_summary.get('max_win', 0.0)  # Правильный ключ!
            worst_trade = pnl_summary.get('max_loss', 0.0)  # Правильный ключ!

            # Подсчет закрытых сделок сегодня
            closed_today = 0
            daily_pnl = 0.0
            try:
                from datetime import date
                today = date.today()
                recent_closed = self.position_manager.get_recent_closed(limit=100)
                for pos in recent_closed:
                    if pos.exit_time and pos.exit_time.date() == today:
                        closed_today += 1
                        daily_pnl += pos.pnl
            except Exception as e:
                logger.error(f"Error counting daily trades: {e}")

            # Формируем данные для отправки
            status_data = {
                'open_positions': open_positions_data,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_unrealized_pnl_pct': total_unrealized_pnl_pct,
                'free_balance': current_free_balance,
                'locked_in_positions': locked_in_positions,
                'total_equity': total_equity,
                'reserve_fund': current_reserve,
                'closed_trades_today': closed_today,
                'daily_pnl': daily_pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_realized_pnl': total_realized_pnl,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'period_changes': period_changes
            }

            # Отправляем через notify_status_report
            self.telegram.notify_status_report(status_data)

            # Возвращаем пустую строку (метод уже отправил сообщение)
            return ""

        except Exception as e:
            logger.error(f"Error in status command handler: {e}", exc_info=True)
            return f"❌ Ошибка при получении статуса: {str(e)}"

    def _handle_ai_command(self) -> str:
        """
        Обработчик команды /ai

        Отправляет панель управления AI ассистентом с кнопками:
        - Активировать/Деактивировать AI
        - AI Status (анализ бота через Claude Sonnet 4.5)

        Returns:
            str: Пустая строка (метод уже отправляет сообщение)
        """
        try:
            if not self.ai_assistant:
                return "❌ AI Assistant не настроен. Проверьте ANTHROPIC_API_KEY в .env"

            # Отправляем панель управления
            self.telegram.send_ai_control_panel()

            # Возвращаем пустую строку (метод уже отправил сообщение)
            return ""

        except Exception as e:
            logger.error(f"Error in /ai command: {e}", exc_info=True)
            return f"❌ Ошибка при вызове AI панели: {str(e)[:100]}"

    def _handle_closeall_command(self) -> str:
        """
        Обработчик команды /closeall

        Отправляет панель подтверждения для закрытия всех открытых позиций.

        Returns:
            str: Пустая строка (метод уже отправляет сообщение)
        """
        try:
            # Получаем открытые позиции
            open_positions = self.position_manager.get_all_open()

            if not open_positions:
                return "ℹ️ <b>НЕТ ОТКРЫТЫХ ПОЗИЦИЙ</b>\n\nВсе позиции уже закрыты."

            # Считаем общую стоимость
            total_value = sum(pos.position_value for pos in open_positions)

            # Отправляем панель подтверждения
            self.telegram.send_closeall_confirmation(
                open_positions_count=len(open_positions),
                total_value=total_value
            )

            # Возвращаем пустую строку (метод уже отправил сообщение)
            return ""

        except Exception as e:
            logger.error(f"Error in /closeall command: {e}", exc_info=True)
            return f"❌ Ошибка при вызове closeall: {str(e)[:100]}"

    def close_all_positions(self) -> Dict:
        """
        Закрытие всех открытых позиций (emergency stop)

        Returns:
            Dict: Результат операции с полями:
                - success: bool - успешно ли выполнено
                - closed_count: int - количество закрытых позиций
                - failed_count: int - количество неудачных попыток
                - total_pnl: float - общий PnL
                - details: List[Dict] - детали по каждой позиции
                - error: str (если success=False)
        """
        logger.warning("="*80)
        logger.warning("⚠️ EMERGENCY CLOSE ALL POSITIONS REQUESTED")
        logger.warning("="*80)

        result = {
            "success": False,
            "closed_count": 0,
            "failed_count": 0,
            "total_pnl": 0.0,
            "details": [],
            "error": None
        }

        try:
            # Получаем все открытые позиции
            open_positions = self.position_manager.get_all_open()

            if not open_positions:
                result["success"] = True
                result["error"] = "Нет открытых позиций"
                logger.info("No open positions to close")
                return result

            logger.info(f"Closing {len(open_positions)} open positions...")

            # Закрываем каждую позицию
            for pos in open_positions:
                try:
                    symbol = pos.symbol
                    logger.info(f"Closing position: {symbol} {pos.direction}")

                    # Получаем текущую цену
                    ticker = self.exchange.get_ticker(symbol)
                    current_price = ticker['lastPrice']

                    # Вычисляем PnL
                    pnl_usdt, pnl_pct = pos.calculate_pnl(current_price)

                    # Закрываем позицию через PositionManager
                    # (это обновит статистику и отправит уведомление)
                    self.position_manager.close_position(
                        symbol=symbol,
                        exit_price=current_price,
                        exit_reason="EMERGENCY_CLOSEALL"
                    )

                    result["closed_count"] += 1
                    result["total_pnl"] += pnl_usdt

                    # Добавляем детали
                    result["details"].append({
                        "symbol": symbol,
                        "direction": pos.direction,
                        "entry_price": pos.entry_price,
                        "exit_price": current_price,
                        "pnl": pnl_usdt,
                        "pnl_pct": pnl_pct
                    })

                    logger.info(f"✅ Closed {symbol}: PnL ${pnl_usdt:+.2f} ({pnl_pct:+.2f}%)")

                except Exception as e:
                    logger.error(f"Failed to close position {pos.symbol}: {e}", exc_info=True)
                    result["failed_count"] += 1
                    result["details"].append({
                        "symbol": pos.symbol,
                        "error": str(e)[:100]
                    })

            # Если хотя бы одна позиция закрыта - успех
            if result["closed_count"] > 0:
                result["success"] = True

            logger.warning("="*80)
            logger.warning(f"✅ CLOSEALL COMPLETED: {result['closed_count']} closed, {result['failed_count']} failed")
            logger.warning(f"   Total PnL: ${result['total_pnl']:+.2f}")
            logger.warning("="*80)

            return result

        except Exception as e:
            logger.error(f"Critical error in close_all_positions: {e}", exc_info=True)
            result["success"] = False
            result["error"] = f"Critical error: {str(e)[:200]}"
            return result

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

        # ✅ Останавливаем Telegram command listener
        if hasattr(self, 'telegram') and self.telegram:
            self.telegram.stop_command_listener()
            logger.info("Telegram command listener stopped")

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
            # ════════════════════════════════════════════════════════════════
            # ВАЛИДАЦИЯ БАЛАНСА: Проверяем корректность расчетов
            # ════════════════════════════════════════════════════════════════
            validation_result = self.exchange.validate_balance()

            print("\n" + "="*80)
            print("💰 FINAL BALANCE")
            print("="*80)

            # Используем get_equity() для правильного учета unrealized PnL (если позиции не закрылись)
            current_equity = self.exchange.get_equity()
            current_free_balance = self.exchange.get_balance('USDT')
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
            print(f"  Free balance:     ${current_free_balance:,.2f}")
            print(f"  Current equity:   ${current_equity:,.2f}")
            print(f"  Net change:       ${current_equity - initial_balance:+,.2f}")

            # Если есть резервный фонд
            if self.reinvestment and self.reinvestment.reserve_fund > 0:
                reserve = self.reinvestment.reserve_fund
                total_equity = current_equity + reserve
                print(f"\n  Reserve fund:     ${reserve:,.2f}")
                print(f"  Total equity:     ${total_equity:,.2f}")
                print(f"  Total change:     ${total_equity - initial_balance:+,.2f}")

            # ROI (на основе equity, не balance)
            roi = ((current_equity - initial_balance) / initial_balance) * 100
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

        # Сохранение всех экспертов (online learning models)
        print("\n💾 Saving expert models...")

        if 'xgb' in self.experts and self.experts['xgb'] is not None:
            try:
                xgb_path = config.MODELS_DIR / 'saved' / 'xgb_expert_online.pkl'
                self.experts['xgb'].save(str(xgb_path))
                print(f"✅ XGBoost saved (trees: {self.experts['xgb'].model.num_boosted_rounds()})")
            except Exception as e:
                print(f"⚠️  Failed to save XGBoost: {e}")

        if 'rf' in self.experts and self.experts['rf'] is not None:
            try:
                rf_path = config.MODELS_DIR / 'saved' / 'rf_expert_online.pkl'
                self.experts['rf'].save(str(rf_path))
                print(f"✅ RandomForest/ARF saved ({self.experts['rf'].train_samples} samples)")
            except Exception as e:
                print(f"⚠️  Failed to save RandomForest/ARF: {e}")

        if 'arf' in self.experts and self.experts['arf'] is not None:
            try:
                arf_path = config.MODELS_DIR / 'saved' / 'arf_expert_online.pkl'
                self.experts['arf'].save(str(arf_path))
                print(f"✅ AdaptiveRF saved ({self.experts['arf'].train_samples} samples)")
            except Exception as e:
                print(f"⚠️  Failed to save AdaptiveRF: {e}")

        if 'nn' in self.experts and self.experts['nn'] is not None:
            try:
                nn_path = config.MODELS_DIR / 'saved' / 'nn_expert_online.pkl'
                self.experts['nn'].save(str(nn_path))
                print(f"✅ NeuralNetwork saved ({self.experts['nn'].train_epochs} epochs)")
            except Exception as e:
                print(f"⚠️  Failed to save NeuralNetwork: {e}")

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
