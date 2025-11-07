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

# Добавляем путь к GGG3
sys.path.insert(0, str(Path(__file__).parent))

# Конфиг
import binance_config as config

# Binance API
from binance_api.client import BinanceClient

# Portfolio
from portfolio.position_manager import PositionManager, Position
from portfolio.reinvestment import ReinvestmentManager

# Strategy
from strategy.coin_selector import CoinSelector
from strategy.threshold_adapter import get_adaptive_threshold
from strategy.kelly_sizer import calculate_kelly_position_size

# Features
from features.builder import BinanceFeatureBuilder
from features.base_logic import BaseLogicMultiTF
from features.target_calculator import calculate_atr, detect_market_phase

# Models
from models.experts import XGBoostExpert, RandomForestExpert, NeuralNetworkExpert

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
            self.exchange = PaperExchange(initial_capital=config.PAPER_INITIAL_BALANCE)
            print(f"  Mode: 📄 PAPER TRADING (${config.PAPER_INITIAL_BALANCE})")
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
        else:
            self.reinvestment = None

        # Telegram notifier
        self.telegram = TelegramNotifier(
            bot_token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
            enabled=config.TELEGRAM_ALERTS_ENABLED
        )

        # Statistics
        self.total_closed_trades = len(self.position_manager.closed_positions)
        self.stats = {
            'win_rate_last_100': self._calculate_win_rate(),
            'calibration_error': 0.05,
            'recent_hour_trades': 0
        }

        print("\n✅ Bot initialized successfully\n")
        print("="*80)

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
        meta_path = config.MODELS_DIR / 'saved' / 'meta_cmaes_model.pkl'

        if not meta_path.exists():
            print(f"    ⚠ META not found (будет использован Average)")
            return None

        try:
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)
            print(f"    ✓ META ({meta_data.get('mode', 'UNKNOWN')} mode)")
            return meta_data
        except Exception as e:
            print(f"    ⚠ META failed: {e}")
            return None

    def _calculate_win_rate(self) -> float:
        """Рассчитывает win rate за последние 100 сделок"""
        closed = self.position_manager.closed_positions[-100:]
        if not closed:
            return 0.5

        wins = sum(1 for p in closed if p.get('pnl', 0) > 0)
        return wins / len(closed)

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
                # Получаем процент падения BTC (если доступно)
                btc_drop = 5.0  # TODO: получить реальное значение из black_swan
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
            calib_error=self.stats['calibration_error']
        )

        print(f"  Adaptive threshold: {p_threshold:.4f}")

        top_opportunities = self.coin_selector.select_top_coins(
            all_pairs=all_pairs,
            get_ticker_func=lambda s: self.exchange.get_ticker(s),
            get_ohlcv_func=lambda s, tf, lim: self.exchange.get_ohlcv(s, tf, lim),
            feature_builder=self.feature_builder,
            calculate_atr_func=calculate_atr,
            calculate_phase_func=detect_market_phase,
            top_n=config.TOP_N_COINS
        )

        print(f"\n  Top-{len(top_opportunities)} opportunities:")
        for i, opp in enumerate(top_opportunities, 1):
            print(f"    {i}. {opp['symbol']:<12} {opp['direction']:<6} "
                  f"p_up={opp['p_up']:.3f}  EV={opp['ev']:.4f}")

        # Закрываем позиции не в топ-10
        top_symbols = {opp['symbol'] for opp in top_opportunities}

        for position in list(self.position_manager.get_all_open()):
            if position.symbol not in top_symbols:
                print(f"\n  Closing {position.symbol}: not in top-10 anymore")
                self.close_position_manual(position)

        # Открываем новые позиции
        for opp in top_opportunities:
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

            # Открываем позицию
            self.open_position(opp)

        final_positions = len(self.position_manager.get_all_open())
        print(f"\n  Portfolio status: {final_positions}/{config.MAX_POSITIONS} positions")

        # Send Telegram notification
        if config.TELEGRAM_ALERT_TYPES.get('rebalance', True):
            # Подсчитываем изменения (приблизительно)
            # В идеале нужно считать до и после, но это упрощенная версия
            stats = {
                'closed_positions': 0,  # TODO: track actual count
                'opened_positions': 0,  # TODO: track actual count
                'total_positions': final_positions,
                'top_opportunities': top_opportunities[:5]  # Top 5
            }
            self.telegram.notify_rebalance_stats(stats)

    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================

    def open_position(self, opportunity: dict):
        """
        Открывает позицию с полным snapshot

        КРИТИЧНО: Сохраняет ВСЕ данные на момент входа (t0)
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

            from features.builder import BinanceFeatureBuilder
            atr = BinanceFeatureBuilder.calculate_atr(df_4h, period=config.ATR_PERIOD).iloc[-1]

            # TP/SL уровни
            if direction == 'LONG':
                tp_price = current_price + config.TP_ATR_MULTIPLIER * atr
                sl_price = current_price - config.SL_ATR_MULTIPLIER * atr
            else:  # SHORT
                tp_price = current_price - config.TP_ATR_MULTIPLIER * atr
                sl_price = current_price + config.SL_ATR_MULTIPLIER * atr

            # Размер позиции (Kelly)
            capital = self.exchange.get_balance('USDT')

            # Apply reinvestment if enabled (paper trading only)
            reserve_fund = None
            trading_capital = None
            total_equity = None

            if self.paper_mode and self.reinvestment and self.reinvestment.enabled:
                reinvest_result = self.reinvestment.calculate_daily_reinvestment(capital)
                capital = reinvest_result['trading_capital']
                reserve_fund = reinvest_result['reserve_fund']
                trading_capital = reinvest_result['trading_capital']
                total_equity = reinvest_result['total_equity']

                # Log reserve fund info
                if reinvest_result['reinvested']:
                    print(f"  [Reinvest] Daily profit: ${reinvest_result['daily_profit']:.2f}")
                    print(f"  [Reinvest] To reserve: ${reinvest_result['to_reserve']:.2f}")
                    print(f"  [Reinvest] Trading capital: ${capital:.2f}")
                    print(f"  [Reinvest] Reserve fund: ${reinvest_result['reserve_fund']:.2f}")

            position_size_usd = calculate_kelly_position_size(
                capital=capital,
                ev=opportunity['ev'],
                p_up=opportunity['p_up'],
                recent_wr=self.stats['win_rate_last_100']
            )

            amount = position_size_usd / current_price

            # ✅ КРИТИЧНО: Создаем snapshot на момент входа (t0)
            entry_snapshot = {
                'features_68d': opportunity.get('features', np.zeros(68)),
                'predictions': opportunity.get('predictions', {}),
                'meta_context': opportunity.get('context', {}),
                'p_meta': opportunity['p_up'],
                'timestamp': time.time()
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

                # Добавляем информацию о резервном фонде (если доступно)
                if reserve_fund is not None:
                    position_data['reserve_fund'] = reserve_fund
                if trading_capital is not None:
                    position_data['trading_capital'] = trading_capital
                if total_equity is not None:
                    position_data['total_equity'] = total_equity

                self.telegram.notify_position_opened(position_data)

        except Exception as e:
            logger.error(f"Error opening position {symbol}: {e}", exc_info=True)
            print(f"  ❌ Error: {e}")

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
                            self.close_position(position, 'TP', position.tp_price)
                            continue
                        elif current_price <= position.sl_price:
                            self.close_position(position, 'SL', position.sl_price)
                            continue
                    else:  # SHORT
                        if current_price <= position.tp_price:
                            self.close_position(position, 'TP', position.tp_price)
                            continue
                        elif current_price >= position.sl_price:
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
        # TODO: Реализовать online learning
        # (требуется record_result() методы в экспертах)

        # Закрываем позицию в manager
        self.position_manager.close_position(position.symbol, exit_price, exit_reason)

        # Обновляем статистику
        self.total_closed_trades += 1
        self.stats['win_rate_last_100'] = self._calculate_win_rate()

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

            # Отменяем TP/SL ордера
            if position.tp_order_id:
                self.exchange.cancel_order(position.symbol, position.tp_order_id)
            if position.sl_order_id:
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

    def shutdown(self):
        """Корректное завершение бота"""
        print("\nShutting down...")
        self.position_manager.save_to_file()
        print("✅ Positions saved")

        # Print reinvestment summary if enabled
        if self.paper_mode and self.reinvestment:
            self.reinvestment.print_summary()

        # Send bot stopped notification
        if config.TELEGRAM_ALERT_TYPES.get('bot_stopped', True):
            self.telegram.notify_bot_stopped(reason="Manual shutdown")

        print("✅ Bot stopped")


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
