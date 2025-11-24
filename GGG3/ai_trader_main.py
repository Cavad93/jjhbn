#!/usr/bin/env python3
"""
AI Trading Module - Main Entry Point

Запуск автономного AI трейдера с использованием Claude Sonnet 4.5
"""
import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Загрузка переменных окружения из .env файла (если существует)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[Setup] Loaded environment variables from {env_path}")
    else:
        print(f"[Warning] .env file not found at {env_path}")
except ImportError:
    print("[Warning] python-dotenv not installed - using system environment variables only")
    pass

# Add GGG3 to path
sys.path.insert(0, os.path.dirname(__file__))

from ai_trading_module.config import AITradingConfig
from ai_trading_module.ai_trader import AITrader
from paper_trading.paper_exchange import PaperExchange
from binance_api.client import BinanceClient
from notifications.telegram_notifier import TelegramNotifier


def setup_logging(config: AITradingConfig):
    """Setup logging configuration"""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_exchange(paper_mode: bool):
    """Create exchange client"""
    if paper_mode:
        print("[Setup] Using Paper Trading Exchange")
        # Load or create paper exchange with separate state for AI module
        state_file = "/home/user/jjhbn/GGG3/ai_trading_module/data/paper_exchange_state.json"
        exchange = PaperExchange(initial_balance=1000.0, state_file=state_file)
        exchange.load_state()
        return exchange
    else:
        print("[Setup] Using Live Binance Exchange")
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')

        if not api_key or not secret_key:
            raise ValueError("Binance API keys not set. Use paper mode or set API keys.")

        return BinanceClient(api_key, secret_key, testnet=False)


def create_telegram_notifier():
    """Create Telegram notifier"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        print("[Warning] Telegram credentials not set. Notifications disabled.")
        return None

    try:
        from notifications.telegram_notifier import TelegramNotifier
        return TelegramNotifier(token, chat_id)
    except Exception as e:
        print(f"[Warning] Failed to initialize Telegram: {e}")
        return None


def print_banner():
    """Print startup banner"""
    print("\n" + "="*80)
    print(" " * 25 + "AI TRADING MODULE")
    print(" " * 20 + "Powered by Claude Sonnet 4.5")
    print("="*80 + "\n")


def print_config_summary(config: AITradingConfig):
    """Print configuration summary"""
    print("Configuration:")
    print(f"  • Initial Capital: ${config.INITIAL_CAPITAL}")
    print(f"  • Paper Trading: {config.PAPER_TRADING}")
    print(f"  • AI Model: {config.AI_MODEL}")
    print(f"  • Max Positions: {config.MAX_POSITIONS}")
    print(f"  • Decision Interval: {config.DECISION_INTERVAL_HOURS}h")
    print(f"  • Position Check Interval: {config.CHECK_POSITIONS_INTERVAL_MINUTES}min")
    print(f"  • Max Daily API Cost: ${config.MAX_DAILY_API_COST_USD}")
    print(f"  • Emergency Stop Drawdown: {config.EMERGENCY_STOP_DRAWDOWN_PCT}%")
    print()


def check_environment():
    """Check required environment variables"""
    errors = []

    if not os.getenv('ANTHROPIC_API_KEY'):
        errors.append("ANTHROPIC_API_KEY not set")

    # Telegram is optional
    if not os.getenv('TELEGRAM_BOT_TOKEN'):
        print("[Info] TELEGRAM_BOT_TOKEN not set - notifications disabled")
    if not os.getenv('TELEGRAM_CHAT_ID'):
        print("[Info] TELEGRAM_CHAT_ID not set - notifications disabled")

    if errors:
        print("\n[ERROR] Missing required environment variables:")
        for error in errors:
            print(f"  • {error}")
        print("\n" + "="*80)
        print("РЕШЕНИЕ:")
        print("="*80)
        print("\nОпция 1: Добавьте переменные в файл .env (в директории GGG3)")
        print("  Создайте или отредактируйте файл GGG3/.env:")
        print("  ANTHROPIC_API_KEY=sk-ant-your_key_here")
        print("  TELEGRAM_BOT_TOKEN=your_token_here")
        print("  TELEGRAM_CHAT_ID=your_chat_id_here")
        print("\nОпция 2: Экспортируйте переменные окружения")
        print("  Windows (PowerShell):")
        print("    $env:ANTHROPIC_API_KEY='your_key_here'")
        print("  Linux/Mac:")
        print("    export ANTHROPIC_API_KEY='your_key_here'")
        print("\n" + "="*80)
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AI Trading Module - Autonomous trading using Claude Sonnet 4.5'
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        default=True,
        help='Use paper trading mode (default: True)'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Use live trading mode (DANGEROUS - requires API keys)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=1000.0,
        help='Initial capital in USDT (default: 1000)'
    )
    parser.add_argument(
        '--decision-hour',
        type=int,
        default=10,
        help='Hour of day for daily decisions in UTC (default: 10)'
    )
    parser.add_argument(
        '--max-api-cost',
        type=float,
        default=50.0,
        help='Maximum daily API cost in USD (default: 50)'
    )

    args = parser.parse_args()

    # Determine mode
    paper_mode = not args.live

    # Print banner
    print_banner()

    # Check environment
    check_environment()

    # Create configuration
    config = AITradingConfig()
    config.PAPER_TRADING = paper_mode
    config.INITIAL_CAPITAL = args.capital
    config.DECISION_TIME_UTC = args.decision_hour
    config.MAX_DAILY_API_COST_USD = args.max_api_cost

    # Setup logging
    setup_logging(config)

    # Print configuration
    print_config_summary(config)

    # Validate configuration
    errors = config.validate()
    if errors:
        print("[ERROR] Configuration validation failed:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)

    # Create exchange
    try:
        exchange = create_exchange(paper_mode)
    except Exception as e:
        print(f"[ERROR] Failed to create exchange: {e}")
        sys.exit(1)

    # Create Telegram notifier
    telegram = create_telegram_notifier()

    # Create AI Trader
    try:
        trader = AITrader(
            exchange=exchange,
            config=config,
            telegram_notifier=telegram
        )
    except Exception as e:
        print(f"[ERROR] Failed to create AI Trader: {e}")
        sys.exit(1)

    # Warning for live trading
    if not paper_mode:
        print("\n" + "!"*80)
        print(" " * 20 + "⚠️  LIVE TRADING MODE ENABLED  ⚠️")
        print("!"*80)
        print("\nThis bot will trade with REAL MONEY on Binance Futures!")
        print("Are you absolutely sure? Type 'YES I UNDERSTAND THE RISKS' to continue:")
        confirmation = input("> ")

        if confirmation != "YES I UNDERSTAND THE RISKS":
            print("\n[Cancelled] Exiting...")
            sys.exit(0)

        print("\nStarting live trading in 5 seconds...")
        import time
        for i in range(5, 0, -1):
            print(f"{i}...")
            time.sleep(1)

    # Run trader
    try:
        trader.run()
    except KeyboardInterrupt:
        print("\n\n[Shutdown] AI Trader stopped by user")
    except Exception as e:
        print(f"\n\n[CRITICAL ERROR] {e}")
        logging.exception("Critical error in main loop")
        if telegram:
            telegram.send_message(f"🚨 AI Trader Critical Error:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
