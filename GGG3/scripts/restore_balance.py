#!/usr/bin/env python3
"""
Restore Balance Script - Восстановление баланса из Capital Tracker

Используется когда paper_exchange_state.json повреждён или потерян.
Восстанавливает баланс из последнего снимка capital tracker.

Usage:
    python3 scripts/restore_balance.py

    или

    cd /home/user/jjhbn/GGG3
    python3 scripts/restore_balance.py

Автор: Claude Code
Дата: 2025-11-12
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def restore_balance_from_capital_tracker():
    """
    Восстанавливает баланс PaperExchange из последнего снимка CapitalTracker

    Returns:
        bool: True если успешно, False если ошибка
    """
    print("="*80)
    print("🔧 RESTORE BALANCE FROM CAPITAL TRACKER")
    print("="*80)
    print()

    # Определяем базовую директорию
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    data_dir = base_dir / "data"

    print(f"📂 Base directory: {base_dir}")
    print(f"📂 Data directory: {data_dir}")
    print()

    # Файлы
    snapshots_file = data_dir / "capital_history.json"
    paper_exchange_file = data_dir / "paper_exchange_state.json"

    # Проверяем наличие capital tracker snapshots
    if not snapshots_file.exists():
        print(f"❌ ERROR: Capital tracker snapshots not found!")
        print(f"   Expected: {snapshots_file}")
        print()
        print("💡 Возможные причины:")
        print("   - Бот ещё не был запущен")
        print("   - Capital tracker не был инициализирован")
        print("   - Файл был удалён")
        return False

    print(f"✅ Found capital tracker snapshots: {snapshots_file}")

    # Загружаем снимки
    try:
        with open(snapshots_file, 'r') as f:
            data = json.load(f)
            # CapitalTracker сохраняет в формате {"history": [...], "last_updated": ...}
            snapshots = data.get('history', [])
    except Exception as e:
        print(f"❌ ERROR: Failed to load capital tracker snapshots: {e}")
        return False

    if not snapshots or len(snapshots) == 0:
        print(f"❌ ERROR: Capital tracker has no snapshots!")
        return False

    print(f"✅ Loaded {len(snapshots)} capital snapshots")
    print()

    # Получаем последний снимок
    latest = snapshots[-1]

    # ВАЖНО: Используем free_balance, а не total_equity!
    # total_equity = free_balance + locked_in_positions + reserve_fund
    # balance в paper_exchange = только свободный баланс
    free_balance = latest.get('free_balance', 0.0)
    locked = latest.get('locked_in_positions', 0.0)
    reserve = latest.get('reserve_fund', 0.0)
    total_equity = latest.get('total_equity', 0.0)
    latest_time = latest.get('timestamp', datetime.utcnow().isoformat())

    print("📊 LAST CAPITAL SNAPSHOT:")
    print(f"   💰 Total Equity: ${total_equity:.2f}")
    print(f"   💵 Free Balance: ${free_balance:.2f}")
    print(f"   🔒 Locked: ${locked:.2f}")
    print(f"   🏦 Reserve: ${reserve:.2f}")
    print(f"   ⏰ Timestamp: {latest_time}")
    print()
    print(f"⚠️  Will restore FREE BALANCE only: ${free_balance:.2f}")
    print("   (Positions are managed by PositionManager separately)")

    # Проверяем разумность значения
    if free_balance <= 0:
        print()
        print(f"⚠️  WARNING: Free balance is ${free_balance:.2f} (suspicious!)")
        print("   Продолжить? (y/n): ", end='')
        try:
            answer = input().lower()
            if answer != 'y':
                print("❌ Cancelled by user")
                return False
        except:
            print()
            print("⚠️  Auto-continuing (no stdin available)")

    print()

    # Проверяем существующий paper_exchange_state.json
    if paper_exchange_file.exists():
        print(f"⚠️  Found existing paper_exchange_state.json")
        try:
            with open(paper_exchange_file, 'r') as f:
                content = f.read()

            # Пытаемся найти старый баланс
            if '"balance":' in content:
                import re
                balance_match = re.search(r'"balance":\s*([\d.]+)', content)
                if balance_match:
                    old_balance = float(balance_match.group(1))
                    print(f"   📊 Old balance: ${old_balance:.2f}")
                    print(f"   📊 New balance: ${free_balance:.2f}")
                    print(f"   📊 Difference: ${free_balance - old_balance:+.2f}")
        except Exception as e:
            print(f"   ⚠️  Could not read old balance: {e}")

        print()
        print("   Will OVERWRITE existing file!")

        # Создаём бэкап
        backup_file = data_dir / f"paper_exchange_state.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            import shutil
            shutil.copy2(paper_exchange_file, backup_file)
            print(f"   ✅ Backup created: {backup_file.name}")
        except Exception as e:
            print(f"   ⚠️  Failed to create backup: {e}")

    print()
    print("🔨 Creating new paper_exchange_state.json...")

    # Создаём новый корректный файл
    # ВАЖНО: balance = только свободный баланс (free_balance)
    # initial_capital оставляем как total_equity для истории
    paper_state = {
        "balance": free_balance,
        "initial_capital": total_equity,  # Для истории сохраняем total equity
        "positions": {},  # Позиции управляются PositionManager отдельно
        "orders": {},
        "trades_history": [],
        "closed_positions": [],  # Required by PaperExchange
        "leverage": 1,
        "timestamp": latest_time,
        "_restored_from_capital_tracker": True,
        "_restoration_timestamp": datetime.utcnow().isoformat(),
        "_restoration_note": f"Restored free_balance={free_balance:.2f} from total_equity={total_equity:.2f}"
    }

    # Сохраняем
    try:
        with open(paper_exchange_file, 'w') as f:
            json.dump(paper_state, f, indent=2)
        print(f"✅ File saved: {paper_exchange_file}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save file: {e}")
        return False

    # Проверяем что файл корректный
    try:
        with open(paper_exchange_file, 'r') as f:
            test_load = json.load(f)
        print("✅ File validated (valid JSON)")
    except Exception as e:
        print(f"❌ ERROR: Saved file is invalid: {e}")
        return False

    print()
    print("="*80)
    print("✅ SUCCESS!")
    print("="*80)
    print()
    print(f"💰 Free Balance restored: ${free_balance:.2f}")
    print(f"💎 Total Equity (with positions): ${total_equity:.2f}")
    print()
    print("🔄 Next steps:")
    print("   1. Start the bot:")
    print(f"      python3 {base_dir.name}/binance_bot_main.py --paper")
    print()
    print("   2. Check logs for:")
    print(f"      Mode: 📄 PAPER TRADING (Loaded: ${free_balance:.2f} USDT)")
    print()
    print("   3. Verify positions loaded from PositionManager")
    print()

    return True


def main():
    """Main entry point"""
    try:
        success = restore_balance_from_capital_tracker()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print()
        print("❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
