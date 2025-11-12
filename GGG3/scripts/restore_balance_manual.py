#!/usr/bin/env python3
"""
Manual Balance Restore Script

Используется когда вы знаете точный баланс который нужно восстановить.

Usage:
    python scripts/restore_balance_manual.py 998.52
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def restore_manual_balance(balance: float):
    """
    Восстанавливает баланс вручную

    Args:
        balance: Баланс для восстановления
    """
    print("="*80)
    print("🔧 MANUAL BALANCE RESTORE")
    print("="*80)
    print()

    # Определяем пути
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    data_dir = base_dir / "data"
    paper_exchange_file = data_dir / "paper_exchange_state.json"

    print(f"📂 Base directory: {base_dir}")
    print(f"📂 Data directory: {data_dir}")
    print()
    print(f"💰 Balance to restore: ${balance:.2f}")
    print()

    # Создаём бэкап если файл существует
    if paper_exchange_file.exists():
        print(f"⚠️  Found existing paper_exchange_state.json")

        backup_file = data_dir / f"paper_exchange_state.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            import shutil
            shutil.copy2(paper_exchange_file, backup_file)
            print(f"✅ Backup created: {backup_file.name}")
        except Exception as e:
            print(f"⚠️  Failed to create backup: {e}")
        print()

    # Создаём новый файл
    print("🔨 Creating new paper_exchange_state.json...")

    paper_state = {
        "balance": balance,
        "initial_capital": balance,
        "positions": {},
        "orders": {},
        "trades_history": [],
        "leverage": 1,
        "timestamp": datetime.utcnow().timestamp(),
        "_restored_manually": True,
        "_restoration_timestamp": datetime.utcnow().isoformat(),
        "_restoration_note": f"Manually restored balance={balance:.2f}"
    }

    # Сохраняем
    try:
        with open(paper_exchange_file, 'w') as f:
            json.dump(paper_state, f, indent=2)
        print(f"✅ File saved: {paper_exchange_file}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save file: {e}")
        return False

    # Проверяем
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
    print(f"💰 Balance restored: ${balance:.2f}")
    print()
    print("🔄 Next steps:")
    print("   1. Start the bot:")
    print(f"      python {base_dir.name}/binance_bot_main.py --paper")
    print()

    return True


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/restore_balance_manual.py <balance>")
        print("Example: python scripts/restore_balance_manual.py 998.52")
        sys.exit(1)

    try:
        balance = float(sys.argv[1])

        if balance <= 0:
            print(f"❌ ERROR: Balance must be positive (got ${balance:.2f})")
            sys.exit(1)

        success = restore_manual_balance(balance)
        sys.exit(0 if success else 1)

    except ValueError:
        print(f"❌ ERROR: Invalid balance value: {sys.argv[1]}")
        print("Balance must be a number (e.g., 998.52)")
        sys.exit(1)
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
