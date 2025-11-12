#!/usr/bin/env python3
"""
ПРОСТОЙ СКРИПТ: Восстановить баланс $998.52

Использование:
    python3 GGG3/scripts/fix_balance_998.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def main():
    """Восстанавливает баланс $998.52"""

    # Правильный баланс после закрытия всех позиций
    CORRECT_BALANCE = 998.52

    print("=" * 80)
    print("🔧 ВОССТАНОВЛЕНИЕ БАЛАНСА $998.52")
    print("=" * 80)
    print()

    # Определяем пути
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    data_dir = base_dir / "data"
    paper_exchange_file = data_dir / "paper_exchange_state.json"

    print(f"📂 Data directory: {data_dir}")
    print(f"💰 Восстанавливаемый баланс: ${CORRECT_BALANCE:.2f}")
    print()

    # Создаём data/ если не существует
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory {data_dir} ready")
    print()

    # Создаём бэкап если файл существует
    if paper_exchange_file.exists():
        print(f"⚠️  Found existing paper_exchange_state.json")

        try:
            with open(paper_exchange_file, 'r') as f:
                old_data = json.load(f)
                old_balance = old_data.get('balance', 0)
                print(f"   Old balance: ${old_balance:.2f}")
                print(f"   New balance: ${CORRECT_BALANCE:.2f}")
                print(f"   Difference: ${CORRECT_BALANCE - old_balance:+.2f}")
        except:
            print(f"   (Could not read old balance)")

        # Бэкап
        backup_file = data_dir / f"paper_exchange_state.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            import shutil
            shutil.copy2(paper_exchange_file, backup_file)
            print(f"   ✅ Backup: {backup_file.name}")
        except Exception as e:
            print(f"   ⚠️  Backup failed: {e}")

        print()

    # Создаём корректный файл
    print("🔨 Creating paper_exchange_state.json with $998.52...")

    paper_state = {
        "balance": CORRECT_BALANCE,
        "initial_capital": CORRECT_BALANCE,
        "positions": {},
        "orders": {},
        "trades_history": [],
        "closed_positions": [],
        "leverage": 1,
        "timestamp": datetime.utcnow().isoformat(),
        "_restored": True,
        "_restoration_timestamp": datetime.utcnow().isoformat(),
        "_restoration_note": f"Fixed balance to ${CORRECT_BALANCE:.2f} (all positions closed)"
    }

    # Сохраняем
    try:
        with open(paper_exchange_file, 'w') as f:
            json.dump(paper_state, f, indent=2)
        print(f"✅ Saved: {paper_exchange_file}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

    # Проверяем
    try:
        with open(paper_exchange_file, 'r') as f:
            test = json.load(f)
            actual_balance = test.get('balance', 0)

            if abs(actual_balance - CORRECT_BALANCE) < 0.01:
                print(f"✅ Validated: balance = ${actual_balance:.2f}")
            else:
                print(f"⚠️  WARNING: balance = ${actual_balance:.2f} (expected ${CORRECT_BALANCE:.2f})")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

    print()
    print("=" * 80)
    print("✅ SUCCESS!")
    print("=" * 80)
    print()
    print(f"💰 Balance restored: ${CORRECT_BALANCE:.2f} USDT")
    print()
    print("🔄 Next steps:")
    print("   1. Запустите бота:")
    print(f"      cd {base_dir}")
    print("      python3 binance_bot_main.py --paper")
    print()
    print("   2. Проверьте лог при старте:")
    print(f"      Mode: 📄 PAPER TRADING (Loaded: ${CORRECT_BALANCE:.2f} USDT)")
    print()
    print("   3. Проверьте в Telegram:")
    print("      /status")
    print()

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
