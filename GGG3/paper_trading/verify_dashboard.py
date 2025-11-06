#!/usr/bin/env python3
"""
Quick verification that dashboard is ready to run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def verify_dashboard():
    """Verify dashboard can be initialized and is ready to run"""
    print("=" * 80)
    print("DASHBOARD VERIFICATION")
    print("=" * 80)
    print()

    print("1. Importing dashboard...", end=" ")
    try:
        from paper_dashboard import PaperDashboard
        print("✓")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("2. Initializing dashboard...", end=" ")
    try:
        dashboard = PaperDashboard()
        print("✓")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("3. Checking app instance...", end=" ")
    try:
        assert dashboard.app is not None, "App is None"
        print("✓")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("4. Checking layout...", end=" ")
    try:
        assert dashboard.app.layout is not None, "Layout is None"
        print("✓")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("5. Checking callbacks...", end=" ")
    try:
        # Dash stores callbacks in _callback_list
        assert len(dashboard.app.callback_map) > 0, "No callbacks registered"
        print(f"✓ ({len(dashboard.app.callback_map)} callbacks)")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print("6. Loading sample data...", end=" ")
    try:
        state = dashboard.loader.load_exchange_state()
        trades = dashboard.loader.load_trades()
        assert state is not None, "Failed to load state"
        assert not trades.empty, "Failed to load trades"
        print("✓")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    print()
    print("=" * 80)
    print("✓ DASHBOARD VERIFICATION SUCCESSFUL")
    print("=" * 80)
    print()
    print("Dashboard is ready to run!")
    print(f"To start the dashboard, run:")
    print(f"    python3 paper_dashboard.py")
    print(f"")
    print(f"Then open your browser to: http://localhost:{dashboard.config.PORT}")
    print()

    return True


if __name__ == '__main__':
    success = verify_dashboard()
    sys.exit(0 if success else 1)
