#!/usr/bin/env python3
"""
Test suite for paper_dashboard.py

Tests dashboard functionality including data loading,
metrics calculation, and layout generation.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all imports work correctly"""
    print("Testing imports...", end=" ")
    try:
        import dash
        from dash import dcc, html, Input, Output
        import plotly.graph_objs as go
        import pandas as pd
        from datetime import datetime, timedelta
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_dashboard_imports():
    """Test dashboard module imports"""
    print("Testing dashboard imports...", end=" ")
    try:
        from paper_dashboard import (
            DashboardConfig,
            DataLoader,
            MetricsCalculator,
            PaperDashboard
        )
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_configuration():
    """Test DashboardConfig class"""
    print("Testing configuration...", end=" ")
    try:
        from paper_dashboard import DashboardConfig

        config = DashboardConfig()

        # Check required attributes
        assert hasattr(config, 'REFRESH_INTERVAL'), "Missing REFRESH_INTERVAL"
        assert hasattr(config, 'PORT'), "Missing PORT"
        assert hasattr(config, 'TRADES_CSV'), "Missing TRADES_CSV"
        assert hasattr(config, 'EXCHANGE_STATE'), "Missing EXCHANGE_STATE"
        assert hasattr(config, 'BOT_STATE'), "Missing BOT_STATE"

        # Check values
        assert config.REFRESH_INTERVAL == 10000, f"Expected 10000, got {config.REFRESH_INTERVAL}"
        assert config.PORT == 8050, f"Expected 8050, got {config.PORT}"

        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_data_loader():
    """Test DataLoader class"""
    print("Testing DataLoader...", end=" ")
    try:
        from paper_dashboard import DashboardConfig, DataLoader

        config = DashboardConfig()
        loader = DataLoader(config)

        # Test load_exchange_state
        state = loader.load_exchange_state()
        assert state is not None, "Failed to load exchange state"
        assert 'balance' in state, "Missing balance in state"
        assert 'positions' in state, "Missing positions in state"
        assert 'equity_history' in state, "Missing equity_history in state"

        # Test load_trades
        trades_df = loader.load_trades()
        assert trades_df is not None, "Failed to load trades"
        assert len(trades_df) > 0, "Trades dataframe is empty"
        assert 'symbol' in trades_df.columns, "Missing symbol column"
        assert 'action' in trades_df.columns, "Missing action column"

        # Test get_equity_history
        equity_history = loader.get_equity_history()
        assert isinstance(equity_history, list), "Equity history should be a list"
        assert len(equity_history) > 0, "Equity history is empty"

        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_metrics_calculator():
    """Test MetricsCalculator class"""
    print("Testing MetricsCalculator...", end=" ")
    try:
        from paper_dashboard import DashboardConfig, DataLoader, MetricsCalculator
        import pandas as pd

        config = DashboardConfig()
        loader = DataLoader(config)

        # Load data
        state = loader.load_exchange_state()
        trades_df = loader.load_trades()

        # Calculate metrics
        metrics = MetricsCalculator.calculate_metrics(trades_df, state)

        # Check required metrics
        required_metrics = [
            'win_rate', 'profit_factor', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'avg_win', 'avg_loss', 'total_trades',
            'winning_trades', 'losing_trades', 'total_pnl', 'roi'
        ]

        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        # Validate metric types and ranges
        assert isinstance(metrics['win_rate'], (int, float)), "win_rate should be numeric"
        assert 0 <= metrics['win_rate'] <= 100, f"win_rate out of range: {metrics['win_rate']}"

        assert isinstance(metrics['total_trades'], int), "total_trades should be integer"
        assert metrics['total_trades'] >= 0, "total_trades should be non-negative"

        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_dashboard_initialization():
    """Test PaperDashboard initialization"""
    print("Testing dashboard initialization...", end=" ")
    try:
        from paper_dashboard import PaperDashboard

        dashboard = PaperDashboard()

        # Check attributes
        assert hasattr(dashboard, 'config'), "Missing config attribute"
        assert hasattr(dashboard, 'loader'), "Missing loader attribute"
        assert hasattr(dashboard, 'app'), "Missing app attribute"

        # Check app is a Dash instance
        import dash
        assert isinstance(dashboard.app, dash.Dash), "app is not a Dash instance"

        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_dashboard_layout():
    """Test dashboard layout generation"""
    print("Testing dashboard layout...", end=" ")
    try:
        from paper_dashboard import PaperDashboard

        dashboard = PaperDashboard()

        # Check layout exists
        assert dashboard.app.layout is not None, "Layout is None"

        # Check layout has children
        assert hasattr(dashboard.app.layout, 'children'), "Layout has no children"

        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_overview_section():
    """Test overview section data generation"""
    print("Testing overview section...", end=" ")
    try:
        from paper_dashboard import PaperDashboard

        dashboard = PaperDashboard()

        # Get current state
        state = dashboard.loader.load_exchange_state()
        trades_df = dashboard.loader.load_trades()

        # Check we can calculate overview metrics
        balance = state.get('balance', 0)
        assert isinstance(balance, (int, float)), "Balance should be numeric"

        open_positions = len(state.get('positions', {}))
        assert isinstance(open_positions, int), "Open positions should be integer"

        closed_trades = trades_df[trades_df['action'] == 'CLOSE']
        total_trades = len(closed_trades)
        assert total_trades > 0, "Should have some closed trades"

        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_equity_curve_data():
    """Test equity curve data preparation"""
    print("Testing equity curve data...", end=" ")
    try:
        from paper_dashboard import DashboardConfig, DataLoader

        config = DashboardConfig()
        loader = DataLoader(config)

        equity_history = loader.get_equity_history()

        assert len(equity_history) > 0, "Equity history is empty"

        # Check data structure
        for point in equity_history:
            assert 'timestamp' in point, "Missing timestamp"
            assert 'equity' in point, "Missing equity"
            assert isinstance(point['timestamp'], (int, float)), "Timestamp should be numeric"
            assert isinstance(point['equity'], (int, float)), "Equity should be numeric"

        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_open_positions_data():
    """Test open positions data preparation"""
    print("Testing open positions data...", end=" ")
    try:
        from paper_dashboard import DashboardConfig, DataLoader

        config = DashboardConfig()
        loader = DataLoader(config)

        state = loader.load_exchange_state()
        positions = state.get('positions', {})

        assert isinstance(positions, dict), "Positions should be a dictionary"

        # Check position structure
        for pos_id, position in positions.items():
            assert 'symbol' in position, f"Position {pos_id} missing symbol"
            assert 'side' in position, f"Position {pos_id} missing side"
            assert 'entry_price' in position, f"Position {pos_id} missing entry_price"
            assert 'amount' in position, f"Position {pos_id} missing amount"

        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 80)
    print("PAPER DASHBOARD TEST SUITE")
    print("=" * 80)
    print()

    tests = [
        ("Imports", test_imports),
        ("Dashboard Imports", test_dashboard_imports),
        ("Configuration", test_configuration),
        ("Data Loader", test_data_loader),
        ("Metrics Calculator", test_metrics_calculator),
        ("Dashboard Init", test_dashboard_initialization),
        ("Dashboard Layout", test_dashboard_layout),
        ("Overview Section", test_overview_section),
        ("Equity Curve Data", test_equity_curve_data),
        ("Open Positions Data", test_open_positions_data),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<30} {status}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print()
        print("=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        return True
    else:
        print()
        print("=" * 80)
        print(f"✗ {total - passed} TEST(S) FAILED")
        print("=" * 80)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
