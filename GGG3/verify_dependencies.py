#!/usr/bin/env python3
"""
Verify all dependencies are installed correctly
"""

import sys

def check_library(name, import_name=None):
    """Check if a library can be imported"""
    if import_name is None:
        import_name = name

    try:
        __import__(import_name)
        print(f"✓ {name:25s} INSTALLED")
        return True
    except ImportError as e:
        print(f"✗ {name:25s} MISSING ({e})")
        return False
    except Exception as e:
        print(f"✗ {name:25s} ERROR ({e})")
        return False


def main():
    """Check all required libraries"""
    print("=" * 80)
    print("DEPENDENCY VERIFICATION")
    print("=" * 80)
    print()

    libraries = [
        # Core runtime
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("requests", "requests"),
        ("urllib3", "urllib3"),
        ("python-dotenv", "dotenv"),
        ("web3", "web3"),
        ("river", "river"),
        ("matplotlib", "matplotlib"),
        ("cma", "cma"),

        # ML models
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),

        # NN head (optional)
        ("torch", "torch"),
        ("libauc", "libauc"),

        # Exchange APIs
        ("python-binance", "binance"),
        ("ccxt", "ccxt"),

        # Technical Analysis
        ("ta-lib", "talib"),

        # Dashboard
        ("plotly", "plotly"),
        ("dash", "dash"),

        # Telegram Bot
        ("python-telegram-bot", "telegram"),

        # Testing
        ("pytest", "pytest"),
    ]

    results = []
    for name, import_name in libraries:
        result = check_library(name, import_name)
        results.append((name, result))

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    installed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"Installed: {installed}/{total}")

    if installed == total:
        print()
        print("✅ All libraries installed successfully!")
        return 0
    else:
        print()
        print("⚠️ Some libraries are missing. Install them with:")
        print("    pip install -r requirements.txt")

        missing = [name for name, result in results if not result]
        print()
        print("Missing libraries:")
        for name in missing:
            print(f"  - {name}")

        return 1


if __name__ == '__main__':
    sys.exit(main())
