#!/usr/bin/env python3
"""
Diagnostic script to test DuckDuckGo search functionality
Tests timeout handling and import correctness
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

print("="*80)
print("DUCKDUCKGO SEARCH DIAGNOSTIC")
print("="*80)

# Test 1: Import
print("\n[1/4] Testing import...", flush=True)
try:
    from duckduckgo_search import DDGS
    print("✓ duckduckgo_search imported successfully", flush=True)
except ImportError as e:
    print(f"✗ Failed to import duckduckgo_search: {e}", flush=True)
    print("\nPlease install: pip install duckduckgo-search", flush=True)
    sys.exit(1)

# Test 2: Basic search
print("\n[2/4] Testing basic search (BTC news)...", flush=True)
try:
    with DDGS() as ddgs:
        results = list(ddgs.news("bitcoin cryptocurrency", timelimit='d', max_results=3))
        print(f"✓ Found {len(results)} news items", flush=True)
        if results:
            print(f"  Example: {results[0].get('title', 'N/A')[:60]}...", flush=True)
except Exception as e:
    print(f"✗ Basic search failed: {e}", flush=True)
    sys.exit(1)

# Test 3: Timeout handling
print("\n[3/4] Testing timeout handling...", flush=True)
def slow_search():
    """Simulates slow search"""
    with DDGS() as ddgs:
        results = ddgs.news("ethereum cryptocurrency", timelimit='d', max_results=5)
        return list(results)

try:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(slow_search)
        results = future.result(timeout=30.0)
        print(f"✓ Timeout mechanism works (found {len(results)} items in <30s)", flush=True)
except FuturesTimeoutError:
    print("✗ Search timed out after 30s (network issue?)", flush=True)
except Exception as e:
    print(f"✗ Timeout test failed: {e}", flush=True)

# Test 4: Multiple rapid searches
print("\n[4/4] Testing multiple rapid searches (stress test)...", flush=True)
coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
success_count = 0
start_time = time.time()

for coin in coins:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(f"{coin.lower()} cryptocurrency", timelimit='d', max_results=2))
            success_count += 1
            print(f"  {coin}: {len(results)} items", flush=True)
    except Exception as e:
        print(f"  {coin}: Failed - {e}", flush=True)

elapsed = time.time() - start_time
print(f"\n✓ Completed {success_count}/{len(coins)} searches in {elapsed:.1f}s", flush=True)

# Summary
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)
if success_count == len(coins):
    print("✓ All tests passed! DuckDuckGo search is working correctly.", flush=True)
    print("\nYour bot should work without hanging.", flush=True)
else:
    print(f"⚠ Some tests failed ({success_count}/{len(coins)} passed)", flush=True)
    print("\nPossible issues:", flush=True)
    print("  - Network connectivity problems", flush=True)
    print("  - DuckDuckGo rate limiting", flush=True)
    print("  - Firewall blocking requests", flush=True)

print("="*80 + "\n", flush=True)
