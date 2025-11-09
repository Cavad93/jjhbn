#!/usr/bin/env python3
"""
Скрипт для проверки конфигурации Haiku Analyzer
"""

import os
import sys

print("=" * 70)
print("ДИАГНОСТИКА: Конфигурация Haiku Analyzer")
print("=" * 70)

# 1. Проверка переменной окружения
api_key = os.getenv('ANTHROPIC_API_KEY', '')
print(f"\n1. ANTHROPIC_API_KEY в окружении:")
if api_key:
    print(f"   ✅ Установлен: {api_key[:10]}...{api_key[-4:]}")
else:
    print(f"   ❌ НЕ установлен (пустая строка)")

# 2. Проверка конфига
sys.path.append('/home/user/jjhbn/GGG3')
try:
    import binance_config as config
    print(f"\n2. Конфигурация в binance_config.py:")
    print(f"   HAIKU_ANALYSIS_ENABLED = {config.HAIKU_ANALYSIS_ENABLED}")
    print(f"   ANTHROPIC_API_KEY = {'[SET]' if config.ANTHROPIC_API_KEY else '[NOT SET]'}")
    print(f"   HAIKU_BATCH_SIZE = {config.HAIKU_BATCH_SIZE}")
    print(f"   MIN_P_UP_FOR_ANALYSIS = {getattr(config, 'MIN_P_UP_FOR_ANALYSIS', 'N/A')}")
except Exception as e:
    print(f"   ❌ Ошибка загрузки конфига: {e}")

# 3. Проверка модуля haiku_analyzer
try:
    from strategy.haiku_analyzer import HaikuConfig, HaikuAnalyzer
    print(f"\n3. Константы HaikuConfig:")
    print(f"   MIN_P_UP_FOR_ANALYSIS = {HaikuConfig.MIN_P_UP_FOR_ANALYSIS}")
    print(f"   SCORE_THRESHOLD_APPROVE = {HaikuConfig.SCORE_THRESHOLD_APPROVE}")
    print(f"   SCORE_THRESHOLD_REJECT = {HaikuConfig.SCORE_THRESHOLD_REJECT}")
except Exception as e:
    print(f"   ❌ Ошибка загрузки haiku_analyzer: {e}")

# 4. Проверка возможности инициализации
print(f"\n4. Проверка инициализации Haiku Analyzer:")
try:
    if config.HAIKU_ANALYSIS_ENABLED and config.ANTHROPIC_API_KEY:
        print(f"   ✅ Haiku будет ВКЛЮЧЕН (enabled=True, key=True)")
        # Пытаемся инициализировать (без реальных запросов)
        try:
            from anthropic import Anthropic
            print(f"   ✅ Модуль 'anthropic' установлен")
        except ImportError:
            print(f"   ❌ Модуль 'anthropic' НЕ установлен (pip install anthropic)")
    else:
        print(f"   ❌ Haiku будет ОТКЛЮЧЕН:")
        if not config.HAIKU_ANALYSIS_ENABLED:
            print(f"      - HAIKU_ANALYSIS_ENABLED = False")
        if not config.ANTHROPIC_API_KEY:
            print(f"      - ANTHROPIC_API_KEY не установлен")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# 5. Рекомендации
print(f"\n" + "=" * 70)
print("РЕКОМЕНДАЦИИ:")
print("=" * 70)

if not api_key:
    print("❌ API ключ Anthropic не установлен!")
    print("")
    print("   Для включения Haiku Analyzer:")
    print("   1. Получите API ключ: https://console.anthropic.com/")
    print("   2. Установите переменную окружения:")
    print("      export ANTHROPIC_API_KEY='your-key-here'")
    print("   3. Или добавьте в .env файл:")
    print("      echo 'ANTHROPIC_API_KEY=your-key' >> .env")
    print("")
    print("   Если Haiku не нужен:")
    print("   - Установите HAIKU_ANALYSIS_ENABLED = False в binance_config.py")
else:
    print("✅ Конфигурация корректна!")
    print("")
    print("   Haiku Analyzer должен работать при следующем запуске бота.")

print("=" * 70)
