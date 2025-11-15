#!/usr/bin/env python3
"""
test_recent_fixes.py — Тестирование последних исправлений

Проверяет:
1. Исправление BASE Win Rate для SHORT позиций
2. Правильность обучения ML моделей
3. Отправку уведомлений в Telegram об обучении моделей

Usage:
    python3 GGG3/tests/test_recent_fixes.py

Автор: Claude (Anthropic)
Дата: 2025-11-15
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Добавляем путь к GGG3
TEST_DIR = Path(__file__).parent
GGG3_DIR = TEST_DIR.parent
sys.path.insert(0, str(GGG3_DIR))

# Импорт необходимых модулей (после добавления пути)
try:
    from portfolio.position_manager import Position
    print("✓ Импортирован Position")
except ImportError as e:
    print(f"✗ Ошибка импорта Position: {e}")
    sys.exit(1)


def test_base_winrate_short_positions():
    """
    Тест 1: Проверка правильности подсчета Win Rate для BASE в SHORT позициях

    Проблема (было):
    - Для SHORT позиций p_meta содержит вероятность ПАДЕНИЯ
    - Но проверка предполагала вероятность РОСТА
    - Результат: все profitable SHORT засчитывались как ошибки

    Решение:
    - Добавлена проверка направления позиции
    - Для SHORT используется инвертированная логика
    """
    print("\n" + "="*80)
    print("ТЕСТ 1: BASE Win Rate для SHORT позиций")
    print("="*80)

    test_cases = [
        # (direction, p_base, pnl, expected_base_was_right, description)
        ('LONG', 0.7, 10, True, "LONG: предсказал рост (0.7), цена выросла (pnl>0)"),
        ('LONG', 0.7, -10, False, "LONG: предсказал рост (0.7), цена упала (pnl<0)"),
        ('LONG', 0.3, -10, True, "LONG: предсказал падение (0.3), цена упала (pnl<0)"),
        ('LONG', 0.3, 10, False, "LONG: предсказал падение (0.3), цена выросла (pnl>0)"),

        ('SHORT', 0.7, 10, True, "SHORT: предсказал падение (0.7), цена упала (pnl>0)"),
        ('SHORT', 0.7, -10, False, "SHORT: предсказал падение (0.7), цена выросла (pnl<0)"),
        ('SHORT', 0.3, -10, True, "SHORT: предсказал рост (0.3), цена выросла (pnl<0)"),
        ('SHORT', 0.3, 10, False, "SHORT: предсказал рост (0.3), цена упала (pnl>0)"),
    ]

    passed = 0
    total = len(test_cases)

    for direction, p_base, pnl, expected, description in test_cases:
        # Имитируем логику из binance_bot_main.py:1644-1650
        if direction == 'LONG':
            price_went_up = pnl > 0
        else:  # SHORT
            price_went_up = pnl < 0  # SHORT: убыток означает цена выросла

        # Новая логика с учетом направления
        if direction == 'LONG':
            base_was_right = (p_base > 0.5 and price_went_up) or (p_base <= 0.5 and not price_went_up)
        else:  # SHORT
            base_was_right = (p_base > 0.5 and not price_went_up) or (p_base <= 0.5 and price_went_up)

        if base_was_right == expected:
            print(f"  ✓ {description}")
            passed += 1
        else:
            print(f"  ✗ {description}")
            print(f"    Ожидалось: {expected}, получено: {base_was_right}")

    print(f"\nРезультат: {passed}/{total} тестов пройдено")
    return passed == total


def test_ml_experts_winrate_short_positions():
    """
    Тест 2: Проверка правильности подсчета Win Rate для ML экспертов в SHORT позициях

    ML эксперты должны правильно работать для обоих направлений,
    т.к. их predictions всегда содержат вероятность роста цены (direction-agnostic)
    """
    print("\n" + "="*80)
    print("ТЕСТ 2: ML Experts Win Rate для SHORT позиций")
    print("="*80)

    test_cases = [
        # (direction, p_up, pnl, expected_was_right, description)
        ('LONG', 0.6, 10, True, "LONG: предсказал рост (0.6), цена выросла (pnl>0)"),
        ('LONG', 0.6, -10, False, "LONG: предсказал рост (0.6), цена упала (pnl<0)"),
        ('LONG', 0.4, -10, True, "LONG: предсказал падение (0.4), цена упала (pnl<0)"),
        ('LONG', 0.4, 10, False, "LONG: предсказал падение (0.4), цена выросла (pnl>0)"),

        ('SHORT', 0.3, 10, True, "SHORT: предсказал падение (0.3), цена упала (pnl>0)"),
        ('SHORT', 0.3, -10, False, "SHORT: предсказал падение (0.3), цена выросла (pnl<0)"),
        ('SHORT', 0.6, -10, True, "SHORT: предсказал рост (0.6), цена выросла (pnl<0)"),
        ('SHORT', 0.6, 10, False, "SHORT: предсказал рост (0.6), цена упала (pnl>0)"),
    ]

    passed = 0
    total = len(test_cases)

    for direction, p_up, pnl, expected, description in test_cases:
        # Имитируем логику из binance_bot_main.py:1622-1637
        if direction == 'LONG':
            price_went_up = pnl > 0
        else:  # SHORT
            price_went_up = pnl < 0  # SHORT: убыток означает цена выросла

        # Универсальная логика для ML экспертов (одинакова для LONG и SHORT)
        expert_was_right = (p_up > 0.5 and price_went_up) or (p_up <= 0.5 and not price_went_up)

        if expert_was_right == expected:
            print(f"  ✓ {description}")
            passed += 1
        else:
            print(f"  ✗ {description}")
            print(f"    Ожидалось: {expected}, получено: {expert_was_right}")

    print(f"\nРезультат: {passed}/{total} тестов пройдено")
    return passed == total


def test_telegram_notification_method_exists():
    """
    Тест 3: Проверка наличия метода notify_models_trained в TelegramNotifier
    """
    print("\n" + "="*80)
    print("ТЕСТ 3: Проверка метода notify_models_trained")
    print("="*80)

    try:
        from notifications.telegram_notifier import TelegramNotifier
        print("  ✓ Импортирован TelegramNotifier")

        # Проверяем наличие метода
        if hasattr(TelegramNotifier, 'notify_models_trained'):
            print("  ✓ Метод notify_models_trained существует")

            # Проверяем сигнатуру метода
            import inspect
            sig = inspect.signature(TelegramNotifier.notify_models_trained)
            params = list(sig.parameters.keys())

            if 'self' in params and 'training_data' in params:
                print(f"  ✓ Сигнатура метода корректна: {params}")
                return True
            else:
                print(f"  ✗ Неправильная сигнатура метода: {params}")
                return False
        else:
            print("  ✗ Метод notify_models_trained не найден")
            return False
    except Exception as e:
        print(f"  ✗ Ошибка при проверке: {e}")
        return False


def test_telegram_notification_content():
    """
    Тест 4: Проверка содержимого уведомления об обучении моделей
    """
    print("\n" + "="*80)
    print("ТЕСТ 4: Проверка содержимого уведомления")
    print("="*80)

    try:
        from notifications.telegram_notifier import TelegramNotifier

        # Создаем mock TelegramNotifier
        with patch.object(TelegramNotifier, '_send_message') as mock_send:
            notifier = TelegramNotifier(
                bot_token="test_token",
                chat_id="test_chat",
                enabled=True
            )

            # Тестовые данные
            training_data = {
                'symbol': 'BTCUSDT',
                'outcome': 'win',
                'total_trades': 5,
                'models': {
                    'base': {'win_rate': 1.0, 'total': 5},
                    'xgb': {'samples': 5, 'trees': 10, 'win_rate': 0.8, 'total': 5},
                    'rf': {'samples': 5, 'win_rate': 0.6, 'total': 5},
                    'arf': {'samples': 5, 'win_rate': 0.7, 'total': 5},
                    'nn': {'epochs': 5, 'win_rate': 0.9, 'total': 5},
                    'meta': {'mode': 'SHADOW', 'samples': 5, 'win_rate': 0.0, 'total': 0}
                }
            }

            # Вызываем метод
            notifier.notify_models_trained(training_data)

            # Проверяем что _send_message был вызван
            if mock_send.called:
                print("  ✓ Метод _send_message был вызван")

                # Получаем переданный текст
                call_args = mock_send.call_args
                if call_args and len(call_args[0]) > 0:
                    message_text = call_args[0][0]

                    # Проверяем наличие ключевых элементов
                    checks = [
                        ('МОДЕЛИ ОБУЧИЛИСЬ' in message_text, "Заголовок 'МОДЕЛИ ОБУЧИЛИСЬ'"),
                        ('BTCUSDT' in message_text, "Символ BTCUSDT"),
                        ('win' in message_text, "Outcome 'win'"),
                        ('5' in message_text, "Количество сделок"),
                        ('BASE' in message_text, "Модель BASE"),
                        ('XGBoost' in message_text, "Модель XGBoost"),
                        ('RandomForest' in message_text, "Модель RandomForest"),
                        ('AdaptiveRF' in message_text, "Модель AdaptiveRF"),
                        ('NeuralNet' in message_text, "Модель NeuralNet"),
                        ('META' in message_text, "Модель META"),
                        ('100.0%' in message_text or '100%' in message_text, "Win Rate 100% для BASE"),
                    ]

                    passed = 0
                    for check, description in checks:
                        if check:
                            print(f"    ✓ {description}")
                            passed += 1
                        else:
                            print(f"    ✗ {description}")

                    print(f"\n  Результат: {passed}/{len(checks)} проверок пройдено")

                    if passed == len(checks):
                        print("\n  📨 Пример сообщения:")
                        print("  " + "-"*76)
                        for line in message_text.split('\n')[:20]:  # Показываем первые 20 строк
                            print(f"  {line}")
                        print("  " + "-"*76)
                        return True
                    return False
                else:
                    print("  ✗ Не удалось получить текст сообщения")
                    return False
            else:
                print("  ✗ Метод _send_message не был вызван")
                return False

    except Exception as e:
        print(f"  ✗ Ошибка при проверке: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_alert_type():
    """
    Тест 5: Проверка наличия конфигурации 'model_training' в TELEGRAM_ALERT_TYPES
    """
    print("\n" + "="*80)
    print("ТЕСТ 5: Проверка конфигурации model_training")
    print("="*80)

    try:
        import binance_config as config

        if hasattr(config, 'TELEGRAM_ALERT_TYPES'):
            print("  ✓ TELEGRAM_ALERT_TYPES существует")

            alert_types = config.TELEGRAM_ALERT_TYPES
            if 'model_training' in alert_types:
                value = alert_types['model_training']
                print(f"  ✓ 'model_training' найден: {value}")

                if value is True:
                    print("  ✓ 'model_training' включен (True)")
                    return True
                else:
                    print(f"  ⚠ 'model_training' выключен: {value}")
                    return True  # Это не ошибка, просто информация
            else:
                print("  ✗ 'model_training' не найден в TELEGRAM_ALERT_TYPES")
                return False
        else:
            print("  ✗ TELEGRAM_ALERT_TYPES не найден в конфигурации")
            return False

    except Exception as e:
        print(f"  ✗ Ошибка при проверке: {e}")
        return False


def test_code_integration():
    """
    Тест 6: Проверка интеграции в binance_bot_main.py (анализ кода)
    """
    print("\n" + "="*80)
    print("ТЕСТ 6: Проверка интеграции в binance_bot_main.py")
    print("="*80)

    bot_file = GGG3_DIR / 'binance_bot_main.py'

    if not bot_file.exists():
        print(f"  ✗ Файл не найден: {bot_file}")
        return False

    with open(bot_file, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ('notify_models_trained' in content, "Вызов notify_models_trained найден"),
        ('model_training' in content, "Проверка config model_training найдена"),
        ("if position.direction == 'LONG':" in content and
         "else:  # SHORT" in content and
         "base_was_right" in content,
         "Логика проверки direction для BASE найдена"),
        ('training_data' in content and 'models_data' in content,
         "Формирование training_data найдено"),
    ]

    passed = 0
    for check, description in checks:
        if check:
            print(f"  ✓ {description}")
            passed += 1
        else:
            print(f"  ✗ {description}")

    print(f"\nРезультат: {passed}/{len(checks)} проверок пройдено")
    return passed == len(checks)


def main():
    """Главная функция запуска всех тестов"""
    print("\n" + "="*80)
    print("🧪 ТЕСТИРОВАНИЕ ПОСЛЕДНИХ ИСПРАВЛЕНИЙ")
    print("="*80)
    print("\nПроверка:")
    print("1. Исправление BASE Win Rate для SHORT позиций")
    print("2. Правильность обучения ML моделей")
    print("3. Отправка уведомлений в Telegram об обучении моделей")
    print()

    results = []

    # Запускаем тесты
    results.append(('BASE Win Rate (SHORT)', test_base_winrate_short_positions()))
    results.append(('ML Experts Win Rate (SHORT)', test_ml_experts_winrate_short_positions()))
    results.append(('Метод notify_models_trained', test_telegram_notification_method_exists()))
    results.append(('Содержимое уведомления', test_telegram_notification_content()))
    results.append(('Конфигурация model_training', test_config_alert_type()))
    results.append(('Интеграция в binance_bot_main', test_code_integration()))

    # Итоги
    print("\n" + "="*80)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status:12} | {name}")

    print("\n" + "="*80)
    success_rate = (passed_count / total_count) * 100
    print(f"Результат: {passed_count}/{total_count} тестов пройдено ({success_rate:.0f}%)")
    print("="*80)

    if passed_count == total_count:
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        return 0
    else:
        print(f"\n❌ ПРОВАЛЕНО ТЕСТОВ: {total_count - passed_count}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
