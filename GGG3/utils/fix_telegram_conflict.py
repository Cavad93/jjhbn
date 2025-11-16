#!/usr/bin/env python3
"""
Утилита для исправления конфликта Telegram Bot API (HTTP 409)

Использование:
    python utils/fix_telegram_conflict.py

Что делает:
    1. Удаляет webhook (если установлен)
    2. Проверяет запущенные процессы Python
    3. Показывает информацию о состоянии бота
"""

import sys
import os
import requests
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

import binance_config as config


def delete_webhook(bot_token: str) -> bool:
    """Удаляет webhook для бота"""
    print("\n" + "=" * 80)
    print("УДАЛЕНИЕ WEBHOOK")
    print("=" * 80)

    try:
        # Проверяем текущий webhook
        webhook_info_url = f"https://api.telegram.org/bot{bot_token}/getWebhookInfo"
        response = requests.get(webhook_info_url, timeout=10)

        if response.ok:
            data = response.json()
            result = data.get("result", {})
            current_webhook = result.get("url", "")

            if current_webhook:
                print(f"Текущий webhook: {current_webhook}")
            else:
                print("Webhook не установлен")

        # Удаляем webhook
        delete_url = f"https://api.telegram.org/bot{bot_token}/deleteWebhook"
        response = requests.get(delete_url, timeout=10)

        if response.ok:
            data = response.json()
            if data.get("ok"):
                print("✅ Webhook успешно удален")
                return True
            else:
                print(f"❌ Ошибка при удалении webhook: {data}")
                return False
        else:
            print(f"❌ HTTP ошибка: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def check_bot_status(bot_token: str):
    """Проверяет статус бота"""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА СТАТУСА БОТА")
    print("=" * 80)

    try:
        # Проверяем что токен работает
        me_url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(me_url, timeout=10)

        if response.ok:
            data = response.json()
            if data.get("ok"):
                bot_info = data["result"]
                print(f"✅ Бот активен:")
                print(f"   Username: @{bot_info.get('username')}")
                print(f"   Name: {bot_info.get('first_name')}")
                print(f"   ID: {bot_info.get('id')}")
            else:
                print(f"❌ Ошибка API: {data}")
        else:
            print(f"❌ HTTP ошибка: {response.status_code}")

        # Проверяем webhook
        webhook_info_url = f"https://api.telegram.org/bot{bot_token}/getWebhookInfo"
        response = requests.get(webhook_info_url, timeout=10)

        if response.ok:
            data = response.json()
            result = data.get("result", {})

            print("\nWebhook:")
            webhook_url = result.get("url", "")
            if webhook_url:
                print(f"  URL: {webhook_url}")
                print(f"  Pending updates: {result.get('pending_update_count', 0)}")
                print(f"  ⚠️  Webhook установлен - это КОНФЛИКТУЕТ с long polling!")
            else:
                print(f"  ✅ Webhook не установлен (long polling работает)")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


def check_running_processes():
    """Проверяет запущенные процессы Python"""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ЗАПУЩЕННЫХ ПРОЦЕССОВ")
    print("=" * 80)

    try:
        import psutil

        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if any('binance_bot_main.py' in arg for arg in cmdline):
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': ' '.join(cmdline)
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if python_processes:
            print(f"⚠️  Найдено {len(python_processes)} процессов бота:")
            for proc in python_processes:
                print(f"   PID {proc['pid']}: {proc['cmdline']}")
            print("\nЕсли бот не запущен сейчас, завершите эти процессы:")
            if sys.platform == 'win32':
                print(f"   taskkill /F /PID <PID>")
            else:
                print(f"   kill -9 <PID>")
        else:
            print("✅ Процессы бота не найдены")

    except ImportError:
        print("⚠️  psutil не установлен, пропускаю проверку процессов")
        print("   Установите: pip install psutil")
    except Exception as e:
        print(f"❌ Ошибка при проверке процессов: {e}")


def main():
    """Главная функция"""
    print("\n" + "=" * 80)
    print("🔧 TELEGRAM BOT CONFLICT FIXER")
    print("=" * 80)

    # Получаем токен из конфига
    try:
        bot_token = config.TELEGRAM_BOT_TOKEN
        print(f"\nИспользую токен из конфига: {bot_token[:10]}...")
    except AttributeError:
        print("\n❌ TELEGRAM_BOT_TOKEN не найден в binance_config.py")
        print("   Убедитесь что конфиг настроен правильно")
        return

    # Проверяем статус бота
    check_bot_status(bot_token)

    # Проверяем запущенные процессы
    check_running_processes()

    # Удаляем webhook
    print("\n" + "=" * 80)
    response = input("\nУдалить webhook? (y/n): ").lower().strip()
    if response == 'y':
        delete_webhook(bot_token)

        # Повторная проверка
        print("\nПовторная проверка статуса...")
        check_bot_status(bot_token)

    print("\n" + "=" * 80)
    print("✅ ГОТОВО")
    print("=" * 80)
    print("\nРекомендации:")
    print("  1. Убедитесь что запущена только одна копия бота")
    print("  2. Webhook должен быть удален (для long polling)")
    print("  3. Перезапустите бота")
    print("\n")


if __name__ == '__main__':
    main()
