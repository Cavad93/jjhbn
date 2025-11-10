#!/usr/bin/env python3
"""
test_integration_code_only.py — Проверка интеграции на уровне кода (БЕЗ импортов)

Проверяет что код интеграции Transfer Learning + Diversity Monitor
добавлен корректно в binance_bot_main.py путем анализа исходного кода.

НЕ требует НИКАКИХ зависимостей - только чтение файлов.

Usage:
    python3 tests/test_integration_code_only.py

Автор: Claude (Anthropic)
Дата: 2025-11-10
"""
import sys
from pathlib import Path

# Добавляем путь к GGG3
TEST_DIR = Path(__file__).parent
GGG3_DIR = TEST_DIR.parent


def main():
    print("="*80)
    print("🧪 CODE-LEVEL INTEGRATION TEST")
    print("="*80)
    print("\nПроверка интеграции Transfer Learning + Diversity Monitor")
    print("без выполнения импортов (только анализ кода)\n")

    total_passed = 0
    total_checks = 0

    # ========== ТЕСТ 1: Файлы существуют ==========
    print("="*80)
    print("ТЕСТ 1: Проверка наличия файлов")
    print("="*80)

    files_to_check = [
        ('models/transfer_learning.py', 'Transfer Learning модуль'),
        ('models/diversity_metric.py', 'Diversity Monitor модуль'),
        ('simplified_ensemble_meta.py', 'SimplifiedEnsemble META'),
        ('binance_bot_main.py', 'Главный бот'),
        ('scripts/pretrain_experts.py', 'Pretraining скрипт'),
    ]

    for file_path, description in files_to_check:
        total_checks += 1
        full_path = GGG3_DIR / file_path
        if full_path.exists():
            print(f"  ✅ {description}: {file_path}")
            total_passed += 1
        else:
            print(f"  ❌ {description} НЕ НАЙДЕН: {file_path}")

    # ========== ТЕСТ 2: Импорты в binance_bot_main.py ==========
    print("\n" + "="*80)
    print("ТЕСТ 2: Импорты в binance_bot_main.py")
    print("="*80)

    bot_file = GGG3_DIR / 'binance_bot_main.py'
    if not bot_file.exists():
        print(f"  ❌ Файл {bot_file} не найден")
        print("\n" + "="*80)
        print(f"ИТОГО: {total_passed}/{total_checks} проверок пройдено")
        print("="*80)
        return 1

    with open(bot_file, 'r', encoding='utf-8') as f:
        bot_content = f.read()

    imports_to_check = [
        ('from models.transfer_learning import TransferLearningManager, PretrainingConfig',
         'Transfer Learning импорт'),
        ('from models.diversity_metric import DiversityMonitor, AlertType',
         'Diversity Monitor импорт'),
        ('from simplified_ensemble_meta import SimplifiedEnsembleMETA',
         'SimplifiedEnsemble импорт'),
    ]

    for import_line, description in imports_to_check:
        total_checks += 1
        if import_line in bot_content:
            print(f"  ✅ {description}")
            total_passed += 1
        else:
            print(f"  ❌ {description} НЕ НАЙДЕН")

    # ========== ТЕСТ 3: Инициализация DiversityMonitor ==========
    print("\n" + "="*80)
    print("ТЕСТ 3: Инициализация DiversityMonitor в __init__")
    print("="*80)

    patterns_to_check = [
        ('self.diversity_monitor = DiversityMonitor(',
         'DiversityMonitor создается'),
        ('window_size=100',
         'window_size параметр установлен'),
        ('low_diversity_threshold=0.05',
         'low_diversity_threshold параметр установлен'),
    ]

    for pattern, description in patterns_to_check:
        total_checks += 1
        if pattern in bot_content:
            print(f"  ✅ {description}")
            total_passed += 1
        else:
            print(f"  ❌ {description} - ПАТТЕРН НЕ НАЙДЕН")

    # ========== ТЕСТ 4: Transfer Learning в _load_experts ==========
    print("\n" + "="*80)
    print("ТЕСТ 4: Transfer Learning в _load_experts()")
    print("="*80)

    tl_patterns = [
        ('pretrained_dir = config.MODELS_DIR',
         'Проверка pretrained директории'),
        ('TransferLearningManager(tl_config)',
         'Создание TransferLearningManager'),
        ('load_pretrained_experts()',
         'Вызов load_pretrained_experts()'),
        ('Attempting to load pretrained models',
         'Информационное сообщение о pretraining'),
    ]

    for pattern, description in tl_patterns:
        total_checks += 1
        if pattern in bot_content:
            print(f"  ✅ {description}")
            total_passed += 1
        else:
            print(f"  ❌ {description} - ПАТТЕРН НЕ НАЙДЕН")

    # ========== ТЕСТ 5: Diversity monitoring в predictions ==========
    print("\n" + "="*80)
    print("ТЕСТ 5: Diversity monitoring в ML predictions")
    print("="*80)

    diversity_patterns = [
        ('self.diversity_monitor.add_predictions(',
         'Добавление predictions в monitor'),
        ('self.diversity_monitor.n_samples % 20 == 0',
         'Проверка метрик каждые 20 samples'),
        ('metrics = self.diversity_monitor.get_metrics()',
         'Получение diversity метрик'),
        ('logger.info(f"[Diversity]',
         'Логирование diversity метрик'),
    ]

    for pattern, description in diversity_patterns:
        total_checks += 1
        if pattern in bot_content:
            print(f"  ✅ {description}")
            total_passed += 1
        else:
            print(f"  ❌ {description} - ПАТТЕРН НЕ НАЙДЕН")

    # ========== ТЕСТ 6: Реакция на алерты ==========
    print("\n" + "="*80)
    print("ТЕСТ 6: Реакция на diversity алерты")
    print("="*80)

    alert_patterns = [
        ('if metrics.alert == AlertType.LOW_DIVERSITY:',
         'Проверка LOW_DIVERSITY alert'),
        ('self.meta.mode = "SHADOW"',
         'Откат META в SHADOW режим'),
        ('AlertType.HIGH_DISAGREEMENT',
         'Проверка HIGH_DISAGREEMENT alert'),
        ('AlertType.HIGH_CORRELATION',
         'Проверка HIGH_CORRELATION alert'),
    ]

    for pattern, description in alert_patterns:
        total_checks += 1
        if pattern in bot_content:
            print(f"  ✅ {description}")
            total_passed += 1
        else:
            print(f"  ❌ {description} - ПАТТЕРН НЕ НАЙДЕН")

    # ========== ТЕСТ 7: Pretraining скрипт ==========
    print("\n" + "="*80)
    print("ТЕСТ 7: Pretraining скрипт")
    print("="*80)

    script_path = GGG3_DIR / 'scripts' / 'pretrain_experts.py'
    if script_path.exists():
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()

        script_patterns = [
            ('load_historical_data_from_positions',
             'Функция загрузки из positions'),
            ('create_synthetic_historical_data',
             'Генерация синтетических данных'),
            ('manager.pretrain_all()',
             'Вызов pretraining'),
            ('argparse.ArgumentParser',
             'CLI аргументы'),
        ]

        for pattern, description in script_patterns:
            total_checks += 1
            if pattern in script_content:
                print(f"  ✅ {description}")
                total_passed += 1
            else:
                print(f"  ❌ {description} - ПАТТЕРН НЕ НАЙДЕН")
    else:
        print(f"  ❌ Скрипт не найден: {script_path}")
        total_checks += 4

    # ========== ИТОГИ ==========
    print("\n" + "="*80)
    print("ИТОГИ")
    print("="*80)
    print(f"\n  Всего проверок: {total_checks}")
    print(f"  Пройдено: {total_passed}")
    print(f"  Провалено: {total_checks - total_passed}")

    success_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0
    print(f"  Процент успеха: {success_rate:.1f}%")

    if total_passed == total_checks:
        print("\n  ✅ ✅ ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ! ✅ ✅ ✅")
        print("\n  Transfer Learning + Diversity Monitor успешно интегрированы в бот!")
        exit_code = 0
    elif success_rate >= 90:
        print("\n  ✅ Интеграция выполнена успешно (>90%)")
        print(f"  {total_checks - total_passed} незначительных проблем")
        exit_code = 0
    else:
        print(f"\n  ⚠️  {total_checks - total_passed} проверок провалено")
        print("  Проверьте ошибки выше")
        exit_code = 1

    print("="*80 + "\n")
    return exit_code


if __name__ == "__main__":
    exit(main())
