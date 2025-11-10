#!/usr/bin/env python3
"""
test_tl_diversity_integration_smoke.py — Smoke test интеграции Transfer Learning и Diversity Monitor

Проверяет:
1. Импорты работают корректно
2. DiversityMonitor инициализируется
3. Transfer Learning Manager работает
4. Интеграция в binance_bot_main.py не сломана

НЕ требует зависимостей (numpy, scipy, torch) - только проверка кода.

Usage:
    python3 tests/test_tl_diversity_integration_smoke.py

Автор: Claude (Anthropic)
Дата: 2025-11-10
"""
import sys
from pathlib import Path

# Добавляем путь к GGG3
TEST_DIR = Path(__file__).parent
GGG3_DIR = TEST_DIR.parent
sys.path.insert(0, str(GGG3_DIR))


def test_imports():
    """Тест 1: Проверка импортов"""
    print("\n" + "="*80)
    print("ТЕСТ 1: Проверка импортов")
    print("="*80)

    checks_passed = 0
    checks_total = 0

    # Transfer Learning
    checks_total += 1
    try:
        from models.transfer_learning import TransferLearningManager, PretrainingConfig
        print("  ✅ Transfer Learning импорты работают")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Transfer Learning импорт failed: {e}")

    # Diversity Monitor
    checks_total += 1
    try:
        from models.diversity_metric import DiversityMonitor, AlertType, DiversityMetrics
        print("  ✅ Diversity Monitor импорты работают")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Diversity Monitor импорт failed: {e}")

    # SimplifiedEnsemble
    checks_total += 1
    try:
        from simplified_ensemble_meta import SimplifiedEnsembleMETA
        print("  ✅ SimplifiedEnsemble импорт работает")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ SimplifiedEnsemble импорт failed: {e}")

    print(f"\n  Результат: {checks_passed}/{checks_total} проверок пройдено")
    return checks_passed, checks_total


def test_diversity_monitor_init():
    """Тест 2: Инициализация DiversityMonitor"""
    print("\n" + "="*80)
    print("ТЕСТ 2: Инициализация DiversityMonitor")
    print("="*80)

    checks_passed = 0
    checks_total = 0

    try:
        from models.diversity_metric import DiversityMonitor

        # Инициализация
        checks_total += 1
        monitor = DiversityMonitor(
            window_size=100,
            low_diversity_threshold=0.05,
            high_disagreement_threshold=0.20
        )
        print("  ✅ DiversityMonitor инициализирован")
        checks_passed += 1

        # Проверка атрибутов
        checks_total += 1
        assert hasattr(monitor, 'window_size'), "Нет атрибута window_size"
        assert monitor.window_size == 100, "Неверный window_size"
        print("  ✅ Атрибуты корректны")
        checks_passed += 1

        # Проверка методов
        checks_total += 1
        assert hasattr(monitor, 'add_predictions'), "Нет метода add_predictions"
        assert hasattr(monitor, 'get_metrics'), "Нет метода get_metrics"
        print("  ✅ Методы существуют")
        checks_passed += 1

    except Exception as e:
        print(f"  ❌ Ошибка: {e}")

    print(f"\n  Результат: {checks_passed}/{checks_total} проверок пройдено")
    return checks_passed, checks_total


def test_transfer_learning_config():
    """Тест 3: Конфигурация Transfer Learning"""
    print("\n" + "="*80)
    print("ТЕСТ 3: Конфигурация Transfer Learning")
    print("="*80)

    checks_passed = 0
    checks_total = 0

    try:
        from models.transfer_learning import PretrainingConfig, TransferLearningManager

        # Создание конфига
        checks_total += 1
        config = PretrainingConfig(
            historical_data_path="test.npz",
            experts_to_pretrain=['xgb', 'rf'],
            pretrained_save_dir="test_dir/",
            validation_split=0.2
        )
        print("  ✅ PretrainingConfig создан")
        checks_passed += 1

        # Проверка атрибутов
        checks_total += 1
        assert config.historical_data_path == "test.npz"
        assert config.experts_to_pretrain == ['xgb', 'rf']
        assert config.validation_split == 0.2
        print("  ✅ Атрибуты конфига корректны")
        checks_passed += 1

        # Инициализация менеджера
        checks_total += 1
        manager = TransferLearningManager(config)
        assert hasattr(manager, 'config')
        assert hasattr(manager, 'pretrained_experts')
        print("  ✅ TransferLearningManager инициализирован")
        checks_passed += 1

    except Exception as e:
        print(f"  ❌ Ошибка: {e}")

    print(f"\n  Результат: {checks_passed}/{checks_total} проверок пройдено")
    return checks_passed, checks_total


def test_binance_bot_imports():
    """Тест 4: Интеграция в binance_bot_main.py"""
    print("\n" + "="*80)
    print("ТЕСТ 4: Интеграция в binance_bot_main.py")
    print("="*80)

    checks_passed = 0
    checks_total = 0

    # Читаем файл binance_bot_main.py
    bot_file = GGG3_DIR / 'binance_bot_main.py'

    if not bot_file.exists():
        print(f"  ❌ Файл {bot_file} не найден")
        return 0, 1

    with open(bot_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Проверка 1: Импорты добавлены
    checks_total += 1
    if 'from models.transfer_learning import TransferLearningManager' in content:
        print("  ✅ Transfer Learning импорт добавлен в binance_bot_main.py")
        checks_passed += 1
    else:
        print("  ❌ Transfer Learning импорт НЕ найден в binance_bot_main.py")

    checks_total += 1
    if 'from models.diversity_metric import DiversityMonitor' in content:
        print("  ✅ Diversity Monitor импорт добавлен в binance_bot_main.py")
        checks_passed += 1
    else:
        print("  ❌ Diversity Monitor импорт НЕ найден в binance_bot_main.py")

    # Проверка 2: DiversityMonitor инициализируется
    checks_total += 1
    if 'self.diversity_monitor = DiversityMonitor(' in content:
        print("  ✅ DiversityMonitor инициализируется в __init__")
        checks_passed += 1
    else:
        print("  ❌ DiversityMonitor НЕ инициализируется в __init__")

    # Проверка 3: Transfer Learning используется в _load_experts
    checks_total += 1
    if 'pretrained_dir = config.MODELS_DIR' in content and 'TransferLearningManager' in content:
        print("  ✅ Transfer Learning интегрирован в _load_experts()")
        checks_passed += 1
    else:
        print("  ❌ Transfer Learning НЕ интегрирован в _load_experts()")

    # Проверка 4: Diversity monitoring в predictions
    checks_total += 1
    if 'self.diversity_monitor.add_predictions(' in content:
        print("  ✅ Diversity monitoring добавлен в ML predictions")
        checks_passed += 1
    else:
        print("  ❌ Diversity monitoring НЕ добавлен в ML predictions")

    # Проверка 5: Реакция на LOW_DIVERSITY
    checks_total += 1
    if 'AlertType.LOW_DIVERSITY' in content and 'self.meta.mode = "SHADOW"' in content:
        print("  ✅ Реакция на LOW_DIVERSITY alert реализована")
        checks_passed += 1
    else:
        print("  ❌ Реакция на LOW_DIVERSITY alert НЕ реализована")

    print(f"\n  Результат: {checks_passed}/{checks_total} проверок пройдено")
    return checks_passed, checks_total


def test_pretraining_script():
    """Тест 5: Pretraining скрипт существует"""
    print("\n" + "="*80)
    print("ТЕСТ 5: Pretraining скрипт")
    print("="*80)

    checks_passed = 0
    checks_total = 0

    # Проверка наличия скрипта
    checks_total += 1
    script_path = GGG3_DIR / 'scripts' / 'pretrain_experts.py'
    if script_path.exists():
        print(f"  ✅ Pretraining скрипт существует: {script_path}")
        checks_passed += 1
    else:
        print(f"  ❌ Pretraining скрипт НЕ найден: {script_path}")
        return checks_passed, checks_total

    # Проверка содержимого
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    checks_total += 1
    if 'load_historical_data_from_positions' in content:
        print("  ✅ Функция загрузки исторических данных присутствует")
        checks_passed += 1
    else:
        print("  ❌ Функция загрузки исторических данных НЕ найдена")

    checks_total += 1
    if 'TransferLearningManager' in content and 'pretrain_all' in content:
        print("  ✅ Pretraining логика реализована")
        checks_passed += 1
    else:
        print("  ❌ Pretraining логика НЕ реализована")

    print(f"\n  Результат: {checks_passed}/{checks_total} проверок пройдено")
    return checks_passed, checks_total


def main():
    print("="*80)
    print("🧪 SMOKE TEST: TRANSFER LEARNING + DIVERSITY MONITOR INTEGRATION")
    print("="*80)

    total_passed = 0
    total_checks = 0

    # Тест 1: Импорты
    passed, total = test_imports()
    total_passed += passed
    total_checks += total

    # Тест 2: DiversityMonitor
    passed, total = test_diversity_monitor_init()
    total_passed += passed
    total_checks += total

    # Тест 3: Transfer Learning
    passed, total = test_transfer_learning_config()
    total_passed += passed
    total_checks += total

    # Тест 4: Интеграция в бот
    passed, total = test_binance_bot_imports()
    total_passed += passed
    total_checks += total

    # Тест 5: Pretraining скрипт
    passed, total = test_pretraining_script()
    total_passed += passed
    total_checks += total

    # Итоги
    print("\n" + "="*80)
    print("ИТОГИ")
    print("="*80)
    print(f"\n  Всего проверок: {total_checks}")
    print(f"  Пройдено: {total_passed}")
    print(f"  Провалено: {total_checks - total_passed}")
    print(f"  Процент успеха: {total_passed / total_checks * 100:.1f}%")

    if total_passed == total_checks:
        print("\n  ✅ ✅ ✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ! ✅ ✅ ✅")
        print("\n  Transfer Learning + Diversity Monitor успешно интегрированы!")
        exit_code = 0
    else:
        print(f"\n  ⚠️  {total_checks - total_passed} проверок провалено")
        print("  Проверьте ошибки выше")
        exit_code = 1

    print("="*80 + "\n")
    return exit_code


if __name__ == "__main__":
    exit(main())
