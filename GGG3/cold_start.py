#!/usr/bin/env python3
"""
Холодный старт бота - полный сброс состояния

Удаляет все данные о работе бота и начинает с нуля:
- Позиции (открытые и закрытые)
- Баланс PaperExchange
- История реинвестирования
- История капитала
- Логи (опционально)
- Модели META (опционально)

Usage:
    python cold_start.py              # Базовый сброс
    python cold_start.py --full       # Полный сброс (включая логи и модели)
    python cold_start.py --keep-logs  # Сбросить всё кроме логов
"""

import os
import sys
import shutil
from pathlib import Path
import argparse
from datetime import datetime
import json


def print_header(text):
    """Красивый заголовок"""
    print("\n" + "="*80)
    print(f"🔄 {text}")
    print("="*80)


def print_success(text):
    """Успешное действие"""
    print(f"  ✅ {text}")


def print_warning(text):
    """Предупреждение"""
    print(f"  ⚠️  {text}")


def print_info(text):
    """Информация"""
    print(f"  ℹ️  {text}")


def delete_file(file_path: Path, description: str) -> bool:
    """
    Удаляет файл если он существует

    Args:
        file_path: Путь к файлу
        description: Описание файла

    Returns:
        True если файл был удалён, False если не существовал
    """
    if file_path.exists():
        file_path.unlink()
        print_success(f"{description} удалён")
        return True
    else:
        print_info(f"{description} не найден (пропускаем)")
        return False


def delete_directory(dir_path: Path, description: str) -> bool:
    """
    Удаляет директорию если она существует

    Args:
        dir_path: Путь к директории
        description: Описание директории

    Returns:
        True если директория была удалена, False если не существовала
    """
    if dir_path.exists():
        shutil.rmtree(dir_path)
        print_success(f"{description} удалена")
        return True
    else:
        print_info(f"{description} не найдена (пропускаем)")
        return False


def create_backup(base_dir: Path) -> Path:
    """
    Создает резервную копию данных перед удалением

    Args:
        base_dir: Базовая директория (GGG3)

    Returns:
        Путь к директории с бэкапом
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = base_dir / f"backup_{timestamp}"

    print_header("СОЗДАНИЕ РЕЗЕРВНОЙ КОПИИ")

    # Создаем директорию для бэкапа
    backup_dir.mkdir(exist_ok=True)

    # Файлы для бэкапа
    files_to_backup = [
        ('data/positions.json', 'Позиции'),
        ('data/paper_exchange_state.json', 'Баланс PaperExchange'),
        ('data/reinvestment_state.json', 'Реинвестирование'),
        ('data/capital_history.json', 'История капитала'),
    ]

    backed_up = 0
    for file_rel_path, description in files_to_backup:
        src = base_dir / file_rel_path
        if src.exists():
            dst = backup_dir / file_rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print_success(f"{description} скопирован")
            backed_up += 1

    if backed_up > 0:
        print_success(f"Резервная копия создана: {backup_dir}")
        return backup_dir
    else:
        print_warning("Нет файлов для бэкапа")
        backup_dir.rmdir()
        return None


def cold_start(
    keep_logs: bool = False,
    keep_models: bool = True,
    create_backup_first: bool = True
):
    """
    Выполняет холодный старт бота

    Args:
        keep_logs: Сохранить логи
        keep_models: Сохранить обученные модели
        create_backup_first: Создать бэкап перед удалением
    """

    # Определяем базовую директорию (GGG3)
    base_dir = Path(__file__).parent

    print_header("ХОЛОДНЫЙ СТАРТ БОТА")
    print(f"Базовая директория: {base_dir}")
    print(f"Сохранить логи: {keep_logs}")
    print(f"Сохранить модели: {keep_models}")

    # Запрос подтверждения
    print("\n⚠️  ВНИМАНИЕ: Это удалит все данные о работе бота!")
    response = input("Продолжить? (yes/no): ").strip().lower()

    if response not in ['yes', 'y', 'да', 'д']:
        print("\n❌ Отменено пользователем")
        return

    # Создаем бэкап
    backup_path = None
    if create_backup_first:
        backup_path = create_backup(base_dir)

    # ============================================================================
    # Удаление данных
    # ============================================================================

    print_header("УДАЛЕНИЕ ДАННЫХ О ПОЗИЦИЯХ")

    data_dir = base_dir / 'data'

    # Позиции
    delete_file(data_dir / 'positions.json', "Позиции (positions.json)")

    # Paper Exchange
    delete_file(data_dir / 'paper_exchange_state.json', "Баланс PaperExchange")

    # Реинвестирование
    delete_file(data_dir / 'reinvestment_state.json', "История реинвестирования")

    # История капитала
    delete_file(data_dir / 'capital_history.json', "История капитала")

    # Логи
    if not keep_logs:
        print_header("УДАЛЕНИЕ ЛОГОВ")
        delete_file(base_dir / 'trading_bot.log', "Лог бота (trading_bot.log)")
        delete_directory(base_dir / 'logs', "Директория логов")
    else:
        print_header("ЛОГИ")
        print_info("Логи сохранены (--keep-logs)")

    # Модели META
    if not keep_models:
        print_header("УДАЛЕНИЕ МОДЕЛЕЙ META")
        models_dir = base_dir / 'models' / 'saved'

        meta_files = [
            'meta_model.json',
            'meta_model.pkl',
            'meta_cem.json',
            'meta_cem.pkl',
            'meta_online_buffer.pkl'
        ]

        for model_file in meta_files:
            delete_file(models_dir / model_file, f"META модель ({model_file})")
    else:
        print_header("МОДЕЛИ")
        print_info("Обученные модели сохранены")

    # ============================================================================
    # Создание свежих файлов
    # ============================================================================

    print_header("ИНИЦИАЛИЗАЦИЯ ЧИСТОГО СОСТОЯНИЯ")

    # Создаем data директорию если не существует
    data_dir.mkdir(exist_ok=True)

    # Создаем пустой positions.json
    positions_data = {
        "open_positions": [],
        "closed_positions": [],
        "last_updated": datetime.now().isoformat()
    }

    with open(data_dir / 'positions.json', 'w') as f:
        json.dump(positions_data, f, indent=2)
    print_success("Создан чистый positions.json")

    # Создаем пустой paper_exchange_state.json
    # Читаем начальный капитал из конфига
    try:
        sys.path.insert(0, str(base_dir))
        import binance_config as config
        initial_capital = config.PAPER_INITIAL_BALANCE
    except:
        initial_capital = 1000.0
        print_warning(f"Не удалось прочитать конфиг, использую капитал по умолчанию: ${initial_capital}")

    exchange_data = {
        "balance": initial_capital,
        "initial_capital": initial_capital,
        "positions": {},
        "orders": {},
        "trades_history": [],
        "closed_positions": [],
        "slippage": 0.0005,
        "commission": 0.001,
        "last_updated": datetime.now().isoformat()
    }

    with open(data_dir / 'paper_exchange_state.json', 'w') as f:
        json.dump(exchange_data, f, indent=2)
    print_success(f"Создан чистый paper_exchange_state.json (баланс: ${initial_capital})")

    # Создаем пустой capital_history.json
    capital_data = {
        "history": [],
        "last_updated": datetime.now().isoformat()
    }

    with open(data_dir / 'capital_history.json', 'w') as f:
        json.dump(capital_data, f, indent=2)
    print_success("Создан чистый capital_history.json")

    # ============================================================================
    # Итоги
    # ============================================================================

    print_header("ХОЛОДНЫЙ СТАРТ ЗАВЕРШЁН")

    print(f"\n✅ Бот готов к запуску с чистого листа!")
    print(f"\n📊 Начальное состояние:")
    print(f"  • Баланс: ${initial_capital:,.2f}")
    print(f"  • Открытые позиции: 0")
    print(f"  • Закрытые позиции: 0")
    print(f"  • Резервный фонд: $0.00")

    if backup_path:
        print(f"\n💾 Резервная копия сохранена: {backup_path}")
        print(f"   (можно удалить если не нужна)")

    print(f"\n🚀 Запуск бота:")
    print(f"   python binance_bot_main.py --paper")

    print("="*80 + "\n")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Холодный старт бота - полный сброс состояния',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python cold_start.py              # Базовый сброс (сохранить модели и логи)
  python cold_start.py --full       # Полный сброс (удалить всё включая модели)
  python cold_start.py --keep-logs  # Сбросить всё кроме логов
  python cold_start.py --no-backup  # Не создавать резервную копию
        """
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='Полный сброс (включая модели META и логи)'
    )

    parser.add_argument(
        '--keep-logs',
        action='store_true',
        help='Сохранить логи'
    )

    parser.add_argument(
        '--keep-models',
        action='store_true',
        help='Сохранить обученные модели META'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Не создавать резервную копию'
    )

    args = parser.parse_args()

    # Определяем параметры
    if args.full:
        # Полный сброс - удаляем всё
        keep_logs = False
        keep_models = False
    else:
        # Базовый сброс - сохраняем модели, логи опционально
        keep_logs = args.keep_logs
        keep_models = True if not args.full else args.keep_models

    create_backup_first = not args.no_backup

    # Выполняем холодный старт
    cold_start(
        keep_logs=keep_logs,
        keep_models=keep_models,
        create_backup_first=create_backup_first
    )


if __name__ == "__main__":
    main()
