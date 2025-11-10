#!/usr/bin/env python3
"""
pretrain_experts.py — Предобучение ML экспертов на исторических данных

Решает проблему холодного старта (cold start) путем обучения экспертов
на исторических данных перед запуском бота.

Workflow:
1. Загружает исторические данные из closed_positions.json
2. Извлекает features и targets (цена выросла/упала)
3. Pretrain экспертов (XGBoost, RandomForest, NeuralNet)
4. Сохраняет pretrained модели в models/pretrained/

Usage:
    python3 scripts/pretrain_experts.py
    python3 scripts/pretrain_experts.py --data-source synthetic  # Для тестирования

После pretraining бот автоматически загрузит pretrained модели при старте.

Автор: Claude (Anthropic)
Дата: 2025-11-10
"""
import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
from typing import Dict, List, Tuple

# Добавляем путь к GGG3
SCRIPT_DIR = Path(__file__).parent
GGG3_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(GGG3_DIR))

# Импорты
from models.transfer_learning import TransferLearningManager, PretrainingConfig, create_synthetic_historical_data


def load_historical_data_from_positions(positions_file: str = 'data/closed_positions.json') -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает исторические данные из closed_positions.json

    Args:
        positions_file: Путь к файлу с закрытыми позициями

    Returns:
        (X, y): Features и targets
    """
    positions_path = GGG3_DIR / positions_file

    if not positions_path.exists():
        raise FileNotFoundError(
            f"Файл {positions_path} не найден!\n"
            f"Запустите бот и дождитесь накопления позиций, "
            f"либо используйте --data-source synthetic для тестирования."
        )

    with open(positions_path, 'r') as f:
        positions = json.load(f)

    if not positions:
        raise ValueError("Нет закрытых позиций для обучения!")

    print(f"[Pretraining] Загружено {len(positions)} закрытых позиций")

    # Извлекаем features и targets
    X_list = []
    y_list = []

    for pos in positions:
        # Проверяем наличие entry_snapshot
        if 'entry_snapshot' not in pos:
            continue

        snapshot = pos['entry_snapshot']

        # Извлекаем features (68D вектор)
        if 'features' not in snapshot:
            continue

        features = np.array(snapshot['features'], dtype=np.float32)

        # Проверяем размерность
        if len(features) != 68:
            continue

        # Target: цена выросла (1) или упала (0)
        # ВАЖНО: Учитываем направление позиции
        pnl = pos.get('pnl', 0)
        direction = pos.get('direction', 'LONG')

        if direction == 'LONG':
            # LONG: PNL > 0 → цена выросла
            target = 1.0 if pnl > 0 else 0.0
        else:  # SHORT
            # SHORT: PNL > 0 → цена упала
            target = 0.0 if pnl > 0 else 1.0

        X_list.append(features)
        y_list.append(target)

    if len(X_list) == 0:
        raise ValueError(
            "Не удалось извлечь features из позиций!\n"
            "Убедитесь что позиции содержат entry_snapshot с features."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"[Pretraining] Извлечено:")
    print(f"  Примеров: {len(X)}")
    print(f"  Features: {X.shape[1]}D")
    print(f"  Class balance: {y.mean():.2%} (UP)")

    return X, y


def save_historical_data(X: np.ndarray, y: np.ndarray, save_path: str):
    """Сохраняет исторические данные в NPZ формате"""
    np.savez(save_path, X=X, y=y)
    print(f"[Pretraining] Сохранено в: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Pretraining ML экспертов')
    parser.add_argument(
        '--data-source',
        type=str,
        choices=['positions', 'synthetic'],
        default='positions',
        help='Источник данных: positions (из бота) или synthetic (генерация)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=100,
        help='Минимальное количество примеров для pretraining'
    )
    parser.add_argument(
        '--experts',
        nargs='+',
        default=['xgb', 'rf', 'nn'],
        help='Какие эксперты обучать (xgb, rf, nn)'
    )

    args = parser.parse_args()

    print("="*80)
    print("🎓 TRANSFER LEARNING: PRETRAINING ML ЭКСПЕРТОВ")
    print("="*80)

    # Пути
    data_dir = GGG3_DIR / 'data'
    models_dir = GGG3_DIR / 'models'
    pretrained_dir = models_dir / 'pretrained'

    data_dir.mkdir(exist_ok=True)
    pretrained_dir.mkdir(exist_ok=True)

    historical_data_path = data_dir / 'historical_data.npz'

    # ========== ШАГ 1: Загрузка/генерация данных ==========
    print(f"\n{'='*80}")
    print(f"ШАГ 1: Подготовка исторических данных")
    print(f"{'='*80}")

    if args.data_source == 'positions':
        # Загружаем из закрытых позиций
        try:
            X, y = load_historical_data_from_positions()

            # Проверка минимального количества
            if len(X) < args.min_samples:
                print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Недостаточно данных!")
                print(f"  Требуется минимум: {args.min_samples}")
                print(f"  Доступно: {len(X)}")
                print(f"  Рекомендация: Запустите бот и накопите больше позиций")
                print(f"  Альтернатива: Используйте --data-source synthetic для тестирования\n")

                response = input("Продолжить с малым количеством данных? (y/n): ")
                if response.lower() != 'y':
                    print("Отменено.")
                    return

            # Сохраняем
            save_historical_data(X, y, str(historical_data_path))

        except Exception as e:
            print(f"\n❌ Ошибка загрузки данных: {e}")
            print(f"Используйте --data-source synthetic для тестирования")
            return

    else:  # synthetic
        # Генерация синтетических данных
        print(f"[Pretraining] Генерация синтетических данных...")
        create_synthetic_historical_data(
            n_samples=5000,
            n_features=68,
            save_path=str(historical_data_path),
            random_state=42
        )

    # ========== ШАГ 2: Pretraining экспертов ==========
    print(f"\n{'='*80}")
    print(f"ШАГ 2: Pretraining экспертов")
    print(f"{'='*80}")

    config = PretrainingConfig(
        historical_data_path=str(historical_data_path),
        experts_to_pretrain=args.experts,
        pretrained_save_dir=str(pretrained_dir),
        validation_split=0.2,
        verbose=True
    )

    manager = TransferLearningManager(config)

    print(f"\n⚠️  Pretraining может занять 1-5 минут...\n")

    try:
        manager.pretrain_all()
    except Exception as e:
        print(f"\n❌ Ошибка pretraining: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== ШАГ 3: Проверка результатов ==========
    print(f"\n{'='*80}")
    print(f"✅ PRETRAINING ЗАВЕРШЕН")
    print(f"{'='*80}")

    print(f"\nPretrained модели сохранены в: {pretrained_dir}")
    print(f"Содержимое:")

    for expert_name in args.experts:
        model_path = pretrained_dir / f"{expert_name}_pretrained.pkl"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {expert_name}_pretrained.pkl ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {expert_name}_pretrained.pkl (не найден)")

    print(f"\n{'='*80}")
    print(f"🚀 ГОТОВО К ИСПОЛЬЗОВАНИЮ")
    print(f"{'='*80}")
    print(f"\nПри следующем запуске бот автоматически загрузит pretrained модели:")
    print(f"  python3 binance_bot_main.py")
    print(f"\nЭксперты будут работать с первого дня благодаря Transfer Learning! 🎓")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
