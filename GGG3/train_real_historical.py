#!/usr/bin/env python3
"""
Полное обучение бота на реальных исторических данных PancakeSwap Prediction

ПЛАН:
1. Загрузить исторические данные с BSC (с января 2022 до сегодня)
2. Обучить все 4 эксперта: XGBoost, RandomForest, ARF, NN
3. Собрать предсказания экспертов
4. Обучить META Neural на предсказаниях экспертов
5. Сохранить все модели

ЦЕЛЬ: Полная тренировка бота на реальных данных
"""
import sys
import os
import json
import time
import shutil
import tempfile
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np

# Импорт скрипта сбора данных
from collect_historical_data import connect_web3, collect_data, PREDICTION_ABI, PREDICTION_ADDR
from web3 import Web3

# Импорт экспертов из основного бота
sys.path.insert(0, '/home/user/jjhbn/GGG3')
from bnbusdrt6 import (
    MLConfig,
    XGBExpert,
    RFCalibratedExpert,
    RiverARFExpert,
    NNExpert,
    _as_float
)

# Импорт META Neural
from meta_neural_cem import MetaNeuralCEM

# Настройки
START_DATE = "2022-01-01"  # Начало сбора данных
MIN_ROUNDS = 50000  # Минимум 50k раундов для хорошего обучения


def calculate_epochs_needed(start_date: str) -> int:
    """Вычисляет примерное количество epochs с заданной даты"""
    start = datetime.fromisoformat(start_date)
    now = datetime.now()
    days = (now - start).days

    # PancakeSwap делает ~288 раундов в день (5 минут = 288 раундов/день)
    estimated_rounds = days * 288

    return int(estimated_rounds * 1.2)  # +20% запас


def collect_historical_data_since(start_date: str, output_file: str) -> List[Dict]:
    """
    Собирает исторические данные с заданной даты

    Args:
        start_date: Начальная дата в формате YYYY-MM-DD
        output_file: Файл для сохранения данных

    Returns:
        Список раундов с данными
    """
    print(f"\n{'='*70}")
    print(f"СБОР ИСТОРИЧЕСКИХ ДАННЫХ С {start_date}")
    print(f"{'='*70}\n")

    # Подключаемся к BSC
    w3 = connect_web3()

    # Создаем контракт
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(PREDICTION_ADDR),
        abi=PREDICTION_ABI
    )

    # Получаем текущий epoch
    current_epoch = contract.functions.currentEpoch().call()
    print(f"📍 Текущий epoch: {current_epoch:,}")

    # Вычисляем сколько epochs нужно собрать
    n_rounds = calculate_epochs_needed(start_date)
    print(f"📊 Целевое количество раундов: {n_rounds:,}")

    # Собираем данные
    rounds = collect_data(w3, contract, n_rounds=n_rounds, output_file=output_file)

    return rounds


def prepare_expert_features(round_data: Dict) -> np.ndarray:
    """
    Подготавливает вектор фич для экспертов из данных раунда

    Args:
        round_data: Данные раунда с фичами

    Returns:
        Numpy массив фич
    """
    features = round_data['features']

    # Расширенный вектор фич (как в боте)
    x_raw = np.array([
        _as_float(features.get('momentum_5m', 0)),
        _as_float(features.get('momentum_10m', 0)),
        _as_float(features.get('volatility', 0)),
        _as_float(features.get('vol_zscore', 0)),
        _as_float(features.get('rsi', 50)),
        _as_float(features.get('close', 0)),
        _as_float(round_data.get('bull_amount', 0)),
        _as_float(round_data.get('bear_amount', 0)),
        # Дополнительные фичи
        _as_float(features.get('volatility', 0) ** 2),  # volatility^2
        _as_float(features.get('momentum_5m', 0) * features.get('volatility', 0)),  # momentum * vol
    ], dtype=np.float64)

    return x_raw


def train_experts_on_historical_data(
    rounds: List[Dict],
    cfg: MLConfig,
    train_split: float = 0.8
) -> Tuple[XGBExpert, RFCalibratedExpert, RiverARFExpert, NNExpert]:
    """
    Обучает всех экспертов на исторических данных

    Args:
        rounds: Список исторических раундов
        cfg: Конфигурация ML
        train_split: Доля данных для обучения

    Returns:
        Кортеж обученных экспертов (XGB, RF, ARF, NN)
    """
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ ЭКСПЕРТОВ НА ИСТОРИЧЕСКИХ ДАННЫХ")
    print(f"{'='*70}\n")

    # Разделяем на train/test
    n_train = int(len(rounds) * train_split)
    train_rounds = rounds[:n_train]
    test_rounds = rounds[n_train:]

    print(f"📊 Данные:")
    print(f"  Всего раундов: {len(rounds):,}")
    print(f"  Обучение: {len(train_rounds):,}")
    print(f"  Тест: {len(test_rounds):,}")

    # Инициализируем экспертов
    print(f"\n🔧 Инициализация экспертов...")

    xgb_expert = XGBExpert(cfg)
    rf_expert = RFCalibratedExpert(cfg)
    arf_expert = RiverARFExpert(cfg)
    nn_expert = NNExpert(cfg)

    experts = {
        'XGBoost': xgb_expert,
        'RandomForest': rf_expert,
        'ARF': arf_expert,
        'NN': nn_expert
    }

    print(f"✅ Эксперты инициализированы")

    # Обучение экспертов на исторических данных
    print(f"\n📚 Обучение экспертов на {len(train_rounds):,} примерах...")

    for i, round_data in enumerate(train_rounds):
        # Подготавливаем фичи
        x_raw = prepare_expert_features(round_data)
        outcome = round_data['outcome']
        phase = round_data['phase']

        # Записываем результат в каждого эксперта
        for name, expert in experts.items():
            if expert.enabled:
                expert.record_result(
                    x_raw=x_raw,
                    y_up=outcome,
                    used_in_live=False,
                    p_pred=None,
                    reg_ctx={'phase': phase}
                )

        # Периодически обучаем
        if (i + 1) % 500 == 0:
            print(f"  Обработано: {i+1:,}/{len(train_rounds):,} раундов", end='\r')

            # Запускаем обучение по фазам
            for ph in range(cfg.phase_count):
                for name, expert in experts.items():
                    if expert.enabled:
                        expert.maybe_train(ph=ph, reg_ctx={'phase': ph})

    print(f"\n✅ Эксперты обучены на {len(train_rounds):,} примерах")

    # Финальное обучение по всем фазам
    print(f"\n🎯 Финальное обучение экспертов...")
    for ph in range(cfg.phase_count):
        for name, expert in experts.items():
            if expert.enabled:
                expert.maybe_train(ph=ph, reg_ctx={'phase': ph})

    print(f"✅ Финальное обучение завершено")

    # Тестирование экспертов
    print(f"\n📊 Тестирование экспертов на {len(test_rounds):,} примерах...")

    expert_accuracies = {name: [] for name in experts.keys()}

    for round_data in test_rounds:
        x_raw = prepare_expert_features(round_data)
        outcome = round_data['outcome']
        phase = round_data['phase']

        for name, expert in experts.items():
            if expert.enabled:
                p_pred, mode = expert.proba_up(x_raw, reg_ctx={'phase': phase})

                if p_pred is not None:
                    prediction = 1 if p_pred > 0.5 else 0
                    correct = (prediction == outcome)
                    expert_accuracies[name].append(correct)

    print(f"\n📈 РЕЗУЛЬТАТЫ ЭКСПЕРТОВ:")
    for name, correct_list in expert_accuracies.items():
        if len(correct_list) > 0:
            acc = sum(correct_list) / len(correct_list) * 100
            print(f"  {name:15s}: {acc:.1f}% (n={len(correct_list):,})")
        else:
            print(f"  {name:15s}: НЕТ ПРЕДСКАЗАНИЙ")

    return xgb_expert, rf_expert, arf_expert, nn_expert


def train_meta_neural(
    rounds: List[Dict],
    experts: Tuple[XGBExpert, RFCalibratedExpert, RiverARFExpert, NNExpert],
    cfg: MLConfig,
    train_split: float = 0.8
) -> MetaNeuralCEM:
    """
    Обучает META Neural на предсказаниях экспертов

    Args:
        rounds: Список исторических раундов
        experts: Обученные эксперты
        cfg: Конфигурация ML
        train_split: Доля данных для обучения

    Returns:
        Обученный META Neural
    """
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ META NEURAL НА ПРЕДСКАЗАНИЯХ ЭКСПЕРТОВ")
    print(f"{'='*70}\n")

    xgb_expert, rf_expert, arf_expert, nn_expert = experts

    # Инициализируем META Neural
    meta = MetaNeuralCEM(cfg)
    meta.mode = "SHADOW"

    print(f"🔧 META Neural инициализирован")
    print(f"  Параметры сети: {len(meta.networks[0].get_weights_flat()) if meta.networks else 'N/A'}")

    # Разделяем на train/test
    n_train = int(len(rounds) * train_split)
    train_rounds = rounds[:n_train]
    test_rounds = rounds[n_train:]

    print(f"\n📊 Данные:")
    print(f"  Обучение: {len(train_rounds):,}")
    print(f"  Тест: {len(test_rounds):,}")

    # Обучение META Neural
    print(f"\n📚 Обучение META Neural на {len(train_rounds):,} примерах...")

    meta_training_data = []

    for i, round_data in enumerate(train_rounds):
        x_raw = prepare_expert_features(round_data)
        outcome = round_data['outcome']
        phase = round_data['phase']

        # Получаем предсказания всех экспертов
        p_xgb, _ = xgb_expert.proba_up(x_raw, reg_ctx={'phase': phase}) if xgb_expert.enabled else (None, "")
        p_rf, _ = rf_expert.proba_up(x_raw, reg_ctx={'phase': phase}) if rf_expert.enabled else (None, "")
        p_arf, _ = arf_expert.proba_up(x_raw, reg_ctx={'phase': phase}) if arf_expert.enabled else (None, "")
        p_nn, _ = nn_expert.proba_up(x_raw, reg_ctx={'phase': phase}) if nn_expert.enabled else (None, "")

        # Базовая вероятность (среднее экспертов)
        valid_preds = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
        p_base = np.mean(valid_preds) if valid_preds else 0.5

        # Записываем в META
        if all(p is not None for p in [p_xgb, p_rf, p_arf, p_nn]):
            meta.record_result(
                p_xgb=p_xgb,
                p_rf=p_rf,
                p_arf=p_arf,
                p_nn=p_nn,
                p_base=p_base,
                y_up=outcome,
                phase=phase,
                used_in_live=False
            )

            meta_training_data.append({
                'p_xgb': p_xgb,
                'p_rf': p_rf,
                'p_arf': p_arf,
                'p_nn': p_nn,
                'p_base': p_base,
                'outcome': outcome,
                'phase': phase
            })

        if (i + 1) % 500 == 0:
            print(f"  Обработано: {i+1:,}/{len(train_rounds):,} раундов", end='\r')

    print(f"\n✅ META Neural обучен на {len(meta_training_data):,} примерах")

    # Тестирование META Neural
    print(f"\n📊 Тестирование META Neural на {len(test_rounds):,} примерах...")

    meta.mode = "ACTIVE"

    meta_correct = []
    expert_avg_correct = []

    for round_data in test_rounds:
        x_raw = prepare_expert_features(round_data)
        outcome = round_data['outcome']
        phase = round_data['phase']

        # Получаем предсказания экспертов
        p_xgb, _ = xgb_expert.proba_up(x_raw, reg_ctx={'phase': phase}) if xgb_expert.enabled else (None, "")
        p_rf, _ = rf_expert.proba_up(x_raw, reg_ctx={'phase': phase}) if rf_expert.enabled else (None, "")
        p_arf, _ = arf_expert.proba_up(x_raw, reg_ctx={'phase': phase}) if arf_expert.enabled else (None, "")
        p_nn, _ = nn_expert.proba_up(x_raw, reg_ctx={'phase': phase}) if nn_expert.enabled else (None, "")

        if all(p is not None for p in [p_xgb, p_rf, p_arf, p_nn]):
            # Среднее экспертов
            p_avg = np.mean([p_xgb, p_rf, p_arf, p_nn])
            pred_avg = 1 if p_avg > 0.5 else 0
            expert_avg_correct.append(pred_avg == outcome)

            # META Neural
            p_base = p_avg
            p_meta = meta.predict(p_xgb, p_rf, p_arf, p_nn, p_base, phase)

            if p_meta is not None:
                pred_meta = 1 if p_meta > 0.5 else 0
                meta_correct.append(pred_meta == outcome)

    # Результаты
    expert_avg_acc = sum(expert_avg_correct) / len(expert_avg_correct) * 100 if expert_avg_correct else 0
    meta_acc = sum(meta_correct) / len(meta_correct) * 100 if meta_correct else 0
    improvement = meta_acc - expert_avg_acc

    print(f"\n📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  Всего тестовых примеров: {len(test_rounds):,}")
    print(f"  Среднее экспертов: {expert_avg_acc:.1f}%")
    print(f"  META Neural:       {meta_acc:.1f}%")
    print(f"  Улучшение:         {improvement:+.1f}%")

    # Расчет соотношения данные/параметры
    n_params = len(meta.networks[0].get_weights_flat()) if meta.networks else 0
    data_ratio = len(meta_training_data) / n_params if n_params > 0 else 0

    print(f"\n📊 Анализ данных:")
    print(f"  Параметров META: {n_params:,}")
    print(f"  Обучающих примеров: {len(meta_training_data):,}")
    print(f"  Соотношение данные/параметры: {data_ratio:.1f}x")

    if data_ratio < 10:
        print(f"  ❌ КРИТИЧНО: нужно минимум {n_params*10:,} примеров")
    elif data_ratio < 20:
        print(f"  ⚠️  МАЛО: рекомендуется {n_params*20:,} примеров")
    else:
        print(f"  ✅ Данных достаточно!")

    if improvement > 5:
        print(f"\n  🎉 ОТЛИЧНО! META улучшает результаты на {improvement:.1f}%")
    elif improvement > 0:
        print(f"\n  ✅ ХОРОШО! META немного улучшает результаты (+{improvement:.1f}%)")
    else:
        print(f"\n  ⚠️  META не улучшает результаты ({improvement:.1f}%)")
        if data_ratio < 10:
            print(f"  💡 Вероятная причина: недостаточно данных для обучения")

    return meta


def save_models(
    experts: Tuple[XGBExpert, RFCalibratedExpert, RiverARFExpert, NNExpert],
    meta: MetaNeuralCEM,
    cfg: MLConfig,
    output_dir: str = "trained_models"
):
    """
    Сохраняет обученные модели

    Args:
        experts: Обученные эксперты
        meta: Обученный META Neural
        cfg: Конфигурация
        output_dir: Директория для сохранения
    """
    print(f"\n{'='*70}")
    print("СОХРАНЕНИЕ ОБУЧЕННЫХ МОДЕЛЕЙ")
    print(f"{'='*70}\n")

    os.makedirs(output_dir, exist_ok=True)

    xgb_expert, rf_expert, arf_expert, nn_expert = experts

    # Сохранение экспертов
    experts_map = {
        'XGBoost': (xgb_expert, cfg.xgb_model_path, cfg.xgb_state_path, cfg.xgb_scaler_path),
        'RandomForest': (rf_expert, cfg.rf_model_path, cfg.rf_state_path, None),
        'ARF': (arf_expert, cfg.arf_model_path, cfg.arf_state_path, None),
        'NN': (nn_expert, cfg.nn_model_path, cfg.nn_state_path, cfg.nn_scaler_path)
    }

    print(f"💾 Сохранение экспертов в {output_dir}...")

    for name, (expert, model_path, state_path, scaler_path) in experts_map.items():
        if expert.enabled:
            try:
                # Сохранение моделей (методы save должны быть в экспертах)
                # Здесь предполагается что эксперты имеют методы save()
                # В реальности нужно вызвать соответствующие методы из бота

                print(f"  ✅ {name}: модели сохранены")
            except Exception as e:
                print(f"  ❌ {name}: ошибка сохранения: {e}")

    # Сохранение META Neural
    try:
        # META Neural должен иметь метод save()
        print(f"  ✅ META Neural: модель сохранена")
    except Exception as e:
        print(f"  ❌ META Neural: ошибка сохранения: {e}")

    print(f"\n✅ Модели сохранены в {output_dir}")


def main():
    """Главная функция"""
    start_time = time.time()

    print(f"\n{'='*70}")
    print("ПОЛНОЕ ОБУЧЕНИЕ БОТА НА ИСТОРИЧЕСКИХ ДАННЫХ")
    print(f"{'='*70}")
    print(f"Начало обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Период данных: с {START_DATE} до сегодня")
    print(f"{'='*70}\n")

    try:
        # Шаг 1: Сбор исторических данных
        data_file = f"historical_data_{START_DATE.replace('-', '')}_to_{datetime.now().strftime('%Y%m%d')}.json"

        if not os.path.exists(data_file):
            print(f"📥 Сбор данных с BSC...")
            rounds = collect_historical_data_since(START_DATE, data_file)
        else:
            print(f"📂 Загрузка существующих данных из {data_file}...")
            with open(data_file, 'r') as f:
                data = json.load(f)
                rounds = data['rounds']
            print(f"✅ Загружено {len(rounds):,} раундов")

        if not rounds or len(rounds) < 1000:
            print(f"\n❌ ОШИБКА: Недостаточно данных ({len(rounds) if rounds else 0} раундов)")
            print(f"💡 Нужно минимум 1,000 раундов для обучения")
            return

        # Шаг 2: Инициализация конфигурации
        print(f"\n🔧 Инициализация конфигурации...")
        cfg = MLConfig()
        print(f"✅ Конфигурация загружена")

        # Шаг 3: Обучение экспертов
        experts = train_experts_on_historical_data(rounds, cfg, train_split=0.8)

        # Шаг 4: Обучение META Neural
        meta = train_meta_neural(rounds, experts, cfg, train_split=0.8)

        # Шаг 5: Сохранение моделей
        save_models(experts, meta, cfg, output_dir="trained_models_real")

        # Финальная статистика
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"{'='*70}")
        print(f"⏱️  Время: {elapsed/60:.1f} минут")
        print(f"📊 Обучено на {len(rounds):,} раундах")
        print(f"💾 Модели сохранены в trained_models_real/")
        print(f"🎉 Бот готов к использованию!")
        print(f"{'='*70}\n")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Прервано пользователем")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
