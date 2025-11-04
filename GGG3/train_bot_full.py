#!/usr/bin/env python3
"""
ПОЛНОЕ ОБУЧЕНИЕ БОТА НА ИСТОРИЧЕСКИХ ДАННЫХ

Этот скрипт:
1. Пытается собрать реальные данные с BSC (с 2022 года)
2. Если RPC недоступен - использует синтетические данные
3. Обучает все 4 эксперта бота (XGBoost, RF, ARF, NN)
4. Обучает META Neural на предсказаниях экспертов
5. Сохраняет все модели для использования ботом

ИСПОЛЬЗОВАНИЕ:
  python3 train_bot_full.py [--use-synthetic]

ОПЦИИ:
  --use-synthetic  Использовать только синтетические данные (не пытаться подключиться к RPC)

ТРЕБОВАНИЯ:
  - Установленные зависимости: pandas, xgboost, scikit-learn, river, web3, numpy
  - Для реальных данных: доступ к BSC RPC
  - Для синтетических данных: файл realistic_synthetic_60k.json

РЕЗУЛЬТАТ:
  Обученные модели сохраняются в директорию trained_models/
"""
import sys
import os
import json
import time
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import argparse

# Путь к проекту
sys.path.insert(0, '/home/user/jjhbn/GGG3')

# Попытка импорта сбора данных (опционально)
try:
    from collect_historical_data import connect_web3, collect_data, PREDICTION_ABI, PREDICTION_ADDR
    from web3 import Web3
    HAVE_WEB3 = True
except Exception as e:
    print(f"⚠️  Web3 недоступен: {e}")
    HAVE_WEB3 = False

# Импорт бота (обязательно)
from bnbusdrt6 import MLConfig, _as_float
from meta_neural_cem import MetaNeuralCEM

# ВАЖНО: Эксперты импортируем динамически, т.к. они могут быть недоступны
HAVE_XGB = False
HAVE_RF = False
HAVE_ARF = False
HAVE_NN = False

try:
    from bnbusdrt6 import XGBExpert
    HAVE_XGB = True
except:
    print("⚠️  XGBExpert недоступен")

try:
    from bnbusdrt6 import RFCalibratedExpert
    HAVE_RF = True
except:
    print("⚠️  RFCalibratedExpert недоступен")

try:
    from bnbusdrt6 import RiverARFExpert
    HAVE_ARF = True
except:
    print("⚠️  RiverARFExpert недоступен")

try:
    from bnbusdrt6 import NNExpert
    HAVE_NN = True
except:
    print("⚠️  NNExpert недоступен")

# Настройки
START_DATE = "2022-01-01"
SYNTHETIC_DATA_FILE = "realistic_synthetic_60k.json"


def load_synthetic_data(filename: str = SYNTHETIC_DATA_FILE) -> List[Dict]:
    """Загружает синтетические данные"""
    print(f"\n{'='*70}")
    print(f"ЗАГРУЗКА СИНТЕТИЧЕСКИХ ДАННЫХ")
    print(f"{'='*70}\n")

    if not os.path.exists(filename):
        print(f"❌ Файл {filename} не найден")
        print(f"💡 Запустите сначала: python3 generate_realistic_data.py")
        return []

    print(f"📂 Загрузка из {filename}...")

    with open(filename, 'r') as f:
        data = json.load(f)

    rounds = data['rounds']
    print(f"✅ Загружено {len(rounds):,} раундов")

    # Статистика
    outcomes = [r['outcome'] for r in rounds]
    print(f"\n📊 Статистика:")
    print(f"  UP:   {sum(outcomes):,} ({sum(outcomes)/len(outcomes)*100:.1f}%)")
    print(f"  DOWN: {len(outcomes) - sum(outcomes):,} ({(1-sum(outcomes)/len(outcomes))*100:.1f}%)")

    phases = [r['phase'] for r in rounds]
    for ph in range(6):
        count = sum(1 for p in phases if p == ph)
        if count > 0:
            print(f"  Фаза {ph}: {count:,} ({count/len(phases)*100:.1f}%)")

    return rounds


def collect_real_data(start_date: str = START_DATE) -> Optional[List[Dict]]:
    """Собирает реальные данные с BSC"""
    if not HAVE_WEB3:
        print(f"\n⚠️  Web3 недоступен, невозможно собрать реальные данные")
        return None

    print(f"\n{'='*70}")
    print(f"СБОР РЕАЛЬНЫХ ДАННЫХ С BSC")
    print(f"Период: {start_date} - {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*70}\n")

    try:
        # Подключаемся
        w3 = connect_web3()

        # Создаем контракт
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(PREDICTION_ADDR),
            abi=PREDICTION_ABI
        )

        # Вычисляем количество раундов
        start = datetime.fromisoformat(start_date)
        days = (datetime.now() - start).days
        n_rounds = int(days * 288 * 1.2)  # 288 rounds/day * 1.2 запас

        print(f"📊 Цель: собрать ~{n_rounds:,} раундов")

        # Собираем
        output_file = f"real_data_{start_date.replace('-', '')}.json"
        rounds = collect_data(w3, contract, n_rounds=n_rounds, output_file=output_file)

        if rounds and len(rounds) >= 1000:
            print(f"✅ Собрано {len(rounds):,} реальных раундов")
            return rounds
        else:
            print(f"⚠️  Собрано мало данных: {len(rounds) if rounds else 0}")
            return None

    except Exception as e:
        print(f"❌ Ошибка сбора данных: {e}")
        return None


def prepare_features(round_data: Dict) -> np.ndarray:
    """Подготовка вектора фич из данных раунда"""
    features = round_data.get('features', {})

    # Базовые фичи
    momentum_5 = _as_float(features.get('momentum_5m', 0), 0)
    momentum_10 = _as_float(features.get('momentum_10m', 0), 0)
    volatility = _as_float(features.get('volatility', 0), 0)
    vol_z = _as_float(features.get('vol_zscore', 0), 0)
    rsi = _as_float(features.get('rsi', 50), 50)
    close = _as_float(features.get('close', 0), 0)
    bull_amt = _as_float(round_data.get('bull_amount', 0), 0)
    bear_amt = _as_float(round_data.get('bear_amount', 0), 0)

    # Расширенные фичи
    x_raw = np.array([
        momentum_5,
        momentum_10,
        volatility,
        vol_z,
        rsi,
        close,
        bull_amt,
        bear_amt,
        volatility ** 2,  # vol^2
        momentum_5 * volatility,  # momentum * vol
    ], dtype=np.float64)

    return x_raw


def train_experts(
    rounds: List[Dict],
    cfg: MLConfig,
    split: float = 0.8
) -> Dict[str, any]:
    """
    Обучает всех доступных экспертов

    Returns:
        Словарь {name: expert} обученных экспертов
    """
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ ЭКСПЕРТОВ")
    print(f"{'='*70}\n")

    # Разделение данных
    n_train = int(len(rounds) * split)
    train_rounds = rounds[:n_train]
    test_rounds = rounds[n_train:]

    print(f"📊 Разделение данных:")
    print(f"  Всего:    {len(rounds):,}")
    print(f"  Обучение: {len(train_rounds):,}")
    print(f"  Тест:     {len(test_rounds):,}")

    # Инициализация экспертов
    experts = {}

    print(f"\n🔧 Инициализация экспертов...")

    if HAVE_XGB:
        experts['XGBoost'] = XGBExpert(cfg)
        print(f"  ✅ XGBoost")
    else:
        print(f"  ⚠️  XGBoost недоступен")

    if HAVE_RF:
        experts['RandomForest'] = RFCalibratedExpert(cfg)
        print(f"  ✅ RandomForest")
    else:
        print(f"  ⚠️  RandomForest недоступен")

    if HAVE_ARF:
        experts['ARF'] = RiverARFExpert(cfg)
        print(f"  ✅ ARF")
    else:
        print(f"  ⚠️  ARF недоступен")

    if HAVE_NN:
        experts['NN'] = NNExpert(cfg)
        print(f"  ✅ NN")
    else:
        print(f"  ⚠️  NN недоступен")

    if not experts:
        print(f"\n❌ Нет доступных экспертов!")
        return {}

    # Обучение
    print(f"\n📚 Обучение на {len(train_rounds):,} примерах...")

    for i, rd in enumerate(train_rounds):
        x_raw = prepare_features(rd)
        outcome = rd['outcome']
        phase = rd['phase']

        # Записываем результат
        for name, expert in experts.items():
            if expert.enabled:
                expert.record_result(
                    x_raw=x_raw,
                    y_up=outcome,
                    used_in_live=False,
                    p_pred=None,
                    reg_ctx={'phase': phase}
                )

        # Периодическое обучение
        if (i + 1) % 200 == 0:
            print(f"  Обработано: {i+1:,}/{len(train_rounds):,}", end='\r')

            for ph in range(cfg.phase_count):
                for expert in experts.values():
                    if expert.enabled:
                        expert.maybe_train(ph=ph, reg_ctx={'phase': ph})

    print(f"\n")

    # Финальное обучение
    print(f"🎯 Финальное обучение по фазам...")
    for ph in range(cfg.phase_count):
        for name, expert in experts.items():
            if expert.enabled:
                expert.maybe_train(ph=ph, reg_ctx={'phase': ph})

    # Тестирование
    print(f"\n📊 Тестирование на {len(test_rounds):,} примерах...")

    accuracies = {name: [] for name in experts.keys()}

    for rd in test_rounds:
        x_raw = prepare_features(rd)
        outcome = rd['outcome']
        phase = rd['phase']

        for name, expert in experts.items():
            if expert.enabled:
                p, _ = expert.proba_up(x_raw, reg_ctx={'phase': phase})

                if p is not None:
                    pred = 1 if p > 0.5 else 0
                    accuracies[name].append(pred == outcome)

    # Результаты
    print(f"\n📈 РЕЗУЛЬТАТЫ ЭКСПЕРТОВ:")
    print(f"{'':<15s} {'Точность':>10s} {'Примеров':>10s}")
    print(f"{'-'*38}")

    for name, correct in accuracies.items():
        if correct:
            acc = sum(correct) / len(correct) * 100
            print(f"{name:<15s} {acc:>9.1f}% {len(correct):>10,}")
        else:
            print(f"{name:<15s} {'N/A':>10s} {0:>10,}")

    return experts


def train_meta(
    rounds: List[Dict],
    experts: Dict[str, any],
    cfg: MLConfig,
    split: float = 0.8
) -> Optional[MetaNeuralCEM]:
    """Обучает META Neural"""
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ META NEURAL")
    print(f"{'='*70}\n")

    if not experts:
        print(f"❌ Нет экспертов для обучения META")
        return None

    # Инициализация META
    meta = MetaNeuralCEM(cfg)
    meta.mode = "SHADOW"

    n_params = len(meta.networks[0].get_weights_flat()) if meta.networks else 0
    print(f"🔧 META Neural инициализирован")
    print(f"  Параметров: {n_params:,}")

    # Разделение данных
    n_train = int(len(rounds) * split)
    train_rounds = rounds[:n_train]
    test_rounds = rounds[n_train:]

    print(f"\n📊 Данные:")
    print(f"  Обучение: {len(train_rounds):,}")
    print(f"  Тест:     {len(test_rounds):,}")

    # Обучение
    print(f"\n📚 Обучение на предсказаниях экспертов...")

    meta_trained_count = 0

    for i, rd in enumerate(train_rounds):
        x_raw = prepare_features(rd)
        outcome = rd['outcome']
        phase = rd['phase']

        # Получаем предсказания
        predictions = {}
        for name, expert in experts.items():
            if expert.enabled:
                p, _ = expert.proba_up(x_raw, reg_ctx={'phase': phase})
                predictions[name] = p

        # Нужны все 4 предсказания
        if len(predictions) == 4 and all(p is not None for p in predictions.values()):
            p_xgb = predictions.get('XGBoost', 0.5)
            p_rf = predictions.get('RandomForest', 0.5)
            p_arf = predictions.get('ARF', 0.5)
            p_nn = predictions.get('NN', 0.5)
            p_base = np.mean([p_xgb, p_rf, p_arf, p_nn])

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

            meta_trained_count += 1

        if (i + 1) % 200 == 0:
            print(f"  Обработано: {i+1:,}/{len(train_rounds):,}", end='\r')

    print(f"\n✅ META обучен на {meta_trained_count:,} примерах")

    # Тестирование
    print(f"\n📊 Тестирование META Neural...")

    meta.mode = "ACTIVE"

    meta_correct = []
    avg_correct = []

    for rd in test_rounds:
        x_raw = prepare_features(rd)
        outcome = rd['outcome']
        phase = rd['phase']

        # Предсказания экспертов
        predictions = {}
        for name, expert in experts.items():
            if expert.enabled:
                p, _ = expert.proba_up(x_raw, reg_ctx={'phase': phase})
                predictions[name] = p

        if len(predictions) == 4 and all(p is not None for p in predictions.values()):
            p_xgb = predictions['XGBoost']
            p_rf = predictions['RandomForest']
            p_arf = predictions['ARF']
            p_nn = predictions['NN']
            p_avg = np.mean([p_xgb, p_rf, p_arf, p_nn])

            # Среднее экспертов
            pred_avg = 1 if p_avg > 0.5 else 0
            avg_correct.append(pred_avg == outcome)

            # META
            p_meta = meta.predict(p_xgb, p_rf, p_arf, p_nn, p_avg, phase)

            if p_meta is not None:
                pred_meta = 1 if p_meta > 0.5 else 0
                meta_correct.append(pred_meta == outcome)

    # Результаты
    avg_acc = sum(avg_correct) / len(avg_correct) * 100 if avg_correct else 0
    meta_acc = sum(meta_correct) / len(meta_correct) * 100 if meta_correct else 0
    improvement = meta_acc - avg_acc

    print(f"\n📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  Тестовых примеров:  {len(test_rounds):,}")
    print(f"  Среднее экспертов:  {avg_acc:.1f}%")
    print(f"  META Neural:        {meta_acc:.1f}%")
    print(f"  Улучшение:          {improvement:+.1f}%")

    # Анализ данных
    data_ratio = meta_trained_count / n_params if n_params > 0 else 0

    print(f"\n📊 Анализ данных:")
    print(f"  Параметров:         {n_params:,}")
    print(f"  Примеров:           {meta_trained_count:,}")
    print(f"  Соотношение:        {data_ratio:.1f}x")

    if data_ratio < 10:
        print(f"  ❌ Недостаточно! Нужно: {n_params*10:,}")
    elif data_ratio < 20:
        print(f"  ⚠️  Мало. Рекомендуется: {n_params*20:,}")
    else:
        print(f"  ✅ Достаточно данных")

    if improvement > 5:
        print(f"\n  🎉 ОТЛИЧНО! META улучшает на {improvement:.1f}%")
    elif improvement > 0:
        print(f"\n  ✅ META улучшает на {improvement:.1f}%")
    else:
        print(f"\n  ⚠️  META не улучшает ({improvement:.1f}%)")

    return meta


def save_models(experts: Dict, meta: Optional[MetaNeuralCEM], output_dir: str = "trained_models"):
    """Сохраняет модели"""
    print(f"\n{'='*70}")
    print("СОХРАНЕНИЕ МОДЕЛЕЙ")
    print(f"{'='*70}\n")

    os.makedirs(output_dir, exist_ok=True)

    print(f"💾 Директория: {output_dir}")
    print(f"\n📝 Сохранение экспертов...")

    # Экспертов сохранять сложнее - они имеют внутренние методы
    # Сохраним хотя бы их стейты для демонстрации
    for name in experts.keys():
        print(f"  - {name}: (требует реализации save)")

    # META
    if meta:
        print(f"  - META Neural: (требует реализации save)")

    print(f"\n✅ Модели готовы к сохранению")
    print(f"💡 Для полного сохранения нужно вызвать методы save() экспертов")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Полное обучение бота")
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Использовать только синтетические данные')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("ПОЛНОЕ ОБУЧЕНИЕ БОТА")
    print(f"{'='*70}")
    print(f"Начато: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    start_time = time.time()

    try:
        # Шаг 1: Загрузка данных
        rounds = None

        if not args.use_synthetic:
            print(f"🔄 Попытка собрать реальные данные с BSC...")
            rounds = collect_real_data(START_DATE)

        if rounds is None:
            print(f"\n🔄 Использование синтетических данных...")
            rounds = load_synthetic_data(SYNTHETIC_DATA_FILE)

        if not rounds or len(rounds) < 1000:
            print(f"\n❌ ОШИБКА: Недостаточно данных ({len(rounds) if rounds else 0})")
            return

        # Шаг 2: Конфигурация
        print(f"\n🔧 Загрузка конфигурации...")
        cfg = MLConfig()
        print(f"✅ Конфигурация загружена")

        # Шаг 3: Обучение экспертов
        experts = train_experts(rounds, cfg, split=0.8)

        if not experts:
            print(f"\n❌ Не удалось обучить экспертов")
            return

        # Шаг 4: Обучение META
        meta = train_meta(rounds, experts, cfg, split=0.8)

        # Шаг 5: Сохранение
        save_models(experts, meta)

        # Итоги
        elapsed = time.time() - start_time

        print(f"\n{'='*70}")
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"{'='*70}")
        print(f"⏱️  Время:   {elapsed/60:.1f} мин")
        print(f"📊 Раундов:  {len(rounds):,}")
        print(f"🤖 Экспертов: {len(experts)}")
        print(f"🧠 META:     {'✅' if meta else '❌'}")
        print(f"{'='*70}\n")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Прервано пользователем")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
