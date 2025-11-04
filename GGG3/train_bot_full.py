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
    """
    Подготовка вектора из 58 фич для экспертов

    Структура:
    1-14:  Базовые технические индикаторы
    15-26: Рыночная микроструктура
    27-35: Производные и взаимодействия
    36-44: Временные лаги и тренды
    45-52: Контекстные фичи рынка
    53-58: Дополнительные кросс-фичи
    """
    features = round_data.get('features', {})

    # ========== 1-14: БАЗОВЫЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ==========
    momentum_5 = _as_float(features.get('momentum_5m', 0), 0)
    momentum_10 = _as_float(features.get('momentum_10m', 0), 0)
    momentum_15 = _as_float(features.get('momentum_15m', 0), 0)
    volatility = _as_float(features.get('volatility', 0), 0)
    vol_z = _as_float(features.get('vol_zscore', 0), 0)
    rsi = _as_float(features.get('rsi', 50), 50)
    rsi_lag1 = _as_float(features.get('rsi_lag1', 50), 50)
    rsi_lag5 = _as_float(features.get('rsi_lag5', 50), 50)
    close = _as_float(features.get('close', 0), 0)
    high = _as_float(features.get('high', close), close)
    low = _as_float(features.get('low', close), close)
    volume = _as_float(features.get('volume', 0), 0)
    atr = _as_float(features.get('atr', 0), 0)
    atr_norm = _as_float(features.get('atr_norm', 1), 1)

    # ========== 15-26: РЫНОЧНАЯ МИКРОСТРУКТУРА ==========
    bull_amt = _as_float(round_data.get('bull_amount', 0), 0)
    bear_amt = _as_float(round_data.get('bear_amount', 0), 0)
    pool_total = bull_amt + bear_amt
    pool_ratio = bull_amt / max(1e-12, bear_amt)
    pool_imbalance = (bull_amt - bear_amt) / max(1e-12, pool_total)

    book_imb = _as_float(features.get('book_imb', 0), 0)
    microprice = _as_float(features.get('microprice', 0), 0)
    spread = _as_float(features.get('spread', 0), 0)
    rel_spread = _as_float(features.get('rel_spread', 0), 0)

    ofi_5s = _as_float(features.get('ofi_5s', 0), 0)
    ofi_15s = _as_float(features.get('ofi_15s', 0), 0)
    ofi_30s = _as_float(features.get('ofi_30s', 0), 0)

    # ========== 27-35: ПРОИЗВОДНЫЕ И ВЗАИМОДЕЙСТВИЯ ==========
    volatility_sq = volatility ** 2
    momentum_vol = momentum_5 * volatility
    rsi_deviation = abs(rsi - 50.0) / 50.0

    # Нормализованная позиция в канале
    price_range = max(1e-12, high - low)
    price_position = (close - low) / price_range if price_range > 0 else 0.5

    # Волатильность RSI
    rsi_volatility = _as_float(features.get('rsi_volatility', 0), 0)

    # Коэффициент вариации доходности
    returns_cv = _as_float(features.get('returns_cv', 0), 0)

    # Trend strength
    trend_sign = _as_float(features.get('trend_sign', 0), 0)
    trend_abs = _as_float(features.get('trend_abs', 0), 0)
    vol_ratio = _as_float(features.get('vol_ratio', 1), 1)

    # ========== 36-44: ВРЕМЕННЫЕ ЛАГИ И ТРЕНДЫ ==========
    close_lag1 = _as_float(features.get('close_lag1', close), close)
    close_lag5 = _as_float(features.get('close_lag5', close), close)
    close_lag15 = _as_float(features.get('close_lag15', close), close)

    returns_1 = (close - close_lag1) / max(1e-12, close_lag1)
    returns_5 = (close - close_lag5) / max(1e-12, close_lag5)
    returns_15 = (close - close_lag15) / max(1e-12, close_lag15)

    volume_lag1 = _as_float(features.get('volume_lag1', volume), volume)
    volume_change = (volume - volume_lag1) / max(1e-12, volume_lag1)

    volatility_lag5 = _as_float(features.get('volatility_lag5', volatility), volatility)

    # ========== 45-52: КОНТЕКСТНЫЕ ФИЧИ РЫНКА ==========
    funding_rate = _as_float(features.get('funding_rate', 0), 0)
    funding_sign = np.sign(funding_rate)
    basis_pct = _as_float(features.get('basis_pct', 0), 0)
    basis_sign = np.sign(basis_pct)

    gas_price = _as_float(features.get('gas_price', 0), 0)
    gas_norm = gas_price / max(1e-12, _as_float(features.get('gas_median', 1), 1))

    jump_flag = _as_float(features.get('jump_flag', 0), 0)
    late_money_share = _as_float(features.get('late_money_share', 0), 0)

    # ========== 53-58: ДОПОЛНИТЕЛЬНЫЕ КРОСС-ФИЧИ ==========
    # Взаимодействия ключевых фичей
    momentum_rsi = momentum_5 * (rsi / 50.0)
    volatility_volume = volatility * np.log1p(volume)
    trend_vol_interaction = trend_sign * vol_ratio
    pool_vol_interaction = pool_imbalance * volatility
    ofi_trend = ofi_15s * trend_sign
    momentum_ofi = momentum_5 * ofi_15s

    # ========== СБОРКА ВЕКТОРА (58 ФИЧ) ==========
    x_raw = np.array([
        # 1-14: Базовые индикаторы
        momentum_5, momentum_10, momentum_15,
        volatility, vol_z,
        rsi, rsi_lag1, rsi_lag5,
        close, high, low, volume,
        atr, atr_norm,

        # 15-26: Микроструктура (12 фич)
        bull_amt, bear_amt, pool_total, pool_ratio, pool_imbalance,
        book_imb, microprice, spread, rel_spread,
        ofi_5s, ofi_15s, ofi_30s,

        # 27-35: Производные (9 фич)
        volatility_sq, momentum_vol, rsi_deviation,
        price_position, rsi_volatility, returns_cv,
        trend_sign, trend_abs, vol_ratio,

        # 36-44: Временные лаги (9 фич)
        close_lag1, close_lag5, close_lag15,
        returns_1, returns_5, returns_15,
        volume_lag1, volume_change, volatility_lag5,

        # 45-52: Контекст рынка (8 фич)
        funding_rate, funding_sign, basis_pct, basis_sign,
        gas_price, gas_norm, jump_flag, late_money_share,

        # 53-58: Кросс-фичи (6 фич)
        momentum_rsi, volatility_volume, trend_vol_interaction,
        pool_vol_interaction, ofi_trend, momentum_ofi,
    ], dtype=np.float64)

    # Защита от NaN/Inf
    x_raw = np.nan_to_num(x_raw, nan=0.0, posinf=10.0, neginf=-10.0)
    x_raw = np.clip(x_raw, -10.0, 10.0)

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

    # ========== ВАЛИДАЦИЯ ФИЧЕЙ ==========
    print(f"\n🔍 Проверка размерности фич...")
    sample_features = prepare_features(train_rounds[0])
    print(f"  Размерность: {sample_features.shape[0]} фич")
    if sample_features.shape[0] != 58:
        print(f"  ⚠️  ПРЕДУПРЕЖДЕНИЕ: Ожидалось 58 фич, получено {sample_features.shape[0]}")
    else:
        print(f"  ✅ Размерность корректна: 58 фич")

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
    print(f"  Входы: 5 предсказаний экспертов + контекст")
    print(f"  Внутренняя размерность: 18 → 36D feature engineering")

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
    """Сохраняет все обученные модели"""
    print(f"\n{'='*70}")
    print("СОХРАНЕНИЕ МОДЕЛЕЙ")
    print(f"{'='*70}\n")

    os.makedirs(output_dir, exist_ok=True)
    print(f"💾 Директория: {output_dir}")
    print(f"\n📝 Сохранение экспертов...")

    saved_count = 0
    failed_count = 0

    # Сохранение экспертов
    for name, expert in experts.items():
        try:
            # Разные эксперты имеют разные методы сохранения
            if name == "XGBoost":
                # XGBoost сохраняет через свой внутренний метод
                expert_dir = os.path.join(output_dir, "xgb")
                os.makedirs(expert_dir, exist_ok=True)

                # Сохраняем модели по фазам
                for ph in range(expert.P):
                    if expert.booster_ph.get(ph) is not None:
                        model_path = os.path.join(expert_dir, f"xgb_ph{ph}.model")
                        expert.booster_ph[ph].save_model(model_path)
                        print(f"  ✅ {name} phase {ph}: {model_path}")

                # Сохраняем глобальную модель
                if expert.booster is not None:
                    global_path = os.path.join(expert_dir, "xgb_global.model")
                    expert.booster.save_model(global_path)
                    print(f"  ✅ {name} global: {global_path}")

                saved_count += 1

            elif name == "RandomForest":
                # RandomForest использует pickle
                expert_dir = os.path.join(output_dir, "rf")
                os.makedirs(expert_dir, exist_ok=True)

                # Сохраняем модели по фазам
                for ph in range(expert.P):
                    if expert.clf_ph.get(ph) is not None:
                        model_path = os.path.join(expert_dir, f"rf_ph{ph}.pkl")
                        with open(model_path, 'wb') as f:
                            pickle.dump(expert.clf_ph[ph], f)
                        print(f"  ✅ {name} phase {ph}: {model_path}")

                # Сохраняем глобальную модель
                if expert.clf is not None:
                    global_path = os.path.join(expert_dir, "rf_global.pkl")
                    with open(global_path, 'wb') as f:
                        pickle.dump(expert.clf, f)
                    print(f"  ✅ {name} global: {global_path}")

                saved_count += 1

            elif name == "ARF":
                # River ARF сохраняет через pickle
                expert_dir = os.path.join(output_dir, "arf")
                os.makedirs(expert_dir, exist_ok=True)

                if expert.clf is not None:
                    model_path = os.path.join(expert_dir, "arf_model.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(expert.clf, f)
                    print(f"  ✅ {name}: {model_path}")

                saved_count += 1

            elif name == "NN":
                # NN сохраняет через свой метод save()
                expert_dir = os.path.join(output_dir, "nn")
                os.makedirs(expert_dir, exist_ok=True)

                # Сохраняем модели по фазам
                for ph in range(expert.P):
                    if ph in expert.nets_ph and expert.nets_ph[ph] is not None:
                        model_path = os.path.join(expert_dir, f"nn_ph{ph}.pkl")
                        expert.nets_ph[ph].save(model_path)
                        print(f"  ✅ {name} phase {ph}: {model_path}")

                # Сохраняем глобальную модель
                if expert.net_global is not None:
                    global_path = os.path.join(expert_dir, "nn_global.pkl")
                    expert.net_global.save(global_path)
                    print(f"  ✅ {name} global: {global_path}")

                saved_count += 1

        except Exception as e:
            print(f"  ❌ {name}: ошибка сохранения - {e}")
            failed_count += 1

    # Сохранение META Neural
    if meta:
        try:
            meta_dir = os.path.join(output_dir, "meta")
            os.makedirs(meta_dir, exist_ok=True)

            # Сохраняем каждую фазовую нейросеть
            for ph, network in meta.networks.items():
                if network is not None:
                    net_path = os.path.join(meta_dir, f"meta_net_ph{ph}.pkl")
                    with open(net_path, 'wb') as f:
                        pickle.dump(network, f)
                    print(f"  ✅ META Neural phase {ph}: {net_path}")

            # Сохраняем состояние META
            meta_state = {
                "mode": meta.mode,
                "seen_ph": meta.seen_ph,
                "wins_ph": meta.wins_ph,
                "total_ph": meta.total_ph
            }
            state_path = os.path.join(meta_dir, "meta_state.json")
            with open(state_path, 'w') as f:
                json.dump(meta_state, f, indent=2)
            print(f"  ✅ META state: {state_path}")

            saved_count += 1

        except Exception as e:
            print(f"  ❌ META Neural: ошибка сохранения - {e}")
            failed_count += 1

    # Итог
    print(f"\n{'='*70}")
    print(f"✅ Успешно сохранено: {saved_count} моделей")
    if failed_count > 0:
        print(f"❌ Ошибок: {failed_count}")
    print(f"📁 Директория: {os.path.abspath(output_dir)}")
    print(f"{'='*70}\n")


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
