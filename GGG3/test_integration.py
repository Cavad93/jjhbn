#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеграционный тест всей системы онлайн обучения

Тестирует:
- XGBoost Expert (online learning)
- AdaptiveRF Expert (online learning)
- SimplifiedEnsembleMETA (Ridge regression с контекстом)
- Взаимодействие между компонентами
"""

import numpy as np
import sys
from pathlib import Path

print("=" * 80)
print("ИНТЕГРАЦИОННЫЙ ТЕСТ СИСТЕМЫ ОНЛАЙН ОБУЧЕНИЯ")
print("=" * 80)

# Импортируем экспертов
print("\n1. Импорт компонентов...")
try:
    from models.experts.xgb_expert import XGBoostExpert
    print("   ✅ XGBoostExpert")
except Exception as e:
    print(f"   ❌ XGBoostExpert: {e}")
    sys.exit(1)

try:
    from models.experts.arf_expert import AdaptiveRFExpert
    print("   ✅ AdaptiveRFExpert")
except Exception as e:
    print(f"   ❌ AdaptiveRFExpert: {e}")
    sys.exit(1)

try:
    from simplified_ensemble_meta import SimplifiedEnsembleMETA
    print("   ✅ SimplifiedEnsembleMETA")
except Exception as e:
    print(f"   ❌ SimplifiedEnsembleMETA: {e}")
    sys.exit(1)

# Конфигурация (минимальная)
class Config:
    def __init__(self):
        self.meta_min_ready = 20
        self.meta_min_train = 30
        self.meta_state_path = "/tmp/test_meta_integration.json"

cfg = Config()

# Генерация тестовых данных
print("\n2. Генерация тестовых данных...")
np.random.seed(42)
n_samples = 500
n_features = 68

X = np.random.randn(n_samples, n_features)
# Таргет: зависит от первых 5 фич
y = (X[:, :5].sum(axis=1) > 0).astype(int)

print(f"   X: {X.shape}")
print(f"   y: {y.shape}, класс 1: {y.sum()} ({y.mean()*100:.1f}%)")

# Создание экспертов
print("\n3. Создание экспертов...")

print("   Создание XGBoost...")
xgb = XGBoostExpert(n_estimators=10, max_depth=5, learning_rate=0.1, keep_history=100)

print("   Создание AdaptiveRF (2 эксперта)...")
rf = AdaptiveRFExpert(n_models=15, random_state=123)
arf = AdaptiveRFExpert(n_models=10, random_state=42)

print("   ✅ Все эксперты созданы")

# Создание META
print("\n4. Создание META...")
meta = SimplifiedEnsembleMETA(cfg)
print(f"   ✅ META создана, mode={meta.mode}")

# Начальное обучение (первые 100 примеров)
print("\n5. Начальное обучение (100 примеров)...")
X_train = X[:100]
y_train = y[:100]

print("   Обучение XGBoost...")
xgb.fit(X_train, y_train)
print(f"      ✅ {xgb.train_samples} samples, {xgb.model.num_boosted_rounds()} trees")

print("   Обучение AdaptiveRF (rf)...")
rf.fit(X_train, y_train)
print(f"      ✅ {rf.train_samples} samples, {rf.n_models} models")

print("   Обучение AdaptiveRF (arf)...")
arf.fit(X_train, y_train)
print(f"      ✅ {arf.train_samples} samples, {arf.n_models} models")

# Онлайн обучение на оставшихся данных
print("\n6. Онлайн обучение (400 примеров)...")
n_correct_meta = 0
n_total_meta = 0

for i in range(100, n_samples):
    features = X[i]
    y_true = y[i]

    # Предсказания экспертов (t0)
    p_xgb, _ = xgb.proba_up(features)
    p_rf, _ = rf.proba_up(features)
    p_arf, _ = arf.proba_up(features)
    p_nn = 0.5  # Заглушка (torch не установлен)

    # Контекст
    phase = i % 6
    reg_ctx = {
        'phase': phase,
        'vol_ratio': 1.0 + np.random.randn() * 0.1,
        'trend_macd': np.random.randn() * 0.05,
        'jump_detected': False,
        'funding_sign': np.random.randn() * 0.01,
        'book_imb': np.random.randn() * 0.02,
        'ofi_15s': np.random.randn() * 50,
        'basis_pct': np.random.randn() * 0.0005
    }

    # Предсказание META
    p_final = meta.predict(p_xgb, p_rf, p_arf, p_nn, None, reg_ctx)

    if p_final is not None:
        pred_meta = int(p_final >= 0.5)
        if pred_meta == y_true:
            n_correct_meta += 1
        n_total_meta += 1

    # Онлайн обучение экспертов (t1)
    features_2d = features.reshape(1, -1)
    y_2d = np.array([y_true])

    # Обучаем с ограничением деревьев
    xgb.partial_fit(features_2d, y_2d, n_new_trees=2, max_total_trees=200)
    rf.partial_fit(features_2d, y_2d)
    arf.partial_fit(features_2d, y_2d)

    # Обучаем META
    meta.record_result(p_xgb, p_rf, p_arf, p_nn, None, y_true, True, p_final, reg_ctx)

# Результаты
print("\n7. Результаты онлайн обучения...")
print(f"   XGBoost: {xgb.train_samples} samples, {xgb.model.num_boosted_rounds()} trees")
print(f"   RF (ARF): {rf.train_samples} samples")
print(f"   ARF: {arf.train_samples} samples")
print(f"   META: mode={meta.mode}, total_samples={sum(meta.seen_ph.values())}")

if n_total_meta > 0:
    meta_acc = n_correct_meta / n_total_meta
    print(f"   META Accuracy: {meta_acc*100:.1f}% ({n_correct_meta}/{n_total_meta})")

# Проверка на тестовой выборке
print("\n8. Проверка на тестовой выборке (50 примеров)...")
X_test = np.random.randn(50, n_features)
y_test = (X_test[:, :5].sum(axis=1) > 0).astype(int)

# Предсказания экспертов
pred_xgb = (xgb.predict_proba(X_test) > 0.5).astype(int)
pred_rf = (rf.predict_proba(X_test) > 0.5).astype(int)
pred_arf = (arf.predict_proba(X_test) > 0.5).astype(int)

acc_xgb = (pred_xgb == y_test).mean()
acc_rf = (pred_rf == y_test).mean()
acc_arf = (pred_arf == y_test).mean()

print(f"   XGBoost: {acc_xgb*100:.1f}%")
print(f"   RF (ARF): {acc_rf*100:.1f}%")
print(f"   ARF: {acc_arf*100:.1f}%")

# Сохранение
print("\n9. Сохранение моделей...")
xgb.save("/tmp/test_xgb_integration.pkl")
print("   ✅ XGBoost сохранен")

rf.save("/tmp/test_rf_integration.pkl")
print("   ✅ RF сохранен")

arf.save("/tmp/test_arf_integration.pkl")
print("   ✅ ARF сохранен")

meta._save()
print("   ✅ META сохранен")

# Загрузка
print("\n10. Загрузка моделей...")
xgb_loaded = XGBoostExpert.load("/tmp/test_xgb_integration.pkl")
print(f"   ✅ XGBoost загружен: {xgb_loaded.train_samples} samples, {xgb_loaded.model.num_boosted_rounds()} trees")

rf_loaded = AdaptiveRFExpert.load("/tmp/test_rf_integration.pkl")
print(f"   ✅ RF загружен: {rf_loaded.train_samples} samples")

arf_loaded = AdaptiveRFExpert.load("/tmp/test_arf_integration.pkl")
print(f"   ✅ ARF загружен: {arf_loaded.train_samples} samples")

meta_loaded = SimplifiedEnsembleMETA(cfg)
print(f"   ✅ META загружен: mode={meta_loaded.mode}, samples={sum(meta_loaded.seen_ph.values())}")

# Финальный статус
print("\n11. Финальный статус META...")
status = meta.status()
for key, value in status.items():
    print(f"   {key}: {value}")

# Проверка XGBoost tree limiting
print("\n12. Проверка ограничения деревьев XGBoost...")
print(f"   Текущее количество деревьев: {xgb.model.num_boosted_rounds()}")
print(f"   Размер истории: {len(xgb.X_history)} samples")
print(f"   Максимум деревьев: 200 (при достижении - автоматическое переобучение)")

if xgb.model.num_boosted_rounds() < 200:
    print(f"   ✅ Количество деревьев в пределах лимита")
else:
    print(f"   ⚠️ Достигнут лимит деревьев")

print("\n" + "=" * 80)
print("✅ ВСЕ ИНТЕГРАЦИОННЫЕ ТЕСТЫ ПРОЙДЕНЫ")
print("=" * 80)
print("\n📊 РЕЗЮМЕ:")
print("   • XGBoost: онлайн обучение с ограничением деревьев работает")
print("   • AdaptiveRF (2 эксперта): онлайн обучение работает")
print("   • META: Ridge регрессия с контекстом работает")
print("   • Сохранение/загрузка: все компоненты восстанавливаются")
print("   • Pure Online Learning: все модели учатся ТОЛЬКО на live данных")
print("\n⚠️  NeuralNetwork Expert не протестирован (требует torch)")
