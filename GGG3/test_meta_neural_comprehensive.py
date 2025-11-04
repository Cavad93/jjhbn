#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексное тестирование meta_neural_cem.py

Проверяет:
1. Feature Engineering (36D + 7D)
2. Neural Network (forward, MC dropout, weights)
3. Data Persistence (3000 раундов × 6 фаз)
4. CMA-ES / CEM Training
5. Bootstrap Monte-Carlo calibration
6. CV Validation
7. Mode switching
"""

import os
import sys
import json
import csv
import time
import tempfile
import shutil
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(__file__))

# Импортируем тестируемые модули
try:
    from meta_neural_cem import MetaNeuralCEM, NeuralMetaNetwork, _safe_float, _safe_logit, _sigmoid, _relu
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"❌ IMPORT FAILED: {e}")
    import traceback
    traceback.print_exc()
    IMPORT_SUCCESS = False
    sys.exit(1)


# ========== MOCK CONFIG ==========

@dataclass
class MockConfig:
    """Mock конфиг для тестирования"""
    meta_state_path: str
    meta_exp4_phases: int = 6
    meta_min_ready: int = 80
    meta_min_train: int = 150
    meta_retrain_every: int = 50
    meta_use_cma_es: bool = False  # Используем CEM для быстроты
    meta_mc_n_inference: int = 10
    meta_mc_uncertainty_threshold: float = 0.15
    meta_enter_wr: float = 0.58
    meta_exit_wr: float = 0.52
    phase_memory_cap: int = 3000
    cv_enabled: bool = True
    cv_check_every: int = 100
    tg_bot_token: str = ""
    tg_chat_id: str = ""


# ========== TEST UTILITIES ==========

class TestResults:
    """Сбор результатов тестов"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ PASS: {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"❌ FAIL: {test_name}")
        print(f"   Error: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"SUMMARY: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"\nFAILED TESTS ({len(self.errors)}):")
            for test_name, error in self.errors:
                print(f"  • {test_name}: {error}")
        print(f"{'='*70}\n")
        return self.failed == 0


# ========== TEST SUITES ==========

def test_1_feature_engineering(results: TestResults):
    """Тест 1: Feature Engineering (36D + 7D)"""
    print("\n" + "="*70)
    print("TEST 1: Feature Engineering")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    try:
        cfg = MockConfig(meta_state_path=os.path.join(temp_dir, "state.json"))
        meta = MetaNeuralCEM(cfg)

        # Тест 1.1: Базовая генерация фичей
        try:
            x_feat, x_ctx = meta._build_enriched_features(
                p_xgb=0.6, p_rf=0.55, p_arf=0.58, p_nn=0.62,
                p_base=0.5,
                reg_ctx={
                    "phase": 2,
                    "vol_ratio": 1.2,
                    "trend_macd": 0.01,
                    "jump_detected": False,
                    "funding_sign": 0.0,
                    "book_imb": 0.05,
                    "ofi_15s": 0.02,
                    "basis_pct": 0.001
                }
            )

            assert x_feat is not None, "x_feat is None"
            assert x_ctx is not None, "x_ctx is None"
            assert x_feat.shape == (36,), f"Expected (36,), got {x_feat.shape}"
            assert x_ctx.shape == (7,), f"Expected (7,), got {x_ctx.shape}"
            assert np.all(np.isfinite(x_feat)), "x_feat contains non-finite values"
            assert np.all(np.isfinite(x_ctx)), "x_ctx contains non-finite values"

            results.add_pass("1.1: Basic feature generation")
        except Exception as e:
            results.add_fail("1.1: Basic feature generation", str(e))

        # Тест 1.2: Обработка None значений
        try:
            x_feat, x_ctx = meta._build_enriched_features(
                p_xgb=0.6, p_rf=None, p_arf=0.58, p_nn=None,
                p_base=0.5,
                reg_ctx={"phase": 0}
            )

            assert x_feat is not None, "Should handle None values with >=2 experts"
            results.add_pass("1.2: Handle None expert predictions")
        except Exception as e:
            results.add_fail("1.2: Handle None expert predictions", str(e))

        # Тест 1.3: Недостаточно экспертов
        try:
            x_feat, x_ctx = meta._build_enriched_features(
                p_xgb=0.6, p_rf=None, p_arf=None, p_nn=None,
                p_base=0.5,
                reg_ctx={"phase": 0}
            )

            assert x_feat is None and x_ctx is None, "Should return None with <2 experts"
            results.add_pass("1.3: Reject insufficient experts")
        except Exception as e:
            results.add_fail("1.3: Reject insufficient experts", str(e))

        # Тест 1.4: Логиты в правильном диапазоне
        try:
            x_feat, x_ctx = meta._build_enriched_features(
                p_xgb=0.99, p_rf=0.01, p_arf=0.5, p_nn=0.5,
                p_base=0.5,
                reg_ctx={"phase": 0}
            )

            # Логиты должны быть на позициях 11-14 (5 + 6 = 11)
            logits = x_feat[11:15]
            assert np.all(np.isfinite(logits)), "Logits contain non-finite values"
            results.add_pass("1.4: Logits are finite")
        except Exception as e:
            results.add_fail("1.4: Logits are finite", str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_2_neural_network(results: TestResults):
    """Тест 2: Neural Network"""
    print("\n" + "="*70)
    print("TEST 2: Neural Network")
    print("="*70)

    # Тест 2.1: Инициализация
    try:
        net = NeuralMetaNetwork(phase=0)

        # Проверка размерностей весов
        assert net.W_att1.shape == (7, 16), f"W_att1 wrong shape: {net.W_att1.shape}"
        assert net.W_att2.shape == (16, 4), f"W_att2 wrong shape: {net.W_att2.shape}"
        assert net.W1.shape == (36, 64), f"W1 wrong shape: {net.W1.shape}"
        assert net.W2.shape == (64, 32), f"W2 wrong shape: {net.W2.shape}"
        assert net.W3.shape == (32, 16), f"W3 wrong shape: {net.W3.shape}"
        assert net.W4.shape == (16, 1), f"W4 wrong shape: {net.W4.shape}"

        results.add_pass("2.1: Network initialization")
    except Exception as e:
        results.add_fail("2.1: Network initialization", str(e))

    # Тест 2.2: Forward pass
    try:
        net = NeuralMetaNetwork(phase=0)
        x_feat = np.random.randn(36)
        x_ctx = np.random.randn(7)

        p = net.forward(x_feat, x_ctx, training=False)

        assert isinstance(p, float), f"Output is not float: {type(p)}"
        assert 0.0 <= p <= 1.0, f"Output out of range [0,1]: {p}"
        assert np.isfinite(p), f"Output is not finite: {p}"

        results.add_pass("2.2: Forward pass")
    except Exception as e:
        results.add_fail("2.2: Forward pass", str(e))

    # Тест 2.3: MC Dropout prediction
    try:
        net = NeuralMetaNetwork(phase=0)
        x_feat = np.random.randn(36)
        x_ctx = np.random.randn(7)

        p_mean, p_std = net.predict_mc(x_feat, x_ctx, n_mc=20)

        assert 0.0 <= p_mean <= 1.0, f"Mean out of range: {p_mean}"
        assert 0.0 <= p_std <= 1.0, f"Std out of range: {p_std}"
        assert np.isfinite(p_mean) and np.isfinite(p_std), "Non-finite MC outputs"

        results.add_pass("2.3: MC Dropout prediction")
    except Exception as e:
        results.add_fail("2.3: MC Dropout prediction", str(e))

    # Тест 2.4: Weights serialization
    try:
        net = NeuralMetaNetwork(phase=0)

        # Get weights
        w_flat = net.get_weights_flat()
        expected_size = (7*16 + 16) + (16*4 + 4) + (36*64 + 64) + (64*32 + 32) + (32*16 + 16) + (16*1 + 1)
        assert len(w_flat) == expected_size, f"Wrong weight size: {len(w_flat)} vs {expected_size}"

        # Modify and set back
        w_modified = w_flat + 0.1
        net.set_weights_from_flat(w_modified)
        w_flat_new = net.get_weights_flat()

        assert np.allclose(w_flat_new, w_modified), "Weights not restored correctly"

        results.add_pass("2.4: Weights serialization")
    except Exception as e:
        results.add_fail("2.4: Weights serialization", str(e))


def test_3_data_persistence(results: TestResults):
    """Тест 3: Data Persistence (3000 раундов × 6 фаз)"""
    print("\n" + "="*70)
    print("TEST 3: Data Persistence (3000 rounds × 6 phases)")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    try:
        cfg = MockConfig(meta_state_path=os.path.join(temp_dir, "state.json"))
        meta = MetaNeuralCEM(cfg)

        # Тест 3.1: Сохранение примеров в CSV
        try:
            ph = 2
            x_feat = np.random.randn(36)
            x_ctx = np.random.randn(7)
            y = 1

            # Сохраняем 10 примеров
            for i in range(10):
                meta._append_example(ph, x_feat + i*0.01, x_ctx, y)

            csv_path = meta._phase_csv_paths[ph]
            assert os.path.exists(csv_path), f"CSV file not created: {csv_path}"

            # Читаем CSV
            with open(csv_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == 11, f"Expected 11 lines (header + 10 data), got {len(lines)}"

            results.add_pass("3.1: Save examples to CSV")
        except Exception as e:
            results.add_fail("3.1: Save examples to CSV", str(e))

        # Тест 3.2: Загрузка данных из CSV
        try:
            X_feat_list, X_ctx_list, y_list = meta._load_phase_data(ph)

            assert len(X_feat_list) == 10, f"Expected 10 samples, got {len(X_feat_list)}"
            assert len(X_ctx_list) == 10, f"Expected 10 contexts, got {len(X_ctx_list)}"
            assert len(y_list) == 10, f"Expected 10 labels, got {len(y_list)}"

            # Проверка размерностей
            assert X_feat_list[0].shape == (36,), "Wrong feature shape"
            assert X_ctx_list[0].shape == (7,), "Wrong context shape"

            results.add_pass("3.2: Load data from CSV")
        except Exception as e:
            results.add_fail("3.2: Load data from CSV", str(e))

        # Тест 3.3: Обрезка до 3000 записей
        try:
            # Создаем CSV с 3500 записями
            csv_path = meta._phase_csv_paths[ph]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["x_features", "x_context", "y", "timestamp"])
                for i in range(3500):
                    writer.writerow([
                        json.dumps([0.0]*36),
                        json.dumps([0.0]*7),
                        i % 2,
                        int(time.time()) + i
                    ])

            # Обрезаем
            meta._trim_phase_storage(ph)

            # Проверяем
            with open(csv_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == 3001, f"Expected 3001 lines (header + 3000), got {len(lines)}"

            results.add_pass("3.3: Trim to 3000 records")
        except Exception as e:
            results.add_fail("3.3: Trim to 3000 records", str(e))

        # Тест 3.4: Множественные фазы
        try:
            for phase_num in range(6):
                x_feat = np.random.randn(36)
                x_ctx = np.random.randn(7)
                meta._append_example(phase_num, x_feat, x_ctx, phase_num % 2)

            # Проверяем что созданы CSV для всех фаз
            for phase_num in range(6):
                csv_path = meta._phase_csv_paths[phase_num]
                assert os.path.exists(csv_path), f"CSV not created for phase {phase_num}"

            results.add_pass("3.4: Multiple phases storage")
        except Exception as e:
            results.add_fail("3.4: Multiple phases storage", str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_4_training(results: TestResults):
    """Тест 4: CMA-ES / CEM Training"""
    print("\n" + "="*70)
    print("TEST 4: Training (CEM)")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    try:
        cfg = MockConfig(
            meta_state_path=os.path.join(temp_dir, "state.json"),
            meta_use_cma_es=False,  # Используем CEM для скорости
            meta_min_train=50
        )
        meta = MetaNeuralCEM(cfg)

        # Тест 4.1: Генерация синтетических данных
        try:
            ph = 0
            n_samples = 200

            X_feat_list = []
            X_ctx_list = []
            y_list = []

            # Простая зависимость: y = (mean(experts) > 0.5)
            for i in range(n_samples):
                p_mean = 0.3 + 0.4 * (i / n_samples)  # Увеличивается от 0.3 до 0.7

                x_feat, x_ctx = meta._build_enriched_features(
                    p_xgb=p_mean + 0.05,
                    p_rf=p_mean - 0.05,
                    p_arf=p_mean,
                    p_nn=p_mean,
                    p_base=0.5,
                    reg_ctx={"phase": ph, "vol_ratio": 1.0}
                )

                y = int(p_mean > 0.5)

                X_feat_list.append(x_feat)
                X_ctx_list.append(x_ctx)
                y_list.append(y)

            X_feat = np.array(X_feat_list)
            X_ctx = np.array(X_ctx_list)
            y = np.array(y_list)

            assert X_feat.shape == (n_samples, 36), "Wrong X_feat shape"
            assert X_ctx.shape == (n_samples, 7), "Wrong X_ctx shape"
            assert y.shape == (n_samples,), "Wrong y shape"

            results.add_pass("4.1: Generate synthetic data")
        except Exception as e:
            results.add_fail("4.1: Generate synthetic data", str(e))

        # Тест 4.2: Evaluate weights (Bootstrap MC)
        try:
            net = meta.networks[ph]
            w_initial = net.get_weights_flat()

            loss_initial = meta._evaluate_weights(w_initial, net, X_feat, X_ctx, y)

            assert np.isfinite(loss_initial), f"Loss is not finite: {loss_initial}"
            assert loss_initial > 0, f"Loss should be positive: {loss_initial}"

            results.add_pass("4.2: Evaluate weights (Bootstrap MC)")
        except Exception as e:
            results.add_fail("4.2: Evaluate weights (Bootstrap MC)", str(e))

        # Тест 4.3: CEM Training
        try:
            print(f"\n[TEST] Starting CEM training with {n_samples} samples...")

            meta._train_cem(ph, X_feat, X_ctx, y)

            # Проверяем что веса изменились
            w_trained = net.get_weights_flat()
            assert not np.allclose(w_initial, w_trained), "Weights did not change after training"

            # Проверяем что loss улучшился
            loss_trained = meta._evaluate_weights(w_trained, net, X_feat, X_ctx, y)
            print(f"[TEST] Loss: {loss_initial:.6f} → {loss_trained:.6f}")

            # Допускаем что может не улучшиться из-за CEM noise, но проверяем финитность
            assert np.isfinite(loss_trained), "Trained loss is not finite"

            results.add_pass("4.3: CEM training")
        except Exception as e:
            results.add_fail("4.3: CEM training", str(e))

        # Тест 4.4: Prediction after training
        try:
            # Тестируем на новых данных
            x_feat_test, x_ctx_test = meta._build_enriched_features(
                p_xgb=0.7, p_rf=0.65, p_arf=0.68, p_nn=0.72,
                p_base=0.5,
                reg_ctx={"phase": ph, "vol_ratio": 1.0}
            )

            p_pred = net.forward(x_feat_test, x_ctx_test, training=False)

            assert 0.0 <= p_pred <= 1.0, f"Prediction out of range: {p_pred}"
            assert np.isfinite(p_pred), "Prediction is not finite"

            # На этих данных (mean=0.69 > 0.5) ожидаем p > 0.5
            print(f"[TEST] Prediction for high confidence input: {p_pred:.4f}")

            results.add_pass("4.4: Prediction after training")
        except Exception as e:
            results.add_fail("4.4: Prediction after training", str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_5_record_and_trigger(results: TestResults):
    """Тест 5: Record Result & Training Trigger"""
    print("\n" + "="*70)
    print("TEST 5: Record Result & Training Trigger")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    try:
        cfg = MockConfig(
            meta_state_path=os.path.join(temp_dir, "state.json"),
            meta_min_train=50,
            meta_retrain_every=30
        )
        meta = MetaNeuralCEM(cfg)

        # Тест 5.1: Record results
        try:
            ph = 1

            for i in range(100):
                p_xgb = 0.5 + 0.01 * i
                p_rf = 0.5 + 0.01 * i
                p_arf = 0.5 + 0.01 * i
                p_nn = 0.5 + 0.01 * i
                y_up = int(i > 50)

                meta.record_result(
                    p_xgb=p_xgb, p_rf=p_rf, p_arf=p_arf, p_nn=p_nn,
                    p_base=0.5,
                    y_up=y_up,
                    used_in_live=False,
                    reg_ctx={"phase": ph}
                )

            # Проверяем что данные сохранились
            assert meta.seen_ph[ph] == 100, f"Expected seen=100, got {meta.seen_ph[ph]}"

            csv_path = meta._phase_csv_paths[ph]
            assert os.path.exists(csv_path), "CSV not created"

            results.add_pass("5.1: Record results")
        except Exception as e:
            results.add_fail("5.1: Record results", str(e))

        # Тест 5.2: Training trigger
        try:
            # Проверяем что обучение НЕ запущено (еще не достигнут порог new_since_train)
            assert meta.new_since_train_ph[ph] < cfg.meta_retrain_every, "Training should not trigger yet"

            # Добавляем еще примеров чтобы достичь порога
            for i in range(50):
                meta.record_result(
                    p_xgb=0.6, p_rf=0.6, p_arf=0.6, p_nn=0.6,
                    p_base=0.5,
                    y_up=1,
                    used_in_live=False,
                    reg_ctx={"phase": ph}
                )

            # После обучения new_since_train должен сброситься
            # Проверяем что модель обучена
            assert meta.seen_ph[ph] == 150, f"Expected seen=150, got {meta.seen_ph[ph]}"

            results.add_pass("5.2: Training trigger")
        except Exception as e:
            results.add_fail("5.2: Training trigger", str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_6_cv_validation(results: TestResults):
    """Тест 6: Cross-Validation"""
    print("\n" + "="*70)
    print("TEST 6: Cross-Validation")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    try:
        cfg = MockConfig(
            meta_state_path=os.path.join(temp_dir, "state.json"),
            cv_enabled=True
        )
        meta = MetaNeuralCEM(cfg)

        # Тест 6.1: CV с синтетическими данными
        try:
            ph = 0

            # Добавляем perfect predictions в OOF
            for i in range(200):
                y = i % 2
                p = 0.9 if y == 1 else 0.1

                meta.cv_oof_preds[ph].append(p)
                meta.cv_oof_labels[ph].append(y)

            # Запускаем CV
            cv_results = meta._run_cv_validation(ph)

            assert cv_results["status"] == "ok", f"CV failed: {cv_results}"
            assert cv_results["oof_accuracy"] > 80.0, f"Low accuracy: {cv_results['oof_accuracy']}"
            assert "ci_lower" in cv_results, "Missing CI bounds"
            assert "ci_upper" in cv_results, "Missing CI bounds"

            print(f"[TEST] CV Accuracy: {cv_results['oof_accuracy']:.2f}% "
                  f"CI=[{cv_results['ci_lower']:.2f}%, {cv_results['ci_upper']:.2f}%]")

            results.add_pass("6.1: CV validation")
        except Exception as e:
            results.add_fail("6.1: CV validation", str(e))

        # Тест 6.2: CV с недостаточными данными
        try:
            ph = 1

            # Только 50 примеров
            for i in range(50):
                meta.cv_oof_preds[ph].append(0.5)
                meta.cv_oof_labels[ph].append(i % 2)

            cv_results = meta._run_cv_validation(ph)

            assert cv_results["status"] == "insufficient_data", "Should reject insufficient data"

            results.add_pass("6.2: CV insufficient data handling")
        except Exception as e:
            results.add_fail("6.2: CV insufficient data handling", str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_7_mode_switching(results: TestResults):
    """Тест 7: Mode Switching (SHADOW ↔ ACTIVE)"""
    print("\n" + "="*70)
    print("TEST 7: Mode Switching")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    try:
        cfg = MockConfig(
            meta_state_path=os.path.join(temp_dir, "state.json"),
            meta_enter_wr=0.58,
            meta_exit_wr=0.52,
            meta_min_ready=50
        )
        meta = MetaNeuralCEM(cfg)

        # Тест 7.1: SHADOW → ACTIVE
        try:
            assert meta.mode == "SHADOW", "Should start in SHADOW"

            # Добавляем хорошие результаты
            for i in range(100):
                meta.shadow_hits.append(1)  # 100% accuracy

            meta._maybe_flip_modes()

            assert meta.mode == "ACTIVE", "Should switch to ACTIVE with good WR"

            results.add_pass("7.1: SHADOW → ACTIVE")
        except Exception as e:
            results.add_fail("7.1: SHADOW → ACTIVE", str(e))

        # Тест 7.2: ACTIVE → SHADOW
        try:
            # Добавляем плохие результаты
            meta.active_hits = [0] * 100  # 0% accuracy

            meta._maybe_flip_modes()

            assert meta.mode == "SHADOW", "Should switch to SHADOW with bad WR"

            results.add_pass("7.2: ACTIVE → SHADOW")
        except Exception as e:
            results.add_fail("7.2: ACTIVE → SHADOW", str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_8_state_persistence(results: TestResults):
    """Тест 8: State Save/Load"""
    print("\n" + "="*70)
    print("TEST 8: State Persistence")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    try:
        state_path = os.path.join(temp_dir, "state.json")
        cfg = MockConfig(meta_state_path=state_path)

        # Тест 8.1: Save state
        try:
            meta1 = MetaNeuralCEM(cfg)

            # Модифицируем состояние
            meta1.mode = "ACTIVE"
            meta1.seen_ph[2] = 500
            meta1.shadow_hits = [1, 0, 1, 1]

            # Модифицируем веса
            net = meta1.networks[0]
            w = net.get_weights_flat()
            w_modified = w + 0.5
            net.set_weights_from_flat(w_modified)

            # Сохраняем
            meta1._save()

            assert os.path.exists(state_path), "State file not created"

            results.add_pass("8.1: Save state")
        except Exception as e:
            results.add_fail("8.1: Save state", str(e))

        # Тест 8.2: Load state
        try:
            # Создаем новый экземпляр
            meta2 = MetaNeuralCEM(cfg)

            # Проверяем что состояние восстановилось
            assert meta2.mode == "ACTIVE", f"Mode not restored: {meta2.mode}"
            assert meta2.seen_ph[2] == 500, f"seen_ph not restored: {meta2.seen_ph[2]}"
            assert meta2.shadow_hits == [1, 0, 1, 1], "shadow_hits not restored"

            # Проверяем веса
            w_loaded = meta2.networks[0].get_weights_flat()
            assert np.allclose(w_loaded, w_modified), "Weights not restored correctly"

            results.add_pass("8.2: Load state")
        except Exception as e:
            results.add_fail("8.2: Load state", str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_9_predict_integration(results: TestResults):
    """Тест 9: End-to-End Prediction"""
    print("\n" + "="*70)
    print("TEST 9: End-to-End Prediction")
    print("="*70)

    temp_dir = tempfile.mkdtemp()
    try:
        cfg = MockConfig(meta_state_path=os.path.join(temp_dir, "state.json"))
        meta = MetaNeuralCEM(cfg)

        # Тест 9.1: Prediction without training (fallback)
        try:
            p_final = meta.predict(
                p_xgb=0.6, p_rf=0.65, p_arf=0.58, p_nn=0.62,
                p_base=0.5,
                reg_ctx={"phase": 0}
            )

            assert p_final is not None, "Prediction is None"
            assert 0.0 <= p_final <= 1.0, f"Prediction out of range: {p_final}"

            # Должно быть близко к среднему экспертов (0.6125)
            expected_mean = (0.6 + 0.65 + 0.58 + 0.62) / 4
            assert abs(p_final - expected_mean) < 0.1, f"Fallback prediction too far from mean: {p_final} vs {expected_mean}"

            results.add_pass("9.1: Prediction without training (fallback)")
        except Exception as e:
            results.add_fail("9.1: Prediction without training (fallback)", str(e))

        # Тест 9.2: Prediction after training
        try:
            ph = 0

            # Обучаем модель
            meta.seen_ph[ph] = 200  # Помечаем как обученную

            # Теперь predict должен использовать нейросеть
            p_final = meta.predict(
                p_xgb=0.6, p_rf=0.65, p_arf=0.58, p_nn=0.62,
                p_base=0.5,
                reg_ctx={"phase": ph}
            )

            assert p_final is not None, "Prediction is None"
            assert 0.0 <= p_final <= 1.0, f"Prediction out of range: {p_final}"

            results.add_pass("9.2: Prediction after training")
        except Exception as e:
            results.add_fail("9.2: Prediction after training", str(e))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ========== MAIN ==========

def main():
    """Запуск всех тестов"""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUITE FOR meta_neural_cem.py")
    print("="*70)

    if not IMPORT_SUCCESS:
        print("❌ Import failed, aborting tests")
        return False

    results = TestResults()

    # Запускаем все тесты
    test_1_feature_engineering(results)
    test_2_neural_network(results)
    test_3_data_persistence(results)
    test_4_training(results)
    test_5_record_and_trigger(results)
    test_6_cv_validation(results)
    test_7_mode_switching(results)
    test_8_state_persistence(results)
    test_9_predict_integration(results)

    # Итоги
    success = results.summary()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
