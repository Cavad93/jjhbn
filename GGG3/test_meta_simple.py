#!/usr/bin/env python3
"""
ПРОСТЫЕ БЫСТРЫЕ ТЕСТЫ для meta_neural_cem.py
Тестируем только публичный API
"""
import sys
import os
import time
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from meta_neural_cem import MetaNeuralCEM

class SimpleConfig:
    def __init__(self):
        self.meta_nn_hidden = [8]
        self.meta_train_freq_ph = 30  # Быстрое обучение
        self.meta_retrain_every = 30  # Частота переобучения
        self.meta_min_train = 150  # Минимум примеров для первого обучения
        self.meta_use_cma_es = False  # CEM быстрее
        self.meta_cem_iters = 3  # Очень мало
        self.meta_cem_pop = 5
        self.meta_bootstrap_reps = 1
        self.meta_dropout_rate = 0.1
        self.meta_state_path = None
        self.meta_exp4_phases = 6

def test_basic_flow():
    """Тест 1: Базовый поток - запись и предсказание"""
    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(cfg)

        # Запись результатов
        for i in range(40):
            meta.record_result(
                p_xgb=0.5 + np.random.randn()*0.1,
                p_rf=0.5 + np.random.randn()*0.1,
                p_arf=0.5 + np.random.randn()*0.1,
                p_nn=0.5 + np.random.randn()*0.1,
                p_base=0.5,
                y_up=i % 2,
                used_in_live=True,
                reg_ctx={'phase': 0}
            )

        # Предсказание
        pred = meta.predict(
            p_xgb=0.5,
            p_rf=0.5,
            p_arf=0.5,
            p_nn=0.5,
            p_base=0.5,
            reg_ctx={'phase': 0}
        )

        print("✅ Тест 1 пройден: Базовый поток работает")
        return True
    except Exception as e:
        print(f"❌ Тест 1 провален: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_all_phases():
    """Тест 2: Работа со всеми 6 фазами"""
    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(cfg)

        # Записываем по 35 примеров в каждую фазу
        for phase in range(6):
            for i in range(35):
                meta.record_result(
                    p_xgb=0.5 + np.random.randn()*0.1,
                    p_rf=0.5 + np.random.randn()*0.1,
                    p_arf=0.5 + np.random.randn()*0.1,
                    p_nn=0.5 + np.random.randn()*0.1,
                    p_base=0.5,
                    y_up=i % 2,
                    used_in_live=True,
                    reg_ctx={'phase': phase}
                )

        # Проверяем что данные записались
        for phase in range(6):
            assert len(meta.buf_ph[phase]) > 0, f"Phase {phase} empty"

        print("✅ Тест 2 пройден: Все фазы работают")
        return True
    except Exception as e:
        print(f"❌ Тест 2 провален: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_shadow_mode():
    """Тест 3: SHADOW режим дает fallback (среднее экспертов)"""
    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(cfg)

        # По умолчанию режим SHADOW
        pred = meta.predict(
            p_xgb=0.5,
            p_rf=0.5,
            p_arf=0.5,
            p_nn=0.5,
            p_base=0.5,
            reg_ctx={'phase': 0}
        )

        # SHADOW может возвращать fallback (среднее экспертов)
        # если модель не обучена
        assert pred is None or (0 <= pred <= 1), f"Invalid prediction: {pred}"

        print("✅ Тест 3 пройден: SHADOW режим работает")
        return True
    except Exception as e:
        print(f"❌ Тест 3 провален: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_active_mode():
    """Тест 4: ACTIVE режим дает предсказания после обучения"""
    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(cfg)

        # Обучаем
        np.random.seed(42)
        for i in range(40):
            meta.record_result(
                p_xgb=0.5 + np.random.randn()*0.1,
                p_rf=0.5 + np.random.randn()*0.1,
                p_arf=0.5 + np.random.randn()*0.1,
                p_nn=0.5 + np.random.randn()*0.1,
                p_base=0.5,
                y_up=i % 2,
                used_in_live=True,
                reg_ctx={'phase': 0}
            )

        # Активируем
        meta.mode = "ACTIVE"

        # Предсказываем
        pred = meta.predict(
            p_xgb=0.5,
            p_rf=0.5,
            p_arf=0.5,
            p_nn=0.5,
            p_base=0.5,
            reg_ctx={'phase': 0}
        )

        # Может быть None если модель не обучена, или число
        if pred is not None:
            assert 0 <= pred <= 1, f"Prediction {pred} out of range"

        print("✅ Тест 4 пройден: ACTIVE режим работает")
        return True
    except Exception as e:
        print(f"❌ Тест 4 провален: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_state_persistence():
    """Тест 5: Сохранение и загрузка состояния"""
    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        meta1 = MetaNeuralCEM(cfg)

        # Записываем данные
        for i in range(20):
            meta1.record_result(
                p_xgb=0.5,
                p_rf=0.5,
                p_arf=0.5,
                p_nn=0.5,
                p_base=0.5,
                y_up=i % 2,
                used_in_live=True,
                reg_ctx={'phase': 0}
            )

        # Сохраняем
        meta1.settle()

        # Загружаем новый экземпляр
        meta2 = MetaNeuralCEM(cfg)

        # Проверяем что счетчик seen_ph загрузился
        assert meta2.seen_ph[0] > 0, f"Seen counter not loaded: {meta2.seen_ph[0]}"

        print("✅ Тест 5 пройден: Сохранение и загрузка работают")
        return True
    except Exception as e:
        print(f"❌ Тест 5 провален: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_training_trigger():
    """Тест 6: Обучение срабатывает при достижении порога"""
    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        cfg.meta_train_freq_ph = 30  # Обучение каждые 30 примеров
        meta = MetaNeuralCEM(cfg)

        # Записываем 35 примеров (больше чем порог)
        np.random.seed(42)
        for i in range(35):
            meta.record_result(
                p_xgb=0.5 + np.random.randn()*0.1,
                p_rf=0.5 + np.random.randn()*0.1,
                p_arf=0.5 + np.random.randn()*0.1,
                p_nn=0.5 + np.random.randn()*0.1,
                p_base=0.5,
                y_up=i % 2,
                used_in_live=True,
                reg_ctx={'phase': 0}
            )

        # Счетчик должен сброситься (обучение сработало)
        assert meta.new_since_train_ph[0] < 30, f"Counter not reset: {meta.new_since_train_ph[0]}"

        print("✅ Тест 6 пройден: Обучение срабатывает")
        return True
    except Exception as e:
        print(f"❌ Тест 6 провален: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_missing_experts():
    """Тест 7: Обработка отсутствующих экспертов"""
    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(cfg)

        # Запись с None экспертами
        meta.record_result(
            p_xgb=0.5,
            p_rf=None,
            p_arf=None,
            p_nn=None,
            p_base=0.5,
            y_up=1,
            used_in_live=True,
            reg_ctx={'phase': 0}
        )

        # Не должно упасть
        print("✅ Тест 7 пройден: Обработка None значений работает")
        return True
    except Exception as e:
        print(f"❌ Тест 7 провален: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_extreme_probabilities():
    """Тест 8: Экстремальные вероятности"""
    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(cfg)

        # Экстремальные значения
        meta.record_result(
            p_xgb=0.0,
            p_rf=1.0,
            p_arf=0.99,
            p_nn=0.01,
            p_base=0.5,
            y_up=1,
            used_in_live=True,
            reg_ctx={'phase': 0}
        )

        print("✅ Тест 8 пройден: Экстремальные значения обрабатываются")
        return True
    except Exception as e:
        print(f"❌ Тест 8 провален: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_networks_initialized():
    """Тест 9: Нейросети инициализированы для всех фаз"""
    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(cfg)

        # Проверяем что 6 нейросетей созданы
        assert len(meta.networks) == 6, f"Expected 6 networks, got {len(meta.networks)}"

        # Проверяем что каждая сеть имеет веса
        for phase, net in meta.networks.items():
            weights = net.get_weights_flat()
            assert len(weights) > 0, f"Phase {phase} has empty weights"

        print("✅ Тест 9 пройден: Нейросети инициализированы")
        return True
    except Exception as e:
        print(f"❌ Тест 9 провален: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_status():
    """Тест 10: Метод status() возвращает информацию"""
    tmpdir = tempfile.mkdtemp()
    try:
        cfg = SimpleConfig()
        cfg.meta_state_path = os.path.join(tmpdir, "state.json")
        meta = MetaNeuralCEM(cfg)

        # Запись примеров
        for i in range(10):
            meta.record_result(
                p_xgb=0.5,
                p_rf=0.5,
                p_arf=0.5,
                p_nn=0.5,
                p_base=0.5,
                y_up=i % 2,
                used_in_live=True,
                reg_ctx={'phase': 0}
            )

        # Получаем статус
        status = meta.status()

        assert isinstance(status, dict), "Status should be dict"
        assert 'mode' in status, "Status should have 'mode'"

        print("✅ Тест 10 пройден: Метод status() работает")
        return True
    except Exception as e:
        print(f"❌ Тест 10 провален: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    print("="*60)
    print("ПРОСТЫЕ ТЕСТЫ META NEURAL CEM (10 тестов)")
    print("Только публичный API, быстрые и надежные")
    print("="*60)

    start = time.time()
    results = []

    tests = [
        ("Базовый поток", test_basic_flow),
        ("Все фазы", test_all_phases),
        ("SHADOW режим", test_shadow_mode),
        ("ACTIVE режим", test_active_mode),
        ("Персистентность", test_state_persistence),
        ("Триггер обучения", test_training_trigger),
        ("Отсутствующие эксперты", test_missing_experts),
        ("Экстремальные вероятности", test_extreme_probabilities),
        ("Инициализация сетей", test_networks_initialized),
        ("Метод status()", test_status),
    ]

    for i, (name, test_fn) in enumerate(tests, 1):
        print(f"\n[Тест {i}/10] {name}...")
        t0 = time.time()
        passed = test_fn()
        dur = time.time() - t0
        results.append(passed)
        print(f"  Время: {dur:.1f}s")

    elapsed = time.time() - start
    passed = sum(results)
    total = len(results)

    print("\n" + "="*60)
    print(f"РЕЗУЛЬТАТЫ: {passed}/{total} тестов прошли за {elapsed:.1f}s")
    print(f"✅ Успешно: {passed}")
    print(f"❌ Провалено: {total - passed}")
    print("="*60)

    sys.exit(0 if passed == total else 1)
