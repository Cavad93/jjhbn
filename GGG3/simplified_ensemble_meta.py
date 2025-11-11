# -*- coding: utf-8 -*-
"""
simplified_ensemble_meta.py — Упрощенная META с контекстными фичами

=== АРХИТЕКТУРА ===
- Ridge регрессия для комбинирования экспертов с учетом контекста
- Phase-specific модели: 6 фаз × Ridge модель
- Входные фичи (11): 4 эксперта + 7 контекстных
  • Эксперты: p_xgb, p_rf, p_arf, p_nn
  • Контекст: vol_ratio, trend_macd, jump_detected, funding_sign, book_imb, ofi_15s, basis_pct
- SHADOW/ACTIVE режимы с валидацией
- ADWIN drift detection

=== ПРЕИМУЩЕСТВА ===
✅ META "видит" рынок через контекстные фичи
✅ Ridge регрессия: быстрая, интерпретируемая, устойчивая к overfitting
✅ Фичи фиксируются в момент ОТКРЫТИЯ позиции (t0)
✅ Результат фиксируется в момент ЗАКРЫТИЯ позиции
✅ Прозрачность (можно анализировать коэффициенты модели)
✅ Стабильность (L2 регуляризация предотвращает overfitting)

=== РЕЖИМЫ ===
- SHADOW: Обучается на данных, но не влияет на решения
- ACTIVE: Используется для предсказаний (переключается при WR >= 58%)

Автор: Claude (Anthropic)
Дата: 2025-11-11 (Enhanced with context features)
"""
from __future__ import annotations

import os
import json
import math
import traceback
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

import numpy as np

# Scikit-learn для Ridge регрессии
try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    print("WARNING: sklearn not installed. Install with: pip install scikit-learn")

# ========== ВНЕШНИЕ ЗАВИСИМОСТИ ==========

# River ADWIN для drift detection
try:
    from river.drift import ADWIN
    HAVE_RIVER = True
except Exception:
    ADWIN = None
    HAVE_RIVER = False

# Безопасное сохранение
try:
    from state_safety import atomic_save_json
except Exception:
    def atomic_save_json(path: str, obj: dict):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

# Извлечение фазы
try:
    from meta_ctx import phase_from_ctx
except Exception:
    def phase_from_ctx(ctx: Optional[dict]) -> int:
        return int(ctx.get("phase", 0) if isinstance(ctx, dict) else 0)


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def _safe_float(v, default=0.5) -> float:
    """Безопасное преобразование в float"""
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def extract_context_features(reg_ctx: Optional[dict]) -> np.ndarray:
    """
    Извлекает контекстные фичи из reg_ctx

    Args:
        reg_ctx: Словарь с контекстными фичами

    Returns:
        np.ndarray: [vol_ratio, trend_macd, jump_detected, funding_sign, book_imb, ofi_15s, basis_pct]
    """
    if reg_ctx is None:
        reg_ctx = {}

    return np.array([
        _safe_float(reg_ctx.get('vol_ratio', 1.0), 1.0),
        _safe_float(reg_ctx.get('trend_macd', 0.0), 0.0),
        float(reg_ctx.get('jump_detected', False)),  # bool -> 0.0 или 1.0
        _safe_float(reg_ctx.get('funding_sign', 0.0), 0.0),
        _safe_float(reg_ctx.get('book_imb', 0.0), 0.0),
        _safe_float(reg_ctx.get('ofi_15s', 0.0), 0.0),
        _safe_float(reg_ctx.get('basis_pct', 0.0), 0.0)
    ])


# ========== PHASE-SPECIFIC ENSEMBLE ==========

@dataclass
class PhaseEnsemble:
    """
    Ансамбль для одной фазы рынка с контекстными фичами

    Модель: Ridge регрессия
    Входные фичи (11): [p_xgb, p_rf, p_arf, p_nn, vol_ratio, trend_macd, jump_detected,
                        funding_sign, book_imb, ofi_15s, basis_pct]
    Выход: p_final (вероятность роста)
    """
    phase: int
    model: Optional[Ridge] = None  # Ridge регрессия
    scaler: Optional[StandardScaler] = None  # Нормализация фич

    # Данные для обучения
    X: List[List[float]] = None  # История: [4 эксперта + 7 контекстных фич]
    y: List[int] = None  # История таргетов

    # Статистика
    n_samples: int = 0
    n_trained: int = 0
    last_score: float = 0.0  # R² score

    def __post_init__(self):
        if not HAVE_SKLEARN:
            raise ImportError("sklearn required for context-aware META")

        if self.model is None:
            # Ridge с L2 регуляризацией
            self.model = Ridge(alpha=1.0, fit_intercept=True)

        if self.scaler is None:
            self.scaler = StandardScaler()

        if self.X is None:
            self.X = []
        if self.y is None:
            self.y = []

    def predict(
        self,
        p_xgb: float,
        p_rf: float,
        p_arf: float,
        p_nn: float,
        reg_ctx: Optional[dict] = None
    ) -> float:
        """
        Предсказание с учетом контекста

        Args:
            p_xgb, p_rf, p_arf, p_nn: Вероятности от экспертов
            reg_ctx: Контекстные фичи

        Returns:
            p_final: Финальная вероятность
        """
        # Формируем вектор фич
        expert_preds = np.array([
            _safe_float(p_xgb, 0.5),
            _safe_float(p_rf, 0.5),
            _safe_float(p_arf, 0.5),
            _safe_float(p_nn, 0.5)
        ])

        context_feats = extract_context_features(reg_ctx)
        full_features = np.concatenate([expert_preds, context_feats])

        # Если модель не обучена, возвращаем среднее экспертов
        if self.n_trained == 0:
            return float(np.clip(np.mean(expert_preds), 0.0, 1.0))

        try:
            # Нормализуем и предсказываем
            X_scaled = self.scaler.transform(full_features.reshape(1, -1))
            p_final = self.model.predict(X_scaled)[0]

            # Clip в [0, 1]
            return float(np.clip(p_final, 0.0, 1.0))

        except Exception as e:
            # Fallback: среднее экспертов
            return float(np.clip(np.mean(expert_preds), 0.0, 1.0))

    def add_sample(
        self,
        p_xgb: float,
        p_rf: float,
        p_arf: float,
        p_nn: float,
        reg_ctx: Optional[dict],
        y_true: int
    ):
        """Добавить пример для обучения"""
        expert_preds = [
            _safe_float(p_xgb, 0.5),
            _safe_float(p_rf, 0.5),
            _safe_float(p_arf, 0.5),
            _safe_float(p_nn, 0.5)
        ]

        context_feats = extract_context_features(reg_ctx).tolist()
        full_features = expert_preds + context_feats

        self.X.append(full_features)
        self.y.append(int(y_true))
        self.n_samples += 1

        # Ограничиваем историю
        max_history = 2000
        if len(self.X) > max_history:
            self.X = self.X[-max_history:]
            self.y = self.y[-max_history:]

    def train_model(self, min_samples: int = 50) -> bool:
        """
        Обучает Ridge регрессию на накопленных данных

        Returns:
            bool: True если обучение прошло успешно
        """
        if len(self.X) < min_samples:
            return False

        X_array = np.array(self.X)
        y_array = np.array(self.y)

        try:
            # Нормализуем фичи
            X_scaled = self.scaler.fit_transform(X_array)

            # Обучаем Ridge
            self.model.fit(X_scaled, y_array)

            # Оцениваем качество
            self.last_score = self.model.score(X_scaled, y_array)
            self.n_trained += 1

            return True

        except Exception as e:
            print(f"[PhaseEnsemble] Training failed for phase {self.phase}: {e}")
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Возвращает важность фич (коэффициенты Ridge)

        Returns:
            Dict с важностью каждой фичи
        """
        if self.n_trained == 0:
            return {}

        feature_names = [
            'xgb', 'rf', 'arf', 'nn',
            'vol_ratio', 'trend_macd', 'jump_detected',
            'funding_sign', 'book_imb', 'ofi_15s', 'basis_pct'
        ]

        coefs = self.model.coef_

        return {name: float(coef) for name, coef in zip(feature_names, coefs)}


# ========== SIMPLIFIED ENSEMBLE META ==========

class SimplifiedEnsembleMETA:
    """
    Упрощенная META модель на основе взвешенного ансамбля

    Основные принципы:
    - Простота > Сложность
    - Прозрачность > Black box
    - Скорость > Точность (в разумных пределах)

    Параметры: 24 (6 фаз × 4 веса) вместо 31200
    Обучение: <1 секунда вместо 10-30 минут
    """

    def __init__(self, cfg):
        """
        Инициализация упрощенной META

        Args:
            cfg: Конфигурация с параметрами
        """
        self.cfg = cfg
        self.state_path = getattr(cfg, "meta_state_path", "simplified_meta_state.json")
        self.enabled = True
        self.mode = "SHADOW"  # Начинаем в SHADOW режиме

        # Phase-specific ensembles
        self.ensembles: Dict[int, PhaseEnsemble] = {}
        for ph in range(6):
            self.ensembles[ph] = PhaseEnsemble(phase=ph)

        # ADWIN для drift detection
        self.adwin = ADWIN(delta=0.002) if HAVE_RIVER else None

        # Счетчики для режимов
        self.active_hits = []  # Hit rate в ACTIVE режиме
        self.shadow_hits = []  # Hit rate в SHADOW режиме

        # Статистика
        self.seen_ph: Dict[int, int] = {ph: 0 for ph in range(6)}
        self.new_since_train_ph: Dict[int, int] = {ph: 0 for ph in range(6)}

        # Сохранение
        self._unsaved = 0
        self._save_every = 50

        # Загружаем состояние если есть
        if os.path.exists(self.state_path):
            self._load()

    # ========== ПРЕДСКАЗАНИЕ ==========

    def predict(
        self,
        p_xgb: Optional[float],
        p_rf: Optional[float],
        p_arf: Optional[float],
        p_nn: Optional[float],
        p_base: Optional[float],
        reg_ctx: Optional[dict] = None
    ) -> Optional[float]:
        """
        Предсказание финальной вероятности с учетом контекста

        Args:
            p_xgb, p_rf, p_arf, p_nn: Вероятности от экспертов
            p_base: Вероятность от BASE логики (не используется)
            reg_ctx: Контекст с фазой рынка и контекстными фичами

        Returns:
            p_final: Финальная вероятность или None
        """
        if not self.enabled:
            return None

        ph = phase_from_ctx(reg_ctx)
        ensemble = self.ensembles.get(ph)

        if ensemble is None:
            return None

        # Проверяем минимальное количество примеров
        min_ready = int(getattr(self.cfg, "meta_min_ready", 80))
        if self.seen_ph[ph] < min_ready:
            # Еще не готовы - возвращаем простое среднее
            preds = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
            if len(preds) == 0:
                return None
            return float(np.clip(np.mean(preds), 0.0, 1.0))

        # Предсказание с учетом контекста
        try:
            p_final = ensemble.predict(p_xgb, p_rf, p_arf, p_nn, reg_ctx)
            return p_final
        except Exception as e:
            print(f"[SimplifiedMETA] Prediction error: {e}")
            return None

    # ========== ЗАПИСЬ РЕЗУЛЬТАТА ==========

    def record_result(
        self,
        p_xgb: Optional[float],
        p_rf: Optional[float],
        p_arf: Optional[float],
        p_nn: Optional[float],
        p_base: Optional[float],
        y_up: int,
        used_in_live: bool,
        p_final_used: Optional[float] = None,
        reg_ctx: Optional[dict] = None
    ) -> None:
        """
        Записывает результат и триггерит обучение

        Args:
            p_xgb, p_rf, p_arf, p_nn: Вероятности от экспертов (из t0)
            p_base: BASE логика (не используется)
            y_up: Фактический результат (0 или 1) (из момента закрытия)
            used_in_live: Использовался ли в живой торговле
            p_final_used: Финальная вероятность (для метрик)
            reg_ctx: Контекст с фазой и контекстными фичами (из t0)
        """
        try:
            ph = phase_from_ctx(reg_ctx)
            ensemble = self.ensembles.get(ph)

            if ensemble is None:
                return

            # Добавляем пример с контекстными фичами
            ensemble.add_sample(p_xgb, p_rf, p_arf, p_nn, reg_ctx, y_up)
            self.seen_ph[ph] += 1
            self.new_since_train_ph[ph] += 1

            # Обновление hit rate
            if p_final_used is not None:
                p_for_gate = p_final_used
            else:
                # Предсказываем на основе текущей модели
                min_ready = int(getattr(self.cfg, "meta_min_ready", 80))
                if self.seen_ph[ph] >= min_ready:
                    p_for_gate = ensemble.predict(p_xgb, p_rf, p_arf, p_nn, reg_ctx)
                else:
                    preds = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
                    p_for_gate = float(np.mean(preds)) if preds else 0.5

            hit = int((p_for_gate >= 0.5) == bool(y_up))

            # Обновляем метрики в зависимости от режима
            if self.mode == "ACTIVE" and used_in_live:
                self.active_hits.append(hit)

                # ADWIN drift detection
                if self.adwin is not None:
                    in_drift = self.adwin.update(1 - hit)
                    if in_drift:
                        self.mode = "SHADOW"
                        self.active_hits = []
                        print(f"[SimplifiedMETA] 🔄 ACTIVE→SHADOW: drift detected")
            else:
                self.shadow_hits.append(hit)

            # Ограничиваем историю
            self.active_hits = self.active_hits[-2000:]
            self.shadow_hits = self.shadow_hits[-2000:]

            # Обучение при готовности
            retrain_every = int(getattr(self.cfg, "meta_retrain_every", 100))
            if self.new_since_train_ph[ph] >= retrain_every:
                self._train_phase(ph)

            # Переключение режимов
            self._maybe_flip_modes()

            # Периодическое сохранение
            self._unsaved += 1
            if self._unsaved >= self._save_every:
                self._save()
                self._unsaved = 0

        except Exception as e:
            print(f"[SimplifiedMETA] record_result error: {e}")
            traceback.print_exc()

    def settle(self, *args, **kwargs):
        """Alias для record_result (для совместимости)"""
        return self.record_result(*args, **kwargs)

    # ========== ОБУЧЕНИЕ ==========

    def _train_phase(self, ph: int):
        """
        Обучение Ridge модели для одной фазы

        Быстро: Ridge регрессия за <1 секунду
        """
        ensemble = self.ensembles.get(ph)
        if ensemble is None:
            return

        min_samples = int(getattr(self.cfg, "meta_min_train", 150))
        if ensemble.n_samples < min_samples:
            return

        print(f"[SimplifiedMETA] 🎯 Training phase {ph} ({ensemble.n_samples} samples)")

        try:
            # Обучение Ridge модели
            success = ensemble.train_model(min_samples=min_samples)

            if success:
                # Выводим feature importance
                importance = ensemble.get_feature_importance()
                print(f"[SimplifiedMETA] ✅ Phase {ph} trained (R²={ensemble.last_score:.3f}):")
                print(f"    Experts: XGB={importance.get('xgb', 0):.3f}, "
                      f"RF={importance.get('rf', 0):.3f}, "
                      f"ARF={importance.get('arf', 0):.3f}, "
                      f"NN={importance.get('nn', 0):.3f}")
                print(f"    Context: vol_ratio={importance.get('vol_ratio', 0):.3f}, "
                      f"book_imb={importance.get('book_imb', 0):.3f}, "
                      f"funding={importance.get('funding_sign', 0):.3f}")

                self.new_since_train_ph[ph] = 0
            else:
                print(f"[SimplifiedMETA] ⚠️ Phase {ph} training failed")

        except Exception as e:
            print(f"[SimplifiedMETA] ❌ Training failed for phase {ph}: {e}")
            traceback.print_exc()

    # ========== ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ ==========

    def _maybe_flip_modes(self):
        """
        Переключение SHADOW ↔ ACTIVE на основе win rate

        Критерии:
        - SHADOW → ACTIVE: WR >= 58% на 80+ примерах
        - ACTIVE → SHADOW: WR < 52% или drift detected
        """
        enter_wr = float(getattr(self.cfg, "meta_enter_wr", 0.58))
        exit_wr = float(getattr(self.cfg, "meta_exit_wr", 0.52))
        min_ready = int(getattr(self.cfg, "meta_min_ready", 80))

        def _wr(hits):
            return sum(hits) / len(hits) if hits else None

        wr_shadow = _wr(self.shadow_hits[-100:]) if len(self.shadow_hits) >= 50 else None
        wr_active = _wr(self.active_hits[-100:]) if len(self.active_hits) >= 50 else None

        # SHADOW → ACTIVE
        if self.mode == "SHADOW" and wr_shadow is not None and len(self.shadow_hits) >= min_ready:
            if wr_shadow >= enter_wr:
                self.mode = "ACTIVE"
                print(f"[SimplifiedMETA] 🚀 SHADOW→ACTIVE: WR={wr_shadow:.2%}")

        # ACTIVE → SHADOW
        if self.mode == "ACTIVE" and wr_active is not None:
            if wr_active < exit_wr:
                self.mode = "SHADOW"
                print(f"[SimplifiedMETA] 🛑 ACTIVE→SHADOW: WR={wr_active:.2%}")

    # ========== СТАТУС ==========

    def status(self) -> Dict[str, str]:
        """Возвращает статус META"""
        def _wr(hits):
            return sum(hits) / len(hits) if hits else None

        wr_shadow = _wr(self.shadow_hits[-100:])
        wr_active = _wr(self.active_hits[-100:])

        return {
            "mode": self.mode,
            "enabled": str(self.enabled),
            "wr_shadow": f"{wr_shadow:.2%}" if wr_shadow is not None else "N/A",
            "wr_active": f"{wr_active:.2%}" if wr_active is not None else "N/A",
            "n_shadow": str(len(self.shadow_hits)),
            "n_active": str(len(self.active_hits)),
            "total_samples": str(sum(self.seen_ph.values()))
        }

    def get_phase_weights(self, ph: int) -> Optional[Dict[str, float]]:
        """
        Возвращает feature importance для конкретной фазы

        УСТАРЕЛО: Используйте get_phase_importance() для новой версии
        """
        return self.get_phase_importance(ph)

    def get_phase_importance(self, ph: int) -> Optional[Dict[str, float]]:
        """Возвращает важность фич для конкретной фазы"""
        ensemble = self.ensembles.get(ph)
        if ensemble is None:
            return None
        return ensemble.get_feature_importance()

    # ========== СОХРАНЕНИЕ/ЗАГРУЗКА ==========

    def _save(self):
        """Сохранение состояния (модели + метаданные)"""
        try:
            import pickle

            # Сохраняем каждую модель отдельно (бинарные данные)
            models_dir = os.path.dirname(self.state_path)
            os.makedirs(models_dir, exist_ok=True)

            ensembles_state = {}
            for ph, ens in self.ensembles.items():
                # Сохраняем Ridge модель и scaler
                model_path = os.path.join(models_dir, f"meta_phase_{ph}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': ens.model,
                        'scaler': ens.scaler
                    }, f)

                ensembles_state[str(ph)] = {
                    'model_path': model_path,
                    'n_samples': ens.n_samples,
                    'n_trained': ens.n_trained,
                    'last_score': ens.last_score
                }

            # Метаданные в JSON
            state = {
                'version': '2.0',  # Новая версия с Ridge
                'mode': self.mode,
                'enabled': self.enabled,
                'ensembles': ensembles_state,
                'seen_ph': self.seen_ph,
                'new_since_train_ph': self.new_since_train_ph,
                'active_hits': self.active_hits[-500:],
                'shadow_hits': self.shadow_hits[-500:]
            }

            atomic_save_json(self.state_path, state)

        except Exception as e:
            print(f"[SimplifiedMETA] Save error: {e}")
            traceback.print_exc()

    def _load(self):
        """Загрузка состояния (модели + метаданные)"""
        try:
            import pickle

            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            version = state.get("version", "1.0")
            self.mode = state.get("mode", "SHADOW")
            self.enabled = state.get("enabled", True)

            # Восстановление ансамблей
            ensembles_state = state.get("ensembles", {})
            for p_str, ens_data in ensembles_state.items():
                ph = int(p_str)
                if ph in self.ensembles:
                    # Загружаем модель
                    if version == "2.0" and 'model_path' in ens_data:
                        model_path = ens_data['model_path']
                        if os.path.exists(model_path):
                            with open(model_path, 'rb') as f:
                                model_data = pickle.load(f)
                                self.ensembles[ph].model = model_data['model']
                                self.ensembles[ph].scaler = model_data['scaler']
                        else:
                            print(f"[SimplifiedMETA] ⚠️ Model file not found for phase {ph}: {model_path}")
                    elif 'weights' in ens_data:
                        # Старая версия (v1.0) - игнорируем веса, т.к. теперь используем Ridge
                        print(f"[SimplifiedMETA] ⚠️ Phase {ph}: old format detected, will retrain")

                    # Метаданные
                    self.ensembles[ph].n_samples = ens_data.get('n_samples', 0)
                    self.ensembles[ph].n_trained = ens_data.get('n_trained', 0)
                    self.ensembles[ph].last_score = ens_data.get('last_score', 0.0)

            self.seen_ph = state.get("seen_ph", {ph: 0 for ph in range(6)})
            # Преобразуем ключи в int
            self.seen_ph = {int(k): v for k, v in self.seen_ph.items()}

            self.new_since_train_ph = state.get("new_since_train_ph", {ph: 0 for ph in range(6)})
            self.new_since_train_ph = {int(k): v for k, v in self.new_since_train_ph.items()}

            self.active_hits = state.get("active_hits", [])
            self.shadow_hits = state.get("shadow_hits", [])

            print(f"[SimplifiedMETA] ✅ Loaded state: version={version}, mode={self.mode}, "
                  f"samples={sum(self.seen_ph.values())}")

        except Exception as e:
            print(f"[SimplifiedMETA] Load error: {e}")
            traceback.print_exc()


# ========== ТЕСТИРОВАНИЕ ==========

if __name__ == "__main__":
    print("="*80)
    print("ТЕСТ SIMPLIFIED ENSEMBLE META")
    print("="*80)

    # Mock конфигурация
    class MockConfig:
        meta_state_path = "/tmp/test_simplified_meta.json"
        meta_min_ready = 50
        meta_min_train = 100
        meta_retrain_every = 50
        meta_enter_wr = 0.58
        meta_exit_wr = 0.52

    cfg = MockConfig()

    # Создание META
    print("\n1. Создание META...")
    meta = SimplifiedEnsembleMETA(cfg)
    print(f"   ✅ Создано: {len(meta.ensembles)} phase ensembles")

    # Симуляция данных
    print("\n2. Симуляция данных...")
    np.random.seed(42)
    n_samples = 200

    for i in range(n_samples):
        # Генерируем предсказания экспертов
        p_xgb = np.random.uniform(0.3, 0.7)
        p_rf = p_xgb + np.random.normal(0, 0.1)
        p_arf = p_xgb + np.random.normal(0, 0.15)
        p_nn = p_xgb + np.random.normal(0, 0.12)

        # Генерируем контекстные фичи
        ph = i % 6
        reg_ctx = {
            "phase": ph,
            "vol_ratio": np.random.uniform(0.8, 1.2),
            "trend_macd": np.random.uniform(-0.1, 0.1),
            "jump_detected": np.random.choice([True, False], p=[0.1, 0.9]),
            "funding_sign": np.random.uniform(-0.01, 0.01),
            "book_imb": np.random.uniform(-0.05, 0.05),
            "ofi_15s": np.random.uniform(-100, 100),
            "basis_pct": np.random.uniform(-0.001, 0.001)
        }

        # Таргет коррелирован с экспертами + контекстом
        p_true = (p_xgb + p_rf + p_arf + p_nn) / 4.0
        # Добавляем влияние контекста
        if reg_ctx['vol_ratio'] > 1.1:
            p_true += 0.05  # Высокая волатильность -> выше шанс роста
        if reg_ctx['book_imb'] > 0:
            p_true += reg_ctx['book_imb']  # Order book imbalance влияет

        y_up = int(p_true > 0.5)

        # Предсказание
        p_final = meta.predict(p_xgb, p_rf, p_arf, p_nn, 0.5, reg_ctx)

        # Запись результата
        meta.record_result(p_xgb, p_rf, p_arf, p_nn, 0.5, y_up, True, p_final, reg_ctx)

    print(f"   ✅ Обработано {n_samples} примеров")

    # Статус
    print("\n3. Статус META...")
    status = meta.status()
    for key, value in status.items():
        print(f"   {key}: {value}")

    # Feature importance по фазам
    print("\n4. Feature importance по фазам...")
    for ph in range(6):
        importance = meta.get_phase_importance(ph)
        if importance:
            print(f"   Phase {ph}:")
            print(f"      Experts: XGB={importance.get('xgb', 0):.3f}, "
                  f"RF={importance.get('rf', 0):.3f}, "
                  f"ARF={importance.get('arf', 0):.3f}, "
                  f"NN={importance.get('nn', 0):.3f}")
            print(f"      Context: vol_ratio={importance.get('vol_ratio', 0):.3f}, "
                  f"book_imb={importance.get('book_imb', 0):.3f}")

    # Сохранение
    print("\n5. Сохранение...")
    meta._save()
    print("   ✅ Сохранено")

    # Загрузка
    print("\n6. Загрузка...")
    meta2 = SimplifiedEnsembleMETA(cfg)
    print(f"   ✅ Загружено: mode={meta2.mode}, samples={sum(meta2.seen_ph.values())}")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("="*80)
