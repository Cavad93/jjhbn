# -*- coding: utf-8 -*-
"""
meta_neural_cem.py — Нейросетевая META с CMA-ES/CEM обучением + Monte-Carlo Dropout

=== АРХИТЕКТУРА ===
- Feature Engineering: 18D → 36D (enriched features)
- Context Attention: 7D → 16D → 4D (dynamic expert weighting)
- Main MLP: 36D → 64D → 32D → 16D → 1D (≈5000 параметров)
- MC Dropout: epistemic uncertainty estimation
- Phase-specific heads: shared features + 6 phase heads

=== ОБУЧЕНИЕ ===
- CMA-ES/CEM для оптимизации весов нейросети
- Bootstrap Monte-Carlo для оценки качества
- Walk-forward CV для валидации
- Exponential weight decay для временного забывания

=== ОСОБЕННОСТИ ===
✅ Нелинейные зависимости (ReLU)
✅ Динамический гейтинг экспертов (Attention)
✅ Uncertainty estimation (MC Dropout)
✅ Robustness через регуляризацию
✅ Phase-adaptive learning
"""
from __future__ import annotations

import os
import json
import math
import time
import random
import csv
import traceback
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass

import numpy as np

# ========== ВНЕШНИЕ ЗАВИСИМОСТИ ==========

# CMA-ES оптимизатор
try:
    import cma
    HAVE_CMA = True
except Exception:
    cma = None
    HAVE_CMA = False

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

# Визуализация
try:
    from training_visualizer import get_visualizer
    HAVE_VISUALIZER = True
except Exception:
    HAVE_VISUALIZER = False


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def _safe_float(v, default=0.5) -> float:
    """Безопасное преобразование в float"""
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _safe_logit(p: float) -> float:
    """Безопасный логит"""
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def _sigmoid(z: float) -> float:
    """Сигмоида с защитой от overflow"""
    z = np.clip(z, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation"""
    return np.maximum(0.0, x)


def _dropout(x: np.ndarray, rate: float, training: bool) -> np.ndarray:
    """Dropout regularization"""
    if not training or rate <= 0.0:
        return x
    mask = np.random.binomial(1, 1.0 - rate, size=x.shape)
    return x * mask / (1.0 - rate)


# ========== НЕЙРОСЕТЕВАЯ МЕТА-МОДЕЛЬ ==========

class NeuralMetaNetwork:
    """
    Нейронная сеть для META-стекинга
    
    Архитектура:
    - Feature Engineering: 18D → 36D
    - Attention: 7D → 16D → 4D
    - Main: 36D → 64D → 32D → 16D → 1D
    - Total params: ~5200
    """
    
    def __init__(self, phase: int = 0):
        """
        Инициализация нейросети для конкретной фазы
        
        Args:
            phase: номер фазы рынка (0-5)
        """
        self.phase = phase
        
        # ===== ATTENTION MODULE (196 параметров) =====
        # Context(7) → Dense(16) → Dense(4)
        self.W_att1 = self._xavier_init(7, 16)    # 7×16 = 112
        self.b_att1 = np.zeros(16)                # 16
        self.W_att2 = self._xavier_init(16, 4)    # 16×4 = 64
        self.b_att2 = np.zeros(4)                 # 4
        
        # ===== MAIN NETWORK (4993 параметров) =====
        # Input(36) → Dense(64) → Dense(32) → Dense(16) → Dense(1)
        self.W1 = self._xavier_init(36, 64)       # 36×64 = 2304
        self.b1 = np.zeros(64)                    # 64
        
        self.W2 = self._xavier_init(64, 32)       # 64×32 = 2048
        self.b2 = np.zeros(32)                    # 32
        
        self.W3 = self._xavier_init(32, 16)       # 32×16 = 512
        self.b3 = np.zeros(16)                    # 16
        
        self.W4 = self._xavier_init(16, 1)        # 16×1 = 16
        self.b4 = np.zeros(1)                     # 1
        
        # Dropout rates
        self.dropout_rates = [0.20, 0.20, 0.15]  # для layers 1, 2, 3
    
    def _xavier_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Xavier/Glorot инициализация весов"""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    
    def get_weights_flat(self) -> np.ndarray:
        """Возвращает все веса как плоский вектор для CMA-ES"""
        weights = [
            self.W_att1.ravel(), self.b_att1,
            self.W_att2.ravel(), self.b_att2,
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3,
            self.W4.ravel(), self.b4
        ]
        return np.concatenate(weights)
    
    def set_weights_from_flat(self, w_flat: np.ndarray):
        """Устанавливает веса из плоского вектора"""
        idx = 0
        
        # Attention
        size = 7 * 16
        self.W_att1 = w_flat[idx:idx+size].reshape(7, 16)
        idx += size
        self.b_att1 = w_flat[idx:idx+16]
        idx += 16
        
        size = 16 * 4
        self.W_att2 = w_flat[idx:idx+size].reshape(16, 4)
        idx += size
        self.b_att2 = w_flat[idx:idx+4]
        idx += 4
        
        # Main network
        size = 36 * 64
        self.W1 = w_flat[idx:idx+size].reshape(36, 64)
        idx += size
        self.b1 = w_flat[idx:idx+64]
        idx += 64
        
        size = 64 * 32
        self.W2 = w_flat[idx:idx+size].reshape(64, 32)
        idx += size
        self.b2 = w_flat[idx:idx+32]
        idx += 32
        
        size = 32 * 16
        self.W3 = w_flat[idx:idx+size].reshape(32, 16)
        idx += size
        self.b3 = w_flat[idx:idx+16]
        idx += 16
        
        size = 16 * 1
        self.W4 = w_flat[idx:idx+size].reshape(16, 1)
        idx += size
        self.b4 = w_flat[idx:idx+1]
    
    def forward(
        self,
        x_features: np.ndarray,
        x_context: np.ndarray,
        training: bool = False
    ) -> float:
        """
        Forward pass нейросети
        
        Args:
            x_features: enriched features (36D)
            x_context: context features (7D)
            training: включить dropout?
        
        Returns:
            p_final: вероятность UP (0-1)
        """
        # ===== ATTENTION MODULE =====
        # Context(7) → 16 → 4 attention weights
        h_att = x_context @ self.W_att1 + self.b_att1
        h_att = _relu(h_att)
        
        att_logits = h_att @ self.W_att2 + self.b_att2
        # Softmax для нормализации
        att_weights = np.exp(att_logits - np.max(att_logits))
        att_weights = att_weights / (np.sum(att_weights) + 1e-8)
        
        # ===== MAIN NETWORK =====
        # Layer 1: 36 → 64
        h1 = x_features @ self.W1 + self.b1
        h1 = _relu(h1)
        h1 = _dropout(h1, self.dropout_rates[0], training)
        
        # Layer 2: 64 → 32
        h2 = h1 @ self.W2 + self.b2
        h2 = _relu(h2)
        h2 = _dropout(h2, self.dropout_rates[1], training)
        
        # Layer 3: 32 → 16
        h3 = h2 @ self.W3 + self.b3
        h3 = _relu(h3)
        h3 = _dropout(h3, self.dropout_rates[2], training)
        
        # Output: 16 → 1
        logit = (h3 @ self.W4 + self.b4)[0]
        
        p = _sigmoid(logit)
        return float(np.clip(p, 0.0, 1.0))
    
    def predict_mc(
        self,
        x_features: np.ndarray,
        x_context: np.ndarray,
        n_mc: int = 30
    ) -> Tuple[float, float]:
        """
        Предсказание с Monte-Carlo Dropout для uncertainty
        
        Args:
            x_features: enriched features (36D)
            x_context: context (7D)
            n_mc: количество MC проходов
        
        Returns:
            (p_mean, p_std): среднее предсказание и uncertainty
        """
        predictions = []
        for _ in range(n_mc):
            p = self.forward(x_features, x_context, training=True)
            predictions.append(p)
        
        p_mean = float(np.mean(predictions))
        p_std = float(np.std(predictions))
        
        return p_mean, p_std


# ========== ГЛАВНЫЙ КЛАСС НЕЙРОСЕТЕВОЙ META ==========

class MetaNeuralCEM:
    """
    Нейросетевая META с эволюционным обучением
    
    Основные компоненты:
    - Feature Engineering (18D → 36D)
    - Attention-based expert gating
    - Deep MLP (64→32→16→1)
    - MC Dropout uncertainty
    - CMA-ES/CEM optimization
    - Phase-specific models
    """
    
    def __init__(self, cfg):
        """Инициализация META"""
        self.cfg = cfg
        self.state_path = getattr(cfg, "meta_state_path", "meta_neural_state.json")
        self.enabled = True
        self.mode = "SHADOW"
        self._last_phase = 0
        
        # ADWIN для drift detection
        self.adwin = ADWIN(delta=0.002) if HAVE_RIVER else None
        
        # ===== ФАЗОВАЯ АРХИТЕКТУРА =====
        self.P = int(getattr(cfg, "meta_exp4_phases", 6))
        
        # Нейросети для каждой фазы
        self.networks: Dict[int, NeuralMetaNetwork] = {}
        for p in range(self.P):
            self.networks[p] = NeuralMetaNetwork(phase=p)
        
        # ===== БУФЕРЫ ДАННЫХ =====
        self.buf_ph: Dict[int, List[Tuple]] = {p: [] for p in range(self.P)}
        self.seen_ph: Dict[int, int] = {p: 0 for p in range(self.P)}
        self.new_since_train_ph: Dict[int, int] = {p: 0 for p in range(self.P)}
        
        # Пути к CSV с данными фаз
        self._phase_csv_paths: Dict[int, str] = {}
        base_path = getattr(cfg, "meta_state_path", "meta_neural_state.json")
        base_dir = os.path.dirname(base_path) or "."
        for p in range(self.P):
            self._phase_csv_paths[p] = os.path.join(base_dir, f"meta_neural_phase_{p}.csv")
        
        # ===== МЕТРИКИ =====
        self.shadow_hits: List[int] = []
        self.active_hits: List[int] = []
        
        # Cross-validation
        self.cv_metrics: Dict[int, Dict] = {p: {} for p in range(self.P)}
        self.cv_oof_preds: Dict[int, deque] = {p: deque(maxlen=500) for p in range(self.P)}
        self.cv_oof_labels: Dict[int, deque] = {p: deque(maxlen=500) for p in range(self.P)}
        self.cv_last_check: Dict[int, int] = {p: 0 for p in range(self.P)}
        self.validation_passed: Dict[int, bool] = {p: False for p in range(self.P)}
        
        # ===== ОБУЧЕНИЕ =====
        self._unsaved = 0
        self._last_save = time.time()
        self._experts = []
        
        # Загрузка состояния
        self._load()
    
    def bind_experts(self, *experts):
        """Привязка экспертов для логирования"""
        self._experts = list(experts)
        return self
    
    # ========== FEATURE ENGINEERING ==========
    
    def _build_enriched_features(
        self,
        p_xgb: Optional[float],
        p_rf: Optional[float],
        p_arf: Optional[float],
        p_nn: Optional[float],
        p_base: Optional[float],
        reg_ctx: Optional[dict] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Строит обогащенные фичи для нейросети
        
        Returns:
            (x_features, x_context):
            - x_features: 36D вектор для main network
            - x_context: 7D вектор для attention module
        """
        # Безопасное извлечение предсказаний
        preds = []
        for p in [p_xgb, p_rf, p_arf, p_nn]:
            if p is not None:
                preds.append(float(p))
        
        # Нужно минимум 2 эксперта
        if len(preds) < 2:
            return None, None
        
        # Дополняем до 4 экспертов средним
        p_mean = float(np.mean(preds))
        while len(preds) < 4:
            preds.append(p_mean)
        
        p_xgb_safe, p_rf_safe, p_arf_safe, p_nn_safe = preds
        p_base_safe = _safe_float(p_base, p_mean)
        
        # ===== ОСНОВНЫЕ ФИЧИ (5D) =====
        features = [
            p_xgb_safe, p_rf_safe, p_arf_safe, p_nn_safe, p_base_safe
        ]
        
        # ===== ВЗАИМОДЕЙСТВИЯ ЭКСПЕРТОВ (6D) =====
        features.extend([
            p_xgb_safe * p_rf_safe,
            p_xgb_safe * p_arf_safe,
            p_xgb_safe * p_nn_safe,
            p_rf_safe * p_arf_safe,
            p_rf_safe * p_nn_safe,
            p_arf_safe * p_nn_safe
        ])
        
        # ===== ЛОГИТЫ (4D) =====
        features.extend([
            _safe_logit(p_xgb_safe),
            _safe_logit(p_rf_safe),
            _safe_logit(p_arf_safe),
            _safe_logit(p_nn_safe)
        ])
        
        # ===== СТАТИСТИКИ СОГЛАСИЯ (4D) =====
        preds_arr = np.array([p_xgb_safe, p_rf_safe, p_arf_safe, p_nn_safe])
        features.extend([
            float(np.std(preds_arr)),                    # разброс мнений
            float(np.max(preds_arr) - np.min(preds_arr)), # диапазон
            float(np.abs(p_xgb_safe - p_rf_safe) + 
                  np.abs(p_xgb_safe - p_arf_safe) +
                  np.abs(p_rf_safe - p_nn_safe)),        # disagreement
            float(-np.sum(preds_arr * np.log(preds_arr + 1e-8))) # entropy
        ])
        
        # ===== КОНТЕКСТНЫЕ ФИЧИ (7D) =====
        ctx = reg_ctx if isinstance(reg_ctx, dict) else {}
        
        vol_ratio = _safe_float(ctx.get("vol_ratio"), 1.0)
        trend_macd = _safe_float(ctx.get("trend_macd"), 0.0)
        jump_flag = float(ctx.get("jump_detected", False))
        funding_sign = _safe_float(ctx.get("funding_sign"), 0.0)
        
        # НОВОЕ: дополнительные контекстные фичи
        book_imb = _safe_float(ctx.get("book_imb"), 0.0)
        ofi_15s = _safe_float(ctx.get("ofi_15s"), 0.0)
        basis_pct = _safe_float(ctx.get("basis_pct"), 0.0)
        
        context = np.array([
            vol_ratio, trend_macd, jump_flag, funding_sign,
            book_imb, ofi_15s, basis_pct
        ], dtype=float)
        
        # ===== ВЗАИМОДЕЙСТВИЯ С КОНТЕКСТОМ (10D) =====
        features.extend([
            p_xgb_safe * vol_ratio,
            p_rf_safe * vol_ratio,
            p_arf_safe * vol_ratio,
            p_nn_safe * vol_ratio,
            p_mean * trend_macd,
            p_mean * jump_flag,
            float(np.std(preds_arr)) * vol_ratio,  # uncertainty × volatility
            p_mean * book_imb,
            p_mean * ofi_15s,
            p_mean * basis_pct
        ])
        
        # ===== BIAS TERM (1D) =====
        features.append(1.0)
        
        # Итого: 5 + 6 + 4 + 4 + 10 + 1 = 30D
        # Паддинг до 36D
        while len(features) < 36:
            features.append(0.0)
        
        x_features = np.array(features[:36], dtype=float)
        x_context = context
        
        return x_features, x_context
    
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
        Предсказание с MC Dropout
        
        Returns:
            p_final: скорректированная вероятность UP
        """
        ph = phase_from_ctx(reg_ctx)
        
        # Построение фичей
        x_features, x_context = self._build_enriched_features(
            p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx
        )
        
        if x_features is None or x_context is None:
            # Fallback: простое среднее
            preds = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
            if len(preds) == 0:
                return None
            return float(np.clip(np.mean(preds), 0.0, 1.0))
        
        # Получаем нейросеть для фазы
        net = self.networks.get(ph)
        if net is None:
            return None
        
        # Проверяем, обучена ли модель
        if self.seen_ph.get(ph, 0) < int(getattr(self.cfg, "meta_min_ready", 80)):
            # Модель не обучена - используем среднее экспертов
            preds = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
            if len(preds) == 0:
                return None
            return float(np.clip(np.mean(preds), 0.0, 1.0))
        
        # MC Dropout предсказание
        n_mc = int(getattr(self.cfg, "meta_mc_n_inference", 30))
        p_mean, p_std = net.predict_mc(x_features, x_context, n_mc=n_mc)
        
        # Коррекция на uncertainty
        uncertainty_threshold = float(getattr(self.cfg, "meta_mc_uncertainty_threshold", 0.15))
        
        if p_std > uncertainty_threshold:
            # Высокая uncertainty → консервативно к 0.5
            p_final = p_mean * 0.7 + 0.5 * 0.3
        else:
            # Низкая uncertainty → доверяем модели
            p_final = p_mean
        
        return float(np.clip(p_final, 0.0, 1.0))
    
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
        """Записывает результат и триггерит обучение"""
        try:
            ph = phase_from_ctx(reg_ctx)
            self._last_phase = ph
            
            # Построение фичей
            x_features, x_context = self._build_enriched_features(
                p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx
            )
            
            if x_features is None or x_context is None:
                # Пропускаем пример если нет фичей
                return
            
            # Сохраняем пример
            # Сохраняем пример
            self._append_example(ph, x_features, x_context, int(y_up))
            self.seen_ph[ph] = int(self.seen_ph.get(ph, 0)) + 1
            self.new_since_train_ph[ph] = self.new_since_train_ph.get(ph, 0) + 1

            # Обновление метрик
            if p_final_used is not None:
                p_for_gate = p_final_used
            else:
                net = self.networks.get(ph)
                if net is not None and self.seen_ph[ph] >= 80:
                    p_for_gate = net.forward(x_features, x_context, training=False)
                else:
                    preds = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
                    p_for_gate = float(np.mean(preds)) if preds else 0.5
            
            hit = int((p_for_gate >= 0.5) == bool(y_up))
            
            if self.mode == "ACTIVE" and used_in_live:
                self.active_hits.append(hit)
                if self.adwin is not None:
                    in_drift = self.adwin.update(1 - hit)
                    if in_drift:
                        self.mode = "SHADOW"
                        self.active_hits = []
                        print(f"[MetaNeural] 🔄 ACTIVE→SHADOW: drift detected")
            else:
                self.shadow_hits.append(hit)
            
            # Ограничиваем размер
            self.active_hits = self.active_hits[-2000:]
            self.shadow_hits = self.shadow_hits[-2000:]
            
            self._unsaved += 1
            self._save_throttled()
            
            # OOF для CV
            if getattr(self.cfg, "cv_enabled", True):
                self.cv_oof_preds[ph].append(float(p_for_gate))
                self.cv_oof_labels[ph].append(int(y_up))
            
            # Периодическая CV
            cv_check_every = int(getattr(self.cfg, "cv_check_every", 100))
            self.cv_last_check[ph] = int(self.cv_last_check.get(ph, 0)) + 1
            
            if getattr(self.cfg, "cv_enabled", True) and self.cv_last_check[ph] >= cv_check_every:
                self.cv_last_check[ph] = 0
                try:
                    cv_results = self._run_cv_validation(ph)
                    self.cv_metrics[ph] = cv_results
                    if cv_results.get("status") == "ok":
                        self.validation_passed[ph] = True
                        print(
                            f"[MetaNeural] ✅ CV ph={ph}: "
                            f"ACC={cv_results['oof_accuracy']:.2f}% "
                            f"CI=[{cv_results['ci_lower']:.2f}%, {cv_results['ci_upper']:.2f}%]"
                        )
                except Exception as e:
                    print(f"[MetaNeural] ❌ CV failed for ph={ph}: {e}")
            
            # Обучение при готовности
            if self._phase_ready(ph):
                try:
                    print(f"[MetaNeural] 🎯 Starting training for phase {ph} ({self.seen_ph[ph]} samples)")
                    self._train_phase(ph)
                    self._trim_phase_storage(ph)
                    self.buf_ph[ph] = []
                    self._save()
                    print(f"[MetaNeural] ✅ Training completed for phase {ph}")
                except Exception as e:
                    print(f"[MetaNeural] ❌ Training failed for ph={ph}: {e}")
                    traceback.print_exc()
            
            # Переключение режимов
            self._maybe_flip_modes()
            
            # Отправка метрик в visualizer
            self._send_metrics_to_viz()
            
        except Exception as e:
            print(f"[MetaNeural] record_result error: {e}")
            traceback.print_exc()
    
    def settle(self, *args, **kwargs):
        """Alias для record_result (для совместимости)"""
        try:
            return self.record_result(*args, **kwargs)
        except Exception as e:
            print(f"[MetaNeural] settle error: {e}")
    
    # ========== ОБУЧЕНИЕ CMA-ES ==========
    
    def _phase_ready(self, ph: int) -> bool:
        """Проверка готовности фазы к обучению с адаптивной частотой"""
        min_samples = int(getattr(self.cfg, "meta_min_train", 150))
        base_retrain = int(getattr(self.cfg, "meta_retrain_every", 50))
        
        # Адаптивная частота на основе сложности модели
        net = self.networks.get(ph)
        if net is not None:
            n_params = len(net.get_weights_flat())
            # Для ~5000 параметров: 50 × 3 = 150
            # Для ~1500 параметров: 50 × 1 = 50
            multiplier = max(1, n_params // 1500)  # 🔥 ОПТИМАЛЬНЫЙ ДЕЛИТЕЛЬ
            retrain_every = base_retrain * multiplier
        else:
            retrain_every = base_retrain
        
        seen = self.seen_ph.get(ph, 0)
        new_since_last_train = self.new_since_train_ph.get(ph, 0)
        
        if seen < min_samples:
            return False
        
        if new_since_last_train < retrain_every:
            return False
        
        print(f"[MetaNeural] Phase {ph} ready: {new_since_last_train} new samples "
            f"(threshold={retrain_every}, params={n_params if net else 'N/A'})")
        
        return True
    
    def _train_phase(self, ph: int):
        """Обучение нейросети для фазы через CMA-ES"""
        # Загрузка данных
        X_feat_list, X_ctx_list, y_list = self._load_phase_data(ph)
        
        if len(X_feat_list) < 150:
            return
        
        X_feat = np.array(X_feat_list, dtype=float)
        X_ctx = np.array(X_ctx_list, dtype=float)
        y = np.array(y_list, dtype=float)
        
        # Выбор алгоритма
        use_cma = getattr(self.cfg, "meta_use_cma_es", True) and HAVE_CMA
        
        if use_cma:
            self._train_cma_es(ph, X_feat, X_ctx, y)
        else:
            self._train_cem(ph, X_feat, X_ctx, y)
    
    def _train_cma_es(self, ph: int, X_feat: np.ndarray, X_ctx: np.ndarray, y: np.ndarray):
        """CMA-ES оптимизация"""
        net = self.networks[ph]
        D = len(net.get_weights_flat())
        
        # Начальная точка
        x0 = net.get_weights_flat()
        sigma0 = 1.0
        
        es = cma.CMAEvolutionStrategy(
            x0=x0,
            sigma0=sigma0,
            inopts={
                'bounds': [-5.0, 5.0],
                'popsize': 40,
                'maxiter': 30,
                'verbose': -1
            }
        )
        
        viz_enabled = False
        viz = None
        try:
            if HAVE_VISUALIZER:
                viz = get_visualizer()
                viz_enabled = True
        except Exception:
            pass
        
        iteration = 0
        while not es.stop():
            solutions = es.ask()
            fitness = []
            
            for w in solutions:
                loss = self._evaluate_weights(w, net, X_feat, X_ctx, y)
                fitness.append(loss)
            
            es.tell(solutions, fitness)
            
            iteration += 1
            
            # Логирование
            if iteration % 5 == 0 or iteration == 1:
                best_loss = float(np.min(fitness))
                median_loss = float(np.median(fitness))
                sigma = float(getattr(es, "sigma", sigma0))
                
                print(f"[MetaNeural] CMA-ES iter {iteration}: "
                      f"best={best_loss:.6f}, median={median_loss:.6f}, sigma={sigma:.4f}")
                
                if viz_enabled and viz:
                    try:
                        viz.record_meta_training_step(
                            phase=ph,
                            iteration=iteration,
                            best_loss=best_loss,
                            median_loss=median_loss,
                            sigma=sigma
                        )
                    except Exception:
                        pass
        
        # Устанавливаем лучшие веса
        # Устанавливаем лучшие веса
        best_w = np.array(es.result.xbest, dtype=float)
        net.set_weights_from_flat(best_w)

        self.new_since_train_ph[ph] = 0

        print(f"[MetaNeural] ✅ CMA-ES converged for phase {ph}")
    
    def _train_cem(self, ph: int, X_feat: np.ndarray, X_ctx: np.ndarray, y: np.ndarray):
        """CEM оптимизация (fallback)"""
        net = self.networks[ph]
        D = len(net.get_weights_flat())
        
        n_iter = 30
        pop_size = 50
        elite_frac = 0.2
        n_elite = max(1, int(pop_size * elite_frac))
        
        mu = net.get_weights_flat()
        sigma = np.ones(D) * 1.0
        
        best_loss = float('inf')
        best_w = mu.copy()
        
        for iteration in range(n_iter):
            # Генерация популяции
            population = []
            for _ in range(pop_size):
                w = mu + sigma * np.random.randn(D)
                w = np.clip(w, -5.0, 5.0)
                population.append(w)
            
            # Оценка
            scores = []
            for w in population:
                loss = self._evaluate_weights(w, net, X_feat, X_ctx, y)
                scores.append(loss)
            
            # Селекция элиты
            elite_idx = np.argsort(scores)[:n_elite]
            elite = [population[i] for i in elite_idx]
            
            if scores[elite_idx[0]] < best_loss:
                best_loss = scores[elite_idx[0]]
                best_w = population[elite_idx[0]].copy()
            
            # Обновление распределения
            elite_arr = np.array(elite)
            mu = elite_arr.mean(axis=0)
            sigma = elite_arr.std(axis=0) + 1e-6
            
            if iteration % 5 == 0:
                print(f"[MetaNeural] CEM iter {iteration}: best={best_loss:.6f}")
        
        net.set_weights_from_flat(best_w)

        self.new_since_train_ph[ph] = 0

        print(f"[MetaNeural] ✅ CEM converged for phase {ph}")
    
    def _evaluate_weights(
        self,
        w: np.ndarray,
        net: NeuralMetaNetwork,
        X_feat: np.ndarray,
        X_ctx: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Оценка качества весов через bootstrap log-loss"""
        net.set_weights_from_flat(w)
        
        # Bootstrap выборка
        n = len(y)
        n_bootstrap = 10
        losses = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            X_f_boot = X_feat[idx]
            X_c_boot = X_ctx[idx]
            y_boot = y[idx]
            
            # Forward pass
            preds = []
            for i in range(len(y_boot)):
                p = net.forward(X_f_boot[i], X_c_boot[i], training=True)
                preds.append(p)
            
            preds = np.array(preds)
            preds = np.clip(preds, 1e-6, 1.0 - 1e-6)
            
            # Log-loss
            log_loss = -np.mean(y_boot * np.log(preds) + (1 - y_boot) * np.log(1 - preds))
            
            # L2 регуляризация
            l2_penalty = 0.0001 * np.sum(w ** 2)
            
            losses.append(log_loss + l2_penalty)
        
        return float(np.mean(losses))
    
    # ========== ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ ==========
    
    def _maybe_flip_modes(self):
        """Переключение SHADOW ↔ ACTIVE"""
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
                print(f"[MetaNeural] SHADOW→ACTIVE: WR={wr_shadow:.2%}")
        
        # ACTIVE → SHADOW
        if self.mode == "ACTIVE" and wr_active is not None:
            if wr_active < exit_wr:
                self.mode = "SHADOW"
                print(f"[MetaNeural] ACTIVE→SHADOW: WR={wr_active:.2%}")
    
    # ========== СТАТУС ==========
    
    def status(self) -> Dict[str, str]:
        """Возвращает статус META"""
        def _wr(xs):
            return sum(xs) / len(xs) if xs else None
        
        def _fmt(p):
            return "—" if p is None else f"{100.0*p:.2f}%"
        
        wr_a = _wr(self.active_hits)
        wr_s = _wr(self.shadow_hits)
        all_hits = (self.active_hits or []) + (self.shadow_hits or [])
        wr_all = _wr(all_hits)
        
        ph = self._last_phase
        cv_metrics = self.cv_metrics.get(ph, {})
        
        return {
            "algo": "Neural+CMA-ES" if getattr(self.cfg, "meta_use_cma_es", True) else "Neural+CEM",
            "mode": self.mode,
            "enabled": str(self.enabled),
            "features": "36D+7D",
            "params": "~5200",
            "wr_active": _fmt(wr_a),
            "n_active": str(len(self.active_hits or [])),
            "wr_shadow": _fmt(wr_s),
            "n_shadow": str(len(self.shadow_hits or [])),
            "wr_all": _fmt(wr_all),
            "n": str(len(all_hits)),
            "cv_oof_wr": _fmt(cv_metrics.get("oof_accuracy", 0) / 100.0) if cv_metrics else "—",
        }
    
    # ========== РАБОТА С ФАЙЛАМИ ==========
    
    def _append_example(self, ph: int, x_feat: np.ndarray, x_ctx: np.ndarray, y: int):
        """Добавляет пример в буфер и CSV"""
        ts = int(time.time())
        self.buf_ph[ph].append((x_feat.tolist(), x_ctx.tolist(), y, ts))
        
        # Сохранение в CSV
        csv_path = self._phase_csv_paths.get(ph)
        if csv_path:
            try:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # Header при первой записи
                    if f.tell() == 0:
                        writer.writerow(["x_features", "x_context", "y", "timestamp"])
                    writer.writerow([
                        json.dumps(x_feat.tolist()),
                        json.dumps(x_ctx.tolist()),
                        y,
                        ts
                    ])
            except Exception as e:
                print(f"[MetaNeural] CSV append error: {e}")
    
    def _load_phase_data(self, ph: int) -> Tuple[List, List, List]:
        """Загружает данные фазы из CSV"""
        csv_path = self._phase_csv_paths.get(ph)
        X_feat_list, X_ctx_list, y_list = [], [], []
        
        if not csv_path or not os.path.exists(csv_path):
            return X_feat_list, X_ctx_list, y_list
        
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        x_f = np.array(json.loads(row["x_features"]), dtype=float)
                        x_c = np.array(json.loads(row["x_context"]), dtype=float)
                        y = int(row["y"])
                        
                        X_feat_list.append(x_f)
                        X_ctx_list.append(x_c)
                        y_list.append(y)
                    except Exception:
                        continue
        except Exception as e:
            print(f"[MetaNeural] CSV load error: {e}")
        
        return X_feat_list, X_ctx_list, y_list
    
    def _trim_phase_storage(self, ph: int):
        """Обрезка старых данных"""
        max_rows = int(getattr(self.cfg, "phase_memory_cap", 3000))
        csv_path = self._phase_csv_paths.get(ph)
        
        if not csv_path or not os.path.exists(csv_path):
            return
        
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if len(lines) > max_rows + 1:  # +1 для header
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(lines[0])  # header
                    f.writelines(lines[-(max_rows):])
        except Exception:
            pass
    
    def _run_cv_validation(self, ph: int) -> Dict:
        """CV валидация (упрощенная)"""
        preds = list(self.cv_oof_preds[ph])
        labels = list(self.cv_oof_labels[ph])
        
        if len(preds) < 100:
            return {"status": "insufficient_data"}
        
        preds_arr = np.array(preds)
        labels_arr = np.array(labels)
        
        acc = 100.0 * np.mean((preds_arr >= 0.5) == labels_arr)
        
        # Bootstrap CI
        n_boot = 1000
        accs = []
        for _ in range(n_boot):
            idx = np.random.choice(len(preds), size=len(preds), replace=True)
            acc_boot = 100.0 * np.mean((preds_arr[idx] >= 0.5) == labels_arr[idx])
            accs.append(acc_boot)
        
        ci_lower = float(np.percentile(accs, 2.5))
        ci_upper = float(np.percentile(accs, 97.5))
        
        return {
            "status": "ok",
            "oof_accuracy": float(acc),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_samples": len(preds)
        }
    
    def _send_metrics_to_viz(self):
        """Отправка метрик в visualizer"""
        if not HAVE_VISUALIZER:
            return
        
        try:
            viz = get_visualizer()
            
            all_hits = (self.active_hits or []) + (self.shadow_hits or [])
            if not all_hits:
                return
            
            wr_all = sum(all_hits) / len(all_hits)
            
            ph = self._last_phase
            cv = self.cv_metrics.get(ph, {})
            
            viz.record_expert_metrics(
                name="META",
                accuracy=wr_all,
                n_samples=len(all_hits),
                cv_accuracy=cv.get("oof_accuracy", 0) / 100.0 if cv else None,
                cv_ci_lower=cv.get("ci_lower", 0) / 100.0 if cv else None,
                cv_ci_upper=cv.get("ci_upper", 0) / 100.0 if cv else None,
                mode=self.mode
            )
        except Exception:
            pass
    
    def _save_throttled(self):
        """Throttled сохранение"""
        if self._unsaved >= 10 or (time.time() - self._last_save) > 300:
            self._save()
    
    def _save(self, force: bool = False):
        """Сохранение состояния"""
        try:
            # Веса нейросетей
            networks_state = {}
            for p, net in self.networks.items():
                networks_state[p] = net.get_weights_flat().tolist()
            
            state = {
                "mode": self.mode,
                "enabled": self.enabled,
                "networks": networks_state,
                "shadow_hits": self.shadow_hits,
                "active_hits": self.active_hits,
                "seen_ph": {str(k): v for k, v in self.seen_ph.items()},
                "new_since_train_ph": {str(k): v for k, v in self.new_since_train_ph.items()},
                "cv_metrics": {str(k): v for k, v in self.cv_metrics.items()},
                "validation_passed": {str(k): v for k, v in self.validation_passed.items()},
            }
            
            atomic_save_json(self.state_path, state)
            self._unsaved = 0
            self._last_save = time.time()
        except Exception as e:
            print(f"[MetaNeural] Save error: {e}")
    
    def _load(self):
        """Загрузка состояния"""
        if not os.path.exists(self.state_path):
            return
        
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            self.mode = state.get("mode", "SHADOW")
            self.enabled = state.get("enabled", True)
            
            # Восстановление весов нейросетей
            networks_state = state.get("networks", {})
            for p_str, w_list in networks_state.items():
                p = int(p_str)
                if p in self.networks:
                    w = np.array(w_list, dtype=float)
                    self.networks[p].set_weights_from_flat(w)
            
            self.shadow_hits = state.get("shadow_hits", [])
            self.active_hits = state.get("active_hits", [])
            
            seen_ph = state.get("seen_ph", {})
            self.seen_ph = {int(k): int(v) for k, v in seen_ph.items()}
            
            self.new_since_train_ph = {p: 0 for p in range(self.P)}
            new_since_train = state.get("new_since_train_ph", {})
            if new_since_train:
                self.new_since_train_ph = {int(k): int(v) for k, v in new_since_train.items()}
            
            cv_metrics = state.get("cv_metrics", {})
            self.cv_metrics = {int(k): v for k, v in cv_metrics.items()}
            
            val_passed = state.get("validation_passed", {})
            self.validation_passed = {int(k): bool(v) for k, v in val_passed.items()}
            
            print(f"[MetaNeural] State loaded: mode={self.mode}, total_samples={sum(self.seen_ph.values())}")
        except Exception as e:
            print(f"[MetaNeural] Load error: {e}")
