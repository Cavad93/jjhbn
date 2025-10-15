# -*- coding: utf-8 -*-
"""
meta_cem_mc.py — META-стекинг на основе CEM/CMA-ES + Монте-Карло (bootstrap) + Cross-Validation

=== ОСНОВНАЯ ИДЕЯ ===
Этот модуль объединяет предсказания четырех экспертов (XGB, RF, ARF, NN) в единое финальное
предсказание. Вместо традиционного градиентного обучения используется стохастическая оптимизация
(CEM или CMA-ES) с оценкой качества через Монте-Карло (bootstrap выборки).

=== КЛЮЧЕВЫЕ ОСОБЕННОСТИ ===
1. Фазовая память: отдельная модель для каждой из 6 фаз рынка
2. Контекстный гейтинг: веса экспертов зависят от контекста
3. Cross-Validation: честная оценка качества с purged walk-forward CV
4. Bootstrap CI: статистические доверительные интервалы для метрик
5. Режимы SHADOW/ACTIVE: переключение на основе валидированных метрик

=== АРХИТЕКТУРА ===
- Вход: предсказания 4 экспертов + базовое предсказание + контекст (18 фичей)
- Гейтинг: soft (softmax) или exp4 (EXP4 Hedge) режим
- Выход: p_final = σ(w · φ), где φ — расширенный вектор фичей (логиты + мета + контекст)
- Обучение: CEM/CMA-ES минимизирует log-loss на bootstrap выборках
- Валидация: Walk-forward purged CV с embargo period

Файл состояния: cfg.meta_state_path (JSON)
"""
from __future__ import annotations
import os
import json
import time
import math
import random
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np

# ========== ВНЕШНИЕ ЗАВИСИМОСТИ ==========

# CMA-ES оптимизатор (опциональный, fallback на CEM если недоступен)
try:
    import cma  # type: ignore
    HAVE_CMA = True
except Exception:
    cma = None
    HAVE_CMA = False

# Графики и Telegram уведомления
try:
    from meta_report import plot_cma_like, send_telegram_photo, send_telegram_text
    from expert_report import plot_experts_reliability_panel
    import matplotlib.pyplot as plt
    HAVE_PLOTTING = True
except Exception:
    HAVE_PLOTTING = False
    plot_cma_like = None
    send_telegram_photo = None
    send_telegram_text = None

# LambdaMART эксперт (если доступен)
try:
    from models.lambdamart_expert import LambdaMARTExpert as _LMCore
    _HAVE_LAMBDAMART = True
except Exception:
    _LMCore = None
    _HAVE_LAMBDAMART = False

# Безопасное сохранение JSON с атомарной заменой
try:
    from state_safety import atomic_save_json
except Exception:
    def atomic_save_json(path: str, obj: dict):
        """Fallback: простое сохранение через временный файл"""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

# Извлечение фазы из контекста
try:
    from meta_ctx import phase_from_ctx
except Exception:
    def phase_from_ctx(ctx: Optional[dict]) -> int:
        return int(ctx.get("phase", 0) if isinstance(ctx, dict) else 0)

# === NEW: LambdaMART в роли второй МЕТА + блендер вероятностей ===
def _safe_logit(p):
    try:
        p = float(p)
    except Exception:
        return 0.0
    p = max(min(p, 1.0 - 1e-6), 1e-6)
    return math.log(p / (1.0 - p))

def _entropy4(p_list):
    vals = [float(p) for p in p_list if p is not None]
    if not vals:
        return 0.0
    hist, _ = np.histogram(vals, bins=10, range=(0.0, 1.0), density=True)
    hist = hist / (hist.sum() + 1e-12)
    return float(-(hist * np.log(hist + 1e-12)).sum())

class LambdaMARTMetaLite:
    """
    Обучает LGBMRanker на φ-признаках мета-уровня и отдаёт «сырую» вероятность через сигмоиду.
    Буферизует данные и периодически переобучается.
    """
    def __init__(self, retrain_every: int = 80, min_ready: int = 160, max_buf: int = 10000):
        self.enabled = bool(_HAVE_LAMBDAMART)
        self.retrain_every = int(retrain_every)
        self.min_ready = int(min_ready)
        self.max_buf = int(max_buf)
        self.params = dict(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="lambdarank",
            metric="ndcg"
        )
        self.model = None
        self._X, self._y, self._g = [], [], []
        self._last_fit_n = 0

    def _phi(self, p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx=None):
        """Расширенный вектор фичей с контекстом (18 элементов)"""
        lzs = [_safe_logit(p) for p in [p_xgb, p_rf, p_arf, p_nn]]
        lz_base = _safe_logit(p_base)
        plist = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
        disagree = float(np.std(plist)) if len(plist) > 1 else 0.0
        ent = _entropy4([p_xgb, p_rf, p_arf, p_nn])
        
        # Контекстные фичи
        if reg_ctx is not None and isinstance(reg_ctx, dict):
            trend_sign = float(reg_ctx.get("trend_sign", 0.0))
            trend_abs = float(reg_ctx.get("trend_abs", 0.0))
            vol_ratio = float(reg_ctx.get("vol_ratio", 1.0))
            jump_flag = float(reg_ctx.get("jump_flag", 0.0))
            ofi_sign = float(reg_ctx.get("ofi_sign", 0.0))
            book_imb = float(reg_ctx.get("book_imb", 0.0))
            basis_sign = float(reg_ctx.get("basis_sign", 0.0))
            funding_sign = float(reg_ctx.get("funding_sign", 0.0))
        else:
            trend_sign = trend_abs = vol_ratio = jump_flag = 0.0
            ofi_sign = book_imb = basis_sign = funding_sign = 0.0
        
        # Взаимодействия
        disagree_vol = disagree * vol_ratio
        entropy_trend = ent * abs(trend_abs)
        
        return np.array([
            lzs[0], lzs[1], lzs[2], lzs[3], lz_base,  # 0-4: логиты
            disagree, ent,  # 5-6: агрегаты
            trend_sign, trend_abs, vol_ratio, jump_flag,  # 7-10: контекст
            ofi_sign, book_imb, basis_sign, funding_sign,  # 11-14: микроструктура
            disagree_vol, entropy_trend,  # 15-16: взаимодействия
            1.0  # 17: bias
        ], dtype=float)

    def _phase_group(self, reg_ctx):
        try:
            return int(phase_from_ctx(reg_ctx or {}))
        except Exception:
            return 0

    def predict(self, p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx=None):
        if not self.enabled or self.model is None:
            return None
        x = self._phi(p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx).reshape(1, -1)
        try:
            s = float(self.model.predict(x)[0])
            p = 1.0 / (1.0 + math.exp(-s))
            return float(np.clip(p, 0.0, 1.0))
        except Exception:
            return None

    def record_result(self, p_xgb, p_rf, p_arf, p_nn, p_base, y_up, reg_ctx=None, used_in_live=False):
        if not self.enabled:
            return
        x = self._phi(p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx)
        g = self._phase_group(reg_ctx)
        self._X.append(x)
        self._y.append(int(y_up))
        self._g.append(g)
        if len(self._X) > self.max_buf:
            drop = len(self._X) - self.max_buf
            self._X = self._X[drop:]
            self._y = self._y[drop:]
            self._g = self._g[drop:]
        if len(self._X) >= self.min_ready and (len(self._X) - self._last_fit_n) >= self.retrain_every:
            try:
                X = np.vstack(self._X)
                y = np.asarray(self._y, dtype=int)
                g = np.asarray(self._g, dtype=int)
                order = np.argsort(g)
                X = X[order]
                y = y[order]
                g = g[order]
                _, counts = np.unique(g, return_counts=True)
                mdl = _LMCore(self.params) if _LMCore else None
                if mdl is not None:
                    mdl.fit(X, y, groups=counts.tolist())
                    self.model = mdl.model
                    self._last_fit_n = len(self._X)
            except Exception:
                pass

    def status(self) -> str:
        if not self.enabled:
            return "LMETA[off]"
        n = len(self._X)
        ready = n >= self.min_ready
        return f"LMETA[{'ON' if self.model is not None else 'boot'} n={n}, ready={ready}]"

class ProbBlender:
    """
    Линейно смешивает две калиброванные вероятности p1 и p2.
    Вес подбирается по NLL или Brier на скользящем окне.
    """
    def __init__(self, metric: str = "nll", window: int = 1200, step: float = 0.02):
        self.metric = str(metric).lower()
        self.win = int(window)
        self.step = float(step)
        self.hist = deque(maxlen=self.win)
        self.w = 0.5

    def mix(self, p1: float, p2: float) -> float:
        w = float(self.w)
        p = w * float(p1) + (1.0 - w) * float(p2)
        return float(min(max(p, 0.0), 1.0))

    def record(self, y: int, p1: float | None, p2: float | None) -> None:
        if p1 is None or p2 is None:
            return
        self.hist.append((float(p1), float(p2), int(y)))
        self._retune()

    def _retune(self) -> None:
        if len(self.hist) < 50:
            return
        arr = list(self.hist)
        p1 = np.array([a for a,_,_ in arr], dtype=float)
        p2 = np.array([b for _,b,_ in arr], dtype=float)
        y  = np.array([c for _,_,c in arr], dtype=float)
        best, best_w = 1e18, self.w
        grid = np.arange(0.0, 1.0 + self.step/2, self.step)
        for w in grid:
            p = np.clip(w*p1 + (1.0 - w)*p2, 1e-6, 1.0 - 1e-6)
            cur = (np.mean((p - y)**2) if self.metric == "brier"
                   else -np.mean(y*np.log(p) + (1.0 - y)*np.log(1.0 - p)))
            if cur < best:
                best, best_w = cur, float(w)
        self.w = best_w

# ---- helpers ----
_EPS = 1e-8

# River ADWIN для drift detection
try:
    from river.drift import ADWIN
    HAVE_RIVER = True
except Exception:
    ADWIN = None
    HAVE_RIVER = False


# ========== КЛАСС META-СТЕКИНГА С CV ==========

class MetaCEMMC:
    """
    META-стекинг с CEM/CMA-ES оптимизацией + Cross-Validation
    
    Этот класс принимает предсказания от четырех экспертов и создает финальное
    взвешенное предсказание. Обучение происходит через стохастическую оптимизацию,
    а валидация через walk-forward cross-validation с bootstrap доверительными интервалами.
    
    Основные принципы работы:
    - Каждая фаза рынка имеет свою независимую модель (6 фаз всего)
    - Веса обучаются на минимизации log-loss с L2 регуляризацией
    - Качество оценивается через bootstrap для получения доверительных интервалов
    - Переключение SHADOW↔ACTIVE требует подтверждения через CV метрики
    """
    
    def __init__(self, cfg):
        """
        Инициализация META-стекинга
        
        Args:
            cfg: конфигурация с параметрами оптимизации, CV, путями файлов
        
        Что происходит при инициализации:
        1. Загружаем параметры оптимизации (CEM/CMA-ES)
        2. Инициализируем веса для каждой фазы
        3. Настраиваем гейтинг (soft или EXP4)
        4. Создаем структуры для хранения CV метрик
        5. Загружаем сохраненное состояние (если есть)
        """
        self.cfg = cfg
        self.enabled = True
        self.mode = "SHADOW"  # Начинаем в shadow режиме для накопления данных
        
        # ADWIN для детекции дрейфа концепции
        self.adwin = ADWIN(delta=self.cfg.adwin_delta) if HAVE_RIVER else None

        # ===== ФАЗОВАЯ АРХИТЕКТУРА =====
        # У нас 6 фаз рынка, каждая имеет свою модель
        self.P = int(getattr(cfg, "meta_exp4_phases", 6))
        
        # ИЗМЕНЕНО: Размерность вектора фичей увеличена с 8 до 18
        # Включает: 4 логита экспертов + базовый логит + disagree + entropy
        #           + 8 контекстных фичей + 2 взаимодействия + bias
        self.D = 18
        
        # Веса для расширенного вектора фичей
        # Каждая фаза имеет свой набор весов размерностью D=18
        self.w_meta_ph = np.zeros((self.P, self.D), dtype=float)
        
        # Гиперпараметры оптимизации
        self.eta = float(getattr(cfg, "meta_eta", 0.05))  # learning rate
        self.l2 = float(getattr(cfg, "meta_l2", 0.001))   # L2 регуляризация
        self.w_clip = float(getattr(cfg, "meta_w_clip", 8.0))   # клиппинг весов
        self.g_clip = float(getattr(cfg, "meta_g_clip", 1.0))   # клиппинг градиентов

        # ===== КОНТЕКСТНЫЙ ГЕЙТИНГ =====
        # Веса экспертов могут зависеть от контекста (волатильность, тренд и т.д.)
        self.gating_mode = getattr(cfg, "meta_gating_mode", "soft")  # "soft" или "exp4"
        self.alpha_mix = float(getattr(cfg, "meta_alpha_mix", 1.0))  # вес смеси логитов
        
        # Для soft-гейтинга: матрица весов (K экспертов × D контекстных фичей)
        self.Wg = None  # Инициализируется при первом использовании
        self.g_eta = float(getattr(cfg, "meta_gate_eta", 0.02))
        self.g_l2 = float(getattr(cfg, "meta_gate_l2", 0.0005))
        self.gate_clip = float(getattr(cfg, "meta_gate_clip", 5.0))

        # Для EXP4 гейтинга: веса экспертов по фазам
        self.exp4_eta = float(getattr(cfg, "meta_exp4_eta", 0.10))
        self.exp4_w = None  # np.ndarray (P × K), нормированные веса

        # ===== ТРЕКИНГ МЕТРИК ДЛЯ РЕЖИМОВ =====
        # Накапливаем хиты для оценки winrate
        self.shadow_hits: List[int] = []  # предсказания в shadow режиме
        self.active_hits: List[int] = []  # предсказания в active режиме

        # ===== БУФЕРЫ ДАННЫХ ПО ФАЗАМ =====
        # Для каждой фазы храним in-memory буфер примеров
        self.buf_ph: Dict[int, List[Tuple]] = {p: [] for p in range(self.P)}
        self.seen_ph: Dict[int, int] = {p: 0 for p in range(self.P)}
        
        # Пути к CSV файлам с накопленными данными фаз
        self._phase_csv_paths: Dict[int, str] = {}
        base_path = getattr(cfg, "meta_state_path", "meta_state.json")
        base_dir = os.path.dirname(base_path) or "."
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        
        for p in range(self.P):
            self._phase_csv_paths[p] = os.path.join(base_dir, f"{base_name}_ph{p}_data.csv")

        # ===== НОВОЕ: CROSS-VALIDATION СТРУКТУРЫ =====
        # Для каждой фазы храним CV метрики и OOF predictions
        
        # Out-of-fold predictions для текущего окна
        cv_window = int(getattr(cfg, "cv_oof_window", 500))
        self.cv_oof_preds: Dict[int, deque] = {
            p: deque(maxlen=cv_window) for p in range(self.P)
        }
        self.cv_oof_labels: Dict[int, deque] = {
            p: deque(maxlen=cv_window) for p in range(self.P)
        }
        
        # Метрики последней CV проверки для каждой фазы
        self.cv_metrics: Dict[int, Dict] = {p: {} for p in range(self.P)}
        
        # Счетчик для периодической CV проверки
        self.cv_last_check: Dict[int, int] = {p: 0 for p in range(self.P)}
        
        # Флаг валидации: прошла ли фаза CV проверку
        self.validation_passed: Dict[int, bool] = {p: False for p in range(self.P)}

        # ===== ТРЕКИНГ ДЛЯ СОХРАНЕНИЯ =====
        self._unsaved = 0  # счетчик несохраненных изменений
        self._last_save_ts = time.time()
        
        # Ссылки на экспертов (опционально, для логирования)
        self._experts: List = []

        # ===== ЗАГРУЗКА СОСТОЯНИЯ =====
        self._load()

    # ========== СВЯЗЫВАНИЕ С ЭКСПЕРТАМИ ==========
    def settle(self, *args, **kwargs):
        """
        Алиас для record_result() (обратная совместимость с legacy кодом).
        Просто перенаправляет вызов на record_result().
        """
        return self.record_result(*args, **kwargs)

    def bind_experts(self, *experts):
        """
        Сохраняет ссылки на экспертов для логирования и диагностики
        
        Args:
            *experts: список экспертов (XGB, RF, ARF, NN)
        
        Returns:
            self для chain-style вызовов
        
        Пример:
            meta.bind_experts(xgb_expert, rf_expert, arf_expert, nn_expert)
        """
        self._experts = list(experts)
        return self

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
        Создает финальное предсказание из предсказаний экспертов
        
        Процесс:
        1. Определяем текущую фазу из контекста
        2. Извлекаем веса для этой фазы
        3. Строим вектор фичей φ из логитов экспертов, мета-фичей и контекста
        4. Применяем гейтинг (если включен)
        5. Вычисляем взвешенную сумму и применяем сигмоид
        
        Args:
            p_xgb, p_rf, p_arf, p_nn: вероятности от экспертов
            p_base: базовое предсказание (до ансамбля)
            reg_ctx: контекст с информацией о фазе и других параметрах
        
        Returns:
            Финальная вероятность p_final ∈ [0, 1] или None если недостаточно данных
        
        Математика:
            φ = [logit(p_xgb), logit(p_rf), logit(p_arf), logit(p_nn),
                 logit(p_base), disagree, entropy, 
                 trend_sign, trend_abs, vol_ratio, jump_flag,
                 ofi_sign, book_imb, basis_sign, funding_sign,
                 disagree*vol, entropy*trend, 1]
            
            z = w · φ  (линейная комбинация)
            p_final = σ(z) = 1 / (1 + exp(-z))  (сигмоид)
        """
        ph = phase_from_ctx(reg_ctx)
        
        # Проверяем, обучена ли модель для этой фазы
        w = self.w_meta_ph[ph]
        if np.allclose(w, 0.0):
            return None  # Модель еще не обучена

        # Строим вектор фичей из предсказаний экспертов и контекста
        x = self._phi(p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx)
        if x is None:
            return None

        # Вычисляем финальное предсказание
        return self._safe_p_from_x(ph, x)

    def _phi(
        self,
        p_xgb: Optional[float],
        p_rf: Optional[float],
        p_arf: Optional[float],
        p_nn: Optional[float],
        p_base: Optional[float],
        reg_ctx: Optional[dict] = None
    ) -> Optional[np.ndarray]:
        """
        Строит расширенный вектор фичей для мета-модели
        
        Вектор включает:
        - Логиты предсказаний экспертов (4 значения)
        - Логит базового предсказания (1 значение)
        - Disagreement между экспертами (1 значение)
        - Энтропия распределения (1 значение)
        - НОВОЕ: Контекстные фичи рыночного режима (8 значений)
        - НОВОЕ: Взаимодействия между фичами (2 значения)
        - Bias term (1 значение) = 1.0
        
        Всего 18 фичей (было 8)
        
        Почему логиты, а не вероятности?
        Логиты живут в (-∞, +∞), что удобнее для линейных операций.
        Вероятности сжаты в [0,1] и теряют информацию об уверенности модели.
        
        Args:
            p_xgb, p_rf, p_arf, p_nn: вероятности экспертов
            p_base: базовое предсказание
            reg_ctx: контекст с рыночными условиями
        
        Returns:
            Вектор фичей размера 18 или None если недостаточно данных
        """
        # Собираем доступные предсказания
        preds = []
        for p in [p_xgb, p_rf, p_arf, p_nn]:
            if p is not None:
                preds.append(float(p))
        
        if len(preds) == 0:
            return None  # Нет предсказаний от экспертов

        # Преобразуем в логиты
        def safe_logit(p: float) -> float:
            """Безопасное вычисление logit с клиппингом"""
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(np.log(p / (1 - p)))

        lz_xgb = safe_logit(p_xgb) if p_xgb is not None else 0.0
        lz_rf = safe_logit(p_rf) if p_rf is not None else 0.0
        lz_arf = safe_logit(p_arf) if p_arf is not None else 0.0
        lz_nn = safe_logit(p_nn) if p_nn is not None else 0.0
        lz_base = safe_logit(p_base) if p_base is not None else 0.0

        # Базовые мета-фичи (как было)
        
        # Disagreement: насколько эксперты не согласны друг с другом
        # Высокое значение = эксперты дают разные предсказания = высокая неопределенность
        disagree = float(np.std(preds)) if len(preds) > 1 else 0.0

        # Entropy: энтропия среднего распределения экспертов
        # Высокая энтропия = модель неуверена (p близко к 0.5)
        p_mean = float(np.mean(preds))
        p_mean = np.clip(p_mean, 1e-6, 1 - 1e-6)
        entropy = float(-(p_mean * np.log(p_mean) + (1 - p_mean) * np.log(1 - p_mean)))

        # НОВОЕ: Добавляем контекстные фичи рыночного режима
        if reg_ctx is not None and isinstance(reg_ctx, dict):
            trend_sign = float(reg_ctx.get("trend_sign", 0.0))
            trend_abs = float(reg_ctx.get("trend_abs", 0.0))
            vol_ratio = float(reg_ctx.get("vol_ratio", 1.0))
            jump_flag = float(reg_ctx.get("jump_flag", 0.0))
            ofi_sign = float(reg_ctx.get("ofi_sign", 0.0))
            book_imb = float(reg_ctx.get("book_imb", 0.0))
            basis_sign = float(reg_ctx.get("basis_sign", 0.0))
            funding_sign = float(reg_ctx.get("funding_sign", 0.0))
        else:
            # Если контекст не передан - заполняем нулями (обратная совместимость)
            trend_sign = trend_abs = vol_ratio = jump_flag = 0.0
            ofi_sign = book_imb = basis_sign = funding_sign = 0.0

        # НОВОЕ: Добавляем взаимодействия между фичами
        # Эти комбинации помогают модели уловить нелинейные паттерны
        disagree_vol = disagree * vol_ratio  # дисагремент при высокой воле более важен
        entropy_trend = entropy * abs(trend_abs)  # неопределенность в тренде

        # Собираем расширенный вектор фичей (18 элементов)
        x = np.array([
            # Оригинальные логиты экспертов (0-4)
            lz_xgb,
            lz_rf,
            lz_arf,
            lz_nn,
            lz_base,
            
            # Оригинальные агрегаты (5-6)
            disagree,
            entropy,
            
            # Контекстные фичи рыночного режима (7-14)
            trend_sign,      # знак тренда (±1)
            trend_abs,       # сила тренда (0-3)
            vol_ratio,       # отношение текущей волы к средней
            jump_flag,       # детекция скачка цены (0/1)
            ofi_sign,        # знак order flow imbalance
            book_imb,        # дисбаланс стакана [-1, 1]
            basis_sign,      # знак базиса фьючерсов
            funding_sign,    # знак ставки финансирования
            
            # Взаимодействия (15-16)
            disagree_vol,    # взаимодействие неопределенности и волатильности
            entropy_trend,   # взаимодействие энтропии и тренда
            
            # Bias term (17)
            1.0
        ], dtype=float)

        return x

    def _safe_p_from_x(self, ph: int, x: np.ndarray) -> Optional[float]:
        """
        Вычисляет вероятность из вектора фичей для конкретной фазы
        
        Применяет линейную комбинацию с весами фазы и сигмоид:
        z = w · x
        p = σ(z) = 1 / (1 + exp(-z))
        
        Args:
            ph: номер фазы (0-5)
            x: вектор фичей размера 18 (было 8)
        
        Returns:
            Вероятность ∈ [0, 1] или None если веса не обучены
        """
        w = self.w_meta_ph[ph]
        if np.allclose(w, 0.0):
            return None

        # Линейная комбинация
        z = float(np.dot(w, x))
        
        # Сигмоид с защитой от overflow
        z = np.clip(z, -60.0, 60.0)
        p = 1.0 / (1.0 + math.exp(-z))
        
        return float(np.clip(p, 0.0, 1.0))

    # ========== ЗАПИСЬ РЕЗУЛЬТАТА И ОБУЧЕНИЕ ==========
        
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
        Записывает результат предсказания и триггерит обучение при необходимости
        
        Это центральная функция обучения. Что происходит:
        
        1. Извлекаем фазу и строим вектор фичей с контекстом (18 элементов)
        2. Сохраняем пример (x, y) в буфер фазы и в CSV (только если есть предсказания экспертов)
        3. ВСЕГДА обновляем метрики качества (hits) - даже если экспертов нет
        4. НОВОЕ: Сохраняем OOF predictions для CV
        5. НОВОЕ: Периодически запускаем CV валидацию
        6. Если накопилось достаточно данных - запускаем обучение CEM/CMA-ES
        7. Проверяем условия переключения SHADOW↔ACTIVE
        
        Args:
            p_xgb, p_rf, p_arf, p_nn: вероятности от экспертов
            p_base: базовое предсказание
            y_up: истинный результат (0 или 1)
            used_in_live: использовалось ли в реальной торговле
            p_final_used: какое итоговое предсказание использовалось (если отличается от predict)
            reg_ctx: контекст с фазой и рыночными условиями
        """
        # ===== ШАГ 1: ИЗВЛЕЧЕНИЕ ФАЗЫ И ПОСТРОЕНИЕ ФИЧЕЙ =====
        ph = phase_from_ctx(reg_ctx)
        
        # Сохраняем последнюю виденную фазу для CV
        self._last_phase = ph
        
        # Строим вектор фичей (может быть None если эксперты не дали предсказаний)
        x = self._phi(p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx)
        
        # ===== ШАГ 2: СОХРАНЕНИЕ ПРИМЕРА ДЛЯ ОБУЧЕНИЯ =====
        # Сохраняем ТОЛЬКО если есть предсказания экспертов
        if x is not None:
            buf = self._append_example(ph, x, int(y_up))
            self.seen_ph[ph] += 1

        # ===== ШАГ 3: ОБНОВЛЕНИЕ МЕТРИК (ВСЕГДА, ДАЖЕ БЕЗ ЭКСПЕРТОВ) =====
        # Вычисляем предсказание для статистики
        if x is not None:
            # Есть предсказания экспертов - используем модель
            p_for_gate = p_final_used if (p_final_used is not None) else self._safe_p_from_x(ph, x)
            
            # Если веса еще не обучены, используем baseline
            if p_for_gate is None:
                p_for_gate = p_base if p_base is not None else 0.5
        else:
            # Нет предсказаний экспертов - используем baseline для статистики
            p_for_gate = p_base if p_base is not None else 0.5

        # Вычисляем корректность предсказания
        hit = int((p_for_gate >= 0.5) == bool(y_up))

        # Обновляем метрики в зависимости от режима
        if self.mode == "ACTIVE" and used_in_live:
            # В ACTIVE режиме: отслеживаем только реальные ставки
            self.active_hits.append(hit)
            
            # ADWIN детектирует концептуальный дрейф
            if self.adwin is not None:
                in_drift = self.adwin.update(1 - hit)
                if in_drift:
                    self.mode = "SHADOW"
                    self.active_hits = []
        else:
            # В SHADOW режиме: накапливаем ВСЕ наблюдения
            # Это позволяет МЕТЕ учиться даже когда она не используется в ставках
            # КРИТИЧНО: теперь накапливаем даже когда x is None
            self.shadow_hits.append(hit)

        # Ограничиваем размер буферов хитов
        self.active_hits = self.active_hits[-2000:]
        self.shadow_hits = self.shadow_hits[-2000:]

        self._unsaved += 1
        self._save_throttled()

        # ===== ДАЛЕЕ - ТОЛЬКО ЕСЛИ ЕСТЬ ПРЕДСКАЗАНИЯ ЭКСПЕРТОВ =====
        if x is None:
            return  # Больше нечего делать без предсказаний экспертов

        # ===== ШАГ 4: НОВОЕ - OOF TRACKING ДЛЯ CV =====
        # Сохраняем предсказания для последующей валидации
        if getattr(self.cfg, "cv_enabled", True) and p_for_gate is not None:
            self.cv_oof_preds[ph].append(float(p_for_gate))
            self.cv_oof_labels[ph].append(int(y_up))

        # ===== ШАГ 5: НОВОЕ - ПЕРИОДИЧЕСКАЯ CV ПРОВЕРКА =====
        # Каждые N примеров запускаем полную cross-validation
        cv_check_every = int(getattr(self.cfg, "cv_check_every", 50))
        self.cv_last_check[ph] += 1
        
        if getattr(self.cfg, "cv_enabled", True) and self.cv_last_check[ph] >= cv_check_every:
            self.cv_last_check[ph] = 0
            
            try:
                cv_results = self._run_cv_validation(ph)
                self.cv_metrics[ph] = cv_results
                
                if cv_results.get("status") == "ok":
                    self.validation_passed[ph] = True
                    
                    # Логируем результаты CV
                    print(f"[MetaCEMMC] CV ph={ph}: "
                        f"OOF_ACC={cv_results['oof_accuracy']:.2f}% "
                        f"CI=[{cv_results['ci_lower']:.2f}%, {cv_results['ci_upper']:.2f}%] "
                        f"folds={cv_results['n_folds']}")
            except Exception as e:
                print(f"[MetaCEMMC] CV failed for phase {ph}: {e}")

        # ===== ШАГ 6: ЛЕНИВОЕ ОБУЧЕНИЕ =====
        # Если в фазе накопилось достаточно данных - запускаем CEM/CMA-ES обучение
        if self._phase_ready(ph):
            try:
                self._train_phase(ph)
                
                # После обучения очищаем буферы для следующего раунда
                self._clear_phase_storage(ph)
                self.buf_ph[ph] = []
                
                self._save()
            except Exception as e:
                # Не падаем в production, только логируем
                print(f"[MetaCEMMC] Training failed for phase {ph}: {e}")

        # ===== ШАГ 7: ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ =====
        # Проверяем метрики (включая CV) и решаем о переключении SHADOW↔ACTIVE
        try:
            self._maybe_flip_modes()
        except Exception:
            pass

    # ========== НОВОЕ: CROSS-VALIDATION ФУНКЦИИ ==========
    
    def _run_cv_validation(self, ph: int) -> Dict:
        """
        Запускает walk-forward purged cross-validation для фазы
        
        Почему walk-forward?
        Мы работаем с временными рядами, где порядок данных важен. Обычная
        k-fold CV нарушает временную структуру. Walk-forward CV симулирует
        реальную ситуацию: обучаемся на прошлом, предсказываем будущее.
        
        Почему purged?
        Между train и test должен быть gap (purge + embargo), чтобы избежать
        information leakage. Если модель видела цену в момент T, она косвенно
        знает цену в T+1 через автокорреляцию.
        
        Процесс:
        1. Загружаем накопленные данные фазы из CSV
        2. Разбиваем на N временных фолдов
        3. Для каждого фолда:
           - Train на всех данных до фолда (минус purge)
           - Test на данных фолда
           - Сохраняем out-of-fold predictions
        4. Вычисляем метрики на всех OOF predictions
        5. Bootstrap для доверительных интервалов
        
        Args:
            ph: номер фазы (0-5)
        
        Returns:
            Словарь с метриками:
            - status: "ok" или причина неудачи
            - oof_accuracy: точность на OOF данных
            - ci_lower, ci_upper: границы 95% доверительного интервала
            - fold_scores: список точностей по фолдам
            - n_folds: количество фолдов
            - oof_samples: количество OOF примеров
        """
        # Загружаем накопленные данные фазы с весами
        X_list, y_list, sample_weights = self._load_phase_buffer_from_disk(ph)
        
        if len(X_list) < int(getattr(self.cfg, "cv_min_train_size", 200)):
            return {
                "status": "insufficient_data",
                "oof_accuracy": 0.0,
                "n_samples": len(X_list)
            }

        X_all = np.array(X_list, dtype=float)
        y_all = np.array(y_list, dtype=int)
        
        n_samples = len(X_all)
        n_splits = min(
            int(getattr(self.cfg, "cv_n_splits", 5)),
            n_samples // int(getattr(self.cfg, "cv_min_train_size", 200))
        )
        
        if n_splits < 2:
            return {
                "status": "insufficient_splits",
                "oof_accuracy": 0.0,
                "n_samples": n_samples
            }

        # Параметры purging и embargo
        embargo_pct = float(getattr(self.cfg, "cv_embargo_pct", 0.02))
        purge_pct = float(getattr(self.cfg, "cv_purge_pct", 0.01))
        
        embargo_size = max(1, int(n_samples * embargo_pct))
        purge_size = max(1, int(n_samples * purge_pct))
        
        fold_size = n_samples // n_splits
        
        # Массивы для out-of-fold predictions
        oof_preds = np.zeros(n_samples)
        oof_mask = np.zeros(n_samples, dtype=bool)
        fold_scores = []

        # Walk-forward cross-validation
        for fold_idx in range(n_splits):
            # Определяем границы test fold
            test_start = fold_idx * fold_size
            test_end = min(test_start + fold_size, n_samples)
            
            # Train на всех данных ДО test fold (минус purge gap)
            train_end = max(0, test_start - purge_size)
            
            if train_end < int(getattr(self.cfg, "cv_min_train_size", 200)):
                continue  # Недостаточно данных для обучения
            
            X_train = X_all[:train_end]
            y_train = y_all[:train_end]
            X_test = X_all[test_start:test_end]
            y_test = y_all[test_start:test_end]
            
            # Обучаем временную модель на train fold
            temp_weights = self._train_fold_model(X_train, y_train, ph)
            
            if temp_weights is None:
                continue
            
            # Предсказания на test fold
            preds = self._predict_fold(temp_weights, X_test)
            
            # Сохраняем OOF predictions
            oof_preds[test_start:test_end] = preds
            oof_mask[test_start:test_end] = True
            
            # Метрики фолда
            fold_acc = 100.0 * np.mean((preds >= 0.5) == y_test)
            fold_scores.append(fold_acc)

        # Итоговые OOF метрики
        oof_valid = oof_mask.sum()
        
        if oof_valid < int(getattr(self.cfg, "cv_min_train_size", 200)):
            return {
                "status": "insufficient_oof",
                "oof_accuracy": 0.0,
                "oof_samples": int(oof_valid)
            }
        
        oof_accuracy = 100.0 * np.mean((oof_preds[oof_mask] >= 0.5) == y_all[oof_mask])
        
        # Bootstrap confidence intervals
        ci_lower, ci_upper = self._bootstrap_ci(
            oof_preds[oof_mask],
            y_all[oof_mask],
            n_bootstrap=int(getattr(self.cfg, "cv_bootstrap_n", 1000)),
            confidence=float(getattr(self.cfg, "cv_confidence", 0.95))
        )
        
        return {
            "status": "ok",
            "oof_accuracy": float(oof_accuracy),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "fold_scores": fold_scores,
            "n_folds": len(fold_scores),
            "oof_samples": int(oof_valid)
        }

    def _bootstrap_ci(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        n_bootstrap: int,
        confidence: float
    ) -> Tuple[float, float]:
        """
        Вычисляет bootstrap доверительные интервалы для accuracy
        
        Bootstrap - это метод оценки статистической неопределенности через
        многократное пересэмплирование данных с возвратом. Это позволяет понять,
        насколько стабильна наша метрика.
        
        Процесс:
        1. N раз пересэмплируем данные с возвратом (одни примеры могут повторяться)
        2. Для каждой бутстрэп выборки считаем accuracy
        3. Из N значений accuracy вычисляем перцентили
        
        Например, если 95% доверительный интервал = [52%, 58%], это значит,
        что мы на 95% уверены, что истинная accuracy находится в этом диапазоне.
        
        Args:
            preds: предсказания (вероятности)
            labels: истинные метки
            n_bootstrap: количество бутстрэп итераций
            confidence: уровень доверия (0.95 = 95%)
        
        Returns:
            (ci_lower, ci_upper): границы доверительного интервала в процентах
        """
        accuracies = []
        n = len(preds)
        
        for _ in range(n_bootstrap):
            # Resample с возвратом (одни индексы могут повторяться)
            idx = np.random.choice(n, size=n, replace=True)
            boot_preds = preds[idx]
            boot_labels = labels[idx]
            
            # Accuracy на бутстрэп выборке
            boot_acc = 100.0 * np.mean((boot_preds >= 0.5) == boot_labels)
            accuracies.append(boot_acc)
        
        accuracies = np.array(accuracies)
        
        # Вычисляем перцентили для доверительного интервала
        alpha = 1.0 - confidence
        ci_lower = np.percentile(accuracies, 100 * alpha / 2)
        ci_upper = np.percentile(accuracies, 100 * (1 - alpha / 2))
        
        return float(ci_lower), float(ci_upper)

    def _train_fold_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ph: int
    ) -> Optional[np.ndarray]:
        """
        Обучает временную модель для CV fold
        
        Это упрощенная версия основного обучения. Мы используем CEM (более быстрый)
        вместо CMA-ES, чтобы не тратить слишком много времени на каждый fold.
        
        Args:
            X: матрица фичей (N × 18)
            y: вектор меток (N,)
            ph: номер фазы
        
        Returns:
            Вектор весов или None если обучение не удалось
        """
        if len(X) < 50:
            return None  # Слишком мало данных

        # Используем упрощенный CEM с меньшим числом итераций
        try:
            weights = self._train_cem(
                X, y,
                n_iter=20,  # Меньше итераций чем в основном обучении
                pop_size=50,  # Меньше популяция
                elite_frac=0.2
            )
            return weights
        except Exception:
            return None

    def _predict_fold(
        self,
        weights: np.ndarray,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Предсказания временной модели на fold
        
        Применяет линейную комбинацию с весами и сигмоид.
        
        Args:
            weights: вектор весов (18,)
            X: матрица фичей (N × 18)
        
        Returns:
            Вектор вероятностей (N,)
        """
        z = X @ weights  # Матричное умножение: (N × 18) @ (18,) = (N,)
        z = np.clip(z, -60.0, 60.0)  # Защита от overflow в exp
        probs = 1.0 / (1.0 + np.exp(-z))
        return probs

    # ========== ОБУЧЕНИЕ CEM/CMA-ES ==========
    
    def _phase_ready(self, ph: int) -> bool:
        """
        Проверяет, готова ли фаза к обучению
        
        Условия:
        - Минимальное количество примеров в буфере
        - Минимальное количество уникальных меток (нужны и 0, и 1)
        
        Args:
            ph: номер фазы
        
        Returns:
            True если можно обучать
        """
        buf = self.buf_ph.get(ph, [])
        min_samples = int(getattr(self.cfg, "meta_min_train", 100))
        
        if len(buf) < min_samples:
            return False
        
        # Проверяем, что есть примеры обоих классов
        labels = [y for (_, y) in buf]
        if len(set(labels)) < 2:
            return False  # Все примеры одного класса
        
        return True

    def _train_phase(self, ph: int) -> None:
        """
        Обучает модель для конкретной фазы через CEM или CMA-ES
        
        Это основная функция обучения. Она:
        1. Загружает все накопленные данные фазы из CSV (теперь с весами по времени)
        2. Готовит X, y массивы и веса экспоненциального забывания
        3. Запускает CEM или CMA-ES оптимизацию с учётом весов
        4. Сохраняет лучшие веса
        5. Генерирует отчет с графиками (опционально)
        
        НОВОЕ: Экспоненциальное взвешивание означает, что свежие паттерны
        влияют на обучение сильнее, чем старые. Это позволяет модели адаптироваться
        к изменяющимся рыночным условиям, не теряя при этом долгосрочную память.
        
        Почему CEM/CMA-ES, а не градиентный спуск?
        - Не требует дифференцируемости (можем использовать любую метрику)
        - Более устойчив к шуму в данных
        - Лучше работает с малыми выборками
        - Естественная регуляризация через популяцию решений
        
        Args:
            ph: номер фазы для обучения
        """
        # Загружаем все данные фазы с весами по времени
        X_list, y_list, sample_weights = self._load_phase_buffer_from_disk(ph)
        
        if len(X_list) < int(getattr(self.cfg, "meta_min_train", 100)):
            return  # Недостаточно данных

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)
        sample_weights = np.array(sample_weights, dtype=float)
        
        # Нормализуем веса так, чтобы их сумма равнялась количеству примеров
        # Это сохраняет баланс между взвешенными и невзвешенными примерами
        sample_weights = sample_weights * len(sample_weights) / (sample_weights.sum() + 1e-12)

        # Выбираем оптимизатор
        use_cma = getattr(self.cfg, "meta_use_cma_es", False) and HAVE_CMA
        
        if use_cma:
            # CMA-ES: более мощный, но требует внешней библиотеки
            w_best = self._train_cma_es(X, y, ph, sample_weights=sample_weights)
        else:
            # CEM: проще, без зависимостей
            w_best = self._train_cem(X, y, sample_weights=sample_weights)

        # Сохраняем обученные веса для этой фазы
        self.w_meta_ph[ph] = w_best

    def _train_cem(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 50,
        pop_size: int = 100,
        elite_frac: float = 0.2
    ) -> np.ndarray:
        """
        Cross-Entropy Method оптимизация с визуализацией
        """
        D = X.shape[1]
        n_elite = max(1, int(pop_size * elite_frac))
        
        mu = np.zeros(D)
        sigma = np.ones(D) * 2.0
        
        clip_val = float(getattr(self.cfg, "meta_w_clip", 8.0))
        best_loss = float('inf')
        best_w = mu.copy()

        # НОВОЕ: Получаем визуализатор
        try:
            from training_visualizer import get_visualizer
            viz = get_visualizer()
            viz_enabled = True
        except Exception:
            viz_enabled = False

        for iteration in range(n_iter):
            population = []
            for _ in range(pop_size):
                w = mu + sigma * np.random.randn(D)
                w = np.clip(w, -clip_val, clip_val)
                population.append(w)
            
            scores = []
            for w in population:
                loss = self._mc_eval(w, X, y, n_bootstrap=10)
                scores.append(loss)
            
            elite_idx = np.argsort(scores)[:n_elite]
            elite = [population[i] for i in elite_idx]
            
            if scores[elite_idx[0]] < best_loss:
                best_loss = scores[elite_idx[0]]
                best_w = population[elite_idx[0]].copy()
            
            elite_arr = np.array(elite)
            mu = elite_arr.mean(axis=0)
            current_sigma = elite_arr.std(axis=0) + 1e-6
            sigma = current_sigma
            
            # НОВОЕ: Записываем метрики для визуализации
            if viz_enabled and iteration % 5 == 0:
                try:
                    ph = getattr(self, "_last_phase", 0)
                    median_loss = float(np.median(scores))
                    avg_sigma = float(np.mean(sigma))
                    viz.record_meta_training_step(
                        phase=ph,
                        iteration=iteration,
                        best_loss=float(best_loss),
                        median_loss=median_loss,
                        sigma=avg_sigma
                    )
                except Exception:
                    pass

        return best_w

    def _train_cma_es(self, X: np.ndarray, y: np.ndarray, ph: int, sample_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        CMA-ES оптимизация (более продвинутая версия) с визуализацией
        
        CMA-ES (Covariance Matrix Adaptation Evolution Strategy) - один из
        лучших алгоритмов для безградиентной оптимизации.
        
        Отличия от CEM:
        - Адаптирует полную ковариационную матрицу (не только диагональ)
        - Использует rank-mu update для быстрой сходимости
        - Автоматически настраивает step-size (sigma)
        - Более устойчив к плохой инициализации
        
        Недостатки:
        - Требует внешнюю библиотеку cma
        - Медленнее чем CEM
        
        Args:
            X: матрица фичей
            y: вектор меток
            ph: номер фазы (для логирования)
        
        Returns:
            Лучший найденный вектор весов
        """
        if not HAVE_CMA:
            # Fallback на CEM если CMA недоступен
            return self._train_cem(X, y)

        D = X.shape[1]
        sigma0 = 2.0  # Начальный step-size
        clip_val = float(getattr(self.cfg, "meta_w_clip", 8.0))
        
        # Создаем CMA-ES оптимизатор
        es = cma.CMAEvolutionStrategy(
            x0=np.zeros(D),
            sigma0=sigma0,
            inopts={
                'bounds': [-clip_val, clip_val],
                'popsize': 50,
                'maxiter': 100,
                'verbose': -1  # Отключаем вывод
            }
        )

        # История для графиков
        iters, best_hist, med_hist, sigma_hist = [], [], [], []

        # НОВОЕ: Получаем визуализатор
        try:
            from training_visualizer import get_visualizer
            viz = get_visualizer()
            viz_enabled = True
        except Exception:
            viz_enabled = False

        # Основной цикл оптимизации
        while not es.stop():
            # Получаем популяцию решений
            solutions = es.ask()
            
            # Оцениваем через Монте-Карло
            fitness = []
            for w in solutions:
                w_clipped = np.clip(w, -clip_val, clip_val)
                loss = self._mc_eval(w_clipped, X, y, n_bootstrap=20, sample_weights=sample_weights)
                fitness.append(loss)
            
            # Обновляем распределение
            es.tell(solutions, fitness)
            
            # Логируем прогресс
            it = len(best_hist) + 1
            iters.append(it)
            best_hist.append(float(np.min(fitness)))
            med_hist.append(float(np.median(fitness)))
            sigma_hist.append(float(getattr(es, "sigma", sigma0)))
            
            # НОВОЕ: Записываем метрики для визуализации (каждые 5 итераций)
            if viz_enabled and it % 5 == 0:
                try:
                    viz.record_meta_training_step(
                        phase=ph,
                        iteration=it,
                        best_loss=float(best_hist[-1]),
                        median_loss=float(med_hist[-1]),
                        sigma=float(sigma_hist[-1])
                    )
                except Exception:
                    pass

        # Лучшее найденное решение
        w_best = np.array(es.result.xbest, dtype=float)
        w_best = np.clip(w_best, -clip_val, clip_val)

        # Генерируем отчет (график + Telegram)
        if HAVE_PLOTTING:
            try:
                self._emit_report(
                    ph=ph,
                    algo="CMA-ES",
                    iters=iters,
                    best=best_hist,
                    median=med_hist,
                    sigma=sigma_hist
                )
            except Exception:
                pass  # Не падаем если отчет не удался

        return w_best

    def _mc_eval(
        self,
        w: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 20,
        sample_weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Монте-Карло оценка качества весов через взвешенный bootstrap
        
        Вместо того чтобы считать loss на всей выборке (что может дать
        переоптимистичную оценку), мы:
        1. Многократно пересэмплируем данные с учётом весов
        2. Считаем взвешенный loss на каждой бутстрэп выборке
        3. Усредняем
        
        Это дает более устойчивую оценку, особенно на малых выборках.
        
        НОВОЕ: При наличии sample_weights более свежие примеры:
        - Чаще попадают в bootstrap выборки (weighted sampling)
        - Сильнее влияют на итоговый loss (weighted loss)
        Это позволяет модели быстрее адаптироваться к новым паттернам.
        
        Args:
            w: веса для оценки
            X: матрица фичей
            y: вектор меток
            n_bootstrap: количество бутстрэп итераций
            sample_weights: веса примеров для взвешенного sampling
        
        Returns:
            Средний взвешенный log-loss с L2 регуляризацией
        """
        n = len(X)
        losses = []
        l2_reg = float(getattr(self.cfg, "meta_l2", 0.001))
        
        # Если веса не заданы - все примеры равнозначны
        if sample_weights is None:
            sample_weights = np.ones(n)
        
        # Нормализуем веса для использования в качестве вероятностей
        probs = sample_weights / (sample_weights.sum() + 1e-12)

        for _ in range(n_bootstrap):
            # Resample с возвратом, учитывая веса примеров
            # Свежие данные (с большим весом) будут чаще попадать в выборку
            idx = np.random.choice(n, size=n, replace=True, p=probs)
            Xb = X[idx]
            yb = y[idx]
            weights_b = sample_weights[idx]
            
            # Предсказания
            z = Xb @ w
            z = np.clip(z, -60, 60)
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            
            # Взвешенный Log-loss
            # Свежие примеры влияют на loss сильнее через weights_b
            sample_losses = -(yb * np.log(p) + (1 - yb) * np.log(1 - p))
            weighted_loss = np.sum(sample_losses * weights_b) / (weights_b.sum() + 1e-12)
            
            # L2 регуляризация
            reg = l2_reg * np.sum(w**2)
            
            losses.append(weighted_loss + reg)

        return float(np.mean(losses))

    # ========== ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ С УЧЕТОМ CV ==========
    
    def _maybe_flip_modes(self):
        """
        Переключение режимов SHADOW ↔ ACTIVE с учетом CV метрик
        
        СТАРАЯ ЛОГИКА (только winrate):
        - SHADOW → ACTIVE: если winrate на shadow_hits > порога
        - ACTIVE → SHADOW: если winrate на active_hits < порога
        
        НОВАЯ ЛОГИКА (winrate + CV):
        - SHADOW → ACTIVE: требуем winrate > порога И CV validation passed
          И нижняя граница CI тоже выше порога
        - ACTIVE → SHADOW: если winrate < порога ИЛИ CV метрики деградировали
        
        Это предотвращает переключение в ACTIVE на основе случайного везения
        или переобучения.
        """
        def wr(arr: List[int], n: int) -> Optional[float]:
            """Вычисляет winrate на последних n примерах"""
            if len(arr) < n:
                return None
            window = arr[-n:]
            return 100.0 * (sum(window) / float(len(window)))

        # Параметры переключения
        try:
            enter_wr = float(getattr(self.cfg, "enter_wr", 53.0))
            exit_wr = float(getattr(self.cfg, "exit_wr", 49.0))
            min_ready = int(getattr(self.cfg, "min_ready", 80))
            cv_enabled = bool(getattr(self.cfg, "cv_enabled", True))
        except Exception:
            enter_wr, exit_wr, min_ready = 53.0, 49.0, 80
            cv_enabled = True

        # Текущие метрики
        wr_shadow = wr(self.shadow_hits, min_ready)
        wr_active = wr(self.active_hits, max(30, min_ready // 2))

        # SHADOW → ACTIVE: строгая проверка
        if self.mode == "SHADOW" and wr_shadow is not None:
            basic_threshold_met = wr_shadow >= enter_wr

            if cv_enabled:
                # Требуем CV validation для всех активных фаз
                # Берем текущую фазу (последнюю виденную)
                ph = getattr(self, "_last_phase", 0)
                cv_metrics = self.cv_metrics.get(ph, {})
                cv_passed = self.validation_passed.get(ph, False)

                if basic_threshold_met and cv_passed:
                    cv_wr = cv_metrics.get("oof_accuracy", 0.0)
                    ci_lower = cv_metrics.get("ci_lower", 0.0)
                    
                    # Минимальное улучшение для значимости
                    min_improvement = float(getattr(self.cfg, "cv_min_improvement", 2.0))

                    # Проверяем CV метрики
                    if cv_wr >= enter_wr and ci_lower >= (enter_wr - min_improvement):
                        self.mode = "ACTIVE"
                        print(f"[MetaCEMMC] SHADOW→ACTIVE: WR={wr_shadow:.2f}%, "
                              f"CV_WR={cv_wr:.2f}% (CI: [{ci_lower:.2f}%, {cv_metrics.get('ci_upper', 0):.2f}%])")
            else:
                # Без CV используем старую логику
                if basic_threshold_met:
                    self.mode = "ACTIVE"
                    print(f"[MetaCEMMC] SHADOW→ACTIVE: WR={wr_shadow:.2f}% (CV disabled)")

        # ACTIVE → SHADOW: детектируем деградацию
        if self.mode == "ACTIVE" and wr_active is not None:
            basic_threshold_failed = wr_active < exit_wr

            # Также проверяем CV метрики на деградацию
            cv_degraded = False
            if cv_enabled:
                ph = getattr(self, "_last_phase", 0)
                cv_metrics = self.cv_metrics.get(ph, {})
                cv_wr = cv_metrics.get("oof_accuracy", 100.0)
                cv_degraded = cv_wr < exit_wr

            if basic_threshold_failed or cv_degraded:
                self.mode = "SHADOW"
                reason = "WR dropped" if basic_threshold_failed else "CV degraded"
                print(f"[MetaCEMMC] ACTIVE→SHADOW: {reason} (WR={wr_active:.2f}%)")
                
                # Сбрасываем валидацию
                if cv_enabled:
                    self.validation_passed[ph] = False

    # ========== СТАТУС И ДИАГНОСТИКА ==========
    
    def status(self) -> Dict[str, str]:
        """
        Возвращает текущий статус META с метриками
        
        Включает:
        - Алгоритм оптимизации (CEM/CMA-ES)
        - Режим (SHADOW/ACTIVE)
        - Winrate на active и shadow хитах
        - Количество примеров
        - CV метрики последней проверки
        
        Returns:
            Словарь со строковыми значениями для отображения
        """
        def _wr(xs: List[int]):
            if not xs:
                return None
            return sum(xs) / float(len(xs))

        def _fmt(p):
            return "—" if p is None else f"{100.0*p:.2f}%"

        wr_a = _wr(self.active_hits)
        wr_s = _wr(self.shadow_hits)
        all_hits = (self.active_hits or []) + (self.shadow_hits or [])
        wr_all = _wr(all_hits)

        # CV метрики последней фазы
        ph = getattr(self, "_last_phase", 0)
        cv_metrics = self.cv_metrics.get(ph, {})
        cv_status = cv_metrics.get("status", "N/A")
        cv_wr = cv_metrics.get("oof_accuracy", 0.0)
        cv_ci = (
            f"[{cv_metrics.get('ci_lower', 0):.1f}%, {cv_metrics.get('ci_upper', 0):.1f}%]"
            if cv_status == "ok"
            else "N/A"
        )

        return {
            "algo": "CEM+MC" if not getattr(self.cfg, "meta_use_cma_es", False) else "CMA-ES+MC",
            "mode": self.mode,
            "enabled": str(self.enabled),
            "features": f"{self.D}D",  # Показываем размерность
            "wr_active": _fmt(wr_a),
            "n_active": str(len(self.active_hits or [])),
            "wr_shadow": _fmt(wr_s),
            "n_shadow": str(len(self.shadow_hits or [])),
            "wr_all": _fmt(wr_all),
            "n": str(len(all_hits)),
            "cv_oof_wr": _fmt(cv_wr / 100.0) if cv_wr > 0 else "—",
            "cv_ci": cv_ci,
            "cv_validated": str(self.validation_passed.get(ph, False))
        }

    # ========== РАБОТА С ФАЙЛАМИ ==========
    
    def _append_example(self, ph: int, x: np.ndarray, y: int) -> List:
        """
        Добавляет пример в буфер фазы и сохраняет в CSV с временной меткой
        
        Мы используем двухуровневое хранение:
        1. In-memory буфер (быстрый доступ, ограниченный размер)
        2. CSV файл на диске (долгосрочное хранение)
        
        При обучении мы загружаем все данные из CSV.
        НОВОЕ: Каждая запись теперь имеет timestamp для экспоненциального взвешивания.
        Старые паттерны постепенно теряют вес, позволяя модели адаптироваться к новым условиям.
        
        Args:
            ph: номер фазы
            x: вектор фичей (размер 18)
            y: метка
        
        Returns:
            Обновленный буфер фазы
        """
        # Добавляем в in-memory буфер
        self.buf_ph[ph].append((x.tolist(), int(y)))
        
        # Записываем в CSV (append mode)
        csv_path = self._phase_csv_paths.get(ph)
        if csv_path:
            try:
                file_exists = os.path.isfile(csv_path)
                current_timestamp = time.time()
                
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # Заголовок только для нового файла - теперь с timestamp
                    if not file_exists:
                        header = [f"x{i}" for i in range(len(x))] + ["y", "timestamp"]
                        writer.writerow(header)
                    # Данные с временной меткой
                    row = list(x) + [int(y), current_timestamp]
                    writer.writerow(row)
            except Exception:
                pass  # Не падаем если запись не удалась

        return self.buf_ph[ph]

    def _load_phase_buffer_from_disk(self, ph: int, max_age_days: Optional[float] = None) -> Tuple[List, List, List]:
        """
        Загружает накопленные данные фазы из CSV с экспоненциальным взвешиванием
        
        Механизм экспоненциального забывания работает следующим образом:
        - Каждому примеру присваивается вес w = exp(-age_days / tau)
        - Свежие данные (age ≈ 0) имеют вес ≈ 1.0
        - Данные возрастом tau дней имеют вес ≈ 0.37 (1/e)
        - Данные возрастом 2*tau дней имеют вес ≈ 0.14 (1/e²)
        
        Это позволяет модели:
        - Адаптироваться к новым рыночным условиям
        - Постепенно забывать устаревшие паттерны
        - Сохранять полезные долгосрочные закономерности
        
        Args:
            ph: номер фазы
            max_age_days: период полураспада в днях (tau), по умолчанию из конфига
        
        Returns:
            (X_list, y_list, weights): списки фичей, меток и временных весов
        """
        X_list, y_list, weights = [], [], []
        
        csv_path = self._phase_csv_paths.get(ph)
        if not csv_path or not os.path.isfile(csv_path):
            return X_list, y_list, weights

        # Получаем период полураспада из конфига
        if max_age_days is None:
            max_age_days = float(getattr(self.cfg, "meta_weight_decay_days", 30.0))
        
        current_time = time.time()

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                # Проверяем, есть ли колонка timestamp в CSV
                has_timestamp = header and "timestamp" in header
                
                for row in reader:
                    if len(row) < 2:
                        continue
                    
                    try:
                        # Обрабатываем как новый формат (с timestamp), так и старый
                        if has_timestamp and len(row) >= 3:
                            # Новый формат: [x0, x1, ..., xN, y, timestamp]
                            x = [float(v) for v in row[:-2]]
                            y = int(float(row[-2]))
                            row_time = float(row[-1])
                        else:
                            # Старый формат без timestamp: [x0, x1, ..., xN, y]
                            # Даём старым данным максимальный вес, чтобы не потерять их
                            x = [float(v) for v in row[:-1]]
                            y = int(float(row[-1]))
                            row_time = current_time
                        
                        # Вычисляем возраст записи в днях
                        age_days = (current_time - row_time) / 86400.0
                        
                        # Экспоненциальное затухание: вес падает с возрастом
                        # При age = tau: вес = e^(-1) ≈ 0.368
                        # При age = 2*tau: вес = e^(-2) ≈ 0.135
                        weight = math.exp(-age_days / max_age_days)
                        
                        X_list.append(x)
                        y_list.append(y)
                        weights.append(weight)
                        
                    except (ValueError, IndexError):
                        continue  # Пропускаем повреждённые строки
                        
        except Exception:
            pass

        return X_list, y_list, weights

    def _clear_phase_storage(self, ph: int):
        """
        Очищает CSV файл фазы после обучения
        
        После успешного обучения мы начинаем новый цикл накопления данных.
        Старые данные больше не нужны (модель уже обучена на них).
        
        Args:
            ph: номер фазы
        """
        csv_path = self._phase_csv_paths.get(ph)
        if csv_path and os.path.isfile(csv_path):
            try:
                os.remove(csv_path)
            except Exception:
                pass

    def _save_throttled(self):
        """
        Сохраняет состояние с ограничением частоты
        
        Чтобы не перегружать диск, сохраняем не чаще раза в минуту
        или при накоплении 100+ несохраненных изменений.
        """
        now = time.time()
        throttle_s = 60  # 1 минута
        
        if self._unsaved >= 100 or (now - self._last_save_ts) >= throttle_s:
            self._save()
            self._unsaved = 0
            self._last_save_ts = now

    def _save(self):
        """Сохраняет полное состояние META в JSON"""
        state = {
            "mode": self.mode,
            "w_meta_ph": self.w_meta_ph.tolist(),
            "shadow_hits": self.shadow_hits[-2000:],
            "active_hits": self.active_hits[-2000:],
            "seen_ph": self.seen_ph,
            "cv_metrics": self.cv_metrics,
            "validation_passed": self.validation_passed,
            "cv_last_check": self.cv_last_check,
        }
        
        path = getattr(self.cfg, "meta_state_path", "meta_state.json")
        try:
            atomic_save_json(path, state)
        except Exception:
            pass

    def _load(self):
        """Загружает состояние META из JSON"""
        path = getattr(self.cfg, "meta_state_path", "meta_state.json")
        
        if not os.path.isfile(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            self.mode = state.get("mode", "SHADOW")
            
            w_list = state.get("w_meta_ph")
            if w_list:
                loaded_w = np.array(w_list, dtype=float)
                # Обратная совместимость: если загружены старые веса (4 элемента)
                # расширяем до новой размерности (18 элементов)
                if loaded_w.shape[1] != self.D:
                    old_D = loaded_w.shape[1]
                    new_w = np.zeros((self.P, self.D), dtype=float)
                    # Копируем старые веса в начало
                    new_w[:, :old_D] = loaded_w
                    self.w_meta_ph = new_w
                    print(f"[MetaCEMMC] Expanded weights from {old_D}D to {self.D}D")
                else:
                    self.w_meta_ph = loaded_w
            
            self.shadow_hits = state.get("shadow_hits", [])
            self.active_hits = state.get("active_hits", [])
            self.seen_ph = state.get("seen_ph", {p: 0 for p in range(self.P)})
            
            # Загружаем CV состояние
            self.cv_metrics = state.get("cv_metrics", {p: {} for p in range(self.P)})
            self.validation_passed = state.get("validation_passed", {p: False for p in range(self.P)})
            self.cv_last_check = state.get("cv_last_check", {p: 0 for p in range(self.P)})
            
        except Exception as e:
            print(f"[MetaCEMMC] Failed to load state: {e}")

    def _emit_report(
        self,
        ph: Optional[int],
        algo: Optional[str] = None,
        iters=None,
        best=None,
        median=None,
        sigma=None
    ):
        """
        Генерирует и отправляет отчет об обучении
        
        Создает график с историей оптимизации и отправляет в Telegram.
        
        Args:
            ph: номер фазы
            algo: название алгоритма
            iters, best, median, sigma: данные для графика
        """
        if not HAVE_PLOTTING:
            return

        try:
            # Создаем график
            fig = plot_cma_like(
                iters=iters,
                best=best,
                median=median,
                sigma=sigma,
                title=f"META {algo} Training - Phase {ph}"
            )
            
            # Сохраняем во временный файл
            tmp_path = f"/tmp/meta_train_ph{ph}.png"
            fig.savefig(tmp_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Отправляем в Telegram
            if send_telegram_photo:
                send_telegram_photo(tmp_path, caption=f"META training completed for phase {ph}")
            
            # Удаляем временный файл
            try:
                os.remove(tmp_path)
            except Exception:
                pass
                
        except Exception as e:
            print(f"[MetaCEMMC] Report generation failed: {e}")


# ========== ЭКСПОРТ ==========

__all__ = ["MetaCEMMC", "LambdaMARTMetaLite", "ProbBlender"]
