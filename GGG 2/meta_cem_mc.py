# -*- coding: utf-8 -*-
"""
meta_cem_mc.py — META на основе CEM/CMA-ES + Монте‑Карло (bootstrap).

Идея:
- У нас есть 4 эксперта (XGB, RF, ARF, NN) + базовый p_base до ансамбля.
- МЕТА строит p_final = σ( w · φ ), где φ = [logit(pxgb), logit(prf), logit(parf), logit(pnn),
  logit(p_base), disagree, entropy, 1].
- w обучаем не градиентно, а стохастической оптимизацией по лог‑лоссу,
  причём оценку качества считаем по Монте‑Карло (бутстрэп выборок из буфера)
  — это повышает устойчивость к шуму/дрейфу, особенно при малых выборках.

- Реализованы два оптимизатора:
  1) CEM (Cross‑Entropy Method) — без внешних зависимостей, по умолчанию.
  2) CMA‑ES (если установлен пакет `cma`). Включается cfg.meta_use_cma_es=True.

- Память/обучение ведём ПО ФАЗАМ (φ in [0..P‑1]) — как и в остальном коде.
- Тренировка запускается «лениво» из record_result() при накоплении новых примеров.

API совместим с текущей MetaStacking:
    predict(...), record_result(...), bind_experts(...), status()

Файл состояния: cfg.meta_state_path (JSON)
"""
from __future__ import annotations
import os, json, time, math, random, csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# опционально используем CMA‑ES, если установлен пакет `cma`
# опционально используем CMA-ES, если установлен пакет `cma`
try:
    import cma  # type: ignore
    HAVE_CMA = True
except Exception:
    cma = None
    HAVE_CMA = False

# NEW: графики и Telegram
# NEW: графики и Telegram
from meta_report import plot_cma_like, send_telegram_photo, send_telegram_text
from expert_report import plot_experts_reliability_panel  # ← новый модуль
import matplotlib.pyplot as plt  # на случай headless backends

from collections import deque  # ← NEW

# === NEW: LambdaMART-вторая МЕТА и блендер вероятностей ===
try:
    from models.lambdamart_expert import LambdaMARTExpert as _LMCore
    _HAVE_LAMBDAMART = True
except Exception:
    _LMCore = None
    _HAVE_LAMBDAMART = False

# безопасное сохранение — уже есть в проекте
try:
    from state_safety import atomic_save_json
except Exception:
    def atomic_save_json(path: str, obj: dict):  # fallback
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

# фазы/контекст
try:
    from meta_ctx import phase_from_ctx
except Exception:
    def phase_from_ctx(ctx: Optional[dict]) -> int:
        return int(ctx.get("phase", 0) if isinstance(ctx, dict) else 0)

# ---- helpers ----
_EPS = 1e-8

def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    if isinstance(z, np.ndarray):
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))
    else:
        z = _clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + math.exp(-z))

def logit(p: float) -> float:
    p = _clip(float(p), _EPS, 1.0 - _EPS)
    return math.log(p / (1.0 - p))

def _entropy(ps: List[Optional[float]]) -> float:
    p = np.array([q for q in ps if q is not None], dtype=float)
    if p.size == 0:
        return 0.0
    p = np.clip(p, _EPS, 1.0 - _EPS)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(h.mean())

@dataclass
class _CEMCfg:
    pop_size: int = 96
    elite_frac: float = 0.2
    iters: int = 25
    init_scale: float = 0.5
    l2: float = 1e-3
    weight_clip: float = 5.0
    n_boot: int = 64              # сколько бутстрэп‑реплик для оценки одной особи
    boot_size: int = 256          # размер бутстрэп выборки (<= фактических наблюдений)
    min_ready: int = 50          # минимум наблюдений фазы для старта оптимизации
    retrain_every: int = 50       # дообучать каждые N новых примеров
    max_buffer_per_phase: int = 10_000

class MetaCEMMC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = True
        self.mode = "SHADOW"  # совместимо с текущей логикой переключений
        self._experts = []

        # гист/метрики для status()
        self.active_hits: List[int] = []
        self.shadow_hits: List[int] = []

        # NEW: троттлинг сохранений
        self._unsaved = 0
        self._last_save_ts = 0.0

        # конфиг оптимизации (часть можно прокинуть из MLConfig, если добавите поля)
        self.opt = _CEMCfg()
        try:  # необязательные поля из MLConfig
            self.opt.min_ready = int(getattr(cfg, "phase_min_ready", self.opt.min_ready))
            self.opt.max_buffer_per_phase = int(getattr(cfg, "phase_memory_cap", self.opt.max_buffer_per_phase))
            self.opt.retrain_every = int(getattr(cfg, "meta_retrain_every", self.opt.retrain_every))
        except Exception:
            pass

        # параметры по фазам: φ -> w( D )
        self.P: int = int(getattr(cfg, "phase_count", 6))
        self.D: int = 8  # размер φ‑вектора
        self.w_ph: Dict[int, np.ndarray] = {}
        self.seen_ph: Dict[int, int] = {k: 0 for k in range(self.P)}

        # директория для файловых буферов примеров (переживёт рестарты)
        self.examples_dir = getattr(cfg, "meta_examples_dir", "meta_examples")
        os.makedirs(self.examples_dir, exist_ok=True)

        # буферы обучающих примеров по фазам
        # каждый элемент: (x: np.ndarray[D], y: int)
        self.buf_ph: Dict[int, List[Tuple[np.ndarray, int]]] = self._load_phase_buffers()  # ← грузим буферы из файлов

        # куда сохраняем состояние
        self.state_path = getattr(cfg, "meta_state_path", "meta_state.json")


        # Telegram + директория отчётов
        # 1) приоритетно берём из cfg (устанавливается в основном файле),
        # 2) иначе — из переменных окружения,

        # 3) иначе — None (отправка будет пропущена).
        tok = getattr(cfg, "tg_bot_token", None) or os.getenv("TG_BOT_TOKEN")
        cid = getattr(cfg, "tg_chat_id",   None) or os.getenv("TG_CHAT_ID")

        self.tg_token   = str(tok) if tok else None
        self.tg_chat_id = str(cid) if (cid is not None) else None
        self.rep_dir    = getattr(cfg, "meta_report_dir", getattr(cfg, "reports_dir", "meta_reports"))

        # последняя сводка по фазам для дельт на подписи
        self.last_rep: Dict[int, Dict[str, float]] = {}
        self._load()


    # ---- совместимый API ----
    def bind_experts(self, *experts):
        self._experts = list(experts)
        return self

    def predict(
        self,
        p_xgb: Optional[float],
        p_rf: Optional[float],
        p_arf: Optional[float],
        p_nn: Optional[float],
        p_base: Optional[float],
        reg_ctx: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        x = self._phi(p_xgb, p_rf, p_arf, p_nn, p_base)
        ph = phase_from_ctx(reg_ctx)
        w = self.w_ph.get(ph)
        if w is None:
            # запасной путь — равномерная смесь логитов
            lz = [logit(p) for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
            if len(lz) == 0:
                return p_base  # вообще нет экспертов
            z = float(np.mean(lz))
            return float(sigmoid(z))
        z = float(np.dot(w, x))
        return float(sigmoid(z))

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
        reg_ctx: Optional[Dict[str, float]] = None,
    ):
        """Вызывается на settle. Сохраняем (x,y), триггерим обучение.
        Также обновляем WR‑метрики для status().
        """
        x = self._phi(p_xgb, p_rf, p_arf, p_nn, p_base)
        ph = phase_from_ctx(reg_ctx)

        # накопление примеров
        buf = self._append_example(ph, x, int(y_up))  # ← сразу пишем в файл и обновляем in-memory
        # перенос записи в файл: см. _append_example(ph, x, y)

        self.seen_ph[ph] += 1

        # метрики
        p_for_gate = p_final_used if (p_final_used is not None) else self._safe_p_from_x(ph, x)
        if p_for_gate is not None:
            hit = int((p_for_gate >= 0.5) == bool(y_up))
            if used_in_live and self.mode == "ACTIVE":
                self.active_hits.append(hit)
            else:
                self.shadow_hits.append(hit)
            self.active_hits = self.active_hits[-2000:]
            self.shadow_hits = self.shadow_hits[-2000:]

            self._unsaved += 1
            self._save_throttled()

        # ленивое обучение
        if self._phase_ready(ph):
            try:
                self._train_phase(ph)
                self._clear_phase_storage(ph)   # ← очищаем корзину, начинаем новый раунд накопления
                self.buf_ph[ph] = []            # ← синхронизируем оперативную корзину
                self._save()
            except Exception as e:
                # не падаем в лайве
                print(f"[meta-cem] train failed for phase {ph}: {e}")

        # авто‑переключение SHADOW↔ACTIVE по wr, как в MetaStacking
        try:
            self._maybe_flip_modes()
        except Exception:
            pass

    def status(self) -> Dict[str, str]:
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
        return {
            "algo": "CEM+MC" if not getattr(self.cfg, "meta_use_cma_es", False) else "CMA-ES+MC",
            "mode": self.mode,
            "enabled": self.enabled,
            "wr_active": _fmt(wr_a),
            "n_active": len(self.active_hits or []),
            "wr_shadow": _fmt(wr_s),
            "n_shadow": len(self.shadow_hits or []),
            "wr_all": _fmt(wr_all),
            "n": len(all_hits),
        }

    # ---- внутрянка ----
    def _phi(
        self,
        p_xgb: Optional[float],
        p_rf: Optional[float],
        p_arf: Optional[float],
        p_nn: Optional[float],
        p_base: Optional[float],
    ) -> np.ndarray:
        # логиты экспертов (None -> 0 в логе — нейтрально)
        lzs = [logit(p) if (p is not None) else 0.0 for p in [p_xgb, p_rf, p_arf, p_nn]]
        lz_base = logit(p_base) if (p_base is not None) else 0.0
        plist = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
        disagree = float(np.mean([abs(p - 0.5) for p in plist])) if plist else 0.0
        ent = _entropy([p_xgb, p_rf, p_arf, p_nn])
        return np.array([lzs[0], lzs[1], lzs[2], lzs[3], lz_base, disagree, ent, 1.0], dtype=float)

    def _safe_p_from_x(self, ph: int, x: np.ndarray) -> Optional[float]:
        w = self.w_ph.get(ph)
        if w is None:
            return None
        return float(sigmoid(float(np.dot(w, x))))

    def _loss_bootstrap(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, n_boot: int, boot_size: int) -> float:
        """MC‑оценка среднего лог‑лосса по бутстрэп‑подвыборкам.
        Возвращает loss (меньше — лучше).
        """
        N = X.shape[0]
        boot_size = min(boot_size, N)
        losses = []
        for _ in range(n_boot):
            idx = np.random.randint(0, N, size=boot_size)
            Xb = X[idx]
            yb = y[idx]
            z = Xb @ w
            p = sigmoid(z)
            p = np.clip(p, _EPS, 1.0 - _EPS)
            ll = -(yb * np.log(p) + (1 - yb) * np.log(1 - p)).mean()
            losses.append(ll)
        loss = float(np.mean(losses)) + float(l2) * float(np.dot(w, w))
        return loss

    def _save_throttled(self, force: bool = False):
        try:
            now = time.time()
            if force or (now - getattr(self, "_last_save_ts", 0) >= 60) or (getattr(self, "_unsaved", 0) >= 10):
                self._save()
                self._last_save_ts = now
                self._unsaved = 0
        except Exception:
            pass

    # ---- файловое хранилище примеров по фазам ----
    def _phase_path(self, ph: int) -> str:
        return os.path.join(self.examples_dir, f"phase_{ph}.csv")

    def _load_phase_buffers(self) -> Dict[int, List[Tuple[np.ndarray, int]]]:
        # читаем последние max_buffer_per_phase примеров в память (для совместимости того, что уже есть)
        os.makedirs(self.examples_dir, exist_ok=True)
        out: Dict[int, List[Tuple[np.ndarray, int]]] = {}
        for k in range(self.P):
            path = self._phase_path(k)
            buf: List[Tuple[np.ndarray, int]] = []
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            row = line.strip().split(",")
                            if len(row) != self.D + 1:
                                continue
                            xs = np.array([float(v) for v in row[:self.D]], dtype=float)
                            y  = int(float(row[-1]))
                            buf.append((xs, y))
                except Exception:
                    pass
            out[k] = buf[-int(self.opt.max_buffer_per_phase):]
        return out

    def _append_example(self, ph: int, x: np.ndarray, y: int) -> List[Tuple[np.ndarray, int]]:
        os.makedirs(self.examples_dir, exist_ok=True)
        path = self._phase_path(ph)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(",".join([f"{float(v):.10f}" for v in x.tolist()] + [str(int(y))]) + "\n")
        except Exception:
            pass
        # поддерживаем короткий in-memory буфер как раньше
        buf = self.buf_ph.get(ph, [])
        buf.append((x, y))
        if len(buf) > int(self.opt.max_buffer_per_phase):
            del buf[: len(buf) - int(self.opt.max_buffer_per_phase)]
        self.buf_ph[ph] = buf
        return buf

    def _phase_count_on_disk(self, ph: int) -> int:
        path = self._phase_path(ph)
        if not os.path.exists(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _phase_ready(self, ph: int) -> bool:
        n = self._phase_count_on_disk(ph)
        return (n >= int(self.opt.min_ready)) and (n % int(self.opt.retrain_every) == 0)

    def _read_phase_dataset(self, ph: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._phase_path(ph)
        Xs, ys = [], []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    row = line.strip().split(",")
                    if len(row) != self.D + 1:
                        continue
                    Xs.append([float(v) for v in row[:self.D]])
                    ys.append(int(float(row[-1])))
        if not Xs:
            return np.empty((0, self.D)), np.empty((0,), dtype=float)
        X = np.array(Xs, dtype=float)
        y = np.array(ys, dtype=float)
        return X, y

    def _clear_phase_storage(self, ph: int):
        path = self._phase_path(ph)
        try:
            open(path, "w").close()  # просто обнуляем файл после обучения
        except Exception:
            pass



    def _train_phase(self, ph: int):
        """Обучение веса w_φ стохастической оптимизацией.
        Если установлен пакет `cma` и cfg.meta_use_cma_es=True — используем CMA‑ES.
        Иначе — CEM.
        """
        # читаем весь датасет фазы c диска (переживает рестарты)
        X, y = self._read_phase_dataset(ph)
        if len(y) < int(self.opt.min_ready):
            return
        D = X.shape[1]
        assert D == self.D, f"Unexpected feature size {D} != {self.D}"

        # инициализация
        w0 = self.w_ph.get(ph)
        if w0 is None:
            w0 = np.zeros(D, dtype=float)
        if getattr(self.cfg, "meta_use_cma_es", False) and HAVE_CMA:
            w = self._train_cma_es(ph, X, y, w0)
        else:
            w = self._train_cem(ph, X, y, w0)

        self._emit_experts_report(ph, X, y)
        self.w_ph[ph] = w
        self._save()



    # ---- CEM ----
    def _train_cem(self, ph: int, X: np.ndarray, y: np.ndarray, w0: np.ndarray) -> np.ndarray:
        pop = int(self.opt.pop_size)
        elite_k = max(1, int(self.opt.elite_frac * pop))
        iters = int(self.opt.iters)
        l2 = float(self.opt.l2)
        n_boot = int(self.opt.n_boot)
        bsz = int(self.opt.boot_size)
        clip_val = float(self.opt.weight_clip)

        D = X.shape[1]
        mean = w0.copy()
        cov = (self.opt.init_scale ** 2) * np.eye(D)
        best_w = w0.copy()
        best_f = self._loss_bootstrap(X, y, best_w, l2, n_boot, bsz)

        best_hist, med_hist, iters_ax = [], [], []
        for t in range(iters):
            Ws = np.random.multivariate_normal(mean, cov + 1e-6*np.eye(D), size=pop)
            Ws = np.clip(Ws, -clip_val, clip_val)
            fitness = np.array([
                self._loss_bootstrap(X, y, w, l2, n_boot, bsz) for w in Ws
            ], dtype=float)
            elite_idx = np.argsort(fitness)[:elite_k]
            elites = Ws[elite_idx]
            mean = elites.mean(axis=0)
            cov = np.cov(elites.T) + 1e-6*np.eye(D)

            # история для графика
            iters_ax.append(t + 1)
            best_hist.append(float(np.min(fitness)))
            med_hist.append(float(np.median(fitness)))

            if float(fitness[elite_idx[0]]) < float(best_f):
                best_f = float(fitness[elite_idx[0]])
                best_w = elites[0].copy()

        # отчёт (график + Telegram) — фикс: ph передаём явно
        self._emit_report(ph=ph, algo="CEM", iters=iters_ax,
                          best=best_hist, median=med_hist, sigma=None)

        return np.clip(best_w, -clip_val, clip_val)

    # ---- CMA-ES (опционально) ----
    def _train_cma_es(self, ph: int, X: np.ndarray, y: np.ndarray, w0: np.ndarray) -> np.ndarray:
        if not HAVE_CMA:
            # фикс: пробрасываем ph в CEM
            return self._train_cem(ph, X, y, w0)

        l2 = float(self.opt.l2)
        n_boot = int(self.opt.n_boot)
        bsz = int(self.opt.boot_size)
        clip_val = float(self.opt.weight_clip)

        def f(w: List[float]) -> float:
            wv = np.array(w, dtype=float)
            wv = np.clip(wv, -clip_val, clip_val)
            return self._loss_bootstrap(X, y, wv, l2, n_boot, bsz)

        sigma0 = float(self.opt.init_scale)
        opts = {"maxiter": int(self.opt.iters), "verb_disp": 0}
        es = cma.CMAEvolutionStrategy(w0.tolist(), sigma0, opts)

        # история для графика
        iters, best_hist, med_hist, sigma_hist = [], [], [], []

        while not es.stop():
            solutions = es.ask()
            fitness = [f(s) for s in solutions]
            es.tell(solutions, fitness)

            it = len(best_hist) + 1
            iters.append(it)
            best_hist.append(float(np.min(fitness)))
            med_hist.append(float(np.median(fitness)))
            sigma_hist.append(float(getattr(es, "sigma", sigma0)))

        w_best = np.array(es.result.xbest, dtype=float)
        w_best = np.clip(w_best, -clip_val, clip_val)

        # отчёт (график + Telegram) — фикс: ph передаём явно
        self._emit_report(ph=ph, algo="CMA-ES", iters=iters,
                          best=best_hist, median=med_hist, sigma=sigma_hist)

        return w_best


    def _maybe_flip_modes(self):
        def wr(arr: List[int], n: int) -> Optional[float]:
            if len(arr) < n:
                return None
            window = arr[-n:]
            return 100.0 * (sum(window) / float(len(window)))
        try:
            enter_wr = float(getattr(self.cfg, "enter_wr", 53.0))
            exit_wr  = float(getattr(self.cfg, "exit_wr", 49.0))
            min_ready = int(getattr(self.cfg, "min_ready", 80))
        except Exception:
            enter_wr, exit_wr, min_ready = 53.0, 49.0, 80
        wr_shadow = wr(self.shadow_hits, min_ready)
        if self.mode == "SHADOW" and wr_shadow is not None and wr_shadow >= enter_wr:
            self.mode = "ACTIVE"
        wr_active = wr(self.active_hits, max(30, min_ready // 2))
        if self.mode == "ACTIVE" and (wr_active is not None and wr_active < exit_wr):
            self.mode = "SHADOW"

    def _emit_report(self, ph: Optional[int], algo: Optional[str] = None,
                    iters=None, best=None, median=None, sigma=None,
                    attach_last: bool=False):
        """
        Рендер графика и отправка в Telegram.
        Если iters/best/median/sigma=None — значит график уже построен ранее в _train_*,
        и здесь мы только используем сохранённые значения из self._last_plot.
        """
        try:
            if iters is None or best is None:
                return  # ничего не рисовать
            phase = int(ph) if ph is not None else 0
            algo  = algo or ("CMA-ES" if getattr(self.cfg, "meta_use_cma_es", False) and HAVE_CMA else "CEM")
            path  = plot_cma_like(iters, best, median, sigma, phase=phase, algo=algo, out_dir=self.rep_dir)

            # посчитаем дельты от прошлого отчёта (на уровне фазы)
            last = self.last_rep.get(phase, {})
            cur_best = float(best[-1])
            cur_med  = float(median[-1]) if (median is not None and len(median)>0) else None
            cur_sig  = float(sigma[-1]) if (sigma is not None and len(sigma)>0) else None
            ts_now   = int(time.time())
            # подпиcь (укладываемся в лимит Telegram caption ~1024)
            lines = []
            lines.append(f"📈 <b>{algo}</b> — фаза {phase}")
            lines.append(f"best={cur_best:.4g}" + (f" (Δ{cur_best - last.get('best', cur_best):+.4g})" if 'best' in last else ""))
            if cur_med is not None:
                lines.append(f"median={cur_med:.4g}" + (f" (Δ{cur_med - last.get('median', cur_med):+.4g})" if 'median' in last else ""))
            if cur_sig is not None:
                lines.append(f"σ={cur_sig:.4g}" + (f" (Δ{cur_sig - last.get('sigma', cur_sig):+.4g})" if 'sigma' in last else ""))
            lines.append("Кривые: лучшая/медианная пригодность (лог-ось), правая ось — шаг σ (адаптивный).")
            caption = "\n".join(lines)

            # отправка
            if self.tg_token and self.tg_chat_id:
                send_telegram_photo(self.tg_token, self.tg_chat_id, path, caption)
                # короткое текстовое пояснение при желании:
                # send_telegram_text(self.tg_token, self.tg_chat_id, "Пояснение: убывание best→ сходимость; рост σ→ поиск расширяется.")
            else:
                print(f"[meta-report] {caption} | file={path}")

            # обновим last_rep и сохраним
            self.last_rep[phase] = {"best": cur_best}
            if cur_med is not None: self.last_rep[phase]["median"] = cur_med
            if cur_sig is not None: self.last_rep[phase]["sigma"]  = cur_sig
            self.last_rep[phase]["ts"] = ts_now
            self._save()
        except Exception as e:
            print(f"[meta-report] emit failed: {e}")

    def _emit_experts_report(self, ph: int, X: np.ndarray, y: np.ndarray):
        """Строим панель калибровки 4 экспертов из φ-датасета и шлём в Telegram."""
        try:
            # ограничим объём для скорости (последние 2000 наблюдений этой фазы, если есть)
            if len(y) > 2000:
                X, y = X[-2000:], y[-2000:]
            path = plot_experts_reliability_panel(X, y, phase=int(ph), out_dir=self.rep_dir)
            caption = f"🧪 Эксперты: калибровка/качество (φ={int(ph)}). " \
                    f"Данные: {len(y)} примеров. Бины=12."
            if self.tg_token and self.tg_chat_id:
                send_telegram_photo(self.tg_token, self.tg_chat_id, path, caption)
            else:
                print(f"[expert-report] {caption} | file={path}")
        except Exception as e:
            print(f"[expert-report] skipped: {e}")


    # ---- I/O ----
    def _save(self):
        try:
            atomic_save_json(self.state_path, {
                "_magic": "META_CEM_MC_v1",
                "_ts": int(time.time()),
                "P": self.P,
                "D": self.D,
                "w_ph": {int(k): v.tolist() for k, v in self.w_ph.items()},
                "seen_ph": self.seen_ph,
                "last_rep": {int(k): v for k, v in self.last_rep.items()},
                "shadow_hits": self.shadow_hits[-2000:],   # NEW
                "active_hits": self.active_hits[-2000:],
                "mode": self.mode,   # NEW
            })


        except Exception:
            pass


    def _load(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
                if st.get("_magic") in ("META_CEM_MC_v1", "MLSTATE_v1"):
                    self.P = int(st.get("P", self.P))
                    self.D = int(st.get("D", self.D))
                    self.w_ph = {int(k): np.array(v, dtype=float) for k, v in st.get("w_ph", {}).items()}
                    self.seen_ph.update({int(k): int(v) for k, v in st.get("seen_ph", {}).items()})
                    self.last_rep = {int(k): dict(v) for k, v in st.get("last_rep", {}).items()}
                    self.shadow_hits = list(st.get("shadow_hits", []))  # NEW
                    self.active_hits = list(st.get("active_hits", []))  # NEW
                    self.mode = st.get("mode", self.mode)
                    self._maybe_flip_modes()  # NEW: пересчитать режим по сохранённой истории
        except Exception:
            pass
# ... (ниже идут существующие классы MetaCEMMC, утилиты, save/load и т.д.)

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

    def _phi(self, p_xgb, p_rf, p_arf, p_nn, p_base):
        lzs = [_safe_logit(p) for p in [p_xgb, p_rf, p_arf, p_nn]]
        lz_base = _safe_logit(p_base)
        plist = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
        disagree = float(np.mean([abs(p - 0.5) for p in plist])) if plist else 0.0
        ent = _entropy4([p_xgb, p_rf, p_arf, p_nn])
        return np.array([lzs[0], lzs[1], lzs[2], lzs[3], lz_base, disagree, ent, 1.0], dtype=float)

    def _phase_group(self, reg_ctx):
        try:
            from meta_ctx import phase_from_ctx
            return int(phase_from_ctx(reg_ctx or {}))
        except Exception:
            return 0

    def predict(self, p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx=None):
        if not self.enabled or self.model is None:
            return None
        x = self._phi(p_xgb, p_rf, p_arf, p_nn, p_base).reshape(1, -1)
        try:
            s = float(self.model.predict(x)[0])
            p = 1.0 / (1.0 + math.exp(-s))
            return float(np.clip(p, 0.0, 1.0))
        except Exception:
            return None

    def record_result(self, p_xgb, p_rf, p_arf, p_nn, p_base, y_up, reg_ctx=None, used_in_live=False):
        if not self.enabled:
            return
        x = self._phi(p_xgb, p_rf, p_arf, p_nn, p_base)
        g = self._phase_group(reg_ctx)
        self._X.append(x); self._y.append(int(y_up)); self._g.append(g)
        if len(self._X) > self.max_buf:
            drop = len(self._X) - self.max_buf
            self._X = self._X[drop:]; self._y = self._y[drop:]; self._g = self._g[drop:]
        if len(self._X) >= self.min_ready and (len(self._X) - self._last_fit_n) >= self.retrain_every:
            try:
                X = np.vstack(self._X); y = np.asarray(self._y, dtype=int); g = np.asarray(self._g, dtype=int)
                order = np.argsort(g); X = X[order]; y = y[order]; g = g[order]
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
        n = len(self._X); ready = n >= self.min_ready
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
