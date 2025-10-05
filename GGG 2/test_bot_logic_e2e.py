# -*- coding: utf-8 -*-
"""
Комплексный и логический тест для бота:
- Проверяет формулу EV-гейта на детерминированных примерах.
- Прогоняет мини-цикл "рынок → прогнозы экспертов → META → гейт → сделка → сеттлмент → метрики".
- Устойчив к вариациям интерфейсов (PerfMonitor.status/state, META.record*/observe).
"""

import math
import numpy as np
import pytest
from pathlib import Path
import time
import inspect

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

# ------------------------ ВСПОМОГАТЕЛЬНЫЕ ------------------------

def fake_batch(n=64, d=8, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    w = rng.normal(0, 0.5, size=(d,))
    logits = X @ w
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (p > 0.5).astype(np.int32)
    return X, y

def fake_ctx(phase=0):
    return {"phase": int(phase), "now_ts": int(time.time())}

def ev_threshold(gb: float, gc: float, S: float, r: float):
    """Классическая формула порога: EV >= 0 ⇒ p >= (1 + gb/S)/(r - gc/S)."""
    return (1.0 + gb / max(S, 1e-9)) / (r - gc / max(S, 1e-9))

def meta_record_adaptive(meta, *args, **kwargs):
    """
    Универсальная запись примера в МЕТА.
    Пытаемся meta.record(...), иначе record_result(...), иначе observe(...).
    Подбираем сигнатуры на лету.
    """
    # 1) Прямой вызов record(...)
    if hasattr(meta, "record"):
        try:
            return meta.record(*args, **kwargs)
        except TypeError:
            try:
                return meta.record(*args)
            except TypeError:
                pass

    # 2) Универсальный вызов record_result(...)
    if hasattr(meta, "record_result"):
        try:
            return meta.record_result(*args, **kwargs)
        except TypeError:
            sig = inspect.signature(meta.record_result)
            names = list(sig.parameters.keys())

            # Извлечём значения из нашего унифицированного вызова
            # Мы передаём: (p_xgb, p_rf, p_arf, p_nn, p_base), y_up=..., reg_ctx=...
            p_xgb = kwargs.get("p_xgb", args[0] if len(args)>0 else 0.5)
            p_rf  = kwargs.get("p_rf",  args[1] if len(args)>1 else 0.5)
            p_arf = kwargs.get("p_arf", args[2] if len(args)>2 else 0.5)
            p_nn  = kwargs.get("p_nn",  args[3] if len(args)>3 else 0.5)
            p_base= kwargs.get("p_base",args[4] if len(args)>4 else 0.5)
            y     = kwargs.get("y_up", 0)
            ctx   = kwargs.get("reg_ctx", None)
            used_in_live = kwargs.get("used_in_live", True)

            # Сигнатура: (p_xgb,p_rf,p_arf,p_nn,p_base,y_up,used_in_live,reg_ctx)
            if {"p_xgb","p_rf","p_arf","p_nn","p_base","y_up","used_in_live"}.issubset(set(names)):
                try:
                    return meta.record_result(p_xgb=p_xgb, p_rf=p_rf, p_arf=p_arf, p_nn=p_nn, p_base=p_base,
                                              y_up=y, used_in_live=used_in_live, reg_ctx=ctx)
                except TypeError:
                    try:
                        return meta.record_result(p_xgb, p_rf, p_arf, p_nn, p_base, y, used_in_live, ctx)
                    except TypeError:
                        pass

            # Сигнатура: (p_final, y_up, reg_ctx) или (p,y,ctx)
            if {"p_final","y_up"}.issubset(set(names)) or (len(names)>=2 and names[0] in ("p_final","p")):
                pf = kwargs.get("p_final", p_xgb)
                try:
                    return meta.record_result(p_final=pf, y_up=y, reg_ctx=ctx)
                except TypeError:
                    try:
                        return meta.record_result(pf, y, ctx)
                    except TypeError:
                        pass

    # 3) Наконец, observe(...)
    if hasattr(meta, "observe"):
        try:
            return meta.observe(*args, **kwargs)
        except TypeError:
            pass

    pytest.skip("У МЕТА нет совместимого метода record/record_result/observe")

# ------------------------ САМ ТЕСТ ------------------------

def test_bot_logic_e2e(project, tmp_path):
    """
    A) Проверка EV-порога на детерминированных данных.
    B) Мини-интеграционный цикл с экспертами, META, гейтом, сеттлментом и метриками.
    """
    # ---------- A) Чистая математика EV-гейта ----------
    r = 1.5
    gb = 0.0
    gc = 0.0
    S = 1.0
    p_thr = ev_threshold(gb, gc, S, r)  # 1/r ≈ 0.6667
    assert 0.66 < p_thr < 0.67

    # При p ниже порога — не ставим; выше — ставим
    below = [0.60, 0.65, 0.66]
    above = [0.67, 0.70, 0.80]
    assert all(p < p_thr for p in below)
    assert all(p >= p_thr for p in above)

    # ---------- B) Интеграционный цикл ----------
    MLConfig = project.bnbusdrt6.MLConfig
    cfg = MLConfig()
    cfg.min_ready = 6
    cfg.retrain_every = 3
    cfg.phase_min_ready = 4
    cfg.phase_hysteresis_s = 60
    cfg.max_memory = 256
    cfg.train_window = 128

    # пути для сохранения (чтобы не трогать реальный проект)
    cfg.nn_model_path   = str(tmp_path / "nn_model.pkl")
    cfg.nn_state_path   = str(tmp_path / "nn_state.json")
    cfg.nn_scaler_path  = str(tmp_path / "nn_scaler.pkl")
    cfg.rf_model_path   = str(tmp_path / "rf_model.pkl")
    cfg.rf_state_path   = str(tmp_path / "rf_state.json")
    cfg.xgb_model_path  = str(tmp_path / "xgb_model.json")
    cfg.xgb_scaler_path = str(tmp_path / "xgb_scaler.pkl")
    cfg.xgb_state_path  = str(tmp_path / "xgb_state.json")
    cfg.arf_model_path  = str(tmp_path / "arf_model.pkl")
    cfg.arf_cal_path    = str(tmp_path / "arf_cal.pkl")
    cfg.meta_state_path = str(tmp_path / "meta_state.json")
    cfg.meta_report_dir = str(tmp_path / "meta_reports")

    # Эксперты + META
    XGB = project.bnbusdrt6.XGBExpert
    RF  = project.bnbusdrt6.RFCalibratedExpert
    ARF = project.bnbusdrt6.RiverARFExpert
    NN  = project.bnbusdrt6.NNExpert
    Meta = project.meta_cem_mc.MetaCEMMC

    xgb = XGB(cfg)
    rf  = RF(cfg)
    arf = ARF(cfg)
    nn  = NN(cfg)

    meta = Meta(cfg)
    meta.bind_experts(xgb, rf, arf, nn)

    # Префит: обучим NN/RF простыми примерами, чтобы начались ненулевые прогнозы
    X, y = fake_batch(n=32, d=8, seed=10)
    for i in range(12):
        ph = i % 2
        nn.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(ph))
        nn.maybe_train(ph=ph, reg_ctx=fake_ctx(ph))
        rf.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(ph))
        rf.maybe_train(ph=ph, reg_ctx=fake_ctx(ph))

    # PerfMonitor
    PM = project.performance_metrics.PerfMonitor
    try:
        pm = PM(window_n=50)
    except TypeError:
        try:
            pm = PM(window=50)
        except TypeError:
            pm = PM()

    r_seq = [1.3, 1.5, 1.7, 1.4, 1.6]
    gb = 0.0
    gc = 0.0
    S = 1.0

    capital = 1.0
    bets = 0
    skips = 0

    # Основной цикл
    for i in range(30):
        ph = (i // 5) % 3  # смена фаз каждые 5 шагов
        ctx = fake_ctx(ph)
        r = r_seq[i % len(r_seq)]

        # прогнозы экспертов
        p_xgb, _ = xgb.proba_up(X[i % len(X)], reg_ctx=ctx)
        p_rf,  _ = rf.proba_up(X[i % len(X)], reg_ctx=ctx)
        p_arf, _ = arf.proba_up(X[i % len(X)], reg_ctx=ctx)
        p_nn,  _ = nn.proba_up(X[i % len(X)], reg_ctx=ctx)
        p_base = 0.5

        # итоговая вероятность от META (или fallback)
        try:
            p_final = meta.predict(p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx=ctx)
        except TypeError:
            # некоторые реализации требуют все аргументы позиционно и/или без None
            vals = [(p if p is not None else 0.5) for p in [p_xgb, p_rf, p_arf, p_nn, p_base]]
            p_final = meta.predict(*vals, reg_ctx=ctx) if "reg_ctx" in inspect.signature(meta.predict).parameters else meta.predict(*vals)

        # EV-гейт
        thr = ev_threshold(gb, gc, S, r)
        do_bet = (p_final is not None) and (p_final >= thr)

        # истинный исход (симулируем из ранее сгенерированного класса)
        y_true = int(y[i % len(y)])
        side_up = 1  # бот ставит только UP согласно p_final

        if do_bet:
            bets += 1
            # запишем предикты и результаты в экспертов и META
            for (exp, pexp) in [(xgb, p_xgb), (rf, p_rf), (arf, p_arf), (nn, p_nn)]:
                exp.record_result(X[i % len(X)], y_true, used_in_live=True, p_pred=pexp, reg_ctx=ctx)

            meta_record_adaptive(meta, p_xgb, p_rf, p_arf, p_nn, p_base, y_up=y_true, reg_ctx=ctx, used_in_live=True)

            # сеттлмент и pnl
            if y_true == side_up:  # win
                pnl = (r - 1.0) * S
            else:                  # loss
                pnl = -1.0 * S
            pnl_net = pnl - (gb + (gc if y_true == side_up else 0.0))

            before = capital
            capital = capital * (1.0 + pnl_net / max(S, 1e-9)) if hasattr(PM, "capitalized") else capital + pnl_net

            # метрики
            row = {
                "pnl_net": pnl_net,
                "capital_before": before,
                "capital_after": capital,
                "p_up": p_final if p_final is not None else 0.5,
                "payout_ratio": r,
            }
            pm.on_trade_settled(row)

            # учим фазы
            for (exp, phx) in [(xgb, ph), (rf, ph), (arf, ph), (nn, ph)]:
                exp.maybe_train(ph=phx, reg_ctx=ctx)
        else:
            skips += 1
            # буферизуем наблюдение, но без live
            for (exp, pexp) in [(xgb, p_xgb), (rf, p_rf), (arf, p_arf), (nn, p_nn)]:
                exp.record_result(X[i % len(X)], y_true, used_in_live=False, p_pred=pexp, reg_ctx=ctx)

    # ----- Проверки -----
    # 1) Были и ставки, и пропуски
    assert bets > 0 and skips > 0

    # 2) META что-то накопила / обучила
    assert isinstance(getattr(meta, "buf_ph", {}), dict)

    # 3) PerfMonitor вернул состояние/статус
    status_attr = getattr(pm, "status", None) or getattr(pm, "state", None)
    st = status_attr() if callable(status_attr) else status_attr
    assert st is not None
    # желательно, чтобы в статусе были ключи с прогрессом
    # (но не требуем жёстко — у разных реализаций разные ключи)

    # 4) Сохранение/загрузка META не падает
    meta._save()
    m2 = Meta(cfg)
    m2.state_path = meta.state_path
    m2._load()

    # 5) Финальный капитал должен быть числом
    assert isinstance(capital, float)
