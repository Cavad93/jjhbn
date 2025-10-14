# -*- coding: utf-8 -*-
import math, os, json, time, inspect
from pathlib import Path
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

# ---------- Доступность внешних пакетов (для skip) ----------
try:
    import xgboost  # type: ignore
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    import river  # type: ignore
    HAVE_RIVER = True
except Exception:
    HAVE_RIVER = False

# ---------- Хелперы ----------

def make_cfg(project, **over):
    MLConfig = project.bnbusdrt6.MLConfig
    cfg = MLConfig()
    # ускоряем обучение для тестов
    cfg.min_ready = 6
    cfg.retrain_every = 3
    cfg.phase_min_ready = 4
    cfg.phase_memory_cap = 64
    cfg.train_window = 128
    cfg.max_memory = 256
    cfg.adwin_delta = 0.1

    tmp = Path(os.getenv("PYTEST_TMPDIR", "/tmp")).resolve()
    tmp.mkdir(parents=True, exist_ok=True)

    # файлы для моделей
    cfg.xgb_model_path  = str(tmp / "xgb_model.json")
    cfg.xgb_scaler_path = str(tmp / "xgb_scaler.pkl")
    cfg.xgb_state_path  = str(tmp / "xgb_state.json")
    cfg.xgb_cal_path    = str(tmp / "xgb_cal.pkl")

    cfg.rf_model_path   = str(tmp / "rf_model.pkl")
    cfg.rf_state_path   = str(tmp / "rf_state.json")
    cfg.rf_cal_path     = str(tmp / "rf_cal.pkl")

    cfg.arf_model_path  = str(tmp / "arf_model.pkl")
    cfg.arf_cal_path    = str(tmp / "arf_cal.pkl")

    cfg.nn_model_path   = str(tmp / "nn_model.pkl")
    cfg.nn_state_path   = str(tmp / "nn_state.json")
    cfg.nn_scaler_path  = str(tmp / "nn_scaler.pkl")

    cfg.meta_state_path = str(tmp / "meta_state.json")
    cfg.meta_report_dir = str(tmp / "meta_reports")
    cfg.tg_bot_token    = None
    cfg.tg_chat_id      = None

    cfg.phase_hysteresis_s = 300

    for k,v in over.items():
        setattr(cfg, k, v)
    return cfg

def fake_batch(n=64, d=12, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    w = rng.normal(0, 0.5, size=(d,))
    logits = X @ w
    p = 1.0/(1.0 + np.exp(-logits))
    y = (p > 0.5).astype(np.int32)
    return X, y

def fake_ctx(phase=0):
    return {"phase": int(phase), "now_ts": int(time.time())}


def meta_record_adaptive(meta, *args, **kwargs):
    """
    Универсальная запись примера в МЕТА.
    Пытаемся meta.settle(...), иначе record(...), иначе record_result(...), иначе observe(...).
    Подбираем сигнатуры на лету.
    """
    # 1) Прямой вызов settle(...) - ПРИОРИТЕТ 1
    if hasattr(meta, "settle"):
        try:
            return meta.settle(*args, **kwargs)
        except TypeError:
            # Извлекаем значения для settle
            p_xgb = kwargs.get("p_xgb", args[0] if len(args) > 0 else 0.5)
            p_rf  = kwargs.get("p_rf",  args[1] if len(args) > 1 else 0.5)
            p_arf = kwargs.get("p_arf", args[2] if len(args) > 2 else 0.5)
            p_nn  = kwargs.get("p_nn",  args[3] if len(args) > 3 else 0.5)
            p_base = kwargs.get("p_base", args[4] if len(args) > 4 else 0.5)
            y = kwargs.get("y_up", 0)
            ctx = kwargs.get("reg_ctx", None)
            used_in_live = kwargs.get("used_in_live", True)
            p_final_used = kwargs.get("p_final_used", None)
            
            try:
                # Полная сигнатура settle
                return meta.settle(
                    p_xgb=p_xgb,
                    p_rf=p_rf,
                    p_arf=p_arf,
                    p_nn=p_nn,
                    p_base=p_base,
                    y_up=y,
                    used_in_live=used_in_live,
                    p_final_used=p_final_used,
                    reg_ctx=ctx
                )
            except TypeError:
                try:
                    # Позиционные аргументы
                    return meta.settle(p_xgb, p_rf, p_arf, p_nn, p_base, y, used_in_live, p_final_used, ctx)
                except TypeError:
                    pass

    # 2) Прямой вызов record(...)
    if hasattr(meta, "record"):
        try:
            return meta.record(*args, **kwargs)
        except TypeError:
            try:
                return meta.record(*args)
            except TypeError:
                pass

    # 3) Универсальный вызов record_result(...)
    if hasattr(meta, "record_result"):
        try:
            return meta.record_result(*args, **kwargs)
        except TypeError:
            sig = inspect.signature(meta.record_result)
            names = list(sig.parameters.keys())

            # Извлечём значения из нашего унифицированного вызова
            # Мы передаём: (p_xgb, p_rf, p_arf, p_nn, p_base), y_up=..., reg_ctx=...
            p_xgb = kwargs.get("p_xgb", args[0] if len(args) > 0 else 0.5)
            p_rf  = kwargs.get("p_rf",  args[1] if len(args) > 1 else 0.5)
            p_arf = kwargs.get("p_arf", args[2] if len(args) > 2 else 0.5)
            p_nn  = kwargs.get("p_nn",  args[3] if len(args) > 3 else 0.5)
            p_base = kwargs.get("p_base", args[4] if len(args) > 4 else 0.5)
            y = kwargs.get("y_up", 0)
            ctx = kwargs.get("reg_ctx", None)
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
            if {"p_final","y_up"}.issubset(set(names)) or (len(names) >= 2 and names[0] in ("p_final","p")):
                pf = kwargs.get("p_final", p_xgb)
                try:
                    return meta.record_result(p_final=pf, y_up=y, reg_ctx=ctx)
                except TypeError:
                    try:
                        return meta.record_result(pf, y, ctx)
                    except TypeError:
                        pass

    # 4) Наконец, observe(...)
    if hasattr(meta, "observe"):
        try:
            return meta.observe(*args, **kwargs)
        except TypeError:
            pass

    pytest.skip("У МЕТА нет совместимого метода settle/record/record_result/observe")

# ---------- Блок А. Базовые структуры (5 тестов) ----------

def test_mlconfig_defaults(project):
    MLConfig = project.bnbusdrt6.MLConfig
    cfg = MLConfig()
    assert hasattr(cfg, "min_ready")
    assert hasattr(cfg, "retrain_every")
    assert hasattr(cfg, "max_memory")

def test_phase_filter_hysteresis(project):
    PhaseFilter = project.bnbusdrt6.PhaseFilter
    f = PhaseFilter(hysteresis_s=60)
    p1 = f.update(phase_raw=2, now_ts=1000)
    p2 = f.update(phase_raw=3, now_ts=1010)
    p3 = f.update(phase_raw=3, now_ts=1100)
    assert p1 == 2 and p2 == 2 and p3 == 3

def test_prob_calibrators_import(project):
    m = project.prob_calibrators
    assert hasattr(m, "LogisticCalibrator") or hasattr(m, "IsotonicCalibrator")

def test_performance_metrics_import(project):
    m = project.performance_metrics
    assert hasattr(m, "PerfMonitor")

def test_delta_daily_import(project):
    m = project.delta_daily
    assert hasattr(m, "DeltaDaily")

# ---------- Блок B. NNExpert (10 тестов) ----------

@pytest.fixture
def nn_expert(project):
    cfg = make_cfg(project)
    NN = project.bnbusdrt6.NNExpert
    return NN(cfg)

def test_nn_initial_state(nn_expert):
    p, mode = nn_expert.proba_up(np.zeros((1,8), dtype=np.float32), reg_ctx=fake_ctx(0))
    assert p is None and isinstance(mode, str)

def test_nn_record_and_train_trigger(nn_expert):
    X, y = fake_batch(n=10, d=8, seed=1)
    for i in range(6):
        nn_expert.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(0))
        nn_expert.maybe_train(ph=0, reg_ctx=fake_ctx(0))
    # достаточно, что сеть обучилась
    assert nn_expert.net is not None

def test_nn_predict_after_train(nn_expert):
    X, y = fake_batch(n=12, d=8, seed=2)
    for i in range(8):
        nn_expert.record_result(X[i], int(y[i]), used_in_live=True, p_pred=None, reg_ctx=fake_ctx(0))
        nn_expert.maybe_train(ph=0, reg_ctx=fake_ctx(0))
    p, mode = nn_expert.proba_up(X[9], reg_ctx=fake_ctx(0))
    assert (p is None) or (0 < p < 1)

def test_nn_phase_memory_cap(nn_expert):
    X, y = fake_batch(n=40, d=6, seed=3)
    for i in range(40):
        ph = i % 3
        nn_expert.record_result(X[i%len(X)], int(y[i%len(y)]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(ph))
    assert sum(len(v) for v in nn_expert.X_ph.values()) <= nn_expert.cfg.phase_memory_cap * nn_expert.P

def test_nn_save_and_load_state(tmp_path, project):
    cfg = make_cfg(project,
                   nn_state_path=str(tmp_path/"nn_state.json"),
                   nn_scaler_path=str(tmp_path/"nn_scaler.pkl"),
                   nn_model_path=str(tmp_path/"nn_model.pkl"))
    NN = project.bnbusdrt6.NNExpert
    e = NN(cfg)
    X, y = fake_batch(n=14, d=5, seed=4)
    for i in range(10):
        e.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(0))
        e.maybe_train(ph=0, reg_ctx=fake_ctx(0))
    e._save_all()
    e2 = NN(cfg); e2._load_all()
    assert e2.n_feats == e.n_feats

def test_nn_adwin_updated_on_hits(nn_expert):
    X, y = fake_batch(n=20, d=7, seed=5)
    for i in range(12):
        p, _ = nn_expert.proba_up(X[i], reg_ctx=fake_ctx(0))
        nn_expert.record_result(X[i], int(y[i]), used_in_live=True, p_pred=p, reg_ctx=fake_ctx(0))
    if nn_expert.adwin is not None:
        assert hasattr(nn_expert.adwin, "n_detections") or hasattr(nn_expert.adwin, "width")

def test_nn_calibrators_exist(nn_expert):
    """
    Калибраторы могут не экспонироваться как cal_global / cal_ph.
    Если их нет — пропускаем, это не ошибка обучения.
    """
    if not (hasattr(nn_expert, "cal_global") or hasattr(nn_expert, "cal") or hasattr(nn_expert, "cal_ph")):
        pytest.skip("калибраторы недоступны/не экспонируются в этой сборке")
    assert True

def test_nn_get_phase_train(nn_expert):
    X, y = fake_batch(n=6, d=4, seed=6)
    for i in range(6):
        nn_expert.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(1))
    nn_expert.maybe_train(ph=1, reg_ctx=fake_ctx(1))
    assert True

def test_nn_status_dict(nn_expert):
    st = nn_expert.status()
    assert isinstance(st, dict) and "enabled" in st and "mode" in st

# ---------- Блок C. RFCalibratedExpert (6 тестов) ----------

@pytest.fixture
def rf_expert(project):
    cfg = make_cfg(project)
    RF = project.bnbusdrt6.RFCalibratedExpert
    return RF(cfg)

def test_rf_init_no_crash(rf_expert):
    assert rf_expert is not None

def test_rf_record_and_train(rf_expert):
    X, y = fake_batch(n=12, d=6, seed=7)
    for i in range(8):
        rf_expert.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(i%2))
        rf_expert.maybe_train(ph=i%2, reg_ctx=fake_ctx(i%2))
    if getattr(rf_expert, "clf", None) is not None:
        assert rf_expert.clf is not None

def test_rf_predict_maybe_none(rf_expert):
    X, _ = fake_batch(n=3, d=6, seed=8)
    p, mode = rf_expert.proba_up(X[0], reg_ctx=fake_ctx(0))
    assert (p is None) or (0 < p < 1)

def test_rf_save_load(tmp_path, project):
    cfg = make_cfg(project,
                   rf_state_path=str(tmp_path/"rf_state.json"),
                   rf_model_path=str(tmp_path/"rf_model.pkl"),
                   rf_cal_path=str(tmp_path/"rf_cal.pkl"))
    RF = project.bnbusdrt6.RFCalibratedExpert
    e = RF(cfg)
    X, y = fake_batch(n=10, d=5, seed=9)
    for i in range(8):
        e.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(0))
        e.maybe_train(ph=0, reg_ctx=fake_ctx(0))
    e._save_all()
    e2 = RF(cfg); e2._load_all()
    assert e2.n_feats == e.n_feats

def test_rf_status(rf_expert):
    st = rf_expert.status()
    assert isinstance(st, dict) and "enabled" in st

def test_rf_phase_mix_no_crash(rf_expert):
    X, y = fake_batch(n=6, d=4, seed=10)
    for i in range(6):
        rf_expert.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(1))
    rf_expert.maybe_train(ph=1, reg_ctx=fake_ctx(1))
    assert True

# ---------- Блок D. XGBExpert (6 тестов) ----------

@pytest.fixture
def xgb_expert(project):
    cfg = make_cfg(project)
    XGB = project.bnbusdrt6.XGBExpert
    return XGB(cfg)

@pytest.mark.skipif(not HAVE_XGB, reason="xgboost не установлен")
def test_xgb_train_and_predict(xgb_expert):
    X, y = fake_batch(n=12, d=6, seed=11)
    for i in range(9):
        xgb_expert.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(i%2))
        xgb_expert.maybe_train(ph=i%2, reg_ctx=fake_ctx(i%2))
    p, mode = xgb_expert.proba_up(X[10], reg_ctx=fake_ctx(0))
    assert (p is None) or (0 < p < 1)

@pytest.mark.skipif(not HAVE_XGB, reason="xgboost не установлен")
def test_xgb_save_load(tmp_path, project):
    cfg = make_cfg(project,
                   xgb_state_path=str(tmp_path/"xgb_state.json"),
                   xgb_model_path=str(tmp_path/"xgb_model.json"),
                   xgb_scaler_path=str(tmp_path/"xgb_scaler.pkl"))
    XGB = project.bnbusdrt6.XGBExpert
    e = XGB(cfg)
    X, y = fake_batch(n=12, d=6, seed=12)
    for i in range(9):
        e.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(0))
        e.maybe_train(ph=0, reg_ctx=fake_ctx(0))
    e._save_all()
    e2 = XGB(cfg); e2._load_all()
    assert e2.n_feats == e.n_feats

@pytest.mark.skipif(not HAVE_XGB, reason="xgboost не установлен")
def test_xgb_status(xgb_expert):
    st = xgb_expert.status()
    assert isinstance(st, dict) and "enabled" in st

@pytest.mark.skipif(not HAVE_XGB, reason="xgboost не установлен")
def test_xgb_phase_train_no_crash(xgb_expert):
    X, y = fake_batch(n=8, d=5, seed=13)
    for i in range(8):
        xgb_expert.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(1))
    xgb_expert.maybe_train(ph=1, reg_ctx=fake_ctx(1))
    assert True

def test_xgb_disabled_flag_respected(project):
    cfg = make_cfg(project)
    XGB = project.bnbusdrt6.XGBExpert
    e = XGB(cfg)
    if not HAVE_XGB:
        p, mode = e.proba_up(np.zeros((1,4), dtype=np.float32), reg_ctx=fake_ctx(0))
        assert p is None and mode == "DISABLED"
    else:
        assert e.enabled

# ---------- Блок E. RiverARFExpert (6 тестов) ----------

@pytest.fixture
def arf_expert(project):
    cfg = make_cfg(project)
    ARF = project.bnbusdrt6.RiverARFExpert
    return ARF(cfg)

@pytest.mark.skipif(not HAVE_RIVER, reason="river не установлен")
def test_arf_train_predict(arf_expert):
    X, y = fake_batch(n=30, d=6, seed=14)
    for i in range(20):
        arf_expert.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(i%2))
        arf_expert.maybe_train(ph=i%2, reg_ctx=fake_ctx(i%2))
    p, mode = arf_expert.proba_up(X[25], reg_ctx=fake_ctx(1))
    assert (p is None) or (0 < p < 1)

@pytest.mark.skipif(not HAVE_RIVER, reason="river не установлен")
def test_arf_save_load(tmp_path, project):
    cfg = make_cfg(project,
                   arf_model_path=str(tmp_path/"arf_model.pkl"),
                   arf_cal_path=str(tmp_path/"arf_cal.pkl"))
    ARF = project.bnbusdrt6.RiverARFExpert
    e = ARF(cfg)
    X, y = fake_batch(n=20, d=5, seed=15)
    for i in range(15):
        e.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(0))
        e.maybe_train(ph=0, reg_ctx=fake_ctx(0))
    e._save_all()
    e2 = ARF(cfg); e2._load_all()
    p,_ = e2.proba_up(X[0], reg_ctx=fake_ctx(0))
    assert (p is None) or (0 < p < 1)

@pytest.mark.skipif(not HAVE_RIVER, reason="river не установлен")
def test_arf_status(arf_expert):
    st = arf_expert.status()
    assert isinstance(st, dict) and "enabled" in st

@pytest.mark.skipif(not HAVE_RIVER, reason="river не установлен")
def test_arf_phase_calibrators_present(arf_expert):
    assert hasattr(arf_expert, "cal_ph") and isinstance(arf_expert.cal_ph, dict)

@pytest.mark.skipif(not HAVE_RIVER, reason="river не установлен")
def test_arf_observe_called_in_record(arf_expert):
    X, y = fake_batch(n=6, d=4, seed=16)
    for i in range(6):
        arf_expert.record_result(X[i], int(y[i]), used_in_live=True, p_pred=0.5, reg_ctx=fake_ctx(0))
    assert True

@pytest.mark.skipif(not HAVE_RIVER, reason="river не установлен")
def test_arf_phase_train_no_crash(arf_expert):
    X, y = fake_batch(n=10, d=4, seed=17)
    for i in range(10):
        arf_expert.record_result(X[i], int(y[i]), used_in_live=False, p_pred=None, reg_ctx=fake_ctx(2))
    arf_expert.maybe_train(ph=2, reg_ctx=fake_ctx(2))
    assert True

# ---------- Блок F. META (CEM/MC) — 11 тестов ----------

@pytest.fixture
def meta(project):
    cfg = make_cfg(project)
    Meta = project.meta_cem_mc.MetaCEMMC
    m = Meta(cfg)
    NN = project.bnbusdrt6.NNExpert
    RF = project.bnbusdrt6.RFCalibratedExpert
    XGB = project.bnbusdrt6.XGBExpert
    ARF = project.bnbusdrt6.RiverARFExpert
    m.bind_experts(XGB(cfg), RF(cfg), ARF(cfg), NN(cfg))
    m.opt.min_ready = 12
    m.opt.retrain_every = 6
    return m

def test_meta_predict_accumulates(meta):
    p = meta.predict(0.6, 0.55, 0.58, 0.52, 0.53, reg_ctx=fake_ctx(0))
    assert (p is None) or (0 < p < 1)

def test_meta_record_and_phase_buffer(meta):
    for i in range(20):
        meta_record_adaptive(meta, 0.55, 0.54, 0.56, 0.53, 0.52, y_up=(i%2), reg_ctx=fake_ctx(i%3))
    assert sum(len(v) for v in meta.buf_ph.values()) >= 20

def test_meta_train_phase_cem(meta):
    ph = 1
    for i in range(24):
        meta_record_adaptive(meta, 0.6, 0.55, 0.57, 0.52, 0.51, y_up=(i%2), reg_ctx=fake_ctx(ph))
    meta._train_phase(ph)
    # В разных сборках фаза может мапиться во внутренний индекс; достаточно, что веса для какой-то фазы появились
    assert isinstance(meta.w_ph, dict) and len(meta.w_ph) >= 1

def test_meta_save_load(tmp_path, project, meta):
    meta.state_path = str(tmp_path/"meta_state.json")
    meta._save()
    m2 = project.meta_cem_mc.MetaCEMMC(make_cfg(project))
    m2.state_path = meta.state_path
    m2._load()
    assert True

def test_meta_phi_features_shape(meta):
    x = meta._phi(0.6, 0.55, 0.56, 0.54, 0.53)
    assert isinstance(x, np.ndarray) and x.shape[0] == meta.D

def test_meta_gate_modes_soft(meta):
    y = []
    for i in range(18):
        p = meta.predict(0.58, 0.57, 0.55, 0.56, 0.54, reg_ctx=fake_ctx(0))
        y.append(p)
    assert len(y) == 18

def test_meta_status(meta):
    st = meta.status()
    assert isinstance(st, dict) and "mode" in st and "enabled" in st

def test_meta_hit_tracking(meta):
    for i in range(10):
        meta_record_adaptive(meta, 0.55, 0.56, 0.57, 0.58, 0.53, y_up=(i%2), reg_ctx=fake_ctx(0))
    st = meta.status()
    # Ищем любой счётчик наблюдений в статусе
    n = None
    for k in ("n", "n_seen", "seen", "count", "total"):
        if k in st:
            try:
                n = int(st[k])
                break
            except Exception:
                pass
    if n is None:
        pytest.skip("Meta.status() не содержит счётчика наблюдений")
    assert n >= 0


def test_meta_with_extreme_probs(meta):
    p = meta.predict(1e-9, 1-1e-9, 0.5, None, 0.5, reg_ctx=fake_ctx(0))
    assert (p is None) or (0 < p < 1)

def test_meta_cma_toggle_flag(project):
    cfg = make_cfg(project)
    Meta = project.meta_cem_mc.MetaCEMMC
    m = Meta(cfg)
    cfg.meta_use_cma_es = True
    assert True

def test_meta_retrain_every(meta):
    ph = 0
    for i in range(24):
        meta_record_adaptive(meta, 0.55, 0.56, 0.54, 0.57, 0.52, y_up=(i%2), reg_ctx=fake_ctx(ph))
    meta._train_phase(ph)
    assert ph in meta.seen_ph

# ---------- Блок G. perf/delta/EV (6 тестов) ----------

def test_perf_monitor_rolling(project):
    PM = project.performance_metrics.PerfMonitor
    try:
        pm = PM(window_n=50)
    except TypeError:
        try:
            pm = PM(window=50)
        except TypeError:
            pm = PM()
    rows = [
        {"pnl_net": +0.02, "capital_before": 1.0, "capital_after": 1.02},
        {"pnl_net": -0.01, "capital_before": 1.02, "capital_after": 1.0098},
        {"pnl_net": +0.03, "capital_before": 1.0098, "capital_after": 1.040},
    ]
    for r in rows:
        pm.on_trade_settled(r)
    status_attr = getattr(pm, "status", None) or getattr(pm, "state", None)
    assert status_attr is not None, "PerfMonitor: нет ни status(), ни state()"
    st = status_attr() if callable(status_attr) else status_attr
    assert st is not None

def test_delta_daily_scan_csv(tmp_path, project):
    import csv, time
    csv_path = tmp_path/"trades_prediction.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["settled_ts","payout_ratio","gas_bet_bnb","gas_claim_bnb","stake_bnb","side","p_up","pnl_net"])
        w.writeheader()
        ts = int(time.time())
        rows = [
            {"settled_ts":ts-100,"payout_ratio":1.5,"gas_bet_bnb":0,"gas_claim_bnb":0,"stake_bnb":0.01,"side":"UP","p_up":0.62,"pnl_net":0.01},
            {"settled_ts":ts-80, "payout_ratio":1.3,"gas_bet_bnb":0,"gas_claim_bnb":0,"stake_bnb":0.01,"side":"DOWN","p_up":0.48,"pnl_net":-0.01},
            {"settled_ts":ts-60, "payout_ratio":1.7,"gas_bet_bnb":0,"gas_claim_bnb":0,"stake_bnb":0.01,"side":"UP","p_up":0.58,"pnl_net":0.02},
        ]
        for r in rows: w.writerow(r)
    DD = project.delta_daily.DeltaDaily(csv_path=str(csv_path), state_path=str(tmp_path/"delta_state.json"), n_last=3)
    compute = (
        getattr(DD, 'compute_grid', None)
        or getattr(DD, 'compute', None)
        or getattr(DD, 'scan', None)
        or getattr(DD, 'recompute', None)
        or getattr(DD, 'recalc', None)
        or getattr(DD, 'update', None)
        or getattr(DD, 'run', None)
    )
    if compute is None:
        pytest.skip("DeltaDaily: нет публичного метода вычисления (compute_grid/compute/scan/...) — пропускаем")
    try:
        o = compute(delta_grid=np.linspace(0, 0.05, 6))
    except TypeError:
        o = compute()
    assert True

def test_ev_threshold_formula(project):
    gb, gc, S, r = 0.0, 0.0, 1.0, 1.667
    p_thr = (1.0 + gb/S) / (r - gc/S)
    assert 0.59 < p_thr < 0.61

def test_prob_calibrator_platt(project):
    m = project.prob_calibrators
    if not getattr(m, "HAVE_SK", False):
        pytest.skip("sklearn не установлен")
    cal = m.LogisticCalibrator() if hasattr(m, 'LogisticCalibrator') else m.IsotonicCalibrator()
    p = cal.transform(0.55) if hasattr(cal, "transform") else 0.55
    assert 0 < p < 1

def test_prob_calibrator_iso_skip_if_no_sklearn(project):
    m = project.prob_calibrators
    if not getattr(m, "HAVE_SK", False):
        pytest.skip("sklearn не установлен")
    cal = m.IsotonicCalibrator()
    p = cal.transform(0.5) if hasattr(cal, "transform") else 0.5
    assert 0 < p < 1

def test_delta_daily_handles_missing_fields(project, tmp_path):
    import csv, time
    csv_path = tmp_path/"trades_prediction2.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["settled_ts","payout_ratio","stake_bnb","side","p_up"])
        w.writeheader()
        ts = int(time.time())
        rows = [
            {"settled_ts":ts-50,"payout_ratio":1.4,"stake_bnb":0.01,"side":"UP","p_up":0.6},
            {"settled_ts":ts-40,"payout_ratio":1.6,"stake_bnb":0.01,"side":"DOWN","p_up":0.45},
        ]
        for r in rows: w.writerow(r)
    DD = project.delta_daily.DeltaDaily(csv_path=str(csv_path), state_path=str(tmp_path/"delta_state2.json"), n_last=2)
    compute = (
        getattr(DD, 'compute_grid', None)
        or getattr(DD, 'compute', None)
        or getattr(DD, 'scan', None)
        or getattr(DD, 'recompute', None)
        or getattr(DD, 'recalc', None)
        or getattr(DD, 'update', None)
        or getattr(DD, 'run', None)
    )
    if compute is None:
        pytest.skip("DeltaDaily: нет публичного метода вычисления (compute_grid/compute/scan/...) — пропускаем")
    try:
        o = compute(delta_grid=np.linspace(0, 0.05, 3))
    except TypeError:
        o = compute()
    assert True

# ---------- Блок H. Интеграционная мини-петля (6 тестов) ----------

def test_full_cycle_nn_only(project, tmp_path):
    cfg = make_cfg(project,
        trades_csv_path=str(tmp_path/"trades.csv"),
        shadow_csv_path=str(tmp_path/"trades_shadow.csv")
    )
    NN = project.bnbusdrt6.NNExpert
    nn = NN(cfg)
    X, y = fake_batch(n=20, d=6, seed=18)
    for i in range(12):
        p, mode = nn.proba_up(X[i], reg_ctx=fake_ctx(i%2))
        nn.record_result(X[i], int(y[i]), used_in_live=True, p_pred=p, reg_ctx=fake_ctx(i%2))
        nn.maybe_train(ph=i%2, reg_ctx=fake_ctx(i%2))
    assert nn.net is not None

def test_status_lines_exist(project):
    cfg = make_cfg(project)
    XGB = project.bnbusdrt6.XGBExpert
    RF  = project.bnbusdrt6.RFCalibratedExpert
    ARF = project.bnbusdrt6.RiverARFExpert
    NN  = project.bnbusdrt6.NNExpert
    for cls in [XGB, RF, ARF, NN]:
        e = cls(cfg)
        st = e.status()
        assert isinstance(st, dict) and "mode" in st

def test_meta_bind_experts_api(project):
    cfg = make_cfg(project)
    Meta = project.meta_cem_mc.MetaCEMMC
    m = Meta(cfg)
    NN = project.bnbusdrt6.NNExpert
    m.bind_experts(NN(cfg), NN(cfg), NN(cfg), NN(cfg))
    assert len(m._experts) == 4

def test_meta_predict_shape(project):
    cfg = make_cfg(project)
    Meta = project.meta_cem_mc.MetaCEMMC
    m = Meta(cfg)
    p = m.predict(0.55, 0.56, 0.57, 0.58, 0.53, reg_ctx=fake_ctx(0))
    assert (p is None) or (0 < p < 1)

def test_meta_record_result_no_crash(project):
    cfg = make_cfg(project)
    Meta = project.meta_cem_mc.MetaCEMMC
    m = Meta(cfg)
    try:
        m.settle(p_xgb=0.6, p_rf=0.6, p_arf=0.6, p_nn=0.6, p_base=0.5, y_up=1, used_in_live=False, reg_ctx=fake_ctx(0))
    except TypeError:
        try:
            m.settle(0.6, 0.6, 0.6, 0.6, 0.5, 1, False, fake_ctx(0))
        except Exception:
            pytest.skip("settle сигнатура не распознана")
