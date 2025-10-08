# -*- coding: utf-8 -*-
"""
Live paper-trading для PancakeSwap Prediction (BNB):
- Реальные rounds/тайминги с контракта Prediction V2 (BSC mainnet)
- Цены/объёмы: Binance Spot /api/v3/klines (без ключей)
- Базовая модель: фичи (Momentum/VWAP/Keltner/Bollinger/ATR-chop + Vol Z) -> softmax + EMA/Super Smoother
- NN-калибратор: онлайн логистическая регрессия (обучаем по факту исхода)

- +++ ML-АНСАМБЛЬ:
    Четыре «эксперта» выдают вероятность UP по расширенному вектору фич (см. ExtendedMLFeatures):
      1) XGBoost (+ ADWIN-гейтинг).
      2) RandomForest + CalibratedClassifierCV (sigmoid), батч-дообучение + ADWIN-гейтинг.
      3) River Adaptive Random Forest (онлайн) + ADWIN-гейтинг.
      4) NNExpert — компактная MLP (1 скрытый слой, tanh + sigmoid), батч-дообучение, калибровка температурой + ADWIN-гейтинг.
    Над ними — МЕТА-оценщик: онлайн логистическая регрессия по логитам [p_xgb,p_rf,p_arf,p_nn,p_base] + доп. статистики.

    Режимы:
      * SHADOW: все учатся/мониторятся, но в ставках НЕ участвуют.
      * ACTIVE: в ставках используется ТОЛЬКО p_final от мета-оценщика (без смешивания с базой).
    Персист: модели/состояния/скейлеры/веса.

- Менеджмент: старт 2 BNB, ставка:
    * первые 500 сделок — фикс. 1% капитала;
    * далее — 1/2 Kelly, но ограничено 0.5%..3% и кэп на раунд ≤3% капитала.
- Газ/учёт/телега/EV-гейт: без изменений.
- EV-порог p_thr:
    * p_thr = 0.51, пока нет 500 закрытых сделок ИЛИ если нет ни одной закрытой сделки за последний час;
    * иначе — классический EV-порог с учётом газа и payout.

Примечание по зависимостям:
- xgboost — опционально (эксперт XGB).
- scikit-learn — для RandomForest + калибровка (эксперт RF) и StandardScaler.
- river — для ADWIN и ARF (эксперт ARF и гейтинг).
Если либы отсутствуют — соответствующий эксперт/гейтинг будет отключён, остальные работают.
"""

import os
import csv
import math
import time
import json
import pickle

# --- numeric guards ---
def _is_finite_num(x) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False

def _as_float(x, default=float("nan")):
    """Безопасная конвертация в float с обработкой None, pd.NA, np.nan"""
    try:
        # Явная проверка на None и pd.NA
        if x is None:
            return default
        # Проверка на pandas NA (для совместимости с pandas < 2.0)
        if hasattr(x, '__class__') and x.__class__.__name__ == 'NAType':
            return default
        
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from gating_no_r import compute_p_thr_no_r
from no_r_auto import pick_no_r_mode
from delta_daily import DeltaDaily

from proj_scenarios import try_send_projection

from html import escape
from requests import RequestException
# вверху bnbusdrt6.py
from prob_calibrators import make_calibrator, _BaseCal

from performance_metrics import PerfMonitor

# === НОВОЕ: контекстная калибровка p и r̂-таблица, EV-гейт по «марже к рынку» ===
from ctx_calibration import p_ctx_calibrated
from rhat_quantile2d import RHat2D
from ev_margin_gate import loss_margin_q, p_thr_from_ev

from error_logger import setup_error_logging, log_exception, get_logger

from dotenv import load_dotenv; load_dotenv()

# инициализируем отдельный error-лог (GGG/errors.log)
setup_error_logging(log_dir=".", filename="errors.log")



def _proj_mark_once(path: str, day: str) -> bool:
    import json, os
    try:
        st = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                st = json.load(f)
        if st.get("last_day") == day:
            return False
        st["last_day"] = day
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st, f)
    except Exception:
        pass
    return True


from daily_report import try_send as try_send_daily

# /report: лёгкий слушатель команд
from report_cmd import start_report_listener

from datetime import datetime, timezone
# --- NEW: addons ---
from microstructure import MicrostructureClient


# --- FIX: совместимый импорт ZoneInfo для Python 3.8/3.9+ ---
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    try:
        from backports.zoneinfo import ZoneInfo  # для Python < 3.9
    except Exception:
        ZoneInfo = None  # fallback, если ни один импорт не удался

def _get_proj_tz():
    # стараемся вернуть Europe/Berlin; если нет базы часовых поясов — откатываемся на UTC
    if ZoneInfo is not None:
        try:
            return ZoneInfo("Europe/Berlin")
        except Exception:
            pass
    # предупреждение можно убрать, если не нужно
    print("[proj] warning: tz database unavailable; using UTC")
    return timezone.utc


rpc_fail_streak = 0
RPC_FAIL_MAX = 3

# часовой пояс для ежедневной проекции и файл-маркер "раз в день"
PROJ_TZ = _get_proj_tz()
PROJ_STATE_PATH = os.path.join(os.path.dirname(__file__), "proj_state.json")

from futures_ctx import FuturesContext
from pool_features import PoolFeaturesCtx
from extra_features import realized_metrics, jump_flag_from_rv_bv_rq, amihud_illiq, kyle_lambda
from extra_features import intraday_time_features, idio_features, GasHistory, pack_vector

from r_hat_improved import (
    estimate_r_hat_improved,
    analyze_r_hat_accuracy,
    adaptive_quantile
)

import requests


def fmtf(x, nd=4, dash="—"):
    """Безопасно форматирует число с nd знаками после точки или возвращает '—'."""
    try:
        if x is None:
            return dash
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return dash
        return f"{xf:.{nd}f}"
    except Exception:
        return dash

def fmt_pct(x, nd=2, dash="—"):
    """Проценты: 12.34% или '—'."""
    s = fmtf(x, nd=nd, dash=dash)
    return s if s == dash else f"{s}%"

def update_capital_atomic(capital_state, new_capital: float, ts: int, csv_row: dict) -> float:
    """
    Атомарно обновляет капитал и сохраняет строку в CSV.
    Гарантирует согласованность: сначала капитал, потом CSV.
    Если что-то пойдет не так, возвращает последний сохраненный капитал.
    """
    try:
        # Сначала сохраняем капитал атомарно через временный файл
        temp_path = capital_state.path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump({"capital": new_capital, "ts": ts}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, capital_state.path)
        
        # Только после успешного сохранения капитала пишем в CSV
        try:
            append_csv_row(CSV_PATH, csv_row)
        except Exception as e:
            print(f"[csv ] write failed but capital saved: {e}")
        
        return new_capital
    except Exception as e:
        print(f"[capital] save failed: {e}")
        # В случае ошибки возвращаем последнее корректное значение
        return capital_state.load()


def fmt_prob(x):
    """Вероятности p∈[0,1] до 4 знаков или '—'."""
    return fmtf(x, nd=4)




def _tail_df(path, n=300):
    try:
        df = _read_csv_df(path)  # у тебя уже есть этот ридер
        df = df.dropna(subset=["outcome"])
        return df.sort_values("settled_ts").tail(n).copy()
    except Exception:
        return None

# bnbusdrt6.py
def rolling_calib_error(path: str, n: int = 200) -> float:
    """Средняя |y - p_side| по последним n сеттлам как прокси ECE/Brier."""
    df = _tail_df(path, n)
    if df is None or df.empty:
        return 0.10
    # безопасная типизация и фильтрация
    side = df.get("side")
    p_up = pd.to_numeric(df.get("p_up"), errors="coerce")
    y = (df.get("outcome") == "win").astype(float)

    mask = side.astype(str).str.upper().isin(["UP", "DOWN"]) & p_up.notna() & y.notna()
    if not mask.any():
        return 0.10

    side_u = side[mask].astype(str).str.upper()
    p_u = p_up[mask].astype(float).to_numpy()
    y_u = y[mask].astype(float).to_numpy()

    p_side = np.where(side_u == "UP", p_u, 1.0 - p_u)
    p_side = np.clip(p_side, 1e-6, 1 - 1e-6)

    err = np.abs(y_u - p_side)
    m = float(np.mean(err)) if np.isfinite(err).all() else float(np.nanmean(err))
    if not math.isfinite(m):
        return 0.10
    return m


def realized_sigma_g(path: str, n: int = 200) -> float:
    """Стд.кв. лог-роста на сделку по последним n."""
    df = _tail_df(path, n)
    if df is None or df.empty: return 0.01
    cb = pd.to_numeric(df["capital_before"], errors="coerce").to_numpy()
    ca = pd.to_numeric(df["capital_after"],  errors="coerce").to_numpy()
    mask = np.isfinite(cb) & np.isfinite(ca) & (cb>0) & (ca>0)
    if not np.any(mask): return 0.01
    g = np.log(ca[mask] / cb[mask])
    return float(np.std(g, ddof=1))


# --- end helpers ---



# +++ ДОБАВЛЕНО для проверок целостности:
from state_safety import (
    atomic_save_json, safe_load_json, sane_vec, sane_prob,
    file_sha256, atomic_write_bytes
)


_TG_FAILS = 0  # счётчик подряд неудачных отправок (чтобы не вешать бота)

_TG_MUTED_UNTIL = 0.0    # unix-ts до которого молчим
_TG_LAST_ERR = ""        # последняя причина

from dataclasses import dataclass

# NEW: контекст для меты
from meta_ctx import build_regime_ctx, pack_ctx

# --- добавили для финальной пост-калибровки мета-вероятности ---
from collections import deque
from calib.selector import CalibratorSelector  # <— наш селектор калибратора

from meta_cem_mc import MetaCEMMC, LambdaMARTMetaLite, ProbBlender  # ← NEW

import numpy as np
import pandas as pd
from web3 import Web3, HTTPProvider
try:
    from web3.middleware import geth_poa_middleware  # для BSC/PoA
    HAVE_POA = True
except Exception:
    HAVE_POA = False
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Telegram config (globals) ---
# --- Telegram config (globals) ---
from typing import Final
import threading  # ← добавили


# --- буферы «сырых» p_meta и исходов для окна калибровки ---
_CALIB_P_META = deque(maxlen=20000)  # p_meta_raw до калибровки
_CALIB_Y_META = deque(maxlen=20000)  # outcome: 1=win, 0=loss

TG_TOKEN: Final[str] = os.getenv("TG_TOKEN", "").strip()
# Важно: для чатов/каналов ID может быть отрицательным (например, -100...).
def _env_int(name: str, default: int = 0) -> int:
    try:
        raw = os.getenv(name, str(default)).strip()
        return int(raw) if raw else default
    except Exception:
        return default

TG_CHAT_ID: Final[int] = _env_int("TG_CHAT_ID", 0)
TG_API: Final[str] = f"https://api.telegram.org/bot{TG_TOKEN}"

_REPORT_THREAD = None  # поток слушателя /report; поднимаем максимум один


# Сессия с ретраями (чтобы tg_send был устойчивее)
SESSION = requests.Session()
_adapter = HTTPAdapter(
    max_retries=Retry(
        total=3,                # 3 повторных
        backoff_factor=0.3,     # 0.3s, 0.6s, 1.2s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"])
    )
)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)

# Вспомогательная проверка — «включён ли» Telegram
# =============================
# Telegram
# =============================
def tg_enabled() -> bool:
    # включено + есть реальные реквизиты (из ENV или констант)
    has_token = bool((TG_TOKEN if 'TG_TOKEN' in globals() else "") or
                     (TELEGRAM_BOT_TOKEN if 'TELEGRAM_BOT_TOKEN' in globals() else ""))
    has_chat  = bool((TG_CHAT_ID if 'TG_CHAT_ID' in globals() else 0) or
                     (TELEGRAM_CHAT_ID if 'TELEGRAM_CHAT_ID' in globals() else ""))
    return bool(TELEGRAM_ENABLED and has_token and has_chat)




from wr_pnl_tracker import StatsTracker, RestState, RestConfig
from reserve_fund import ReserveFund
import math
from requests.exceptions import Timeout as ReqTimeout, ReadTimeout, ConnectionError as ReqConnError
from web3.exceptions import TimeExhausted


# =============================
# ML зависимости (опционально)
# =============================
HAVE_XGB = False
HAVE_RIVER = False
HAVE_SKLEARN = False
try:
    import xgboost as xgb  # бустинг + загрузка/сохранение
    HAVE_XGB = True
except Exception:
    pass

try:
    from river.drift import ADWIN  # детектор дрейфа
    from river import forest as river_forest
    HAVE_RIVER = True
except Exception:
    ADWIN = None
    river_forest = None
    HAVE_RIVER = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except Exception:
    # sklearn может отсутствовать — даём безопасную "заглушку" StandardScaler
    class StandardScaler:
        def fit(self, X, y=None): return self
        def transform(self, X):    return X
        def fit_transform(self, X, y=None): return X
    HAVE_SKLEARN = False

# -----------------------------
# ПАРАМЕТРЫ
# -----------------------------
START_CAPITAL_BNB = 2.0
BET_FRACTION = 0.01  # legacy

SYMBOL = "BNBUSDT"
BINANCE_INTERVAL = "1m"
BINANCE_LIMIT = 1000

CSV_PATH = "trades_prediction.csv"
DELTA_STATE_PATH = "delta_state.json"   # ← новое
CSV_SHADOW_PATH = "trades_shadow.csv"   # ← добавили здесь (нужно при init DeltaDaily)
MIN_TRADES_FOR_DELTA = 50  # Минимум сделок для расчета delta

# Анти-спам/таймаут ожидания oracleCalled
MAX_WAIT_POLLS = 20
WAIT_PRINT_EVERY = 5

# «болото» ATR
ATR_LEN = 14
ATR_SMOOTH = 50
USE_PCT_CHOP = True
CHOP_PCT = 20.0
CHOP_RATIO = 0.6

# Фичи
M1, M2, M3 = 1, 3, 5
VWAP_LOOK = 10
KC_LEN = 20
KC_MULT = 2.0
BB_LEN = 20
BB_Z = 1.2
VOL_LEN = 50
VOL_BOOST = 0.15
RB_LEN = 20
USE_LORENTZ = True
C_M, C_S, C_B, C_R = 2.5, 3.0, 2.0, 1.6

# Сглаживание вероятностей
SMOOTH_N = 8
USE_SUPER_SMOOTHER = True
SS_LEN = 8

# --- OU ДОБАВКИ ---
OU_SKEW_USE = True
OU_SKEW_DT_UNIT = 60.0
OU_SKEW_DECAY = 0.997
OU_SKEW_THR = 0.15
OU_SKEW_LAMBDA_MAX = 0.45
OU_SKEW_Z_CLIP = 3.0

LOGIT_OU_USE = True
LOGIT_OU_HALF_LIFE_SEC = 120.0
LOGIT_OU_MU_BETA = 0.985
LOGIT_OU_Z_CLIP = 5.5
# -------------------------------------

# NN (логистическая регрессия калибратор)
NN_USE = True
ETA = 0.02
L2 = 0.002
BLEND_NN = 0.35
W_CLIP = 10.0
G_CLIP = 1.0

# Walk-Forward веса
WF_USE = True
WF_ETA = 0.02
WF_L2 = 0.003
WF_G_CLIP = 1.0
WF_W_CLIP = 7.0
WF_WEIGHTS_PATH = "wf_weights.json"
WF_INIT_W = [0.35, 0.20, 0.20, 0.25]

# Triple Screen Элдера
ELDER_HTF = "15min"
ELDER_MID = "5min"
ELDER_ALPHA = 0.60
STOCH_LEN = 14
STOCH_OS = 20.0
STOCH_OB = 80.0

# Газ
GAS_USED_BET = 93_132
GAS_USED_CLAIM = 86_500

# Трежери-фии
TREASURY_FEE = 0.03

# --- Новое: тайминг и защитные параметры
GUARD_SECONDS   = 30      # решаем судьбу ставки только в последние 15с до lock
SEND_WIN_LOW    = 12      # «окно отправки»: нижняя граница (для реальной торговли; сейчас paper)
SEND_WIN_HIGH   = 8       # верхняя граница (для реальной торговли; сейчас paper)
DELTA_PROTECT   = 0.04    # δ — страховой зазор поверх EV-порога
USE_STRESS_R15  = True    # использовать стресс по медианному притоку за 15с





# Коррелированные активы
USE_CROSS_ASSETS = True
CROSS_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
STABLE_SYMBOLS = ["USDCUSDT", "FDUSDUSDT", "TUSDUSDT"]
CROSS_SHIFT_BARS = 0
CROSS_ALPHA = 0.50
CROSS_W_MOM = 0.18
CROSS_W_VWAP = 0.12
STABLE_W_MOM = 0.06
STABLE_W_VWAP = 0.04

# СТАВКА: жёсткий кэп на раунд
MAX_STAKE_FRACTION = 0.01  # ≤3% капитала

TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TG_CHAT_ID", "").strip()

TELEGRAM_ENABLED   = True

TG_MUTE_AFTER      = 3       # после скольких фейлов уходим в mute
TG_COOLDOWN_S      = 300     # базовый кулдаун (сек) до следующей пробы
TG_PROBE_EVERY_S   = 30      # в mute: как часто «прощупывать» линию

# мост к старым именам, которые использует tg_send()
TG_TOKEN = TELEGRAM_BOT_TOKEN
TG_CHAT_ID = TELEGRAM_CHAT_ID
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"

# RPC
# RPC
RPC_URLS = [
    "https://bsc-dataseed.bnbchain.org",
    "https://bsc-dataseed1.bnbchain.org",
    "https://bsc-dataseed2.bnbchain.org",
]
RPC_REQUEST_KW = {"timeout": 8}  # короче, чем прежние 20s — меньше зависаний

PREDICTION_ADDR = Web3.to_checksum_address("0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA")

PREDICTION_ABI = json.loads(r"""
[
  {"inputs":[],"name":"currentEpoch","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
  {"inputs":[{"internalType":"uint256","name":"epoch","type":"uint256"}],"name":"rounds","outputs":[
    {"internalType":"uint256","name":"epoch","type":"uint256"},
    {"internalType":"uint256","name":"startTimestamp","type":"uint256"},
    {"internalType":"uint256","name":"lockTimestamp","type":"uint256"},
    {"internalType":"uint256","name":"closeTimestamp","type":"uint256"},
    {"internalType":"int256","name":"lockPrice","type":"int256"},
    {"internalType":"int256","name":"closePrice","type":"int256"},
    {"internalType":"uint256","name":"lockOracleId","type":"uint256"},
    {"internalType":"uint256","name":"closeOracleId","type":"uint256"},
    {"internalType":"uint256","name":"totalAmount","type":"uint256"},
    {"internalType":"uint256","name":"bullAmount","type":"uint256"},
    {"internalType":"uint256","name":"bearAmount","type":"uint256"},
    {"internalType":"uint256","name":"rewardBaseCalAmount","type":"uint256"},
    {"internalType":"uint256","name":"rewardAmount","type":"uint256"},
    {"internalType":"bool","name":"oracleCalled","type":"bool"}
  ],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"intervalSeconds","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"bufferSeconds","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"minBetAmount","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]
""")

# =============================
# HTTP session с ретраями
# =============================
def make_requests_session():
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

SESSION: requests.Session = make_requests_session()

import atexit
atexit.register(lambda: SESSION.close())


# =============================
# УТИЛИТЫ Web3 / Binance
# =============================
# =============================
# … Web3 / Binance
# =============================
def connect_web3() -> Web3:
    for url in RPC_URLS:
        # короче таймаут и единая конфигурация
        w3 = Web3(HTTPProvider(url, request_kwargs=RPC_REQUEST_KW))
        try:
            # для BSC (PoA) важно вставить middleware
            if HAVE_POA:
                try:
                    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                except Exception:
                    # если уже вставлен — игнорируем
                    pass

            ok = False
            if hasattr(w3, "is_connected"):
                ok = w3.is_connected()
            elif hasattr(w3, "isConnected"):
                ok = w3.isConnected()
            if ok:
                return w3
        except Exception:
            pass
    raise RuntimeError("не удалось подключиться к BSC RPC")



def connect_web3_resilient(retries=9999):
    delay = 1.0
    for _ in range(retries):
        try:
            return connect_web3()
        except Exception as e:
            print(f"[init] RPC connect failed: {e}; retrying in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 1.7, 30)
    raise RuntimeError("RPC connect: exhausted retries")

def get_gas_price_wei(w3: Web3) -> int:
    return w3.eth.gas_price

def get_prediction_contract(w3: Web3):
    return w3.eth.contract(address=PREDICTION_ADDR, abi=PREDICTION_ABI)

def get_min_bet_bnb(c) -> float:
    try:
        wei = int(c.functions.minBetAmount().call())
        return wei / 1e18
    except Exception:
        return 0.0

def binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = dict(symbol=symbol, interval=interval, startTime=start_ms, endTime=end_ms, limit=BINANCE_LIMIT)
    out = []
    while True:
        r = SESSION.get(url, params=params, timeout=20)
        if r.status_code == 400:
            return pd.DataFrame()
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out += rows
        last_open = rows[-1][0]
        if last_open >= end_ms or len(rows) < BINANCE_LIMIT:
            break
        params["startTime"] = last_open + 1
        time.sleep(0.2)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "qav","trades","taker_base","taker_quote","ignore"
    ])
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df[["open_time","close_time","open","high","low","close","volume"]].set_index("close_time")

def ensure_klines_cover(df: Optional[pd.DataFrame], symbol: str, interval: str, need_until_ms: int, back_hours: int = 8) -> pd.DataFrame:
    if df is not None and not df.empty:
        have_last_ms = int(df.index[-1].timestamp() * 1000)
        if have_last_ms >= need_until_ms - 30_000:
            return df
    end_ms = need_until_ms
    start_ms = end_ms - back_hours * 3600 * 1000
    new_df = binance_klines(symbol, interval, start_ms, end_ms)
    return new_df

def ensure_klines_cover_map(df_map: Dict[str, Optional[pd.DataFrame]], symbols: List[str], interval: str, need_until_ms: int, back_hours: int = 8) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            cur = df_map.get(s)
            out[s] = ensure_klines_cover(cur, s, interval, need_until_ms, back_hours)
        except Exception:
            out[s] = pd.DataFrame()
    return out

def nearest_close_price_ms(symbol: str, ts_ms: int) -> Optional[float]:
    df = binance_klines(symbol, "1m", ts_ms - 3*60_000, ts_ms + 2*60_000)
    if df is None or df.empty:
        return None
    tgt = pd.to_datetime(ts_ms, unit="ms", utc=True)
    i = df.index.get_indexer([tgt], method="pad")[0]
    if i == -1:
        return float(df["close"].iloc[0])
    return float(df["close"].iloc[i])

# =============================
# Техиндикаторы/фичи
# =============================
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()

def stdev(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).std(ddof=0)

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": (df["high"] - prev_close).abs(),
        "lc": (df["low"] - prev_close).abs()
    }).max(axis=1)
    return tr

def rma(x: pd.Series, n: int) -> pd.Series:
    alpha = 1.0 / float(n)
    r = x.copy()
    if len(x) == 0:
        return x
    r.iloc[0] = x.iloc[:n].mean() if len(x) >= n else x.iloc[0]
    for i in range(1, len(x)):
        r.iloc[i] = alpha * x.iloc[i] + (1 - alpha) * r.iloc[i-1]
    return r

def atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    return rma(true_range(df), n)

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return ema(true_range(df), n)

def lorentz(x: np.ndarray, c: float) -> np.ndarray:
    return x / (1.0 + (x / c) ** 2)

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def norm_feat(x: np.ndarray, gain: float, c: float, use_lorentz: bool) -> np.ndarray:
    return lorentz(x, c) if use_lorentz else _tanh(gain * x)

def sigmoid(x: float) -> float:
    x = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))

def softmax2(z_up: float, z_dn: float) -> Tuple[float, float]:
    m = max(z_up, z_dn)
    e_up = math.exp(z_up - m)
    e_dn = math.exp(z_dn - m)
    s = e_up + e_dn
    return e_up / s, e_dn / s

def session_vwap(df: pd.DataFrame, src: pd.Series) -> pd.Series:
    day = df.index.tz_convert("UTC").date
    grp = pd.Series(day, index=df.index)
    pv = (src * df["volume"]).groupby(grp).cumsum()
    vv = (df["volume"]).groupby(grp).cumsum().replace(0.0, np.nan)
    vwap = (pv / vv).ffill()
    return vwap

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff().fillna(0.0)
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    rs = rma(up, n) / (rma(dn, n) + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def ema_pair_spread(series: pd.Series, fast: int, slow: int) -> pd.Series:
    return ema(series, fast) - ema(series, slow)

class EhlersSuperSmoother:
    def __init__(self, period: int):
        self.period = max(3, int(period))
        a1 = math.exp(-math.sqrt(2.0) * math.pi / self.period)
        self.c2 = 2.0 * a1 * math.cos(math.sqrt(2.0) * math.pi / self.period)
        self.c3 = -a1 * a1
        self.c1 = 1.0 - self.c2 - self.c3
        self.y1 = None
        self.y2 = None
        self.x1 = None

    def update(self, x: float) -> float:
        if self.y1 is None:
            self.y1 = x
            self.y2 = x
            self.x1 = x
            return x
        y = self.c1 * 0.5 * (x + (self.x1 if self.x1 is not None else x)) + self.c2 * self.y1 + self.c3 * self.y2
        self.y2, self.y1 = self.y1, y
        self.x1 = x
        return float(y)

# ========= OU HELPERS =========
def _phi_a_from_ols(n: float, Sx: float, Sy: float, Sxx: float, Sxy: float) -> Tuple[Optional[float], Optional[float]]:
    den = (n * Sxx - Sx * Sx)
    if den <= 1e-12:
        return None, None
    phi = (n * Sxy - Sx * Sy) / den
    a = (Sy - phi * Sx) / n
    return phi, a

class OUOnlineSkew:
    def __init__(self, dt_unit: float = 60.0, decay: float = 0.997):
        self.dt_unit = float(dt_unit)
        self.decay = float(decay)
        self.n = 0.0
        self.Sx = 0.0
        self.Sy = 0.0
        self.Sxx = 0.0
        self.Sxy = 0.0
        self.Syy = 0.0
        self.last_x = None

    def update_pair(self, x_prev: float, x_now: float):
        d = self.decay
        self.n = d * self.n + 1.0
        self.Sx = d * self.Sx + x_prev
        self.Sy = d * self.Sy + x_now
        self.Sxx = d * self.Sxx + x_prev * x_prev
        self.Sxy = d * self.Sxy + x_prev * x_now
        self.Syy = d * self.Syy + x_now * x_now
        self.last_x = x_now

    def _params(self) -> Optional[Tuple[float, float, float]]:
        if self.n < 20:
            return None
        phi, a = _phi_a_from_ols(self.n, self.Sx, self.Sy, self.Sxx, self.Sxy)
        if phi is None:
            return None
        phi = float(np.clip(phi, 1e-6, 0.999999))
        a = float(a)
        sse_over_n = (self.Syy
                      - 2.0 * a * self.Sy
                      - 2.0 * phi * self.Sxy
                      + 2.0 * a * phi * self.Sx
                      + (a * a) * self.n
                      + (phi * phi) * self.Sxx) / max(1.0, self.n)
        var_eps = max(1e-8, float(sse_over_n))
        kappa = -math.log(phi) / self.dt_unit
        if not math.isfinite(kappa) or kappa <= 0:
            return None
        mu = a / (1.0 - phi)
        denom = (1.0 - math.exp(-2.0 * kappa * self.dt_unit))
        sigma2 = max(1e-12, 2.0 * kappa * var_eps / max(1e-12, denom))
        return kappa, mu, sigma2

    def prob_above_zero(self, x_now: float, horizon_sec: float) -> Optional[Tuple[float, float]]:
        pars = self._params()
        if pars is None:
            return None
        kappa, mu, sigma2 = pars
        dt = max(0.0, float(horizon_sec))
        expk = math.exp(-kappa * dt)
        m = mu + (x_now - mu) * expk
        v = (sigma2 / (2.0 * kappa)) * (1.0 - math.exp(-2.0 * kappa * dt))
        s = max(1e-12, math.sqrt(v))
        z = (0.0 - m) / s
        p = 0.5 * math.erfc(z / math.sqrt(2.0))
        strength = 1.0 - expk
        return float(np.clip(p, 1e-6, 1.0 - 1e-6)), float(np.clip(strength, 0.0, 1.0))

class LogitOUSmoother:
    def __init__(self, half_life_sec: float = 120.0, mu_beta: float = 0.985, z_clip: float = 5.5):
        self.kappa = math.log(2.0) / max(1e-3, half_life_sec)
        self.mu = 0.0
        self.beta = float(np.clip(mu_beta, 0.0, 1.0))
        self.z_clip = float(z_clip)

    def update_mu(self, z_now: float):
        self.mu = self.beta * self.mu + (1.0 - self.beta) * z_now

    def predict_future(self, z_now: float, horizon_sec: float) -> float:
        dt = max(0.0, float(horizon_sec))
        expk = math.exp(-self.kappa * dt)
        z_now = float(np.clip(z_now, -self.z_clip, self.z_clip))
        z_pred = self.mu + (z_now - self.mu) * expk
        return float(np.clip(z_pred, -self.z_clip, self.z_clip))

# =============================

@dataclass
class RoundInfo:
    epoch: int
    start_ts: int
    lock_ts: int
    close_ts: int
    lock_price: float
    close_price: float
    bull_amount: float
    bear_amount: float
    reward_base: float
    reward_amt: float
    oracle_called: bool

    @property
    def payout_ratio(self) -> Optional[float]:
        if self.oracle_called and self.reward_base > 0:
            return self.reward_amt / self.reward_base
        return None

class OnlineLogReg:
    def __init__(self, eta=ETA, l2=L2, w_clip=W_CLIP, g_clip=G_CLIP, state_path: str = "calib_logreg_state.json"):
        self.w = np.zeros(5, dtype=float)  # 4 features + bias
        self.eta = eta
        self.l2 = l2
        self.w_clip = w_clip
        self.g_clip = g_clip
        self.state_path = state_path
        self._load()

    def _load(self):
        try:
            with open(self.state_path, "r") as f:
                obj = json.load(f)
            w = obj.get("w", [])
            if isinstance(w, list) and len(w) == len(self.w):
                self.w = np.array(w, dtype=float)
        except Exception:
            pass

    def save(self):
        try:
            with open(self.state_path, "w") as f:
                json.dump({"w": self.w.tolist()}, f)
        except Exception:
            pass

    def predict(self, phi: np.ndarray) -> float:
        z = float(np.dot(self.w, phi))
        return sigmoid(z)

    def update(self, phi: np.ndarray, y: float):
        p = self.predict(phi)
        g = p - y
        g = max(min(g, self.g_clip), -self.g_clip)
        grad = g * phi + self.l2 * self.w
        self.w -= self.eta * grad
        self.w = np.clip(self.w, -self.w_clip, self.w_clip)

class WalkForwardWeighter:
    def __init__(self, eta=WF_ETA, l2=WF_L2, g_clip=WF_G_CLIP, w_clip=WF_W_CLIP, path=WF_WEIGHTS_PATH):
        self.eta = eta
        self.l2 = l2
        self.g_clip = g_clip
        self.w_clip = w_clip
        self.path = path
        self.w = np.array(WF_INIT_W, dtype=float)
        self.load()

    def load(self):
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
                if "w" in data and len(data["w"]) == 4:
                    self.w = np.array(data["w"], dtype=float)
        except Exception:
            pass

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump({"w": self.w.tolist()}, f)
        except Exception:
            pass

    def predict_prob(self, phi_diff: np.ndarray) -> float:
        z = float(np.dot(self.w, phi_diff))
        return sigmoid(z)

    def update(self, phi_diff: np.ndarray, y_up: float):
        p = self.predict_prob(phi_diff)
        g = p - y_up
        g = max(min(g, self.g_clip), -self.g_clip)
        grad = g * phi_diff + self.l2 * self.w
        self.w -= self.eta * grad
        self.w = np.clip(self.w, -self.w_clip, self.w_clip)

# --------- извлечение rounds ----------
# --- BSC Prediction helpers ---
def get_round(w3: Web3, c, epoch: int, retries: int = 2) -> Optional[RoundInfo]:
    """
    Надёжный вызов rounds(epoch) с короткими ретраями и backoff.
    Возвращает RoundInfo или None (чтобы цикл мог пропустить раунд при RPC-проблемах).
    """
    for i in range(retries + 1):
        try:
            r = c.functions.rounds(epoch).call()
            return RoundInfo(
        epoch=int(r[0]),
        start_ts=int(r[1]),
        lock_ts=int(r[2]),
        close_ts=int(r[3]),
        lock_price=float(r[4]),
        close_price=float(r[5]),
        bull_amount=float(r[9]),
        bear_amount=float(r[10]),
        reward_base=float(r[11]),
        reward_amt=float(r[12]),
        oracle_called=bool(r[13])
            )
        except (ReqTimeout, ReadTimeout, ReqConnError, TimeExhausted) as e:
            backoff = 0.5 * (2 ** i)
            print(f"[rpc ] timeout on rounds({epoch}), retry in {backoff:.1f}s ({e.__class__.__name__})")
            time.sleep(backoff)
        except KeyboardInterrupt:
            # немедленно пробрасываем — без лишних сетевых попыток
            raise
        except Exception as e:
            print(f"[rpc ] error on rounds({epoch}): {e}")
            break
    return None  # сигнал наверх: пропускаем раунд

def get_current_epoch(w3, c):
    return c.functions.currentEpoch().call()

# --------- фичи из свечей ----------
def features_from_binance(df: pd.DataFrame) -> Dict[str, pd.Series]:
    ln_hl = np.log(df["high"] / df["low"]).clip(lower=1e-12)
    sigP = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (ln_hl ** 2))
    ln_co = np.log(df["close"] / df["open"]).fillna(0.0)
    sigGK = np.sqrt(np.maximum(0.0, 0.5 * (ln_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (ln_co ** 2)))
    ln_hc = np.log(df["high"] / df["close"]).clip(lower=1e-12)
    ln_ho = np.log(df["high"] / df["open"]).clip(lower=1e-12)
    ln_lc = np.log(df["low"] / df["close"]).clip(lower=1e-12)
    ln_lo = np.log(df["low"] / df["open"]).clip(lower=1e-12)
    rsVar = ln_hc * ln_ho + ln_lc * ln_lo
    sigRS = np.sqrt(np.maximum(0.0, rsVar))
    sigRB = pd.Series((sigP + sigGK + sigRS) / 3.0, index=df.index)
    sigRB = ema(sigRB, RB_LEN)
    normGain = 1.0 / np.maximum(sigRB.values, 1e-10)

    atr_series = atr_wilder(df, ATR_LEN)
    atr_sma = sma(atr_series, ATR_SMOOTH)

    r1 = np.log(df["close"] / df["close"].shift(M1)).fillna(0.0)
    r2 = np.log(df["close"] / df["close"].shift(M2)).fillna(0.0)
    r3 = np.log(df["close"] / df["close"].shift(M3)).fillna(0.0)
    Mraw = 0.6 * r1 + 0.3 * r2 + 0.1 * r3
    M_up = norm_feat(Mraw.values * normGain, 2.5, C_M, USE_LORENTZ)
    M_dn = norm_feat(-Mraw.values * normGain, 2.5, C_M, USE_LORENTZ)

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap = session_vwap(df, tp)
    vslp = ((vwap - vwap.shift(VWAP_LOOK)) / vwap.replace(0.0, np.nan)).fillna(0.0)
    S_up = norm_feat(vslp.values * normGain, 3.0, C_S, USE_LORENTZ)
    S_dn = norm_feat(-vslp.values * normGain, 3.0, C_S, USE_LORENTZ)

    basisKC = ema(df["close"], KC_LEN)
    rngKC = atr_wilder(df, KC_LEN)
    upKC = basisKC + KC_MULT * rngKC
    dnKC = basisKC - KC_MULT * rngKC
    distUp = ((df["close"] - upKC) / (KC_MULT * rngKC.replace(0, np.nan))).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    distDn = ((dnKC - df["close"]) / (KC_MULT * rngKC.replace(0, np.nan))).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    B_up = norm_feat(distUp.values, 2.0, C_B, USE_LORENTZ)
    B_dn = norm_feat(distDn.values, 2.0, C_B, USE_LORENTZ)

    bb_basis = sma(df["close"], BB_LEN)
    bb_dev = stdev(df["close"], BB_LEN).replace(0.0, np.nan)
    Zs = ((df["close"] - bb_basis) / bb_dev).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    R_up = norm_feat(np.maximum(0.0, -Zs - BB_Z).values, 1.6, C_R, USE_LORENTZ)
    R_dn = norm_feat(np.maximum(0.0,  Zs - BB_Z).values, 1.6, C_R, USE_LORENTZ)

    vol_usd = df["volume"] * df["close"]
    vMean = sma(vol_usd, VOL_LEN)
    vStd = stdev(vol_usd, VOL_LEN).replace(0.0, np.nan)
    volZ = ((vol_usd - vMean) / vStd).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    volAmp = np.clip(1.0 + VOL_BOOST * np.maximum(0.0, volZ.values), 0.8, 1.8)

    return dict(
        M_up=pd.Series(M_up, index=df.index),
        M_dn=pd.Series(M_dn, index=df.index),
        S_up=pd.Series(S_up, index=df.index),
        S_dn=pd.Series(S_dn, index=df.index),
        B_up=pd.Series(B_up, index=df.index),
        B_dn=pd.Series(B_dn, index=df.index),
        R_up=pd.Series(R_up, index=df.index),
        R_dn=pd.Series(R_dn, index=df.index),
        atr=pd.Series(atr_series, index=df.index),
        atr_sma=pd.Series(atr_sma, index=df.index),
        volAmp=pd.Series(volAmp, index=df.index),
        close=df["close"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        Zs=Zs,
    )

def _index_pad(series: pd.Series, t: pd.Timestamp) -> Optional[int]:
    idx = series.index.get_indexer([t], method="pad")
    return None if idx[0] == -1 else int(idx[0])

def _np_percentile_linear(arr: np.ndarray, q: float) -> float:
    try:
        return float(np.percentile(arr, q, method="linear"))
    except TypeError:
        return float(np.percentile(arr, q, interpolation="linear"))

def is_chop_at_time(feats: Dict[str, pd.Series], tstamp: pd.Timestamp) -> bool:
    end_loc = _index_pad(feats["atr"], tstamp)
    if end_loc is None:
        return True
    start_loc = max(0, end_loc - ATR_SMOOTH + 1)
    window_atr = feats["atr"].iloc[start_loc:end_loc + 1].dropna()
    if window_atr.empty:
        return True
    atr_now = float(feats["atr"].iloc[end_loc])
    if USE_PCT_CHOP:
        pct = _np_percentile_linear(window_atr.values, CHOP_PCT)
        return atr_now <= pct
    else:
        atr_sma_val = float(feats["atr_sma"].iloc[end_loc])
        return atr_now < atr_sma_val * CHOP_RATIO

# ============== Triple Screen / Elder ==============
def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first","high": "max","low": "min","close": "last","volume": "sum"}
    out = df[["open", "high", "low", "close", "volume"]].resample(rule, label="right", closed="right").agg(agg)
    return out.dropna(how="any")

def macd_hist(close: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    macd = ema(close, fast) - ema(close, slow)
    signal = ema(macd, sig)
    return macd - signal

def stoch_k(df: pd.DataFrame, n: int = 14) -> pd.Series:
    ll = df["low"].rolling(n, min_periods=1).min()
    hh = df["high"].rolling(n, min_periods=1).max()
    return 100.0 * (df["close"] - ll) / (hh - ll + 1e-12)

def to_logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))

def from_logit(z: float) -> float:
    z = max(min(z, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-z))

def elder_logit_adjust(df_1m: pd.DataFrame, tstamp: pd.Timestamp, p_up_hat: float) -> float:
    try:
        htf = resample_ohlc(df_1m, ELDER_HTF)
        if htf.empty:
            return p_up_hat
        i = htf.index.get_indexer([tstamp], method="pad")[0]
        if i <= 0:
            return p_up_hat
        hist = macd_hist(htf["close"])
        sgn = 1.0 if hist.iloc[i-1] > 0 else (-1.0 if hist.iloc[i-1] < 0 else 0.0)

        mtf = resample_ohlc(df_1m, ELDER_MID)
        if mtf.empty:
            return p_up_hat
        j = mtf.index.get_indexer([tstamp], method="pad")[0]
        if j <= 0:
            return p_up_hat
        k = float(stoch_k(mtf, STOCH_LEN).iloc[j-1])
        rng = max(1.0, (STOCH_OB - STOCH_OS))
        t = float(np.clip((k - STOCH_OS)/rng, 0.0, 1.0))
        pullback = 1.0 - 2.0 * t  # [-1..1]

        z = to_logit(p_up_hat) + ELDER_ALPHA * (sgn * pullback)
        return from_logit(z)
    except Exception:
        return p_up_hat

# ====== Вероятности из фич (+ авто-веса WF) ======
def prob_up_down_at_time(feats: Dict[str, pd.Series], tstamp: pd.Timestamp, w_dyn: Optional[np.ndarray] = None) -> Tuple[float, float, Dict[str, float]]:
    i = _index_pad(feats["M_up"], tstamp)
    if i is None:
        return 0.5, 0.5, {}
    M_up = float(feats["M_up"].iloc[i]); M_dn = float(feats["M_dn"].iloc[i])
    S_up = float(feats["S_up"].iloc[i]); S_dn = float(feats["S_dn"].iloc[i])
    B_up = float(feats["B_up"].iloc[i]); B_dn = float(feats["B_dn"].iloc[i])
    R_up = float(feats["R_up"].iloc[i]); R_dn = float(feats["R_dn"].iloc[i])
    volAmp = float(feats["volAmp"].iloc[i])

    if w_dyn is None or len(w_dyn) != 4:
        w_mom, w_vwp, w_brk, w_rev = 0.35, 0.20, 0.20, 0.25
    else:
        w_mom, w_vwp, w_brk, w_rev = [float(x) for x in w_dyn]
        # анти-усушка: если ||w|| слишком мала — откат к стартовым
        if float(np.linalg.norm([w_mom, w_vwp, w_brk, w_rev])) < 0.15:
            w_mom, w_vwp, w_brk, w_rev = 0.35, 0.20, 0.20, 0.25


    Z_up = (w_mom * M_up * volAmp) + (w_vwp * S_up) + (w_brk * B_up) + (w_rev * R_up)
    Z_dn = (w_mom * M_dn * volAmp) + (w_vwp * S_dn) + (w_brk * B_dn) + (w_rev * R_dn)
    P_up, P_dn = softmax2(Z_up, Z_dn)

    phi_wf = np.array([
        (M_up - M_dn) * volAmp,
        (S_up - S_dn),
        (B_up - B_dn),
        (R_up - R_dn)
    ], dtype=float)
    return P_up, P_dn, {"phi_wf0": phi_wf[0], "phi_wf1": phi_wf[1], "phi_wf2": phi_wf[2], "phi_wf3": phi_wf[3]}

# ====== Кросс-активы ======
def features_for_symbols(df_map: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.Series]]:
    out: Dict[str, Dict[str, pd.Series]] = {}
    for sym, df in df_map.items():
        if df is not None and not df.empty:
            try:
                out[sym] = features_from_binance(df)
            except Exception:
                pass
    return out

def _idx_with_shift(series: pd.Series, tstamp: pd.Timestamp, shift_bars: int = 0) -> Optional[int]:
    i = _index_pad(series, tstamp)
    if i is None:
        return None
    i = int(i) - int(shift_bars)
    if i < 0 or i >= len(series):
        return None
    return i

def cross_up_down_contrib(feats_map: Dict[str, Dict[str, pd.Series]],
                          tstamp: pd.Timestamp,
                          symbols: List[str],
                          w_mom: float,
                          w_vwap: float,
                          shift_bars: int = 0) -> Tuple[float, float]:
    z_up_sum, z_dn_sum = 0.0, 0.0
    for sym in symbols:
        f = feats_map.get(sym)
        if not f:
            continue
        i = _idx_with_shift(f["M_up"], tstamp, shift_bars)
        if i is None:
            continue
        vA = float(f["volAmp"].iloc[i])
        M_up = float(f["M_up"].iloc[i]); M_dn = float(f["M_dn"].iloc[i])
        S_up = float(f["S_up"].iloc[i]); S_dn = float(f["S_dn"].iloc[i])
        z_up = (w_mom * M_up * vA) + (w_vwap * S_up)
        z_dn = (w_mom * M_dn * vA) + (w_vwap * S_dn)
        z_up_sum += z_up
        z_dn_sum += z_dn
    return z_up_sum, z_dn_sum

# =============================
# Расширенная фабрика признаков для экспертов
# =============================
class ExtendedMLFeatures:
    """
    Собирает расширенный x-вектор под экспертов (XGB/RF/ARF/NN) из минутных фич и «сырых» OHLC:
      - базовые диффы (Mom/VWAP/Keltner/Bollinger)
      - BB z-score, Keltner position/width
      - ATR норм. / wick imbalance
      - RSI, тренд RSI (slope)
      - парные интеракции (умеренно)
    """
    def __init__(self, use_interactions: bool = True):
        self.use_interactions = use_interactions
        base_dim = 11
        inter_dim = 3 if use_interactions else 0
        self.dim = base_dim + inter_dim  # 14 при use_interactions=True

    def _keltner(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        basis = ema(df["close"], KC_LEN)
        rng = atr_wilder(df, KC_LEN)
        return basis, rng

    def build(self, df_1m: pd.DataFrame, feats: Dict[str, pd.Series], tstamp: pd.Timestamp) -> np.ndarray:
        i = _index_pad(feats["M_up"], tstamp)
        if i is None:
            return np.zeros(self.dim, dtype=float)

        m_diff = float(feats["M_up"].iloc[i] - feats["M_dn"].iloc[i])
        s_diff = float(feats["S_up"].iloc[i] - feats["S_dn"].iloc[i])
        b_diff = float(feats["B_up"].iloc[i] - feats["B_dn"].iloc[i])
        r_diff = float(feats["R_up"].iloc[i] - feats["R_dn"].iloc[i])

        z_bb = float(feats.get("Zs", pd.Series(index=df_1m.index, dtype=float)).iloc[i])

        basisKC, rngKC = self._keltner(df_1m)
        kc_pos = float(((df_1m["close"].iloc[i] - basisKC.iloc[i]) / (KC_MULT * rngKC.iloc[i] + 1e-12)))
        kc_w = float((KC_MULT * rngKC.iloc[i]) / max(1e-9, df_1m["close"].iloc[i]))

        atr_now = float(feats["atr"].iloc[i])
        atr_sma_now = float(feats["atr_sma"].iloc[i])
        atr_norm = float(atr_now / (atr_sma_now + 1e-12))

        rsi_series = rsi(df_1m["close"], 14).fillna(50.0)
        rsi_now = float(rsi_series.iloc[i])
        rsi_norm = (rsi_now - 50.0) / 50.0
        i_prev = max(0, i - 3)
        trend_rsi = float((rsi_series.iloc[i] - rsi_series.iloc[i_prev]) / 100.0)

        hi = float(df_1m["high"].iloc[i]); lo = float(df_1m["low"].iloc[i])
        op = float(df_1m["open"].iloc[i]); cl = float(df_1m["close"].iloc[i])
        rng = max(1e-12, hi - lo)
        up_w = hi - max(op, cl)
        dn_w = min(op, cl) - lo
        wick_imb = float((up_w - dn_w) / rng)

        feats_vec = [
            m_diff, s_diff, b_diff, r_diff,
            z_bb, kc_pos, atr_norm, rsi_norm,
            wick_imb, trend_rsi, kc_w
        ]
        if self.use_interactions:
            feats_vec += [
                m_diff * s_diff,
                m_diff * rsi_norm,
                s_diff * kc_pos,
            ]
        x = np.array(feats_vec, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        x = np.clip(x, -5.0, 5.0)
        return x

# =============================
# CSV / KPI

# =============================
CSV_COLUMNS = [
    "settled_ts","epoch","side","p_up",
    "p_meta_raw","p_meta2_raw","p_blend","blend_w","calib_src",
    "p_thr_used","p_thr_src","edge_at_entry",
    "stake","gas_bet_bnb","gas_claim_bnb",
    "gas_price_bet_gwei","gas_price_claim_gwei",
    "outcome","pnl","capital_before","capital_after",
    "lock_ts","close_ts","lock_price","close_price","payout_ratio","up_won",
    "r_hat_used","r_hat_source","r_hat_error_pct"  # ← НОВОЕ
]



# 👇 Единая схема типов для наших CSV
CSV_DTYPES = {
    "settled_ts":           "Int64",
    "epoch":                "Int64",
    "side":                 "string",
    "p_up":                 "float64",
    "p_meta_raw":           "float64",
    "p_meta2_raw":          "float64",   # ← NEW
    "p_blend":              "float64",   # ← NEW
    "blend_w":              "float64",   # ← NEW                       # ← ДОБАВИЛИ
    "calib_src":            "string", 
    "p_thr_used":           "float64",
    "p_thr_src":            "string",
    "edge_at_entry":        "float64",
    "stake":                "float64",
    "gas_bet_bnb":          "float64",
    "gas_claim_bnb":        "float64",
    "gas_price_bet_gwei":   "float64",
    "gas_price_claim_gwei": "float64",
    "outcome":              "string",    # ← важно: строка
    "pnl":                  "float64",
    "capital_before":       "float64",
    "capital_after":        "float64",
    "lock_ts":              "Int64",
    "close_ts":             "Int64",
    "lock_price":           "float64",
    "close_price":          "float64",
    "payout_ratio":         "float64",
    "up_won":               "boolean",
    "r_hat_used":           "float64",      # ← НОВОЕ
    "r_hat_source":         "string",       # ← НОВОЕ
    "r_hat_error_pct":      "float64",      # ← НОВОЕ   # ← важно: логический
}

# --- Порог для включения δ от тюнера ---
MIN_TRADES_FOR_DELTA = 500  # до этого числа завершённых сделок δ принудительно 0.000

def _settled_trades_count(path: str) -> int:
    """
    Считает КОЛИЧЕСТВО завершённых сделок (win/loss/draw) в trades CSV.
    Этого достаточно, чтобы решить: включать ли δ из тюнера или держать 0.000.
    """
    try:
        df = _read_csv_df(path)
        if df is None or df.empty:
            return 0
        out = df.get("outcome")
        if out is None:
            return 0
        # outcome хранится как строка; считаем только win/loss/draw
        out = out.astype("string").str.lower()
        return int(out.isin(["win", "loss", "draw"]).sum())
    except Exception:
        return 0



def _coerce_csv_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализуем типы столбцов согласно CSV_DTYPES (мягко, без падений)."""
    
    # ✅ Заменяем pd.NA на пустую строку для строковых колонок
    obj_like = list(df.select_dtypes(include=["object", "string"]).columns)
    if obj_like:
        df[obj_like] = df[obj_like].fillna("")  # безопаснее чем .where()
    
    for col, dtype in CSV_DTYPES.items():
        # bnbusdrt6.py  (функция _coerce_csv_dtypes)
        if col not in df.columns:
            # ✅ Используем np.nan вместо pd.NA
            if dtype in ("float64", "Int64"):
                df[col] = np.nan
            elif dtype == "boolean":
                # было: df[col] = pd.Series(dtype="boolean")  ← длина 0 → рассинхрон с индексом df
                df[col] = pd.Series([pd.NA]*len(df), dtype="boolean")
            else:
                df[col] = ""
            continue

        
        try:
            if dtype in ("float64",):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(np.nan).astype(dtype)
            elif dtype in ("Int64",):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # ✅ Int64 поддерживает pd.NA, но безопаснее через astype
                df[col] = df[col].astype("Int64")
            # bnbusdrt6.py  (функция _coerce_csv_dtypes)
            elif dtype == "boolean":
                # Надёжная нормализация булевых значений из старых CSV:
                # поддерживает 'True'/'False', '1'/'0', 't'/'f', 'y'/'n', 'yes'/'no' (регистр/пробелы игнорируем)
                s_str = df[col].astype("string").str.strip().str.lower()
                _map = {
                    "true": True, "false": False,
                    "1": True, "0": False,
                    "t": True, "f": False,
                    "y": True, "n": False,
                    "yes": True, "no": False,
                }
                df[col] = pd.Series(s_str.map(_map), dtype="boolean")
            else:
                df[col] = df[col].astype(dtype)
        except Exception as e:
            print(f"[warn] failed to coerce {col} to {dtype}: {e}")
            pass
    
    return df

def ensure_csv_header(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

def append_trade_row(path: str, row: Dict):
    ensure_csv_header(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in CSV_COLUMNS])


# ========= SHADOW CSV (для off-policy δ) =========
def ensure_shadow_header(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

def append_shadow_row(path: str, row: Dict):
    ensure_shadow_header(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in CSV_COLUMNS])
        

def try_settle_shadow_rows(path: str, w3: Web3, c, cur_epoch: int) -> None:
    """Закрыть теневые строки (outcome пуст) для уже завершённых эпох (< cur_epoch)."""
    if not os.path.exists(path):
        return

    df = _read_csv_df(path)
    if df.empty:
        return

    # Открытые тени: outcome отсутствует/пуст и epoch < cur_epoch
    outcome_series = df.get("outcome")
    if outcome_series is None:
        return

    open_mask = (
        outcome_series.isna() |
        (outcome_series.astype(str).str.len() == 0)
    ) & (pd.to_numeric(df["epoch"], errors="coerce") < int(cur_epoch))

    if not bool(open_mask.any()):
        return

    changed_any = False

    for idx, row in df.loc[open_mask].iterrows():
        try:
            # --- Безопасные геттеры чисел ---
            def _sf(x, default=0.0):
                """Безопасно конвертирует в float с обработкой pd.NA"""
                try:
                    # ✅ Явная проверка на pd.NA (для pandas < 2.0)
                    if pd.isna(x):  # работает и для pd.NA, и для np.nan
                        return default
                    
                    v = float(pd.to_numeric(x, errors="coerce"))
                    return v if math.isfinite(v) else default
                except (TypeError, ValueError):
                    return default

            epoch = int(pd.to_numeric(row.get("epoch"), errors="coerce"))
            rd = get_round(w3, c, epoch)
            if not rd or not getattr(rd, "oracle_called", False):
                # Раунд ещё не закрыт — попробуем позже
                continue

            side = str(row.get("side", "UP")).upper()
            stake = _sf(row.get("stake", 0.0), 0.0)

            up_won   = (rd.close_price > rd.lock_price)
            down_won = (rd.close_price < rd.lock_price)
            draw     = (rd.close_price == rd.lock_price)

            # Газ из зафиксированных в тени значений
            gb = _sf(row.get("gas_bet_bnb", 0.0), 0.0)    # bet gas
            gc = _sf(row.get("gas_claim_bnb", 0.0), 0.0)  # claim gas

            # Коэффициент выплат; если пуст/некорректен — используем 1.9 как дефолт
            ratio = _sf(getattr(rd, "payout_ratio", None), 1.9)
            if not math.isfinite(ratio) or ratio <= 0:
                ratio = 1.9

            # --- PnL с учётом газа (off-policy) ---
            if draw:
                pnl = -(gb + gc)
                outcome = "draw"
            else:
                win = (up_won and side == "UP") or (down_won and side == "DOWN")
                if win:
                    pnl = stake * (ratio - 1.0) - (gb + gc)
                    outcome = "win"
                else:
                    pnl = -stake - gb
                    outcome = "loss"

            # --- пополняем окно калибратора мета-вероятностей ---
            # --- пополняем окно калибратора мета-вероятностей ---
            try:
                if outcome in ("win", "loss"):
                    # сырое p_meta_raw, которое положили в bets[epoch] при решении
                    p_logged_raw = float(b.get("p_meta_raw", b.get("p_up", float('nan'))))
                    _CALIB_P_META.append(p_logged_raw)
                    _CALIB_Y_META.append(1 if outcome == "win" else 0)
                    # обновим онлайн-менеджер (если включен)
                    if os.getenv("CALIB_ENABLE","1")=="1":
                        settled_ts = int(time.time())
                        globals()["_CALIB_MGR"].update(p_logged_raw, 1 if outcome=="win" else 0, settled_ts)
            except Exception:
                pass


            # --- Заполняем


            # --- Заполняем поля в исходном df ---
            df.at[idx, "outcome"]       = outcome
            df.at[idx, "pnl"]           = pnl
            df.at[idx, "settled_ts"]    = int(time.time())
            df.at[idx, "lock_ts"]       = getattr(rd, "lock_ts", None)
            df.at[idx, "close_ts"]      = getattr(rd, "close_ts", None)
            df.at[idx, "lock_price"]    = getattr(rd, "lock_price", float("nan"))
            df.at[idx, "close_price"]   = getattr(rd, "close_price", float("nan"))
            df.at[idx, "payout_ratio"]  = ratio
            df.at[idx, "up_won"]        = bool(up_won)

            # Гипотетический capital_after для офф-полиси анализа
            cap_before = _sf(row.get("capital_before", float("nan")), float("nan"))
            if math.isfinite(cap_before):
                df.at[idx, "capital_after"] = cap_before + pnl

            changed_any = True

        except Exception as e:
            print(f"[shadow] settle error for epoch={row.get('epoch')} : {e}")

    if changed_any:
        df.to_csv(path, index=False, encoding="utf-8-sig")



def _read_csv_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        empty = pd.DataFrame({c: pd.Series(np.nan, dtype=CSV_DTYPES.get(c, "string")) for c in CSV_COLUMNS})
        return empty
    
    # ✅ Читаем БЕЗ dtype="string" и сразу заменяем pd.NA
    df = pd.read_csv(path, keep_default_na=True, encoding="utf-8-sig")
    
    # ✅ КРИТИЧНО: заменить pd.NA на np.nan ДО любых операций
    df = df.fillna(np.nan)  # более надёжно чем .replace()
    
    # Дополнительная зачистка
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].replace({"<NA>": np.nan, "NaN": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
    
    return _coerce_csv_dtypes(df)


def upgrade_csv_schema_if_needed(path: str) -> None:
    if not os.path.exists(path):
        return
    import pandas as pd
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return
    need_cols = {"p_thr_used","p_thr_src","edge_at_entry","p_meta_raw","p_meta2_raw","p_blend","blend_w","calib_src"}  # ← NEW
    missing = [c for c in need_cols if c not in df.columns]
    if not missing:
        return
    # добавим недостающие с пустыми значениями и перезапишем
    for c in missing:
        if c in ("p_thr_src","calib_src"):
            df[c] = ""
        else:
            df[c] = float("nan")
    # упорядочим по новой схеме
    cols = [c for c in CSV_COLUMNS if c in df.columns] + [c for c in CSV_COLUMNS if c not in df.columns]
    df = df.reindex(columns=cols)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _period_return_pct(df: pd.DataFrame, start_ts: int, now_ts: int) -> Optional[float]:
    if df.empty:
        return None
    df = df.sort_values("settled_ts")
    df_in = df[df["settled_ts"] >= start_ts]
    if df_in.empty:
        return None
    before_df = df[df["settled_ts"] < start_ts]
    if not before_df.empty:
        base_cap = float(before_df.iloc[-1]["capital_after"])
    else:
        base_cap = float(df_in.iloc[0]["capital_before"])
    end_cap = float(df.iloc[-1]["capital_after"])
    if base_cap <= 0:
        return None
    return (end_cap - base_cap) / base_cap * 100.0

def compute_stats_from_csv(path: str) -> Dict[str, Optional[float]]:
    df = _read_csv_df(path)
    if df.empty:
        return dict(total=0, wins=0, losses=0, winrate=None, roi_24h=None, roi_7d=None, roi_30d=None)
    df = df.dropna(subset=["outcome"])
    df_tr = df[df["outcome"].isin(["win","loss"])]
    wins = int((df_tr["outcome"] == "win").sum())
    losses = int((df_tr["outcome"] == "loss").sum())
    total = wins + losses
    winrate = (wins / total * 100.0) if total > 0 else None

    now_ts = int(time.time())
    roi_24 = _period_return_pct(df, now_ts - 24*3600, now_ts)
    roi_7d = _period_return_pct(df, now_ts - 7*24*3600, now_ts)
    roi_30 = _period_return_pct(df, now_ts - 30*24*3600, now_ts)

    return dict(total=total, wins=wins, losses=losses, winrate=winrate, roi_24h=roi_24, roi_7d=roi_7d, roi_30d=roi_30)

def print_stats(stats: Dict[str, Optional[float]]):
    wr = "—" if stats["winrate"] is None else f"{stats['winrate']:.2f}%"
    r24 = "—" if stats["roi_24h"] is None else f"{stats['roi_24h']:.2f}%"
    r7  = "—" if stats["roi_7d"]  is None else f"{stats['roi_7d']:.2f}%"
    r30 = "—" if stats["roi_30d"] is None else f"{stats['roi_30d']:.2f}%"
    print(f"[stats] trades={stats['total']}  wins={stats['wins']}  losses={stats['losses']}  winrate={wr}  "
          f"ROI: 24h={r24} | 7d={r7} | 30d={r30}")

def last3_ev_estimates(path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    df = _read_csv_df(path)
    if df.empty:
        return None, None, None
    df = df.dropna(subset=["outcome"]).sort_values("settled_ts")
    df = df[df["outcome"].isin(["win","loss","draw"])]
    tail = df.tail(3)
    r_vals = pd.to_numeric(tail.get("payout_ratio", pd.Series(dtype=float)), errors="coerce").dropna().values
    gb_vals = pd.to_numeric(tail.get("gas_bet_bnb", pd.Series(dtype=float)), errors="coerce").dropna().values
    gc_vals = pd.to_numeric(tail.get("gas_claim_bnb", pd.Series(dtype=float)), errors="coerce").dropna().values
    r_med = float(np.median(r_vals)) if len(r_vals) else None
    gb_med = float(np.median(gb_vals)) if len(gb_vals) else None
    gc_med = float(np.median(gc_vals)) if len(gc_vals) else None
    return r_med, gb_med, gc_med


def r_ewma_by_side(path: str, side_up: bool, alpha: float = 0.25,
                   max_epoch_exclusive: Optional[int] = None) -> Optional[float]:
    """
    Основная оценка r̂ без заглядывания: EWMA(λ=alpha) по payout_ratio прошлых сеттлов на нужной стороне.
    """
    df = _read_csv_df(path)
    if df.empty:
        return None
    df = df.dropna(subset=["outcome"]).sort_values("settled_ts")
    df = df[df["outcome"].isin(["win","loss","draw"])]
    if max_epoch_exclusive is not None and "epoch" in df.columns:
        try:
            df = df[df["epoch"] < int(max_epoch_exclusive)]
        except Exception:
            pass
    if df.empty:
        return None
    side_series = df.get("side", pd.Series(dtype="string")).astype(str).str.upper()
    df = df[side_series == ("UP" if side_up else "DOWN")]
    if df.empty:
        return None
    r = pd.to_numeric(df.get("payout_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    if r.empty:
        return None
    # EWMA с alpha=λ (adjust=False, чтобы не заглядывать назад «по-теоретически»)
    ew = r.ewm(alpha=float(alpha), adjust=False).mean()
    val = float(ew.iloc[-1])
    if not np.isfinite(val) or val <= 1.0:
        return None
    return val


def r_tod_percentile(path: str, side_up: bool, hour_utc: Optional[int] = None, q: float = 0.50,
                     max_epoch_exclusive: Optional[int] = None) -> Optional[float]:
    """
    Бэкап-оценка r̂: перцентиль payout_ratio по текущему часу суток (UTC) на стороне.
    Если выборка часа слишком мала, берём перцентиль по всей истории.
    """
    df = _read_csv_df(path)
    if df.empty:
        return None
    df = df.dropna(subset=["outcome"]).sort_values("settled_ts")
    df = df[df["outcome"].isin(["win","loss","draw"])]
    if max_epoch_exclusive is not None and "epoch" in df.columns:
        try:
            df = df[df["epoch"] < int(max_epoch_exclusive)]
        except Exception:
            pass
    if df.empty:
        return None
    side_series = df.get("side", pd.Series(dtype="string")).astype(str).str.upper()
    df = df[side_series == ("UP" if side_up else "DOWN")]
    if df.empty:
        return None

    ts_col = None
    for c in ["lock_ts", "open_ts", "settled_ts"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        return None

    ts = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.assign(_ts=ts)
    df = df[np.isfinite(df["_ts"])]
    if df.empty:
        return None

    hours = pd.to_datetime(df["_ts"], unit="s", utc=True).dt.hour
    df = df.assign(_hour=hours)

    if hour_utc is None:
        try:
            hour_utc = int(pd.Timestamp.utcnow().hour)
        except Exception:
            hour_utc = 0

    same_hour = df[df["_hour"] == int(hour_utc)]
    r_ser = pd.to_numeric(same_hour.get("payout_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    if r_ser.size < 5:
        r_ser = pd.to_numeric(df.get("payout_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    if r_ser.empty:
        return None

    q = float(min(max(q, 0.0), 1.0))
    val = float(np.quantile(r_ser.to_numpy(), q))
    if not np.isfinite(val) or val <= 1.0:
        return None
    return val


def rolling_winrate_laplace(path: str, n: int = 50, max_epoch_exclusive: Optional[int] = None) -> Optional[float]:
    """
    Laplace-сглажённый винрейт по последним n закрытым сделкам ДО текущего раунда.
    Возвращает значение в [0,1] или None, если выборка пуста.
    """
    df = _read_csv_df(path)
    if df.empty:
        return None
    df = df.dropna(subset=["outcome"]).sort_values("settled_ts")
    df = df[df["outcome"].isin(["win","loss"])]
    if max_epoch_exclusive is not None and "epoch" in df.columns:
        try:
            df = df[df["epoch"] < int(max_epoch_exclusive)]
        except Exception:
            pass
    if df.empty:
        return None
    tail = df.tail(int(n))
    wins = int((tail["outcome"] == "win").sum())
    total = int(len(tail))
    if total <= 0:
        return None
    # Laplace smoothing: (wins + 1) / (total + 2)
    return (wins + 1.0) / (total + 2.0)

# НОВОЕ: масштаб в просадке — f ← f * max(0.25, 1 - DD/0.30)
def _dd_scale_factor(path: str) -> float:
    try:
        df = _read_csv_df(path)
    except Exception:
        return 1.0
    if df.empty:
        return 1.0
    df = df.sort_values("settled_ts")
    eq = pd.to_numeric(df.get("capital_after", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
    if eq.size == 0:
        return 1.0
    peak = float(np.nanmax(eq))
    last = float(eq[-1])
    if peak <= 0:
        return 1.0
    dd = max(0.0, (peak - last) / peak)
    return float(max(0.25, 1.0 - dd / 0.30))


def implied_payout_ratio(side_up: bool, rd: RoundInfo, fee: float = TREASURY_FEE) -> Optional[float]:
    total = float(rd.bull_amount + rd.bear_amount)
    side_amt = float(rd.bull_amount if side_up else rd.bear_amount)
    if side_amt <= 0.0 or total <= 0.0:
        return None
    return (total / side_amt) * (1.0 - fee)


# === KPI: число закрытых и «тишина» 1ч ===
def settled_trades_count(path: str) -> int:
    df = _read_csv_df(path)
    if df.empty:
        return 0
    df = df.dropna(subset=["outcome"])
    df = df[df["outcome"].isin(["win","loss","draw"])]
    return int(len(df))

def had_trade_in_last_hours(path: str, hours: float = 1.0) -> bool:
    df = _read_csv_df(path)
    if df.empty:
        return False
    df = df.dropna(subset=["outcome"])
    df = df[df["outcome"].isin(["win","loss","draw"])]
    if df.empty:
        return False
    now_ts = int(time.time())
    cutoff = now_ts - int(hours * 3600)
    ts = pd.to_numeric(df.get("settled_ts", pd.Series(dtype=float)), errors="coerce").dropna()
    return bool((ts >= cutoff).any())

# =============================
# ПЕРСИСТЕНТНОСТЬ КАПИТАЛА
# =============================
def _restore_capital_from_csv(path: str) -> Optional[float]:
    """
    Возвращает capital_after из последней завершённой строки CSV.
    Если нет ни одной завершённой — пытается взять последнюю capital_before.
    Если CSV отсутствует/пуст — возвращает None.
    """
    try:
        df = _read_csv_df(path)
        if df.empty:
            return None
        # Берём только строки с outcome и capital_after
        df2 = df.dropna(subset=["outcome", "capital_after"])
        df2 = df2[df2["outcome"].isin(["win","loss","draw"])]
        if not df2.empty:
            cap = float(df2.iloc[-1]["capital_after"])
            if math.isfinite(cap) and cap > 0:
                return cap
        # Фолбэк — capital_before
        df3 = df.dropna(subset=["capital_before"])
        if not df3.empty:
            cap = float(df3.iloc[-1]["capital_before"])
            if math.isfinite(cap) and cap > 0:
                return cap
        return None
    except Exception as e:
        get_logger().warning("failed to restore capital from CSV", exc_info=True)
        return None


class CapitalState:
    def __init__(self, path: str = "capital_state.json"):
        self.path = path

    def load(self, default: float) -> float:
        try:
            with open(self.path, "r") as f:
                obj = json.load(f)
            cap = float(obj.get("capital", default))
            if not math.isfinite(cap) or cap <= 0:
                return default
            return cap
        except Exception:
            return default

    def save(self, capital: float, ts: Optional[int] = None) -> None:
        try:
            obj = {"capital": float(capital), "ts": int(ts or time.time())}
            with open(self.path, "w") as f:
                json.dump(obj, f)
        except Exception as e:
            get_logger().error("failed to save capital_state", exc_info=True)



# =============================
# Telegram
def tg_enabled() -> bool:
    # включено + есть реальные реквизиты (из ENV или констант)
    has_token = bool((TG_TOKEN if 'TG_TOKEN' in globals() else '') or
                     (TELEGRAM_BOT_TOKEN if 'TELEGRAM_BOT_TOKEN' in globals() else ''))
    has_chat  = bool((TG_CHAT_ID if 'TG_CHAT_ID' in globals() else 0) or
                     (TELEGRAM_CHAT_ID if 'TELEGRAM_CHAT_ID' in globals() else ''))
    return bool(TELEGRAM_ENABLED and has_token and has_chat)



# --- Telegram helpers ---
def _html_safe_allow_basic(text: str) -> str:
    """
    Экранируем всё, но разрешаем базовые теги, которые ты можешь оставить в шаблоне:
    <b>, </b>, <i>, </i>, <code>, </code>, <pre>, </pre>.
    ВАЖНО: динамические значения (числа, массивы) вставляй "сырыми" — эта функция их экранирует.
    """
    s = escape(text, quote=False)  # превращает &, <, > в сущности
    # Разрешаем базовые теги обратно (если они были в шаблонной части строки)
    allow = {
        "&lt;b&gt;": "<b>", "&lt;/b&gt;": "</b>",
        "&lt;i&gt;": "<i>", "&lt;/i&gt;": "</i>",
        "&lt;code&gt;": "<code>", "&lt;/code&gt;": "</code>",
        "&lt;pre&gt;": "<pre>", "&lt;/pre&gt;": "</pre>",
    }
    for k, v in allow.items():
        s = s.replace(k, v)
    return s

def tg_send(text: str, html: bool = True, **kwargs) -> bool:
    """
    Отправляет сообщение в TG. По умолчанию HTML-режим с автоэкранированием.
    Возвращает True/False, не бросает исключения (чтобы не ронять основной цикл).
    """
    global _TG_FAILS, _TG_MUTED_UNTIL, _TG_LAST_ERR
    # --- мост: берём токен/чат либо из ENV (TG_*), либо из констант (TELEGRAM_*) ---
    token = (TG_TOKEN.strip() if 'TG_TOKEN' in globals() and TG_TOKEN else
             (TELEGRAM_BOT_TOKEN.strip() if 'TELEGRAM_BOT_TOKEN' in globals() and TELEGRAM_BOT_TOKEN else ""))
    chat_id = (TG_CHAT_ID if 'TG_CHAT_ID' in globals() and TG_CHAT_ID else
               (int(str(TELEGRAM_CHAT_ID).strip()) if 'TELEGRAM_CHAT_ID' in globals() and TELEGRAM_CHAT_ID else 0))

    # ранний выход: Телега выключена или нет токена/чата
    if not TELEGRAM_ENABLED or not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    now = time.time()

    # cooldown/half-open вместо вечного mute
    if _TG_FAILS >= TG_MUTE_AFTER:
        if now < _TG_MUTED_UNTIL:
            print(f"[tg ] muted; until {time.strftime('%H:%M:%S', time.localtime(_TG_MUTED_UNTIL))} ({_TG_LAST_ERR})")
            return False
        # half-open: разрешаем одну пробу

    try:
        if html:
            payload = {
                "chat_id": chat_id,  # ← используем вычисленный chat_id
                "text": _html_safe_allow_basic(text),
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
        else:
            payload = {
                "chat_id": chat_id,  # ← и здесь тоже
                "text": text,
                "disable_web_page_preview": True,
            }

        r = SESSION.post(url, json=payload, timeout=(3.05, 5))
        if r.status_code == 400 and html:
            payload.pop("parse_mode", None)
            payload["text"] = text
            r = SESSION.post(url, json=payload, timeout=(3.05, 5))
        if r.status_code >= 400:
            try:
                desc = r.json().get("description", r.text)
            except Exception:
                desc = r.text
            raise RequestException(f"{r.status_code} {desc}")
        _TG_FAILS = 0
        _TG_MUTED_UNTIL = 0.0
        _TG_LAST_ERR = ""
        return True


    except RequestException as e:
        _TG_FAILS += 1
        _TG_LAST_ERR = str(e)
        if _TG_FAILS >= TG_MUTE_AFTER:
            _TG_MUTED_UNTIL = time.time() + TG_COOLDOWN_S
            print(f"[tg ] muted after {_TG_FAILS} fails for {TG_COOLDOWN_S}s ({_TG_LAST_ERR})")
            return False
        else:
            print(f"[tg ] send failed ({_TG_FAILS}/{TG_MUTE_AFTER}): {_TG_LAST_ERR}")
        return False



# --- TG utils: HTML escape + чанкование для 4096-лимита
def _tg_html_escape(s: str) -> str:
    if s is None: return ""
    # Минимум: &, <, >. Кавычки экранировать не обязательно для <pre><code>
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def tg_send_chunks(text: str, chat_id: str = TELEGRAM_CHAT_ID, parse_mode: str = "HTML"):
    """
    Дробит длинные сообщения < 4096 символов и шлёт по частям.
    """
    if not TELEGRAM_ENABLED:
        return
    MAX = 4000  # запас от 4096 из-за HTML-энтити
    parts = [text[i:i+MAX] for i in range(0, len(text), MAX)] or [text]
    for idx, part in enumerate(parts, 1):
        suffix = f" ({idx}/{len(parts)})" if len(parts) > 1 else ""
        try:
            tg_send(part + suffix, html=(str(parse_mode).upper() == "HTML"))
        except Exception as e:
            # не валим основной цикл из-за телеги
            print(f"[tg ] send failed: {e}")
            continue

def notify_ev_decision(title: str,
                       epoch: int,
                       side_txt: str,
                       p_side: float,
                       p_thr: float,
                       p_thr_src: str,
                       r_hat: float,
                       gb_hat: float,
                       gc_hat: float,
                       stake: float,
                       delta15: float = None,
                       extra_lines: list = None,
                       delta_eff: float | None = None):
    """
    Отправляет компактное уведомление с полным разбором порога.
    """
    try:
        # Безопасная конвертация всех параметров
        d = _as_float(DELTA_PROTECT if (delta_eff is None) else delta_eff, 0.0)
        p_side = _as_float(p_side, 0.5)
        p_thr = _as_float(p_thr, 0.5)
        r_hat = _as_float(r_hat, 1.9)
        gb_hat = _as_float(gb_hat, 0.0)
        gc_hat = _as_float(gc_hat, 0.0)
        stake = _as_float(stake, 0.0)

        head = f"<b>{_tg_html_escape(str(title))}</b> — epoch <code>{int(epoch)}</code>\n"
        lines = [
            f"side:       {str(side_txt)}",
            f"p_side:     {p_side:.4f}",
            f"p_thr:      {p_thr:.4f}  [{str(p_thr_src)}]",
            f"p_thr+δ:    {(p_thr + d):.4f}  (δ={d:.2f})",
            f"edge:       {p_side - (p_thr + d):+.4f}",
            f"r_hat:      {r_hat:.6f}",
            f"gb_hat:     {gb_hat:.8f}  (BNB)",
            f"gc_hat:     {gc_hat:.8f}  (BNB)",
            f"S (stake):  {stake:.6f}    (BNB)",
        ]
        
        if USE_STRESS_R15 and delta15 is not None:
            _d = _as_float(delta15, 0.0)
            if _d > 1e6:
                _d /= 1e18
            if math.isfinite(_d):
                lines.append(f"Δ15_med:   {_d:.6f}  (BNB)")

        if extra_lines:
            for x in extra_lines:
                if x:
                    lines.append(str(x))

        block = "<pre><code>" + _tg_html_escape("\n".join(lines)) + "</code></pre>"
        tg_send_chunks(head + block, parse_mode="HTML")
    except Exception as e:
        print(f"[tg ] notify_ev_decision failed: {e}")


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or not isinstance(x, (float, int)) or not math.isfinite(float(x)):
        return "—"
    return f"{x:.2f}%"

def _period_winrate(df: pd.DataFrame, start_ts: int, end_ts: int) -> Optional[float]:
    sub = df[(df["settled_ts"] >= start_ts) & (df["settled_ts"] < end_ts)]
    sub = sub[sub["outcome"].isin(["win","loss"])]
    n = len(sub)
    if n == 0:
        return None
    wins = int((sub["outcome"] == "win").sum())
    return wins / n * 100.0

def winrate_explanation(path: str) -> str:
    df = _read_csv_df(path)
    if df.empty:
        return "Объяснение: данных ещё нет."
    now_ts = int(time.time())
    last24 = _period_winrate(df, now_ts - 24*3600, now_ts)
    prev24 = _period_winrate(df, now_ts - 48*3600, now_ts - 24*3600)
    note_parts = []
    if last24 is None:
        return "Объяснение: за последние 24ч нет закрытых сделок."
    if prev24 is None:
        note_parts.append("База сравнения пустая — смотрим только текущие 24ч.")
        prev24 = last24
    diff = last24 - prev24
    direction = "вырос" if diff > 0 else ("упал" if diff < 0 else "без изменений")
    note_parts.append(f"Винрейт {direction} на {abs(diff):.2f} п.п. (текущие 24ч: {last24:.2f}%, предыдущие 24ч: {prev24:.2f}%).")
    def _avg(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.mean()) if len(s) else None
    cur_df = df[(df["settled_ts"] >= now_ts - 24*3600) & (df["outcome"].isin(["win","loss"]))]
    prv_df = df[(df["settled_ts"] >= now_ts - 48*3600) & (df["settled_ts"] < now_ts - 24*3600) & (df["outcome"].isin(["win","loss"]))]
    avg_p_cur = _avg(cur_df.get("p_up", pd.Series(dtype=float)))
    avg_p_prv = _avg(prv_df.get("p_up", pd.Series(dtype=float)))
    med_r_cur = pd.to_numeric(cur_df.get("payout_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    med_r_cur = float(np.median(med_r_cur)) if len(med_r_cur) else None
    med_r_prv = pd.to_numeric(prv_df.get("payout_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    med_r_prv = float(np.median(med_r_prv)) if len(med_r_prv) else None
    n_cur = len(cur_df)
    if avg_p_cur is not None and avg_p_prv is not None and abs(avg_p_cur - avg_p_prv) >= 0.01:
        note_parts.append(f"Средняя p_up изменилась на {avg_p_cur-avg_p_prv:+.3f} (было {avg_p_prv:.3f} → стало {avg_p_cur:.3f}).")
    if med_r_cur is not None and med_r_prv is not None and abs(med_r_cur - med_r_prv) >= 0.02:
        note_parts.append(f"Медианный payout изменился на {med_r_cur-med_r_prv:+.2f} (было {med_r_prv:.2f} → стало {med_r_cur:.2f}).")
    if n_cur < 10:
        note_parts.append(f"Выборка за 24ч мала (n={n_cur}), возможен шум.")
    return " ".join(note_parts)

def build_stats_message(stats: Dict[str, Optional[float]]) -> str:
    wr = "—" if stats["winrate"] is None else f"{stats['winrate']:.2f}%"
    r24 = "—" if stats["roi_24h"] is None else f"{stats['roi_24h']:.2f}%"
    r7  = "—" if stats["roi_7d"]  is None else f"{stats['roi_7d']:.2f}%"
    r30 = "—" if stats["roi_30d"] is None else f"{stats['roi_30d']:.2f}%"
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    total = stats.get("total", 0)

    # Reserve balance
    try:
        from reserve_fund import ReserveFund
        _reserve_path = os.path.join(os.path.dirname(__file__), "reserve_state.json")
        _rf = ReserveFund(path=_reserve_path)
        reserve_line = f"Reserve: <b>{_rf.balance:.6f} BNB</b>\n"
    except Exception:
        reserve_line = ""
    
    # ← НОВОЕ: анализ точности r̂ из модуля
    r_hat_line = ""
    try:
        from r_hat_improved import analyze_r_hat_accuracy
        acc = analyze_r_hat_accuracy(CSV_PATH, n=200)
        if acc and acc.get("n_samples", 0) >= 20:
            mae = acc["mae_pct"]
            bias = acc["bias_pct"]
            n = acc["n_samples"]
            r_hat_line = f"r̂ accuracy: MAE={mae:.1f}%, bias={bias:+.1f}% (n={n})\n"
    except Exception:
        pass

    msg = (f"<b>Статистика</b>\n"
           f"Trades: {total} | Wins: {wins} | Losses: {losses}\n"
           f"Winrate: <b>{wr}</b>\n"
           f"ROI: 24h={r24} | 7d={r7} | 30d={r30}\n"
           f"{reserve_line}"
           f"{r_hat_line}")
    return msg


def send_round_snapshot(prefix: str, extra_lines: List[str]):
    stats_dict = compute_stats_from_csv(CSV_PATH)
    stats_msg = build_stats_message(stats_dict)
    explain = winrate_explanation(CSV_PATH)
    text = f"{prefix}\n" + "\n".join(extra_lines) + "\n\n" + stats_msg + f"<i>{explain}</i>"
    tg_send(text)

# =============================
# ML: КОНФИГИ
# =============================
@dataclass
class MLConfig:
    # общие пороги гейтинга (как было)
    min_ready: int = 80
    enter_wr: float = 3.0
    exit_wr: float = 1.0
    retrain_every: int = 40
    adwin_delta: float = 0.002
    max_memory: int = 5000
    train_window: int = 1500

    # XGB
    xgb_model_path: str = "gb_model.json"
    xgb_scaler_path: str = "gb_scaler.pkl"
    xgb_state_path: str = "gb_state.json"
    xgb_cal_path: str = "gb_cal.pkl"                 # 👈 НОВОЕ
    xgb_calibration_method: str = "platt"   
    xgb_max_depth: int = 4
    xgb_eta: float = 0.08
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_min_child_weight: int = 2
    xgb_rounds_cold: int = 60
    xgb_rounds_warm: int = 30

    # RF
    rf_model_path: str = "rf_calibrated.pkl"
    rf_state_path: str = "rf_state.json"
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 2
    rf_calibration_method: str = "sigmoid"  # 'sigmoid' или 'isotonic'

    # ARF (River)
    arf_state_path: str = "arf_state.json"
    arf_model_path: str = "arf_model.pkl"
    arf_cal_path: str = "arf_cal.pkl"                # 👈 НОВОЕ
    arf_calibration_method: str = "platt"
    arf_n_models: int = 10
    arf_max_depth: Optional[int] = None

    # NN Expert (MLP)
    nn_state_path: str = "nn_state.json"
    nn_model_path: str = "nn_model.pkl"
    nn_scaler_path: str = "nn_scaler.pkl"
    nn_hidden: int = 16
    nn_eta: float = 0.01
    nn_l2: float = 0.0005
    nn_epochs: int = 30
    nn_retrain_every: int = 40
    nn_calib_every: int = 200

    # META
    meta_state_path: str = "meta_state.json"
    meta_eta: float = 0.05
    meta_l2: float = 0.001
    meta_w_clip: float = 8.0
    meta_g_clip: float = 1.0

    # NEW: контекстный гейтинг
    meta_gating_mode: str = "soft"   # "soft" | "exp4"
    meta_alpha_mix: float = 1.0      # вес смеси логитов экспертов
    meta_gate_eta: float = 0.02      # шаг для Wg (soft)
    meta_gate_l2: float = 0.0005     # L2 для Wg
    meta_gate_clip: float = 5.0      # клип градиента Wg

    # EXP4 вариант (по фазам)
    meta_exp4_eta: float = 0.10      # темп обновления весов EXP4
    meta_exp4_phases: int = 6        # число фаз (см. phase_from_ctx)

    use_two_window_drop: bool = False
# =============================
    # ====== ФАЗОВАЯ ПАМЯТЬ / КАЛИБРОВКА ======
    use_phase_memory: bool = True
    phase_count: int = 6                   # дубль meta_exp4_phases для удобства
    phase_memory_cap: int = 10_000         # на ФАЗУ
    phase_min_ready: int = 50              # минимум для «включения» фазы
    phase_mix_global_share: float = 0.30   # если < phase_min_ready: доля глобального хвоста
    phase_hysteresis_s: int = 300     
    meta_use_cma_es: bool = True  # ← включаем CMA-ES     # залипание фазы (анти-дрожь)
    phase_state_path: str = "phase_state.json"

    # для файлов калибраторов по фазе (будем апеллировать к существующим путям)
    # у XGB/NN: base_path → base_path_ph{φ}.pkl
    # Cross-validation parameters
    cv_enabled: bool = True
    cv_n_splits: int = 5              # количество фолдов
    cv_embargo_pct: float = 0.02      # 2% gap между train/test
    cv_purge_pct: float = 0.01        # 1% purge перед test
    cv_min_train_size: int = 200      # минимум для обучения
    cv_bootstrap_n: int = 1000        # итераций bootstrap для CI
    cv_confidence: float = 0.95       # уровень доверия (95%)
    cv_min_improvement: float = 0.02  # минимум +2% для значимости
    
    # Validation tracking
    cv_oof_window: int = 500          # окно out-of-fold predictions
    cv_check_every: int = 50          # проверка каждые N примеров


# ===== Фильтр фазы с гистерезисом =====
class PhaseFilter:
    def __init__(self, hysteresis_s: int = 300):
        self.hysteresis_s = int(max(0, hysteresis_s))
        self.last_phase: Optional[int] = None
        self.last_change_ts: Optional[int] = None

    def update(self, phase_raw: int, now_ts: int) -> int:
        # первый раз — принять как есть
        if self.last_phase is None:
            self.last_phase, self.last_change_ts = int(phase_raw), int(now_ts)
            return self.last_phase
        # если новая фаза = старая — просто обновим время
        if int(phase_raw) == int(self.last_phase):
            self.last_change_ts = int(now_ts)
            return self.last_phase
        # если прошло мало времени — «залипаем»
        if self.last_change_ts is not None and (now_ts - self.last_change_ts) < self.hysteresis_s:
            return self.last_phase
        # иначе позволим смениться
        self.last_phase = int(phase_raw)
        self.last_change_ts = int(now_ts)
        return self.last_phase


# Базовый интерфейс экспертов
# =============================
class _BaseExpert:
    def proba_up(self, x_raw: np.ndarray, reg_ctx: Optional[dict] = None) -> tuple[Optional[float], str]:
        raise NotImplementedError
    def record_result(self, x_raw: np.ndarray, y_up: int, used_in_live: bool, p_pred: Optional[float] = None, reg_ctx: Optional[dict] = None) -> None:
        raise NotImplementedError
    def maybe_train(self, ph: Optional[int] = None, reg_ctx: Optional[dict] = None) -> None:
        pass
    def status(self) -> Dict[str, str]:
        return {"mode":"DISABLED", "wr":"—", "n":"0", "enabled":"False"}
# =============================

# ---------- XGB ----------
class XGBExpert(_BaseExpert):
    def __init__(self, cfg: MLConfig):
        self.cfg = cfg
        self.enabled = HAVE_XGB
        self.mode = "SHADOW"

        # модель XGBoost
        self.booster = None
        # скейлер
        self.scaler: Optional[StandardScaler] = None
        # детектор дрейфа
        self.adwin = ADWIN(delta=self.cfg.adwin_delta) if HAVE_RIVER else None

        # ===== глобальная память (хвост) =====
        self.X: List[List[float]] = []
        self.y: List[int] = []
        self.new_since_train = 0

        # ===== фазовая память =====
        self.P = int(self.cfg.phase_count)  # 6 фаз
        self.X_ph: Dict[int, List[List[float]]] = {p: [] for p in range(self.P)}
        self.y_ph: Dict[int, List[int]] = {p: [] for p in range(self.P)}
        self.new_since_train_ph: Dict[int, int] = {p: 0 for p in range(self.P)}
        self._last_seen_phase: int = 0

        # ===== фазовые калибраторы =====
        self.cal_ph: Dict[int, Optional[_BaseCal]] = {p: None for p in range(self.P)}
        self.cal_global: Optional[_BaseCal] = None  # для обратной совместимости

        # хиты/диагностика
        self.shadow_hits: List[int] = []
        self.active_hits: List[int] = []

        # Cross-validation tracking (per phase)
        self.cv_oof_preds: Dict[int, deque] = {p: deque(maxlen=cfg.cv_oof_window) for p in range(self.P)}
        self.cv_oof_labels: Dict[int, deque] = {p: deque(maxlen=cfg.cv_oof_window) for p in range(self.P)}
        self.cv_metrics: Dict[int, Dict] = {p: {} for p in range(self.P)}
        self.cv_last_check: Dict[int, int] = {p: 0 for p in range(self.P)}
        
        # Validation mode tracking
        self.validation_passed: Dict[int, bool] = {p: False for p in range(self.P)}

        self.n_feats: Optional[int] = None

        # загрузка калибратора (глобального) и стейтов
        try:
            self.cal_global = _BaseCal.load(self.cfg.xgb_cal_path)
        except Exception:
            self.cal_global = None

        self._load_all()

        # вспомогалка для путей фазовых калибраторов
        import os as _os
        self._cal_path = lambda base, ph: f"{_os.path.splitext(base)[0]}_ph{ph}{_os.path.splitext(base)[1]}"

        # попытка подгрузить фазовые калибраторы
        for p in range(self.P):
            try:
                self.cal_ph[p] = _BaseCal.load(self._cal_path(self.cfg.xgb_cal_path, p))
            except Exception:
                self.cal_ph[p] = None


    def _load_all(self):
        try:
            if os.path.exists(self.cfg.xgb_state_path):
                with open(self.cfg.xgb_state_path, "r") as f:
                    st = json.load(f)

                # базовые поля
                self.mode = st.get("mode", "SHADOW")
                self.shadow_hits = st.get("shadow_hits", [])[-1000:]
                self.active_hits = st.get("active_hits", [])[-1000:]
                self.n_feats = st.get("n_feats", None)

                # 👇 восстановление памяти
                self.X = st.get("X", [])
                self.y = st.get("y", [])

                X_ph = st.get("X_ph", {})
                y_ph = st.get("y_ph", {})
                if isinstance(X_ph, dict) and isinstance(y_ph, dict):
                    self.X_ph = {int(k): v for k, v in X_ph.items()}
                    self.y_ph = {int(k): v for k, v in y_ph.items()}

                # счётчики тренировки по фазам
                self.new_since_train_ph = {p: 0 for p in range(self.P)}
                if isinstance(st.get("new_since_train_ph"), dict):
                    for k, v in st["new_since_train_ph"].items():
                        try:
                            self.new_since_train_ph[int(k)] = int(v)
                        except Exception:
                            pass

                self._last_seen_phase = int(st.get("_last_seen_phase", 0))

                # безопасные обрезки
                mm = getattr(self.cfg, "max_memory", None)
                if isinstance(mm, int) and mm > 0 and len(self.X) > mm:
                    self.X = self.X[-mm:]
                    self.y = self.y[-mm:]

                cap = int(getattr(self.cfg, "phase_memory_cap", 10_000))
                for p in range(self.P):
                    if len(self.X_ph.get(p, [])) > cap:
                        self.X_ph[p] = self.X_ph[p][-cap:]
                        self.y_ph[p] = self.y_ph[p][-cap:]
        except Exception:
            pass

        # scaler/booster — без изменений
        try:
            if os.path.exists(self.cfg.xgb_scaler_path):
                with open(self.cfg.xgb_scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
        except Exception:
            self.scaler = None
        try:
            if HAVE_XGB and os.path.exists(self.cfg.xgb_model_path):
                bst = xgb.Booster()
                bst.load_model(self.cfg.xgb_model_path)
                self.booster = bst
        except Exception:
            self.booster = None


    def _save_all(self):
        # --- state (режим, хиты, память) ---
        try:
            # обрезка глобального хвоста
            X_tail, y_tail = self.X, self.y
            mm = getattr(self.cfg, "max_memory", None)
            if isinstance(mm, int) and mm > 0:
                X_tail = self.X[-mm:]
                y_tail = self.y[-mm:]

            # обрезка фазовых буферов
            cap = int(getattr(self.cfg, "phase_memory_cap", 10_000))
            X_ph_tail = {p: self.X_ph.get(p, [])[-cap:] for p in range(self.P)}
            y_ph_tail = {p: self.y_ph.get(p, [])[-cap:] for p in range(self.P)}

            st = {
                "mode": self.mode,
                "shadow_hits": self.shadow_hits[-1000:],
                "active_hits": self.active_hits[-1000:],
                "n_feats": self.n_feats,

                # 👇 память
                "X": X_tail, "y": y_tail,
                "X_ph": X_ph_tail, "y_ph": y_ph_tail,
                "new_since_train_ph": {int(p): int(self.new_since_train_ph.get(p, 0)) for p in range(self.P)},
                "_last_seen_phase": int(self._last_seen_phase),
                "P": int(self.P),
            }
            with open(self.cfg.xgb_state_path, "w") as f:
                json.dump(st, f)
        except Exception as e:
            print(f"[xgb ] _save_all state error: {e}")

        # --- scaler / booster ---
        try:
            if self.scaler is not None:
                with open(self.cfg.xgb_scaler_path, "wb") as f:
                    pickle.dump(self.scaler, f)
        except Exception:
            pass
        try:
            if HAVE_XGB and self.booster is not None:
                self.booster.save_model(self.cfg.xgb_model_path)
        except Exception:
            pass



    # ---------- утилиты ----------
    def _ensure_dim(self, x_raw: np.ndarray):
        d = int(x_raw.reshape(1, -1).shape[1])
        if self.n_feats is None or self.n_feats != d:
            # смена размерности — чистим всё
            self.n_feats = d
            self.X, self.y = [], []
            self.X_ph = {p: [] for p in range(self.P)}
            self.y_ph = {p: [] for p in range(self.P)}
            self.new_since_train = 0
            self.new_since_train_ph = {p: 0 for p in range(self.P)}
            self._last_seen_phase = 0
            self.booster = None
            self.scaler = None

    def _transform_one(self, x_raw: np.ndarray) -> np.ndarray:
        self._ensure_dim(x_raw)
        xr = x_raw.astype(np.float32).reshape(1, -1)
        if self.scaler is None:
            return xr
        return self.scaler.transform(xr).astype(np.float32)

    def _transform_many(self, X_raw: np.ndarray) -> np.ndarray:
        X_raw = X_raw.astype(np.float32).reshape(-1, self.n_feats or X_raw.shape[1])
        if self.scaler is None:
            return X_raw
        return self.scaler.transform(X_raw).astype(np.float32)

    def _predict_raw(self, x_raw: np.ndarray) -> Optional[float]:
        if not self.enabled or self.booster is None:
            return None
        Xt = self._transform_one(x_raw)
        d = xgb.DMatrix(Xt)
        p = float(self.booster.predict(d)[0])
        return float(min(max(p, 1e-6), 1.0 - 1e-6))

    def _get_global_tail(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n <= 0 or not self.X:
            return np.empty((0, self.n_feats or 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        Xg = np.array(self.X[-n:], dtype=np.float32)
        yg = np.array(self.y[-n:], dtype=np.int32)
        return Xg, yg

    def _get_past_phases_tail(self, ph: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        # Берет последние n записей только из фаз 0..ph (включительно)
        # Исключает данные из будущих фаз → нет утечки
        if n <= 0:
            return np.empty((0, self.n_feats or 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        
        # Собираем все данные из фаз 0..ph
        X_past = []
        y_past = []
        for p in range(min(ph + 1, self.P)):
            if self.X_ph.get(p):
                X_past.extend(self.X_ph[p])
                y_past.extend(self.y_ph[p])
        
        if not X_past:
            return np.empty((0, self.n_feats or 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        
        # Берем последние n записей
        X_past = X_past[-n:]
        y_past = y_past[-n:]
        
        return np.array(X_past, dtype=np.float32), np.array(y_past, dtype=np.int32)


    def _get_phase_train(self, ph: int) -> Tuple[np.ndarray, np.ndarray]:
        # X_phase
        Xp = np.array(self.X_ph[ph], dtype=np.float32) if self.X_ph[ph] else np.empty((0, self.n_feats or 0), dtype=np.float32)
        yp = np.array(self.y_ph[ph], dtype=np.int32)   if self.y_ph[ph]  else np.empty((0,), dtype=np.int32)

        if len(Xp) >= int(self.cfg.phase_min_ready):
            return Xp, yp

        # иначе смешиваем X_phase ∪ X_past_phases_tail (70/30 по умолчанию)
        # ИСПРАВЛЕНИЕ: используем только прошлые фазы (0..ph), а не весь глобальный хвост
        share = float(self.cfg.phase_mix_global_share)  # 0.30
        need_g = int(round(len(Xp) * share / max(1e-9, (1.0 - share))))
        need_g = max(need_g, int(self.cfg.phase_min_ready) - len(Xp))   # не менее, чтобы достичь порога
        
        # Считаем сколько доступно в фазах 0..ph
        available_past = sum(len(self.X_ph.get(p, [])) for p in range(min(ph + 1, self.P)))
        need_g = min(need_g, available_past)  # не больше, чем доступно в прошлых фазах
        
        Xg, yg = self._get_past_phases_tail(ph, need_g)
        if len(Xg) == 0:
            return Xp, yp

        X = np.concatenate([Xp, Xg], axis=0)
        y = np.concatenate([yp, yg], axis=0)
        return X, y

    def _maybe_train_phase(self, ph: int):
        if not self.enabled or self.n_feats is None:
            return
        if self.new_since_train_ph.get(ph, 0) < int(self.cfg.retrain_every):
            return
        X_all, y_all = self._get_phase_train(ph)
        if len(X_all) < int(self.cfg.phase_min_ready):
            return
        if len(X_all) > int(self.cfg.train_window):
            X_all = X_all[-int(self.cfg.train_window):]
            y_all = y_all[-int(self.cfg.train_window):]

        try:
            # масштабирование как раньше (global scaler ок, но лучше по батчу фазы)
            self.scaler = StandardScaler().fit(X_all)
            Xt = self.scaler.transform(X_all)
            dtrain = xgb.DMatrix(Xt, label=y_all)

            params = dict(
                objective="binary:logistic",
                eval_metric="logloss",
                eta=getattr(self.cfg, "xgb_eta", 0.1),
                max_depth=getattr(self.cfg, "xgb_max_depth", 4),
                subsample=getattr(self.cfg, "xgb_subsample", 0.9),
                colsample_bytree=getattr(self.cfg, "xgb_colsample_bytree", 0.8),
                min_child_weight=getattr(self.cfg, "xgb_min_child_weight", 1.0),
                tree_method="auto",
            )
            num_round = int(self.cfg.xgb_rounds_cold if (self.booster is None) else self.cfg.xgb_rounds_warm)
            self.booster = xgb.train(params, dtrain, num_boost_round=num_round, xgb_model=self.booster)
            self.new_since_train_ph[ph] = 0
            self._save_all()
        except Exception as e:
            print(f"[xgb ] train error (ph={ph}): {e}")

    def _run_cv_validation(self, ph: int) -> Dict:
        """
        Walk-forward purged cross-validation для фазы ph.
        Возвращает метрики: accuracy, CI bounds, fold scores.
        """
        X_all, y_all = self._get_phase_train(ph)
        
        if len(X_all) < self.cfg.cv_min_train_size:
            return {"status": "insufficient_data", "oof_accuracy": 0.0}
        
        n_samples = len(X_all)
        n_splits = min(self.cfg.cv_n_splits, n_samples // self.cfg.cv_min_train_size)
        
        if n_splits < 2:
            return {"status": "insufficient_splits", "oof_accuracy": 0.0}
        
        # Walk-forward splits с purge и embargo
        embargo_size = max(1, int(n_samples * self.cfg.cv_embargo_pct))
        purge_size = max(1, int(n_samples * self.cfg.cv_purge_pct))
        
        fold_size = n_samples // n_splits
        oof_preds = np.zeros(n_samples)
        oof_mask = np.zeros(n_samples, dtype=bool)
        fold_scores = []
        
        for fold_idx in range(n_splits):
            # Test fold
            test_start = fold_idx * fold_size
            test_end = min(test_start + fold_size, n_samples)
            
            # Train: всё до (test_start - purge_size)
            train_end = max(0, test_start - purge_size)
            
            if train_end < self.cfg.cv_min_train_size:
                continue
            
            X_train = X_all[:train_end]
            y_train = y_all[:train_end]
            X_test = X_all[test_start:test_end]
            y_test = y_all[test_start:test_end]
            
            # Обучаем временную модель на train fold
            temp_model = self._train_fold_model(X_train, y_train, ph)
            
            # Предсказания на test fold
            preds = self._predict_fold(temp_model, X_test, ph)
            
            # Сохраняем OOF predictions
            oof_preds[test_start:test_end] = preds
            oof_mask[test_start:test_end] = True
            
            # Метрики фолда
            fold_acc = np.mean((preds >= 0.5) == y_test)
            fold_scores.append(fold_acc)
        
        # Итоговые OOF метрики
        oof_valid = oof_mask.sum()
        if oof_valid < self.cfg.cv_min_train_size:
            return {"status": "insufficient_oof", "oof_accuracy": 0.0}
        
        oof_accuracy = 100.0 * np.mean((oof_preds[oof_mask] >= 0.5) == y_all[oof_mask])
        
        # Bootstrap confidence intervals
        ci_lower, ci_upper = self._bootstrap_ci(
            oof_preds[oof_mask], 
            y_all[oof_mask],
            n_bootstrap=self.cfg.cv_bootstrap_n,
            confidence=self.cfg.cv_confidence
        )
        
        return {
            "status": "ok",
            "oof_accuracy": oof_accuracy,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "fold_scores": fold_scores,
            "n_folds": len(fold_scores),
            "oof_samples": int(oof_valid)
        }

    def _bootstrap_ci(self, preds: np.ndarray, labels: np.ndarray, 
                    n_bootstrap: int, confidence: float) -> tuple:
        """
        Bootstrap confidence intervals для accuracy.
        """
        accuracies = []
        n = len(preds)
        
        for _ in range(n_bootstrap):
            # Resample с возвратом
            idx = np.random.choice(n, size=n, replace=True)
            boot_preds = preds[idx]
            boot_labels = labels[idx]
            boot_acc = 100.0 * np.mean((boot_preds >= 0.5) == boot_labels)
            accuracies.append(boot_acc)
        
        accuracies = np.array(accuracies)
        alpha = 1.0 - confidence
        ci_lower = np.percentile(accuracies, 100 * alpha / 2)
        ci_upper = np.percentile(accuracies, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper

    def _train_fold_model(self, X: np.ndarray, y: np.ndarray, ph: int):
        """
        Обучает временную модель для CV fold.
        Реализация зависит от типа эксперта (XGB/RF/ARF/NN).
        """
        # Пример для XGB
        if not HAVE_XGB:
            return None
        
        scaler = StandardScaler().fit(X) if HAVE_SKLEARN else None
        X_scaled = scaler.transform(X) if scaler else X
        
        dtrain = xgb.DMatrix(X_scaled, label=y)
        model = xgb.train(
            params={
                "objective": "binary:logistic",
                "max_depth": self.cfg.xgb_max_depth,
                "eta": self.cfg.xgb_eta,
                "subsample": self.cfg.xgb_subsample,
                "colsample_bytree": self.cfg.xgb_colsample_bytree,
                "min_child_weight": self.cfg.xgb_min_child_weight,
                "eval_metric": "logloss",
            },
            dtrain=dtrain,
            num_boost_round=self.cfg.xgb_rounds_warm,
            verbose_eval=False
        )
        
        return {"model": model, "scaler": scaler}

    def _predict_fold(self, fold_model, X: np.ndarray, ph: int) -> np.ndarray:
        """
        Предсказания временной модели CV fold.
        """
        if fold_model is None:
            return np.full(len(X), 0.5)
        
        scaler = fold_model.get("scaler")
        model = fold_model.get("model")
        
        X_scaled = scaler.transform(X) if scaler else X
        dtest = xgb.DMatrix(X_scaled)
        preds = model.predict(dtest)
        
        return preds


    # ---------- инференс / запись ----------
    def proba_up(self, x_raw: np.ndarray, reg_ctx: Optional[dict] = None) -> Tuple[Optional[float], str]:
        if not self.enabled:
            return (None, "DISABLED")
        if self.booster is None:
            try:
                self._ensure_dim(x_raw)
            except Exception:
                pass
            return (None, self.mode)

        try:
            self._ensure_dim(x_raw)

            # сырой прогноз
            Xt = self._transform_one(x_raw)
            d = xgb.DMatrix(Xt)
            p = float(self.booster.predict(d)[0])
            p = float(min(max(p, 1e-6), 1.0 - 1e-6))

            # фазовый калибратор
            ph = int(reg_ctx.get("phase")) if isinstance(reg_ctx, dict) and "phase" in reg_ctx else 0
            self._last_seen_phase = ph
            cal = self.cal_ph.get(ph) or self.cal_global
            if cal is not None and getattr(cal, "ready", False):
                try:
                    p = float(cal.transform(p))
                except Exception:
                    pass

            p = float(min(max(p, 1e-6), 1.0 - 1e-6))
            return (p, self.mode)
        except Exception:
            return (None, self.mode)

    def record_result(self, x_raw: np.ndarray, y_up: int, used_in_live: bool,
                    p_pred: Optional[float] = None, reg_ctx: Optional[dict] = None) -> None:
        """
        Записывает результат предсказания и обновляет модель.
        
        Теперь включает:
        - Сохранение в глобальную и фазовую память
        - Трекинг хитов для оценки качества
        - Out-of-fold predictions для cross-validation
        - Периодическую CV проверку для валидации модели
        - Обучение модели при накоплении данных
        - Переключение режимов SHADOW/ACTIVE на основе метрик
        """
        
        # ========== БЛОК 1: ИНИЦИАЛИЗАЦИЯ И ПРОВЕРКА РАЗМЕРНОСТИ ==========
        # Убеждаемся, что размерность фичей корректна и инициализирована
        self._ensure_dim(x_raw)

        # ========== БЛОК 2: СОХРАНЕНИЕ В ГЛОБАЛЬНУЮ ПАМЯТЬ ==========
        # Глобальная память используется как fallback, когда в фазе мало данных
        self.X.append(x_raw.astype(np.float32).ravel().tolist())
        self.y.append(int(y_up))
        
        # Ограничиваем размер глобальной памяти, чтобы не раздувалась
        if len(self.X) > int(getattr(self.cfg, "max_memory", 10_000)):
            self.X = self.X[-self.cfg.max_memory:]
            self.y = self.y[-self.cfg.max_memory:]
        
        self.new_since_train += 1

        # ========== БЛОК 3: ОПРЕДЕЛЕНИЕ ФАЗЫ ==========
        # Извлекаем текущую фазу из контекста (0-5 для 6 фаз)
        ph = 0
        if isinstance(reg_ctx, dict):
            ph = int(reg_ctx.get("phase", 0))
        self._last_seen_phase = ph

        # ========== БЛОК 4: СОХРАНЕНИЕ В ФАЗОВУЮ ПАМЯТЬ ==========
        # Каждая фаза хранит свою собственную историю примеров
        # Это позволяет модели специализироваться на разных рыночных режимах
        self.X_ph[ph].append(x_raw.astype(np.float32).ravel().tolist())
        self.y_ph[ph].append(int(y_up))
        
        # Ограничиваем размер фазовой памяти
        cap = int(getattr(self.cfg, "phase_memory_cap", 10_000))
        if len(self.X_ph[ph]) > cap:
            self.X_ph[ph] = self.X_ph[ph][-cap:]
            self.y_ph[ph] = self.y_ph[ph][-cap:]
        
        self.new_since_train_ph[ph] = self.new_since_train_ph.get(ph, 0) + 1

        # ========== БЛОК 5: ТРЕКИНГ ХИТОВ И DRIFT DETECTION ==========
        # Оцениваем качество предсказания и отслеживаем дрейф концепции
        if p_pred is not None:
            try:
                # Считаем hit: правильно ли предсказали направление?
                hit = int((float(p_pred) >= 0.5) == bool(y_up))
                
                if self.mode == "ACTIVE" and used_in_live:
                    # В активном режиме отслеживаем реальные сделки
                    self.active_hits.append(hit)
                    
                    # ADWIN детектирует дрейф распределения ошибок
                    if self.adwin is not None:
                        in_drift = self.adwin.update(1 - hit)  # 1=correct, 0=error
                        if in_drift:
                            # Обнаружен дрейф - возвращаемся в shadow режим
                            self.mode = "SHADOW"
                            self.active_hits = []
                else:
                    # В shadow режиме накапливаем "что было бы, если бы входили"
                    self.shadow_hits.append(hit)
            except Exception:
                pass

        # ========== БЛОК 6: НОВОЕ - СОХРАНЕНИЕ OOF PREDICTIONS ДЛЯ CV ==========
        # Out-of-fold predictions нужны для расчета метрик cross-validation
        # Эти предсказания были сделаны на данных, которые модель НЕ видела при обучении
        if self.cfg.cv_enabled and p_pred is not None:
            self.cv_oof_preds[ph].append(float(p_pred))
            self.cv_oof_labels[ph].append(int(y_up))

        # ========== БЛОК 7: ФАЗОВАЯ КАЛИБРОВКА ==========
        # Калибратор корректирует вероятности для каждой фазы отдельно
        # Это важно, потому что модель может быть по-разному откалибрована в разных режимах
        try:
            p_raw = self._predict_raw(x_raw)
            if p_raw is not None:
                # Инициализируем калибратор для этой фазы, если его нет
                if self.cal_ph[ph] is None:
                    self.cal_ph[ph] = make_calibrator(self.cfg.xgb_calibration_method)
                
                # Показываем калибратору истинную пару (предсказание, результат)
                self.cal_ph[ph].observe(float(p_raw), int(y_up))
                
                # Периодически пересчитываем калибровку
                if self.cal_ph[ph].maybe_fit(min_samples=200, every=100):
                    cal_path = self._cal_path(self.cfg.xgb_cal_path, ph)
                    self.cal_ph[ph].save(cal_path)
        except Exception:
            pass

        # ========== БЛОК 8: НОВОЕ - ПЕРИОДИЧЕСКАЯ CV ПРОВЕРКА ==========
        # Каждые N примеров запускаем полную cross-validation для оценки реального качества
        # Это защищает от переобучения и дает честную оценку обобщающей способности
        self.cv_last_check[ph] += 1
        
        if self.cfg.cv_enabled and self.cv_last_check[ph] >= self.cfg.cv_check_every:
            # Сбрасываем счетчик
            self.cv_last_check[ph] = 0
            
            # Запускаем полную walk-forward cross-validation с purging
            cv_results = self._run_cv_validation(ph)
            
            # Сохраняем результаты для использования в _maybe_flip_modes
            self.cv_metrics[ph] = cv_results
            
            # Если CV прошла успешно, помечаем фазу как валидированную
            if cv_results.get("status") == "ok":
                self.validation_passed[ph] = True
            
            # Логируем результаты для мониторинга
            if cv_results.get("status") == "ok":
                print(f"[{self.__class__.__name__}] CV ph={ph}: "
                    f"OOF_ACC={cv_results['oof_accuracy']:.2f}% "
                    f"CI=[{cv_results['ci_lower']:.2f}%, {cv_results['ci_upper']:.2f}%] "
                    f"folds={cv_results['n_folds']}")

        # ========== БЛОК 9: ОБУЧЕНИЕ МОДЕЛИ ПО ФАЗЕ ==========
        # Когда накопилось достаточно новых примеров в фазе, запускаем переобучение
        self._maybe_train_phase(ph)

        # ========== БЛОК 10: ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ ==========
        # Проверяем метрики (включая CV) и решаем, переключать ли SHADOW ↔ ACTIVE
        self._maybe_flip_modes()
        
        # ========== БЛОК 11: СОХРАНЕНИЕ СОСТОЯНИЯ ==========
        # Периодически сохраняем все на диск для восстановления после перезапуска
        self._save_all()

    # ---------- режимы ----------
    def _maybe_flip_modes(self):
        """
        Улучшенное переключение режимов с учётом:
        1. Cross-validation метрик
        2. Статистической значимости (bootstrap CI)
        3. Out-of-fold predictions
        """
        if not self.cfg.cv_enabled:
            # fallback к старой логике
            self._maybe_flip_modes_simple()
            return
        
        def wr(arr, n):
            if len(arr) < n: return None
            window = arr[-n:]
            return 100.0 * (sum(window)/len(window))
        
        # Текущие метрики
        wr_shadow = wr(self.shadow_hits, self.cfg.min_ready)
        wr_active = wr(self.active_hits, max(30, self.cfg.min_ready // 2))
        
        # Получаем CV метрики текущей фазы
        ph = self._last_seen_phase
        cv_metrics = self.cv_metrics.get(ph, {})
        cv_passed = self.validation_passed.get(ph, False)
        
        # SHADOW → ACTIVE: требуем CV validation + bootstrap CI
        if self.mode == "SHADOW" and wr_shadow is not None:
            basic_threshold_met = wr_shadow >= self.cfg.enter_wr
            
            if basic_threshold_met and cv_passed:
                # Проверяем статистическую значимость
                cv_wr = cv_metrics.get("oof_accuracy", 0.0)
                cv_ci_lower = cv_metrics.get("ci_lower", 0.0)
                
                # Нужно: OOF accuracy > порог И нижняя граница CI тоже
                if cv_wr >= self.cfg.enter_wr and cv_ci_lower >= (self.cfg.enter_wr - self.cfg.cv_min_improvement):
                    self.mode = "ACTIVE"
                    if HAVE_RIVER:
                        self.adwin = ADWIN(delta=self.cfg.adwin_delta)
                    print(f"[{self.__class__.__name__}] SHADOW→ACTIVE ph={ph}: WR={wr_shadow:.2f}%, CV_WR={cv_wr:.2f}% (CI: [{cv_ci_lower:.2f}%, {cv_metrics.get('ci_upper', 0):.2f}%])")
        
        # ACTIVE → SHADOW: детектируем деградацию
        if self.mode == "ACTIVE" and wr_active is not None:
            basic_threshold_failed = wr_active < self.cfg.exit_wr
            
            # Также проверяем CV метрики на деградацию
            cv_wr = cv_metrics.get("oof_accuracy", 100.0)
            cv_degraded = cv_wr < self.cfg.exit_wr
            
            if basic_threshold_failed or cv_degraded:
                self.mode = "SHADOW"
                self.validation_passed[ph] = False
                print(f"[{self.__class__.__name__}] ACTIVE→SHADOW ph={ph}: WR={wr_active:.2f}%, CV_WR={cv_wr:.2f}%")

    def _maybe_flip_modes_simple(self):
        """Старая логика для backward compatibility"""
        def wr(arr, n):
            if len(arr) < n: return None
            window = arr[-n:]
            return 100.0 * (sum(window)/len(window))
        wr_shadow = wr(self.shadow_hits, self.cfg.min_ready)
        if self.mode == "SHADOW" and wr_shadow is not None and wr_shadow >= self.cfg.enter_wr:
            self.mode = "ACTIVE"
            if HAVE_RIVER:
                self.adwin = ADWIN(delta=self.cfg.adwin_delta)
        wr_active = wr(self.active_hits, max(30, self.cfg.min_ready // 2))
        if self.mode == "ACTIVE" and (wr_active is not None and wr_active < self.cfg.exit_wr):
            self.mode = "SHADOW"

    # ---------- совместимая обёртка ----------
    def maybe_train(self, ph: Optional[int] = None, reg_ctx: Optional[dict] = None) -> None:
        """Тренируем по текущей фазе (без «глобального» рефита)."""
        if not self.enabled or self.n_feats is None:
            return
        if ph is None:
            if isinstance(reg_ctx, dict) and "phase" in reg_ctx:
                ph = int(reg_ctx["phase"])
            else:
                ph = int(getattr(self, "_last_seen_phase", 0))
        self._maybe_train_phase(int(ph))

    # ---------- статус ----------
    def status(self):
        def _wr(xs):
            if not xs: return None
            return sum(xs) / float(len(xs))
        def _fmt_pct(p):
            return "—" if p is None else f"{100.0*p:.2f}%"
        
        wr_a = _wr(self.active_hits)
        wr_s = _wr(self.shadow_hits)
        all_hits = (self.active_hits or []) + (self.shadow_hits or [])
        wr_all = _wr(all_hits)
        
        # CV метрики текущей фазы
        ph = self._last_seen_phase
        cv_metrics = self.cv_metrics.get(ph, {})
        cv_status = cv_metrics.get("status", "N/A")
        cv_wr = cv_metrics.get("oof_accuracy", 0.0)
        cv_ci = f"[{cv_metrics.get('ci_lower', 0):.1f}%, {cv_metrics.get('ci_upper', 0):.1f}%]" if cv_status == "ok" else "N/A"
        
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "wr_active": _fmt_pct(wr_a),
            "n_active": len(self.active_hits or []),
            "wr_shadow": _fmt_pct(wr_s),
            "n_shadow": len(self.shadow_hits or []),
            "wr_all": _fmt_pct(wr_all),
            "n": len(all_hits),
            "cv_oof_wr": _fmt_pct(cv_wr / 100.0) if cv_wr > 0 else "—",
            "cv_ci": cv_ci,
            "cv_validated": str(self.validation_passed.get(ph, False))
        }





# ---------- RF ----------
class RFCalibratedExpert(_BaseExpert):
    def __init__(self, cfg: MLConfig):
        self.cfg = cfg
        self.enabled = HAVE_SKLEARN
        self.mode = "SHADOW"

        # Калиброванный RF (калибровка внутри CalibratedClassifierCV)
        self.clf: Optional[CalibratedClassifierCV] = None
        # --- НОВОЕ: модели по фазам ---
        self.clf_ph: Dict[int, Optional[CalibratedClassifierCV]] = {}


        # Детектор дрейфа (как было)
        self.adwin = ADWIN(delta=self.cfg.adwin_delta) if HAVE_RIVER else None

        # ===== ГЛОБАЛЬНАЯ ПАМЯТЬ (хвост) =====
        self.X: List[List[float]] = []
        self.y: List[int] = []
        self.new_since_train: int = 0

        # ===== ФАЗОВАЯ ПАМЯТЬ =====
        self.P: int = int(self.cfg.phase_count)  # 6 фаз: bull/bear/flat × low/high
        self.X_ph: Dict[int, List[List[float]]] = {p: [] for p in range(self.P)}
        self.y_ph: Dict[int, List[int]]         = {p: [] for p in range(self.P)}
        self.new_since_train_ph: Dict[int, int] = {p: 0  for p in range(self.P)}

        self.clf_ph = {p: None for p in range(self.P)}
        # Последняя стабильная фаза — пригодится, если maybe_train() вызовут без reg_ctx
        self._last_seen_phase: int = 0

        # Хиты/диагностика
        self.shadow_hits: List[int] = []
        self.active_hits: List[int] = []
        # Cross-validation tracking (per phase)
        self.cv_oof_preds: Dict[int, deque] = {p: deque(maxlen=cfg.cv_oof_window) for p in range(self.P)}
        self.cv_oof_labels: Dict[int, deque] = {p: deque(maxlen=cfg.cv_oof_window) for p in range(self.P)}
        self.cv_metrics: Dict[int, Dict] = {p: {} for p in range(self.P)}
        self.cv_last_check: Dict[int, int] = {p: 0 for p in range(self.P)}
        
        # Validation mode tracking
        self.validation_passed: Dict[int, bool] = {p: False for p in range(self.P)}


        # Техническое: число фич (заполним при первом вызове)
        self.n_feats: Optional[int] = None

        # Загрузка стейта (если сериализуете память/модель)
        self._load_all()

    # ---------- ВСПОМОГАТЕЛЬНЫЕ ----------
    def _get_global_tail(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n <= 0 or not self.X:
            return np.empty((0, self.n_feats or 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        Xg = np.array(self.X[-n:], dtype=np.float32)
        yg = np.array(self.y[-n:], dtype=np.int32)
        return Xg, yg

    def _get_past_phases_tail(self, ph: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        # Берет последние n записей только из фаз 0..ph (включительно)
        # Исключает данные из будущих фаз → нет утечки
        if n <= 0:
            return np.empty((0, self.n_feats or 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        
        # Собираем все данные из фаз 0..ph
        X_past = []
        y_past = []
        for p in range(min(ph + 1, self.P)):
            if self.X_ph.get(p):
                X_past.extend(self.X_ph[p])
                y_past.extend(self.y_ph[p])
        
        if not X_past:
            return np.empty((0, self.n_feats or 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        
        # Берем последние n записей
        X_past = X_past[-n:]
        y_past = y_past[-n:]
        
        return np.array(X_past, dtype=np.float32), np.array(y_past, dtype=np.int32)

    def _get_phase_train(self, ph: int) -> Tuple[np.ndarray, np.ndarray]:
        # X_phase
        Xp = np.array(self.X_ph[ph], dtype=np.float32) if self.X_ph[ph] else np.empty((0, self.n_feats or 0), dtype=np.float32)
        yp = np.array(self.y_ph[ph], dtype=np.int32)   if self.y_ph[ph]  else np.empty((0,), dtype=np.int32)

        if len(Xp) >= int(self.cfg.phase_min_ready):
            return Xp, yp

        # иначе смешиваем X_phase ∪ X_past_phases_tail (70/30 по умолчанию)
        # ИСПРАВЛЕНИЕ: используем только прошлые фазы (0..ph), а не весь глобальный хвост
        share = float(self.cfg.phase_mix_global_share)  # 0.30
        need_g = int(round(len(Xp) * share / max(1e-9, (1.0 - share))))
        need_g = max(need_g, int(self.cfg.phase_min_ready) - len(Xp))   # не менее, чтобы достичь порога
        
        # Считаем сколько доступно в фазах 0..ph
        available_past = sum(len(self.X_ph.get(p, [])) for p in range(min(ph + 1, self.P)))
        need_g = min(need_g, available_past)  # не больше, чем доступно в прошлых фазах
        
        Xg, yg = self._get_past_phases_tail(ph, need_g)
        if len(Xg) == 0:
            return Xp, yp

        X = np.concatenate([Xp, Xg], axis=0)
        y = np.concatenate([yp, yg], axis=0)
        return X, y

    def _maybe_train_phase(self, ph: int) -> None:
        # тренируем ровно по фазе ph (с подмешиванием прошлых фаз 0..ph при нехватке)
        if self.n_feats is None or not self.enabled:
            return
        if self.new_since_train_ph.get(ph, 0) < int(self.cfg.retrain_every):
            return

        X_all, y_all = self._get_phase_train(ph)
        if len(X_all) < int(self.cfg.phase_min_ready):
            return


        # ограничим окно обучения
        if len(X_all) > int(self.cfg.train_window):
            X_all = X_all[-int(self.cfg.train_window):]
            y_all = y_all[-int(self.cfg.train_window):]

        try:
            # (пере)инициализация модели при необходимости
            if self.clf is None:
                # (пере)инициализация МОДЕЛИ ДЛЯ КОНКРЕТНОЙ ФАЗЫ
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.calibration import CalibratedClassifierCV

                model = self.clf_ph.get(ph)
                if model is None:
                    base = RandomForestClassifier(
                        n_estimators=getattr(self.cfg, "rf_n_estimators", 300),
                        max_depth=getattr(self.cfg, "rf_max_depth", None),
                        min_samples_leaf=getattr(self.cfg, "rf_min_samples_leaf", 2),
                        n_jobs=-1,
                        random_state=42,
                        class_weight=None
                    )
                    cal_method = getattr(self.cfg, "rf_calibration_method", "sigmoid")
                    try:
                        model = CalibratedClassifierCV(estimator=base, method=cal_method, cv=3)
                    except TypeError:
                        model = CalibratedClassifierCV(base_estimator=base, method=cal_method, cv=3)

                # обучение на фазовом батче
                model.fit(X_all, y_all)

                # сохранить в контейнер фаз
                self.clf_ph[ph] = model
                # для обратной совместимости оставим ссылку на "последнюю обученную"
                self.clf = model


            self.new_since_train_ph[ph] = 0
            # опционально: self._save_all()
        except Exception as e:
            print(f"[rf  ] train error (ph={ph}): {e}")

    def _run_cv_validation(self, ph: int) -> Dict:
        """
        Walk-forward purged cross-validation для фазы ph.
        Возвращает метрики: accuracy, CI bounds, fold scores.
        """
        X_all, y_all = self._get_phase_train(ph)
        
        if len(X_all) < self.cfg.cv_min_train_size:
            return {"status": "insufficient_data", "oof_accuracy": 0.0}
        
        n_samples = len(X_all)
        n_splits = min(self.cfg.cv_n_splits, n_samples // self.cfg.cv_min_train_size)
        
        if n_splits < 2:
            return {"status": "insufficient_splits", "oof_accuracy": 0.0}
        
        # Walk-forward splits с purge и embargo
        embargo_size = max(1, int(n_samples * self.cfg.cv_embargo_pct))
        purge_size = max(1, int(n_samples * self.cfg.cv_purge_pct))
        
        fold_size = n_samples // n_splits
        oof_preds = np.zeros(n_samples)
        oof_mask = np.zeros(n_samples, dtype=bool)
        fold_scores = []
        
        for fold_idx in range(n_splits):
            # Test fold
            test_start = fold_idx * fold_size
            test_end = min(test_start + fold_size, n_samples)
            
            # Train: всё до (test_start - purge_size)
            train_end = max(0, test_start - purge_size)
            
            if train_end < self.cfg.cv_min_train_size:
                continue
            
            X_train = X_all[:train_end]
            y_train = y_all[:train_end]
            X_test = X_all[test_start:test_end]
            y_test = y_all[test_start:test_end]
            
            # Обучаем временную модель на train fold
            temp_model = self._train_fold_model(X_train, y_train, ph)
            
            # Предсказания на test fold
            preds = self._predict_fold(temp_model, X_test, ph)
            
            # Сохраняем OOF predictions
            oof_preds[test_start:test_end] = preds
            oof_mask[test_start:test_end] = True
            
            # Метрики фолда
            fold_acc = np.mean((preds >= 0.5) == y_test)
            fold_scores.append(fold_acc)
        
        # Итоговые OOF метрики
        oof_valid = oof_mask.sum()
        if oof_valid < self.cfg.cv_min_train_size:
            return {"status": "insufficient_oof", "oof_accuracy": 0.0}
        
        oof_accuracy = 100.0 * np.mean((oof_preds[oof_mask] >= 0.5) == y_all[oof_mask])
        
        # Bootstrap confidence intervals
        ci_lower, ci_upper = self._bootstrap_ci(
            oof_preds[oof_mask], 
            y_all[oof_mask],
            n_bootstrap=self.cfg.cv_bootstrap_n,
            confidence=self.cfg.cv_confidence
        )
        
        return {
            "status": "ok",
            "oof_accuracy": oof_accuracy,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "fold_scores": fold_scores,
            "n_folds": len(fold_scores),
            "oof_samples": int(oof_valid)
        }

    def _bootstrap_ci(self, preds: np.ndarray, labels: np.ndarray, 
                    n_bootstrap: int, confidence: float) -> tuple:
        """
        Bootstrap confidence intervals для accuracy.
        """
        accuracies = []
        n = len(preds)
        
        for _ in range(n_bootstrap):
            # Resample с возвратом
            idx = np.random.choice(n, size=n, replace=True)
            boot_preds = preds[idx]
            boot_labels = labels[idx]
            boot_acc = 100.0 * np.mean((boot_preds >= 0.5) == boot_labels)
            accuracies.append(boot_acc)
        
        accuracies = np.array(accuracies)
        alpha = 1.0 - confidence
        ci_lower = np.percentile(accuracies, 100 * alpha / 2)
        ci_upper = np.percentile(accuracies, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper

    def _train_fold_model(self, X: np.ndarray, y: np.ndarray, ph: int):
        """
        Обучает временную модель для CV fold.
        Реализация зависит от типа эксперта (XGB/RF/ARF/NN).
        """
        # Пример для XGB
        if not HAVE_XGB:
            return None
        
        scaler = StandardScaler().fit(X) if HAVE_SKLEARN else None
        X_scaled = scaler.transform(X) if scaler else X
        
        dtrain = xgb.DMatrix(X_scaled, label=y)
        model = xgb.train(
            params={
                "objective": "binary:logistic",
                "max_depth": self.cfg.xgb_max_depth,
                "eta": self.cfg.xgb_eta,
                "subsample": self.cfg.xgb_subsample,
                "colsample_bytree": self.cfg.xgb_colsample_bytree,
                "min_child_weight": self.cfg.xgb_min_child_weight,
                "eval_metric": "logloss",
            },
            dtrain=dtrain,
            num_boost_round=self.cfg.xgb_rounds_warm,
            verbose_eval=False
        )
        
        return {"model": model, "scaler": scaler}

    def _predict_fold(self, fold_model, X: np.ndarray, ph: int) -> np.ndarray:
        """
        Предсказания временной модели CV fold.
        """
        if fold_model is None:
            return np.full(len(X), 0.5)
        
        scaler = fold_model.get("scaler")
        model = fold_model.get("model")
        
        X_scaled = scaler.transform(X) if scaler else X
        dtest = xgb.DMatrix(X_scaled)
        preds = model.predict(dtest)
        
        return preds

    def _ensure_dim(self, x_raw: np.ndarray):
        d = int(x_raw.reshape(1, -1).shape[1])
        if self.n_feats is None:
            self.n_feats = d
            self.X, self.y = [], []
            self.clf = None
            self.new_since_train = 0
            # также обнулим фазовые буферы и счётчики
            self.X_ph = {p: [] for p in range(self.P)}
            self.y_ph = {p: [] for p in range(self.P)}
            self.new_since_train_ph = {p: 0 for p in range(self.P)}
            self._last_seen_phase = 0
        elif self.n_feats != d:
            # смена размерности — сбросить все накопленные данные
            self.n_feats = d
            self.X, self.y = [], []
            self.clf = None
            self.new_since_train = 0
            self.X_ph = {p: [] for p in range(self.P)}
            self.y_ph = {p: [] for p in range(self.P)}
            self.new_since_train_ph = {p: 0 for p in range(self.P)}
            self._last_seen_phase = 0

    # ---------- ЗАГРУЗКА/СОХРАНЕНИЕ ----------
    def _load_all(self) -> None:
        # --- загружаем состояние эксперта (JSON) ---
        try:
            if os.path.exists(self.cfg.rf_state_path):
                with open(self.cfg.rf_state_path, "r") as f:
                    st = json.load(f)

                # Базовые поля (как раньше)
                self.mode = st.get("mode", self.mode if hasattr(self, "mode") else "SHADOW")
                self.shadow_hits = st.get("shadow_hits", [])[-1000:]
                self.active_hits = st.get("active_hits", [])[-1000:]
                self.n_feats = st.get("n_feats", self.n_feats if hasattr(self, "n_feats") else None)

                # NEW: глобальная память (если есть в стейте)
                self.X = st.get("X", self.X if hasattr(self, "X") else [])
                self.y = st.get("y", self.y if hasattr(self, "y") else [])

                # NEW: фазовая память (ключи → int; если нет — создаём пустые буферы)
                X_ph = st.get("X_ph")
                y_ph = st.get("y_ph")
                if isinstance(X_ph, dict) and isinstance(y_ph, dict):
                    self.X_ph = {int(k): v for k, v in X_ph.items()}
                    self.y_ph = {int(k): v for k, v in y_ph.items()}
                else:
                    # старый формат без фаз — инициализируем по умолчанию
                    self.X_ph = {p: [] for p in range(self.P)}
                    self.y_ph = {p: [] for p in range(self.P)}

                # NEW: счётчики retrain по фазам
                self.new_since_train_ph = {p: 0 for p in range(self.P)}
                if isinstance(st.get("new_since_train_ph"), dict):
                    for k, v in st["new_since_train_ph"].items():
                        try:
                            self.new_since_train_ph[int(k)] = int(v)
                        except Exception:
                            pass

                # NEW: последняя увиденная фаза (для maybe_train без reg_ctx)
                try:
                    self._last_seen_phase = int(st.get("_last_seen_phase", 0))
                except Exception:
                    self._last_seen_phase = 0

                # Безопасные обрезки по капам (на случай старых больших стейтов)
                max_mem = getattr(self.cfg, "max_memory", None)
                if isinstance(max_mem, int) and max_mem > 0 and len(self.X) > max_mem:
                    self.X = self.X[-max_mem:]
                    self.y = self.y[-max_mem:]

                cap = int(self.cfg.phase_memory_cap)
                for p in range(self.P):
                    if len(self.X_ph.get(p, [])) > cap:
                        self.X_ph[p] = self.X_ph[p][-cap:]
                        self.y_ph[p] = self.y_ph[p][-cap:]
        except Exception as e:
            print(f"[rf  ] _load_all state error: {e}")

        # --- загружаем модель RF+калибратор (pickle) ---
        try:
            if os.path.exists(self.cfg.rf_model_path):
                with open(self.cfg.rf_model_path, "rb") as f:
                    self.clf = pickle.load(f)
        except Exception as e:
            print(f"[rf  ] _load_all model error: {e}")
            self.clf = None

        # --- НОВОЕ: загрузка фазовых моделей ---
        try:
            root, ext = os.path.splitext(self.cfg.rf_model_path)
            for p in range(self.P):
                ph_path = f"{root}_ph{p}{ext}"
                if os.path.exists(ph_path):
                    with open(ph_path, "rb") as f:
                        self.clf_ph[p] = pickle.load(f)
        except Exception as e:
            print(f"[rf  ] _load_all per-phase model error: {e}")

    def _save_all(self) -> None:
        # --- сохраняем состояние эксперта (JSON) ---
        try:
            # Перед сохранением гарантируем обрезку по капам
            max_mem = getattr(self.cfg, "max_memory", None)
            X_tail = self.X
            y_tail = self.y
            if isinstance(max_mem, int) and max_mem > 0:
                X_tail = self.X[-max_mem:]
                y_tail = self.y[-max_mem:]

            cap = int(self.cfg.phase_memory_cap)
            X_ph_tail: Dict[int, List[List[float]]] = {}
            y_ph_tail: Dict[int, List[int]] = {}
            for p in range(self.P):
                Xp = self.X_ph.get(p, [])
                yp = self.y_ph.get(p, [])
                X_ph_tail[p] = Xp[-cap:]
                y_ph_tail[p] = yp[-cap:]

            st = {
                # базовые поля
                "mode": self.mode,
                "shadow_hits": self.shadow_hits[-1000:],
                "active_hits": self.active_hits[-1000:],
                "n_feats": self.n_feats,

                # NEW: глобальная и фазовая память
                "X": X_tail,
                "y": y_tail,
                "X_ph": X_ph_tail,
                "y_ph": y_ph_tail,
                "new_since_train_ph": {int(p): int(self.new_since_train_ph.get(p, 0)) for p in range(self.P)},
                "_last_seen_phase": int(self._last_seen_phase),

                # на всякий случай — пишем P (для отладки/совместимости)
                "P": int(self.P),
            }

            with open(self.cfg.rf_state_path, "w") as f:
                json.dump(st, f)
        except Exception as e:
            print(f"[rf  ] _save_all state error: {e}")

        # --- сохраняем модель RF+калибратор (pickle) ---
        try:
            if self.clf is not None:
                with open(self.cfg.rf_model_path, "wb") as f:
                    pickle.dump(self.clf, f)
        except Exception as e:
            print(f"[rf  ] _save_all model error: {e}")

        # --- НОВОЕ: сохранение фазовых моделей ---
        try:
            root, ext = os.path.splitext(self.cfg.rf_model_path)
            for p, model in (self.clf_ph or {}).items():
                if model is None:
                    continue
                ph_path = f"{root}_ph{int(p)}{ext}"
                with open(ph_path, "wb") as f:
                    pickle.dump(model, f)
        except Exception as e:
            print(f"[rf  ] _save_all per-phase model error: {e}")


    # ---------- ИНФЕРЕНС / ОБУЧЕНИЕ ----------
    def proba_up(self, x_raw: np.ndarray, reg_ctx: Optional[dict] = None) -> Tuple[Optional[float], str]:
        # гарантируем корректную размерность
        try:
            self._ensure_dim(x_raw)
        except Exception:
            pass

        # выбрать модель фазы, при отсутствии — глобальную
        model = None
        if isinstance(reg_ctx, dict):
            ph = int(reg_ctx.get("phase", 0))
            self._last_seen_phase = ph
            model = self.clf_ph.get(ph)
        if model is None:
            model = self.clf

        if not self.enabled or model is None:
            return (None, self.mode)

        # определим фазу (стабилизированную — вы добавляете в reg_ctx["phase"])
        ph = 0
        if isinstance(reg_ctx, dict):
            ph = int(reg_ctx.get("phase", 0))
        self._last_seen_phase = ph

        xx = x_raw.astype(np.float32).reshape(1, -1)
        if self.n_feats is None:
            self.n_feats = xx.shape[1]

        try:
            p = float(model.predict_proba(xx)[0, 1])   # калибровка внутри
            p = max(1e-6, min(1.0 - 1e-6, p))
            return (p, self.mode)
        except Exception:
            return (None, self.mode)

    def record_result(self, x_raw: np.ndarray, y_up: int, used_in_live: bool,
                    p_pred: Optional[float] = None, reg_ctx: Optional[dict] = None) -> None:
        """
        Записывает результат предсказания и обновляет модель.
        
        Теперь включает:
        - Сохранение в глобальную и фазовую память
        - Трекинг хитов для оценки качества
        - Out-of-fold predictions для cross-validation
        - Периодическую CV проверку для валидации модели
        - Обучение модели при накоплении данных
        - Переключение режимов SHADOW/ACTIVE на основе метрик
        """
        
        # ========== БЛОК 1: ИНИЦИАЛИЗАЦИЯ И ПРОВЕРКА РАЗМЕРНОСТИ ==========
        # Убеждаемся, что размерность фичей корректна и инициализирована
        self._ensure_dim(x_raw)

        # ========== БЛОК 2: СОХРАНЕНИЕ В ГЛОБАЛЬНУЮ ПАМЯТЬ ==========
        # Глобальная память используется как fallback, когда в фазе мало данных
        self.X.append(x_raw.astype(np.float32).ravel().tolist())
        self.y.append(int(y_up))
        
        # Ограничиваем размер глобальной памяти, чтобы не раздувалась
        if len(self.X) > int(getattr(self.cfg, "max_memory", 10_000)):
            self.X = self.X[-self.cfg.max_memory:]
            self.y = self.y[-self.cfg.max_memory:]
        
        self.new_since_train += 1

        # ========== БЛОК 3: ОПРЕДЕЛЕНИЕ ФАЗЫ ==========
        # Извлекаем текущую фазу из контекста (0-5 для 6 фаз)
        ph = 0
        if isinstance(reg_ctx, dict):
            ph = int(reg_ctx.get("phase", 0))
        self._last_seen_phase = ph

        # ========== БЛОК 4: СОХРАНЕНИЕ В ФАЗОВУЮ ПАМЯТЬ ==========
        # Каждая фаза хранит свою собственную историю примеров
        # Это позволяет модели специализироваться на разных рыночных режимах
        self.X_ph[ph].append(x_raw.astype(np.float32).ravel().tolist())
        self.y_ph[ph].append(int(y_up))
        
        # Ограничиваем размер фазовой памяти
        cap = int(getattr(self.cfg, "phase_memory_cap", 10_000))
        if len(self.X_ph[ph]) > cap:
            self.X_ph[ph] = self.X_ph[ph][-cap:]
            self.y_ph[ph] = self.y_ph[ph][-cap:]
        
        self.new_since_train_ph[ph] = self.new_since_train_ph.get(ph, 0) + 1

        # ========== БЛОК 5: ТРЕКИНГ ХИТОВ И DRIFT DETECTION ==========
        # Оцениваем качество предсказания и отслеживаем дрейф концепции
        if p_pred is not None:
            try:
                # Считаем hit: правильно ли предсказали направление?
                hit = int((float(p_pred) >= 0.5) == bool(y_up))
                
                if self.mode == "ACTIVE" and used_in_live:
                    # В активном режиме отслеживаем реальные сделки
                    self.active_hits.append(hit)
                    
                    # ADWIN детектирует дрейф распределения ошибок
                    if self.adwin is not None:
                        in_drift = self.adwin.update(1 - hit)  # 1=correct, 0=error
                        if in_drift:
                            # Обнаружен дрейф - возвращаемся в shadow режим
                            self.mode = "SHADOW"
                            self.active_hits = []
                else:
                    # В shadow режиме накапливаем "что было бы, если бы входили"
                    self.shadow_hits.append(hit)
            except Exception:
                pass

        # ========== БЛОК 6: НОВОЕ - СОХРАНЕНИЕ OOF PREDICTIONS ДЛЯ CV ==========
        # Out-of-fold predictions нужны для расчета метрик cross-validation
        # Эти предсказания были сделаны на данных, которые модель НЕ видела при обучении
        if self.cfg.cv_enabled and p_pred is not None:
            self.cv_oof_preds[ph].append(float(p_pred))
            self.cv_oof_labels[ph].append(int(y_up))

        # ========== БЛОК 7: ФАЗОВАЯ КАЛИБРОВКА ==========
        # Калибратор корректирует вероятности для каждой фазы отдельно
        # Это важно, потому что модель может быть по-разному откалибрована в разных режимах
        try:
            p_raw = self._predict_raw(x_raw)
            if p_raw is not None:
                # Инициализируем калибратор для этой фазы, если его нет
                if self.cal_ph[ph] is None:
                    self.cal_ph[ph] = make_calibrator(self.cfg.xgb_calibration_method)
                
                # Показываем калибратору истинную пару (предсказание, результат)
                self.cal_ph[ph].observe(float(p_raw), int(y_up))
                
                # Периодически пересчитываем калибровку
                if self.cal_ph[ph].maybe_fit(min_samples=200, every=100):
                    cal_path = self._cal_path(self.cfg.xgb_cal_path, ph)
                    self.cal_ph[ph].save(cal_path)
        except Exception:
            pass

        # ========== БЛОК 8: НОВОЕ - ПЕРИОДИЧЕСКАЯ CV ПРОВЕРКА ==========
        # Каждые N примеров запускаем полную cross-validation для оценки реального качества
        # Это защищает от переобучения и дает честную оценку обобщающей способности
        self.cv_last_check[ph] += 1
        
        if self.cfg.cv_enabled and self.cv_last_check[ph] >= self.cfg.cv_check_every:
            # Сбрасываем счетчик
            self.cv_last_check[ph] = 0
            
            # Запускаем полную walk-forward cross-validation с purging
            cv_results = self._run_cv_validation(ph)
            
            # Сохраняем результаты для использования в _maybe_flip_modes
            self.cv_metrics[ph] = cv_results
            
            # Если CV прошла успешно, помечаем фазу как валидированную
            if cv_results.get("status") == "ok":
                self.validation_passed[ph] = True
            
            # Логируем результаты для мониторинга
            if cv_results.get("status") == "ok":
                print(f"[{self.__class__.__name__}] CV ph={ph}: "
                    f"OOF_ACC={cv_results['oof_accuracy']:.2f}% "
                    f"CI=[{cv_results['ci_lower']:.2f}%, {cv_results['ci_upper']:.2f}%] "
                    f"folds={cv_results['n_folds']}")

        # ========== БЛОК 9: ОБУЧЕНИЕ МОДЕЛИ ПО ФАЗЕ ==========
        # Когда накопилось достаточно новых примеров в фазе, запускаем переобучение
        self._maybe_train_phase(ph)

        # ========== БЛОК 10: ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ ==========
        # Проверяем метрики (включая CV) и решаем, переключать ли SHADOW ↔ ACTIVE
        self._maybe_flip_modes()
        
        # ========== БЛОК 11: СОХРАНЕНИЕ СОСТОЯНИЯ ==========
        # Периодически сохраняем все на диск для восстановления после перезапуска
        self._save_all()

    def _maybe_flip_modes(self):
        """
        Улучшенное переключение режимов с учётом:
        1. Cross-validation метрик
        2. Статистической значимости (bootstrap CI)
        3. Out-of-fold predictions
        """
        if not self.cfg.cv_enabled:
            # fallback к старой логике
            self._maybe_flip_modes_simple()
            return
        
        def wr(arr, n):
            if len(arr) < n: return None
            window = arr[-n:]
            return 100.0 * (sum(window)/len(window))
        
        # Текущие метрики
        wr_shadow = wr(self.shadow_hits, self.cfg.min_ready)
        wr_active = wr(self.active_hits, max(30, self.cfg.min_ready // 2))
        
        # Получаем CV метрики текущей фазы
        ph = self._last_seen_phase
        cv_metrics = self.cv_metrics.get(ph, {})
        cv_passed = self.validation_passed.get(ph, False)
        
        # SHADOW → ACTIVE: требуем CV validation + bootstrap CI
        if self.mode == "SHADOW" and wr_shadow is not None:
            basic_threshold_met = wr_shadow >= self.cfg.enter_wr
            
            if basic_threshold_met and cv_passed:
                # Проверяем статистическую значимость
                cv_wr = cv_metrics.get("oof_accuracy", 0.0)
                cv_ci_lower = cv_metrics.get("ci_lower", 0.0)
                
                # Нужно: OOF accuracy > порог И нижняя граница CI тоже
                if cv_wr >= self.cfg.enter_wr and cv_ci_lower >= (self.cfg.enter_wr - self.cfg.cv_min_improvement):
                    self.mode = "ACTIVE"
                    if HAVE_RIVER:
                        self.adwin = ADWIN(delta=self.cfg.adwin_delta)
                    print(f"[{self.__class__.__name__}] SHADOW→ACTIVE ph={ph}: WR={wr_shadow:.2f}%, CV_WR={cv_wr:.2f}% (CI: [{cv_ci_lower:.2f}%, {cv_metrics.get('ci_upper', 0):.2f}%])")
        
        # ACTIVE → SHADOW: детектируем деградацию
        if self.mode == "ACTIVE" and wr_active is not None:
            basic_threshold_failed = wr_active < self.cfg.exit_wr
            
            # Также проверяем CV метрики на деградацию
            cv_wr = cv_metrics.get("oof_accuracy", 100.0)
            cv_degraded = cv_wr < self.cfg.exit_wr
            
            if basic_threshold_failed or cv_degraded:
                self.mode = "SHADOW"
                self.validation_passed[ph] = False
                print(f"[{self.__class__.__name__}] ACTIVE→SHADOW ph={ph}: WR={wr_active:.2f}%, CV_WR={cv_wr:.2f}%")

    def _maybe_flip_modes_simple(self):
        """Старая логика для backward compatibility"""
        def wr(arr, n):
            if len(arr) < n: return None
            window = arr[-n:]
            return 100.0 * (sum(window)/len(window))
        wr_shadow = wr(self.shadow_hits, self.cfg.min_ready)
        if self.mode == "SHADOW" and wr_shadow is not None and wr_shadow >= self.cfg.enter_wr:
            self.mode = "ACTIVE"
            if HAVE_RIVER:
                self.adwin = ADWIN(delta=self.cfg.adwin_delta)
        wr_active = wr(self.active_hits, max(30, self.cfg.min_ready // 2))
        if self.mode == "ACTIVE" and (wr_active is not None and wr_active < self.cfg.exit_wr):
            self.mode = "SHADOW"

    def maybe_train(self, ph: Optional[int] = None, reg_ctx: Optional[dict] = None) -> None:
        """Совместимая обёртка: тренируем по текущей фазе (без глобального рефита)."""
        if not self.enabled or self.n_feats is None:
            return
        if ph is None:
            if isinstance(reg_ctx, dict) and "phase" in reg_ctx:
                ph = int(reg_ctx["phase"])
            else:
                ph = int(getattr(self, "_last_seen_phase", 0))
        self._maybe_train_phase(int(ph))



    def status(self):
        def _wr(xs):
            if not xs: return None
            return sum(xs) / float(len(xs))
        def _fmt_pct(p):
            return "—" if p is None else f"{100.0*p:.2f}%"
        
        wr_a = _wr(self.active_hits)
        wr_s = _wr(self.shadow_hits)
        all_hits = (self.active_hits or []) + (self.shadow_hits or [])
        wr_all = _wr(all_hits)
        
        # CV метрики текущей фазы
        ph = self._last_seen_phase
        cv_metrics = self.cv_metrics.get(ph, {})
        cv_status = cv_metrics.get("status", "N/A")
        cv_wr = cv_metrics.get("oof_accuracy", 0.0)
        cv_ci = f"[{cv_metrics.get('ci_lower', 0):.1f}%, {cv_metrics.get('ci_upper', 0):.1f}%]" if cv_status == "ok" else "N/A"
        
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "wr_active": _fmt_pct(wr_a),
            "n_active": len(self.active_hits or []),
            "wr_shadow": _fmt_pct(wr_s),
            "n_shadow": len(self.shadow_hits or []),
            "wr_all": _fmt_pct(wr_all),
            "n": len(all_hits),
            "cv_oof_wr": _fmt_pct(cv_wr / 100.0) if cv_wr > 0 else "—",
            "cv_ci": cv_ci,
            "cv_validated": str(self.validation_passed.get(ph, False))
        }



# ---------- ARF (River) ----------
class RiverARFExpert(_BaseExpert):
    def __init__(self, cfg: MLConfig):
        self.cfg = cfg
        self.enabled = HAVE_RIVER and (river_forest is not None)
        self.mode = "SHADOW"
        self.adwin = ADWIN(delta=self.cfg.adwin_delta) if HAVE_RIVER else None
        self.clf = None
        if self.enabled:
            try:
                self.clf = river_forest.ARFClassifier(n_models=self.cfg.arf_n_models, seed=42)
            except Exception:
                self.clf = None
                self.enabled = False

        self.shadow_hits: List[int] = []
        self.active_hits: List[int] = []

        from collections import deque
        self._seen_epochs = deque(maxlen=5000)   # ← кэш последних обработанных epoch

        # 👇 ДОБАВКА: загрузка/инициализация калибратора вероятностей ARF
        # __init__
        self.P = int(getattr(self.cfg, "phase_count", 6))
        self.cal_ph = {p: None for p in range(self.P)}
        self._last_seen_phase = 0

        def _cal_path(base: str, ph: int) -> str:
            root, ext = os.path.splitext(base)
            return f"{root}_ph{ph}{ext}"

        for p in range(self.P):
            try:
                self.cal_ph[p] = _BaseCal.load(_cal_path(self.cfg.arf_cal_path, p))
            except Exception:
                self.cal_ph[p] = None

        # Cross-validation tracking (per phase)
        self.cv_oof_preds: Dict[int, deque] = {p: deque(maxlen=cfg.cv_oof_window) for p in range(self.P)}
        self.cv_oof_labels: Dict[int, deque] = {p: deque(maxlen=cfg.cv_oof_window) for p in range(self.P)}
        self.cv_metrics: Dict[int, Dict] = {p: {} for p in range(self.P)}
        self.cv_last_check: Dict[int, int] = {p: 0 for p in range(self.P)}
        
        # Validation mode tracking
        self.validation_passed: Dict[int, bool] = {p: False for p in range(self.P)}

        self._load_all()

    def _ensure_dim(self, x_raw: np.ndarray):
        """Проверяет и обновляет размерность признаков при необходимости"""
        d = int(x_raw.reshape(1, -1).shape[1])
        if self.n_feats is None:
            self.n_feats = d
        elif self.n_feats != d:
            # смена размерности - сбрасываем модель
            self.n_feats = d
            self.clf = None
            if self.enabled:
                try:
                    self.clf = river_forest.ARFClassifier(n_models=self.cfg.arf_n_models, seed=42)
                except Exception:
                    self.clf = None
                    self.enabled = False

    def _load_all(self):
        try:
            if os.path.exists(self.cfg.arf_state_path):
                with open(self.cfg.arf_state_path, "r") as f:
                    st = json.load(f)
                self.mode = st.get("mode", "SHADOW")
                self.shadow_hits = st.get("shadow_hits", [])
                self.active_hits = st.get("active_hits", [])
        except Exception:
            pass
        if self.enabled:
            try:
                if os.path.exists(self.cfg.arf_model_path):
                    with open(self.cfg.arf_model_path, "rb") as f:
                        self.clf = pickle.load(f)
            except Exception:
                pass

    def _save_all(self):
        # 1) сохраняем state (режим и последние хиты)
        try:
            with open(self.cfg.arf_state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "mode": self.mode,
                    "shadow_hits": self.shadow_hits[-1000:],
                    "active_hits": self.active_hits[-1000:],
                }, f)
        except Exception:
            pass

        # 2) сохраняем модель ARF
        if self.enabled and self.clf is not None:
            try:
                with open(self.cfg.arf_model_path, "wb") as f:
                    pickle.dump(self.clf, f)
            except Exception:
                pass

        # 3) сохраняем калибратор вероятностей (если готов)
        # 3) сохраняем калибраторы вероятностей по фазам (если готовы)
        try:
            root, ext = os.path.splitext(self.cfg.arf_cal_path)
            for ph, cal in (self.cal_ph or {}).items():
                if cal is not None and getattr(cal, "ready", False):
                    cal_path = f"{root}_ph{int(ph)}{ext}"
                    try:
                        cal.save(cal_path)
                    except Exception:
                        pass
        except Exception:
            pass



    def _to_dict(self, x_raw: np.ndarray) -> Dict[str, float]:
        return {f"f{k}": float(v) for k, v in enumerate(x_raw.ravel().tolist())}


    def _predict_raw(self, x_raw: np.ndarray) -> Optional[float]:
        if not self.enabled or self.clf is None:
            return None
        pmap = self.clf.predict_proba_one(self._to_dict(x_raw))
        p = float(pmap.get(True, pmap.get(1, 0.5)))
        return float(min(max(p, 1e-6), 1.0 - 1e-6))


    def proba_up(self, x_raw: np.ndarray, reg_ctx: Optional[dict] = None) -> Tuple[Optional[float], str]:
        if not self.enabled or self.clf is None:
            return (None, "DISABLED" if not self.enabled else self.mode)
        try:
            # сырой прогноз
            p = self._predict_raw(x_raw)
            if p is None:
                return (None, self.mode)

            # фаза → фазовый калибратор
            ph = int(reg_ctx.get("phase")) if isinstance(reg_ctx, dict) and "phase" in reg_ctx else 0
            self._last_seen_phase = ph
            cal = self.cal_ph.get(ph)
            if cal is not None and getattr(cal, "ready", False):
                try:
                    p = float(cal.transform(float(p)))
                except Exception:
                    pass

            p = float(min(max(p, 1e-6), 1.0 - 1e-6))
            return (p, self.mode)
        except Exception:
            return (None, self.mode)


    def record_result(self, x_raw: np.ndarray, y_up: int, used_in_live: bool,
                    p_pred: Optional[float] = None, reg_ctx: Optional[dict] = None) -> None:
        """
        Записывает результат предсказания и обновляет модель.
        
        Теперь включает:
        - Сохранение в глобальную и фазовую память
        - Трекинг хитов для оценки качества
        - Out-of-fold predictions для cross-validation
        - Периодическую CV проверку для валидации модели
        - Обучение модели при накоплении данных
        - Переключение режимов SHADOW/ACTIVE на основе метрик
        """
        
        # ========== БЛОК 1: ИНИЦИАЛИЗАЦИЯ И ПРОВЕРКА РАЗМЕРНОСТИ ==========
        # Убеждаемся, что размерность фичей корректна и инициализирована
        self._ensure_dim(x_raw)

        # ========== БЛОК 2: СОХРАНЕНИЕ В ГЛОБАЛЬНУЮ ПАМЯТЬ ==========
        # Глобальная память используется как fallback, когда в фазе мало данных
        self.X.append(x_raw.astype(np.float32).ravel().tolist())
        self.y.append(int(y_up))
        
        # Ограничиваем размер глобальной памяти, чтобы не раздувалась
        if len(self.X) > int(getattr(self.cfg, "max_memory", 10_000)):
            self.X = self.X[-self.cfg.max_memory:]
            self.y = self.y[-self.cfg.max_memory:]
        
        self.new_since_train += 1

        # ========== БЛОК 3: ОПРЕДЕЛЕНИЕ ФАЗЫ ==========
        # Извлекаем текущую фазу из контекста (0-5 для 6 фаз)
        ph = 0
        if isinstance(reg_ctx, dict):
            ph = int(reg_ctx.get("phase", 0))
        self._last_seen_phase = ph

        # ========== БЛОК 4: СОХРАНЕНИЕ В ФАЗОВУЮ ПАМЯТЬ ==========
        # Каждая фаза хранит свою собственную историю примеров
        # Это позволяет модели специализироваться на разных рыночных режимах
        self.X_ph[ph].append(x_raw.astype(np.float32).ravel().tolist())
        self.y_ph[ph].append(int(y_up))
        
        # Ограничиваем размер фазовой памяти
        cap = int(getattr(self.cfg, "phase_memory_cap", 10_000))
        if len(self.X_ph[ph]) > cap:
            self.X_ph[ph] = self.X_ph[ph][-cap:]
            self.y_ph[ph] = self.y_ph[ph][-cap:]
        
        self.new_since_train_ph[ph] = self.new_since_train_ph.get(ph, 0) + 1

        # ========== БЛОК 5: ТРЕКИНГ ХИТОВ И DRIFT DETECTION ==========
        # Оцениваем качество предсказания и отслеживаем дрейф концепции
        if p_pred is not None:
            try:
                # Считаем hit: правильно ли предсказали направление?
                hit = int((float(p_pred) >= 0.5) == bool(y_up))
                
                if self.mode == "ACTIVE" and used_in_live:
                    # В активном режиме отслеживаем реальные сделки
                    self.active_hits.append(hit)
                    
                    # ADWIN детектирует дрейф распределения ошибок
                    if self.adwin is not None:
                        in_drift = self.adwin.update(1 - hit)  # 1=correct, 0=error
                        if in_drift:
                            # Обнаружен дрейф - возвращаемся в shadow режим
                            self.mode = "SHADOW"
                            self.active_hits = []
                else:
                    # В shadow режиме накапливаем "что было бы, если бы входили"
                    self.shadow_hits.append(hit)
            except Exception:
                pass

        # ========== БЛОК 6: НОВОЕ - СОХРАНЕНИЕ OOF PREDICTIONS ДЛЯ CV ==========
        # Out-of-fold predictions нужны для расчета метрик cross-validation
        # Эти предсказания были сделаны на данных, которые модель НЕ видела при обучении
        if self.cfg.cv_enabled and p_pred is not None:
            self.cv_oof_preds[ph].append(float(p_pred))
            self.cv_oof_labels[ph].append(int(y_up))

        # ========== БЛОК 7: ФАЗОВАЯ КАЛИБРОВКА ==========
        # Калибратор корректирует вероятности для каждой фазы отдельно
        # Это важно, потому что модель может быть по-разному откалибрована в разных режимах
        try:
            p_raw = self._predict_raw(x_raw)
            if p_raw is not None:
                # Инициализируем калибратор для этой фазы, если его нет
                if self.cal_ph[ph] is None:
                    self.cal_ph[ph] = make_calibrator(self.cfg.xgb_calibration_method)
                
                # Показываем калибратору истинную пару (предсказание, результат)
                self.cal_ph[ph].observe(float(p_raw), int(y_up))
                
                # Периодически пересчитываем калибровку
                if self.cal_ph[ph].maybe_fit(min_samples=200, every=100):
                    cal_path = self._cal_path(self.cfg.xgb_cal_path, ph)
                    self.cal_ph[ph].save(cal_path)
        except Exception:
            pass

        # ========== БЛОК 8: НОВОЕ - ПЕРИОДИЧЕСКАЯ CV ПРОВЕРКА ==========
        # Каждые N примеров запускаем полную cross-validation для оценки реального качества
        # Это защищает от переобучения и дает честную оценку обобщающей способности
        self.cv_last_check[ph] += 1
        
        if self.cfg.cv_enabled and self.cv_last_check[ph] >= self.cfg.cv_check_every:
            # Сбрасываем счетчик
            self.cv_last_check[ph] = 0
            
            # Запускаем полную walk-forward cross-validation с purging
            cv_results = self._run_cv_validation(ph)
            
            # Сохраняем результаты для использования в _maybe_flip_modes
            self.cv_metrics[ph] = cv_results
            
            # Если CV прошла успешно, помечаем фазу как валидированную
            if cv_results.get("status") == "ok":
                self.validation_passed[ph] = True
            
            # Логируем результаты для мониторинга
            if cv_results.get("status") == "ok":
                print(f"[{self.__class__.__name__}] CV ph={ph}: "
                    f"OOF_ACC={cv_results['oof_accuracy']:.2f}% "
                    f"CI=[{cv_results['ci_lower']:.2f}%, {cv_results['ci_upper']:.2f}%] "
                    f"folds={cv_results['n_folds']}")

        # ========== БЛОК 9: ОБУЧЕНИЕ МОДЕЛИ ПО ФАЗЕ ==========
        # Когда накопилось достаточно новых примеров в фазе, запускаем переобучение
        self._maybe_train_phase(ph)

        # ========== БЛОК 10: ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ ==========
        # Проверяем метрики (включая CV) и решаем, переключать ли SHADOW ↔ ACTIVE
        self._maybe_flip_modes()
        
        # ========== БЛОК 11: СОХРАНЕНИЕ СОСТОЯНИЯ ==========
        # Периодически сохраняем все на диск для восстановления после перезапуска
        self._save_all()

    def _run_cv_validation(self, ph: int) -> Dict:
        """
        Walk-forward purged cross-validation для фазы ph.
        Возвращает метрики: accuracy, CI bounds, fold scores.
        """
        X_all, y_all = self._get_phase_train(ph)
        
        if len(X_all) < self.cfg.cv_min_train_size:
            return {"status": "insufficient_data", "oof_accuracy": 0.0}
        
        n_samples = len(X_all)
        n_splits = min(self.cfg.cv_n_splits, n_samples // self.cfg.cv_min_train_size)
        
        if n_splits < 2:
            return {"status": "insufficient_splits", "oof_accuracy": 0.0}
        
        # Walk-forward splits с purge и embargo
        embargo_size = max(1, int(n_samples * self.cfg.cv_embargo_pct))
        purge_size = max(1, int(n_samples * self.cfg.cv_purge_pct))
        
        fold_size = n_samples // n_splits
        oof_preds = np.zeros(n_samples)
        oof_mask = np.zeros(n_samples, dtype=bool)
        fold_scores = []
        
        for fold_idx in range(n_splits):
            # Test fold
            test_start = fold_idx * fold_size
            test_end = min(test_start + fold_size, n_samples)
            
            # Train: всё до (test_start - purge_size)
            train_end = max(0, test_start - purge_size)
            
            if train_end < self.cfg.cv_min_train_size:
                continue
            
            X_train = X_all[:train_end]
            y_train = y_all[:train_end]
            X_test = X_all[test_start:test_end]
            y_test = y_all[test_start:test_end]
            
            # Обучаем временную модель на train fold
            temp_model = self._train_fold_model(X_train, y_train, ph)
            
            # Предсказания на test fold
            preds = self._predict_fold(temp_model, X_test, ph)
            
            # Сохраняем OOF predictions
            oof_preds[test_start:test_end] = preds
            oof_mask[test_start:test_end] = True
            
            # Метрики фолда
            fold_acc = np.mean((preds >= 0.5) == y_test)
            fold_scores.append(fold_acc)
        
        # Итоговые OOF метрики
        oof_valid = oof_mask.sum()
        if oof_valid < self.cfg.cv_min_train_size:
            return {"status": "insufficient_oof", "oof_accuracy": 0.0}
        
        oof_accuracy = 100.0 * np.mean((oof_preds[oof_mask] >= 0.5) == y_all[oof_mask])
        
        # Bootstrap confidence intervals
        ci_lower, ci_upper = self._bootstrap_ci(
            oof_preds[oof_mask], 
            y_all[oof_mask],
            n_bootstrap=self.cfg.cv_bootstrap_n,
            confidence=self.cfg.cv_confidence
        )
        
        return {
            "status": "ok",
            "oof_accuracy": oof_accuracy,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "fold_scores": fold_scores,
            "n_folds": len(fold_scores),
            "oof_samples": int(oof_valid)
        }

    def _bootstrap_ci(self, preds: np.ndarray, labels: np.ndarray, 
                    n_bootstrap: int, confidence: float) -> tuple:
        """
        Bootstrap confidence intervals для accuracy.
        """
        accuracies = []
        n = len(preds)
        
        for _ in range(n_bootstrap):
            # Resample с возвратом
            idx = np.random.choice(n, size=n, replace=True)
            boot_preds = preds[idx]
            boot_labels = labels[idx]
            boot_acc = 100.0 * np.mean((boot_preds >= 0.5) == boot_labels)
            accuracies.append(boot_acc)
        
        accuracies = np.array(accuracies)
        alpha = 1.0 - confidence
        ci_lower = np.percentile(accuracies, 100 * alpha / 2)
        ci_upper = np.percentile(accuracies, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper

    def _train_fold_model(self, X: np.ndarray, y: np.ndarray, ph: int):
        """
        Обучает временную модель для CV fold.
        Реализация зависит от типа эксперта (XGB/RF/ARF/NN).
        """
        # Пример для XGB
        if not HAVE_XGB:
            return None
        
        scaler = StandardScaler().fit(X) if HAVE_SKLEARN else None
        X_scaled = scaler.transform(X) if scaler else X
        
        dtrain = xgb.DMatrix(X_scaled, label=y)
        model = xgb.train(
            params={
                "objective": "binary:logistic",
                "max_depth": self.cfg.xgb_max_depth,
                "eta": self.cfg.xgb_eta,
                "subsample": self.cfg.xgb_subsample,
                "colsample_bytree": self.cfg.xgb_colsample_bytree,
                "min_child_weight": self.cfg.xgb_min_child_weight,
                "eval_metric": "logloss",
            },
            dtrain=dtrain,
            num_boost_round=self.cfg.xgb_rounds_warm,
            verbose_eval=False
        )
        
        return {"model": model, "scaler": scaler}

    def _predict_fold(self, fold_model, X: np.ndarray, ph: int) -> np.ndarray:
        """
        Предсказания временной модели CV fold.
        """
        if fold_model is None:
            return np.full(len(X), 0.5)
        
        scaler = fold_model.get("scaler")
        model = fold_model.get("model")
        
        X_scaled = scaler.transform(X) if scaler else X
        dtest = xgb.DMatrix(X_scaled)
        preds = model.predict(dtest)
        
        return preds

    def _maybe_flip_modes(self):
        """
        Улучшенное переключение режимов с учётом:
        1. Cross-validation метрик
        2. Статистической значимости (bootstrap CI)
        3. Out-of-fold predictions
        """
        if not self.cfg.cv_enabled:
            # fallback к старой логике
            self._maybe_flip_modes_simple()
            return
        
        def wr(arr, n):
            if len(arr) < n: return None
            window = arr[-n:]
            return 100.0 * (sum(window)/len(window))
        
        # Текущие метрики
        wr_shadow = wr(self.shadow_hits, self.cfg.min_ready)
        wr_active = wr(self.active_hits, max(30, self.cfg.min_ready // 2))
        
        # Получаем CV метрики текущей фазы
        ph = self._last_seen_phase
        cv_metrics = self.cv_metrics.get(ph, {})
        cv_passed = self.validation_passed.get(ph, False)
        
        # SHADOW → ACTIVE: требуем CV validation + bootstrap CI
        if self.mode == "SHADOW" and wr_shadow is not None:
            basic_threshold_met = wr_shadow >= self.cfg.enter_wr
            
            if basic_threshold_met and cv_passed:
                # Проверяем статистическую значимость
                cv_wr = cv_metrics.get("oof_accuracy", 0.0)
                cv_ci_lower = cv_metrics.get("ci_lower", 0.0)
                
                # Нужно: OOF accuracy > порог И нижняя граница CI тоже
                if cv_wr >= self.cfg.enter_wr and cv_ci_lower >= (self.cfg.enter_wr - self.cfg.cv_min_improvement):
                    self.mode = "ACTIVE"
                    if HAVE_RIVER:
                        self.adwin = ADWIN(delta=self.cfg.adwin_delta)
                    print(f"[{self.__class__.__name__}] SHADOW→ACTIVE ph={ph}: WR={wr_shadow:.2f}%, CV_WR={cv_wr:.2f}% (CI: [{cv_ci_lower:.2f}%, {cv_metrics.get('ci_upper', 0):.2f}%])")
        
        # ACTIVE → SHADOW: детектируем деградацию
        if self.mode == "ACTIVE" and wr_active is not None:
            basic_threshold_failed = wr_active < self.cfg.exit_wr
            
            # Также проверяем CV метрики на деградацию
            cv_wr = cv_metrics.get("oof_accuracy", 100.0)
            cv_degraded = cv_wr < self.cfg.exit_wr
            
            if basic_threshold_failed or cv_degraded:
                self.mode = "SHADOW"
                self.validation_passed[ph] = False
                print(f"[{self.__class__.__name__}] ACTIVE→SHADOW ph={ph}: WR={wr_active:.2f}%, CV_WR={cv_wr:.2f}%")

    def _maybe_flip_modes_simple(self):
        """Старая логика для backward compatibility"""
        def wr(arr, n):
            if len(arr) < n: return None
            window = arr[-n:]
            return 100.0 * (sum(window)/len(window))
        wr_shadow = wr(self.shadow_hits, self.cfg.min_ready)
        if self.mode == "SHADOW" and wr_shadow is not None and wr_shadow >= self.cfg.enter_wr:
            self.mode = "ACTIVE"
            if HAVE_RIVER:
                self.adwin = ADWIN(delta=self.cfg.adwin_delta)
        wr_active = wr(self.active_hits, max(30, self.cfg.min_ready // 2))
        if self.mode == "ACTIVE" and (wr_active is not None and wr_active < self.cfg.exit_wr):
            self.mode = "SHADOW"

    def status(self):
        def _wr(xs):
            if not xs: return None
            return sum(xs) / float(len(xs))
        def _fmt_pct(p):
            return "—" if p is None else f"{100.0*p:.2f}%"
        
        wr_a = _wr(self.active_hits)
        wr_s = _wr(self.shadow_hits)
        all_hits = (self.active_hits or []) + (self.shadow_hits or [])
        wr_all = _wr(all_hits)
        
        # CV метрики текущей фазы
        ph = self._last_seen_phase
        cv_metrics = self.cv_metrics.get(ph, {})
        cv_status = cv_metrics.get("status", "N/A")
        cv_wr = cv_metrics.get("oof_accuracy", 0.0)
        cv_ci = f"[{cv_metrics.get('ci_lower', 0):.1f}%, {cv_metrics.get('ci_upper', 0):.1f}%]" if cv_status == "ok" else "N/A"
        
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "wr_active": _fmt_pct(wr_a),
            "n_active": len(self.active_hits or []),
            "wr_shadow": _fmt_pct(wr_s),
            "n_shadow": len(self.shadow_hits or []),
            "wr_all": _fmt_pct(wr_all),
            "n": len(all_hits),
            "cv_oof_wr": _fmt_pct(cv_wr / 100.0) if cv_wr > 0 else "—",
            "cv_ci": cv_ci,
            "cv_validated": str(self.validation_passed.get(ph, False))
        }




# =============================
# NNExpert — компактная MLP с калибровкой температурой (ПО ФАЗАМ)
# =============================
class _SimpleMLP:
    def __init__(self, n_in: int, n_h: int, eta: float, l2: float):
        rng = np.random.default_rng(42)
        self.n_in, self.n_h = int(n_in), int(n_h)
        self.W1 = rng.normal(0, 0.1, size=(n_in, n_h)).astype(np.float32)
        self.b1 = np.zeros(n_h, dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, size=(n_h,)).astype(np.float32)
        self.b2 = np.float32(0.0)
        self.eta = float(eta)
        self.l2 = float(l2)

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -60.0, 60.0)
        return 1.0/(1.0 + np.exp(-z))

    @staticmethod
    def _tanh(z):
        return np.tanh(z)

    def forward_logits(self, X: np.ndarray) -> np.ndarray:
        H = self._tanh(X @ self.W1 + self.b1)
        z = H @ self.W2 + self.b2
        return z.astype(np.float32), H.astype(np.float32)

    def predict_proba(self, X: np.ndarray, T: float = 1.0) -> np.ndarray:
        z, _ = self.forward_logits(X)
        zT = z / max(1e-3, float(T))
        return self._sigmoid(zT).astype(np.float32)

    def fit_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 128):
        n = len(X)
        if n <= 0:
            return
        idx = np.arange(n)
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            sl = idx[start:start+batch_size]
            xb, yb = X[sl], y[sl]
            z, H = self.forward_logits(xb)
            p = self._sigmoid(z)
            g = (p - yb).reshape(-1, 1)  # (B,1)
            # grads
            dW2 = (H * g).mean(axis=0) + self.l2 * self.W2
            db2 = g.mean()
            dH = g * (1.0 - H*H)  # tanh'
            dW1 = (xb.T @ dH).astype(np.float32)/len(xb) + self.l2 * self.W1
            db1 = dH.mean(axis=0)
            # step
            self.W2 -= self.eta * dW2
            self.b2 -= self.eta * db2
            self.W1 -= self.eta * dW1
            self.b1 -= self.eta * db1

    def save(self, path: str):
        obj = dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, n_in=self.n_in, n_h=self.n_h, eta=self.eta, l2=self.l2)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(path: str) -> Optional["_SimpleMLP"]:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            m = _SimpleMLP(obj["n_in"], obj["n_h"], obj["eta"], obj["l2"])
            m.W1, m.b1, m.W2, m.b2 = obj["W1"], obj["b1"], obj["W2"], obj["b2"]
            return m
        except Exception:
            return None


class NNExpert(_BaseExpert):
    def __init__(self, cfg: MLConfig):
        self.cfg = cfg
        self.enabled = True  # без внешних зависимостей
        self.mode = "SHADOW"
        self.adwin = ADWIN(delta=self.cfg.adwin_delta) if HAVE_RIVER else None

        # Скейлер + сеть
        self.scaler: Optional[StandardScaler] = None
        self.net: Optional[_SimpleMLP] = None
        # --- НОВОЕ: перефазные сети и скейлеры ---
        self.net_ph: Dict[int, Optional[_SimpleMLP]] = {}
        self.scaler_ph: Dict[int, Optional[StandardScaler]] = {}
        self.n_feats: Optional[int] = None

        # ===== Глобальная память (хвост), как было =====
        self.X: List[List[float]] = []
        self.y: List[int] = []
        self.new_since_train: int = 0

        # ===== ФАЗОВАЯ ПАМЯТЬ =====
        self.P: int = int(getattr(self.cfg, "phase_count", 6))  # 6 фаз по умолчанию
        self.X_ph: Dict[int, List[List[float]]] = {p: [] for p in range(self.P)}
        self.y_ph: Dict[int, List[int]]         = {p: [] for p in range(self.P)}
        self.new_since_train_ph: Dict[int, int] = {p: 0  for p in range(self.P)}
        self._last_seen_phase: int = 0

        # ===== Калибровка температурой ПО ФАЗАМ =====
        # T_ph[φ] — температура фазы; seen_since_calib_ph[φ] — накопленные наблюдения для пересчёта
        self.T_ph: Dict[int, float] = {p: 1.0 for p in range(self.P)}
        self.seen_since_calib_ph: Dict[int, int] = {p: 0 for p in range(self.P)}
        # Старое поле T оставляем для обратной совместимости (не используется в прогнозе)
        self.T: float = 1.0
        # Cross-validation tracking (per phase)
        self.cv_oof_preds: Dict[int, deque] = {p: deque(maxlen=cfg.cv_oof_window) for p in range(self.P)}
        self.cv_oof_labels: Dict[int, deque] = {p: deque(maxlen=cfg.cv_oof_window) for p in range(self.P)}
        self.cv_metrics: Dict[int, Dict] = {p: {} for p in range(self.P)}
        self.cv_last_check: Dict[int, int] = {p: 0 for p in range(self.P)}
        
        # Validation mode tracking
        self.validation_passed: Dict[int, bool] = {p: False for p in range(self.P)}

        # Хиты
        self.shadow_hits: List[int] = []
        self.active_hits: List[int] = []

        self._load_all()

    # ---------- ВСПОМОГАТЕЛЬНЫЕ УТИЛЫ ----------
    @staticmethod
    def _clip01(p: float) -> float:
        return float(min(max(p, 1e-6), 1 - 1e-6))

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = float(np.clip(z, -60.0, 60.0))
        return 1.0/(1.0 + math.exp(-z))

    @staticmethod
    def _logit(p: float) -> float:
        p = NNExpert._clip01(p)
        return math.log(p / (1 - p))

    def _ensure_dim(self, x_raw: np.ndarray):
        d = int(x_raw.reshape(1, -1).shape[1])
        if self.n_feats is None or self.n_feats != d:
            # сбрасываем состояния под новую размерность
            self.n_feats = d
            self.net = None
            self.scaler = None
            self.X, self.y = [], []
            self.new_since_train = 0
            # фазовые буферы и счётчики тоже лучше сбросить, чтобы не мешались разные d
            self.X_ph = {p: [] for p in range(self.P)}
            self.y_ph = {p: [] for p in range(self.P)}
            self.new_since_train_ph = {p: 0 for p in range(self.P)}
            self.net_ph     = {p: None for p in range(self.P)}
            self.scaler_ph  = {p: None for p in range(self.P)}
            self.seen_since_calib_ph = {p: 0 for p in range(self.P)}
            self.T_ph = {p: 1.0 for p in range(self.P)}

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X.astype(np.float32)
        return self.scaler.transform(X.astype(np.float32))

    def _get_global_tail(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n <= 0 or not self.X:
            return np.empty((0, self.n_feats or 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        Xg = np.array(self.X[-n:], dtype=np.float32)
        yg = np.array(self.y[-n:], dtype=np.int32)
        return Xg, yg

    def _get_past_phases_tail(self, ph: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        # Берет последние n записей только из фаз 0..ph (включительно)
        # Исключает данные из будущих фаз → нет утечки
        if n <= 0:
            return np.empty((0, self.n_feats or 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        
        # Собираем все данные из фаз 0..ph
        X_past = []
        y_past = []
        for p in range(min(ph + 1, self.P)):
            if self.X_ph.get(p):
                X_past.extend(self.X_ph[p])
                y_past.extend(self.y_ph[p])
        
        if not X_past:
            return np.empty((0, self.n_feats or 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        
        # Берем последние n записей
        X_past = X_past[-n:]
        y_past = y_past[-n:]
        
        return np.array(X_past, dtype=np.float32), np.array(y_past, dtype=np.int32)

    def _get_phase_train(self, ph: int) -> Tuple[np.ndarray, np.ndarray]:
        # X_phase
        Xp = np.array(self.X_ph[ph], dtype=np.float32) if self.X_ph[ph] else np.empty((0, self.n_feats or 0), dtype=np.float32)
        yp = np.array(self.y_ph[ph], dtype=np.int32)   if self.y_ph[ph]  else np.empty((0,), dtype=np.int32)

        if len(Xp) >= int(self.cfg.phase_min_ready):
            return Xp, yp

        # иначе смешиваем X_phase ∪ X_past_phases_tail (70/30 по умолчанию)
        # ИСПРАВЛЕНИЕ: используем только прошлые фазы (0..ph), а не весь глобальный хвост
        share = float(self.cfg.phase_mix_global_share)  # 0.30
        need_g = int(round(len(Xp) * share / max(1e-9, (1.0 - share))))
        need_g = max(need_g, int(self.cfg.phase_min_ready) - len(Xp))   # не менее, чтобы достичь порога
        
        # Считаем сколько доступно в фазах 0..ph
        available_past = sum(len(self.X_ph.get(p, [])) for p in range(min(ph + 1, self.P)))
        need_g = min(need_g, available_past)  # не больше, чем доступно в прошлых фазах
        
        Xg, yg = self._get_past_phases_tail(ph, need_g)
        if len(Xg) == 0:
            return Xp, yp

        X = np.concatenate([Xp, Xg], axis=0)
        y = np.concatenate([yp, yg], axis=0)
        return X, y

    def _nll_with_T(self, z_list: np.ndarray, y: np.ndarray, T: float) -> float:
        zT = z_list / float(max(T, 1e-6))
        p = 1.0 / (1.0 + np.exp(-np.clip(zT, -60, 60)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _maybe_recalibrate_T(self, ph: int) -> None:
        """Подбор температуры T_ph[ph] по логлоссу на фазовом батче."""
        every = int(getattr(self.cfg, "nn_temp_recalib_every", 200))
        if self.seen_since_calib_ph.get(ph, 0) < every:
            return
        if self.net is None:
            return

        X_all, y_all = self._get_phase_train(ph)
        min_samples = int(getattr(self.cfg, "nn_temp_min_samples", 400))
        if len(X_all) < min_samples:
            return

        # трансформ и логиты
        Xt = self._transform(X_all)
        z, _ = self.net.forward_logits(Xt)
        z = z.astype(np.float64)
        y = np.array(y_all[:len(z)], dtype=np.int32)

        lo = float(getattr(self.cfg, "nn_temp_grid_lo", 0.5))
        hi = float(getattr(self.cfg, "nn_temp_grid_hi", 3.0))
        steps = int(getattr(self.cfg, "nn_temp_grid_steps", 25))
        Ts = np.linspace(lo, hi, num=max(2, steps))

        best_T = float(self.T_ph.get(ph, 1.0))
        best_nll = float("inf")
        for T in Ts:
            nll = self._nll_with_T(z, y, float(T))
            if nll < best_nll:
                best_nll, best_T = nll, float(T)

        self.T_ph[ph] = float(max(0.05, min(10.0, best_T)))
        self.seen_since_calib_ph[ph] = 0

    # ---------- API ЭКСПЕРТА ----------
    def proba_up(self, x_raw: np.ndarray, reg_ctx: Optional[dict] = None) -> Tuple[Optional[float], str]:
        try:
            self._ensure_dim(x_raw)
        except Exception:
            pass

        # стабильная фаза приходит в reg_ctx["phase"] (гистерезис выше по коду)
        ph = 0
        if isinstance(reg_ctx, dict):
            ph = int(reg_ctx.get("phase", 0))
        self._last_seen_phase = ph

        # сеть / скейлер по фазе (fallback на глобальные, если нет)
        net = (self.net_ph.get(ph) if hasattr(self, "net_ph") else None) or self.net
        scaler = (self.scaler_ph.get(ph) if hasattr(self, "scaler_ph") else None) or self.scaler
        if net is None:
            return (None, self.mode)

        try:
            X = x_raw.reshape(1, -1).astype(np.float32)
            Xt = scaler.transform(X) if scaler is not None else X
            T = float(self.T_ph.get(ph, 1.0))
            p = float(net.predict_proba(Xt, T=T)[0])
            p = self._clip01(p)
            return (p, self.mode)
        except Exception:
            return (None, self.mode)


    def record_result(self, x_raw: np.ndarray, y_up: int, used_in_live: bool,
                    p_pred: Optional[float] = None, reg_ctx: Optional[dict] = None) -> None:
        """
        Записывает результат предсказания и обновляет модель.
        
        Теперь включает:
        - Сохранение в глобальную и фазовую память
        - Трекинг хитов для оценки качества
        - Out-of-fold predictions для cross-validation
        - Периодическую CV проверку для валидации модели
        - Обучение модели при накоплении данных
        - Переключение режимов SHADOW/ACTIVE на основе метрик
        """
        
        # ========== БЛОК 1: ИНИЦИАЛИЗАЦИЯ И ПРОВЕРКА РАЗМЕРНОСТИ ==========
        # Убеждаемся, что размерность фичей корректна и инициализирована
        self._ensure_dim(x_raw)

        # ========== БЛОК 2: СОХРАНЕНИЕ В ГЛОБАЛЬНУЮ ПАМЯТЬ ==========
        # Глобальная память используется как fallback, когда в фазе мало данных
        self.X.append(x_raw.astype(np.float32).ravel().tolist())
        self.y.append(int(y_up))
        
        # Ограничиваем размер глобальной памяти, чтобы не раздувалась
        if len(self.X) > int(getattr(self.cfg, "max_memory", 10_000)):
            self.X = self.X[-self.cfg.max_memory:]
            self.y = self.y[-self.cfg.max_memory:]
        
        self.new_since_train += 1

        # ========== БЛОК 3: ОПРЕДЕЛЕНИЕ ФАЗЫ ==========
        # Извлекаем текущую фазу из контекста (0-5 для 6 фаз)
        ph = 0
        if isinstance(reg_ctx, dict):
            ph = int(reg_ctx.get("phase", 0))
        self._last_seen_phase = ph

        # ========== БЛОК 4: СОХРАНЕНИЕ В ФАЗОВУЮ ПАМЯТЬ ==========
        # Каждая фаза хранит свою собственную историю примеров
        # Это позволяет модели специализироваться на разных рыночных режимах
        self.X_ph[ph].append(x_raw.astype(np.float32).ravel().tolist())
        self.y_ph[ph].append(int(y_up))
        
        # Ограничиваем размер фазовой памяти
        cap = int(getattr(self.cfg, "phase_memory_cap", 10_000))
        if len(self.X_ph[ph]) > cap:
            self.X_ph[ph] = self.X_ph[ph][-cap:]
            self.y_ph[ph] = self.y_ph[ph][-cap:]
        
        self.new_since_train_ph[ph] = self.new_since_train_ph.get(ph, 0) + 1

        # ========== БЛОК 5: ТРЕКИНГ ХИТОВ И DRIFT DETECTION ==========
        # Оцениваем качество предсказания и отслеживаем дрейф концепции
        if p_pred is not None:
            try:
                # Считаем hit: правильно ли предсказали направление?
                hit = int((float(p_pred) >= 0.5) == bool(y_up))
                
                if self.mode == "ACTIVE" and used_in_live:
                    # В активном режиме отслеживаем реальные сделки
                    self.active_hits.append(hit)
                    
                    # ADWIN детектирует дрейф распределения ошибок
                    if self.adwin is not None:
                        in_drift = self.adwin.update(1 - hit)  # 1=correct, 0=error
                        if in_drift:
                            # Обнаружен дрейф - возвращаемся в shadow режим
                            self.mode = "SHADOW"
                            self.active_hits = []
                else:
                    # В shadow режиме накапливаем "что было бы, если бы входили"
                    self.shadow_hits.append(hit)
            except Exception:
                pass

        # ========== БЛОК 6: НОВОЕ - СОХРАНЕНИЕ OOF PREDICTIONS ДЛЯ CV ==========
        # Out-of-fold predictions нужны для расчета метрик cross-validation
        # Эти предсказания были сделаны на данных, которые модель НЕ видела при обучении
        if self.cfg.cv_enabled and p_pred is not None:
            self.cv_oof_preds[ph].append(float(p_pred))
            self.cv_oof_labels[ph].append(int(y_up))

        # ========== БЛОК 7: ФАЗОВАЯ КАЛИБРОВКА ==========
        # Калибратор корректирует вероятности для каждой фазы отдельно
        # Это важно, потому что модель может быть по-разному откалибрована в разных режимах
        try:
            p_raw = self._predict_raw(x_raw)
            if p_raw is not None:
                # Инициализируем калибратор для этой фазы, если его нет
                if self.cal_ph[ph] is None:
                    self.cal_ph[ph] = make_calibrator(self.cfg.xgb_calibration_method)
                
                # Показываем калибратору истинную пару (предсказание, результат)
                self.cal_ph[ph].observe(float(p_raw), int(y_up))
                
                # Периодически пересчитываем калибровку
                if self.cal_ph[ph].maybe_fit(min_samples=200, every=100):
                    cal_path = self._cal_path(self.cfg.xgb_cal_path, ph)
                    self.cal_ph[ph].save(cal_path)
        except Exception:
            pass

        # ========== БЛОК 8: НОВОЕ - ПЕРИОДИЧЕСКАЯ CV ПРОВЕРКА ==========
        # Каждые N примеров запускаем полную cross-validation для оценки реального качества
        # Это защищает от переобучения и дает честную оценку обобщающей способности
        self.cv_last_check[ph] += 1
        
        if self.cfg.cv_enabled and self.cv_last_check[ph] >= self.cfg.cv_check_every:
            # Сбрасываем счетчик
            self.cv_last_check[ph] = 0
            
            # Запускаем полную walk-forward cross-validation с purging
            cv_results = self._run_cv_validation(ph)
            
            # Сохраняем результаты для использования в _maybe_flip_modes
            self.cv_metrics[ph] = cv_results
            
            # Если CV прошла успешно, помечаем фазу как валидированную
            if cv_results.get("status") == "ok":
                self.validation_passed[ph] = True
            
            # Логируем результаты для мониторинга
            if cv_results.get("status") == "ok":
                print(f"[{self.__class__.__name__}] CV ph={ph}: "
                    f"OOF_ACC={cv_results['oof_accuracy']:.2f}% "
                    f"CI=[{cv_results['ci_lower']:.2f}%, {cv_results['ci_upper']:.2f}%] "
                    f"folds={cv_results['n_folds']}")

        # ========== БЛОК 9: ОБУЧЕНИЕ МОДЕЛИ ПО ФАЗЕ ==========
        # Когда накопилось достаточно новых примеров в фазе, запускаем переобучение
        self._maybe_train_phase(ph)

        # ========== БЛОК 10: ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ ==========
        # Проверяем метрики (включая CV) и решаем, переключать ли SHADOW ↔ ACTIVE
        self._maybe_flip_modes()
        
        # ========== БЛОК 11: СОХРАНЕНИЕ СОСТОЯНИЯ ==========
        # Периодически сохраняем все на диск для восстановления после перезапуска
        self._save_all()

    def _maybe_flip_modes(self):
        """
        Улучшенное переключение режимов с учётом:
        1. Cross-validation метрик
        2. Статистической значимости (bootstrap CI)
        3. Out-of-fold predictions
        """
        if not self.cfg.cv_enabled:
            # fallback к старой логике
            self._maybe_flip_modes_simple()
            return
        
        def wr(arr, n):
            if len(arr) < n: return None
            window = arr[-n:]
            return 100.0 * (sum(window)/len(window))
        
        # Текущие метрики
        wr_shadow = wr(self.shadow_hits, self.cfg.min_ready)
        wr_active = wr(self.active_hits, max(30, self.cfg.min_ready // 2))
        
        # Получаем CV метрики текущей фазы
        ph = self._last_seen_phase
        cv_metrics = self.cv_metrics.get(ph, {})
        cv_passed = self.validation_passed.get(ph, False)
        
        # SHADOW → ACTIVE: требуем CV validation + bootstrap CI
        if self.mode == "SHADOW" and wr_shadow is not None:
            basic_threshold_met = wr_shadow >= self.cfg.enter_wr
            
            if basic_threshold_met and cv_passed:
                # Проверяем статистическую значимость
                cv_wr = cv_metrics.get("oof_accuracy", 0.0)
                cv_ci_lower = cv_metrics.get("ci_lower", 0.0)
                
                # Нужно: OOF accuracy > порог И нижняя граница CI тоже
                if cv_wr >= self.cfg.enter_wr and cv_ci_lower >= (self.cfg.enter_wr - self.cfg.cv_min_improvement):
                    self.mode = "ACTIVE"
                    if HAVE_RIVER:
                        self.adwin = ADWIN(delta=self.cfg.adwin_delta)
                    print(f"[{self.__class__.__name__}] SHADOW→ACTIVE ph={ph}: WR={wr_shadow:.2f}%, CV_WR={cv_wr:.2f}% (CI: [{cv_ci_lower:.2f}%, {cv_metrics.get('ci_upper', 0):.2f}%])")
        
        # ACTIVE → SHADOW: детектируем деградацию
        if self.mode == "ACTIVE" and wr_active is not None:
            basic_threshold_failed = wr_active < self.cfg.exit_wr
            
            # Также проверяем CV метрики на деградацию
            cv_wr = cv_metrics.get("oof_accuracy", 100.0)
            cv_degraded = cv_wr < self.cfg.exit_wr
            
            if basic_threshold_failed or cv_degraded:
                self.mode = "SHADOW"
                self.validation_passed[ph] = False
                print(f"[{self.__class__.__name__}] ACTIVE→SHADOW ph={ph}: WR={wr_active:.2f}%, CV_WR={cv_wr:.2f}%")

    def _maybe_flip_modes_simple(self):
        """Старая логика для backward compatibility"""
        def wr(arr, n):
            if len(arr) < n: return None
            window = arr[-n:]
            return 100.0 * (sum(window)/len(window))
        wr_shadow = wr(self.shadow_hits, self.cfg.min_ready)
        if self.mode == "SHADOW" and wr_shadow is not None and wr_shadow >= self.cfg.enter_wr:
            self.mode = "ACTIVE"
            if HAVE_RIVER:
                self.adwin = ADWIN(delta=self.cfg.adwin_delta)
        wr_active = wr(self.active_hits, max(30, self.cfg.min_ready // 2))
        if self.mode == "ACTIVE" and (wr_active is not None and wr_active < self.cfg.exit_wr):
            self.mode = "SHADOW"

    # --- обучение NN по ФАЗЕ ---
    # --- обучение NN по ФАЗЕ ---
    def _maybe_train_phase(self, ph: int) -> None:
        # перезапуск обучения только когда в фазе накопилось достаточно новых примеров
        if self.new_since_train_ph.get(ph, 0) < int(getattr(self.cfg, "nn_retrain_every", 100)):
            return

        # собираем батч конкретной фазы
        X_all, y_all = self._get_phase_train(ph)
        if len(X_all) < int(self.cfg.phase_min_ready):
            return

        # ограничим окно обучения по свежести
        train_window = int(getattr(self.cfg, "train_window", 5000))
        if len(X_all) > train_window:
            X_all = X_all[-train_window:]
            y_all = y_all[-train_window:]

        # гарантируем сеть и скейлер для КОНКРЕТНОЙ фазы
        net = self.net_ph.get(ph)
        if net is None and self.n_feats is not None:
            net = _SimpleMLP(
                n_in=self.n_feats,
                n_h=int(getattr(self.cfg, "nn_hidden", 32)),
                eta=float(getattr(self.cfg, "nn_eta", 0.01)),
                l2=float(getattr(self.cfg, "nn_l2", 0.0)),
            )

        # скейлер по батчу фазы
        scaler = None
        if HAVE_SKLEARN:
            try:
                scaler = StandardScaler().fit(X_all)
            except Exception:
                scaler = None

        # трансформация и обучение
        Xt = scaler.transform(X_all) if (scaler is not None) else X_all
        y_float = y_all.astype(np.float32)

        try:
            epochs = int(getattr(self.cfg, "nn_epochs", 1))
            for _ in range(max(1, epochs)):
                net.fit_epoch(Xt, y_float, batch_size=128)

            # сохранить модель/скейлер фазы и обновить «последние» глобальные ссылки
            self.net_ph[ph] = net
            self.scaler_ph[ph] = scaler
            self.net = net
            self.scaler = scaler

            # сброс счётчика «новых с последнего тренинга» для этой фазы
            self.new_since_train_ph[ph] = 0
        except Exception as e:
            print(f"[nn  ] train error (ph={ph}): {e}")
    def _run_cv_validation(self, ph: int) -> Dict:
        """
        Walk-forward purged cross-validation для фазы ph.
        Возвращает метрики: accuracy, CI bounds, fold scores.
        """
        X_all, y_all = self._get_phase_train(ph)
        
        if len(X_all) < self.cfg.cv_min_train_size:
            return {"status": "insufficient_data", "oof_accuracy": 0.0}
        
        n_samples = len(X_all)
        n_splits = min(self.cfg.cv_n_splits, n_samples // self.cfg.cv_min_train_size)
        
        if n_splits < 2:
            return {"status": "insufficient_splits", "oof_accuracy": 0.0}
        
        # Walk-forward splits с purge и embargo
        embargo_size = max(1, int(n_samples * self.cfg.cv_embargo_pct))
        purge_size = max(1, int(n_samples * self.cfg.cv_purge_pct))
        
        fold_size = n_samples // n_splits
        oof_preds = np.zeros(n_samples)
        oof_mask = np.zeros(n_samples, dtype=bool)
        fold_scores = []
        
        for fold_idx in range(n_splits):
            # Test fold
            test_start = fold_idx * fold_size
            test_end = min(test_start + fold_size, n_samples)
            
            # Train: всё до (test_start - purge_size)
            train_end = max(0, test_start - purge_size)
            
            if train_end < self.cfg.cv_min_train_size:
                continue
            
            X_train = X_all[:train_end]
            y_train = y_all[:train_end]
            X_test = X_all[test_start:test_end]
            y_test = y_all[test_start:test_end]
            
            # Обучаем временную модель на train fold
            temp_model = self._train_fold_model(X_train, y_train, ph)
            
            # Предсказания на test fold
            preds = self._predict_fold(temp_model, X_test, ph)
            
            # Сохраняем OOF predictions
            oof_preds[test_start:test_end] = preds
            oof_mask[test_start:test_end] = True
            
            # Метрики фолда
            fold_acc = np.mean((preds >= 0.5) == y_test)
            fold_scores.append(fold_acc)
        
        # Итоговые OOF метрики
        oof_valid = oof_mask.sum()
        if oof_valid < self.cfg.cv_min_train_size:
            return {"status": "insufficient_oof", "oof_accuracy": 0.0}
        
        oof_accuracy = 100.0 * np.mean((oof_preds[oof_mask] >= 0.5) == y_all[oof_mask])
        
        # Bootstrap confidence intervals
        ci_lower, ci_upper = self._bootstrap_ci(
            oof_preds[oof_mask], 
            y_all[oof_mask],
            n_bootstrap=self.cfg.cv_bootstrap_n,
            confidence=self.cfg.cv_confidence
        )
        
        return {
            "status": "ok",
            "oof_accuracy": oof_accuracy,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "fold_scores": fold_scores,
            "n_folds": len(fold_scores),
            "oof_samples": int(oof_valid)
        }

    def _bootstrap_ci(self, preds: np.ndarray, labels: np.ndarray, 
                    n_bootstrap: int, confidence: float) -> tuple:
        """
        Bootstrap confidence intervals для accuracy.
        """
        accuracies = []
        n = len(preds)
        
        for _ in range(n_bootstrap):
            # Resample с возвратом
            idx = np.random.choice(n, size=n, replace=True)
            boot_preds = preds[idx]
            boot_labels = labels[idx]
            boot_acc = 100.0 * np.mean((boot_preds >= 0.5) == boot_labels)
            accuracies.append(boot_acc)
        
        accuracies = np.array(accuracies)
        alpha = 1.0 - confidence
        ci_lower = np.percentile(accuracies, 100 * alpha / 2)
        ci_upper = np.percentile(accuracies, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper

    def _train_fold_model(self, X: np.ndarray, y: np.ndarray, ph: int):
        """
        Обучает временную модель для CV fold.
        Реализация зависит от типа эксперта (XGB/RF/ARF/NN).
        """
        # Пример для XGB
        if not HAVE_XGB:
            return None
        
        scaler = StandardScaler().fit(X) if HAVE_SKLEARN else None
        X_scaled = scaler.transform(X) if scaler else X
        
        dtrain = xgb.DMatrix(X_scaled, label=y)
        model = xgb.train(
            params={
                "objective": "binary:logistic",
                "max_depth": self.cfg.xgb_max_depth,
                "eta": self.cfg.xgb_eta,
                "subsample": self.cfg.xgb_subsample,
                "colsample_bytree": self.cfg.xgb_colsample_bytree,
                "min_child_weight": self.cfg.xgb_min_child_weight,
                "eval_metric": "logloss",
            },
            dtrain=dtrain,
            num_boost_round=self.cfg.xgb_rounds_warm,
            verbose_eval=False
        )
        
        return {"model": model, "scaler": scaler}

    def _predict_fold(self, fold_model, X: np.ndarray, ph: int) -> np.ndarray:
        """
        Предсказания временной модели CV fold.
        """
        if fold_model is None:
            return np.full(len(X), 0.5)
        
        scaler = fold_model.get("scaler")
        model = fold_model.get("model")
        
        X_scaled = scaler.transform(X) if scaler else X
        dtest = xgb.DMatrix(X_scaled)
        preds = model.predict(dtest)
        
        return preds


    def maybe_train(self, ph: Optional[int] = None, reg_ctx: Optional[dict] = None) -> None:
        if ph is None:
            if isinstance(reg_ctx, dict) and "phase" in reg_ctx:
                ph = int(reg_ctx["phase"])
            else:
                ph = int(getattr(self, "_last_seen_phase", 0))
        self._maybe_train_phase(int(ph))

    # ---------- сохранение/загрузка ----------
    def _load_all(self) -> None:
        # state (mode, hits, T, n_feats, фазовые буферы, температуры по фазам)
        try:
            if os.path.exists(self.cfg.nn_state_path):
                with open(self.cfg.nn_state_path, "r") as f:
                    st = json.load(f)

                self.mode = st.get("mode", "SHADOW")
                self.shadow_hits = st.get("shadow_hits", [])[-1000:]
                self.active_hits = st.get("active_hits", [])[-1000:]
                self.n_feats = st.get("n_feats", None)

                # глобальная память
                self.X = st.get("X", [])
                self.y = st.get("y", [])

                # фазовые буферы
                X_ph = st.get("X_ph"); y_ph = st.get("y_ph")
                if isinstance(X_ph, dict) and isinstance(y_ph, dict):
                    self.X_ph = {int(k): v for k, v in X_ph.items()}
                    self.y_ph = {int(k): v for k, v in y_ph.items()}
                else:
                    self.X_ph = {p: [] for p in range(self.P)}
                    self.y_ph = {p: [] for p in range(self.P)}

                # счётчики тренировки
                self.new_since_train_ph = {p: 0 for p in range(self.P)}
                if isinstance(st.get("new_since_train_ph"), dict):
                    for k, v in st["new_since_train_ph"].items():
                        try:
                            self.new_since_train_ph[int(k)] = int(v)
                        except Exception:
                            pass

                # температуры по фазам + счётчики для перекалибровки
                T_ph = st.get("T_ph")
                self.T_ph = {p: 1.0 for p in range(self.P)}
                if isinstance(T_ph, dict):
                    for k, v in T_ph.items():
                        try:
                            self.T_ph[int(k)] = float(v)
                        except Exception:
                            pass
                ssc = st.get("seen_since_calib_ph")
                self.seen_since_calib_ph = {p: 0 for p in range(self.P)}
                if isinstance(ssc, dict):
                    for k, v in ssc.items():
                        try:
                            self.seen_since_calib_ph[int(k)] = int(v)
                        except Exception:
                            pass

                self._last_seen_phase = int(st.get("_last_seen_phase", 0))
                # старое поле T (для обратной совместимости)
                try:
                    self.T = float(st.get("T", 1.0))
                except Exception:
                    self.T = 1.0

                # обрезки по капам
                if isinstance(getattr(self.cfg, "max_memory", None), int) and self.cfg.max_memory > 0:
                    self.X = self.X[-self.cfg.max_memory:]
                    self.y = self.y[-self.cfg.max_memory:]
                cap = int(getattr(self.cfg, "phase_memory_cap", 10_000))
                for p in range(self.P):
                    self.X_ph[p] = self.X_ph.get(p, [])[-cap:]
                    self.y_ph[p] = self.y_ph.get(p, [])[-cap:]
        except Exception as e:
            print(f"[nn  ] _load_all state error: {e}")

        # scaler
        try:
            if os.path.exists(self.cfg.nn_scaler_path):
                with open(self.cfg.nn_scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
        except Exception as e:
            print(f"[nn  ] _load_all scaler error: {e}")
            self.scaler = None

        # model
        try:
            self.net = _SimpleMLP.load(self.cfg.nn_model_path)
        except Exception as e:
            print(f"[nn  ] _load_all model error: {e}")
            self.net = None

        # --- НОВОЕ: загрузка перефазных сетей и скейлеров ---
        try:
            root_m, ext_m = os.path.splitext(self.cfg.nn_model_path)
            root_s, ext_s = os.path.splitext(self.cfg.nn_scaler_path)
            for p in range(self.P):
                mp = f"{root_m}_ph{p}{ext_m}"
                sp = f"{root_s}_ph{p}{ext_s}"
                # сеть
                try:
                    self.net_ph[p] = _SimpleMLP.load(mp)
                except Exception:
                    self.net_ph[p] = None
                # скейлер
                try:
                    with open(sp, "rb") as f:
                        self.scaler_ph[p] = pickle.load(f)
                except Exception:
                    self.scaler_ph[p] = None
        except Exception as e:
            print(f"[nn  ] _load_all per-phase error: {e}")


    def _save_all(self) -> None:
        try:
            # обрезки
            X_tail, y_tail = self.X, self.y
            if isinstance(getattr(self.cfg, "max_memory", None), int) and self.cfg.max_memory > 0:
                X_tail = self.X[-self.cfg.max_memory:]
                y_tail = self.y[-self.cfg.max_memory:]
            cap = int(getattr(self.cfg, "phase_memory_cap", 10_000))
            X_ph_tail = {p: self.X_ph.get(p, [])[-cap:] for p in range(self.P)}
            y_ph_tail = {p: self.y_ph.get(p, [])[-cap:] for p in range(self.P)}

            st = {
                "mode": self.mode,
                "shadow_hits": self.shadow_hits[-1000:],
                "active_hits": self.active_hits[-1000:],
                "n_feats": self.n_feats,

                "X": X_tail, "y": y_tail,
                "X_ph": X_ph_tail, "y_ph": y_ph_tail,
                "new_since_train_ph": {int(p): int(self.new_since_train_ph.get(p, 0)) for p in range(self.P)},
                "_last_seen_phase": int(self._last_seen_phase),

                # температуры и счётчики
                "T_ph": {int(p): float(self.T_ph.get(p, 1.0)) for p in range(self.P)},
                "seen_since_calib_ph": {int(p): int(self.seen_since_calib_ph.get(p, 0)) for p in range(self.P)},

                # для обратной совместимости
                "T": float(self.T),
                "P": int(self.P),
            }
            with open(self.cfg.nn_state_path, "w") as f:
                json.dump(st, f)
        except Exception as e:
            print(f"[nn  ] _save_all state error: {e}")

        # scaler
        try:
            if self.scaler is not None:
                with open(self.cfg.nn_scaler_path, "wb") as f:
                    pickle.dump(self.scaler, f)
        except Exception as e:
            print(f"[nn  ] _save_all scaler error: {e}")

        # model
        try:
            if self.net is not None:
                self.net.save(self.cfg.nn_model_path)
        except Exception as e:
            print(f"[nn  ] _save_all model error: {e}")

        # --- НОВОЕ: сохранение перефазных сетей и скейлеров ---
        try:
            root_m, ext_m = os.path.splitext(self.cfg.nn_model_path)
            root_s, ext_s = os.path.splitext(self.cfg.nn_scaler_path)
            for p in range(self.P):
                net = self.net_ph.get(p)
                if net is not None:
                    net.save(f"{root_m}_ph{p}{ext_m}")
                sc = self.scaler_ph.get(p)
                if sc is not None:
                    with open(f"{root_s}_ph{p}{ext_s}", "wb") as f:
                        pickle.dump(sc, f)
        except Exception as e:
            print(f"[nn  ] _save_all per-phase error: {e}")


    # ---------- статус ----------
    def status(self):
        def _wr(xs):
            if not xs: return None
            return sum(xs) / float(len(xs))
        def _fmt_pct(p):
            return "—" if p is None else f"{100.0*p:.2f}%"
        
        wr_a = _wr(self.active_hits)
        wr_s = _wr(self.shadow_hits)
        all_hits = (self.active_hits or []) + (self.shadow_hits or [])
        wr_all = _wr(all_hits)
        
        # CV метрики текущей фазы
        ph = self._last_seen_phase
        cv_metrics = self.cv_metrics.get(ph, {})
        cv_status = cv_metrics.get("status", "N/A")
        cv_wr = cv_metrics.get("oof_accuracy", 0.0)
        cv_ci = f"[{cv_metrics.get('ci_lower', 0):.1f}%, {cv_metrics.get('ci_upper', 0):.1f}%]" if cv_status == "ok" else "N/A"
        
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "wr_active": _fmt_pct(wr_a),
            "n_active": len(self.active_hits or []),
            "wr_shadow": _fmt_pct(wr_s),
            "n_shadow": len(self.shadow_hits or []),
            "wr_all": _fmt_pct(wr_all),
            "n": len(all_hits),
            "cv_oof_wr": _fmt_pct(cv_wr / 100.0) if cv_wr > 0 else "—",
            "cv_ci": cv_ci,
            "cv_validated": str(self.validation_passed.get(ph, False))
        }




# =============================
# META-оценщик (расширенный, 5 логитов)
# =============================

class MetaStacking:
    """
    Мета с контекстным гейтингом.
    Вход: p_xgb, p_rf, p_arf, p_nn, p_base, reg_ctx (ψ).
    Режимы:
      - gating_mode="soft": g = softmax(Wg @ ψ_ext). z = Σ g_k*logit(p_k) * alpha_mix + (w_meta · [lz_base, disagree, entropy, 1]).
      - gating_mode="exp4": по фазе φ обновляются веса экспертов Hedge/EXP4; z = Σ w_k(φ)*logit(p_k) * alpha_mix + (w_meta · ...).
    Обновление ТОЛЬКО после settle.
    """
    def __init__(self, cfg: MLConfig):
        self.cfg = cfg
        self.enabled = True
        self.mode = "SHADOW"
        self.adwin = ADWIN(delta=self.cfg.adwin_delta) if HAVE_RIVER else None

        # базовая линейка меты: [lz_b, disagree, ent, 1]
        self.P = int(cfg.meta_exp4_phases)
        self.w_meta_ph = np.zeros((self.P, 4), dtype=float)  # [lz_b, disagree, ent, 1]
        self.eta = cfg.meta_eta
        self.l2  = cfg.meta_l2
        self.w_clip = cfg.meta_w_clip
        self.g_clip = cfg.meta_g_clip

        # soft гейтер
        self.gating_mode = cfg.meta_gating_mode
        self.alpha_mix   = float(cfg.meta_alpha_mix)
        self.Wg = None            # (K x d_ctx), реализуем как (K, D) где D=len(ψ_ext)
        self.g_eta = cfg.meta_gate_eta
        self.g_l2  = cfg.meta_gate_l2
        self.gate_clip = cfg.meta_gate_clip

        # EXP4 веса по фазам: P x K (нормированы по K)
        self.P = int(cfg.meta_exp4_phases)
        self.exp4_eta = float(cfg.meta_exp4_eta)
        self.exp4_w = None  # np.ndarray (P,K)

        # трекинг для flip режимов
        self.shadow_hits: List[int] = []
        self.active_hits: List[int] = []

        self._load()

    def bind_experts(self, *experts):
        """
        Сохраняет ссылки на экспертов (для статуса/логов и будущих расширений).
        Возвращает self для chain-style.
        """
        self._experts = list(experts)
        return self

    # ---------- служебки ----------
    @staticmethod
    def _lz(p: Optional[float]) -> float:
        if p is None: return 0.0
        return to_logit(float(np.clip(p, 1e-6, 1-1e-6)))

    @staticmethod
    def _entropy(p_list: List[Optional[float]]) -> float:
        out, n = 0.0, 0
        for p in p_list:
            if p is None: 
                continue
            pp = float(np.clip(p, 1e-6, 1-1e-6))
            out += -(pp*math.log(pp) + (1-pp)*math.log(1-pp))
            n += 1
        return out / max(1, n)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(np.clip(x, -60, 60))
        s = ex.sum()
        if s <= 0:  # fallback
            return np.ones_like(ex)/len(ex)
        return ex / s

    from state_safety import atomic_save_json
    def _save(self):
        try:
            atomic_save_json(self.cfg.meta_state_path,{
                    "mode": self.mode,
                    "w_meta_ph": self.w_meta_ph.tolist(),
                    "shadow_hits": self.shadow_hits[-1000:],
                    "active_hits": self.active_hits[-1000:],
                    "gating_mode": self.gating_mode,
                    "alpha_mix": self.alpha_mix,
                    "Wg": (self.Wg.tolist() if self.Wg is not None else []),
                    "P": self.P,
                    "exp4_w": (self.exp4_w.tolist() if self.exp4_w is not None else []),
                })
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(self.cfg.meta_state_path):
                with open(self.cfg.meta_state_path, "r") as f:
                    st = json.load(f)
                self.mode = st.get("mode", "SHADOW")
                # обратная совместимость
                wm_ph = st.get("w_meta_ph", None)
                if wm_ph:
                    self.w_meta_ph = np.array(wm_ph, dtype=float)
                elif "w_meta" in st:
                    w = np.array(st["w_meta"], dtype=float)
                    self.w_meta_ph = np.vstack([w for _ in range(self.P)])  # тиражируем
                elif "w" in st and isinstance(st["w"], list):
                    w_old = np.array(st["w"], dtype=float)
                    w = np.zeros(4, dtype=float)
                    if len(w_old) >= 8:
                        w[0], w[1], w[2], w[3] = w_old[4], w_old[5], w_old[6], w_old[7]
                    self.w_meta_ph = np.vstack([w for _ in range(self.P)])
                self.shadow_hits = st.get("shadow_hits", [])
                self.active_hits = st.get("active_hits", [])
                self.gating_mode = st.get("gating_mode", self.gating_mode)
                self.alpha_mix   = float(st.get("alpha_mix", self.alpha_mix))
                Wg = st.get("Wg", [])
                if Wg:
                    self.Wg = np.array(Wg, dtype=float)
                P = int(st.get("P", self.P))
                self.P = P
                exp4_w = st.get("exp4_w", [])
                if exp4_w:
                    self.exp4_w = np.array(exp4_w, dtype=float)
        except Exception:
            pass

    # ---------- гейтеры ----------
    def _ensure_Wg(self, d_ctx: int, K: int):
        if self.Wg is None or self.Wg.shape != (K, d_ctx):
            self.Wg = np.zeros((K, d_ctx), dtype=float)

    def _ensure_exp4(self, K: int):
        if self.exp4_w is None or self.exp4_w.shape != (self.P, K):
            self.exp4_w = np.ones((self.P, K), dtype=float) / float(K)

    def _gating_soft(self, psi_ext: np.ndarray, avail_mask: np.ndarray) -> np.ndarray:
        """
        Возвращает g на доступных экспертов (маска avail_mask), перенормируя.
        """
        K = len(avail_mask)
        self._ensure_Wg(len(psi_ext), K)
        scores = (self.Wg @ psi_ext)  # (K,)
        g = self._softmax(scores)
        # занулим недоступных, перенормируем
        g = g * avail_mask
        s = g.sum()
        if s <= 0:
            g = avail_mask / max(1, avail_mask.sum())
        else:
            g = g / s
        return g

    def _gating_exp4(self, phase_id: int, avail_mask: np.ndarray) -> np.ndarray:
        K = len(avail_mask)
        self._ensure_exp4(K)
        w = self.exp4_w[phase_id].copy()
        w = w * avail_mask
        s = w.sum()
        if s <= 0:
            w = avail_mask / max(1, avail_mask.sum())
        else:
            w = w / s
        return w

    # ---------- публичные API ----------
    def predict(
        self,
        p_xgb: Optional[float],
        p_rf: Optional[float],
        p_arf: Optional[float],
        p_nn: Optional[float],
        p_base: Optional[float],
        reg_ctx: Optional[Dict[str, float]] = None
    ) -> float:
        # логиты экспертов
        lzs = np.array([
            self._lz(p_xgb), self._lz(p_rf), self._lz(p_arf), self._lz(p_nn)
        ], dtype=float)
        avail = np.array([p_xgb is not None, p_rf is not None, p_arf is not None, p_nn is not None], dtype=float)

        # смесь экспертов по контексту
        mix_logit = 0.0
        if reg_ctx is not None and avail.any():
            from meta_ctx import pack_ctx, phase_from_ctx
            psi_ext, _ = pack_ctx(reg_ctx)
            if self.gating_mode == "soft":
                g = self._gating_soft(psi_ext, avail)
            else:  # "exp4"
                ph = phase_from_ctx(reg_ctx)
                g = self._gating_exp4(ph, avail)
            mix_logit = float(np.dot(g, lzs))
        else:
            # равные веса по доступным
            s = avail.sum()
            if s > 0:
                mix_logit = float(np.dot(lzs, avail / s))

        # базовые агрегаты меты
        lz_b = self._lz(p_base)
        plist = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
        disagree = float(np.mean([abs(p - 0.5) for p in plist])) if plist else 0.0
        ent = self._entropy([p_xgb, p_rf, p_arf, p_nn])

        phi_meta = np.array([lz_b, disagree, ent, 1.0], dtype=float)
        ph = int(reg_ctx.get("phase")) if isinstance(reg_ctx, dict) and "phase" in reg_ctx else (
            phase_from_ctx(reg_ctx) if reg_ctx is not None else 0)
        w_phi = self.w_meta_ph[ph]
        z = self.alpha_mix * mix_logit + float(np.dot(w_phi, phi_meta))
        return sigmoid(z)


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
        reg_ctx: Optional[Dict[str, float]] = None
    ):
        # --- шаг по мета-весам (logistic loss)
        lz_b = self._lz(p_base)
        plist = [p for p in [p_xgb, p_rf, p_arf, p_nn] if p is not None]
        disagree = float(np.mean([abs(p - 0.5) for p in plist])) if plist else 0.0
        ent = self._entropy([p_xgb, p_rf, p_arf, p_nn])
        phi_meta = np.array([lz_b, disagree, ent, 1.0], dtype=float)

        # === NEW: считаем смесь логитов экспертов как в predict()
        lzs = np.array([
            self._lz(p_xgb), self._lz(p_rf), self._lz(p_arf), self._lz(p_nn)
        ], dtype=float)
        avail = np.array([p_xgb is not None, p_rf is not None, p_arf is not None, p_nn is not None], dtype=float)

        mix_logit = 0.0
        if avail.any():
            if reg_ctx is not None:
                if getattr(self, "gating_mode", "soft") == "soft":
                    from meta_ctx import pack_ctx
                    psi_ext, _ = pack_ctx(reg_ctx)
                    self._ensure_Wg(len(psi_ext), len(avail))
                    g = self._gating_soft(psi_ext, avail)
                else:
                    from meta_ctx import phase_from_ctx
                    ph = phase_from_ctx(reg_ctx)
                    g = self._gating_exp4(ph, avail)
            else:
                g = (avail / avail.sum())
            mix_logit = float(np.dot(g, lzs))

        # теперь используем ту же формулу, что и в predict()
        ph = int(reg_ctx.get("phase")) if isinstance(reg_ctx, dict) and "phase" in reg_ctx else (
            phase_from_ctx(reg_ctx) if reg_ctx is not None else 0)
        w_phi = self.w_meta_ph[ph]
        p_hat = sigmoid(self.alpha_mix * mix_logit + float(np.dot(w_phi, phi_meta)))
        g = float(np.clip(p_hat - float(y_up), -self.g_clip, self.g_clip))
        w_phi = w_phi - self.eta * (g * phi_meta + self.l2 * w_phi)
        w_phi = np.clip(w_phi, -self.w_clip, self.w_clip)
        self.w_meta_ph[ph] = w_phi


        # --- soft: шаг по Wg
        if self.gating_mode == "soft" and reg_ctx is not None:
            from meta_ctx import pack_ctx
            psi_ext, _ = pack_ctx(reg_ctx)
            lzs = np.array([
                self._lz(p_xgb), self._lz(p_rf), self._lz(p_arf), self._lz(p_nn)
            ], dtype=float)
            avail = np.array([p_xgb is not None, p_rf is not None, p_arf is not None, p_nn is not None], dtype=float)
            if avail.any():
                self._ensure_Wg(len(psi_ext), len(avail))
                # текущая смесь
                scores = (self.Wg @ psi_ext)
                g_soft = self._softmax(scores)
                g_soft = g_soft * avail
                s = g_soft.sum()
                g_soft = (g_soft / s) if s > 0 else (avail / max(1, avail.sum()))
                # целевой логит = logit(y_up) ~ +inf/-inf, используем градиент через сигмоиду:
                mix_logit = float(np.dot(g_soft, lzs))
                p_mix = sigmoid(self.alpha_mix * mix_logit)
                gm = float(np.clip(p_mix - float(y_up), -self.gate_clip, self.gate_clip))
                # dL/dscores_k = alpha_mix * gm * (lzs_k - Σ g*lzs) * g_k * (1 - g_k)
                lzs_mean = float(np.dot(g_soft, lzs))
                delta = (lzs - lzs_mean) * g_soft * (1.0 - g_soft) * (self.alpha_mix * gm)
                # Wg -= η * (delta_k * psi_ext + l2 * Wg_k)
                for k in range(self.Wg.shape[0]):
                    self.Wg[k, :] -= self.g_eta * (delta[k] * psi_ext + self.g_l2 * self.Wg[k, :])
                self.Wg = np.clip(self.Wg, -20.0, 20.0)

        # --- EXP4: обновление весов по фазе (Hedge)
        if self.gating_mode == "exp4" and reg_ctx is not None:
            from meta_ctx import phase_from_ctx
            ph = phase_from_ctx(reg_ctx)
            lzs = np.array([
                self._lz(p_xgb), self._lz(p_rf), self._lz(p_arf), self._lz(p_nn)
            ], dtype=float)
            avail = np.array([p_xgb is not None, p_rf is not None, p_arf is not None, p_nn is not None], dtype=float)
            K = len(avail)
            self._ensure_exp4(K)
            # log-loss каждого эксперта на исходе
            def _ll(p):
                if p is None: return 0.0
                p = float(np.clip(p, 1e-6, 1-1e-6))
                return -(y_up*math.log(p) + (1-y_up)*math.log(1-p))
            losses = np.array([_ll(p_xgb), _ll(p_rf), _ll(p_arf), _ll(p_nn)], dtype=float)
            # обновляем только доступных
            for k in range(K):
                if avail[k] > 0:
                    self.exp4_w[ph, k] *= math.exp(-self.exp4_eta * losses[k])
            # нормировка
            s = self.exp4_w[ph, :].sum()
            if s > 0: self.exp4_w[ph, :] /= s

        # --- гейтинг / дрейф: апдейт hit-rate
        p_for_gate = p_final_used if (p_final_used is not None) else p_hat
        hit = int((p_for_gate >= 0.5) == bool(y_up))
        if self.mode == "ACTIVE" and used_in_live:
            self.active_hits.append(hit)
            if self.adwin is not None:
                try:
                    in_drift = self.adwin.update(1 - hit)
                    if in_drift:
                        self.mode = "SHADOW"
                        self.active_hits = []
                except Exception:
                    pass
        else:
            self.shadow_hits.append(hit)

        self._maybe_flip_modes()
        self._save()

    def _maybe_flip_modes(self):
        def wr(arr, n):
            if len(arr) < n: return None
            window = arr[-n:]
            return 100.0 * (sum(window)/len(window))
        wr_shadow = wr(self.shadow_hits, self.cfg.min_ready)
        if self.mode == "SHADOW" and wr_shadow is not None and wr_shadow >= self.cfg.enter_wr:
            self.mode = "ACTIVE"
            if HAVE_RIVER:
                self.adwin = ADWIN(delta=self.cfg.adwin_delta)
        wr_active = wr(self.active_hits, max(30, self.cfg.min_ready // 2))
        if self.mode == "ACTIVE" and (wr_active is not None and wr_active < self.cfg.exit_wr):
            self.mode = "SHADOW"

    def status(self):
        def _wr(xs):
            if not xs: 
                return None
            return sum(xs) / float(len(xs))
        def _fmt_pct(p):
            return "—" if p is None else f"{100.0*p:.2f}%"

        wr_a = _wr(self.active_hits)
        wr_s = _wr(self.shadow_hits)
        all_hits = (self.active_hits or []) + (self.shadow_hits or [])
        wr_all = _wr(all_hits)

        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "wr_active": _fmt_pct(wr_a),
            "n_active": len(self.active_hits or []),
            "wr_shadow": _fmt_pct(wr_s),
            "n_shadow": len(self.shadow_hits or []),
            "wr_all": _fmt_pct(wr_all),
            "n": len(all_hits)
        }




# =============================
# REST MODE
# =============================

def _prune_bets(bets: Dict[int, Dict], keep_settled_last: int = 500, keep_other_last: int = 200):
    settled = sorted([e for e, b in bets.items() if b.get("settled")])
    to_drop = settled[:-keep_settled_last] if len(settled) > keep_settled_last else []
    for e in to_drop:
        bets.pop(e, None)
    # чистим старые «не закрытые», включая skipped=True
    others = sorted([e for e, b in bets.items() if not b.get("settled")])
    to_drop2 = others[:-keep_other_last] if len(others) > keep_other_last else []
    for e in to_drop2:
        bets.pop(e, None)

def _settled_trades_count(csv_path: str) -> int:
    """Подсчет количества завершенных сделок в CSV."""
    try:
        if not os.path.exists(csv_path):
            return 0
        df = pd.read_csv(csv_path)
        # Считаем только сделки с результатом (settled=1 или has payout_ratio)
        if "settled" in df.columns:
            return int(df[df["settled"] == 1].shape[0])
        elif "payout_ratio" in df.columns:
            return int(df[df["payout_ratio"].notna()].shape[0])
        else:
            return len(df)
    except Exception:
        return 0

# =============================
# ГЛАВНЫЙ ЦИКЛ (с ансамблем)
# =============================
def main_loop():
    global rpc_fail_streak
    global DELTA_PROTECT  # будем менять модульную константу
    # --- Фаза-гистерезис --
    hours = None
    ml_cfg = MLConfig() 
    phase_filter = PhaseFilter(hysteresis_s=ml_cfg.phase_hysteresis_s)
    # (опционально можно загрузить last_phase/last_change_ts из JSON, если нужно переживать рестарт)
    # восстановить состояние
    try:
        if os.path.exists(ml_cfg.phase_state_path):
            with open(ml_cfg.phase_state_path, "r") as f:
                st = json.load(f)
            phase_filter.last_phase = st.get("last_phase", None)
            phase_filter.last_change_ts = st.get("last_change_ts", None)
    except Exception:
        pass
        # === δ: суточный подбор по последним 100 сделкам ===
    try:
        # Проверяем количество доступных сделок
        n_trades = _settled_trades_count(CSV_PATH)
        
        delta_daily = DeltaDaily(csv_path=CSV_PATH, state_path=DELTA_STATE_PATH,
                                n_last=min(100, n_trades),  # Не больше чем есть сделок
                                grid_start=0.000, grid_stop=0.100, grid_step=0.005,
                                csv_shadow_path=CSV_SHADOW_PATH,
                                window_hours=24,
                                opt_mode="dr_lcb")  # ✳️ анализируем последние 24 часа
        st = delta_daily.load_or_recompute_every_hours(period_hours=4)
        
        if st and isinstance(st.get("delta"), (int, float)):
            if n_trades < MIN_TRADES_FOR_DELTA:
                DELTA_PROTECT = 0.0
                print(
                    "[delta] startup(4h/24h): δ=0.000 (FORCED) "
                    f"| trades={n_trades}/{MIN_TRADES_FOR_DELTA} — копим статистику; "
                    f"p_opt={st.get('p_thr_opt', float('nan')):.4f} | avg_used={st.get('avg_p_thr_used', float('nan')):.4f}"
                )
            else:
                DELTA_PROTECT = float(st["delta"])
                method = str(st.get("method","?")).lower()
                if method == "dr_lcb":
                    # Проверяем наличие lcb15 в правильном формате
                    lcb_value = st.get('lcb15', 0.0)
                    # Защита от -1e9 при выводе
                    if lcb_value < -1000:
                        lcb_str = "N/A"
                    else:
                        lcb_str = f"{lcb_value:.6f}"
                    
                    print(
                        "[delta] startup(4h/24h): "
                        f"δ={DELTA_PROTECT:.3f} | method=DR-LCB | p_opt={st.get('p_thr_opt', float('nan')):.4f} | "
                        f"N={st.get('sample_size','?')}, picked={st.get('selected_n','?')} | "
                        f"LCB15={lcb_str} | window={st.get('window_hours','?')}h"
                    )
                elif method == "grid_pnl":
                    print(
                        "[delta] startup(4h/24h): "
                        f"δ={DELTA_PROTECT:.3f} | method=GRID-PnL | p_opt={st.get('p_thr_opt', float('nan')):.4f} | "
                        f"N={st.get('sample_size','?')}, picked={st.get('selected_n','?')} | "
                        f"P&L*={st.get('pnl_at_opt', float('nan')):.6f} BNB | window={st.get('window_hours','?')}h"
                    )
                else:
                    print(
                        "[delta] startup(4h/24h): "
                        f"δ={DELTA_PROTECT:.3f} | method=P*-AVG_USED | p_opt={st.get('p_thr_opt', float('nan')):.4f} | "
                        f"avg_used={st.get('avg_p_thr_used', float('nan')):.4f} | "
                        f"N={st.get('sample_size','?')}, picked={st.get('selected_n','?')} | "
                        f"P&L*={st.get('pnl_at_opt', float('nan')):.6f} BNB | window={st.get('window_hours','?')}h"
                    )

    except Exception as e:
        print(f"[warn] delta_daily init failed: {e}")
    
    # --- init web3/contract
    w3 = connect_web3_resilient()
    c = get_prediction_contract(w3)
    interval_sec = int(c.functions.intervalSeconds().call())
    buffer_sec   = int(c.functions.bufferSeconds().call())
    min_bet_bnb  = get_min_bet_bnb(c)
    print(f"[init] Connected. interval={interval_sec}s buffer={buffer_sec}s minBet={min_bet_bnb:.6f} BNB")
    if tg_enabled():
        tg_send(f"🤖 Bot online. interval={interval_sec}s, buffer={buffer_sec}s, minBet={min_bet_bnb:.6f} BNB.")

    # --- восстановим капитал из CSV (или из capital_state.json, если CSV пуст)
    capital_state = CapitalState(path=os.path.join(os.path.dirname(__file__), "capital_state.json"))
    cap_csv = _restore_capital_from_csv(CSV_PATH)
    if cap_csv is not None:
        capital = cap_csv
        cap_src = "trades_prediction.csv"
    else:
        capital = capital_state.load(START_CAPITAL_BNB)
        cap_src = "capital_state.json (fallback)" if os.path.exists(capital_state.path) else "default"
    print(f"[init] Capital restored: {capital:.6f} BNB (source={cap_src})")

    # --- монитор производительности (EV & log-growth)
    try:
        perf = PerfMonitor(
            path=os.path.join(os.path.dirname(__file__), "perf_state.json"),
            window_trades=500,              # можно 300–1000
            min_trades_for_report=50,       # минимум для отчёта
            fees_net=True                   # pnl уже NET → c=0 в p_BE
        )
        print("[init] PerfMonitor ready")
    except Exception as e:
        perf = None
        print(f"[warn] perf init failed: {e}")


    # --- резервный фонд: загрузка состояния
    try:
        reserve = ReserveFund(path=os.path.join(os.path.dirname(__file__), "reserve_state.json"), checkpoint_hour=23)
        # опционально покажем баланс при старте
        print(f"[init] Reserve balance: {reserve.balance:.6f} BNB")
    except Exception as e:
        reserve = None
        print(f"[warn] reserve init failed: {e}")




    # REST/WR трекер (используем CSV_PATH)
    # REST/WR трекер (используем CSV_PATH)
    stats = StatsTracker(csv_path=CSV_PATH)
    rest  = RestState.load(path="rest_state.json")
    rest_cfg = RestConfig(drop_for_rest4h=0.10, drop_for_rest24h=0.15,
                        min_trades_per_window=40, min_trades_after_rest4h=10)
    # 👇 добавляем атрибут напрямую (сработает даже если у класса нет __init__ с kwargs)
    rest_cfg.min_total_trades_for_rest = 500


    # сглаживание базовых вероятностей
    alpha = 2.0 / (SMOOTH_N + 1.0)
    p_up_ema = None
    p_ss = EhlersSuperSmoother(SS_LEN) if USE_SUPER_SMOOTHER else None

    logreg = OnlineLogReg(state_path="calib_logreg_state.json") if NN_USE else None
    wf = WalkForwardWeighter() if WF_USE else None
    if wf:
        print(f"[wf  ] init weights = {wf.w}")

    # --- ансамбль экспертов + мета
    xgb_exp = XGBExpert(ml_cfg)
    rf_exp  = RFCalibratedExpert(ml_cfg)
    arf_exp = RiverARFExpert(ml_cfg)
    nn_exp  = NNExpert(ml_cfg)

    # Если в этом файле уже есть переменные с токеном/чатом — подставляем их в cfg:
    ml_cfg.meta_report_dir = "meta_reports"    # опционально, куда сохранять PNG
    ml_cfg.phase_min_ready = 50                # ← старт обучения с 50 примеров/фазу
    ml_cfg.meta_retrain_every = 50             # ← тренироваться каждые 50 новых примеров
    meta    = MetaCEMMC(ml_cfg)

    meta.bind_experts(xgb_exp, rf_exp, arf_exp, nn_exp)

    # --- калибровщики и вторая МЕТА + блендер ---
    from calib.manager import OnlineCalibManager
    _CALIB_MGR = globals().get("_CALIB_MGR")
    if _CALIB_MGR is None:
        _CALIB_MGR = OnlineCalibManager()
        globals()["_CALIB_MGR"] = _CALIB_MGR
        try:
            import pandas as pd, numpy as np
            if os.path.exists(CSV_PATH):
                df_hist = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
                if {"p_meta_raw","outcome"}.issubset(df_hist.columns):
                    y_hist = (df_hist["outcome"].astype(str).str.lower()=="win").astype(int).to_numpy()
                    p_hist = df_hist["p_meta_raw"].astype(float).to_numpy()
                    mask = np.isfinite(p_hist)
                    if mask.sum() >= int(os.getenv("CALIB_MIN_N","300")):
                        _CALIB_MGR.fit_global(p_hist[mask], y_hist[mask])
        except Exception:
            pass

    _CALIB_MGR2 = globals().get("_CALIB_MGR2")
    if _CALIB_MGR2 is None:
        _CALIB_MGR2 = OnlineCalibManager()
        globals()["_CALIB_MGR2"] = _CALIB_MGR2

    _LM_META = globals().get("_LM_META")
    if _LM_META is None:
        _LM_META = LambdaMARTMetaLite(
            retrain_every=int(os.getenv("LM_RETRAIN_EVERY","80")),
            min_ready=int(os.getenv("LM_MIN_READY","160")),
            max_buf=int(os.getenv("LM_MAX_BUF","10000"))
        )
        globals()["_LM_META"] = _LM_META

    _BLENDER = globals().get("_BLENDER")
    if _BLENDER is None:
        _BLENDER = ProbBlender(
            metric=os.getenv("BLEND_METRIC","nll"),
            window=int(os.getenv("BLEND_WIN","1200")),
            step=float(os.getenv("BLEND_STEP","0.02"))
        )
        globals()["_BLENDER"] = _BLENDER

        # поднимем "глобальный" калибратор на истории CSV, если есть данные
        try:
            import pandas as pd, numpy as np
            if os.path.exists(CSV_PATH):
                df_hist = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
                if {"p_meta_raw","outcome"}.issubset(df_hist.columns):
                    y_hist = (df_hist["outcome"].astype(str).str.lower()=="win").astype(int).to_numpy()
                    p_hist = df_hist["p_meta_raw"].astype(float).to_numpy()
                    mask = np.isfinite(p_hist)
                    if mask.sum() >= int(os.getenv("CALIB_MIN_N","300")):
                        _CALIB_MGR.fit_global(p_hist[mask], y_hist[mask])
        except Exception:
            pass


    import atexit, signal, sys

    def _meta_flush(*_):
        try: meta._save_throttled(force=True)
        except: pass

    atexit.register(_meta_flush)
    try:
        signal.signal(signal.SIGTERM, _meta_flush)              # OK: мягко флашим при SIGTERM
        signal.signal(signal.SIGINT,  signal.default_int_handler)  # ← вернуть дефолт
    except Exception:
        pass
 

    def _status_line(name, st):
        return (f"{name}: enabled={st['enabled']}, mode={st['mode']}, "
                f"wr_act={st.get('wr_active','—')} (n={st.get('n_active','0')}), "
                f"wr_sh={st.get('wr_shadow','—')} (n={st.get('n_shadow','0')}), "
                f"wr_all={st.get('wr_all','—')} (n={st.get('n','0')})")

    print("[ens ] " + _status_line("XGB", xgb_exp.status()))
    print("[ens ] " + _status_line("RF ", rf_exp.status()))
    print("[ens ] " + _status_line("ARF", arf_exp.status()))
    print("[ens ] " + _status_line("NN ", nn_exp.status()))
    print("[ens ] " + _status_line("META", meta.status()))
    if tg_enabled():
        tg_send("🧠 Ensemble init:\n" +
                _status_line("XGB", xgb_exp.status()) + "\n" +
                _status_line("RF ", rf_exp.status()) + "\n" +
                _status_line("ARF", arf_exp.status()) + "\n" +
                _status_line("NN ",  nn_exp.status())  + "\n" +
                _status_line("META", meta.status()))
        # --- NEW: contexts for addons ---
    micro = MicrostructureClient(SESSION, SYMBOL)
    fut   = FuturesContext(SESSION, SYMBOL, min_refresh_sec=30)
    # bnbusdrt6.py (инициализация контекстов)
    pool  = PoolFeaturesCtx(k=10, late_sec=30)

    # НОВОЕ: персистентная 2D-таблица квантилей r̂ по (t_rem × pool)
    r2d   = RHat2D(state_path="rhat2d_state.json", pending_path="rhat2d_pending.json")

    gas_hist = GasHistory(maxlen=1200)  # ~20 минут при шаге 1с



    # кешы свечей/фич
    kl_df: Optional[pd.DataFrame] = None
    cross_df_map: Dict[str, Optional[pd.DataFrame]] = {}
    stab_df_map: Dict[str, Optional[pd.DataFrame]] = {}
    feats: Optional[Dict[str, pd.Series]] = None
    cross_feats_map: Dict[str, Dict[str, pd.Series]] = {}
    stab_feats_map: Dict[str, Dict[str, pd.Series]] = {}

    # фабрика расширенных признаков для экспертов
    ext_builder = ExtendedMLFeatures()

    bets: Dict[int, Dict] = {}
    last_seen_epoch = None
    print("[loop] Press Ctrl+C to stop.")

    rpc_fail_streak = 0
    RPC_FAIL_MAX = 5
    _last_gc = 0  # unix-ts последнего ручного GC

    # OU/Logit-OU
    ou_skew = OUOnlineSkew(dt_unit=OU_SKEW_DT_UNIT, decay=OU_SKEW_DECAY) if OU_SKEW_USE else None
    logit_ou = LogitOUSmoother(half_life_sec=LOGIT_OU_HALF_LIFE_SEC,
                               mu_beta=LOGIT_OU_MU_BETA,
                               z_clip=LOGIT_OU_Z_CLIP) if LOGIT_OU_USE else None

    def notify_ens_used(p_base: Optional[float],
                        px: Optional[float], prf: Optional[float], parf: Optional[float], pnn: Optional[float],
                        p_final: Optional[float], used: bool, meta_mode: str):
        try:
            if used and p_final is not None:
                tg_send(
                    "📊 ENS used=<b>YES</b>\n"
                    f"mode=<b>{meta_mode}</b>, "
                    f"p_base={fmt_prob(p_base)}, "
                    f"p_xgb={fmt_prob(px)}, "
                    f"p_rf={fmt_prob(prf)}, "
                    f"p_arf={fmt_prob(parf)}, "
                    f"p_nn={fmt_prob(pnn)}, "
                    f"p_final={fmt_prob(p_final)}"
                )

            else:
                s_x = xgb_exp.status(); s_r = rf_exp.status(); s_a = arf_exp.status(); s_n = nn_exp.status(); s_m = meta.status()
                tg_send("📊 ENS used=no\n"
                        + _status_line("XGB", s_x) + "\n"
                        + _status_line("RF ", s_r) + "\n"
                        + _status_line("ARF", s_a) + "\n"
                        + _status_line("NN ", s_n) + "\n"
                        + _status_line("META", s_m))
        except Exception:
            pass

    while True:
        try:
            now = int(time.time())

            # — пересчёт «хороших часов» по окну 14 дней (по умолчанию) каждые 4 часа
            try:
                if 'hours' in globals() and hours is not None:
                    hours.maybe_recompute(now_ts=now)
            except Exception as e:
                print(f"[hours] recompute failed: {e}")

            # — ежедневная отсечка резерва в 23:00 UTC (при первом тике после 23:00)
            try:
                if reserve is not None:
                    evt = reserve.maybe_eod_rebalance(now_ts=now, capital=capital)
                    if evt and evt.get("changed"):
                        capital = float(evt["capital"])
                        try:
                            capital_state.save(capital, ts=now)  # сохраняем новый рабочий капитал
                        except Exception as e:
                            print(f"[warn] capital_state save failed: {e}")
                        # информируем в TG (тихо игнорим сбои)
                        try:
                            tg_send(evt["message"])
                        except Exception:
                            pass
            except Exception as e:
                print(f"[reserve] eod rebalance failed: {e}")



            # --- текущий epoch
            try:
                cur = int(c.functions.currentEpoch().call())
                rpc_fail_streak = 0
            except Exception as e:
                print(f"[rpc ] currentEpoch failed: {e}")
                rpc_fail_streak += 1
                if rpc_fail_streak >= RPC_FAIL_MAX:
                    try:
                        w3 = connect_web3()
                        c = get_prediction_contract(w3)
                        rpc_fail_streak = 0
                        print("[rpc ] reconnected")
                    except Exception as ee:
                        print(f"[rpc ] reconnect failed: {ee}")
                time.sleep(1.0)
                continue

            if last_seen_epoch != cur:
                print(f"\n[epoch] currentEpoch={cur} (time={now})")
                last_seen_epoch = cur
            try:
                try_settle_shadow_rows(CSV_SHADOW_PATH, w3, c, cur)
            except Exception as e:
                print(f"[shadow] settle pass failed: {e}")


            try:
                st = delta_daily.maybe_update_every_hours(period_hours=4, now_ts=now)
                if st and isinstance(st.get("delta"), (int, float)):
                    n_trades = _settled_trades_count(CSV_PATH)

                    if n_trades < MIN_TRADES_FOR_DELTA:
                        DELTA_PROTECT = 0.0
                        try:
                            tg_send(
                                "⚙️ Обновление δ (каждые 4ч)\n"
                                f"δ=<b>0.000</b> (FORCED: trades={n_trades}/{MIN_TRADES_FOR_DELTA})\n"
                                f"p_opt=<b>{st.get('p_thr_opt', float('nan')):.4f}</b> | "
                                f"avg_used=<b>{st.get('avg_p_thr_used', float('nan')):.4f}</b>\n"
                                f"N={st.get('sample_size','?')}  взяли={st.get('selected_n','?')}\n"
                                f"P&L*={st.get('pnl_at_opt', float('nan')):.6f} BNB\n"
                                "<i>* по подмножеству исторически взятых сделок</i>"
                            )
                        except Exception:
                            pass

                    elif (meta.mode != "ACTIVE") or (not had_trade_in_last_hours(CSV_PATH, 1.0)):
                        DELTA_PROTECT = 0.0
                        reason = "meta≠ACTIVE" if meta.mode != "ACTIVE" else "idle≥1h"
                        try:
                            tg_send(
                                "⚙️ Обновление δ (каждые 4ч)\n"
                                f"δ=<b>0.000</b> (DISABLED: {reason})\n"
                                f"p_opt=<b>{st.get('p_thr_opt', float('nan')):.4f}</b> | "
                                f"avg_used=<b>{st.get('avg_p_thr_used', float('nan')):.4f}</b>\n"
                                f"N={st.get('sample_size','?')}  взяли={st.get('selected_n','?')}\n"
                                f"P&L*={st.get('pnl_at_opt', float('nan')):.6f} BNB\n"
                                "<i>* по подмножеству исторически взятых сделок</i>"
                            )
                        except Exception:
                            pass

                    else:
                        DELTA_PROTECT = float(st["delta"])
                        try:
                            tg_send(
                                "⚙️ Обновление δ (каждые 4ч)\n"
                                f"δ=<b>{DELTA_PROTECT:.3f}</b>\n"
                                f"p_opt=<b>{st.get('p_thr_opt', float('nan')):.4f}</b> | "
                                f"avg_used=<b>{st.get('avg_p_thr_used', float('nan')):.4f}</b>\n"
                                f"N={st.get('sample_size','?')}  взяли={st.get('selected_n','?')}\n"
                                f"P&L*={st.get('pnl_at_opt', float('nan')):.6f} BNB\n"
                                "<i>* по подмножеству исторически взятых сделок</i>"
                            )
                        except Exception:
                            pass
            except Exception as e:
                print(f"[delta] update failed: {e}")



            # проверяем активные/пропущенные
            pending = sorted([e for e, b in bets.items() if not b.get("settled") and e < cur - 1])[-50:]

            for epoch in [cur, cur - 1] + pending:
                if epoch <= 0:
                    continue

                try:
                    rd = get_round(w3, c, epoch)
                    if rd is None:
                        print(f"[skip] epoch={epoch} (rpc timeout)")
                        continue  # переходим к следующему циклу/ожиданию
                    rpc_fail_streak = 0
                except Exception as e:
                    print(f"[rpc ] get_round({epoch}) failed: {e}")
                    rpc_fail_streak += 1
                    if rpc_fail_streak >= RPC_FAIL_MAX:
                        try:
                            w3 = connect_web3()
                            c = get_prediction_contract(w3)
                            rpc_fail_streak = 0
                            print("[rpc ] reconnected")
                        except Exception as ee:
                            print(f"[rpc ] reconnect failed: {ee}")
                    continue

                # ============= ЗАМЕНИТЬ БЛОК (строки ~975-995) =============

                if epoch not in bets:
                    # === COOLING PERIOD: умная пауза после серии проигрышей ===
                    if epoch == cur and now < rd.lock_ts:
                        try:
                            df_recent = _read_csv_df(CSV_PATH).sort_values("settled_ts")
                            if not df_recent.empty:
                                # Смотрим на последние 5 сделок (вместо 3)
                                recent_trades = df_recent[df_recent["outcome"].isin(["win", "loss"])].tail(5)
                                
                                if len(recent_trades) >= 5:
                                    # Считаем количество проигрышей
                                    losses = (recent_trades["outcome"] == "loss").sum()
                                    
                                    # Анализируем КАЧЕСТВО проигрышей (edge_at_entry)
                                    loss_rows = recent_trades[recent_trades["outcome"] == "loss"]
                                    
                                    # Безопасно извлекаем edge_at_entry
                                    loss_edges = pd.to_numeric(
                                        loss_rows.get("edge_at_entry", pd.Series(dtype=float)), 
                                        errors="coerce"
                                    ).dropna()
                                    
                                    avg_loss_edge = float(loss_edges.mean()) if len(loss_edges) > 0 else 0.0
                                    
                                    # === УСЛОВИЯ ДЛЯ COOLING ===
                                    # 1) 3+ проигрыша из последних 5
                                    # 2) Средний edge проигрышей >= 0.03 (не маргинальные ставки)
                                    cooling_needed = (losses >= 3) and (avg_loss_edge >= 0.03)
                                    
                                    if cooling_needed:
                                        last_loss_ts = int(recent_trades[recent_trades["outcome"] == "loss"].iloc[-1]["settled_ts"])
                                        hours_since = (now - last_loss_ts) / 3600.0
                                        COOLDOWN_HOURS = 1.0  # было 2.0
                                        
                                        if hours_since < COOLDOWN_HOURS:
                                            bets[epoch] = dict(
                                                skipped=True, 
                                                reason="cooling_period", 
                                                wait_polls=0, 
                                                settled=False
                                            )
                                            
                                            # Детальное сообщение
                                            print(f"[cool] epoch={epoch} COOLING: {losses}/5 losses "
                                                f"(avg_edge={avg_loss_edge:.3f}) | "
                                                f"wait {COOLDOWN_HOURS-hours_since:.1f}h more")
                                            
                                            send_round_snapshot(
                                                prefix=f"🧊 <b>Cooling</b> epoch={epoch}",
                                                extra_lines=[
                                                    f"Пауза после {losses}/5 проигрышей (последний час).",
                                                    f"Средний edge проигрышей: {avg_loss_edge:.3f}",
                                                    f"Осталось: {COOLDOWN_HOURS-hours_since:.1f}ч"
                                                ]
                                            )
                                            
                                            notify_ens_used(None, None, None, None, None, None, False, meta.mode)
                                            continue
                                
                        except Exception as e:
                            print(f"[cool] check failed: {e}")
                    
                    # --- стадия принятия решения
                    if epoch == cur and now < rd.lock_ts:
                        # --- Guard: ждём до окна последних 15 секунд перед lock
                        time_left = rd.lock_ts - now
                        if time_left > GUARD_SECONDS:
                            # ещё рано принимать решение: накапливаем снапшоты пулов и продолжаем
                            pool.observe(epoch, now, rd.bull_amount, rd.bear_amount)
                            continue




                        # --- тики/фичи к lock-1s
                        t_lock = pd.to_datetime((rd.lock_ts - 1) * 1000, unit="ms", utc=True)
                        need_until_ms = int(t_lock.timestamp() * 1000)

                        kl_df = ensure_klines_cover(kl_df, SYMBOL, BINANCE_INTERVAL, need_until_ms)
                        if kl_df is None or kl_df.empty:
                            continue
                        feats = features_from_binance(kl_df)

                        if USE_CROSS_ASSETS:
                            cross_df_map = ensure_klines_cover_map(cross_df_map, CROSS_SYMBOLS, BINANCE_INTERVAL, need_until_ms)
                            stab_df_map  = ensure_klines_cover_map(stab_df_map,  STABLE_SYMBOLS, BINANCE_INTERVAL, need_until_ms)
                            cross_feats_map = features_for_symbols(cross_df_map)
                            stab_feats_map  = features_for_symbols(stab_df_map)

                        if is_chop_at_time(feats, t_lock):
                            bets[epoch] = dict(skipped=True, reason="chop", wait_polls=0, settled=False)
                            print(f"[skip] epoch={epoch} (chop)")
                            send_round_snapshot(
                                prefix=f"⛔ <b>Skip</b> epoch={epoch} (болото ATR/CHOP)",
                                extra_lines=[f"Причина: chop (низкая волатильность)."]
                            )
                            notify_ens_used(None, None, None, None, None, None, False, meta.mode)
                            continue

                        # --- базовая вероятность (как раньше)
                        w_for_prob = wf.w if (wf is not None) else None
                        P_up, P_dn, wf_phi_dict = prob_up_down_at_time(feats, t_lock, w_for_prob)

                        # отладка амплитуды базовой вероятности
                        try:
                            phi_dbg = np.array([
                                wf_phi_dict.get("phi_wf0", 0.0),
                                wf_phi_dict.get("phi_wf1", 0.0),
                                wf_phi_dict.get("phi_wf2", 0.0),
                                wf_phi_dict.get("phi_wf3", 0.0),
                            ], dtype=float)
                            w_dbg = (wf.w if (wf is not None) else np.array([0.35, 0.20, 0.20, 0.25], dtype=float))
                            z_up = float(np.dot(w_dbg, phi_dbg))
                            print(f"[base] ||w||={np.linalg.norm(w_dbg):.3f} logit={z_up:+.4f} P_up_raw={P_up:.4f}")
                        except Exception:
                            pass


                        if USE_SUPER_SMOOTHER and p_ss is not None:
                            P_up = float(np.clip(p_ss.update(P_up), 0.0, 1.0))
                            P_dn = 1.0 - P_up
                        else:
                            p_up_ema = P_up if p_up_ema is None else (alpha * P_up + (1 - alpha) * p_up_ema)
                            P_up = p_up_ema
                            P_dn = 1.0 - P_up

                        # NN-калибратор поверх фич (старый)
                        phi, i = None, _index_pad(feats["M_up"], t_lock)
                        if NN_USE and i is not None:
                            phi = np.array([
                                float(feats["M_up"].iloc[i] - feats["M_dn"].iloc[i]),
                                float(feats["S_up"].iloc[i] - feats["S_dn"].iloc[i]),
                                float(feats["B_up"].iloc[i] - feats["B_dn"].iloc[i]),
                                float(feats["R_up"].iloc[i] - feats["R_dn"].iloc[i]),
                                1.0
                            ], dtype=float)
                            if logreg is not None:
                                p_nncal = logreg.predict(phi)
                                P_up = (1.0 - BLEND_NN) * P_up + BLEND_NN * p_nncal
                                P_dn = 1.0 - P_up

                        # кросс-активы
                        if USE_CROSS_ASSETS:
                            zc_up1, zc_dn1 = cross_up_down_contrib(cross_feats_map, t_lock, CROSS_SYMBOLS, CROSS_W_MOM, CROSS_W_VWAP, CROSS_SHIFT_BARS)
                            zc_up2, zc_dn2 = cross_up_down_contrib(stab_feats_map,  t_lock, STABLE_SYMBOLS, STABLE_W_MOM, STABLE_W_VWAP, CROSS_SHIFT_BARS)
                            delta_logit = CROSS_ALPHA * ((zc_up1 + zc_up2) - (zc_dn1 + zc_dn2))
                            P_up = from_logit(to_logit(P_up) + float(delta_logit))
                            P_up = float(np.clip(P_up, 0.0, 1.0))
                            P_dn = 1.0 - P_up

                        P_up = elder_logit_adjust(kl_df, t_lock, P_up)
                        P_dn = 1.0 - P_up

                        # OU-добавки
                        if OU_SKEW_USE and "Zs" in feats:
                            j = _index_pad(feats["Zs"], t_lock)
                            if j is not None and j > 0:
                                z_prev = float(np.clip(feats["Zs"].iloc[j-1], -OU_SKEW_Z_CLIP, OU_SKEW_Z_CLIP))
                                z_now  = float(np.clip(feats["Zs"].iloc[j],   -OU_SKEW_Z_CLIP, OU_SKEW_Z_CLIP))
                                ou_skew.update_pair(z_prev, z_now)
                                horizon_sec = max(1.0, rd.close_ts - rd.lock_ts)
                                res = ou_skew.prob_above_zero(z_now, horizon_sec)
                                if res is not None:
                                    p_ou_up, strength = res
                                    z_base = to_logit(P_up)
                                    z_ou = to_logit(p_ou_up)
                                    amp = float(np.clip((abs(z_now) - OU_SKEW_THR) / max(1e-6, OU_SKEW_THR), 0.0, 1.0))
                                    lam = min(OU_SKEW_LAMBDA_MAX, strength) * amp
                                    z_mix = (1.0 - lam) * z_base + lam * z_ou
                                    P_up = from_logit(z_mix)
                                    P_dn = 1.0 - P_up

                        if LOGIT_OU_USE and logit_ou is not None:
                            z_now = to_logit(P_up)
                            logit_ou.update_mu(z_now)
                            horizon_sec = max(1.0, rd.close_ts - rd.lock_ts)
                            z_pred = logit_ou.predict_future(z_now, horizon_sec)
                            P_up = from_logit(z_pred)
                            P_dn = 1.0 - P_up



                        # --- NEW: microstructure/futures/pools/jumps/liquidity/time/gas/idio

                        # 1) Микроструктура к lock-1s
                        end_ms = int(t_lock.timestamp()*1000)
                        micro_feats = micro.compute(end_ms)  # rel_spread, book_imb, microprice_delta, ofi_5s/15s/30s, ob_slope, mid

                        # 2) Фьючерсы (refresh ≈ раз в 30с)
                        fut.refresh()
                        spot_mid = micro_feats.get("mid", float(kl_df["close"].iloc[-1]))
                        fut_feats = fut.features(spot_mid)

                        # 3) Пулы Prediction: копим наблюдения и извлекаем фичи к lock-1s
                        pool.observe(epoch, now, rd.bull_amount, rd.bear_amount)
                        pool.update_streak_from_rounds(lambda e: get_round(w3, c, e), cur)
                        pool_feats = pool.features(epoch, rd.lock_ts)

                        # 4) Волатильность/джампы (BV/RQ/RV) на окнах 20/60/120 баров
                        RV20,BV20,RQ20,n20 = realized_metrics(kl_df["close"], 20)
                        RV60,BV60,RQ60,n60 = realized_metrics(kl_df["close"], 60)
                        RV120,BV120,RQ120,n120 = realized_metrics(kl_df["close"], 120)
                        jump20 = jump_flag_from_rv_bv_rq(RV20,BV20,RQ20,n20, z_thr=3.0)
                        jump60 = jump_flag_from_rv_bv_rq(RV60,BV60,RQ60,n60, z_thr=3.0)

                        # 5) Ликвидность/импакт
                        amihud = amihud_illiq(kl_df, win=20)
                        kyle   = kyle_lambda(kl_df, win=20)

                        # 6) Интрадей-профиль времени
                        time_feats = intraday_time_features(t_lock)

                        # 7) Кросс-активы: «очищенный» ретёрн и динамика беты
                        btc_df = cross_df_map.get("BTCUSDT")
                        eth_df = cross_df_map.get("ETHUSDT")
                        idio = idio_features(kl_df, btc_df, eth_df, look_min=240)

                        # 8) Газ — дельта/волатильность
                        gas_gwei_now = float(get_gas_price_wei(w3))/1e9
                        gas_hist.push(now, gas_gwei_now)
                        gas_feats = gas_hist.features(now)

                        # Сбор всех новых фич в единый вектор (фиксированный порядок ключей):
                        addon_dict = {}
                        addon_dict.update({
                            "rel_spread": micro_feats.get("rel_spread", 0.0),
                            "book_imb": micro_feats.get("book_imb", 0.0),
                            "microprice_delta": micro_feats.get("microprice_delta", 0.0),
                            "ofi_5s": micro_feats.get("ofi_5s", 0.0),
                            "ofi_15s": micro_feats.get("ofi_15s", 0.0),
                            "ofi_30s": micro_feats.get("ofi_30s", 0.0),
                            "ob_slope": micro_feats.get("ob_slope", 0.0),
                            "funding_sign": fut_feats.get("funding_sign", 0.0),
                            "funding_timeleft": fut_feats.get("funding_timeleft", 0.0),
                            "dOI_1m": fut_feats.get("dOI_1m", 0.0),
                            "dOI_5m": fut_feats.get("dOI_5m", 0.0),
                            "basis_now": fut_feats.get("basis_now", 0.0),
                            "pool_logit": pool_feats.get("pool_logit", 0.0),
                            "pool_logit_d30": pool_feats.get("pool_logit_d30", 0.0),
                            "pool_logit_d60": pool_feats.get("pool_logit_d60", 0.0),
                            "late_money_share": pool_feats.get("late_money_share", 0.0),
                            "last_k_outcomes_mean": pool_feats.get("last_k_outcomes_mean", 0.0),
                            "last_k_payout_median": pool_feats.get("last_k_payout_median", 0.0),
                            "bv_over_rv_20": (BV20/max(1e-12, RV20)),
                            "bv_over_rv_60": (BV60/max(1e-12, RV60)),
                            "rq_norm_20": RQ20,
                            "rq_norm_60": RQ60,
                            "jump20": float(jump20),
                            "jump60": float(jump60),
                            "amihud_illq": amihud,
                            "kyle_lambda": kyle,
                            "resid_ret_1m": idio.get("resid_ret_1m", 0.0),
                            "beta_sum": idio.get("beta_sum", 0.0),
                            "beta_sum_d60": idio.get("beta_sum_d60", 0.0),
                            "gas_d1m": gas_feats.get("gas_d1m", 0.0),
                            "gas_vol5m": gas_feats.get("gas_vol5m", 0.0),
                        })
                        addon_dict.update(time_feats)

                        addon_names = [
                            "rel_spread","book_imb","microprice_delta","ofi_5s","ofi_15s","ofi_30s","ob_slope",
                            "funding_sign","funding_timeleft","dOI_1m","dOI_5m","basis_now",
                            "pool_logit","pool_logit_d30","pool_logit_d60","late_money_share",
                            "last_k_outcomes_mean","last_k_payout_median",
                            "bv_over_rv_20","bv_over_rv_60","rq_norm_20","rq_norm_60","jump20","jump60",
                            "amihud_illq","kyle_lambda","resid_ret_1m","beta_sum","beta_sum_d60",
                            "gas_d1m","gas_vol5m",
                            "tod_sin","tod_cos","EU","US","ASIA",
                            "dow_0","dow_1","dow_2","dow_3","dow_4","dow_5","dow_6"
                        ]
                        x_addon, _ = pack_vector(addon_dict, addon_names)

                        # --- ТВОЙ СТАРЫЙ x_ml ---
                        x_ml = ext_builder.build(kl_df, feats, t_lock)

                        # --- КОНКАТЕНАЦИЯ ---
                        x_ml = np.concatenate([x_ml, x_addon], axis=0)

                        # --- NEW: режим (ψ) для контекстного гейтинга
                        reg_ctx = build_regime_ctx(
                            kl_df, feats, t_lock,
                            micro_feats=micro_feats,
                            fut_feats=fut_feats,
                            jump_flag=max(float(jump20), float(jump60))
                        )


                        # анти-дрожь фазы
                        from meta_ctx import phase_from_ctx
                        phase_raw = int(phase_from_ctx(reg_ctx))
                        phase_stable = int(phase_filter.update(phase_raw, now_ts=int(t_lock.timestamp())))
                        reg_ctx["phase_raw"] = phase_raw
                        reg_ctx["phase"] = phase_stable  # ← использовать везде далее
                        try:
                            with open(ml_cfg.phase_state_path, "w") as f:
                                json.dump({
                                    "last_phase": int(phase_stable),
                                    "last_change_ts": int(t_lock.timestamp()),
                                }, f)
                        except Exception:
                            pass
                       

                        p_xgb, m_xgb = xgb_exp.proba_up(x_ml, reg_ctx=reg_ctx)
                        p_rf,  m_rf  = rf_exp.proba_up(x_ml,  reg_ctx=reg_ctx)
                        p_arf, m_arf = arf_exp.proba_up(x_ml, reg_ctx=reg_ctx)
                        p_nn,  m_nn  = nn_exp.proba_up(x_ml,   reg_ctx=reg_ctx)



                        p_base_before_ens = P_up
                        p_final = meta.predict(p_xgb, p_rf, p_arf, p_nn, p_base_before_ens, reg_ctx=reg_ctx)
                        ens_used = False
                        if meta.mode == "ACTIVE" and p_final is not None:
                            # 1) «сырое» p от основной МЕТА
                            p_meta_raw = float(np.clip(p_final, 0.0, 1.0))

                            # 2) «сырое» p от LambdaMART-МЕТА (может быть None до обучения)
                            LM = globals().get("_LM_META")
                            p_meta2_raw = None
                            try:
                                p_meta2_raw = LM.predict(p_xgb, p_rf, p_arf, p_nn, p_base_before_ens, reg_ctx=reg_ctx) if LM else None
                            except Exception:
                                p_meta2_raw = None

                            # 3) калибровка обеих мет
                            calib1 = globals().get("_CALIB_MGR")
                            calib2 = globals().get("_CALIB_MGR2")
                            p1_cal = float(calib1.transform(p_meta_raw)) if calib1 else p_meta_raw
                            p2_cal = (float(calib2.transform(p_meta2_raw)) if (calib2 and p_meta2_raw is not None) else p1_cal)

                            calib_src  = "calib[roll/global]" if calib1 else "calib[off]"
                            calib2_src = ("calib2[roll/global]" if (calib2 and p_meta2_raw is not None) else "calib2[off]")

                            # 4) смешивание по NLL/Brier на скользящем окне
                            BL = globals().get("_BLENDER")
                            blend_w = float(BL.w) if BL else 1.0
                            try:
                                P_up = float(BL.mix(p1_cal, p2_cal)) if BL else float(p1_cal)
                                calib_src = f"blend[{BL.metric},w={BL.w:.2f}]" if BL else calib_src
                            except Exception:
                                P_up = float(p1_cal)
                                calib_src = f"blend[fallback]"

                            # для логов/CSV
                            p_blend = float(P_up)

                            P_dn = 1.0 - P_up
                            ens_used = True


                        # --- выбор стороны
                        # --- выбор стороны
                        bet_up = P_up >= P_dn
                        p_side_raw = P_up if bet_up else P_dn
                        
                        # === SHRINKAGE: подтягиваем к 0.5 для снижения overconfidence ===
                        # === АДАПТИВНЫЙ Shrinkage: меньше для высокого края ===
                        edge_est = abs(p_side_raw - 0.5)

                        if edge_est > 0.10:  # очень уверенный прогноз
                            shrinkage = 0.05  # 5% - почти не трогаем
                        elif edge_est > 0.06:  # средняя уверенность
                            shrinkage = 0.10  # 10%
                        else:  # низкая уверенность
                            shrinkage = 0.15  # 15%

                        p_side = 0.5 + (p_side_raw - 0.5) * (1.0 - shrinkage)
                        print(f"[shrink] p_raw={p_side_raw:.4f} → p_conservative={p_side:.4f} (Δ={p_side-p_side_raw:+.4f}, shrink={shrinkage:.2f})")
                        
                        side = "UP" if bet_up else "DOWN"



                        # gas now
                        # gas now
                        gas_price_wei = 0
                        try:
                            gas_price_wei = get_gas_price_wei(w3)
                            rpc_fail_streak = 0
                        except Exception as e:
                            print(f"[rpc ] gas_price failed: {e}")
                            rpc_fail_streak += 1
                            gas_price_wei = 3_000_000_000
                            if rpc_fail_streak >= RPC_FAIL_MAX:
                                try:
                                    w3 = connect_web3()
                                    c = get_prediction_contract(w3)
                                    rpc_fail_streak = 0
                                    print("[rpc ] reconnected successfully")
                                except Exception as ee:
                                    print(f"[rpc ] reconnect failed: {ee}")
                            
                        gas_bet_bnb_cur = _as_float(GAS_USED_BET * gas_price_wei / 1e18, 0.0)
                        gas_claim_bnb_cur = _as_float(GAS_USED_CLAIM * gas_price_wei / 1e18, 0.0)

                        # =============================
                        # УЛУЧШЕННАЯ ОЦЕНКА r̂
                        # =============================
                        
                        # Подготовка: обновить 2D-таблицу и газовые оценки
                        r_med, gb_med, gc_med = last3_ev_estimates(CSV_PATH)
                        r_med = _as_float(r_med, None)
                        gb_med = _as_float(gb_med, None)
                        gc_med = _as_float(gc_med, None)
                        
                        try:
                            r2d.ingest_settled(CSV_PATH)
                        except Exception:
                            pass
                        
                        _now_ts = int(time.time())
                        t_rem_s = max(0, int(_as_float(getattr(rd, "lock_ts", _now_ts), _now_ts) - _now_ts))
                        if t_rem_s <= 2:
                            bets[epoch] = dict(skipped=True, reason="late", wait_polls=0, settled=False)
                            print(f"[late] epoch={epoch} missed betting window")
                            notify_ens_used(None, None, None, None, None, None, False, meta.mode)
                            continue
                        pool_tot = _as_float(getattr(rd, "bull_amount", 0.0), 0.0) + _as_float(getattr(rd, "bear_amount", 0.0), 0.0)
                        
                        try:
                            r2d.observe_epoch(epoch=int(epoch), t_rem_s=int(t_rem_s), pool_total_bnb=float(pool_tot))
                        except Exception:
                            pass
                        
                        # НОВАЯ ФУНКЦИЯ из модуля: приоритет IMPLIED → историческим методам
                        # НОВАЯ ФУНКЦИЯ из модуля: приоритет IMPLIED → историческим методам
                        try:
                            r_hat, r_hat_source = estimate_r_hat_improved(
                                rd=rd,
                                bet_up=bet_up,
                                epoch=epoch,
                                pool=pool,
                                csv_path=CSV_PATH,
                                kl_df=kl_df,
                                treasury_fee=TREASURY_FEE,
                                use_stress_r15=USE_STRESS_R15,
                                r2d=r2d
                            )
                            r_hat = _as_float(r_hat, 1.90)
                            r_hat_source = str(r_hat_source) if r_hat_source else "unknown"
                        except Exception as e:
                            print(f"[r_hat] estimate_r_hat_improved failed: {e}")
                            r_hat = 1.90
                            r_hat_source = "fallback_after_error"
                        
                        print(f"[r_hat] {r_hat:.4f} from {r_hat_source}")
                        
                        # Газовые оценки (без изменений)
                        gb_hat = _as_float(gb_med if (gb_med is not None and math.isfinite(_as_float(gb_med))) else gas_bet_bnb_cur, 0.0)
                        gc_hat = _as_float(gc_med if (gc_med is not None and math.isfinite(_as_float(gc_med))) else gas_claim_bnb_cur, 0.0)


                        total_settled = settled_trades_count(CSV_PATH)
                        has_recent = had_trade_in_last_hours(CSV_PATH, 1.0)
                        bootstrap_phase = (total_settled < MIN_TRADES_FOR_DELTA)

                        cap3 = MAX_STAKE_FRACTION * capital
                        if cap3 < min_bet_bnb:
                            bets[epoch] = dict(skipped=True, reason="cap3_lt_minbet", wait_polls=0, settled=False)
                            print(f"[skip] epoch={epoch} (cap 3% < minBet) cap3={cap3:.6f} minBet={min_bet_bnb:.6f}")
                            send_round_snapshot(
                                prefix=f"⛔ <b>Skip</b> epoch={epoch} (cap 3% ≤ minBet)",
                                extra_lines=[f"cap3={cap3:.6f} BNB ≤ minBet={min_bet_bnb:.6f} BNB"]
                            )
                            notify_ens_used(p_base_before_ens, p_xgb, p_rf, p_arf, p_nn, p_final, False, meta.mode)
                            continue

                        # НОВОЕ: контекстная калибровка p → p_ctx
                        # p_side здесь — «сырое» после ансамбля/сглаживаний; заменим его на p_ctx
                        try:
                            p_ctx = p_ctx_calibrated(p_raw=float(p_side), r_hat=float(r_hat), csv_path=CSV_PATH, max_epoch_exclusive=epoch)
                            p_side = float(np.clip(p_ctx, 0.0, 1.0))
                        except Exception:
                            p_side = float(np.clip(p_side, 0.0, 1.0))

                        if bootstrap_phase:
                            stake = max(min_bet_bnb, 0.01 * capital)
                            stake = min(stake, cap3)
                            kelly_half = None
                        else:
                            # --- Kelly по рынку (риск/выплата r̂): f* = (p*r̂ - 1) / (r̂ - 1) ---
                            # --- Kelly по рынку (риск/выплата r̂): f* = (p*r̂ - 1) / (r̂ - 1) ---
                            denom_r = max(1e-6, float(r_hat) - 1.0)
                            f_kelly_base = float(max(0.0, (p_side * float(r_hat) - 1.0) / denom_r))
                            if not math.isfinite(f_kelly_base):
                                f_kelly_base = 0.0

                            # калибровочная и волатильностная поправки — как и раньше
                            calib_err = rolling_calib_error(CSV_PATH, n=200)   # ~ECE proxy
                            calib_err = float(np.clip(calib_err, 0.0, 0.15))
                            f_calib = float(np.clip(1.0 - 2.0*calib_err, 0.5, 1.0))
                            if not math.isfinite(f_calib):
                                f_calib = 1.0

                            sigma_star = 0.01
                            sigma_realized = realized_sigma_g(CSV_PATH, n=200)
                            sigma_realized = max(sigma_realized, 1e-6)
                            if not math.isfinite(sigma_realized):
                                sigma_realized = 1e-6
                            f_vol = float(np.clip(sigma_star / sigma_realized, 0.5, 2.0))

                            # ============================================
                            # === Kelly/10 с адаптивным капом ===
                            # ============================================
                            
                            KELLY_DIVISOR = 10  # было 16
                            
                            # Вычисляем эффективный Kelly (base * calibration)
                            f_eff = f_kelly_base * f_calib
                            if not math.isfinite(f_eff):
                                f_eff = 0.0
                            
                            # Применяем делитель (Kelly/10)
                            f_eff_scaled = f_eff * (1.0 / float(KELLY_DIVISOR))
                            
                            # Адаптивный кап: зависит от уверенности
                            edge = p_side - (1.0 / r_hat)
                            
                            if edge > 0.08:  # высокая уверенность
                                f_cap = 0.015  # 1.5%
                            elif edge > 0.05:  # средняя уверенность
                                f_cap = 0.010  # 1.0%
                            else:
                                f_cap = 0.006  # 0.6%
                            
                            f_eff_scaled = min(f_eff_scaled, f_cap)
                            
                            # Применяем волатильность
                            frac = float(np.clip(f_eff_scaled, 0.001, 0.015))  # макс 1.5%
                            frac *= f_vol
                            
                            # Масштаб в просадке (без дополнительного множителя 0.5)
                            try:
                                dd_scale = _dd_scale_factor(CSV_PATH)
                                frac *= dd_scale
                                print(f"[kelly] f_base={f_kelly_base:.5f}, f_eff={f_eff:.5f}, "
                                      f"f_scaled={f_eff_scaled:.5f}, edge={edge:.4f}, "
                                      f"dd_scale={dd_scale:.3f}, final frac={frac:.5f}")
                            except Exception:
                                print(f"[kelly] f_base={f_kelly_base:.5f}, f_eff={f_eff:.5f}, "
                                      f"f_scaled={f_eff_scaled:.5f}, edge={edge:.4f}, "
                                      f"final frac={frac:.5f}")
                            
                            # Kelly для информации (совместимость с остальным кодом)
                            kelly_half = f_eff_scaled  # для логов
                            
                            stake = max(min_bet_bnb, frac * capital)
                            stake = min(stake, cap3)




                        if stake <= 0 or capital < min_bet_bnb * 1.0:
                            bets[epoch] = dict(skipped=True, reason="small_cap", wait_polls=0, settled=False)
                            print(f"[skip] epoch={epoch} (capital too small) cap={capital:.6f} minBet={min_bet_bnb:.6f}")
                            send_round_snapshot(
                                prefix=f"⛔ <b>Skip</b> epoch={epoch} (малый капитал)",
                                extra_lines=[f"capital={capital:.6f} BNB, minBet={min_bet_bnb:.6f} BNB"]
                            )
                            notify_ens_used(p_base_before_ens, p_xgb, p_rf, p_arf, p_nn, p_final, False, meta.mode)
                            continue


                        override_reasons = []
                        if bootstrap_phase:
                            override_reasons.append("bootstrap меньше чем 500")
                        if not has_recent:
                            override_reasons.append("idle≥1h")
                        if meta.mode != "ACTIVE":
                            override_reasons.append("meta≠ACTIVE")

                        # === АДАПТИВНАЯ δ: увеличиваем при низком винрейте ===
                        base_delta = float(DELTA_PROTECT)  # 0.06 из глобальных настроек
                        
                        if not bootstrap_phase and has_recent and meta.mode == "ACTIVE":
                            # Проверяем винрейт последних 100 сделок
                            recent_wr = rolling_winrate_laplace(CSV_PATH, n=100, max_epoch_exclusive=epoch)
                            
                            if recent_wr is not None:
                                if recent_wr < 0.50:
                                    delta_eff = base_delta * 1.3  # +50% при плохом винрейте
                                    print(f"[delta] BOOSTED: {delta_eff:.3f} (wr={recent_wr:.2%} < 52%)")
                                elif recent_wr < 0.52:
                                    delta_eff = base_delta * 1.15  # +20% при посредственном
                                    print(f"[delta] slightly increased: {delta_eff:.3f} (wr={recent_wr:.2%} < 54%)")
                                else:
                                    delta_eff = base_delta
                                    print(f"[delta] normal: {delta_eff:.3f} (wr={recent_wr:.2%})")
                            else:
                                delta_eff = base_delta * 1.3  # если нет данных - консервативнее
                                print(f"[delta] conservative (no wr data): {delta_eff:.3f}")
                        else:
                            delta_eff = 0.0

                        # устойчивее проверка override-условий (частичные совпадения)
                        critical_flags = ("bootstrap меньше чем 500", "idle≥1h")
                        has_critical_override = bool(
                            override_reasons and any(flag in r for r in override_reasons for flag in critical_flags)
                        )

                        # ============================================================
                        # === EV-GATE: OR-логика с тремя путями прохождения фильтра ===
                        # ============================================================
                        
                        # Инициализируем переменные для всех веток
                        # Инициализируем переменные для всех веток
                        q70_loss = 0.0
                        q50_loss = 0.0
                        margin_vs_market = 0.0
                        p_thr = 0.0
                        p_thr_ev = 0.0
                        pass_reason = "NA"

                        if has_critical_override:
                            # === РЕЖИМ OVERRIDE: фиксированный порог ===
                            p_thr = 0.51
                            p_thr_src = f"fixed(0.51; {' & '.join(override_reasons)})"
                            
                            # Простая проверка для override
                            if p_side < p_thr:
                                bets[epoch] = dict(
                                    skipped=True, reason="ev_gate_override",
                                    p_side=p_side, p_thr=p_thr, p_thr_src=p_thr_src,
                                    r_hat=r_hat, r_hat_source=r_hat_source,
                                    gb_hat=gb_hat, gc_hat=gc_hat, stake=stake,
                                    delta15=(float(delta15) if (USE_STRESS_R15 and 'delta15' in locals()) else None),
                                    wait_polls=0, settled=False,
                                    p_meta_raw=float(p_meta_raw) if 'p_meta_raw' in locals() else float('nan'),
                                    calib_src=str(calib_src) if 'calib_src' in locals() else "calib[off]"
                                )
                                
                                side_txt = "UP" if bet_up else "DOWN"
                                print(f"[skip] epoch={epoch} side={side_txt} override p={p_side:.4f} < p_thr={p_thr:.4f} [{p_thr_src}]")
                                
                                # === Telegram notification ===
                                try:
                                    notify_ev_decision(
                                        title="⛔ Skip (override)",
                                        epoch=epoch,
                                        side_txt=side_txt,
                                        p_side=p_side,
                                        p_thr=p_thr,
                                        p_thr_src=p_thr_src,
                                        r_hat=r_hat,
                                        gb_hat=gb_hat,
                                        gc_hat=gc_hat,
                                        stake=stake,
                                        delta15=(delta15 if (USE_STRESS_R15 and 'delta15' in locals()) else None),
                                        extra_lines=[],
                                        delta_eff=0.0,
                                    )
                                except Exception as e:
                                    print(f"[tg ] notify skip failed: {e}")
                                
                                # === Snapshot ===
                                send_round_snapshot(
                                    prefix=f"⛔ <b>Skip</b> epoch={epoch} (override)",
                                    extra_lines=[
                                        f"side=<b>{side_txt}</b>, p={p_side:.4f} < p_thr={p_thr:.4f}",
                                        f"Причина: {' & '.join(override_reasons)}"
                                    ]
                                )
                                
                                notify_ens_used(p_base_before_ens, p_xgb, p_rf, p_arf, p_nn, p_final, False, meta.mode)
                                continue

                        else:
                            # === РЕЖИМ ПОЛНОЦЕННОЙ ПРОВЕРКИ: OR-логика ===
                            
                            # Вычисляем ВСЕ метрики заранее
                            # Вычисляем ВСЕ метрики заранее (с защитой от None)
                            try:
                                q70_loss = loss_margin_q(csv_path=CSV_PATH, max_epoch_exclusive=epoch, q=0.70)
                                q70_loss = _as_float(q70_loss, 0.0)
                            except Exception:
                                q70_loss = 0.0
                            
                            try:
                                q50_loss = loss_margin_q(csv_path=CSV_PATH, max_epoch_exclusive=epoch, q=0.50)
                                q50_loss = _as_float(q50_loss, 0.0)
                            except Exception:
                                q50_loss = 0.0
                            
                            try:
                                margin_vs_market = _as_float(p_side, 0.5) - (1.0 / max(1e-9, _as_float(r_hat, 1.9)))
                            except Exception:
                                margin_vs_market = 0.0
                            
                            try:
                                p_thr_ev = p_thr_from_ev(
                                    r_hat=_as_float(r_hat, 1.9),
                                    stake=max(1e-9, _as_float(stake, 0.001)),
                                    gb_hat=_as_float(gb_hat, 0.0),
                                    gc_hat=_as_float(gc_hat, 0.0),
                                    delta=_as_float(delta_eff, 0.0)
                                )
                                p_thr_ev = _as_float(p_thr_ev, 0.51)
                            except Exception as e:
                                print(f"[ev_gate] p_thr_from_ev failed: {e}")
                                p_thr_ev = 0.51
                            
                            # p_thr для совместимости: p_thr + δ = p_thr_ev
                            p_thr = float(max(0.0, p_thr_ev - float(delta_eff)))
                            
                            # Три пути прохождения фильтра (OR-логика):
                            pass_ev_strong = (p_side >= (p_thr + delta_eff))
                            pass_margin_q70 = (margin_vs_market >= q70_loss) and (p_side >= (p_thr + 0.5 * delta_eff))
                            pass_margin_q50 = (margin_vs_market >= q50_loss) and (p_side >= (p_thr + delta_eff))
                            
                            # Определяем источник для логирования
                            if pass_ev_strong:
                                pass_reason = "EV_strong"
                            elif pass_margin_q70:
                                pass_reason = "margin_q70"
                            elif pass_margin_q50:
                                pass_reason = "margin_q50"
                            else:
                                pass_reason = "FAIL"
                            
                            # bnbusdrt6.py  (логирование pass_reason — безопасно к незаданным значениям)
                            _safe_q70 = float(q70_loss) if isinstance(q70_loss, (int, float)) and math.isfinite(float(q70_loss)) else 0.0
                            _safe_q50 = float(q50_loss) if isinstance(q50_loss, (int, float)) and math.isfinite(float(q50_loss)) else 0.0
                            _safe_margin = float(margin_vs_market) if isinstance(margin_vs_market, (int, float)) and math.isfinite(float(margin_vs_market)) else 0.0
                            _safe_r = float(r_hat) if isinstance(r_hat, (int, float)) and math.isfinite(float(r_hat)) else 1.0
                            _safe_reason = pass_reason if isinstance(pass_reason, str) else "NA"

                            p_thr_src = (f"EV|δ+gas; q70={_safe_q70:.4f}, q50={_safe_q50:.4f}; "
                                        f"margin={_safe_margin:+.4f}; r̂={_safe_r:.3f}; "
                                        f"pass={_safe_reason}")

                            
                            # === ПРОВЕРКА: хотя бы одно условие должно пройти ===
                            if not (pass_ev_strong or pass_margin_q70 or pass_margin_q50):
                                # SKIP: все фильтры провалены
                                
                                bets[epoch] = dict(
                                    skipped=True, reason="ev_gate",
                                    p_side=p_side, p_thr=p_thr, p_thr_src=p_thr_src,
                                    r_hat=r_hat, r_hat_source=r_hat_source,
                                    gb_hat=gb_hat, gc_hat=gc_hat, stake=stake,
                                    delta15=(float(delta15) if (USE_STRESS_R15 and 'delta15' in locals()) else None),
                                    wait_polls=0, settled=False,
                                    p_meta_raw=float(p_meta_raw) if 'p_meta_raw' in locals() else float('nan'),
                                    calib_src=str(calib_src) if 'calib_src' in locals() else "calib[off]"
                                )
                                
                                side_txt = "UP" if bet_up else "DOWN"
                                kelly_txt = ("—" if (kelly_half is None or not (isinstance(kelly_half, (int, float)) and math.isfinite(kelly_half)))
                                             else f"{kelly_half:.3f}")
                                
                                # === Telegram notification ===
                                # === Telegram notification ===
                                try:
                                    _safe_margin = float(margin_vs_market) if isinstance(margin_vs_market, (int, float)) and math.isfinite(float(margin_vs_market)) else 0.0
                                    _safe_q70 = float(q70_loss) if isinstance(q70_loss, (int, float)) and math.isfinite(float(q70_loss)) else 0.0
                                    _safe_q50 = float(q50_loss) if isinstance(q50_loss, (int, float)) and math.isfinite(float(q50_loss)) else 0.0

                                    notify_ev_decision(
                                        title="⛔ Skip by EV gate",
                                        epoch=epoch,
                                        side_txt=side_txt,
                                        p_side=p_side,
                                        p_thr=p_thr,
                                        p_thr_src=p_thr_src,
                                        r_hat=r_hat,
                                        gb_hat=gb_hat,
                                        gc_hat=gc_hat,
                                        stake=stake,
                                        delta15=(delta15 if (USE_STRESS_R15 and 'delta15' in locals()) else None),
                                        extra_lines=[
                                            f"Kelly/2:   {kelly_txt}",
                                            f"❌ EV strong: p={_as_float(p_side,0.0):.4f} < p_thr+δ={(_as_float(p_thr)+_as_float(delta_eff,0.0)):.4f}",
                                            f"❌ Margin q70: margin={_safe_margin:+.4f} < q70={_safe_q70:.4f}",
                                            f"❌ Margin q50: margin={_safe_margin:+.4f} < q50={_safe_q50:.4f}",
                                        ],
                                        delta_eff=delta_eff,
                                    )
                                except Exception as e:
                                    print(f"[tg ] notify skip failed: {e}")

                                
                                # === Console log ===
                                print(f"[skip] epoch={epoch} side={side_txt} EV-gate ALL FAIL | "
                                    f"p={_as_float(p_side,0.0):.4f} p_thr+δ={(_as_float(p_thr)+_as_float(delta_eff,0.0)):.4f} | "
                                    f"margin={float(margin_vs_market):+.4f} q70={float(q70_loss):.4f} q50={float(q50_loss):.4f} | "
                                    f"r̂={float(r_hat):.3f} S={float(stake):.6f}")

                                
                                # === Snapshot ===
                                # === Snapshot ===
                                _delta15_str = None
                                if USE_STRESS_R15 and (('delta15' in locals()) or ('delta15' in globals())):
                                    _d15 = _as_float(delta15, float("nan"))
                                    if math.isfinite(_d15):
                                        _delta15_str = f"Δ15_med={(_d15/1e18 if _d15 > 1e6 else _d15):.4f} BNB"


                                _safe_margin = float(margin_vs_market) if isinstance(margin_vs_market, (int, float)) and math.isfinite(float(margin_vs_market)) else 0.0
                                _safe_q70 = float(q70_loss) if isinstance(q70_loss, (int, float)) and math.isfinite(float(q70_loss)) else 0.0
                                _safe_q50 = float(q50_loss) if isinstance(q50_loss, (int, float)) and math.isfinite(float(q50_loss)) else 0.0
                                extra = [
                                    f"p_ctx={p_side:.4f} vs p_thr_ev={(p_thr + delta_eff):.4f} [{p_thr_src}]",
                                    f"❌ EV strong: {p_side:.4f} < {(p_thr + delta_eff):.4f}",
                                    f"❌ Margin q70: {_safe_margin:+.4f} < {_safe_q70:.4f}",
                                    f"❌ Margin q50: {_safe_margin:+.4f} < {_safe_q50:.4f}",
                                    f"r̂={r_hat:.3f} [{r_hat_source}], S={stake:.6f}, gb̂={gb_hat:.8f}, gĉ={gc_hat:.8f}",
                                    _delta15_str,
                                    f"gas_bet≈{gas_bet_bnb_cur:.8f} BNB",
                                    (f"порог-оверрайды: {', '.join(override_reasons)}" if override_reasons else None),
                                    f"Kelly/8={kelly_txt}",
                                ]
                                
                                extra = [x for x in extra if x is not None]
                                
                                send_round_snapshot(
                                    prefix=f"⛔ <b>Skip</b> epoch={epoch} (EV-gate)",
                                    extra_lines=extra
                                )
                                
                                notify_ens_used(p_base_before_ens, p_xgb, p_rf, p_arf, p_nn, p_final, False, meta.mode)
                                
                                # === Теневой лог ===
                                try:
                                    gas_gwei_for_log = float(gas_gwei_now) if 'gas_gwei_now' in locals() else float(get_gas_price_wei(w3)) / 1e9
                                    append_shadow_row(CSV_SHADOW_PATH, {
                                        "settled_ts": "",
                                        "epoch": epoch,
                                        "side": side_txt,
                                        "p_up": float(p_side if side_txt == "UP" else 1.0 - p_side),
                                        "p_thr_used": float(p_thr),
                                        "p_thr_src": str(p_thr_src),
                                        "edge_at_entry": float("nan"),
                                        "stake": float(stake),
                                        "gas_bet_bnb": float(gas_bet_bnb_cur),
                                        "gas_claim_bnb": float(gas_claim_bnb_cur),
                                        "gas_price_bet_gwei": gas_gwei_for_log,
                                        "gas_price_claim_gwei": gas_gwei_for_log,
                                        "outcome": "", "pnl": "",
                                        "capital_before": float(capital),
                                        "capital_after": float(capital),
                                        "lock_ts": "", "close_ts": "",
                                        "lock_price": "", "close_price": "",
                                        "payout_ratio": "", "up_won": ""
                                    })
                                except Exception as e:
                                    print(f"[shadow] append failed: {e}")
                                
                                continue  # ← переход к следующему epoch

                        # ============================================================
                        # === ЕСЛИ ДОШЛИ СЮДА: ФИЛЬТР ПРОЙДЕН, РАЗМЕЩАЕМ СТАВКУ ===
                        # ============================================================
                        
                        # --- считаем запас на входе
                        # --- считаем запас на входе
                        edge_at_entry = float(p_side - (p_thr + delta_eff))

                        _safe_margin = float(margin_vs_market) if isinstance(margin_vs_market, (int, float)) and math.isfinite(float(margin_vs_market)) else 0.0
                        print(f"[bet ] epoch={epoch} side={side} "
                            f"p_side={_as_float(p_side,0.0):.3f} ≥ p_thr+δ={(_as_float(p_thr)+_as_float(delta_eff,0.0)):.3f} "
                            f"edge@entry={edge_at_entry:+.4f} "
                            f"Kelly/2={kelly_txt if 'kelly_txt' in locals() else '—'} r̂={r_hat:.3f} S={stake:.6f}")



                        # --- сохранить контекст ставки
                        phi_wf = np.array([
                            wf_phi_dict.get("phi_wf0", 0.0),
                            wf_phi_dict.get("phi_wf1", 0.0),
                            wf_phi_dict.get("phi_wf2", 0.0),
                            wf_phi_dict.get("phi_wf3", 0.0),
                        ], dtype=float)

                        # быстрый рефреш REST-логики (важно при перезапусках/ресторах)
                        rest.update_from_stats(stats, cfg=rest_cfg)
                        # перед постановкой ставки — проверяем REST
                        if not rest.can_trade_now():
                            print(f"[rest] ⏸ до {rest.rest_until_utc}")
                            continue  # пропускаем этот эпизод/итерацию



                        # --- считаем запас на входе
                        # при ENTER:
                        edge_at_entry = float(p_side - (p_thr + delta_eff))   # здесь p_thr+δ == p_thr_ev
                        # безопасные значения для логирования
                        _safe_margin = float(margin_vs_market) if isinstance(margin_vs_market, (int, float)) and math.isfinite(margin_vs_market) else 0.0
                        _safe_q90 = float(q90_loss) if (('q90_loss' in locals()) or ('q90_loss' in globals())) and isinstance(q90_loss, (int, float)) and math.isfinite(q90_loss) else 0.0
                        # предпочитаем p_thr_ev, если есть; иначе p_thr; иначе 0.0
                        _safe_pthr = None
                        if 'p_thr_ev' in locals() or 'p_thr_ev' in globals():
                            _safe_pthr = p_thr_ev
                        elif 'p_thr' in locals() or 'p_thr' in globals():
                            _safe_pthr = p_thr
                        _safe_pthr = float(_safe_pthr) if isinstance(_safe_pthr, (int, float)) and math.isfinite(_safe_pthr) else 0.0

                        _safe_p = float(p_side) if isinstance(p_side, (int, float)) and math.isfinite(p_side) else 0.0
                        _side_str = (str(side).upper() if ('side' in locals() or 'side' in globals()) else "NA")

                        _safe_p = _as_float(p_side, 0.0)
                        _safe_q70 = _as_float(q70_loss, 0.0)
                        _safe_margin = _as_float(margin_vs_market, 0.0)
                        print(f"[enter] side={_side_str} "
                            f"p_ctx={_safe_p:.4f} ≥ p_thr+δ={_as_float(p_thr) + _as_float(delta_eff, 0.0):.4f} | "
                            f"margin={_safe_margin:+.4f} ≥ q70={_safe_q70:.4f}")
                        bets[epoch] = dict(
                            placed=True, settled=False, wait_polls=0,
                            time=now, t_lock=rd.lock_ts,
                            bet_up=bool(bet_up),
                            p_up=_as_float(P_up),
                            p_side=_as_float(p_side),
                            p_thr=_as_float(p_thr),
                            p_thr_src=str(p_thr_src),
                            r_hat=_as_float(r_hat, 1.0),
                            r_hat_source=str(r_hat_source),
                            gb_hat=_as_float(gb_hat, 0.0),
                            gc_hat=_as_float(gc_hat, 0.0),
                            kelly_half=(None if bootstrap_phase else _as_float(kelly_half, 0.0)),
                            stake=_as_float(stake, 0.0),
                            p_meta_raw=_as_float(locals().get("p_meta_raw")),
                            p_meta2_raw=_as_float(locals().get("p_meta2_raw")),   # ← NEW
                            p_blend=_as_float(locals().get("p_blend")),           # ← NEW
                            blend_w=_as_float(locals().get("blend_w")),           # ← NEW
                            calib_src=str(locals().get("calib_src", "calib[off]")),
                            gas_price_bet_wei=gas_price_wei, gas_bet_bnb=gas_bet_bnb_cur,
                            edge_at_entry=edge_at_entry,
                            delta15=(float(delta15) if (USE_STRESS_R15 and 'delta15' in locals()) else None),
                            phi=phi, phi_wf=phi_wf,
                            ens=dict(
                                x=x_ml.tolist(),
                                p_xgb=(None if p_xgb is None else float(p_xgb)),
                                p_rf=(None if p_rf is None else float(p_rf)),
                                p_arf=(None if p_arf is None else float(p_arf)),
                                p_nn=(None if p_nn is None else float(p_nn)),
                                p_final=float(p_final) if p_final is not None else None,
                                used=bool(ens_used),
                                meta_mode=meta.mode,
                                p_base=float(p_base_before_ens),
                                reg_ctx=reg_ctx,
                            ),
                        )

                        side = "UP" if bet_up else "DOWN"
                        kelly_txt = ("—" if bootstrap_phase else f"{kelly_half:.3f}")

                        print(f"[bet ] epoch={epoch} side={side} "
                            f"... p_side={_as_float(p_side,0.0):.3f} ≥ p_thr+δ={(_as_float(p_thr)+_as_float(delta_eff,0.0)):.3f} ..."
                            f"edge@entry={_as_float(edge_at_entry,0.0):+.4f} "
                            f"Kelly/2={kelly_txt} r̂={_as_float(r_hat,1.0):.3f} S={_as_float(stake,0.0):.6f} gas_bet={_as_float(gas_bet_bnb_cur,0.0):.8f}BNB "
                            f"(lock in {int(_as_float(rd.lock_ts,0)-_as_float(now,0))}s)")

                        _delta15_str = None
                        if USE_STRESS_R15 and 'delta15' in locals():
                            _d15 = _as_float(delta15, float("nan"))
                            if math.isfinite(_d15):
                                _delta15_str = f"Δ15_med={(_d15/1e18 if _d15 > 1e6 else _d15):.4f} BNB"

                        extra = [
                            f"side=<b>{side}</b>, p={_as_float(p_side,0.0):.4f} ≥ p_thr+δ={(_as_float(p_thr)+_as_float(delta_eff,0.0)):.4f} [{p_thr_src}]",
                            f"edge@entry={edge_at_entry:+.4f}",
                            f"S={stake:.6f} BNB (кэп {MAX_STAKE_FRACTION*100:.0f}% от капитала), gas_bet≈{gas_bet_bnb_cur:.8f} BNB",
                            (_delta15_str if USE_STRESS_R15 else None),
                        ]
                        if override_reasons:
                            extra.append(f"p_thr override: {', '.join(override_reasons)}")
                        if not bootstrap_phase:
                            extra.append(f"Kelly/2={kelly_half:.3f}")
                        else:
                            extra.append("Stake=1% bootstrap")

                        extra = [x for x in extra if x is not None]
                        send_round_snapshot(prefix=f"✅ <b>Bet</b> epoch={epoch}", extra_lines=extra)


                        notify_ens_used(p_base_before_ens, p_xgb, p_rf, p_arf, p_nn, p_final, ens_used, meta.mode)

                    elif now >= rd.lock_ts:
                        bets[epoch] = dict(skipped=True, reason="late", wait_polls=0, settled=False)
                        print(f"[late] epoch={epoch} missed betting window")
                        send_round_snapshot(
                            prefix=f"⛔ <b>Skip</b> epoch={epoch} (late)",
                            extra_lines=["Причина: окно размещения закрыто."]
                        )
                        notify_ens_used(None, None, None, None, None, None, False, meta.mode)

                # --- обработка закрытия/сеттла
                b = bets.get(epoch)
                if not b or b.get("settled"):
                    continue

                # пропущенные считаем финализированными сразу после close_ts
                if b.get("skipped") and now > rd.close_ts:
                    b["settled"] = True
                    b["outcome"] = "skipped"
                    send_round_snapshot(
                        prefix=f"ℹ️ <b>Round</b> epoch={epoch} finalized (skip).",
                        extra_lines=["Раунд завершён, пропуск подтверждён."]
                    )
                    continue

                # обычный сеттл — когда oracleCalled
                # обычный сеттл — когда oracleCalled
                if rd.oracle_called:
                    # обновим историю «поздних денег» по только что закрытому раунду
                    # обновим историю «поздних денег» по только что закрытому раунду
                    try:
                        # ✳️ гарантируем снимок на самом lock_ts (после рестартов/лагов его могло не быть)
                        pool.observe(epoch, rd.lock_ts, rd.bull_amount, rd.bear_amount)
                        pool.finalize_epoch(epoch, rd.lock_ts)
                    except Exception:
                        pass
                    # Фолбэк для газа на случай сбоя RPC — НЕ прерываем сеттл из-за газа
                    fallback_wei = 0
                    try:
                        fallback_wei = int(float(b.get("gas_price_bet_wei", 0)) or 0)
                    except Exception:
                        fallback_wei = 0
                    if fallback_wei <= 0:
                        try:
                            # если ранее где-то уже брали газ — используем его
                            fallback_wei = int(gas_price_wei)  # может не существовать — ок
                        except Exception:
                            fallback_wei = 0

                    try:
                        gas_price_claim_wei = get_gas_price_wei(w3)
                        rpc_fail_streak = 0
                    except Exception as e:
                        print(f"[rpc ] gas_price (claim) failed: {e}")
                        rpc_fail_streak += 1
                        # используем фолбэк вместо прерывания
                        gas_price_claim_wei = fallback_wei if fallback_wei > 0 else 3_000_000_000
                        if rpc_fail_streak >= RPC_FAIL_MAX:
                            try:
                                w3 = connect_web3()
                                c = get_prediction_contract(w3)
                                rpc_fail_streak = 0
                                print("[rpc ] reconnected")
                            except Exception as ee:
                                print(f"[rpc ] reconnect failed: {ee}")
                        # ВАЖНО: без continue — идём дальше к отправке исхода
                    # ... далее формирование outcome/pnl и send_round_snapshot(...)


                    outcome = None
                    pnl = 0.0
                    gas_claim_bnb = 0.0

                    bet_up = bool(b.get("bet_up", False))
                    stake = _as_float(b.get("stake", 0.0), 0.0)
                    gas_bet_bnb = _as_float(b.get("gas_bet_bnb", 0.0), 0.0)

                    lock_price = _as_float(getattr(rd, "lock_price", None), 0.0)
                    close_price = _as_float(getattr(rd, "close_price", None), 0.0)

                    up_won = close_price > lock_price
                    down_won = close_price < lock_price
                    draw = close_price == lock_price

                    if NN_USE and logreg is not None and (not draw) and ("phi" in b):
                        try:
                            logreg.update(np.array(b["phi"], dtype=float), 1 if up_won else 0)
                            logreg.save()
                        except Exception:
                            pass

                    capital_before = capital
                    
                    # Вычисляем новый капитал БЕЗ изменения переменной capital
                    if draw:
                        gas_claim_bnb = _as_float(GAS_USED_CLAIM * gas_price_claim_wei / 1e18, 0.0)
                        new_capital = capital - (gas_bet_bnb + gas_claim_bnb)
                        pnl = -(gas_bet_bnb + gas_claim_bnb)
                        outcome = "draw"
                    else:
                        ratio = _as_float(getattr(rd, "payout_ratio", None), 1.9)
                        if ratio <= 1.0:
                            ratio = 1.9
                        
                        if (bet_up and up_won) or ((not bet_up) and down_won):
                            profit = stake * (ratio - 1.0)
                            gas_claim_bnb = _as_float(GAS_USED_CLAIM * gas_price_claim_wei / 1e18, 0.0)
                            new_capital = capital + profit - (gas_bet_bnb + gas_claim_bnb)
                            pnl = profit - (gas_bet_bnb + gas_claim_bnb)
                            outcome = "win"
                        else:
                            new_capital = capital - stake - gas_bet_bnb
                            pnl = -stake - gas_bet_bnb
                            outcome = "loss"

                    b.update(dict(
                        settled=True, outcome=outcome, pnl=pnl,
                        gas_price_claim_wei=gas_price_claim_wei, gas_claim_bnb=gas_claim_bnb,
                        capital_after=new_capital, payout_ratio=rd.payout_ratio
                    ))
                    side = "UP" if bet_up else "DOWN"
                    print(f"[setl] epoch={epoch} side={side} outcome={outcome} pnl={pnl:+.6f} "
                          f"cap={capital:.6f} ratio={rd.payout_ratio if rd.payout_ratio else float('nan'):.3f} up_won={up_won}")

                    # Walk-Forward обновление (заморозка до 500 сделок)
                    try:
                        if WF_USE and ("phi_wf" in b) and (not draw):
                            n_trades = _settled_trades_count(CSV_PATH)  # уже есть в файле
                            if n_trades >= MIN_TRADES_FOR_DELTA:        # MIN_TRADES_FOR_DELTA = 500
                                y_up = 1.0 if up_won else 0.0
                                wf.update(np.array(b["phi_wf"], dtype=float), y_up)
                                wf.save()
                                print(f"[wf  ] updated weights = {wf.w}")
                            else:
                                # до 500 сделок WF не трогаем
                                pass
                    except Exception:
                        pass


                    # Ансамбль: апдейт экспертов и меты
                    try:
                        ens_info = b.get("ens") or {}   # ← если None → {}
                        x_ml = np.array(ens_info.get("x", []), dtype=float)
                        p_xgb = ens_info.get("p_xgb", None)
                        p_rf  = ens_info.get("p_rf", None)
                        p_arf = ens_info.get("p_arf", None)
                        p_nn  = ens_info.get("p_nn", None)
                        p_fin = ens_info.get("p_final", None)
                        p_base = ens_info.get("p_base", None)
                        used_flag = bool(ens_info.get("used", False))

                        reg_ctx = (ens_info.get("reg_ctx", {}) or {})
                        reg_ctx = dict(reg_ctx, epoch=int(epoch))  # ← добавили идентификатор раунда

                        if not draw:
                            y_up_int = 1 if up_won else 0

                            if xgb_exp.enabled and x_ml.size > 0:
                                xgb_exp.record_result(x_ml, y_up=y_up_int, used_in_live=used_flag, p_pred=p_xgb, reg_ctx=reg_ctx)
                                xgb_exp.maybe_train()
                            if rf_exp.enabled and x_ml.size > 0:
                                rf_exp.record_result( x_ml, y_up=y_up_int, used_in_live=used_flag, p_pred=p_rf,  reg_ctx=reg_ctx)
                                rf_exp.maybe_train()
                            if arf_exp.enabled and x_ml.size > 0:
                                arf_exp.record_result(x_ml, y_up=y_up_int, used_in_live=used_flag, p_pred=p_arf, reg_ctx=reg_ctx)
                            if nn_exp.enabled and x_ml.size > 0:
                                nn_exp.record_result( x_ml, y_up=y_up_int, used_in_live=used_flag, p_pred=p_nn,  reg_ctx=reg_ctx)
                                nn_exp.maybe_train()

                            meta.record_result(
                                p_xgb, p_rf, p_arf, p_nn, p_base=p_base,
                                y_up=y_up_int, used_in_live=used_flag, p_final_used=p_fin,
                                reg_ctx=reg_ctx
                            )

                            # NEW: обновляем вторую МЕТА (LambdaMART)
                            try:
                                LM = globals().get("_LM_META")
                                if LM:
                                    LM.record_result(p_xgb, p_rf, p_arf, p_nn, p_base=p_base, y_up=y_up_int, reg_ctx=reg_ctx, used_in_live=used_flag)
                            except Exception:
                                pass

                            # NEW: обновляем калибровщики и блендер на исходе
                            # NEW: обновляем калибровщики и блендер на исходе
                            try:
                                CM1 = globals().get("_CALIB_MGR")
                                CM2 = globals().get("_CALIB_MGR2")
                                BL  = globals().get("_BLENDER")

                                if CM1 and ("p_meta_raw" in b) and _is_finite_num(b["p_meta_raw"]):
                                    CM1.update(_as_float(b["p_meta_raw"]), int(y_up_int), int(time.time()))

                                if CM2 and ("p_meta2_raw" in b) and _is_finite_num(b["p_meta2_raw"]):
                                    CM2.update(_as_float(b["p_meta2_raw"]), int(y_up_int), int(time.time()))

                                if BL:
                                    p1c = _as_float(CM1.transform(_as_float(b.get("p_meta_raw"))) if (CM1 and _is_finite_num(b.get("p_meta_raw"))) else b.get("p_meta_raw"))
                                    p2c = _as_float(CM2.transform(_as_float(b.get("p_meta2_raw"))) if (CM2 and _is_finite_num(b.get("p_meta2_raw"))) else p1c)
                                    BL.record(int(y_up_int), float(p1c), float(p2c))
                            except Exception:
                                pass





                            s_x = xgb_exp.status(); s_r = rf_exp.status(); s_a = arf_exp.status(); s_n = nn_exp.status(); s_m = meta.status()
                            tg_send("🧠 ENS updated:\n" +
                                    _status_line("XGB", s_x) + "\n" +
                                    _status_line("RF ", s_r) + "\n" +
                                    _status_line("ARF", s_a) + "\n" +
                                    _status_line("NN ",  s_n) + "\n" +
                                    _status_line("META", s_m))
                    except Exception as _e:
                        print(f"[ens ] update error: {_e}")

                    
                    # после вычисления outcome/pnl
                    try:
                        side_txt = "UP" if bool(b.get("bet_up", False)) else "DOWN"
                        status = "🏆 WIN" if up_won or down_won else "—"
                        p_side = float(b.get("p_side", 0.0))
                        p_thr  = float(b.get("p_thr",  0.0))
                        p_thr_src = b.get("p_thr_src", "—")
                        r_hat  = float(b.get("r_hat",  0.0))
                        gb_hat = float(b.get("gb_hat", 0.0))
                        gc_hat = float(b.get("gc_hat", 0.0))
                        stake  = float(b.get("stake",  0.0))
                        delta15 = b.get("delta15", None)

                        notify_ev_decision(
                            title=f"{status} Settle",
                            epoch=epoch,
                            side_txt=side_txt,
                            p_side=p_side,
                            p_thr=p_thr,
                            p_thr_src=p_thr_src,
                            r_hat=r_hat,
                            gb_hat=gb_hat,
                            gc_hat=gc_hat,
                            stake=stake,
                            delta15=(delta15 if delta15 is not None else None),
                            extra_lines=[
                                f"outcome:   {'win' if (up_won or down_won) else 'draw'}",
                                f"pnl:       {pnl:+.6f}  (BNB)",
                            ],  # ← запятая обязательна
                            delta_eff=delta_eff,
                        )
                    except Exception as e:
                        print(f"[tg ] notify settle failed: {e}")

                    # CSV-лог
                    row = {
                        "settled_ts": int(time.time()),
                        "epoch": epoch,
                        "side": side,
                        "p_up":           _as_float(b.get("p_up")),
                        "p_meta_raw":     _as_float(b.get("p_meta_raw")),     # ← ДОБАВИЛИ
                        "p_meta2_raw":    _as_float(b.get("p_meta2_raw")),    # ← NEW
                        "p_blend":        _as_float(b.get("p_blend")),        # ← NEW
                        "blend_w":        _as_float(b.get("blend_w")),        # ← NEW
                        "calib_src":      str(b.get("calib_src", "")),
                        "p_thr_used":     _as_float(b.get("p_thr")),
                        "p_thr_src":      str(b.get("p_thr_src", "")),
                        "edge_at_entry":  _as_float(b.get("edge_at_entry")),
                        "stake":          _as_float(stake, 0.0),
                        "gas_bet_bnb":    _as_float(gas_bet_bnb, 0.0),
                        "gas_claim_bnb":  _as_float(gas_claim_bnb, 0.0),
                        "gas_price_bet_gwei":   (_as_float(b.get("gas_price_bet_wei"), 0.0) / 1e9),
                        "gas_price_claim_gwei": (_as_float(gas_price_claim_wei, 0.0) / 1e9),
                        "outcome": outcome,
                        "pnl": pnl,
                        "capital_before": capital_before,
                        "capital_after": capital,
                        "lock_ts": rd.lock_ts,
                        "close_ts": rd.close_ts,
                        "lock_price": rd.lock_price,
                        "close_price": rd.close_price,
                        "payout_ratio": rd.payout_ratio if rd.payout_ratio else float('nan'),
                        "up_won": bool(up_won),
                        "r_hat_used": float(b.get("r_hat", float('nan'))),              # ← НОВОЕ
                        "r_hat_source": str(b.get("r_hat_source", "")),                 # ← НОВОЕ
                        "r_hat_error_pct": float('nan')                                 # ← заполним ниже
                    }
                    
                    # Рассчитываем ошибку оценки r̂
                    try:
                        if rd.payout_ratio and b.get("r_hat"):
                            r_actual = float(rd.payout_ratio)
                            r_pred = float(b["r_hat"])
                            if math.isfinite(r_actual) and math.isfinite(r_pred) and r_actual > 0:
                                error_pct = abs(r_actual - r_pred) / r_actual * 100.0
                                row["r_hat_error_pct"] = float(error_pct)
                    except Exception:
                        pass

                    append_trade_row(CSV_PATH, row)
                    # дублируем капитал в отдельный state-файл (на случай удаления CSV)
                    try:
                        capital_state.save(capital, ts=int(time.time()))
                    except Exception as e:
                        print(f"[warn] capital_state save failed: {e}")

                    # --- Performance monitor: прокинем сделку
                    try:
                        if perf is not None:
                            perf.on_trade_settled(row)
                    except Exception as e:
                        print(f"[perf] on_trade_settled failed: {e}")


                    # обновляем статистику и состояние REST

                    stats.reload()

                    # --- Performance monitor: почасовой отчёт (без дублей)
                    try:
                        if perf is not None:
                            perf.maybe_hourly_report(now_ts=int(time.time()), tg_send_fn=tg_send)
                    except Exception as e:
                        print(f"[perf] hourly report failed: {e}")

                    # --- Ежедневный отчёт (отправляется не чаще 1р/сутки, около полуночи по UTC) ---
                    # --- Ежедневный отчёт (отправляется не чаще 1р/сутки, около полуночи по UTC) ---
                    # Запуск слушателя /report (гарантированно один раз)
                    try:
                        if (TG_TOKEN and TG_CHAT_ID):
                            global _REPORT_THREAD
                            if (_REPORT_THREAD is None) or (not _REPORT_THREAD.is_alive()):
                                _REPORT_THREAD = start_report_listener(SESSION, TG_TOKEN, TG_CHAT_ID, CSV_PATH, tg_send)
                                print("[tg] /report listener started")
                        else:
                            print("[tg] /report listener disabled (no TG_TOKEN/TG_CHAT_ID)")
                    except Exception as e:
                        print(f"[tg] /report listener failed: {e}")



                    # Периодическая отправка суточного отчёта (как было)
                    try:
                        try_send_daily(CSV_PATH, tg_send)  # троттлинг/время внутри
                    except Exception as e:
                        print(f"[warn] daily_report failed: {e}")


                    # Ежедневная проекция в 00:05 Europe/Berlin, не чаще 1 раза в день
                    try:
                        tm = datetime.now(PROJ_TZ)  # требуется: from datetime import datetime
                        if tm.hour == 0 and tm.minute < 10 and _proj_mark_once(PROJ_STATE_PATH, tm.strftime("%Y-%m-%d")):
                            txt = try_send_projection(
                                CSV_PATH,
                                tg_send,
                                horizons=(30, 90, 365),
                                start_cap=capital,
                                threshold=35.0,
                                lookback_days=30,
                                n_paths=8000,
                                block_len=3,
                            )
                            if not txt:
                                print("[proj] send failed (tg_send returned falsy)")
                    except Exception as e:
                        print(f"[warn] projection failed: {e}")





                    rest.notify_trade_executed()
                    rest.update_from_stats(stats, cfg=rest_cfg)
                    rest.save("rest_state.json")



                    emo = "🟢" if outcome == "win" else ("🟡" if outcome == "draw" else "🔴")
                    send_round_snapshot(
                        prefix=f"{emo} <b>Settled</b> epoch={epoch}",
                        extra_lines=[
                            f"side=<b>{side}</b>, outcome=<b>{outcome}</b>, pnl={pnl:+.6f} BNB",
                            f"cap_after={capital:.6f} BNB, ratio={rd.payout_ratio if rd.payout_ratio else float('nan'):.3f}"
                        ]
                    )
                    stats_dict = compute_stats_from_csv(CSV_PATH)
                    print_stats(stats_dict)
                    continue

                # форс-сеттл по таймауту oracleCalled
                if now > rd.close_ts:
                    b["wait_polls"] = int(b.get("wait_polls", 0)) + 1
                    wp = b["wait_polls"]
                    if wp % WAIT_PRINT_EVERY == 0:
                        print(f"[wait] epoch={epoch} waiting oracleCalled (closed, not finalized) polls={wp}/{MAX_WAIT_POLLS}")

                    if wp >= MAX_WAIT_POLLS and b.get("placed"):
                        lock_price_est = rd.lock_price
                        if (not math.isfinite(lock_price_est)) or lock_price_est == 0:
                            lock_price_est = nearest_close_price_ms(SYMBOL, (rd.lock_ts - 1) * 1000)

                        close_price_est = nearest_close_price_ms(SYMBOL, rd.close_ts * 1000)
                        if lock_price_est is None or close_price_est is None:
                            print(f"[wait] epoch={epoch} forced settle postponed (no market price).")
                            continue

                        # форс-сеттл после таймаута ожидания oracleCalled
                        fallback_wei = 0
                        try:
                            fallback_wei = int(float(b.get("gas_price_bet_wei", 0)) or 0)
                        except Exception:
                            fallback_wei = 0

                        try:
                            gas_price_claim_wei = get_gas_price_wei(w3)
                            rpc_fail_streak = 0
                        except Exception as e:
                            print(f"[rpc ] gas_price (claim) failed: {e}")
                            rpc_fail_streak += 1
                            # берём фолбэк и НЕ прерываем обработку
                            gas_price_claim_wei = fallback_wei if fallback_wei > 0 else 3_000_000_000
                            if rpc_fail_streak >= RPC_FAIL_MAX:
                                try:
                                    w3 = connect_web3()
                                    c = get_prediction_contract(w3)
                                    rpc_fail_streak = 0
                                    print("[rpc ] reconnected")
                                except Exception as ee:
                                    print(f"[rpc ] reconnect failed: {ee}")
                        # ВАЖНО: без continue — идём дальше к отправке исхода
                        # ... далее формирование outcome/pnl и send_round_snapshot("Forced settle", ...)


                        bet_up = bool(b.get("bet_up", False))
                        stake = float(b.get("stake", 0.0))
                        gas_bet_bnb = float(b.get("gas_bet_bnb", 0.0))

                        up_won = close_price_est > lock_price_est
                        down_won = close_price_est < lock_price_est
                        draw = close_price_est == lock_price_est

                        if NN_USE and logreg is not None and (not draw) and ("phi" in b):
                            try:
                                logreg.update(np.array(b["phi"], dtype=float), 1 if up_won else 0)
                                logreg.save()
                            except Exception:
                                pass

                        ratio_imp = implied_payout_ratio(bet_up, rd, TREASURY_FEE)
                        ratio_use = ratio_imp if (ratio_imp is not None and math.isfinite(ratio_imp) and ratio_imp > 1.0) else 1.90

                        capital_before = capital
                        gas_claim_bnb = GAS_USED_CLAIM * gas_price_claim_wei / 1e18

                        # Вычисляем новый капитал БЕЗ изменения переменной capital
                        if draw:
                            new_capital = capital - (gas_bet_bnb + gas_claim_bnb)
                            pnl = -(gas_bet_bnb + gas_claim_bnb)
                            outcome = "draw"
                        else:
                            if (bet_up and up_won) or ((not bet_up) and down_won):
                                profit = stake * (ratio_use - 1.0)
                                new_capital = capital + profit - (gas_bet_bnb + gas_claim_bnb)
                                pnl = profit - (gas_bet_bnb + gas_claim_bnb)
                                outcome = "win"
                            else:
                                new_capital = capital - stake - gas_bet_bnb
                                pnl = -stake - gas_bet_bnb
                                outcome = "loss"

                        b.update(dict(
                            settled=True, outcome=outcome, pnl=pnl,
                            gas_price_claim_wei=gas_price_claim_wei, gas_claim_bnb=gas_claim_bnb,
                            capital_after=new_capital, payout_ratio=ratio_use, forced=True
                        ))
                        side = "UP" if bet_up else "DOWN"
                        print(f"[FORC] epoch={epoch} side={side} outcome={outcome} pnl={pnl:+.6f} cap={capital:.6f} "
                              f"ratio_imp={ratio_use:.3f} lock_est={lock_price_est:.4f} close_est={close_price_est:.4f}")

                        try:
                            if WF_USE and ("phi_wf" in b) and (not draw):
                                y_up = 1.0 if up_won else 0.0
                                wf.update(np.array(b["phi_wf"], dtype=float), y_up)
                                wf.save()
                                print(f"[wf  ] updated weights = {wf.w}")
                        except Exception:
                            pass

                        try:
                            ens_info = b.get("ens") or {}   # ← та же защита
                            x_ml = np.array(ens_info.get("x", []), dtype=float)
                            p_xgb = ens_info.get("p_xgb", None)
                            p_rf  = ens_info.get("p_rf", None)
                            p_arf = ens_info.get("p_arf", None)
                            p_nn  = ens_info.get("p_nn", None)
                            p_fin = ens_info.get("p_final", None)
                            p_base = ens_info.get("p_base", None)
                            used_flag = bool(ens_info.get("used", False))

                            reg_ctx = (ens_info.get("reg_ctx", {}) or {})
                            reg_ctx = dict(reg_ctx, epoch=int(epoch))  # ← добавили идентификатор раунда

                            if not draw and x_ml.size > 0:
                                y_up_int = 1 if up_won else 0

                                if xgb_exp.enabled:
                                    xgb_exp.record_result(x_ml, y_up=y_up_int, used_in_live=used_flag, p_pred=p_xgb, reg_ctx=reg_ctx)
                                    xgb_exp.maybe_train()
                                if rf_exp.enabled:
                                    rf_exp.record_result( x_ml, y_up=y_up_int, used_in_live=used_flag, p_pred=p_rf,  reg_ctx=reg_ctx)
                                    rf_exp.maybe_train()
                                if arf_exp.enabled:
                                    arf_exp.record_result(x_ml, y_up=y_up_int, used_in_live=used_flag, p_pred=p_arf, reg_ctx=reg_ctx)
                                if nn_exp.enabled:
                                    nn_exp.record_result( x_ml, y_up=y_up_int, used_in_live=used_flag, p_pred=p_nn,  reg_ctx=reg_ctx)
                                    nn_exp.maybe_train()

                                meta.record_result(
                                    p_xgb, p_rf, p_arf, p_nn, p_base=p_base,
                                    y_up=y_up_int, used_in_live=used_flag, p_final_used=p_fin,
                                    reg_ctx=reg_ctx
                                )

                                # NEW: обновляем вторую МЕТА (LambdaMART)
                                try:
                                    LM = globals().get("_LM_META")
                                    if LM:
                                        LM.record_result(p_xgb, p_rf, p_arf, p_nn, p_base=p_base, y_up=y_up_int, reg_ctx=reg_ctx, used_in_live=used_flag)
                                except Exception:
                                    pass

                                # NEW: обновляем калибровщики и блендер на исходе
                                try:
                                    CM1 = globals().get("_CALIB_MGR")
                                    CM2 = globals().get("_CALIB_MGR2")
                                    BL  = globals().get("_BLENDER")
                                    if CM1 and "p_meta_raw" in b and b["p_meta_raw"] == b["p_meta_raw"]:
                                        CM1.update(float(b["p_meta_raw"]), int(y_up_int), int(time.time()))
                                    if CM2 and "p_meta2_raw" in b and b["p_meta2_raw"] == b["p_meta2_raw"]:
                                        CM2.update(float(b["p_meta2_raw"]), int(y_up_int), int(time.time()))
                                    if BL and "p_meta_raw" in b:
                                        p1c = (CM1.transform(float(b["p_meta_raw"])) if CM1 else float(b["p_meta_raw"]))
                                        p2c = (CM2.transform(float(b["p_meta2_raw"])) if (CM2 and "p_meta2_raw" in b and b["p_meta2_raw"] == b["p_meta2_raw"]) else p1c)
                                        BL.record(int(y_up_int), float(p1c), float(p2c))
                                except Exception:
                                    pass



                        except Exception as _e:
                            print(f"[ens ] forced-settle update error: {_e}")

                        row = {
                            "settled_ts": int(time.time()),
                            "epoch": epoch,
                            "side": side,
                            "p_up": float(b.get("p_up", float('nan'))),
                            "p_meta_raw": float(b.get("p_meta_raw", float('nan'))),   # ← ДОБАВИЛИ
                            "p_meta2_raw": float(b.get("p_meta2_raw", float('nan'))),  # ← NEW
                            "p_blend":     float(b.get("p_blend",     float('nan'))),  # ← NEW
                            "blend_w":     float(b.get("blend_w",     float('nan'))),  # ← NEW
                            "calib_src":  str(b.get("calib_src", "")), 
                            "p_thr_used": float(b.get("p_thr", float('nan'))),
                            "p_thr_src":  str(b.get("p_thr_src", "")),
                            "edge_at_entry": float(b.get("edge_at_entry", float('nan'))),                            
                            "stake": stake,
                            "gas_bet_bnb": gas_bet_bnb,
                            "gas_claim_bnb": gas_claim_bnb,
                            "gas_price_bet_gwei": float(b.get("gas_price_bet_wei", 0.0)) / 1e9,
                            "gas_price_claim_gwei": gas_price_claim_wei / 1e9,
                            "outcome": outcome,
                            "pnl": pnl,
                            "capital_before": capital_before,
                            "capital_after": capital,
                            "lock_ts": rd.lock_ts,
                            "close_ts": rd.close_ts,
                            "lock_price": lock_price_est if rd.lock_price == 0 else rd.lock_price,
                            "close_price": close_price_est,
                            "payout_ratio": ratio_use,
                            "payout_ratio": ratio_use,
                            "up_won": bool(up_won),
                            "r_hat_used": float(b.get("r_hat", float('nan'))),
                            "r_hat_source": str(b.get("r_hat_source", "")),
                            "r_hat_error_pct": float('nan')
                        }
                        
                        # Ошибка для forced settlement
                        try:
                            if ratio_use and b.get("r_hat"):
                                r_actual = float(ratio_use)
                                r_pred = float(b["r_hat"])
                                if math.isfinite(r_actual) and math.isfinite(r_pred) and r_actual > 0:
                                    error_pct = abs(r_actual - r_pred) / r_actual * 100.0
                                    row["r_hat_error_pct"] = float(error_pct)
                        except Exception:
                            pass

                        append_trade_row(CSV_PATH, row)
                        # дублируем капитал в отдельный state-файл (на случай удаления CSV)
                        try:
                            capital_state.save(capital, ts=int(time.time()))
                        except Exception as e:
                            print(f"[warn] capital_state save failed: {e}")
                        # --- Performance monitor: прокинем сделку
                        try:
                            if perf is not None:
                                perf.on_trade_settled(row)
                        except Exception as e:
                            print(f"[perf] on_trade_settled failed: {e}")


                        # обновляем статистику и состояние REST

                        stats.reload()

                        # --- Performance monitor: почасовой отчёт (без дублей)
                        try:
                            if perf is not None:
                                perf.maybe_hourly_report(now_ts=int(time.time()), tg_send_fn=tg_send)
                        except Exception as e:
                            print(f"[perf] hourly report failed: {e}")

                        # --- Ежедневный отчёт (отправляется не чаще 1р/сутки, около полуночи по UTC) ---
                        try:
                            try_send_daily(CSV_PATH, tg_send)  # троттлинг/время внутри
                        except Exception as e:
                            print(f"[warn] daily_report failed: {e}")

                        try:
                            # шлём реже: например, по понедельникам в 00:05 UTC
                            tm = datetime.now(timezone.utc) 
                            if tm.weekday() == 0 and tm.hour == 0 and tm.minute < 10:
                                try_send_projection(CSV_PATH, tg_send,
                                                    horizons=(30, 90, 365),
                                                    start_cap=capital,  # если знаешь текущий
                                                    threshold=35.0,
                                                    lookback_days=30,
                                                    n_paths=8000,
                                                    block_len=3)
                        except Exception as e:
                            print(f"[warn] projection failed: {e}")

                        rest.notify_trade_executed()
                        rest.update_from_stats(stats, cfg=rest_cfg)
                        rest.save("rest_state.json")



                        send_round_snapshot(
                            prefix=f"⚠️ <b>Forced settle</b> epoch={epoch}",
                            extra_lines=[
                                f"Причина: oracleCalled нет {wp} проверок подряд.",
                                f"side=<b>{side}</b>, outcome=<b>{outcome}</b>, pnl={pnl:+.6f} BNB",
                                f"lock≈{(lock_price_est if rd.lock_price == 0 else rd.lock_price):.4f}, close≈{close_price_est:.4f}, ratio≈{ratio_use:.3f}"
                            ]
                        )

                        stats_dict = compute_stats_from_csv(CSV_PATH)
                        print_stats(stats_dict)
                        continue

            _prune_bets(bets, keep_settled_last=500, keep_other_last=200)

            # мягкий сборщик мусора раз в ~10 минут (снижает фрагментацию)
            try:
                import gc
                if (now - _last_gc) >= 600:
                    gc.collect()
                    _last_gc = now
            except Exception:
                pass

            time.sleep(1.0)


        except KeyboardInterrupt:
            print("\n[stop] Ctrl+C")  # не дергаем сеть здесь, просто выходим
            break
        except Exception as e:
            print(f"[warn] {type(e).__name__}: {e}")
            time.sleep(2.0)


def _normalize_existing_csvs():
    """Аккуратно привести оба файла к нужной схеме"""
    for _p in (CSV_PATH, CSV_SHADOW_PATH):
        if os.path.exists(_p):
            try:
                # ✅ Убрали dtype="string" - читаем с автоопределением типов
                raw = pd.read_csv(_p, encoding="utf-8-sig", keep_default_na=True)
                
                # ✅ Сразу заменяем все варианты NA на np.nan
                raw = raw.fillna(np.nan)
                
                # Дополнительная зачистка строковых представлений
                for col in raw.select_dtypes(include=["object"]).columns:
                    raw[col] = raw[col].replace({
                        "<NA>": np.nan, "NaN": np.nan, "nan": np.nan, 
                        "None": np.nan, "": np.nan
                    })
                
                # Мягко приводим типы к целевой схеме
                _df = _coerce_csv_dtypes(raw)
                
                # Финальная зачистка перед сохранением
                _df = _df.fillna(np.nan)
                
                _df.to_csv(_p, index=False, encoding="utf-8-sig")
            except Exception as e:
                print(f"[warn] CSV normalize failed for {_p}: {e!r}")



if __name__ == "__main__":
    ensure_csv_header(CSV_PATH)
    _normalize_existing_csvs()

    upgrade_csv_schema_if_needed(CSV_PATH)
    upgrade_csv_schema_if_needed(CSV_SHADOW_PATH)

    try:
        main_loop()
    except KeyboardInterrupt:
        print("⚠️ Bot stopped (KeyboardInterrupt).")
        try:
            tg_send("⚠️ Bot stopped (KeyboardInterrupt).", html=False)
        except Exception:
            pass
    except Exception as e:
        # пишем стек в GGG/errors.log и даём процессу завершиться с кодом ошибки
        log_exception("Fatal error in main()")
        try:
            tg_send("🔴 Bot crashed: см. GGG/errors.log", html=False)
        except Exception:
            pass
        raise
