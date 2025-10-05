
# -*- coding: utf-8 -*-
"""
delta_daily.py — суточный подбор δ для EV-гейта по последним 100 закрытым раундам.

Идея:
- Берём trades_prediction.csv, сортируем по settled_ts, хвост из N=100 завершённых сделок (win/loss/draw).
- Для каждой сделки считаем p_thr по точной формуле из вашего бота:
    denom = r_hat - (gc_hat / stake)
    p_thr  = (1 + (gb_hat / stake)) / denom
  где r_hat берём как фактический payout_ratio из CSV.
- p_side = p_up (если side=UP) или 1 - p_up (если side=DOWN).
- Для сетки δ (например, 0.000..0.100 шагом 0.005) считаем доход:
    sum(pnl_i) только по тем строкам, где p_side_i >= p_thr_i + δ.
- Выбираем δ*, при котором суммарный PnL максимальный (BNB).
  При равенстве PnL предпочитаем:
    1) больший охват (больше сделок),
    2) меньший δ (более консервативный шаг).
- Сохраняем состояние в JSON (atomic) и возвращаем δ на текущие сутки (UTC 00:00..23:59).

Формат delta_state.json:
{
  "_magic": "MLSTATE_v1",
  "_version": 1,
  "...": "...",
  "valid_for_utc_date": "YYYY-MM-DD",
  "computed_at_utc": 1699999999,
  "delta": 0.035,
  "avg_p_thr": 0.5123,
  "best_pnl_bnb": 0.123456,
  "selected_n": 42,
  "sample_size": 100,
  "grid": "0.000..0.100 step=0.005"
}
"""
from __future__ import annotations

import os, math, time, json, typing as T
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from state_safety import atomic_save_json, safe_load_json

class DeltaDaily:
    def __init__(self,
                 csv_path: str = "trades_prediction.csv",
                 state_path: str = "delta_state.json",
                 n_last: int = 100,
                 grid_start: float = 0.0,
                 grid_stop: float  = 0.10,
                 grid_step: float  = 0.005,
                 csv_shadow_path: str | None = "trades_shadow.csv",
                 window_hours: int = 24,
                 opt_mode: str = "grid_pnl"):
        self.csv_path   = csv_path
        self.csv_shadow_path = csv_shadow_path
        self.state_path = state_path
        self.n_last     = int(n_last)
        self.grid_start = float(grid_start)
        self.grid_stop  = float(grid_stop)
        self.grid_step  = float(grid_step)
        self.window_hours = int(window_hours)
        self.opt_mode = str(opt_mode or "grid_pnl").lower()
        self.state: T.Dict[str, T.Any] = {}


    # --- утилиты ---
    @staticmethod
    def _today_utc_str(ts: int | None = None) -> str:
        if ts is None:
            ts = int(time.time())
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _grid_to_str(a: float, b: float, h: float) -> str:
        return f"{a:.3f}..{b:.3f} step={h:.3f}"

    @staticmethod
    def _p_thr_row(row: pd.Series) -> float | None:
        try:
            stake = float(row["stake"])
            r_hat = float(row["payout_ratio"])
            gb_hat = float(row.get("gas_bet_bnb", 0.0))
            gc_hat = float(row.get("gas_claim_bnb", 0.0))
            if not math.isfinite(stake) or stake <= 0: return None
            if not math.isfinite(r_hat) or r_hat <= 1.0: return None
            denom = (r_hat - (gc_hat / stake))
            if not math.isfinite(denom) or denom <= 1e-12: return None
            p_thr = (1.0 + (gb_hat / stake)) / denom
            if not math.isfinite(p_thr) or p_thr <= 0 or p_thr >= 1: return None
            return float(p_thr)
        except Exception:
            return None

    @staticmethod
    def _pnl_net(row: pd.Series) -> float | None:
        try:
            stake = float(row["stake"])
            gb = float(row.get("gas_bet_bnb", 0.0)) if row.get("gas_bet_bnb") == row.get("gas_bet_bnb") else 0.0
            gc = float(row.get("gas_claim_bnb", 0.0)) if row.get("gas_claim_bnb") == row.get("gas_claim_bnb") else 0.0
            ratio = float(row.get("payout_ratio")) if row.get("payout_ratio") == row.get("payout_ratio") else 1.9
            outc = str(row.get("outcome",""))
            if outc == "draw":
                return -(gb + gc)
            elif outc == "win":
                return stake*(ratio - 1.0) - (gb + gc)
            elif outc == "loss":
                return -stake - gb
            return None
        except Exception:
            return None



    def _find_optimal_p_thr(self, df: pd.DataFrame) -> tuple[float, float, int]:
        """
        Возвращает (p_thr_star, pnl_sum_at_star, selected_n).
        p_thr_star выбирается из множества уникальных p_side (оптимум на «переломах»).
        """
        if df.empty:
            return (0.51, 0.0, 0)
        # кандидаты — уникальные p_side
        cand = sorted(set(pd.to_numeric(df["p_side"], errors="coerce").dropna().values.tolist()))
        best_p, best_pnl, best_n = 0.51, float("-inf"), -1
        for t in cand:
            mask = (df["p_side"] >= t)
            pnl_sum = float(pd.to_numeric(df.loc[mask, "pnl_net"], errors="coerce").fillna(0.0).sum())
            n_sel   = int(mask.sum())
            if (pnl_sum > best_pnl + 1e-12) or (abs(pnl_sum - best_pnl) <= 1e-12 and n_sel > best_n):
                best_p, best_pnl, best_n = float(t), pnl_sum, n_sel
        return best_p, best_pnl, best_n

    def _avg_used_p_thr(self, df: pd.DataFrame) -> float:
        # если есть p_thr_used — берём его (это «как реально решал бот»)
        if "p_thr_used" in df.columns:
            s = pd.to_numeric(df["p_thr_used"], errors="coerce").dropna()
            if len(s) > 0:
                return float(s.mean())
        # иначе — запасной путь: пересчёт EV-порога (как раньше)
        s = pd.to_numeric(df["p_thr"], errors="coerce").dropna()
        return float(s.mean()) if len(s) > 0 else float("nan")
        
    @staticmethod
    def _p_side_row(row: pd.Series) -> float | None:
        try:
            p_up = float(row["p_up"])
            side = str(row["side"]).upper()
            ps = p_up if side == "UP" else (1.0 - p_up)
            if not math.isfinite(ps) or ps <= 0 or ps >= 1: return None
            return float(ps)
        except Exception:
            return None

    def _load_df_tail(self) -> pd.DataFrame:
        dfs = []
        if os.path.exists(self.csv_path):
            dfs.append(pd.read_csv(self.csv_path, encoding="utf-8-sig"))
        if self.csv_shadow_path and os.path.exists(self.csv_shadow_path):
            dfs.append(pd.read_csv(self.csv_shadow_path, encoding="utf-8-sig"))
        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        if df.empty:
            return df
        df = df.dropna(subset=["outcome"])
        df = df[df["outcome"].isin(["win", "loss", "draw"])]

        # сортируем и берём хвост N
        df = df.sort_values("settled_ts").tail(self.n_last).copy()

        # инженерим p_thr и p_side
        df["p_thr"]  = df.apply(self._p_thr_row, axis=1)
        df["p_side"] = df.apply(self._p_side_row, axis=1)

        df = df.dropna(subset=["p_thr", "p_side", "pnl"])
        return df

    def _load_df_window(self, now_ts: int) -> pd.DataFrame:
        """Загрузить сделки за последние self.window_hours часов (по settled_ts)."""
        dfs = []
        if os.path.exists(self.csv_path):
            dfs.append(pd.read_csv(self.csv_path, encoding="utf-8-sig"))
        if self.csv_shadow_path and os.path.exists(self.csv_shadow_path):
            dfs.append(pd.read_csv(self.csv_shadow_path, encoding="utf-8-sig"))
        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        if df.empty:
            return df

        df = df.dropna(subset=["outcome"])
        df = df[df["outcome"].isin(["win", "loss", "draw"])]

        # --- фильтр по времени (UTC unixtime в поле settled_ts)
        horizon = now_ts - self.window_hours * 3600
        st = pd.to_numeric(df.get("settled_ts"), errors="coerce")
        df = df.loc[st >= horizon].copy()

        # если сделок мало (например, < n_last), подстрахуемся хвостом
        if len(df) < self.n_last:
            df_tail = pd.concat(dfs, ignore_index=True).sort_values("settled_ts").tail(self.n_last)
            df = pd.concat([df, df_tail], ignore_index=True).drop_duplicates()

        df = df.sort_values("settled_ts")

        # p_thr и p_side — как и раньше
        # безопасное вычисление p_thr/p_side
        if "p_thr" not in df.columns:
            df["p_thr"]  = df.apply(self._p_thr_row, axis=1)
        if "p_side" not in df.columns:
            df["p_side"] = df.apply(self._p_side_row, axis=1)
        if "pnl" not in df.columns and "pnl_net" in df.columns:
            df["pnl"] = df["pnl_net"]
        df = df.dropna(subset=[c for c in ["p_thr","p_side","pnl"] if c in df.columns])

        return df



    def _grid_search(self, df: pd.DataFrame) -> tuple[float, float, int]:
        # Возвращает (delta, pnl_sum, selected_n)
        if df.empty:
            return (0.03, 0.0, 0)
        a, b, h = self.grid_start, self.grid_stop, self.grid_step
        best_delta, best_pnl, best_n = 0.03, -1e99, -1
        cur = a
        # делаем точную арифметику шагов, чтобы избежать эффекта 0.30000000004
        n_steps = int(round((b - a) / h)) + 1
        for k in range(n_steps):
            delta = a + k * h
            mask = (df["p_side"] >= (df["p_thr"] + delta))
            pnl_sum = float(pd.to_numeric(df.loc[mask, "pnl"], errors="coerce").fillna(0.0).sum())
            n_sel   = int(mask.sum())
            if (pnl_sum > best_pnl + 1e-12) or \
               (abs(pnl_sum - best_pnl) <= 1e-12 and (n_sel > best_n or (n_sel == best_n and delta < best_delta))):
                best_delta, best_pnl, best_n = float(delta), pnl_sum, n_sel
        return best_delta, best_pnl, best_n



    # INSERT BELOW: DR-LCB оптимизатор δ
    def _fit_expectation_model(self, df: pd.DataFrame):
        try:
            from sklearn.linear_model import HuberRegressor
            X = df[["p_side","payout_ratio","stake","gas_bet_bnb","gas_claim_bnb"]].to_numpy(dtype=float)
            y = df["g"].to_numpy(dtype=float)
            mdl = HuberRegressor().fit(X, y)
            def predict(xdf): return mdl.predict(xdf[["p_side","payout_ratio","stake","gas_bet_bnb","gas_claim_bnb"]].to_numpy(dtype=float))
            return predict
        except Exception:
            return lambda xdf: np.full(len(xdf), float(np.nan))


    def _optimize_delta_dr_lcb(self, df: pd.DataFrame, B: int = 1000, q: float = 0.15, ips_shadow: float = 1.0):
        if df.empty: return (0.03, 0.0, 0)
        # g: лог-рост
        def _safe_g(row):
            cb = row.get("capital_before", np.nan)
            ca = row.get("capital_after",  np.nan)
            if np.isfinite(cb) and np.isfinite(ca) and cb>0 and ca>0:
                return float(np.log(ca/cb))
            stake = float(row.get("stake", 0.0) or 0.0)
            pnl   = float(row.get("pnl", 0.0) or 0.0)
            base  = max(stake, 1e-9)
            return float(np.log(1.0 + pnl/base))
        df = df.copy()
        df["g"] = df.apply(_safe_g, axis=1)

        # модель ожидания g_hat(x)
        g_hat = self._fit_expectation_model(df)

        a, b, h = self.grid_start, self.grid_stop, self.grid_step
        best_delta, best_lcb, best_n = 0.03, -1e9, 0
        n_steps = int(round((b - a) / h)) + 1
        rng = np.random.default_rng(42)

        for k in range(n_steps):
            delta = a + k * h
            mask = (df["p_side"] >= (df["p_thr"] + delta))
            sub  = df.loc[mask].copy()
            nsel = int(len(sub))
            if nsel < 10:  # слишком мало — пропускаем
                continue

            gh = g_hat(sub)
            gh = np.where(np.isfinite(gh), gh, sub["g"].to_numpy())  # fallback → чистый IPS (dr=g)
            dr = (1.0 * (sub["g"].to_numpy() - gh)) + gh

            # бутстрап LCB по среднему dr
            means = []

            idxs = np.arange(nsel)
            for _ in range(B):
                bs = rng.choice(idxs, size=nsel, replace=True)
                means.append(float(np.nanmean(dr[bs])))
            lcb = float(np.nanpercentile(means, q*100.0))

            if (lcb > best_lcb + 1e-12) or (abs(lcb - best_lcb) <= 1e-12 and nsel > best_n):
                best_delta, best_lcb, best_n = float(delta), float(lcb), int(nsel)

        return best_delta, best_lcb, best_n


    # --- публичный API ---
    def recompute(self, now_ts: int | None = None) -> T.Dict[str, T.Any]:
        if now_ts is None:
            now_ts = int(time.time())
        today = self._today_utc_str(now_ts)

        # ✳️ новое: берём окно по времени
        df = self._load_df_window(now_ts)
        sample_size = int(len(df))

        # Если данных нет — возвращаем безопасное состояние, без падения
        if df is None or sample_size == 0:
            state = dict(
                computed_at_utc=now_ts,
                window_hours=self.window_hours,
                sample_size=0,
                delta=0.0,               # нейтральный δ на старте
                p_thr_opt=0.51,          # безопасный базовый порог
                avg_p_thr_used=float("nan"),
                pnl_at_opt=0.0,
                selected_n=0,
                valid_for_utc_date=today,
                note="insufficient_data"
            )
            atomic_save_json(self.state_path, state)
            self.state = state
            return state

        # p_side/p_thr уже посчитаны в _load_df_window; добавим pnl_net
        df["pnl_net"] = df.apply(self._pnl_net, axis=1)

        # Если вдруг p_side ещё нет (на всякий случай) — создадим
        if "p_side" not in df.columns:
            df["p_side"] = df.apply(self._p_side_row, axis=1)

        df = df.dropna(subset=["pnl_net", "p_side"])

        # --- выбор δ ---
        if getattr(self, "opt_mode", "p_star").lower() == "dr_lcb":
            # DR-OPE + бутстрап LCB(5%)
            delta, score, nsel = self._optimize_delta_dr_lcb(df, B=1000, q=0.05, ips_shadow=1.0)
            p_thr_opt = float(pd.to_numeric(df["p_thr"], errors="coerce").dropna().mean())

            # ⬇️ ДОБАВКА: avg_used и P&L* на оптимальном δ
            avg_used = self._avg_used_p_thr(df)  # возьмёт p_thr_used, иначе средний p_thr
            delta_f  = float(delta)
            mask_sel = (df["p_side"] >= (df["p_thr"] + delta_f))
            pnl_at_opt = float(pd.to_numeric(df.loc[mask_sel, "pnl"], errors="coerce").fillna(0.0).sum())

            state = dict(
                computed_at_utc=now_ts, window_hours=self.window_hours, sample_size=sample_size,
                delta=float(delta), p_thr_opt=p_thr_opt, lcb5=score, selected_n=int(nsel),
                valid_for_utc_date=today, method="dr_lcb",
                grid=self._grid_to_str(self.grid_start, self.grid_stop, self.grid_step),
                # ⬇️ ДОБАВКА: чтобы не было NaN в логах
                avg_p_thr_used=float(avg_used) if math.isfinite(avg_used) else None,
                pnl_at_opt=pnl_at_opt,
            )



        elif getattr(self, "opt_mode", "p_star").lower() == "grid_pnl":
            # Старая логика по сетке δ: max PnL(δ)
            delta, pnl, nsel = self._grid_search(df)
            p_thr_opt = float(pd.to_numeric(df["p_thr"], errors="coerce").dropna().mean())
            state = dict(
                computed_at_utc=now_ts, window_hours=self.window_hours, sample_size=sample_size,
                delta=float(delta), p_thr_opt=p_thr_opt, pnl_at_opt=pnl, selected_n=int(nsel),
                valid_for_utc_date=today, method="grid_pnl",
                grid=self._grid_to_str(self.grid_start, self.grid_stop, self.grid_step)
            )

        else:
            # РЕЗЕРВ: как было — p_star - avg_used
            p_star, pnl_at_star, n_sel = self._find_optimal_p_thr(df)
            avg_used = self._avg_used_p_thr(df)
            delta = float(p_star - avg_used) if (math.isfinite(p_star) and math.isfinite(avg_used)) else 0.0

            state = dict(
                computed_at_utc=now_ts, window_hours=self.window_hours, sample_size=sample_size,
                delta=delta, p_thr_opt=p_star, avg_p_thr_used=avg_used,
                pnl_at_opt=pnl_at_star, selected_n=n_sel,
                valid_for_utc_date=today, method="p_star_minus_avg_used",
                grid=self._grid_to_str(self.grid_start, self.grid_stop, self.grid_step)
            )

        atomic_save_json(self.state_path, state)
        self.state = state
        return state




    def load_or_recompute_now(self, now_ts: int | None = None) -> T.Optional[T.Dict[str, T.Any]]:
        if now_ts is None:
            now_ts = int(time.time())
        today = self._today_utc_str(now_ts)
        st = safe_load_json(self.state_path)
        if st and st.get("valid_for_utc_date") == today and isinstance(st.get("delta"), (int, float)):
            self.state = st
            return st
        # иначе пересчитаем
        return self.recompute(now_ts)

    def maybe_update_for_date(self, now_ts: int | None = None) -> T.Optional[T.Dict[str, T.Any]]:
        """Если наступили новые сутки (UTC) — пересчитаем δ. Иначе вернём None."""
        if now_ts is None:
            now_ts = int(time.time())
        today = self._today_utc_str(now_ts)
        st = safe_load_json(self.state_path)
        if not st or st.get("valid_for_utc_date") != today:
            return self.recompute(now_ts)
        self.state = st
        return None

    def load_or_recompute_every_hours(self, period_hours: int = 4, now_ts: int | None = None) -> T.Optional[T.Dict[str, T.Any]]:
        if now_ts is None:
            now_ts = int(time.time())
        st = safe_load_json(self.state_path)
        if st and isinstance(st.get("computed_at_utc"), (int, float)):
            # если прошло меньше period_hours — просто вернём текущее состояние
            if now_ts - float(st["computed_at_utc"]) < period_hours * 3600:
                self.state = st
                return st
        # иначе пересчитываем
        return self.recompute(now_ts)

    def maybe_update_every_hours(self, period_hours: int = 4, now_ts: int | None = None) -> T.Optional[T.Dict[str, T.Any]]:
        """Пересчитать δ, если прошло >= period_hours часов с момента последнего пересчёта."""
        if now_ts is None:
            now_ts = int(time.time())
        st = safe_load_json(self.state_path)
        if (not st) or (not isinstance(st.get("computed_at_utc"), (int, float))) or (now_ts - float(st["computed_at_utc"]) >= period_hours * 3600):
            return self.recompute(now_ts)
        self.state = st
        return None
