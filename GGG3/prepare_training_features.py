#!/usr/bin/env python3
"""
ТОЧНОЕ воспроизведение фич из bnbusdrt6.py для обучения на исторических данных

Создает ТЕ ЖЕ 68 фич что использует live бот.
"""
import numpy as np
from typing import Dict, Optional
import pandas as pd


def prepare_features_exact(round_data: Dict, kl_df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    ТОЧНАЯ копия создания фич из bnbusdrt6.py:7747-7871

    Структура:
    - 14 фич: ext_builder.build() (базовые технические)
    - 43 фичи: addon (микроструктура, funding, gas, time)
    - 11 фич: additional (P_up, RSI lags, momentum...)

    Args:
        round_data: Данные раунда с features и pool amounts
        kl_df: OHLCV данные (если доступны)

    Returns:
        np.ndarray: Вектор из 68 фич
    """
    features = round_data.get('features', {})

    # ========== БЛОК 1: EXT_BUILDER.BUILD() - 14 ФИЧ ==========
    # Из bnbusdrt6.py:1574-1589

    # Базовые индикаторы
    m_diff = _safe_float(features.get('M_diff', 0), 0)      # momentum diff
    s_diff = _safe_float(features.get('S_diff', 0), 0)      # VWAP slope diff
    b_diff = _safe_float(features.get('B_diff', 0), 0)      # Bollinger breakout
    r_diff = _safe_float(features.get('R_diff', 0), 0)      # Bollinger reversion

    z_bb = _safe_float(features.get('Zs', 0), 0)            # Bollinger Z-score
    kc_pos = _safe_float(features.get('kc_pos', 0), 0)      # Keltner position
    atr_norm = _safe_float(features.get('atr_norm', 1), 1)  # Normalized ATR

    # RSI normalized
    rsi = _safe_float(features.get('rsi', 50), 50)
    rsi_norm = (rsi - 50.0) / 50.0

    wick_imb = _safe_float(features.get('wick_imb', 0), 0)  # Wick imbalance
    trend_rsi = _safe_float(features.get('trend_rsi', 0), 0)
    kc_w = _safe_float(features.get('kc_w', 0), 0)         # Keltner width

    # Взаимодействия (3 фичи)
    interaction_1 = m_diff * atr_norm
    interaction_2 = rsi_norm * z_bb
    interaction_3 = b_diff * kc_w

    ext_builder_features = np.array([
        m_diff, s_diff, b_diff, r_diff,
        z_bb, kc_pos, atr_norm, rsi_norm,
        wick_imb, trend_rsi, kc_w,
        interaction_1, interaction_2, interaction_3
    ], dtype=np.float64)

    # ========== БЛОК 2: ADDON - 43 ФИЧИ ==========
    # Из bnbusdrt6.py:7747-7758

    # Микроструктура (7)
    rel_spread = _safe_float(features.get('rel_spread', 0), 0)
    book_imb = _safe_float(features.get('book_imb', 0), 0)
    microprice_delta = _safe_float(features.get('microprice_delta', 0), 0)
    ofi_5s = _safe_float(features.get('ofi_5s', 0), 0)
    ofi_15s = _safe_float(features.get('ofi_15s', 0), 0)
    ofi_30s = _safe_float(features.get('ofi_30s', 0), 0)
    ob_slope = _safe_float(features.get('ob_slope', 0), 0)

    # Funding & futures (5)
    funding_sign = _safe_float(features.get('funding_sign', 0), 0)
    funding_timeleft = _safe_float(features.get('funding_timeleft', 0), 0)
    dOI_1m = _safe_float(features.get('dOI_1m', 0), 0)
    dOI_5m = _safe_float(features.get('dOI_5m', 0), 0)
    basis_now = _safe_float(features.get('basis_now', 0), 0)

    # Pool dynamics (4)
    pool_logit = _safe_float(features.get('pool_logit', 0), 0)
    pool_logit_d30 = _safe_float(features.get('pool_logit_d30', 0), 0)
    pool_logit_d60 = _safe_float(features.get('pool_logit_d60', 0), 0)
    late_money_share = _safe_float(features.get('late_money_share', 0), 0)

    # Historical outcomes (2)
    last_k_outcomes_mean = _safe_float(features.get('last_k_outcomes_mean', 0.5), 0.5)
    last_k_payout_median = _safe_float(features.get('last_k_payout_median', 2.0), 2.0)

    # Volatility measures (6)
    bv_over_rv_20 = _safe_float(features.get('bv_over_rv_20', 1), 1)
    bv_over_rv_60 = _safe_float(features.get('bv_over_rv_60', 1), 1)
    rq_norm_20 = _safe_float(features.get('rq_norm_20', 0), 0)
    rq_norm_60 = _safe_float(features.get('rq_norm_60', 0), 0)
    jump20 = _safe_float(features.get('jump20', 0), 0)
    jump60 = _safe_float(features.get('jump60', 0), 0)

    # Market microstructure advanced (4)
    amihud_illq = _safe_float(features.get('amihud_illq', 0), 0)
    kyle_lambda = _safe_float(features.get('kyle_lambda', 0), 0)
    resid_ret_1m = _safe_float(features.get('resid_ret_1m', 0), 0)
    beta_sum = _safe_float(features.get('beta_sum', 0), 0)
    beta_sum_d60 = _safe_float(features.get('beta_sum_d60', 0), 0)

    # Gas (2)
    gas_d1m = _safe_float(features.get('gas_d1m', 0), 0)
    gas_vol5m = _safe_float(features.get('gas_vol5m', 0), 0)

    # Time features (8): tod_sin, tod_cos, EU, US, ASIA, dow_0-6
    tod_sin = _safe_float(features.get('tod_sin', 0), 0)
    tod_cos = _safe_float(features.get('tod_cos', 1), 1)
    EU = _safe_float(features.get('EU', 0), 0)
    US = _safe_float(features.get('US', 0), 0)
    ASIA = _safe_float(features.get('ASIA', 0), 0)
    dow_0 = _safe_float(features.get('dow_0', 0), 0)
    dow_1 = _safe_float(features.get('dow_1', 0), 0)
    dow_2 = _safe_float(features.get('dow_2', 0), 0)
    dow_3 = _safe_float(features.get('dow_3', 0), 0)
    dow_4 = _safe_float(features.get('dow_4', 0), 0)
    dow_5 = _safe_float(features.get('dow_5', 0), 0)
    dow_6 = _safe_float(features.get('dow_6', 0), 0)

    addon_features = np.array([
        rel_spread, book_imb, microprice_delta, ofi_5s, ofi_15s, ofi_30s, ob_slope,
        funding_sign, funding_timeleft, dOI_1m, dOI_5m, basis_now,
        pool_logit, pool_logit_d30, pool_logit_d60, late_money_share,
        last_k_outcomes_mean, last_k_payout_median,
        bv_over_rv_20, bv_over_rv_60, rq_norm_20, rq_norm_60, jump20, jump60,
        amihud_illq, kyle_lambda, resid_ret_1m, beta_sum, beta_sum_d60,
        gas_d1m, gas_vol5m,
        tod_sin, tod_cos, EU, US, ASIA,
        dow_0, dow_1, dow_2, dow_3, dow_4, dow_5, dow_6
    ], dtype=np.float64)

    # ========== БЛОК 3: ADDITIONAL - 11 ФИЧ ==========
    # Из bnbusdrt6.py:7798-7863

    # 1. Базовая вероятность (из другой модели/расчета)
    P_up = _safe_float(features.get('P_up', 0.5), 0.5)

    # 2-3. RSI lags (требуют OHLCV историю)
    rsi_lag1 = _safe_float(features.get('rsi_lag1', rsi), rsi)
    rsi_lag5 = _safe_float(features.get('rsi_lag5', rsi), rsi)

    # 4-5. Momentum на разных таймфреймах
    momentum_5m = _safe_float(features.get('momentum_5m', 0), 0)
    momentum_15m = _safe_float(features.get('momentum_15m', 0), 0)

    # 6. Нормализованная позиция относительно экстремумов
    normalized_position = _safe_float(features.get('normalized_position', 0.5), 0.5)

    # 7. Волатильность RSI
    rsi_volatility = _safe_float(features.get('rsi_volatility', 0), 0)

    # 8. Коэффициент вариации доходности
    returns_cv = _safe_float(features.get('returns_cv', 0), 0)

    # 9-11. Market features (из MarketFeaturesCalculator)
    trend_sign = _safe_float(features.get('trend_sign', 0), 0)
    trend_abs = _safe_float(features.get('trend_abs', 0), 0)
    vol_ratio = _safe_float(features.get('vol_ratio', 1), 1)

    additional_features = np.array([
        P_up,
        rsi_lag1, rsi_lag5,
        momentum_5m, momentum_15m,
        normalized_position,
        rsi_volatility,
        returns_cv,
        trend_sign, trend_abs, vol_ratio
    ], dtype=np.float64)

    # ========== ФИНАЛЬНАЯ КОНКАТЕНАЦИЯ ==========
    x_final = np.concatenate([
        ext_builder_features,  # 14
        addon_features,        # 43
        additional_features    # 11
    ], axis=0)

    # Защита от NaN/Inf
    x_final = np.nan_to_num(x_final, nan=0.0, posinf=5.0, neginf=-5.0)
    x_final = np.clip(x_final, -10.0, 10.0)

    return x_final  # 68 фич


def _safe_float(val, default=0.0):
    """Безопасное преобразование в float"""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return float(default)
        return float(val)
    except (ValueError, TypeError):
        return float(default)


if __name__ == "__main__":
    # Тест
    test_round = {
        'features': {
            'rsi': 55.0,
            'momentum_5m': 0.002,
            'pool_logit': 0.15
        },
        'bull_amount': 100.5,
        'bear_amount': 85.3
    }

    x = prepare_features_exact(test_round)
    print(f"Размерность фич: {x.shape[0]}")
    print(f"Первые 10 фич: {x[:10]}")
    print(f"Последние 10 фич: {x[-10:]}")
