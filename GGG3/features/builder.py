#!/usr/bin/env python3
"""
Построение 68 фич для Binance Futures на основе multi-timeframe данных

Адаптировано из prepare_training_features.py с учётом:
- 4 таймфрейма вместо 1
- Binance Futures вместо PancakeSwap
- Удалены фичи специфичные для PancakeSwap (pool dynamics, gas)
- Добавлены Futures-специфичные фичи
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

# Технические индикаторы
try:
    import talib as ta
except ImportError:
    # Fallback на ta (python-ta)
    import ta as ta_lib
    ta = None

logger = logging.getLogger(__name__)


class BinanceFeatureBuilder:
    """
    Построение 68 фич для Binance Futures

    Входы: 4 таймфрейма (5m, 15m, 30m, 4h)
    Выход: 68D вектор (совместимость с текущими моделями)

    Распределение фич:
    - 5m (momentum): 14 фич
    - 15m (VWAP/Bollinger): 22 фичи
    - 30m (reversion/trend): 21 фича
    - 4h (ATR/phases): 11 фич
    """

    def __init__(self):
        """Инициализация feature builder"""
        self.feature_count = 68

    def build_extended_features(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_30m: pd.DataFrame,
        df_4h: pd.DataFrame,
        current_time: Optional[pd.Timestamp] = None
    ) -> np.ndarray:
        """
        Рассчитывает 68 фич на основе 4 таймфреймов

        Args:
            df_5m: OHLCV данные 5m (минимум 100 свечей)
            df_15m: OHLCV данные 15m (минимум 100 свечей)
            df_30m: OHLCV данные 30m (минимум 100 свечей)
            df_4h: OHLCV данные 4h (минимум 100 свечей)
            current_time: Текущее время (для time features)

        Returns:
            np.ndarray: Вектор из 68 фич
        """
        features = []

        # Блок 1: 5m momentum features (14 фич)
        momentum_features = self._calc_momentum_features(df_5m)
        features.extend(momentum_features)

        # Блок 2: 15m VWAP/Bollinger features (22 фичи)
        vwap_bollinger_features = self._calc_vwap_bollinger_features(df_15m)
        features.extend(vwap_bollinger_features)

        # Блок 3: 30m reversion/trend features (21 фича)
        reversion_trend_features = self._calc_reversion_trend_features(df_30m)
        features.extend(reversion_trend_features)

        # Блок 4: 4h ATR/phase features (11 фич)
        atr_phase_features = self._calc_atr_phase_features(df_4h, current_time)
        features.extend(atr_phase_features)

        # Конвертация в numpy array
        features_array = np.array(features, dtype=np.float64)

        # Защита от NaN/Inf
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=5.0, neginf=-5.0)
        features_array = np.clip(features_array, -10.0, 10.0)

        # Проверка размерности
        if len(features_array) != self.feature_count:
            logger.warning(f"Expected {self.feature_count} features, got {len(features_array)}")
            # Padding если меньше
            if len(features_array) < self.feature_count:
                features_array = np.pad(
                    features_array,
                    (0, self.feature_count - len(features_array)),
                    mode='constant',
                    constant_values=0.0
                )
            # Truncate если больше
            elif len(features_array) > self.feature_count:
                features_array = features_array[:self.feature_count]

        return features_array

    # ========================================================================
    # БЛОК 1: MOMENTUM FEATURES (5m) - 14 ФИЧ
    # ========================================================================

    def _calc_momentum_features(self, df: pd.DataFrame) -> List[float]:
        """
        Momentum индикаторы на 5m таймфрейме (14 фич)

        Быстрые индикаторы для краткосрочного momentum:
        - RSI (2 фичи: значение + slope)
        - MACD (3 фичи: macd, signal, diff)
        - Stochastic (2 фичи: K, D)
        - Price changes (3 фичи: 1, 3, 5 свечей)
        - Volume indicators (2 фичи)
        - Price-volume correlation (1 фича)
        - ATR change (1 фича)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        features = []

        # 1-2. RSI (14) и его slope
        if ta is not None:
            rsi = ta.RSI(close, timeperiod=14)
        else:
            rsi_indicator = ta_lib.momentum.RSIIndicator(close=pd.Series(close), window=14)
            rsi = rsi_indicator.rsi().values

        rsi_current = self._safe_get(rsi, -1, 50.0)
        rsi_prev_5 = self._safe_get(rsi, -6, rsi_current)
        rsi_slope = (rsi_current - rsi_prev_5) / 5.0 if rsi_prev_5 != 0 else 0.0

        features.append((rsi_current - 50.0) / 50.0)  # Нормализация
        features.append(rsi_slope / 10.0)  # Нормализация slope

        # 3-5. MACD (12, 26, 9)
        if ta is not None:
            macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        else:
            macd_indicator = ta_lib.trend.MACD(close=pd.Series(close))
            macd = macd_indicator.macd().values
            macd_signal = macd_indicator.macd_signal().values
            macd_hist = macd_indicator.macd_diff().values

        macd_current = self._safe_get(macd, -1, 0.0)
        macd_signal_current = self._safe_get(macd_signal, -1, 0.0)
        macd_hist_current = self._safe_get(macd_hist, -1, 0.0)

        # Нормализация относительно цены
        price_current = close[-1] if len(close) > 0 else 1.0
        features.append(macd_current / price_current * 100)
        features.append(macd_signal_current / price_current * 100)
        features.append(macd_hist_current / price_current * 100)

        # 6-7. Stochastic (14, 3, 3)
        if ta is not None:
            slowk, slowd = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        else:
            stoch_indicator = ta_lib.momentum.StochasticOscillator(
                high=pd.Series(high),
                low=pd.Series(low),
                close=pd.Series(close),
                window=14,
                smooth_window=3
            )
            slowk = stoch_indicator.stoch().values
            slowd = stoch_indicator.stoch_signal().values

        stoch_k = self._safe_get(slowk, -1, 50.0)
        stoch_d = self._safe_get(slowd, -1, 50.0)

        features.append((stoch_k - 50.0) / 50.0)
        features.append((stoch_d - 50.0) / 50.0)

        # 8-10. Price returns (1, 3, 5 candles back)
        returns_1 = (close[-1] / close[-2] - 1.0) if len(close) >= 2 else 0.0
        returns_3 = (close[-1] / close[-4] - 1.0) if len(close) >= 4 else 0.0
        returns_5 = (close[-1] / close[-6] - 1.0) if len(close) >= 6 else 0.0

        features.append(returns_1 * 100)  # В процентах
        features.append(returns_3 * 100)
        features.append(returns_5 * 100)

        # 11. Relative volume
        vol_current = volume[-1] if len(volume) > 0 else 1.0
        vol_mean_20 = np.mean(volume[-20:]) if len(volume) >= 20 else vol_current
        vol_ratio = vol_current / vol_mean_20 if vol_mean_20 > 0 else 1.0

        features.append(np.log1p(vol_ratio))

        # 12. Volume trend (последние 5 vs предыдущие 5)
        vol_recent = np.mean(volume[-5:]) if len(volume) >= 5 else vol_current
        vol_prev = np.mean(volume[-10:-5]) if len(volume) >= 10 else vol_recent
        vol_trend = (vol_recent / vol_prev - 1.0) if vol_prev > 0 else 0.0

        features.append(vol_trend)

        # 13. Price-volume correlation (последние 20 свечей)
        if len(close) >= 20 and len(volume) >= 20:
            price_returns = np.diff(close[-20:]) / close[-20:-1]
            vol_changes = np.diff(volume[-20:]) / volume[-20:-1]
            # Защита от деления на ноль
            price_returns = np.nan_to_num(price_returns, 0.0)
            vol_changes = np.nan_to_num(vol_changes, 0.0)
            corr = np.corrcoef(price_returns, vol_changes)[0, 1]
            corr = corr if not np.isnan(corr) else 0.0
        else:
            corr = 0.0

        features.append(corr)

        # 14. ATR change (momentum волатильности)
        if ta is not None:
            atr_14 = ta.ATR(high, low, close, timeperiod=14)
        else:
            atr_indicator = ta_lib.volatility.AverageTrueRange(
                high=pd.Series(high),
                low=pd.Series(low),
                close=pd.Series(close),
                window=14
            )
            atr_14 = atr_indicator.average_true_range().values

        atr_current = self._safe_get(atr_14, -1, price_current * 0.01)
        atr_prev_5 = self._safe_get(atr_14, -6, atr_current)
        atr_change = (atr_current / atr_prev_5 - 1.0) if atr_prev_5 > 0 else 0.0

        features.append(atr_change)

        return features

    # ========================================================================
    # БЛОК 2: VWAP/BOLLINGER FEATURES (15m) - 22 ФИЧИ
    # ========================================================================

    def _calc_vwap_bollinger_features(self, df: pd.DataFrame) -> List[float]:
        """
        VWAP и Bollinger индикаторы на 15m (22 фичи)

        Средне-срочные индикаторы для уровней и волатильности:
        - VWAP (3 фичи: slope, distance, volume profile)
        - Bollinger Bands (5 фич: Z-score, width, breakout, reversion, squeeze)
        - Keltner Channels (4 фичи: position, width, Bollinger vs Keltner)
        - Volume profile (3 фичи)
        - Price extremes (3 фичи: high/low distance)
        - Wick analysis (2 фичи)
        - Candlestick patterns (2 фичи)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        features = []
        price_current = close[-1] if len(close) > 0 else 1.0

        # 1-3. VWAP
        vwap = self._calc_vwap(df)
        vwap_current = self._safe_get(vwap, -1, price_current)
        vwap_prev_5 = self._safe_get(vwap, -6, vwap_current)

        # VWAP slope
        vwap_slope = (vwap_current - vwap_prev_5) / 5.0 / price_current * 100

        # Distance to VWAP
        vwap_distance = (price_current - vwap_current) / price_current * 100

        # Volume-weighted VWAP strength
        recent_vol = np.sum(volume[-5:]) if len(volume) >= 5 else 0
        total_vol = np.sum(volume[-20:]) if len(volume) >= 20 else 1
        vwap_strength = recent_vol / total_vol if total_vol > 0 else 0.5

        features.extend([vwap_slope, vwap_distance, vwap_strength])

        # 4-8. Bollinger Bands (20, 2)
        if ta is not None:
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        else:
            bb_indicator = ta_lib.volatility.BollingerBands(close=pd.Series(close), window=20, window_dev=2)
            bb_upper = bb_indicator.bollinger_hband().values
            bb_middle = bb_indicator.bollinger_mavg().values
            bb_lower = bb_indicator.bollinger_lband().values

        bb_upper_current = self._safe_get(bb_upper, -1, price_current * 1.02)
        bb_middle_current = self._safe_get(bb_middle, -1, price_current)
        bb_lower_current = self._safe_get(bb_lower, -1, price_current * 0.98)

        # Z-score (положение относительно полос)
        bb_width = bb_upper_current - bb_lower_current
        z_bb = (price_current - bb_middle_current) / (bb_width / 2) if bb_width > 0 else 0.0

        # Bollinger width (normalized)
        bb_width_norm = bb_width / bb_middle_current if bb_middle_current > 0 else 0.02

        # Breakout signal (цена за пределами полос)
        bb_breakout = 0.0
        if price_current > bb_upper_current:
            bb_breakout = (price_current - bb_upper_current) / bb_middle_current * 100
        elif price_current < bb_lower_current:
            bb_breakout = (price_current - bb_lower_current) / bb_middle_current * 100

        # Reversion signal (близость к полосам)
        bb_reversion = 0.0
        if abs(z_bb) > 1.5:  # За пределами 1.5 sigma
            bb_reversion = -z_bb  # Противоположный сигнал

        # Squeeze (узкие полосы = низкая волатильность)
        bb_width_ma = np.mean([bb_upper[i] - bb_lower[i] for i in range(-20, 0)]) if len(bb_upper) >= 20 else bb_width
        bb_squeeze = (bb_width - bb_width_ma) / bb_width_ma if bb_width_ma > 0 else 0.0

        features.extend([z_bb, bb_width_norm * 100, bb_breakout, bb_reversion, bb_squeeze])

        # 9-12. Keltner Channels
        kc_upper, kc_middle, kc_lower = self._calc_keltner(df)

        kc_upper_current = self._safe_get(kc_upper, -1, price_current * 1.02)
        kc_middle_current = self._safe_get(kc_middle, -1, price_current)
        kc_lower_current = self._safe_get(kc_lower, -1, price_current * 0.98)

        # Keltner position
        kc_width = kc_upper_current - kc_lower_current
        kc_pos = (price_current - kc_middle_current) / (kc_width / 2) if kc_width > 0 else 0.0

        # Keltner width
        kc_width_norm = kc_width / kc_middle_current if kc_middle_current > 0 else 0.02

        # Bollinger vs Keltner (squeeze indicator)
        bb_kc_ratio = bb_width / kc_width if kc_width > 0 else 1.0

        # Bollinger inside Keltner (strong squeeze)
        bb_in_kc = 1.0 if (bb_upper_current < kc_upper_current and bb_lower_current > kc_lower_current) else 0.0

        features.extend([kc_pos, kc_width_norm * 100, bb_kc_ratio, bb_in_kc])

        # 13-15. Volume profile
        # High volume zones
        vol_high_20 = np.percentile(volume[-20:], 80) if len(volume) >= 20 else volume[-1]
        vol_is_high = 1.0 if volume[-1] > vol_high_20 else 0.0

        # Volume concentration (последние 5 vs последние 20)
        vol_recent_pct = np.sum(volume[-5:]) / np.sum(volume[-20:]) if len(volume) >= 20 and np.sum(volume[-20:]) > 0 else 0.25

        # Price-volume agreement (цена и объём в одном направлении)
        price_dir = 1.0 if close[-1] > close[-2] else -1.0 if close[-1] < close[-2] else 0.0
        vol_dir = 1.0 if volume[-1] > volume[-2] else -1.0 if volume[-1] < volume[-2] else 0.0
        pv_agreement = 1.0 if price_dir * vol_dir > 0 else 0.0

        features.extend([vol_is_high, vol_recent_pct, pv_agreement])

        # 16-18. Price extremes (high/low relative to recent range)
        high_20 = np.max(high[-20:]) if len(high) >= 20 else high[-1]
        low_20 = np.min(low[-20:]) if len(low) >= 20 else low[-1]
        range_20 = high_20 - low_20

        # Position in range
        pos_in_range = (price_current - low_20) / range_20 if range_20 > 0 else 0.5

        # Distance to high
        dist_to_high = (high_20 - price_current) / price_current * 100

        # Distance to low
        dist_to_low = (price_current - low_20) / price_current * 100

        features.extend([pos_in_range, dist_to_high, dist_to_low])

        # 19-20. Wick analysis
        # Upper wick (sell pressure)
        body = abs(close[-1] - df['open'].values[-1])
        upper_wick = high[-1] - max(close[-1], df['open'].values[-1])
        wick_upper_ratio = upper_wick / body if body > 0 else 0.0

        # Lower wick (buy pressure)
        lower_wick = min(close[-1], df['open'].values[-1]) - low[-1]
        wick_lower_ratio = lower_wick / body if body > 0 else 0.0

        # Wick imbalance
        wick_imbalance = (upper_wick - lower_wick) / (upper_wick + lower_wick + 1e-10)

        features.extend([wick_imbalance, wick_upper_ratio - wick_lower_ratio])

        # 21-22. Candlestick patterns
        # Doji (small body, large wicks)
        candle_range = high[-1] - low[-1]
        doji_signal = 1.0 if body < candle_range * 0.1 else 0.0

        # Hammer/Hanging Man (small body, long lower wick)
        hammer_signal = 1.0 if (lower_wick > body * 2 and upper_wick < body * 0.5) else 0.0

        features.extend([doji_signal, hammer_signal])

        return features

    # ========================================================================
    # БЛОК 3: REVERSION/TREND FEATURES (30m) - 21 ФИЧА
    # ========================================================================

    def _calc_reversion_trend_features(self, df: pd.DataFrame) -> List[float]:
        """
        Reversion и trend индикаторы на 30m (21 фича)

        Средне-долгосрочные индикаторы для трендов и разворотов:
        - EMA crossovers (4 фичи: 9/21, 21/55)
        - ADX/DI (4 фичи: trend strength, direction)
        - Reversion signals (4 фичи)
        - Momentum oscillators (3 фичи: Williams %R, CCI, ROC)
        - Price action (3 фичи)
        - Time features (3 фичи: session, volatility cycle)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        features = []
        price_current = close[-1] if len(close) > 0 else 1.0

        # 1-4. EMA crossovers
        ema_9 = self._calc_ema(close, 9)
        ema_21 = self._calc_ema(close, 21)
        ema_55 = self._calc_ema(close, 55)

        ema_9_current = self._safe_get(ema_9, -1, price_current)
        ema_21_current = self._safe_get(ema_21, -1, price_current)
        ema_55_current = self._safe_get(ema_55, -1, price_current)

        # EMA 9/21 crossover
        ema_9_21_diff = (ema_9_current - ema_21_current) / price_current * 100

        # EMA 21/55 crossover
        ema_21_55_diff = (ema_21_current - ema_55_current) / price_current * 100

        # Price position relative to EMAs
        price_above_ema9 = (price_current - ema_9_current) / price_current * 100
        price_above_ema55 = (price_current - ema_55_current) / price_current * 100

        features.extend([ema_9_21_diff, ema_21_55_diff, price_above_ema9, price_above_ema55])

        # 5-8. ADX and Directional Indicators
        if ta is not None:
            adx = ta.ADX(high, low, close, timeperiod=14)
            plus_di = ta.PLUS_DI(high, low, close, timeperiod=14)
            minus_di = ta.MINUS_DI(high, low, close, timeperiod=14)
        else:
            adx_indicator = ta_lib.trend.ADXIndicator(
                high=pd.Series(high),
                low=pd.Series(low),
                close=pd.Series(close),
                window=14
            )
            adx = adx_indicator.adx().values
            plus_di = adx_indicator.adx_pos().values
            minus_di = adx_indicator.adx_neg().values

        adx_current = self._safe_get(adx, -1, 20.0)
        plus_di_current = self._safe_get(plus_di, -1, 20.0)
        minus_di_current = self._safe_get(minus_di, -1, 20.0)

        # ADX strength (normalized)
        adx_norm = adx_current / 50.0  # Normalize to 0-2 range

        # Directional movement
        di_diff = plus_di_current - minus_di_current

        # Trend strength (ADX > 25 = strong trend)
        trend_strong = 1.0 if adx_current > 25 else 0.0

        # Trend direction with strength
        trend_direction = di_diff / 50.0 * adx_norm

        features.extend([adx_norm, di_diff / 50.0, trend_strong, trend_direction])

        # 9-12. Reversion signals
        # Mean reversion from SMA
        sma_20 = np.mean(close[-20:]) if len(close) >= 20 else price_current
        reversion_sma = (price_current - sma_20) / sma_20 * 100

        # Bollinger reversion (from 15m, recompute for 30m)
        bb_std = np.std(close[-20:]) if len(close) >= 20 else price_current * 0.01
        z_score = (price_current - sma_20) / bb_std if bb_std > 0 else 0.0
        reversion_bb = -z_score if abs(z_score) > 2.0 else 0.0  # Signal only at extremes

        # RSI reversion (oversold/overbought)
        if ta is not None:
            rsi_30m = ta.RSI(close, timeperiod=14)
        else:
            rsi_indicator = ta_lib.momentum.RSIIndicator(close=pd.Series(close), window=14)
            rsi_30m = rsi_indicator.rsi().values

        rsi_current = self._safe_get(rsi_30m, -1, 50.0)
        reversion_rsi = 0.0
        if rsi_current < 30:
            reversion_rsi = (30 - rsi_current) / 10.0  # Buy signal
        elif rsi_current > 70:
            reversion_rsi = (70 - rsi_current) / 10.0  # Sell signal

        # Price range reversion (extremes of recent range)
        high_50 = np.max(high[-50:]) if len(high) >= 50 else high[-1]
        low_50 = np.min(low[-50:]) if len(low) >= 50 else low[-1]
        range_50 = high_50 - low_50
        pos_in_range_50 = (price_current - low_50) / range_50 if range_50 > 0 else 0.5
        reversion_range = 0.5 - pos_in_range_50  # Signal at extremes

        features.extend([reversion_sma, reversion_bb, reversion_rsi, reversion_range])

        # 13-15. Momentum oscillators
        # Williams %R
        if ta is not None:
            willr = ta.WILLR(high, low, close, timeperiod=14)
        else:
            willr = ta_lib.momentum.WilliamsRIndicator(
                high=pd.Series(high),
                low=pd.Series(low),
                close=pd.Series(close),
                lbp=14
            ).williams_r().values

        willr_current = self._safe_get(willr, -1, -50.0)
        willr_norm = (willr_current + 50.0) / 50.0  # Normalize to -1 to 1

        # CCI (Commodity Channel Index)
        if ta is not None:
            cci = ta.CCI(high, low, close, timeperiod=20)
        else:
            cci = ta_lib.trend.CCIIndicator(
                high=pd.Series(high),
                low=pd.Series(low),
                close=pd.Series(close),
                window=20
            ).cci().values

        cci_current = self._safe_get(cci, -1, 0.0)
        cci_norm = cci_current / 200.0  # Normalize to roughly -1 to 1

        # ROC (Rate of Change)
        if ta is not None:
            roc = ta.ROC(close, timeperiod=10)
        else:
            roc = ta_lib.momentum.ROCIndicator(close=pd.Series(close), window=10).roc().values

        roc_current = self._safe_get(roc, -1, 0.0)

        features.extend([willr_norm, cci_norm, roc_current / 10.0])

        # 16-18. Price action patterns
        # Higher highs / Lower lows (trend continuation)
        hh = 1.0 if (len(high) >= 3 and high[-1] > high[-2] > high[-3]) else 0.0
        ll = 1.0 if (len(low) >= 3 and low[-1] < low[-2] < low[-3]) else 0.0
        trend_continuation = hh - ll

        # Consolidation (low volatility)
        volatility = np.std(close[-20:]) if len(close) >= 20 else price_current * 0.01
        consolidation = 1.0 if volatility / price_current < 0.01 else 0.0

        # Breakout momentum
        range_10 = np.max(high[-10:]) - np.min(low[-10:]) if len(high) >= 10 else high[-1] - low[-1]
        breakout = (close[-1] - close[-2]) / range_10 if range_10 > 0 else 0.0

        features.extend([trend_continuation, consolidation, breakout])

        # 19-21. Time features (session-based)
        # Note: These would ideally use current_time, but for simplicity we'll use price-based proxies
        # In production, you should pass current_time and calculate actual session times

        # Volatility cycle (based on recent volatility)
        vol_ma_short = np.std(close[-10:]) if len(close) >= 10 else volatility
        vol_ma_long = np.std(close[-50:]) if len(close) >= 50 else volatility
        vol_cycle = (vol_ma_short - vol_ma_long) / vol_ma_long if vol_ma_long > 0 else 0.0

        # Price momentum strength
        momentum = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0.0

        # Volume momentum
        vol_momentum = (np.mean(volume[-5:]) - np.mean(volume[-15:-5])) / np.mean(volume[-15:-5]) if len(volume) >= 15 and np.mean(volume[-15:-5]) > 0 else 0.0

        features.extend([vol_cycle, momentum * 100, vol_momentum])

        return features

    # ========================================================================
    # БЛОК 4: ATR/PHASE FEATURES (4h) - 11 ФИЧ
    # ========================================================================

    def _calc_atr_phase_features(
        self,
        df: pd.DataFrame,
        current_time: Optional[pd.Timestamp] = None
    ) -> List[float]:
        """
        ATR и market phase индикаторы на 4h (11 фич)

        Долгосрочные индикаторы для market regime и phases:
        - ATR (3 фичи: value, normalized, change)
        - Market phases (5 фич: accumulation, markup, distribution, markdown, ranging)
        - Regime context (2 фичи: bull/bear, volatility regime)
        - Time features (1 фича: day of week)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        features = []
        price_current = close[-1] if len(close) > 0 else 1.0

        # 1-3. ATR indicators
        if ta is not None:
            atr_14 = ta.ATR(high, low, close, timeperiod=14)
        else:
            atr_indicator = ta_lib.volatility.AverageTrueRange(
                high=pd.Series(high),
                low=pd.Series(low),
                close=pd.Series(close),
                window=14
            )
            atr_14 = atr_indicator.average_true_range().values

        atr_current = self._safe_get(atr_14, -1, price_current * 0.02)

        # ATR normalized (as % of price)
        atr_norm = atr_current / price_current

        # ATR change (volatility momentum)
        atr_prev_10 = self._safe_get(atr_14, -11, atr_current)
        atr_change = (atr_current - atr_prev_10) / atr_prev_10 if atr_prev_10 > 0 else 0.0

        # ATR percentile (current vs historical)
        atr_percentile = np.sum(atr_14[-50:] < atr_current) / len(atr_14[-50:]) if len(atr_14) >= 50 else 0.5

        features.extend([atr_norm * 100, atr_change, atr_percentile])

        # 4-8. Market phases (0-5)
        # Based on Wyckoff cycle: Accumulation, Markup, Distribution, Markdown, Ranging

        # Phase detection based on price action and volume
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else price_current
        sma_200 = np.mean(close[-200:]) if len(close) >= 200 else price_current

        vol_ma = np.mean(volume[-50:]) if len(volume) >= 50 else volume[-1]

        # Phase scores (normalized 0-1)
        # 1. Accumulation: Low volatility, price below MA, increasing volume
        accumulation = 0.0
        if price_current < sma_50 and atr_norm < 0.02:
            vol_trend = np.mean(volume[-10:]) / vol_ma if vol_ma > 0 else 1.0
            accumulation = min(1.0, vol_trend)

        # 2. Markup: Rising price, above MA, strong momentum
        markup = 0.0
        if price_current > sma_50:
            momentum_50 = (price_current - sma_50) / sma_50
            markup = min(1.0, momentum_50 * 10)  # Scale to 0-1

        # 3. Distribution: High volatility, price above MA, decreasing volume
        distribution = 0.0
        if price_current > sma_50 and atr_norm > 0.025:
            vol_trend = vol_ma / np.mean(volume[-10:]) if np.mean(volume[-10:]) > 0 else 1.0
            distribution = min(1.0, vol_trend)

        # 4. Markdown: Falling price, below MA, increasing volume
        markdown = 0.0
        if price_current < sma_50:
            momentum_50 = (sma_50 - price_current) / sma_50
            markdown = min(1.0, momentum_50 * 10)

        # 5. Ranging: Low volatility, price near MA
        ranging = 0.0
        if abs(price_current - sma_50) / sma_50 < 0.02 and atr_norm < 0.015:
            ranging = 1.0

        features.extend([accumulation, markup, distribution, markdown, ranging])

        # 9-10. Regime context
        # Bull/Bear regime (trend over long period)
        regime_bull = 1.0 if sma_50 > sma_200 else 0.0

        # Volatility regime (high/low vol)
        vol_regime = 1.0 if atr_norm > 0.025 else 0.0

        features.extend([regime_bull, vol_regime])

        # 11. Time feature: Day of week (if available)
        if current_time is not None:
            # Cyclic encoding: day of week (0-6)
            dow = current_time.dayofweek
            dow_sin = np.sin(2 * np.pi * dow / 7.0)
        else:
            # Default: mid-week
            dow_sin = 0.0

        features.append(dow_sin)

        return features

    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    # ========================================================================

    def _calc_vwap(self, df: pd.DataFrame) -> np.ndarray:
        """Расчёт VWAP (Volume Weighted Average Price)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.values

    def _calc_keltner(self, df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0) -> tuple:
        """Расчёт Keltner Channels"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Middle line (EMA)
        middle = self._calc_ema(close, period)

        # ATR для ширины канала
        if ta is not None:
            atr = ta.ATR(high, low, close, timeperiod=period)
        else:
            atr_indicator = ta_lib.volatility.AverageTrueRange(
                high=pd.Series(high),
                low=pd.Series(low),
                close=pd.Series(close),
                window=period
            )
            atr = atr_indicator.average_true_range().values

        upper = middle + (atr * atr_mult)
        lower = middle - (atr * atr_mult)

        return upper, middle, lower

    def _calc_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Расчёт EMA"""
        if ta is not None:
            return ta.EMA(data, timeperiod=period)
        else:
            ema = ta_lib.trend.EMAIndicator(close=pd.Series(data), window=period)
            return ema.ema_indicator().values

    def _safe_get(self, arr: np.ndarray, idx: int, default: float) -> float:
        """Безопасное получение элемента массива"""
        try:
            if arr is None or len(arr) == 0:
                return default
            if abs(idx) > len(arr):
                return default
            val = arr[idx]
            if isinstance(val, (np.ndarray, list)):
                val = val[0] if len(val) > 0 else default
            return float(val) if not np.isnan(val) else default
        except (IndexError, TypeError, ValueError):
            return default


# ============================================================================
# MAIN (для тестирования)
# ============================================================================

if __name__ == '__main__':
    # Создание тестовых данных
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=500, freq='5min')

    # Генерация синтетических OHLCV
    base_price = 50000.0
    price_walk = np.cumsum(np.random.randn(500) * 100) + base_price

    df_5m = pd.DataFrame({
        'timestamp': dates,
        'open': price_walk + np.random.randn(500) * 50,
        'high': price_walk + np.abs(np.random.randn(500)) * 100,
        'low': price_walk - np.abs(np.random.randn(500)) * 100,
        'close': price_walk,
        'volume': np.abs(np.random.randn(500)) * 1000 + 5000
    })

    # Ресемплинг для других таймфреймов
    df_15m = df_5m.set_index('timestamp').resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    df_30m = df_5m.set_index('timestamp').resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    df_4h = df_5m.set_index('timestamp').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    # Создание feature builder
    builder = BinanceFeatureBuilder()

    # Расчёт фич
    features = builder.build_extended_features(df_5m, df_15m, df_30m, df_4h)

    print("=" * 80)
    print("ТЕСТ BINANCE FEATURE BUILDER")
    print("=" * 80)
    print(f"✅ Размерность фич: {features.shape[0]}")
    print(f"✅ Ожидалось: 68")
    print(f"✅ Совпадение: {'ДА' if features.shape[0] == 68 else 'НЕТ'}")
    print()
    print("Первые 10 фич (5m momentum):")
    print(features[:10])
    print()
    print("Фичи 14-24 (15m VWAP/Bollinger):")
    print(features[14:24])
    print()
    print("Последние 10 фич (4h ATR/phases):")
    print(features[-10:])
    print()
    print("Статистика:")
    print(f"  Min: {np.min(features):.4f}")
    print(f"  Max: {np.max(features):.4f}")
    print(f"  Mean: {np.mean(features):.4f}")
    print(f"  Std: {np.std(features):.4f}")
    print(f"  NaN count: {np.sum(np.isnan(features))}")
    print(f"  Inf count: {np.sum(np.isinf(features))}")
