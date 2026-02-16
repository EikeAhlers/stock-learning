"""
Advanced Feature Engineering — Deep technical signals that fingerprint
institutional activity, accumulation, and setup patterns BEFORE big moves.

Goes far beyond basic TA: OBV divergence, accumulation/distribution,
dark pool volume proxy, multi-timeframe momentum, Bollinger/Keltner squeeze,
MACD signal, ADX trend strength, money flow, and more.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# Core indicator functions (applied per-ticker)
# ═══════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Average True Range components."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def add_advanced_technicals(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 40+ advanced technical features to the price data.
    All computed per-ticker. These are the raw features; lagging happens later.
    """
    df = prices_df.sort_values(["Ticker", "Date"]).copy()

    for ticker in tqdm(df["Ticker"].unique(), desc="Advanced features"):
        mask = df["Ticker"] == ticker
        t = df.loc[mask].copy()
        close = t["Close"]
        high = t["High"]
        low = t["Low"]
        volume = t["Volume"]
        daily_ret = close.pct_change()

        # ── MACD ──────────────────────────────────────────────────────
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        macd_signal = _ema(macd_line, 9)
        macd_hist = macd_line - macd_signal
        df.loc[mask, "MACD_Hist"] = macd_hist
        df.loc[mask, "MACD_Cross"] = (
            (macd_line > macd_signal).astype(int) -
            (macd_line.shift(1) > macd_signal.shift(1)).astype(int)
        )  # +1 = bullish cross, -1 = bearish cross, 0 = no change

        # ── Bollinger Bands ───────────────────────────────────────────
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mid * 100
        bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)
        df.loc[mask, "BB_Width"] = bb_width
        df.loc[mask, "BB_Pct_B"] = bb_pct_b
        # BB squeeze: width at 6-month low
        df.loc[mask, "BB_Squeeze"] = (
            bb_width <= bb_width.rolling(120, min_periods=60).min() * 1.05
        ).astype(int)

        # ── Keltner Channel ───────────────────────────────────────────
        tr = _true_range(high, low, close)
        atr_20 = tr.rolling(20).mean()
        kc_upper = _ema(close, 20) + 1.5 * atr_20
        kc_lower = _ema(close, 20) - 1.5 * atr_20
        # Bollinger inside Keltner = TTM Squeeze
        df.loc[mask, "TTM_Squeeze"] = (
            (bb_lower > kc_lower) & (bb_upper < kc_upper)
        ).astype(int)

        # ── ATR and ATR ratio ────────────────────────────────────────
        atr_14 = tr.rolling(14).mean()
        df.loc[mask, "ATR_14"] = atr_14
        df.loc[mask, "ATR_Ratio"] = atr_14 / close * 100  # ATR as % of price
        # Is ATR contracting? (current vs 20d ago)
        df.loc[mask, "ATR_Contraction"] = atr_14 / atr_14.shift(20)

        # ── ADX (Average Directional Index) ───────────────────────────
        # Simplified ADX calculation
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        # Zero out the smaller one
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        atr_smooth = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_smooth)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_smooth)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(14).mean()
        df.loc[mask, "ADX"] = adx
        df.loc[mask, "ADX_Rising"] = (adx > adx.shift(5)).astype(int)

        # ── Stochastic RSI ───────────────────────────────────────────
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        stoch_rsi = (rsi - rsi.rolling(14).min()) / (
            rsi.rolling(14).max() - rsi.rolling(14).min()
        ).replace(0, np.nan)
        df.loc[mask, "StochRSI"] = stoch_rsi
        df.loc[mask, "StochRSI_Oversold"] = (stoch_rsi < 0.2).astype(int)

        # ── Williams %R ──────────────────────────────────────────────
        williams_r = (high.rolling(14).max() - close) / (
            high.rolling(14).max() - low.rolling(14).min()
        ).replace(0, np.nan) * -100
        df.loc[mask, "Williams_R"] = williams_r

        # ── On-Balance Volume (OBV) + Divergence ─────────────────────
        obv = (np.sign(daily_ret) * volume).cumsum()
        obv_sma20 = obv.rolling(20).mean()
        df.loc[mask, "OBV"] = obv
        # OBV slope: difference over 10 days (fast approximation vs slow polyfit)
        obv_slope = obv.diff(10) / 10
        df.loc[mask, "OBV_Slope_10d"] = obv_slope
        # OBV divergence: price falling but OBV rising = bullish divergence
        price_change_10 = close.diff(10)
        df.loc[mask, "OBV_Bullish_Div"] = (
            (price_change_10 < 0) & (obv_slope > 0)
        ).astype(int)

        # ── Accumulation/Distribution Line ────────────────────────────
        clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        ad_line = (clv * volume).cumsum()
        df.loc[mask, "AD_Line"] = ad_line
        # AD slope: difference over 10 days (fast approximation)
        ad_slope = ad_line.diff(10) / 10
        df.loc[mask, "AD_Line_Slope_10d"] = ad_slope
        # A/D divergence: price flat/declining but A/D rising = accumulation
        df.loc[mask, "AD_Bullish_Div"] = (
            (price_change_10 < 0) &
            (ad_slope > 0)
        ).astype(int)

        # ── Money Flow Index (MFI) ───────────────────────────────────
        typical_price = (high + low + close) / 3
        raw_mf = typical_price * volume
        mf_positive = raw_mf.where(typical_price > typical_price.shift(1), 0)
        mf_negative = raw_mf.where(typical_price < typical_price.shift(1), 0)
        mf_ratio = mf_positive.rolling(14).sum() / mf_negative.rolling(14).sum().replace(0, np.nan)
        mfi = 100 - (100 / (1 + mf_ratio))
        df.loc[mask, "MFI"] = mfi

        # ── Unusual Volume Score ─────────────────────────────────────
        vol_mean_20 = volume.rolling(20).mean()
        vol_std_20 = volume.rolling(20).std()
        z_vol = (volume - vol_mean_20) / vol_std_20.replace(0, np.nan)
        df.loc[mask, "Vol_ZScore"] = z_vol
        df.loc[mask, "Unusual_Volume"] = (z_vol > 2.0).astype(int)

        # ── Volume-Price Trend (accumulation proxy) ──────────────────
        # Count days in last 10 where volume > avg BUT price barely moved
        # This signals stealth accumulation
        quiet_accum = (
            (volume > vol_mean_20) & (daily_ret.abs() < daily_ret.abs().rolling(20).median())
        ).astype(int)
        df.loc[mask, "Quiet_Accum_Days_10"] = quiet_accum.rolling(10).sum()

        # Volume on up days vs down days (10-day window)
        up_vol = (volume * (daily_ret > 0).astype(int)).rolling(10).sum()
        down_vol = (volume * (daily_ret < 0).astype(int)).rolling(10).sum()
        df.loc[mask, "Up_Down_Vol_Ratio"] = up_vol / down_vol.replace(0, np.nan)

        # ── Range Compression ────────────────────────────────────────
        range_5d = high.rolling(5).max() - low.rolling(5).min()
        range_20d = high.rolling(20).max() - low.rolling(20).min()
        df.loc[mask, "Range_Compression"] = range_5d / range_20d.replace(0, np.nan)
        # Extreme compression = coiling
        df.loc[mask, "Tight_Range"] = (
            (range_5d / close) < (range_20d / close).rolling(60, min_periods=30).quantile(0.1)
        ).astype(int)

        # ── Moving Average Convergence ───────────────────────────────
        sma5 = close.rolling(5).mean()
        sma10 = close.rolling(10).mean()
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        # How close are the MAs? (normalized)
        ma_spread = (sma5.max() - sma50.min()) if sma50.notna().any() else 1
        if isinstance(ma_spread, pd.Series):
            ma_spread = ma_spread.max()
        ma_convergence = pd.concat([sma5, sma10, sma20, sma50], axis=1).std(axis=1)
        df.loc[mask, "MA_Convergence"] = ma_convergence / close * 100

        # ── Price relative to key levels ─────────────────────────────
        df.loc[mask, "Dist_52w_High_Pct"] = (
            (close - close.rolling(252, min_periods=100).max()) /
            close.rolling(252, min_periods=100).max() * 100
        )
        df.loc[mask, "Dist_52w_Low_Pct"] = (
            (close - close.rolling(252, min_periods=100).min()) /
            close.rolling(252, min_periods=100).min() * 100
        )

        # ── Multi-timeframe momentum alignment ───────────────────────
        ret_1d = daily_ret * 100
        ret_3d = close.pct_change(3) * 100
        ret_5d = close.pct_change(5) * 100
        ret_10d = close.pct_change(10) * 100
        ret_20d = close.pct_change(20) * 100
        # Count how many timeframes are positive
        mtf_bull = (
            (ret_1d > 0).astype(int) + (ret_3d > 0).astype(int) +
            (ret_5d > 0).astype(int) + (ret_10d > 0).astype(int) +
            (ret_20d > 0).astype(int)
        )
        df.loc[mask, "MTF_Bullish_Count"] = mtf_bull
        # All negative = potential oversold bounce setup
        df.loc[mask, "MTF_All_Negative"] = (mtf_bull == 0).astype(int)

        # ── Relative Strength vs market ──────────────────────────────
        # (will be filled in by external data module with SPY comparison)

        # ── Consecutive volume buildup (vectorized) ──────────────────
        vol_increasing = (volume > volume.shift(1)).astype(int)
        # Use groupby on cumulative non-increasing to count consecutive increases
        groups = (vol_increasing == 0).cumsum()
        consec_vol_up = vol_increasing.groupby(groups).cumsum()
        df.loc[mask, "Consec_Vol_Up"] = consec_vol_up

        # ── Intraday range relative to recent ────────────────────────
        intraday_range = (high - low) / close * 100
        df.loc[mask, "Intraday_Range_Pct"] = intraday_range
        df.loc[mask, "Intraday_Range_Ratio"] = (
            intraday_range / intraday_range.rolling(20).mean()
        )

        # ── Close location value (where in the day's range did it close?) ──
        df.loc[mask, "Close_Location"] = (close - low) / (high - low).replace(0, np.nan)
        # Bullish close: close near high for multiple days
        close_loc = (close - low) / (high - low).replace(0, np.nan)
        df.loc[mask, "Bullish_Close_Streak"] = (close_loc > 0.7).astype(int).rolling(5).sum()

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Complete list of new features for prediction
# ═══════════════════════════════════════════════════════════════════════════

ADVANCED_RAW_FEATURES = [
    # MACD
    "MACD_Hist", "MACD_Cross",
    # Bollinger
    "BB_Width", "BB_Pct_B", "BB_Squeeze",
    # Keltner/TTM
    "TTM_Squeeze",
    # ATR
    "ATR_Ratio", "ATR_Contraction",
    # ADX
    "ADX", "ADX_Rising",
    # Stochastic
    "StochRSI", "StochRSI_Oversold",
    # Williams
    "Williams_R",
    # Volume analysis
    "OBV_Slope_10d", "OBV_Bullish_Div",
    "AD_Line_Slope_10d", "AD_Bullish_Div",
    "MFI",
    "Vol_ZScore", "Unusual_Volume",
    "Quiet_Accum_Days_10", "Up_Down_Vol_Ratio",
    "Consec_Vol_Up",
    # Range/compression
    "Range_Compression", "Tight_Range",
    "MA_Convergence",
    # Price levels
    "Dist_52w_High_Pct", "Dist_52w_Low_Pct",
    # Multi-timeframe
    "MTF_Bullish_Count", "MTF_All_Negative",
    # Intraday
    "Intraday_Range_Pct", "Intraday_Range_Ratio",
    "Close_Location", "Bullish_Close_Streak",
]
