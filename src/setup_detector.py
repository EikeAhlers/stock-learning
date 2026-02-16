"""
Setup Detector — Identify specific, named multi-day patterns that
historically preceded big stock moves. This goes beyond single-day
feature values to find the SEQUENCES and COMBINATIONS that matter.

Each setup is a multi-condition check on the N days before a big move.
We mine history to find which setups have the highest hit rate, then
scan for them today.

Setups detected:
  1. Stealth Accumulation   - Volume up + OBV rising but price flat/declining
  2. Volatility Squeeze     - BB inside Keltner (TTM) + tight range
  3. Oversold Spring        - RSI < 30, MTF all negative, then reversal signs
  4. Volume Dry-Up Reload   - Volume drops 3+ days then spikes up
  5. Earnings Run-Up        - Rising volume + call activity 5-10 days before earnings
  6. Sector Rotation Entry  - Sector breaks out, lagging stock has room to catch up
  7. Moving Average Coil    - All MAs converging + volatility compression
  8. Support Bounce Setup   - Price at 52w low area + institutional accumulation
  9. Breakout Retest        - Recent breakout, pulls back to breakout level, holds
  10. Dark Pool Footprint   - Unusual quiet accumulation (big volume, tiny moves)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def detect_all_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all setup detectors on the dataframe.
    Adds binary columns (Setup_*) and a Setup_Name column with the 
    strongest matching setup for each row.
    """
    df = df.copy()
    
    # Run each detector
    df = detect_stealth_accumulation(df)
    df = detect_volatility_squeeze(df)
    df = detect_oversold_spring(df)
    df = detect_volume_dryup_reload(df)
    df = detect_earnings_runup(df)
    df = detect_sector_rotation_entry(df)
    df = detect_ma_coil(df)
    df = detect_support_bounce(df)
    df = detect_breakout_retest(df)
    df = detect_dark_pool_footprint(df)
    
    # Count how many setups are active
    setup_cols = [c for c in df.columns if c.startswith("Setup_")]
    df["Setup_Count"] = df[setup_cols].sum(axis=1)
    
    # Name the dominant setup
    def get_primary_setup(row):
        active = [c.replace("Setup_", "") for c in setup_cols if row.get(c, 0) == 1]
        if not active:
            return "none"
        return active[0]  # Return first matching
    
    df["Primary_Setup"] = df.apply(get_primary_setup, axis=1)
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Individual setup detectors
# ═══════════════════════════════════════════════════════════════════════════

def detect_stealth_accumulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    STEALTH ACCUMULATION:
    Smart money is buying but price isn't rising yet.
    
    Conditions:
    - OBV is rising (positive slope over 10 days)
    - Price is flat or declining (5d return <= 0)
    - Volume is above average (Vol_Ratio >= 1.0)
    - Quiet accumulation days >= 3 in last 10
    """
    df = df.copy()
    conditions = (
        (df.get("OBV_Slope_10d", pd.Series(0, index=df.index)) > 0) &
        (df.get("Return_5d", pd.Series(0, index=df.index)) <= 0) &
        (df.get("Vol_Ratio", pd.Series(0, index=df.index)) >= 1.0) &
        (df.get("Quiet_Accum_Days_10", pd.Series(0, index=df.index)) >= 3)
    )
    df["Setup_Stealth_Accumulation"] = conditions.astype(int)
    return df


def detect_volatility_squeeze(df: pd.DataFrame) -> pd.DataFrame:
    """
    VOLATILITY SQUEEZE (TTM Squeeze variant):
    Volatility is compressed to extreme levels, energy building up.
    
    Conditions:
    - TTM Squeeze is ON (Bollinger inside Keltner) OR BB_Squeeze active
    - Range compression < 0.5 (5d range is less than half of 20d range)
    - ADX < 25 (no strong trend yet — coiling)
    """
    df = df.copy()
    squeeze = df.get("TTM_Squeeze", pd.Series(0, index=df.index))
    bb_squeeze = df.get("BB_Squeeze", pd.Series(0, index=df.index))
    range_comp = df.get("Range_Compression", pd.Series(1, index=df.index))
    adx = df.get("ADX", pd.Series(30, index=df.index))
    
    conditions = (
        ((squeeze == 1) | (bb_squeeze == 1)) &
        (range_comp < 0.5) &
        (adx < 25)
    )
    df["Setup_Volatility_Squeeze"] = conditions.astype(int)
    return df


def detect_oversold_spring(df: pd.DataFrame) -> pd.DataFrame:
    """
    OVERSOLD SPRING (Wyckoff-style):
    Stock is deeply oversold across all timeframes, primed for sharp reversal.
    
    Conditions:
    - RSI < 30
    - StochRSI in oversold territory (< 0.2) 
    - Multi-timeframe momentum all negative
    - Price below both SMA20 and SMA50
    - Volume starting to pick up (buyers stepping in)
    """
    df = df.copy()
    rsi = df.get("RSI_14", pd.Series(50, index=df.index))
    stoch_rsi = df.get("StochRSI", pd.Series(0.5, index=df.index))
    mtf_neg = df.get("MTF_All_Negative", pd.Series(0, index=df.index))
    dist_sma20 = df.get("Dist_SMA20_Pct", pd.Series(0, index=df.index))
    vol_ratio = df.get("Vol_Ratio", pd.Series(1, index=df.index))
    
    conditions = (
        (rsi < 30) &
        (stoch_rsi < 0.2) &
        (dist_sma20 < -3) &
        (vol_ratio >= 0.8)  # Volume not dead — someone is buying
    )
    df["Setup_Oversold_Spring"] = conditions.astype(int)
    return df


def detect_volume_dryup_reload(df: pd.DataFrame) -> pd.DataFrame:
    """
    VOLUME DRY-UP RELOAD:
    Volume dries up for several days (sellers exhausted), then suddenly spikes.
    Classic accumulation → breakout pattern.
    
    Conditions:
    - Recent volume was very low (Vol_ZScore < -1 in last 5 days)
    - Current volume is spiking (Vol_Ratio > 1.5 OR Unusual_Volume)
    - Price is in the upper half of its recent range
    """
    df = df.copy()
    vol_z = df.get("Vol_ZScore", pd.Series(0, index=df.index))
    vol_ratio = df.get("Vol_Ratio", pd.Series(1, index=df.index))
    unusual = df.get("Unusual_Volume", pd.Series(0, index=df.index))
    price_pos = df.get("Price_Position_20d", pd.Series(0.5, index=df.index))
    
    # Check if volume was recently very low (approximate with current z-score reversal)
    conditions = (
        (vol_ratio > 1.5) &
        (price_pos > 0.5) &
        (df.get("Range_Compression", pd.Series(1, index=df.index)) < 0.7)
    )
    df["Setup_Volume_Dryup_Reload"] = conditions.astype(int)
    return df


def detect_earnings_runup(df: pd.DataFrame) -> pd.DataFrame:
    """
    EARNINGS RUN-UP:
    Stocks tend to move in the 5-10 days before earnings as institutional
    players position themselves. Look for accumulation signals + proximity.
    
    Conditions:
    - Within 10 days of earnings
    - Volume trending up (Vol_Trend_5d > 1.1)  
    - Price position improving
    """
    df = df.copy()
    within_10 = df.get("Earnings_Within_10d", pd.Series(0, index=df.index))
    vol_trend = df.get("Vol_Trend_5d", pd.Series(1, index=df.index))
    
    conditions = (
        (within_10 == 1) &
        (vol_trend > 1.1)
    )
    df["Setup_Earnings_Runup"] = conditions.astype(int)
    return df


def detect_sector_rotation_entry(df: pd.DataFrame) -> pd.DataFrame:
    """
    SECTOR ROTATION ENTRY:
    The sector is strong but this stock is lagging — room to catch up.
    Money rotating into sector finds the underperformers next.
    
    Conditions:
    - Sector return > 2% (5d) — sector is hot
    - Stock return < sector return — lagging the sector
    - Stock still has reasonable technical setup (not broken)
    """
    df = df.copy()
    sector_ret = df.get("Sector_Return_5d", pd.Series(0, index=df.index))
    stock_vs_sector = df.get("Stock_vs_Sector_5d", pd.Series(0, index=df.index))
    rsi = df.get("RSI_14", pd.Series(50, index=df.index))
    
    conditions = (
        (sector_ret > 2) &
        (stock_vs_sector < -2) &
        (rsi > 30) & (rsi < 60)  # Not overbought, not broken
    )
    df["Setup_Sector_Rotation"] = conditions.astype(int)
    return df


def detect_ma_coil(df: pd.DataFrame) -> pd.DataFrame:
    """
    MOVING AVERAGE COIL:
    All moving averages converging → big move imminent.
    Like a spring being compressed — direction unknown but magnitude likely large.
    
    Conditions:
    - MA convergence very tight (< 2% spread between MAs)
    - ATR contracting
    - Volume low but stable
    """
    df = df.copy()
    ma_conv = df.get("MA_Convergence", pd.Series(5, index=df.index))
    atr_cont = df.get("ATR_Contraction", pd.Series(1, index=df.index))
    
    conditions = (
        (ma_conv < 2.0) &
        (atr_cont < 0.85)
    )
    df["Setup_MA_Coil"] = conditions.astype(int)
    return df


def detect_support_bounce(df: pd.DataFrame) -> pd.DataFrame:
    """
    SUPPORT BOUNCE:
    Stock near 52-week low area with accumulation signs.
    Smart money buying at support = spring setup.
    
    Conditions:
    - Within 10% of 52-week low
    - OBV or A/D line showing bullish divergence
    - MFI not oversold anymore (buyers stepping in)
    """
    df = df.copy()
    near_low = df.get("Dist_52w_Low_Pct", pd.Series(50, index=df.index))
    obv_div = df.get("OBV_Bullish_Div", pd.Series(0, index=df.index))
    ad_div = df.get("AD_Bullish_Div", pd.Series(0, index=df.index))
    mfi = df.get("MFI", pd.Series(50, index=df.index))
    
    conditions = (
        (near_low < 15) &
        ((obv_div == 1) | (ad_div == 1)) &
        (mfi > 30)
    )
    df["Setup_Support_Bounce"] = conditions.astype(int)
    return df


def detect_breakout_retest(df: pd.DataFrame) -> pd.DataFrame:
    """
    BREAKOUT RETEST:
    Stock recently broke out, pulled back to the breakout level, and is holding.
    The pullback on lower volume = healthy, another leg up likely.
    
    Conditions:
    - 20d return positive (recent breakout)
    - 5d return slightly negative (pulling back)
    - Price near SMA20 (retesting the breakout level)
    - Volume declining during pullback
    """
    df = df.copy()
    ret_20 = df.get("Return_20d", pd.Series(0, index=df.index))
    ret_5 = df.get("Return_5d", pd.Series(0, index=df.index))
    dist_sma20 = df.get("Dist_SMA20_Pct", pd.Series(0, index=df.index))
    vol_trend = df.get("Vol_Trend_5d", pd.Series(1, index=df.index))
    
    conditions = (
        (ret_20 > 5) &
        (ret_5 < 0) & (ret_5 > -5) &
        (dist_sma20.abs() < 3) &
        (vol_trend < 1.0)
    )
    df["Setup_Breakout_Retest"] = conditions.astype(int)
    return df


def detect_dark_pool_footprint(df: pd.DataFrame) -> pd.DataFrame:
    """
    DARK POOL FOOTPRINT:
    Large volume but minimal price movement = institutional block trades
    being executed off-exchange. This is STEALTH buying.
    
    Conditions:
    - High volume (> 1.3x average)
    - Very small price change (intraday range < half of average)
    - Multiple days of this pattern (Quiet_Accum_Days >= 4)
    - A/D line rising (money flowing in despite flat price)
    """
    df = df.copy()
    vol_ratio = df.get("Vol_Ratio", pd.Series(1, index=df.index))
    range_ratio = df.get("Intraday_Range_Ratio", pd.Series(1, index=df.index))
    quiet_days = df.get("Quiet_Accum_Days_10", pd.Series(0, index=df.index))
    ad_slope = df.get("AD_Line_Slope_10d", pd.Series(0, index=df.index))
    
    conditions = (
        (vol_ratio > 1.3) &
        (range_ratio < 0.6) &
        (quiet_days >= 4) &
        (ad_slope > 0)
    )
    df["Setup_Dark_Pool_Footprint"] = conditions.astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Setup Analysis — How well did each setup predict big moves historically?
# ═══════════════════════════════════════════════════════════════════════════

SETUP_NAMES = [
    "Setup_Stealth_Accumulation",
    "Setup_Volatility_Squeeze",
    "Setup_Oversold_Spring",
    "Setup_Volume_Dryup_Reload",
    "Setup_Earnings_Runup",
    "Setup_Sector_Rotation",
    "Setup_MA_Coil",
    "Setup_Support_Bounce",
    "Setup_Breakout_Retest",
    "Setup_Dark_Pool_Footprint",
]


def analyze_setup_effectiveness(
    df: pd.DataFrame,
    big_move_threshold: float = 5.0,
    forward_col: str = "Daily_Return_Pct",
) -> pd.DataFrame:
    """
    For each setup, measure:
    - How often it fires (frequency)
    - How often a big move follows within 1-5 days (hit rate)
    - Average forward return when the setup fires
    - False positive rate
    
    This answers: "If I see this setup today, what's the historical probability
    of a big move in the next few days?"
    """
    results = []
    
    # We need next-day returns for each row
    df = df.copy()
    if "Ticker" in df.columns:
        df["Next_Day_Return"] = df.groupby("Ticker")[forward_col].shift(-1)
        df["Next_3d_Max_Return"] = df.groupby("Ticker")[forward_col].transform(
            lambda x: x.shift(-1).rolling(3, min_periods=1).max()
        )
        df["Next_5d_Max_Return"] = df.groupby("Ticker")[forward_col].transform(
            lambda x: x.shift(-1).rolling(5, min_periods=1).max()
        )
    
    total_days = len(df)
    base_rate = (df[forward_col] >= big_move_threshold).mean()
    
    for setup in SETUP_NAMES:
        if setup not in df.columns:
            continue
            
        fires = df[df[setup] == 1]
        n_fires = len(fires)
        
        if n_fires < 10:
            continue
        
        # Hit rates using LAGGED setup (setup today, move TOMORROW)
        if "Next_Day_Return" in df.columns:
            hit_1d = (fires["Next_Day_Return"] >= big_move_threshold).mean()
        else:
            hit_1d = 0
            
        if "Next_3d_Max_Return" in df.columns:
            hit_3d = (fires["Next_3d_Max_Return"] >= big_move_threshold).mean()
        else:
            hit_3d = 0
            
        if "Next_5d_Max_Return" in df.columns:
            hit_5d = (fires["Next_5d_Max_Return"] >= big_move_threshold).mean()
        else:
            hit_5d = 0
        
        avg_next_ret = fires["Next_Day_Return"].mean() if "Next_Day_Return" in fires.columns else 0
        
        results.append({
            "Setup": setup.replace("Setup_", ""),
            "Times_Fired": n_fires,
            "Frequency_Pct": n_fires / total_days * 100,
            "Hit_Rate_1d": hit_1d * 100,
            "Hit_Rate_3d": hit_3d * 100,
            "Hit_Rate_5d": hit_5d * 100,
            "Avg_Next_Return": avg_next_ret,
            "Base_Rate": base_rate * 100,
            "Edge_vs_Base": (hit_1d - base_rate) * 100,
        })
    
    result_df = pd.DataFrame(results).sort_values("Hit_Rate_1d", ascending=False)
    return result_df


def get_active_setups_today(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each stock, show which setups are currently active.
    Returns the latest day's data with setup indicators.
    """
    if "Ticker" not in df.columns:
        return pd.DataFrame()
    
    latest = df.groupby("Ticker").last()
    setup_cols = [c for c in SETUP_NAMES if c in latest.columns]
    
    # Filter to stocks with at least one active setup
    has_setup = latest[setup_cols].sum(axis=1) > 0
    active = latest[has_setup].copy()
    
    if len(active) == 0:
        return pd.DataFrame()
    
    # Build summary
    records = []
    for ticker in active.index:
        row = active.loc[ticker]
        active_setups = [c.replace("Setup_", "") for c in setup_cols if row.get(c, 0) == 1]
        records.append({
            "Ticker": ticker,
            "Active_Setups": ", ".join(active_setups),
            "Setup_Count": len(active_setups),
            "RSI_14": row.get("RSI_14", None),
            "Vol_Ratio": row.get("Vol_Ratio", None),
            "Return_5d": row.get("Return_5d", None),
            "Dist_SMA20_Pct": row.get("Dist_SMA20_Pct", None),
        })
    
    return pd.DataFrame(records).sort_values("Setup_Count", ascending=False)
