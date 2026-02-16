"""
Move Analyzer — Deep dive into WHY each big move happened.

This module takes the top movers and analyzes:
1. What category of move was it? (earnings, breakout, squeeze, etc.)
2. What were the pre-move conditions?
3. How did the stock behave AFTER the big move?
4. What signals were present before the move?
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1. Classify move types based on available data signals
# ---------------------------------------------------------------------------

def classify_move_type(row: pd.Series) -> str:
    """
    Attempt to classify a big move into a category based on
    the technical context around the move.
    
    Categories:
        - 'gap_breakout': Big gap up with high volume (likely news/earnings)
        - 'volume_surge_breakout': Not a gap, but massive volume spike
        - 'momentum_continuation': Already trending up, continued higher
        - 'oversold_bounce': Was deeply oversold (low RSI), then bounced
        - 'compression_breakout': Low volatility squeezed then exploded
        - 'reversal': Was in a downtrend, then reversed sharply
        - 'unknown': Can't classify from technicals alone
    """
    gap = row.get("Gap_Pct", 0)
    vol_ratio = row.get("Vol_Ratio", 1)
    rsi = row.get("RSI_14", 50)
    ret_5d = row.get("Return_5d", 0)
    ret_20d = row.get("Return_20d", 0)
    vol_compression = row.get("Vol_Compression", 1)
    dist_sma20 = row.get("Dist_SMA20_Pct", 0)
    price_pos = row.get("Price_Position_20d", 0.5)
    down_streak = row.get("Down_Streak", 0)
    
    # Gap breakout: opened significantly higher than previous close
    if gap > 3:
        return "gap_breakout"
    
    # Oversold bounce: RSI was low, stock was beaten down then snapped back
    if rsi < 35 and ret_20d < -5:
        return "oversold_bounce"
    
    # Compression breakout: volatility was compressed, then exploded
    if vol_compression < 0.8 and vol_ratio > 1.5:
        return "compression_breakout"
    
    # Reversal: was in downtrend, then reversed
    if ret_20d < -5 and (down_streak >= 2 or ret_5d < -3):
        return "reversal"
    
    # Momentum continuation: already trending up, continued higher
    if ret_5d > 2 and ret_20d > 3:
        return "momentum_continuation"
    
    # Volume surge breakout: massive volume without a big gap
    if vol_ratio > 2 and abs(gap) < 2:
        return "volume_surge_breakout"
    
    # Quiet breakout: low volume, near top of range, intraday move (not gap)
    if price_pos > 0.8 and abs(gap) < 2:
        return "range_breakout"
    
    # Catch-up rally: stock was lagging, then caught up to sector
    if dist_sma20 < 0 and ret_5d > 0:
        return "catch_up_rally"
    
    return "unknown"


def classify_all_movers(top_movers: pd.DataFrame) -> pd.DataFrame:
    """Add move type classification to all top movers."""
    df = top_movers.copy()
    df["Move_Type"] = df.apply(classify_move_type, axis=1)
    return df


# ---------------------------------------------------------------------------
# 2. Analyze pre-move conditions (what was happening BEFORE the big day)
# ---------------------------------------------------------------------------

def score_premove_signals(row: pd.Series) -> Dict[str, float]:
    """
    Score individual pre-move signals on a 0-1 scale.
    These signals describe the CONDITIONS that existed before the big move.
    
    Returns a dict of signal scores.
    """
    signals = {}
    
    # 1. Volume building (accumulation before breakout)
    vol_trend = row.get("Vol_Trend_5d", 1)
    signals["volume_accumulation"] = min(max((vol_trend - 0.8) / 1.2, 0), 1)
    
    # 2. Price near breakout level (high in recent range)
    price_pos = row.get("Price_Position_20d", 0.5)
    signals["near_breakout"] = price_pos  # Already 0-1
    
    # 3. Volatility compression (coiling before explosion)
    vol_comp = row.get("Vol_Compression", 1)
    signals["volatility_squeeze"] = min(max(1 - vol_comp, 0), 1)
    
    # 4. RSI momentum (not overbought, room to run)
    rsi = row.get("RSI_14", 50)
    if rsi < 30:
        signals["rsi_signal"] = 0.9  # Oversold — could bounce
    elif rsi < 50:
        signals["rsi_signal"] = 0.5  # Neutral-low
    elif rsi < 70:
        signals["rsi_signal"] = 0.7  # Healthy momentum
    else:
        signals["rsi_signal"] = 0.3  # Overbought — risky
    
    # 5. Short-term momentum alignment
    ret_5d = row.get("Return_5d", 0)
    ret_10d = row.get("Return_10d", 0)
    if ret_5d > 0 and ret_10d > 0:
        signals["momentum_aligned"] = min(ret_5d / 10, 1)
    elif ret_5d < -5:
        signals["momentum_aligned"] = 0.6  # Oversold bounce potential
    else:
        signals["momentum_aligned"] = 0.3
    
    # 6. Distance from SMA20 (mean reversion potential)
    dist_sma20 = row.get("Dist_SMA20_Pct", 0)
    if dist_sma20 < -5:
        signals["mean_reversion"] = min(abs(dist_sma20) / 15, 1)
    else:
        signals["mean_reversion"] = 0.1
    
    # 7. Gap signal (was there a gap on the move day?)
    gap = row.get("Gap_Pct", 0)
    signals["gap_signal"] = min(max(gap / 10, 0), 1)
    
    return signals


def analyze_premove_conditions(top_movers: pd.DataFrame) -> pd.DataFrame:
    """Add pre-move signal scores to all top movers."""
    df = top_movers.copy()
    
    # Score each mover's pre-conditions
    signal_dicts = df.apply(score_premove_signals, axis=1)
    signal_df = pd.DataFrame(signal_dicts.tolist(), index=df.index)
    
    # Add individual signals
    for col in signal_df.columns:
        df[f"Signal_{col}"] = signal_df[col]
    
    # Composite score (average of all signals)
    df["Composite_Signal_Score"] = signal_df.mean(axis=1)
    
    return df


# ---------------------------------------------------------------------------
# 3. Analyze post-move behavior (what happened AFTER the big day)
# ---------------------------------------------------------------------------

def add_post_move_returns(
    top_movers: pd.DataFrame,
    full_returns: pd.DataFrame,
    forward_days: list = [1, 2, 3, 5, 10, 20],
) -> pd.DataFrame:
    """
    For each top mover event, calculate the forward returns
    (what happened after the big move day).
    
    This helps us understand:
    - Do big movers continue running? (momentum)
    - Do they reverse? (mean reversion)
    - How long does the effect last?
    """
    df = top_movers.reset_index().copy()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    
    for fwd in forward_days:
        df[f"Fwd_Return_{fwd}d"] = np.nan
    
    # Pre-group full_returns by ticker for speed
    grouped = {ticker: grp.sort_index() for ticker, grp in full_returns.groupby("Ticker")}
    
    # For each mover event, look up forward returns
    for i in range(len(df)):
        ticker = df.iloc[i]["Ticker"]
        move_date = pd.Timestamp(df.iloc[i][date_col])
        move_close = df.iloc[i]["Close"]
        
        if ticker not in grouped:
            continue
            
        ticker_data = grouped[ticker]
        ticker_data = ticker_data[ticker_data.index > move_date]
        
        if len(ticker_data) == 0:
            continue
        
        for fwd in forward_days:
            if len(ticker_data) >= fwd:
                future_close = ticker_data.iloc[fwd - 1]["Close"]
                fwd_return = ((future_close - move_close) / move_close) * 100
                df.iloc[i, df.columns.get_loc(f"Fwd_Return_{fwd}d")] = fwd_return
    
    # Restore date index
    df = df.set_index(date_col)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# 4. Summary statistics by move type
# ---------------------------------------------------------------------------

def summarize_by_move_type(analyzed_movers: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table showing:
    - Average return per move type
    - Average pre-move signal scores per type
    - Average forward returns per type
    - Count of occurrences
    """
    signal_cols = [c for c in analyzed_movers.columns if c.startswith("Signal_")]
    fwd_cols = [c for c in analyzed_movers.columns if c.startswith("Fwd_Return_")]
    
    agg_dict = {
        "Daily_Return_Pct": ["mean", "median", "std", "count"],
        "Composite_Signal_Score": ["mean", "median"],
    }
    
    for col in signal_cols:
        agg_dict[col] = "mean"
    
    for col in fwd_cols:
        agg_dict[col] = ["mean", "median"]
    
    summary = analyzed_movers.groupby("Move_Type").agg(agg_dict)
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    
    return summary.sort_values("Daily_Return_Pct_count", ascending=False)


def get_top_signal_patterns(analyzed_movers: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Find the most common pre-move signal combinations among big movers.
    
    This answers: "What conditions were most frequently present
    before a stock made a big move?"
    """
    signal_cols = [c for c in analyzed_movers.columns if c.startswith("Signal_")]
    
    # Discretize signals into High/Low
    signals_discrete = analyzed_movers[signal_cols].copy()
    for col in signal_cols:
        signals_discrete[col] = (signals_discrete[col] > 0.5).map({True: "H", False: "L"})
    
    # Create a pattern string
    signals_discrete["Pattern"] = signals_discrete.apply(
        lambda row: " | ".join([f"{c.replace('Signal_', '')}={row[c]}" for c in signal_cols]),
        axis=1,
    )
    
    # Count patterns
    pattern_counts = signals_discrete["Pattern"].value_counts().head(top_n)
    
    result = pd.DataFrame({
        "Pattern": pattern_counts.index,
        "Count": pattern_counts.values,
        "Pct_of_Total": (pattern_counts.values / len(analyzed_movers) * 100).round(1),
    })
    
    return result
