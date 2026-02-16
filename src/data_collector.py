"""
Data Collector — Pull historical stock data and identify the biggest daily movers.

This module:
1. Downloads historical OHLCV data for a broad universe of stocks
2. Calculates daily returns
3. Identifies the top N gainers and losers each day
4. Enriches each mover with pre-move context (volume, technicals, etc.)
"""
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import DATA_DIR, save_dataframe, load_dataframe, cache_exists, get_sp500_tickers


# ---------------------------------------------------------------------------
# 1. Download historical price data for a universe of stocks
# ---------------------------------------------------------------------------

def download_universe(
    tickers: list = None,
    period: str = "2y",
    interval: str = "1d",
    cache_name: str = "universe_prices",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a list of tickers.
    
    Returns a multi-index DataFrame: columns = (field, ticker), index = date.
    Caches results to disk for fast re-use.
    """
    if not force_refresh and cache_exists(cache_name):
        return load_dataframe(cache_name)
    
    if tickers is None:
        tickers = get_sp500_tickers()
    
    print(f"Downloading data for {len(tickers)} tickers ({period})...")
    
    # Download in batches to avoid rate limits
    all_data = {}
    batch_size = 50
    
    for i in tqdm(range(0, len(tickers), batch_size), desc="Downloading"):
        batch = tickers[i : i + batch_size]
        try:
            data = yf.download(
                batch, 
                period=period, 
                interval=interval, 
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if not data.empty:
                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            ticker_data = data
                        else:
                            ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else None
                        
                        if ticker_data is not None and not ticker_data.empty:
                            # Drop rows where Close is NaN
                            ticker_data = ticker_data.dropna(subset=["Close"])
                            if len(ticker_data) > 50:  # Need enough history
                                all_data[ticker] = ticker_data
                    except Exception:
                        pass
        except Exception as e:
            print(f"  Batch error: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    # Combine into a single DataFrame with ticker column
    records = []
    for ticker, df in all_data.items():
        df = df.copy()
        df["Ticker"] = ticker
        records.append(df)
    
    combined = pd.concat(records)
    combined.index.name = "Date"
    # Normalize to tz-naive for consistency
    if hasattr(combined.index, 'tz') and combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)
    
    save_dataframe(combined, cache_name)
    print(f"Successfully downloaded data for {len(all_data)} tickers")
    return combined


# ---------------------------------------------------------------------------
# 2. Calculate daily returns and identify top movers
# ---------------------------------------------------------------------------

def calculate_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily return columns to the price data."""
    df = prices_df.copy()
    
    # Calculate returns per ticker
    df = df.sort_values(["Ticker", "Date"])
    df["Daily_Return_Pct"] = df.groupby("Ticker")["Close"].pct_change() * 100
    df["Prev_Close"] = df.groupby("Ticker")["Close"].shift(1)
    
    # Remove first row per ticker (no return calculable)
    df = df.dropna(subset=["Daily_Return_Pct"])
    
    return df


def find_top_movers(
    returns_df: pd.DataFrame,
    top_n: int = 10,
    min_return_pct: float = 5.0,
    cache_name: str = "top_movers",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Find the top N biggest daily gainers for each trading day.
    
    Parameters:
        returns_df: DataFrame with daily returns
        top_n: Number of top movers to keep per day
        min_return_pct: Minimum daily return to qualify as a "big mover"
        
    Returns:
        DataFrame of top movers with their returns and context
    """
    if not force_refresh and cache_exists(cache_name):
        return load_dataframe(cache_name)
    
    df = returns_df.copy()
    
    # Filter to significant moves only
    big_movers = df[df["Daily_Return_Pct"] >= min_return_pct].copy()
    
    # Rank within each day
    big_movers["Daily_Rank"] = big_movers.groupby(big_movers.index)["Daily_Return_Pct"].rank(
        ascending=False, method="first"
    )
    
    # Keep top N per day
    top_movers = big_movers[big_movers["Daily_Rank"] <= top_n].copy()
    top_movers = top_movers.sort_values(
        [top_movers.index.name or "Date", "Daily_Rank"]
    )
    
    save_dataframe(top_movers, cache_name)
    print(f"Found {len(top_movers)} top mover events across {top_movers.index.nunique()} trading days")
    return top_movers


# ---------------------------------------------------------------------------
# 3. Enrich movers with pre-move technical context
# ---------------------------------------------------------------------------

def add_technical_context(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators that describe the state of each stock
    BEFORE its big move day. These are the potential predictive signals.
    """
    df = prices_df.sort_values(["Ticker", "Date"]).copy()
    
    for ticker in tqdm(df["Ticker"].unique(), desc="Adding technicals"):
        mask = df["Ticker"] == ticker
        ticker_df = df.loc[mask].copy()
        
        close = ticker_df["Close"]
        volume = ticker_df["Volume"]
        high = ticker_df["High"]
        low = ticker_df["Low"]
        
        # --- Moving Averages ---
        df.loc[mask, "SMA_5"] = close.rolling(5).mean()
        df.loc[mask, "SMA_10"] = close.rolling(10).mean()
        df.loc[mask, "SMA_20"] = close.rolling(20).mean()
        df.loc[mask, "SMA_50"] = close.rolling(50).mean()
        
        # Distance from moving averages (% above/below)
        df.loc[mask, "Dist_SMA20_Pct"] = ((close - close.rolling(20).mean()) / close.rolling(20).mean()) * 100
        df.loc[mask, "Dist_SMA50_Pct"] = ((close - close.rolling(50).mean()) / close.rolling(50).mean()) * 100
        
        # --- Volume Analysis ---
        df.loc[mask, "Vol_SMA20"] = volume.rolling(20).mean()
        df.loc[mask, "Vol_Ratio"] = volume / volume.rolling(20).mean()  # >1 means above-average volume
        
        # Volume trend: average volume ratio over last 5 days
        vol_ratio = volume / volume.rolling(20).mean()
        df.loc[mask, "Vol_Trend_5d"] = vol_ratio.rolling(5).mean()
        
        # --- Momentum ---
        df.loc[mask, "Return_5d"] = close.pct_change(5) * 100
        df.loc[mask, "Return_10d"] = close.pct_change(10) * 100
        df.loc[mask, "Return_20d"] = close.pct_change(20) * 100
        
        # --- Volatility ---
        daily_ret = close.pct_change()
        df.loc[mask, "Volatility_10d"] = daily_ret.rolling(10).std() * 100
        df.loc[mask, "Volatility_20d"] = daily_ret.rolling(20).std() * 100
        
        # Volatility compression: ratio of recent vol to longer-term vol
        vol_10 = daily_ret.rolling(10).std()
        vol_20 = daily_ret.rolling(20).std()
        df.loc[mask, "Vol_Compression"] = vol_10 / vol_20  # <1 means compressing
        
        # --- Price Position ---
        # Where is price relative to its 20-day range? (0=low, 1=high)
        roll_high_20 = high.rolling(20).max()
        roll_low_20 = low.rolling(20).min()
        price_range = roll_high_20 - roll_low_20
        df.loc[mask, "Price_Position_20d"] = np.where(
            price_range > 0,
            (close - roll_low_20) / price_range,
            0.5
        )
        
        # --- Gap Analysis ---
        # Was there a gap up on the big day?
        df.loc[mask, "Gap_Pct"] = ((ticker_df["Open"] - ticker_df["Close"].shift(1)) / ticker_df["Close"].shift(1)) * 100
        
        # --- Consecutive Days ---
        # How many consecutive up/down days before this day?
        daily_returns = close.pct_change()
        up_streak = pd.Series(0, index=ticker_df.index)
        down_streak = pd.Series(0, index=ticker_df.index)
        
        for j in range(1, len(daily_returns)):
            idx = daily_returns.index[j]
            prev_idx = daily_returns.index[j - 1]
            if daily_returns.iloc[j - 1] > 0:
                up_streak.loc[idx] = up_streak.loc[prev_idx] + 1
                down_streak.loc[idx] = 0
            elif daily_returns.iloc[j - 1] < 0:
                down_streak.loc[idx] = down_streak.loc[prev_idx] + 1
                up_streak.loc[idx] = 0
        
        df.loc[mask, "Up_Streak"] = up_streak.values
        df.loc[mask, "Down_Streak"] = down_streak.values
        
        # --- RSI (14-day) ---
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df.loc[mask, "RSI_14"] = 100 - (100 / (1 + rs))
    
    return df


# ---------------------------------------------------------------------------
# 4. Full pipeline: download → returns → top movers → enriched
# ---------------------------------------------------------------------------

def run_full_pipeline(
    tickers: list = None,
    period: str = "2y",
    top_n: int = 10,
    min_return_pct: float = 5.0,
    force_refresh: bool = False,
) -> dict:
    """
    Run the complete data collection pipeline.
    
    Returns dict with:
        - 'prices': Full price history with technicals
        - 'returns': Price data with daily returns  
        - 'top_movers': The biggest daily gainers
    """
    # Step 1: Download price data
    prices = download_universe(tickers=tickers, period=period, force_refresh=force_refresh)
    
    # Step 2: Add technical indicators
    if not cache_exists("universe_with_technicals") or force_refresh:
        prices_enriched = add_technical_context(prices)
        save_dataframe(prices_enriched, "universe_with_technicals")
    else:
        prices_enriched = load_dataframe("universe_with_technicals")
    
    # Step 3: Calculate returns
    returns = calculate_daily_returns(prices_enriched)
    
    # Step 4: Find top movers
    top_movers = find_top_movers(returns, top_n=top_n, min_return_pct=min_return_pct, force_refresh=force_refresh)
    
    return {
        "prices": prices_enriched,
        "returns": returns,
        "top_movers": top_movers,
    }
