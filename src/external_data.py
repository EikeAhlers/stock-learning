"""
External Data — Pull data BEYOND basic OHLCV that reveals institutional
positioning, market regime, earnings catalysts, and options flow.

Data sources:
  - yfinance: VIX, sector ETFs, earnings dates, options chains
  - SEC EDGAR: insider transactions (free API)
  - Computed: relative strength, sector rotation, earnings proximity
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

from src.utils import DATA_DIR, save_dataframe, load_dataframe, cache_exists


# ═══════════════════════════════════════════════════════════════════════════
# 1. Market Regime — VIX level, VIX term structure, SPY trend
# ═══════════════════════════════════════════════════════════════════════════

def download_market_regime(period: str = "2y") -> pd.DataFrame:
    """
    Download VIX and SPY data to characterize the daily market regime.
    
    Returns DataFrame indexed by Date with columns:
      VIX, VIX_SMA20, VIX_Percentile, VIX_Change_5d,
      SPY_Return_5d, SPY_Trend (above/below 50SMA),
      Market_Regime (risk_on / risk_off / transition)
    """
    cache_name = "market_regime"
    if cache_exists(cache_name):
        return load_dataframe(cache_name)

    print("Downloading VIX and SPY for market regime analysis...")
    vix = yf.download("^VIX", period=period, progress=False)
    spy = yf.download("SPY", period=period, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    # Normalize timezone
    if hasattr(vix.index, 'tz') and vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)
    if hasattr(spy.index, 'tz') and spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)

    regime = pd.DataFrame(index=vix.index)
    regime.index.name = "Date"

    # VIX features
    regime["VIX"] = vix["Close"]
    regime["VIX_SMA20"] = regime["VIX"].rolling(20).mean()
    regime["VIX_Percentile"] = regime["VIX"].rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    regime["VIX_Change_5d"] = regime["VIX"].pct_change(5) * 100
    regime["VIX_Spike"] = (regime["VIX"] > regime["VIX_SMA20"] * 1.2).astype(int)

    # SPY features
    spy_close = spy["Close"]
    regime["SPY_Close"] = spy_close
    regime["SPY_SMA50"] = spy_close.rolling(50).mean()
    regime["SPY_Return_1d"] = spy_close.pct_change() * 100
    regime["SPY_Return_5d"] = spy_close.pct_change(5) * 100
    regime["SPY_Above_50SMA"] = (spy_close > spy_close.rolling(50).mean()).astype(int)

    # Market regime classification
    def classify_regime(row):
        vix = row.get("VIX", 20)
        spy_trend = row.get("SPY_Above_50SMA", 1)
        if vix < 18 and spy_trend == 1:
            return "low_vol_bull"
        elif vix < 25 and spy_trend == 1:
            return "normal_bull"
        elif vix >= 25 and spy_trend == 1:
            return "high_vol_bull"
        elif vix < 25 and spy_trend == 0:
            return "normal_bear"
        else:
            return "high_vol_bear"

    regime["Market_Regime"] = regime.apply(classify_regime, axis=1)

    save_dataframe(regime, cache_name)
    return regime


# ═══════════════════════════════════════════════════════════════════════════
# 2. Sector Relative Strength — How is this stock's sector doing?
# ═══════════════════════════════════════════════════════════════════════════

SECTOR_ETFS = {
    "XLK": "Technology", "XLV": "Healthcare", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLY": "Consumer_Disc",
    "XLP": "Consumer_Staples", "XLU": "Utilities", "XLRE": "Real_Estate",
    "XLB": "Materials", "XLC": "Communication",
}

# Map tickers to sectors (rough mapping based on common stocks)
# This will be enhanced dynamically using yfinance sector info
TICKER_SECTOR_MAP = {}


def download_sector_data(period: str = "2y") -> pd.DataFrame:
    """
    Download sector ETF data and compute relative strength metrics.
    """
    cache_name = "sector_data"
    if cache_exists(cache_name):
        return load_dataframe(cache_name)

    print("Downloading sector ETF data...")
    etf_list = list(SECTOR_ETFS.keys())
    data = yf.download(etf_list, period=period, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]

    # Normalize timezone
    if hasattr(close.index, 'tz') and close.index.tz is not None:
        close = close.copy()
        close.index = close.index.tz_localize(None)

    sector_df = pd.DataFrame(index=close.index)
    sector_df.index.name = "Date"

    for etf in etf_list:
        sector_name = SECTOR_ETFS[etf]
        if etf in close.columns:
            px = close[etf]
            sector_df[f"{sector_name}_Return_5d"] = px.pct_change(5) * 100
            sector_df[f"{sector_name}_Return_20d"] = px.pct_change(20) * 100
            sector_df[f"{sector_name}_RS_Rank"] = np.nan  # filled below

    # Rank sectors by 5d return each day (vectorized)
    ret_cols = [c for c in sector_df.columns if c.endswith("_Return_5d")]
    rank_df = sector_df[ret_cols].rank(axis=1, ascending=False)
    for col in ret_cols:
        rank_col = col.replace("_Return_5d", "_RS_Rank")
        sector_df[rank_col] = rank_df[col]

    save_dataframe(sector_df, cache_name)
    return sector_df


def get_ticker_sectors(tickers: list) -> dict:
    """
    Look up sector for each ticker via yfinance.
    Caches results to avoid repeated API calls.
    """
    import json, os
    cache_path = os.path.join(DATA_DIR, "ticker_sectors.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
    else:
        cached = {}

    missing = [t for t in tickers if t not in cached]
    if missing:
        print(f"Looking up sectors for {len(missing)} tickers...")
        for t in tqdm(missing, desc="Sector lookup"):
            try:
                info = yf.Ticker(t).info
                cached[t] = info.get("sector", "Unknown")
            except Exception:
                cached[t] = "Unknown"
            time.sleep(0.1)

        with open(cache_path, "w") as f:
            json.dump(cached, f)

    return cached


def add_sector_features(
    prices_df: pd.DataFrame,
    sector_data: pd.DataFrame,
    ticker_sectors: dict,
) -> pd.DataFrame:
    """
    Merge sector relative strength into per-stock data.
    Adds: Sector, Sector_Return_5d, Sector_Return_20d, Sector_RS_Rank,
          Stock_vs_Sector (outperformance)
    """
    df = prices_df.copy()
    
    # Map ticker → yfinance sector name
    df["Sector"] = df["Ticker"].map(ticker_sectors).fillna("Unknown")
    
    # Map yfinance sector names → our ETF sector names
    yf_to_etf = {
        "Technology": "Technology", "Information Technology": "Technology",
        "Healthcare": "Healthcare", "Health Care": "Healthcare",
        "Financials": "Financials", "Financial Services": "Financials",
        "Energy": "Energy", "Industrials": "Industrials",
        "Consumer Discretionary": "Consumer_Disc", "Consumer Cyclical": "Consumer_Disc",
        "Consumer Staples": "Consumer_Staples", "Consumer Defensive": "Consumer_Staples",
        "Utilities": "Utilities", "Real Estate": "Real_Estate",
        "Materials": "Materials", "Basic Materials": "Materials",
        "Communication Services": "Communication", "Communication": "Communication",
    }
    df["_Sector_ETF"] = df["Sector"].map(yf_to_etf).fillna("Unknown")
    
    # Normalize both indexes to tz-naive for alignment
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    sector_idx = sector_data.index
    if hasattr(sector_idx, 'tz') and sector_idx.tz is not None:
        sector_data = sector_data.copy()
        sector_data.index = sector_data.index.tz_localize(None)
    
    # Initialize columns
    df["Sector_Return_5d"] = np.nan
    df["Sector_Return_20d"] = np.nan
    df["Sector_RS_Rank"] = np.nan
    
    # For each sector, use reindex to align by date (vectorized, fast)
    for sector_name in SECTOR_ETFS.values():
        ret5_col = f"{sector_name}_Return_5d"
        ret20_col = f"{sector_name}_Return_20d"
        rank_col = f"{sector_name}_RS_Rank"
        
        if ret5_col not in sector_data.columns:
            continue
        
        mask = df["_Sector_ETF"] == sector_name
        if mask.sum() == 0:
            continue
        
        # reindex aligns sector_data to the dates in df for this sector's stocks
        aligned_dates = df.index[mask]
        df.loc[mask, "Sector_Return_5d"] = sector_data[ret5_col].reindex(aligned_dates).values
        if ret20_col in sector_data.columns:
            df.loc[mask, "Sector_Return_20d"] = sector_data[ret20_col].reindex(aligned_dates).values
        if rank_col in sector_data.columns:
            df.loc[mask, "Sector_RS_Rank"] = sector_data[rank_col].reindex(aligned_dates).values
    
    # Stock outperformance vs sector
    if "Return_5d" in df.columns:
        df["Stock_vs_Sector_5d"] = df["Return_5d"] - df["Sector_Return_5d"]
    
    df.drop(columns=["_Sector_ETF"], errors="ignore", inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. Earnings Proximity — Distance to next earnings date
# ═══════════════════════════════════════════════════════════════════════════

def get_earnings_dates(tickers: list) -> dict:
    """
    Get upcoming/recent earnings dates for each ticker.
    Returns dict: ticker -> list of earnings dates
    """
    import os, json
    cache_path = os.path.join(DATA_DIR, "earnings_dates.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        # Convert string dates back to datetime-friendly format
        return cached

    cached = {}
    print(f"Fetching earnings dates for {len(tickers)} tickers...")
    for t in tqdm(tickers, desc="Earnings dates"):
        try:
            tk = yf.Ticker(t)
            # Get earnings dates
            cal = tk.get_earnings_dates(limit=20)
            if cal is not None and len(cal) > 0:
                dates = [str(d.date()) for d in cal.index]
                cached[t] = dates
            else:
                cached[t] = []
        except Exception:
            cached[t] = []
        time.sleep(0.15)

    with open(cache_path, "w") as f:
        json.dump(cached, f)

    return cached


def add_earnings_proximity(
    prices_df: pd.DataFrame,
    earnings_dates: dict,
) -> pd.DataFrame:
    """
    Add days-to-next-earnings and days-since-last-earnings features.
    Stocks near earnings tend to have elevated vol and are more likely
    to make big moves.
    """
    df = prices_df.copy()
    df["Days_To_Earnings"] = np.nan
    df["Days_Since_Earnings"] = np.nan
    df["Earnings_Within_5d"] = 0
    df["Earnings_Within_10d"] = 0

    for ticker in df["Ticker"].unique():
        if ticker not in earnings_dates or not earnings_dates[ticker]:
            continue

        mask = df["Ticker"] == ticker
        dates = sorted([pd.Timestamp(d) for d in earnings_dates[ticker]])

        for idx in df.loc[mask].index:
            current_date = pd.Timestamp(idx)

            # Days to next earnings
            future = [d for d in dates if d > current_date]
            if future:
                days_to = (future[0] - current_date).days
                df.loc[idx, "Days_To_Earnings"] = days_to
                if days_to <= 5:
                    df.loc[idx, "Earnings_Within_5d"] = 1
                if days_to <= 10:
                    df.loc[idx, "Earnings_Within_10d"] = 1

            # Days since last earnings
            past = [d for d in dates if d <= current_date]
            if past:
                days_since = (current_date - past[-1]).days
                df.loc[idx, "Days_Since_Earnings"] = days_since

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. Options Flow — Put/Call ratio, unusual call activity
# ═══════════════════════════════════════════════════════════════════════════

def get_options_signals(tickers: list) -> pd.DataFrame:
    """
    Fetch current options data for a list of tickers.
    Computes put/call volume ratio and detects unusual call activity.

    NOTE: This gives a SNAPSHOT of current options, not historical.
    For historical analysis we'd need a paid data source.
    For the forecast model, we use this for TODAY's scoring only.
    """
    records = []
    print(f"Fetching options data for {len(tickers)} tickers...")

    for t in tqdm(tickers, desc="Options scan"):
        try:
            tk = yf.Ticker(t)
            # Get nearest expiration
            expirations = tk.options
            if not expirations:
                continue

            # Look at next 2 expirations
            calls_vol = 0
            puts_vol = 0
            calls_oi = 0
            puts_oi = 0
            unusual_calls = 0

            for exp in expirations[:2]:
                opt = tk.option_chain(exp)
                calls = opt.calls
                puts = opt.puts

                cv = calls["volume"].sum() if "volume" in calls.columns else 0
                pv = puts["volume"].sum() if "volume" in puts.columns else 0
                co = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
                po = puts["openInterest"].sum() if "openInterest" in puts.columns else 0

                calls_vol += cv if pd.notna(cv) else 0
                puts_vol += pv if pd.notna(pv) else 0
                calls_oi += co if pd.notna(co) else 0
                puts_oi += po if pd.notna(po) else 0

                # Unusual activity: volume > 3x open interest on calls
                if co > 0:
                    unusual = calls[calls["volume"] > 3 * calls["openInterest"]]
                    unusual_calls += len(unusual)

            pc_ratio = puts_vol / max(calls_vol, 1)
            pc_oi_ratio = puts_oi / max(calls_oi, 1)

            records.append({
                "Ticker": t,
                "Call_Volume": calls_vol,
                "Put_Volume": puts_vol,
                "PC_Ratio": pc_ratio,
                "Call_OI": calls_oi,
                "Put_OI": puts_oi,
                "PC_OI_Ratio": pc_oi_ratio,
                "Unusual_Call_Activity": unusual_calls,
                "Total_Options_Volume": calls_vol + puts_vol,
            })

        except Exception:
            pass
        time.sleep(0.1)

    if records:
        return pd.DataFrame(records)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Insider Trading — Recent insider buys (from SEC EDGAR, free)
# ═══════════════════════════════════════════════════════════════════════════

def get_insider_trades(tickers: list, days_back: int = 90) -> pd.DataFrame:
    """
    Fetch recent insider transactions from SEC EDGAR (free API).
    Focus on PURCHASES by insiders — a strong bullish signal.
    """
    import requests
    records = []
    headers = {
        "User-Agent": "StockResearch research@example.com",
        "Accept": "application/json",
    }

    print(f"Fetching insider data for {len(tickers)} tickers...")
    cutoff = datetime.now() - timedelta(days=days_back)

    for t in tqdm(tickers, desc="Insider trades"):
        try:
            # SEC EDGAR full-text search for insider filings
            url = f"https://efts.sec.gov/LATEST/search-index?q=%22{t}%22&dateRange=custom&startdt={cutoff.strftime('%Y-%m-%d')}&enddt={datetime.now().strftime('%Y-%m-%d')}&forms=4"
            resp = requests.get(url, headers=headers, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                buy_count = 0
                sell_count = 0
                for hit in hits:
                    source = hit.get("_source", {})
                    # Simplified: count form 4 filings
                    buy_count += 1  # We'd parse XML for P/S, simplified here

                records.append({
                    "Ticker": t,
                    "Insider_Filings_90d": len(hits),
                    "Has_Insider_Activity": 1 if len(hits) > 0 else 0,
                })
        except Exception:
            records.append({
                "Ticker": t,
                "Insider_Filings_90d": 0,
                "Has_Insider_Activity": 0,
            })
        time.sleep(0.15)

    return pd.DataFrame(records) if records else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# 6. Merge all external data into main DataFrame
# ═══════════════════════════════════════════════════════════════════════════

def add_market_regime_features(
    prices_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge market regime data (VIX, SPY) into per-stock data by date."""
    df = prices_df.copy()
    regime = regime_df.copy()

    # Normalize both indexes to tz-naive for alignment
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if hasattr(regime.index, 'tz') and regime.index.tz is not None:
        regime.index = regime.index.tz_localize(None)

    regime_cols = [
        "VIX", "VIX_SMA20", "VIX_Percentile", "VIX_Change_5d", "VIX_Spike",
        "SPY_Return_1d", "SPY_Return_5d", "SPY_Above_50SMA", "Market_Regime",
    ]
    available_cols = [c for c in regime_cols if c in regime.columns]

    # Both DataFrames are indexed by Date — use join (much faster than per-date loop)
    df = df.join(regime[available_cols], how="left")

    return df


EXTERNAL_RAW_FEATURES = [
    # Market regime
    "VIX", "VIX_Percentile", "VIX_Change_5d", "VIX_Spike",
    "SPY_Return_5d", "SPY_Above_50SMA",
    # Sector
    "Sector_Return_5d", "Sector_Return_20d", "Sector_RS_Rank",
    "Stock_vs_Sector_5d",
    # Earnings
    "Days_To_Earnings", "Earnings_Within_5d", "Earnings_Within_10d",
    "Days_Since_Earnings",
]
