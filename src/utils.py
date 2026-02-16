"""
Shared utilities for the stock mover analysis framework.
"""
import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def save_dataframe(df: pd.DataFrame, name: str) -> str:
    """Save a DataFrame to the data directory as CSV."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df.to_csv(path, index=True)
    print(f"Saved {len(df)} rows to {path}")
    return path


def load_dataframe(name: str) -> pd.DataFrame:
    """Load a DataFrame from the data directory."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached data found at {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def cache_exists(name: str) -> bool:
    """Check if cached data exists."""
    return os.path.exists(os.path.join(DATA_DIR, f"{name}.csv"))


def get_sp500_tickers() -> list:
    """Get current S&P 500 tickers. Tries Wikipedia, falls back to hardcoded list."""
    # Try Wikipedia with headers to avoid 403
    try:
        import io
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=10
        )
        resp.raise_for_status()
        table = pd.read_html(io.StringIO(resp.text))
        tickers = table[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return sorted(tickers)
    except Exception as e:
        print(f"Wikipedia fetch failed ({e}), using expanded fallback list")
    
    # Expanded fallback: ~150 major US stocks across all sectors
    return sorted([
        # Tech
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO",
        "CSCO", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "AMAT", "NOW",
        "IBM", "INTU", "ISRG", "NFLX", "PYPL", "SQ", "SHOP", "SNOW", "PLTR",
        "MU", "MRVL", "KLAC", "LRCX", "SNPS", "CDNS", "PANW", "CRWD", "NET",
        "DDOG", "ZS", "FTNT", "UBER", "ABNB", "DASH", "COIN", "MELI",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA",
        "C", "USB", "PNC", "TFC", "COF", "BK", "CME", "ICE", "MCO", "SPGI",
        # Healthcare
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "REGN", "VRTX", "MRNA", "BIIB", "ILMN", "MDT", "SYK",
        "BSX", "EW", "ZTS", "DXCM", "ALGN",
        # Consumer
        "PG", "KO", "PEP", "WMT", "COST", "HD", "LOW", "TGT", "NKE", "SBUX",
        "MCD", "DIS", "CMCSA", "BKNG", "MAR", "YUM", "EL", "CL", "MNST",
        # Industrial
        "CAT", "DE", "HON", "UPS", "RTX", "BA", "LMT", "GE", "MMM", "EMR",
        "ETN", "ITW", "FDX", "NSC", "UNP", "WM",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
        # Real Estate / Utilities
        "AMT", "PLD", "CCI", "EQIX", "NEE", "DUK", "SO", "D", "AEP", "SRE",
        # Other
        "BRK-B", "ACN", "ADP", "FIS", "FISV", "LULU", "ROST", "TJX", "ORLY",
        "AZO", "ODFL", "CTAS", "CPRT", "FAST", "PAYX",
    ])


def format_pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value:+.2f}%"


def trading_days_back(days: int) -> str:
    """Get the date string for N trading days ago (approximate)."""
    # Rough approximation: 252 trading days per year
    calendar_days = int(days * 365 / 252)
    dt = datetime.now() - timedelta(days=calendar_days)
    return dt.strftime("%Y-%m-%d")
