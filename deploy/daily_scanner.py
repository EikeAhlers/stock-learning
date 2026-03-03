#!/usr/bin/env python3
"""
Daily Stock Scanner — Automated daily scan + Telegram notification.

This is the main script that runs on the Le Potato via cron.
It downloads fresh data, trains/loads the model, scores all stocks,
applies filters, and sends the top picks to Telegram.

Usage:
  python3 daily_scanner.py              # Normal daily scan
  python3 daily_scanner.py --manual     # Print results without Telegram
  python3 daily_scanner.py --retrain    # Force model retrain
  python3 daily_scanner.py --force       # Run even on weekends (for testing)
  python3 daily_scanner.py --status     # Show portfolio status
"""
import os
import sys
import gc
import json
import time
import pickle
import argparse
import warnings
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Disable CUDA/NCCL — prevents Bus error on ARM boards without GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NCCL_DEBUG"] = "WARN"
os.environ.setdefault("XGB_CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Setup paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR  # deploy/ is the project root on Le Potato
sys.path.insert(0, PROJECT_DIR)

LOG_DIR = os.path.join(PROJECT_DIR, "logs")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

from telegram_notifier import send_telegram, format_daily_picks, format_portfolio_update
from paper_trader import (
    log_picks, load_portfolio, save_portfolio,
    open_trade, close_trade, update_positions,
    get_performance_summary, format_performance_message,
)
from alpaca_trader import AlpacaTrader


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def load_config():
    cfg_path = os.path.join(SCRIPT_DIR, "config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    log_file = os.path.join(LOG_DIR, f"scanner_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, "a") as f:
        f.write(line + "\n")


def log_analytics(date_str, data):
    """Append a structured JSON record to the analytics log.
    
    This builds a cumulative JSONL file (one JSON object per line per day)
    so we can later load it with pd.read_json(lines=True) and analyze
    model drift, pick quality, prob distributions, feature shifts, etc.
    """
    analytics_path = os.path.join(LOG_DIR, "analytics.jsonl")
    record = {"date": date_str, "timestamp": datetime.now().isoformat(), **data}
    with open(analytics_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD  (streamlined for ARM — no tqdm, minimal memory)
# ═══════════════════════════════════════════════════════════════════════════

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia or fallback."""
    try:
        import io
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        table = pd.read_html(io.StringIO(resp.text))
        tickers = table[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        log(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return sorted(tickers)
    except Exception as e:
        log(f"Wikipedia failed ({e}), using cached/fallback")
        cache = os.path.join(DATA_DIR, "sp500_tickers.json")
        if os.path.exists(cache):
            with open(cache, "r") as f:
                return json.load(f)
        # Minimal fallback
        return sorted([
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
            "JPM", "V", "UNH", "MA", "HD", "PG", "JNJ", "ABBV", "MRK", "LLY",
            "COST", "PEP", "KO", "WMT", "MCD", "NKE", "DIS", "NFLX", "CRM",
            "ORCL", "AMD", "INTC", "QCOM", "TXN", "AMAT", "LRCX", "KLAC",
            "GS", "MS", "BAC", "C", "WFC", "BLK", "SCHW", "AXP",
            "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX",
            "NEE", "DUK", "SO", "AEP", "NRG", "CEG", "VST",
            "UPS", "FDX", "LMT", "GD", "RTX", "BA", "CAT", "DE",
        ])


def download_fresh_data(tickers, period="2y"):
    """Download OHLCV data for all tickers. Optimized for ARM."""
    log(f"Downloading {period} data for {len(tickers)} tickers...")
    
    all_dfs = []
    batch_size = 10  # Smaller batches for 2GB ARM boards
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_str = " ".join(batch)
        
        try:
            data = yf.download(batch_str, period=period, progress=False, threads=False)
            if data.empty:
                continue
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        tk_data = pd.DataFrame({
                            "Open": data[("Open", ticker)],
                            "High": data[("High", ticker)],
                            "Low": data[("Low", ticker)],
                            "Close": data[("Close", ticker)],
                            "Volume": data[("Volume", ticker)],
                        })
                        tk_data["Ticker"] = ticker
                        tk_data.index.name = "Date"
                        if hasattr(tk_data.index, "tz") and tk_data.index.tz is not None:
                            tk_data.index = tk_data.index.tz_localize(None)
                        tk_data = tk_data.dropna(subset=["Close"])
                        if len(tk_data) > 50:
                            all_dfs.append(tk_data)
                    except:
                        pass
            elif len(batch) == 1:
                # Single ticker — no MultiIndex
                data = data.copy()
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                data["Ticker"] = batch[0]
                data.index.name = "Date"
                if hasattr(data.index, "tz") and data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                data = data.dropna(subset=["Close"])
                if len(data) > 50:
                    all_dfs.append(data)
            
            # Free memory every batch on ARM
            del data
            gc.collect()
            
            if (i // batch_size) % 5 == 0:
                log(f"  Downloaded {min(i+batch_size, len(tickers))}/{len(tickers)} tickers")
            time.sleep(0.3)  # Rate limit
            
        except Exception as e:
            log(f"  Batch error at {i}: {e}")
            time.sleep(2)
    
    if not all_dfs:
        raise ValueError("No data downloaded!")
    
    df = pd.concat(all_dfs)
    gc.collect()
    log(f"Downloaded {len(df):,} stock-days for {df['Ticker'].nunique()} tickers")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (self-contained, no src/ dependency)
# ═══════════════════════════════════════════════════════════════════════════

def add_features(df):
    """Add all features needed for the model. Self-contained."""
    log("Engineering features...")
    
    df = df.sort_values(["Ticker", df.index.name or "Date"])
    out_dfs = []
    
    for ticker, grp in df.groupby("Ticker"):
        g = grp.copy().sort_index()
        c = g["Close"]
        v = g["Volume"]
        h = g["High"]
        l = g["Low"]
        
        # Daily return
        g["Daily_Return_Pct"] = c.pct_change() * 100
        
        # Volume features
        g["Vol_Ratio"] = v / v.rolling(20).mean()
        g["Vol_Trend_5d"] = v.rolling(5).mean() / v.rolling(20).mean()
        g["Vol_Compression"] = v.rolling(5).std() / v.rolling(20).std()
        
        # Price features
        g["Price_Position_20d"] = (c - c.rolling(20).min()) / (c.rolling(20).max() - c.rolling(20).min())
        g["RSI_14"] = _compute_rsi(c, 14)
        g["Return_1d"] = c.pct_change() * 100
        g["Return_5d"] = c.pct_change(5) * 100
        g["Return_20d"] = c.pct_change(20) * 100
        g["Return_60d"] = c.pct_change(60) * 100
        
        # ATR
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        g["ATR_14"] = tr.rolling(14).mean()
        g["ATR_Ratio"] = g["ATR_14"] / c
        
        # Bollinger
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        g["BB_Position"] = (c - sma20) / (2 * std20)
        g["BB_Width"] = (4 * std20) / sma20
        
        # MACD
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        g["MACD"] = ema12 - ema26
        g["MACD_Signal"] = g["MACD"].ewm(span=9).mean()
        g["MACD_Hist"] = g["MACD"] - g["MACD_Signal"]
        
        # Moving average features
        g["SMA_50_Dist"] = (c / c.rolling(50).mean() - 1) * 100
        g["SMA_200_Dist"] = (c / c.rolling(200).mean() - 1) * 100
        g["EMA_9_Dist"] = (c / c.ewm(span=9).mean() - 1) * 100
        
        # Stochastic
        low14 = l.rolling(14).min()
        high14 = h.rolling(14).max()
        g["Stoch_K"] = 100 * (c - low14) / (high14 - low14)
        g["Stoch_D"] = g["Stoch_K"].rolling(3).mean()
        
        # Volume-price
        g["OBV_Slope"] = (v * np.sign(c.diff())).rolling(10).sum() / v.rolling(10).sum()
        g["VWAP_Dist"] = (c / ((v * c).rolling(20).sum() / v.rolling(20).sum()) - 1) * 100
        
        # Range/volatility
        g["Intraday_Range"] = (h - l) / c * 100
        g["Range_Ratio"] = g["Intraday_Range"] / g["Intraday_Range"].rolling(20).mean()
        g["Volatility_20d"] = g["Daily_Return_Pct"].rolling(20).std()
        g["Volatility_Ratio"] = g["Volatility_20d"] / g["Daily_Return_Pct"].rolling(60).std()
        
        # Gap
        g["Gap_Pct"] = (g["Open"] / c.shift(1) - 1) * 100 if "Open" in g.columns else 0
        
        # Candle patterns
        body = (c - g["Open"]).abs() if "Open" in g.columns else pd.Series(0, index=g.index)
        g["Body_Range_Ratio"] = body / (h - l).replace(0, np.nan)
        
        # Consecutive moves
        g["Consecutive_Up"] = _consecutive(g["Daily_Return_Pct"] > 0)
        g["Consecutive_Down"] = _consecutive(g["Daily_Return_Pct"] < 0)
        
        # Sector-relative (placeholder — will be 0)
        g["Sector_RS_5d"] = 0
        g["Sector_RS_20d"] = 0
        
        # High/low distance
        g["Dist_52w_High"] = (c / c.rolling(252).max() - 1) * 100
        g["Dist_52w_Low"] = (c / c.rolling(252).min() - 1) * 100
        
        out_dfs.append(g)
    
    result = pd.concat(out_dfs)
    
    # Add "Prev_" prefix (lagged by 1 day to avoid look-ahead)
    feature_cols = [
        "Vol_Ratio", "Vol_Trend_5d", "Price_Position_20d", "Vol_Compression",
        "RSI_14", "Return_1d", "Return_5d", "Return_20d", "Return_60d",
        "ATR_14", "ATR_Ratio", "BB_Position", "BB_Width",
        "MACD", "MACD_Signal", "MACD_Hist",
        "SMA_50_Dist", "SMA_200_Dist", "EMA_9_Dist",
        "Stoch_K", "Stoch_D", "OBV_Slope", "VWAP_Dist",
        "Intraday_Range", "Range_Ratio", "Volatility_20d", "Volatility_Ratio",
        "Gap_Pct", "Body_Range_Ratio", "Consecutive_Up", "Consecutive_Down",
        "Sector_RS_5d", "Sector_RS_20d", "Dist_52w_High", "Dist_52w_Low",
    ]
    
    for col in feature_cols:
        if col in result.columns:
            result[f"Prev_{col}"] = result.groupby("Ticker")[col].shift(1)
    
    features = [f"Prev_{c}" for c in feature_cols if f"Prev_{c}" in result.columns]
    log(f"Engineered {len(features)} features")
    return result, features


def _compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _consecutive(mask):
    """Count consecutive True values."""
    result = mask.copy().astype(int)
    for i in range(1, len(result)):
        if result.iloc[i] == 1:
            result.iloc[i] = result.iloc[i-1] + 1
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def train_model(df, features, big_move_threshold=5.0, hold_days=5):
    """Train XGBoost classifier + magnitude regressor for EV-ranking."""
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
        XGBRegressor = None
    from sklearn.preprocessing import StandardScaler
    
    df = df.copy()
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= big_move_threshold).astype(int)
    
    # Forward N-day return for magnitude prediction (what we actually care about)
    df["Fwd_Return"] = df.groupby("Ticker")["Daily_Return_Pct"].transform(
        lambda x: x.shift(-1).rolling(hold_days, min_periods=1).sum()
    )
    
    # Fill NaN features with 0 (long-lookback features like SMA_200 have many NaNs)
    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(0)
    model_df = df[features + ["Is_Big_Mover", "Fwd_Return"]].dropna(subset=["Is_Big_Mover", "Fwd_Return"])
    
    if len(model_df) < 500:
        log(f"Not enough data for training ({len(model_df)} rows)")
        return None
    
    X = model_df[features]
    y_cls = model_df["Is_Big_Mover"]
    y_reg = model_df["Fwd_Return"]
    
    pos_count = y_cls.sum()
    neg_count = len(y_cls) - pos_count
    scale_pos = neg_count / max(pos_count, 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- Classifier (probability of big move) ---
    clf = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=42,
        verbosity=0, tree_method="hist", device="cpu",
    )
    
    try:
        clf.fit(X_scaled, y_cls)
    except Exception as e:
        log(f"XGBoost fit error: {e}, trying fallback params...")
        clf = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, random_state=42,
            verbosity=0, tree_method="hist", device="cpu",
        )
        clf.fit(X_scaled, y_cls)
    
    # --- Magnitude regressor (expected forward return) ---
    reg = None
    if XGBRegressor is not None:
        try:
            reg = XGBRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, tree_method="hist", device="cpu",
            )
            reg.fit(X_scaled, y_reg)
            log(f"Magnitude regressor trained (median fwd return: {y_reg.median():+.2f}%)")
        except Exception as e:
            log(f"Regressor failed ({e}), will use prob-only ranking")
            reg = None
    
    # AUC estimate (last 20% as validation)
    n_val = int(len(X_scaled) * 0.2)
    val_pred = clf.predict_proba(X_scaled[-n_val:])[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_cls.iloc[-n_val:], val_pred)
    
    log(f"Model trained: {len(model_df):,} rows, AUC={auc:.3f}, pos_rate={pos_count/len(model_df)*100:.1f}%")
    
    return {
        "model": clf, "regressor": reg, "scaler": scaler, "features": features,
        "auc": auc, "train_rows": len(model_df),
        "trained_date": datetime.now().isoformat(),
    }


def save_model(model_dict, name="latest"):
    """Save model to disk."""
    path = os.path.join(MODEL_DIR, f"model_{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model_dict, f)
    log(f"Model saved → {path}")
    return path


def load_model(name="latest"):
    """Load model from disk."""
    path = os.path.join(MODEL_DIR, f"model_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            model_dict = pickle.load(f)
        log(f"Model loaded ← {path}")
        return model_dict
    return None


def needs_retrain(model_dict, retrain_every=63):
    """Check if model needs retraining."""
    if model_dict is None:
        return True
    trained = model_dict.get("trained_date", "")
    if not trained:
        return True
    try:
        trained_dt = datetime.fromisoformat(trained)
        days_old = (datetime.now() - trained_dt).days
        return days_old >= retrain_every
    except:
        return True


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════════════

def run_scan(force_retrain=False, manual=False, force_run=False):
    """Run the full daily scan pipeline."""
    start = time.time()
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    dow = today.weekday()
    
    log("=" * 70)
    log(f"DAILY SCAN — {today_str} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]})")
    log("=" * 70)
    
    # Weekend check
    if dow >= 5 and not force_run:
        log("Weekend — no scan needed. Use --force to override.")
        return
    
    cfg = load_config()
    strat = cfg["strategy"]
    
    # ── Step 1: Download data ─────────────────────────────────────────
    log("Step 1: Downloading fresh market data...")
    tickers = get_sp500_tickers()
    
    # Cache tickers
    with open(os.path.join(DATA_DIR, "sp500_tickers.json"), "w") as f:
        json.dump(tickers, f)
    
    df = download_fresh_data(tickers, period="2y")
    
    # ── Step 2: Feature engineering ───────────────────────────────────
    log("Step 2: Engineering features...")
    df, features = add_features(df)
    gc.collect()
    
    # ── Step 3: Train / load model ────────────────────────────────────
    log("Step 3: Model management...")
    model_dict = load_model()
    
    if force_retrain or needs_retrain(model_dict, strat.get("retrain_every_days", 63)):
        log("Training fresh model...")
        # Train on all data up to yesterday (exclude today)
        yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        train_data = df[df.index < yesterday]
        model_dict = train_model(
            train_data, features,
            big_move_threshold=strat.get("big_move_threshold", 5.0),
            hold_days=strat.get("hold_days", 5),
        )
        
        if model_dict is None:
            log("ERROR: Model training failed!")
            send_telegram("❌ Model training failed! Check logs.")
            return
        
        save_model(model_dict)
        model_dict["features"] = features  # ensure features are current
    else:
        log(f"Using existing model (trained: {model_dict.get('trained_date', 'unknown')})")
        # Update features to match current data
        model_dict["features"] = features
    
    # ── Step 4: Score today's stocks ──────────────────────────────────
    log("Step 4: Scoring stocks...")
    last_date = df.index.max()
    today_df = df[df.index == last_date].copy()
    
    if len(today_df) == 0:
        log("No data for today!")
        return
    
    log(f"Scoring {len(today_df)} stocks for {last_date.date()}")
    
    # Align features — use only features that exist in both model and data
    model_features = model_dict["features"]
    available = [f for f in model_features if f in today_df.columns]
    
    if len(available) < len(model_features) * 0.8:
        log(f"WARNING: Only {len(available)}/{len(model_features)} features available!")
    
    feat_ok = today_df[available].notna().all(axis=1)
    scoreable = today_df[feat_ok]
    
    if len(scoreable) < 50:
        log(f"Only {len(scoreable)} scoreable stocks — too few!")
        return
    
    # Score
    mdl = model_dict["model"]
    scl = model_dict["scaler"]
    
    # Handle feature mismatch
    if len(available) < scl.n_features_in_:
        log("Feature count mismatch — retraining model...")
        train_data = df[df.index < last_date.strftime("%Y-%m-%d")]
        model_dict = train_model(train_data, available, strat.get("big_move_threshold", 5.0))
        if model_dict is None:
            log("Retrain failed!")
            return
        save_model(model_dict)
        mdl = model_dict["model"]
        scl = model_dict["scaler"]
    
    X = scl.transform(scoreable[available].values)
    probs = mdl.predict_proba(X)[:, 1]
    
    # Magnitude prediction (expected forward return)
    reg = model_dict.get("regressor", None)
    if reg is not None:
        try:
            pred_returns = reg.predict(X)
        except Exception as e:
            log(f"Regressor predict failed ({e}), using prob-only")
            pred_returns = probs * 5.0  # fallback: assume 5% if big move
    else:
        pred_returns = probs * 5.0  # fallback
    
    scored = pd.DataFrame({
        "Ticker": scoreable["Ticker"].values,
        "Prob": probs,
        "Pred_Return": pred_returns,
        "Close": scoreable["Close"].values,
    })
    
    # Expected Value = Probability × Predicted Return
    scored["EV"] = scored["Prob"] * scored["Pred_Return"]
    
    # Add momentum
    if "Prev_Return_20d" in scoreable.columns:
        scored["Mom_20d"] = scoreable["Prev_Return_20d"].values
    else:
        scored["Mom_20d"] = 0
    
    # Add volatility for context
    if "Prev_Volatility_20d" in scoreable.columns:
        scored["Volatility"] = scoreable["Prev_Volatility_20d"].values
    else:
        scored["Volatility"] = 0
    
    log(f"Scored {len(scored)} stocks. Prob: {probs.min():.3f}–{probs.max():.3f}, EV: {scored['EV'].min():.2f}–{scored['EV'].max():.2f}")
    
    # ── Step 5: Apply quality filters ─────────────────────────────────
    log("Step 5: Applying quality filters...")
    min_prob = strat.get("min_prob", 0.50)
    max_prob = strat.get("max_prob", 0.85)
    top_n = strat.get("top_n", 3)
    
    # Backtest-proven best strategy: prob-only ranking, 3 picks
    # EV-ranking looked good in theory but underperforms by ~30% in backtest
    filtered = scored[
        (scored["Prob"] >= min_prob) &
        (scored["Prob"] <= max_prob) &
        (scored["Mom_20d"] > 0)  # Only filter: positive 20d momentum
    ].sort_values("Prob", ascending=False)  # Rank by probability (backtest-proven best)
    
    # Skip Wednesday if configured
    if strat.get("skip_wednesday", False) and dow == 2:
        log("Wednesday — skipping signals per config")
        filtered = filtered.head(0)
    
    # Take top N picks by probability
    picks = filtered.head(top_n)
    
    log(f"After filters: {len(filtered)} stocks pass, {len(picks)} selected (prob≥{min_prob}, mom>0%)")
    
    # ── Step 6: Format picks ──────────────────────────────────────────
    picks_list = []
    for _, row in picks.iterrows():
        picks_list.append({
            "ticker": row["Ticker"],
            "prob": float(row["Prob"]),
            "close": float(row["Close"]),
            "momentum_20d": float(row["Mom_20d"]),
            "pred_return": float(row["Pred_Return"]),
            "ev": float(row["EV"]),
            "sector": "",
        })
    
    # ── Step 7: Log picks ─────────────────────────────────────────────
    model_info = {
        "auc": model_dict.get("auc", 0),
        "train_rows": model_dict.get("train_rows", 0),
        "hold_days": strat.get("hold_days", 5),
        "trained_date": model_dict.get("trained_date", ""),
    }
    
    log_picks(today_str, picks_list, model_info)
    
    # ── Step 8: Comprehensive analytics logging ───────────────────────
    log("Step 8: Logging analytics for future analysis...")
    
    # Probability distribution stats
    prob_stats = {
        "mean": float(probs.mean()), "median": float(np.median(probs)),
        "std": float(probs.std()), "min": float(probs.min()), "max": float(probs.max()),
        "p25": float(np.percentile(probs, 25)), "p75": float(np.percentile(probs, 75)),
        "p90": float(np.percentile(probs, 90)), "p95": float(np.percentile(probs, 95)),
        "above_50": int((probs >= 0.50).sum()), "above_70": int((probs >= 0.70).sum()),
        "above_80": int((probs >= 0.80).sum()), "above_85": int((probs >= 0.85).sum()),
    }
    
    # Feature distribution snapshot (top features for drift detection)
    feature_stats = {}
    key_features = ["Prev_ATR_Ratio", "Prev_RSI_14", "Prev_Vol_Ratio",
                    "Prev_Return_20d", "Prev_BB_Position", "Prev_Volatility_20d"]
    for feat in key_features:
        if feat in scoreable.columns:
            vals = scoreable[feat].dropna()
            if len(vals) > 0:
                feature_stats[feat] = {
                    "mean": float(vals.mean()), "std": float(vals.std()),
                    "median": float(vals.median()),
                    "min": float(vals.min()), "max": float(vals.max()),
                }
    
    # Market breadth indicators
    if "Prev_Return_1d" in scoreable.columns:
        r1d = scoreable["Prev_Return_1d"].dropna()
        breadth = {
            "pct_positive": float((r1d > 0).mean() * 100),
            "avg_return_1d": float(r1d.mean()),
            "avg_return_20d": float(scoreable["Prev_Return_20d"].dropna().mean()) if "Prev_Return_20d" in scoreable.columns else None,
            "avg_volatility": float(scoreable["Prev_Volatility_20d"].dropna().mean()) if "Prev_Volatility_20d" in scoreable.columns else None,
        }
    else:
        breadth = {}
    
    # Decision audit trail
    decisions = {
        "total_scored": len(scored),
        "pass_prob_filter": int(((scored["Prob"] >= min_prob) & (scored["Prob"] <= max_prob)).sum()),
        "pass_momentum": int((scored["Mom_20d"] > 0).sum()),
        "pass_all_filters": len(filtered),
        "picks_selected": len(picks_list),
        "filters_used": {"min_prob": min_prob, "max_prob": max_prob, "ranking": "probability"},
    }
    
    # Full top 20 for analysis (did we miss good ones?)
    top20_snapshot = []
    for _, row in filtered.head(20).iterrows():
        top20_snapshot.append({
            "ticker": row["Ticker"], "prob": round(float(row["Prob"]), 4),
            "close": round(float(row["Close"]), 2),
            "momentum_20d": round(float(row["Mom_20d"]), 2),
        })
    
    # Write the full analytics record
    log_analytics(today_str, {
        "scan_date": last_date.strftime("%Y-%m-%d"),
        "day_of_week": ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow],
        "model": {
            "auc": model_dict.get("auc", 0),
            "train_rows": model_dict.get("train_rows", 0),
            "trained_date": model_dict.get("trained_date", ""),
            "n_features": len(available),
            "retrained_today": force_retrain or needs_retrain(load_model(), strat.get("retrain_every_days", 63)),
        },
        "scoring": {
            "n_tickers_downloaded": df["Ticker"].nunique() if "Ticker" in df.columns else 0,
            "n_scoreable": len(scoreable),
            "n_data_rows": len(df),
        },
        "probability_distribution": prob_stats,
        "feature_distributions": feature_stats,
        "market_breadth": breadth,
        "decisions": decisions,
        "picks": picks_list,
        "top20": top20_snapshot,
        "runtime_seconds": round(time.time() - start, 1),
    })
    
    # ── Step 9: Update paper trading ──────────────────────────────────
    log("Step 9: Updating paper portfolio...")
    portfolio = load_portfolio()
    
    # Get current prices for open positions
    held_tickers = [p["ticker"] for p in portfolio["positions"]]
    if held_tickers:
        current_prices = {}
        for p in portfolio["positions"]:
            tk = p["ticker"]
            row = scored[scored["Ticker"] == tk]
            if len(row) > 0:
                current_prices[tk] = float(row["Close"].iloc[0])
            else:
                # Try to get price
                try:
                    px = yf.download(tk, period="1d", progress=False)
                    if len(px) > 0:
                        current_prices[tk] = float(px["Close"].iloc[-1])
                except:
                    pass
        
        portfolio, stopped = update_positions(current_prices, today_str)
        
        # Log stopped-out trades for analytics
        if stopped:
            for s_ticker in stopped:
                log(f"  STOP-LOSS triggered: {s_ticker}")
        
        # Check hold period expiry
        for pos in list(portfolio["positions"]):
            if pos.get("days_held", 0) >= strat.get("hold_days", 5):
                tk = pos["ticker"]
                if tk in current_prices:
                    exit_price = current_prices[tk]
                    entry_price = pos["entry_price"]
                    pnl_pct = (exit_price / entry_price - 1) * 100
                    log(f"  HOLD EXPIRY: {tk} entry=${entry_price:.2f} exit=${exit_price:.2f} P&L={pnl_pct:+.1f}%")
                    close_trade(tk, today_str, exit_price, "hold_expiry")
                    
                    # Log trade outcome for analytics
                    log_analytics(today_str, {
                        "event": "trade_closed",
                        "ticker": tk,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": round(pnl_pct, 2),
                        "entry_date": pos.get("entry_date", ""),
                        "exit_reason": "hold_expiry",
                        "prob_at_entry": pos.get("prob", 0),
                        "days_held": pos.get("days_held", 0),
                    })
    
    # Open new positions for picks
    portfolio = load_portfolio()
    max_pos = strat.get("max_positions", 3)
    open_slots = max_pos - len(portfolio["positions"])
    sizing_method = strat.get("sizing_method", "equal")
    
    if open_slots > 0 and len(picks_list) > 0:
        # Calculate total equity for sizing
        total_val = portfolio["cash"] + sum(
            p.get("current_value", p["shares"] * p["entry_price"])
            for p in portfolio["positions"]
        )
        slot_size = total_val / max_pos
        
        # Tiered: #1 pick gets 70%, #2 gets 30% of their slots
        tiers = [0.70, 0.30] if sizing_method == "tiered" else [0.50, 0.50]
        eligible = []
        for pick in picks_list[:open_slots]:
            tk = pick["ticker"]
            if tk not in [p["ticker"] for p in portfolio["positions"]]:
                eligible.append(pick)
        
        n_buy = len(eligible)
        if n_buy > 0:
            tier_weights = tiers[:n_buy]
            scale = n_buy / sum(tier_weights) if sum(tier_weights) > 0 else 1
        
        for i, pick in enumerate(eligible):
            tk = pick["ticker"]
            tier = tiers[i] if i < len(tiers) else 0.30
            pos_val = min(portfolio["cash"], slot_size * tier * scale)
            shares = int(pos_val / (pick["close"] * 1.0015))
            
            if shares > 0 and portfolio["cash"] > shares * pick["close"] * 1.0015:
                open_trade(
                    tk, today_str, pick["close"],
                    pick["prob"], shares, "daily_scan"
                )
                log_analytics(today_str, {
                    "event": "trade_opened",
                    "ticker": tk,
                    "entry_price": pick["close"],
                    "shares": shares,
                    "prob": pick["prob"],
                    "momentum_20d": pick["momentum_20d"],
                    "position_value": round(shares * pick["close"], 2),
                })
    
    # ── Step 9: Send Telegram ─────────────────────────────────────────
    elapsed = time.time() - start
    log(f"Scan complete in {elapsed:.0f}s")
    
    # Reload portfolio for accurate state
    portfolio = load_portfolio()
    hold_days = strat.get("hold_days", 5)
    
    msg = format_daily_picks(
        picks_list, today_str, model_info,
        portfolio=portfolio, hold_days=hold_days,
    )
    
    # ── Step 10: Alpaca paper trading ─────────────────────────────────
    log("Step 10: Alpaca paper trading...")
    try:
        alpaca = AlpacaTrader()
        if alpaca.enabled:
            # Sell expired positions first
            sold = alpaca.sell_expired()
            if sold:
                for s in sold:
                    log(f"  Alpaca sold {s['ticker']}: {s['reason']} ({s['pnl']:+.1f}%)")
            
            # Buy new picks
            if picks_list:
                result = alpaca.execute_picks(picks_list, max_pos)
                alpaca.record_buys(picks_list)
                for b in result["bought"]:
                    log(f"  Alpaca bought {b['ticker']}: {b['qty']} shares @ ${b['price']:.2f}")
            
            # Ensure all positions have stop-loss orders on Alpaca
            alpaca.ensure_stop_losses()
            
            # Add Alpaca summary to message
            alpaca_summary = alpaca.get_summary()
            if alpaca_summary:
                msg += "\n" + alpaca_summary
        else:
            log("  Alpaca trading disabled in config")
    except Exception as e:
        log(f"  Alpaca error (non-fatal): {e}")
    
    if manual:
        print("\n" + msg.replace("<b>", "").replace("</b>", ""))
    else:
        send_telegram(msg)
    
    # ── Step 10: Also show full top 20 in logs ────────────────────────
    log("\nFull Top 20 (by Probability):")
    for i, (_, row) in enumerate(filtered.head(20).iterrows()):
        log(f"  #{i+1:2d}  {row['Ticker']:<6s}  P={row['Prob']:.2f}  EV={row['EV']:.2f}  PredRet={row['Pred_Return']:+.1f}%  Mom={row['Mom_20d']:+.1f}%")
    
    log("\n" + "=" * 70)
    log("SCAN COMPLETE")
    log("=" * 70)


def show_status():
    """Show current portfolio status and send weekly analytics summary."""
    perf = get_performance_summary()
    msg = format_performance_message()
    
    # Plain text version
    plain = msg.replace("<b>", "").replace("</b>", "")
    print(plain)
    
    # Weekly analytics summary
    analytics_path = os.path.join(LOG_DIR, "analytics.jsonl")
    if os.path.exists(analytics_path):
        try:
            import pandas as _pd
            records = []
            with open(analytics_path) as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except:
                        pass
            
            scans = [r for r in records if "probability_distribution" in r]
            trades = [r for r in records if r.get("event") == "trade_closed"]
            
            if scans:
                recent = scans[-5:]  # last 5 scans
                avg_prob_mean = np.mean([s["probability_distribution"]["mean"] for s in recent])
                avg_above80 = np.mean([s["probability_distribution"]["above_80"] for s in recent])
                avg_runtime = np.mean([s.get("runtime_seconds", 0) for s in recent])
                
                analytics_msg = f"\n\n📈 <b>Weekly Analytics</b>"
                analytics_msg += f"\nScans logged: {len(scans)} total"
                analytics_msg += f"\nAvg prob mean (last 5): {avg_prob_mean:.3f}"
                analytics_msg += f"\nAvg stocks >80% (last 5): {avg_above80:.0f}"
                analytics_msg += f"\nAvg runtime: {avg_runtime:.0f}s"
                
                if trades:
                    pnls = [t["pnl_pct"] for t in trades]
                    analytics_msg += f"\n\nTrades closed: {len(trades)}"
                    analytics_msg += f"\nAvg P&L: {np.mean(pnls):+.2f}%"
                    analytics_msg += f"\nWin rate: {sum(1 for p in pnls if p > 0)/len(pnls)*100:.0f}%"
                    analytics_msg += f"\nBest: {max(pnls):+.2f}% | Worst: {min(pnls):+.2f}%"
                
                msg += analytics_msg
        except Exception as e:
            log(f"Analytics summary error: {e}")
    
    # Send to Telegram
    send_telegram(msg)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Stock Scanner")
    parser.add_argument("--manual", action="store_true", help="Print results, don't send Telegram")
    parser.add_argument("--retrain", action="store_true", help="Force model retrain")
    parser.add_argument("--force", action="store_true", help="Run even on weekends (for testing)")
    parser.add_argument("--status", action="store_true", help="Show portfolio status")
    args = parser.parse_args()
    
    try:
        if args.status:
            show_status()
        else:
            run_scan(force_retrain=args.retrain, manual=args.manual, force_run=args.force)
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())
        try:
            send_telegram(f"❌ Scanner CRASHED:\n{str(e)[:500]}")
        except:
            pass
        sys.exit(1)
