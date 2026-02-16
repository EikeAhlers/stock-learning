#!/usr/bin/env python3
"""Quick test to diagnose bus error on Le Potato"""
import sys
import traceback

def test_step(name, func):
    try:
        result = func()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        return False

print("=" * 60)
print("Stock Scanner Diagnostic Test")
print("=" * 60)

# Step 1: Config
test_step("Load config", lambda: (
    __import__('json').load(open('config.json'))
    and "config loaded"
))

# Step 2: Tickers
def test_tickers():
    from daily_scanner import get_sp500_tickers
    t = get_sp500_tickers()
    return f"{len(t)} tickers"
test_step("Get S&P 500 tickers", test_tickers)

# Step 3: Small data download
def test_download():
    import yfinance as yf
    data = yf.download("AAPL MSFT", period="5d", group_by='ticker', progress=False)
    return f"shape={data.shape}"
test_step("Download 2 stocks (5d)", test_download)

# Step 4: Feature engineering
def test_features():
    import yfinance as yf
    import pandas as pd
    from daily_scanner import add_features
    data = yf.download("AAPL", period="1y", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    featured = add_features(data)
    return f"features={featured.shape[1]}, rows={len(featured)}"
test_step("Feature engineering", test_features)

# Step 5: XGBoost train (tiny)
def test_xgb():
    import numpy as np
    import xgboost as xgb
    X = np.random.randn(100, 10)
    y = (np.random.randn(100) > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    pred = model.predict_proba(X)
    return f"trained, pred shape={pred.shape}"
test_step("XGBoost train (tiny)", test_xgb)

# Step 6: Paper trader
def test_paper():
    from paper_trader import PaperTrader
    pt = PaperTrader('.')
    return f"capital={pt.portfolio['cash']}"
test_step("Paper trader init", test_paper)

# Step 7: Full scan (small batch)
def test_mini_scan():
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from daily_scanner import add_features
    
    # Download just 10 stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']
    raw = yf.download(tickers, period='2y', group_by='ticker', progress=False, threads=False)
    
    all_data = {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[t].copy()
            else:
                df = raw.copy()
            df = df.dropna()
            if len(df) > 100:
                featured = add_features(df)
                if featured is not None and len(featured) > 50:
                    all_data[t] = featured
        except:
            pass
    
    return f"processed {len(all_data)} stocks"
test_step("Mini scan (10 stocks)", test_mini_scan)

print("\n" + "=" * 60)
print("All basic tests complete!")
print("=" * 60)
