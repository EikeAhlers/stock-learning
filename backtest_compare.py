"""
Backtest: Old (prob-only) vs New (EV-ranked) strategy
Runs an honest walk-forward backtest comparing both approaches.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import time
import gc
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

def download_data(tickers, period="2y"):
    """Download OHLCV for a list of tickers."""
    all_dfs = []
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(" ".join(batch), period=period, progress=False, threads=True)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                for tk in batch:
                    try:
                        tk_data = pd.DataFrame({
                            "Open": data[("Open", tk)],
                            "High": data[("High", tk)],
                            "Low": data[("Low", tk)],
                            "Close": data[("Close", tk)],
                            "Volume": data[("Volume", tk)],
                        })
                        tk_data["Ticker"] = tk
                        tk_data.index.name = "Date"
                        if hasattr(tk_data.index, "tz") and tk_data.index.tz is not None:
                            tk_data.index = tk_data.index.tz_localize(None)
                        tk_data = tk_data.dropna(subset=["Close"])
                        if len(tk_data) > 50:
                            all_dfs.append(tk_data)
                    except:
                        pass
            time.sleep(0.2)
        except:
            pass
    return pd.concat(all_dfs) if all_dfs else pd.DataFrame()


def add_features(df):
    """Add technical features (same as daily_scanner)."""
    df = df.sort_values(["Ticker", df.index.name or "Date"])
    out_dfs = []
    for ticker, grp in df.groupby("Ticker"):
        g = grp.copy().sort_index()
        c, v, h, l = g["Close"], g["Volume"], g["High"], g["Low"]
        g["Daily_Return_Pct"] = c.pct_change() * 100
        g["Vol_Ratio"] = v / v.rolling(20).mean()
        g["Vol_Trend_5d"] = v.rolling(5).mean() / v.rolling(20).mean()
        g["Vol_Compression"] = v.rolling(5).std() / v.rolling(20).std()
        g["Price_Position_20d"] = (c - c.rolling(20).min()) / (c.rolling(20).max() - c.rolling(20).min())
        delta = c.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
        g["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        g["Return_1d"] = c.pct_change() * 100
        g["Return_5d"] = c.pct_change(5) * 100
        g["Return_20d"] = c.pct_change(20) * 100
        g["Return_60d"] = c.pct_change(60) * 100
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        g["ATR_14"] = tr.rolling(14).mean()
        g["ATR_Ratio"] = g["ATR_14"] / c
        sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
        g["BB_Position"] = (c - sma20) / (2 * std20)
        g["BB_Width"] = (4 * std20) / sma20
        ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
        g["MACD"] = ema12 - ema26; g["MACD_Signal"] = g["MACD"].ewm(span=9).mean()
        g["MACD_Hist"] = g["MACD"] - g["MACD_Signal"]
        g["SMA_50_Dist"] = (c / c.rolling(50).mean() - 1) * 100
        g["SMA_200_Dist"] = (c / c.rolling(200).mean() - 1) * 100
        g["EMA_9_Dist"] = (c / c.ewm(span=9).mean() - 1) * 100
        low14 = l.rolling(14).min(); high14 = h.rolling(14).max()
        g["Stoch_K"] = 100 * (c - low14) / (high14 - low14)
        g["Stoch_D"] = g["Stoch_K"].rolling(3).mean()
        g["OBV_Slope"] = (v * np.sign(c.diff())).rolling(10).sum() / v.rolling(10).sum()
        g["VWAP_Dist"] = (c / ((v * c).rolling(20).sum() / v.rolling(20).sum()) - 1) * 100
        g["Intraday_Range"] = (h - l) / c * 100
        g["Range_Ratio"] = g["Intraday_Range"] / g["Intraday_Range"].rolling(20).mean()
        g["Volatility_20d"] = g["Daily_Return_Pct"].rolling(20).std()
        g["Volatility_Ratio"] = g["Volatility_20d"] / g["Daily_Return_Pct"].rolling(60).std()
        g["Gap_Pct"] = (g["Open"] / c.shift(1) - 1) * 100 if "Open" in g.columns else 0
        body = (c - g["Open"]).abs() if "Open" in g.columns else pd.Series(0, index=g.index)
        g["Body_Range_Ratio"] = body / (h - l).replace(0, np.nan)
        r = (g["Daily_Return_Pct"] > 0).astype(int)
        for i in range(1, len(r)):
            if r.iloc[i] == 1: r.iloc[i] = r.iloc[i-1] + 1
        g["Consecutive_Up"] = r
        r2 = (g["Daily_Return_Pct"] < 0).astype(int)
        for i in range(1, len(r2)):
            if r2.iloc[i] == 1: r2.iloc[i] = r2.iloc[i-1] + 1
        g["Consecutive_Down"] = r2
        g["Sector_RS_5d"] = 0; g["Sector_RS_20d"] = 0
        g["Dist_52w_High"] = (c / c.rolling(252).max() - 1) * 100
        g["Dist_52w_Low"] = (c / c.rolling(252).min() - 1) * 100
        out_dfs.append(g)
    result = pd.concat(out_dfs)
    feature_cols = [
        "Vol_Ratio","Vol_Trend_5d","Price_Position_20d","Vol_Compression",
        "RSI_14","Return_1d","Return_5d","Return_20d","Return_60d",
        "ATR_14","ATR_Ratio","BB_Position","BB_Width",
        "MACD","MACD_Signal","MACD_Hist","SMA_50_Dist","SMA_200_Dist",
        "EMA_9_Dist","Stoch_K","Stoch_D","OBV_Slope","VWAP_Dist",
        "Intraday_Range","Range_Ratio","Volatility_20d","Volatility_Ratio",
        "Gap_Pct","Body_Range_Ratio","Consecutive_Up","Consecutive_Down",
        "Sector_RS_5d","Sector_RS_20d","Dist_52w_High","Dist_52w_Low",
    ]
    for col in feature_cols:
        if col in result.columns:
            result[f"Prev_{col}"] = result.groupby("Ticker")[col].shift(1)
    features = [f"Prev_{c}" for c in feature_cols if f"Prev_{c}" in result.columns]
    return result, features


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════════════════

def train_model(df, features, big_move_threshold=5.0, hold_days=5, with_regressor=False):
    """Train classifier and optionally a magnitude regressor."""
    from xgboost import XGBClassifier, XGBRegressor

    df = df.copy()
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= big_move_threshold).astype(int)
    df["Fwd_Return"] = df.groupby("Ticker")["Daily_Return_Pct"].transform(
        lambda x: x.shift(-1).rolling(hold_days, min_periods=1).sum()
    )
    # Fill NaN features with 0 (long-lookback features like SMA_200 have many NaNs)
    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(0)
    model_df = df[features + ["Is_Big_Mover", "Fwd_Return"]].dropna(subset=["Is_Big_Mover", "Fwd_Return"])
    if len(model_df) < 500:
        return None

    X = model_df[features]
    y_cls = model_df["Is_Big_Mover"]
    y_reg = model_df["Fwd_Return"]
    scale_pos = (len(y_cls) - y_cls.sum()) / max(y_cls.sum(), 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=42,
        verbosity=0, tree_method="hist",
    )
    clf.fit(X_scaled, y_cls)

    reg = None
    if with_regressor:
        reg = XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, tree_method="hist",
        )
        reg.fit(X_scaled, y_reg)

    return {"model": clf, "regressor": reg, "scaler": scaler, "features": features}


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(df, features, strategy="old", hold_days=5, max_picks=3,
                 max_positions=3, min_prob=0.50, min_mom=0.0, min_ev=0.0,
                 stop_loss=-7.0, retrain_every=63, warmup_days=252,
                 start_capital=100000, cost_bps=15):
    """
    Walk-forward backtest.
    strategy: "old" = prob-only, "new" = EV-ranked with quality filters
    """
    use_ev = (strategy == "new")
    dates = sorted(df.index.unique())
    if len(dates) < warmup_days + 30:
        print(f"Not enough dates: {len(dates)}")
        return {}

    trade_dates = dates[warmup_days:]
    print(f"  Strategy: {strategy.upper()}")
    print(f"  Warmup: {dates[0].date()} → {dates[warmup_days-1].date()}")
    print(f"  Trading: {trade_dates[0].date()} → {trade_dates[-1].date()} ({len(trade_dates)} days)")
    print(f"  Params: picks≤{max_picks}, hold={hold_days}d, prob≥{min_prob:.0%}, mom≥{min_mom}%, EV≥{min_ev}")

    # Initial training
    warmup_data = df[df.index <= dates[warmup_days - 1]]
    model = train_model(warmup_data, features, hold_days=hold_days, with_regressor=use_ev)
    if model is None:
        print("  Training failed!")
        return {}
    days_since_retrain = 0

    capital = start_capital
    positions = []
    trade_log = []
    equity_curve = []

    for day_i, today in enumerate(trade_dates):
        # Retrain periodically
        if days_since_retrain >= retrain_every:
            past = df[df.index < today]
            new_m = train_model(past, features, hold_days=hold_days, with_regressor=use_ev)
            if new_m is not None:
                model = new_m
                days_since_retrain = 0
        days_since_retrain += 1

        # Close expired / stopped positions
        closed = []
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            if len(td) == 0:
                pos["days_held"] += 1
                continue
            cp = float(td["Close"].iloc[0])
            ret = (cp / pos["entry_price"] - 1) * 100
            pos["days_held"] += 1

            if ret <= stop_loss or pos["days_held"] >= hold_days:
                reason = "stop_loss" if ret <= stop_loss else "hold_expiry"
                cost = pos["shares"] * cp * (cost_bps / 10000)
                proceeds = pos["shares"] * cp - cost
                capital += proceeds
                net_ret = (proceeds / (pos["shares"] * pos["entry_price"]) - 1) * 100
                trade_log.append({
                    "Ticker": pos["ticker"], "Entry_Date": pos["entry_date"],
                    "Exit_Date": today, "Return_Pct": net_ret,
                    "Hold_Days": pos["days_held"], "Exit_Reason": reason,
                    "Prob": pos.get("prob", 0), "EV": pos.get("ev", 0),
                })
                closed.append(pos)
        for c in closed:
            positions.remove(c)

        # Open new positions
        open_slots = max_positions - len(positions)
        if open_slots > 0:
            today_df = df[df.index == today].copy()
            if len(today_df) > 10:
                avail_feats = [f for f in features if f in today_df.columns]
                fm = today_df[avail_feats].notna().all(axis=1)
                scoreable = today_df[fm]
                if len(scoreable) > 0:
                    X = model["scaler"].transform(scoreable[avail_feats].values)
                    probs = model["model"].predict_proba(X)[:, 1]

                    scores = pd.DataFrame({
                        "Ticker": scoreable["Ticker"].values,
                        "Prob": probs,
                        "Close": scoreable["Close"].values,
                    })
                    if "Prev_Return_20d" in scoreable.columns:
                        scores["Mom_20d"] = scoreable["Prev_Return_20d"].values
                    else:
                        scores["Mom_20d"] = 0.0

                    # EV scoring
                    if use_ev and model["regressor"] is not None:
                        pred_ret = model["regressor"].predict(X)
                        scores["Pred_Return"] = pred_ret
                        scores["EV"] = scores["Prob"] * scores["Pred_Return"]
                    else:
                        scores["Pred_Return"] = 0.0
                        scores["EV"] = scores["Prob"]

                    # Filters
                    mask = (scores["Prob"] >= min_prob) & (scores["Mom_20d"] >= min_mom)
                    if use_ev:
                        mask &= (scores["EV"] >= min_ev) & (scores["Pred_Return"] > 0)
                    scores = scores[mask]

                    # Sort
                    sort_col = "EV" if use_ev else "Prob"
                    scores = scores.sort_values(sort_col, ascending=False)

                    # Skip what we hold
                    held = [p["ticker"] for p in positions]
                    scores = scores[~scores["Ticker"].isin(held)]

                    n_buy = min(open_slots, max_picks, len(scores))
                    for j in range(n_buy):
                        pick = scores.iloc[j]
                        ep = float(pick["Close"])
                        total_eq = capital + sum(p["shares"] * p["entry_price"] for p in positions)
                        pos_val = min(capital, total_eq / max_positions)
                        cost = pos_val * (cost_bps / 10000)
                        shares = int((pos_val - cost) / ep)
                        if shares > 0 and capital >= shares * ep * (1 + cost_bps/10000):
                            capital -= shares * ep * (1 + cost_bps/10000)
                            positions.append({
                                "ticker": pick["Ticker"], "entry_date": today,
                                "entry_price": ep, "shares": shares, "days_held": 0,
                                "prob": float(pick["Prob"]), "ev": float(pick["EV"]),
                            })

        # Portfolio value
        pv = capital
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            pv += pos["shares"] * (float(td["Close"].iloc[0]) if len(td) > 0 else pos["entry_price"])
        equity_curve.append({"Date": today, "Value": pv})

        if day_i % 50 == 0:
            ret_pct = (pv / start_capital - 1) * 100
            print(f"    Day {day_i:4d} | {today.date()} | ${pv:>10,.0f} ({ret_pct:+.1f}%) | Trades: {len(trade_log)}")

    # Close remaining
    for pos in positions:
        last = df[df["Ticker"] == pos["ticker"]]
        if len(last) > 0:
            ep2 = float(last.iloc[-1]["Close"])
            ret2 = (ep2 / pos["entry_price"] - 1) * 100
            capital += pos["shares"] * ep2
            trade_log.append({
                "Ticker": pos["ticker"], "Entry_Date": pos["entry_date"],
                "Exit_Date": trade_dates[-1], "Return_Pct": ret2,
                "Hold_Days": pos["days_held"], "Exit_Reason": "end",
                "Prob": pos.get("prob", 0), "EV": pos.get("ev", 0),
            })

    eq = pd.DataFrame(equity_curve).set_index("Date")
    trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    return {"equity": eq, "trades": trades, "start": start_capital}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    # Use cached data if available (much faster, no rate limits)
    cached_path = os.path.join("data", "universe_prices.csv")
    if os.path.exists(cached_path):
        print("1. Loading cached data...")
        df = pd.read_csv(cached_path, index_col=0, parse_dates=True)
        n_tickers = df['Ticker'].nunique()
        print(f"   {len(df):,} rows, {n_tickers} tickers (from cache)")
    else:
        try:
            import io, requests
            resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                               headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            tickers = pd.read_html(io.StringIO(resp.text))[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        except:
            tickers = [
                "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","JPM","V",
                "UNH","MA","HD","PG","JNJ","ABBV","MRK","LLY","COST","PEP","KO","WMT",
            ]
        tickers = sorted(tickers)
        print(f"\n1. Downloading data for {len(tickers)} tickers...")
        df = download_data(tickers, period="2y")
        print(f"   {len(df):,} rows, {df['Ticker'].nunique()} tickers")

    print("\n2. Engineering features...")
    df, features = add_features(df)
    gc.collect()
    print(f"   {len(features)} features")

    print("\n" + "=" * 80)
    print("  BACKTEST A: OLD STRATEGY (prob-only, forced 3 picks, mom > 0)")
    print("=" * 80)
    old_result = run_backtest(
        df, features, strategy="old",
        hold_days=5, max_picks=3, max_positions=3,
        min_prob=0.50, min_mom=0.0, min_ev=0.0,
        stop_loss=-7.0, start_capital=100000,
    )

    print("\n" + "=" * 80)
    print("  BACKTEST B: NEW STRATEGY (EV-ranked, quality filters)")
    print("=" * 80)
    new_result = run_backtest(
        df, features, strategy="new",
        hold_days=5, max_picks=3, max_positions=3,
        min_prob=0.60, min_mom=5.0, min_ev=1.5,
        stop_loss=-7.0, start_capital=100000,
    )

    # Also test with max 1 pick (user's preference)
    print("\n" + "=" * 80)
    print("  BACKTEST C: NEW STRATEGY — TOP 1 PICK ONLY")
    print("=" * 80)
    new1_result = run_backtest(
        df, features, strategy="new",
        hold_days=5, max_picks=1, max_positions=1,
        min_prob=0.60, min_mom=5.0, min_ev=1.5,
        stop_loss=-7.0, start_capital=100000,
    )

    # Test: EV-ranked but lower momentum bar + up to 3 picks
    print("\n" + "=" * 80)
    print("  BACKTEST D: EV-RANKED, MOM >= 3%, UP TO 3")
    print("=" * 80)
    new_d = run_backtest(
        df, features, strategy="new",
        hold_days=5, max_picks=3, max_positions=3,
        min_prob=0.55, min_mom=3.0, min_ev=1.0,
        stop_loss=-7.0, start_capital=100000,
    )

    # Test: EV-ranked, 1 pick, lower bars
    print("\n" + "=" * 80)
    print("  BACKTEST E: EV-RANKED, 1 PICK, MOM >= 3%")
    print("=" * 80)
    new_e = run_backtest(
        df, features, strategy="new",
        hold_days=5, max_picks=1, max_positions=1,
        min_prob=0.55, min_mom=3.0, min_ev=1.0,
        stop_loss=-7.0, start_capital=100000,
    )

    # Test: Old strat but with only 1 pick
    print("\n" + "=" * 80)
    print("  BACKTEST F: OLD PROB-ONLY, 1 PICK")
    print("=" * 80)
    old1_result = run_backtest(
        df, features, strategy="old",
        hold_days=5, max_picks=1, max_positions=1,
        min_prob=0.50, min_mom=0.0, min_ev=0.0,
        stop_loss=-7.0, start_capital=100000,
    )

    # Summary
    print("\n" + "=" * 80)
    print("  COMPARISON SUMMARY")
    print("=" * 80)

    for label, r in [("A) OLD (prob-only, 3 picks)", old_result),
                     ("B) NEW (EV-ranked, up to 3)", new_result),
                     ("C) NEW (EV, top 1 only)",     new1_result),
                     ("D) NEW (EV, mom>=3%, 3 picks)", new_d),
                     ("E) NEW (EV, mom>=3%, 1 pick)", new_e),
                     ("F) OLD (prob-only, 1 pick)",  old1_result)]:
        eq = r.get("equity", pd.DataFrame())
        trades = r.get("trades", pd.DataFrame())
        sc = r.get("start", 100000)
        if len(eq) == 0:
            print(f"  {label}: NO DATA")
            continue
        fv = eq["Value"].iloc[-1]
        total_ret = (fv / sc - 1) * 100
        daily_ret = eq["Value"].pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
        max_dd = ((eq["Value"] - eq["Value"].cummax()) / eq["Value"].cummax() * 100).min()
        wr = (trades["Return_Pct"] > 0).mean() * 100 if len(trades) > 0 else 0
        avg_ret = trades["Return_Pct"].mean() if len(trades) > 0 else 0
        n_trades = len(trades)

        print(f"\n  {label}")
        print(f"    Return: {total_ret:+.1f}%  |  ${sc:,.0f} → ${fv:,.0f}")
        print(f"    Sharpe: {sharpe:.2f}  |  Max DD: {max_dd:.1f}%")
        print(f"    Win Rate: {wr:.0f}%  |  Avg P&L/trade: {avg_ret:+.2f}%")
        print(f"    Trades: {n_trades}")

    print("\n" + "=" * 80)
