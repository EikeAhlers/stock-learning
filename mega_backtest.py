"""
MEGA BACKTEST — Comprehensive Strategy Optimization
=====================================================
Tests all relevant parameter combinations across multiple market regimes
to find the statistically optimal strategy configuration.

Grid search dimensions:
  - Hold period:     3, 5, 7, 10 days
  - # of picks:      1, 2, 3
  - Min probability:  0.45, 0.50, 0.55, 0.60
  - Stop loss:       -5%, -7%, -10%, none
  - Ranking:         probability vs EV (expected value)

Realistic cost model:
  - Alpaca: $0 commission
  - Bid-ask spread: ~5 bps per side → 10 bps round trip
  - We use 10 bps total (conservative for S&P 500 large-caps)

German investor considerations:
  - Abgeltungsteuer: 26.375% (25% + 5.5% Soli)
  - Sparerpauschbetrag: €1,000/year tax-free
  - All gains taxed equally (no short/long term distinction)
"""
import pandas as pd
import numpy as np
import os
import sys
import gc
import time
import json
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════

def get_sp500_tickers():
    try:
        import io, requests
        resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                           headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        tickers = pd.read_html(io.StringIO(resp.text))[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        return sorted(tickers)
    except:
        return ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","JPM","V",
                "UNH","MA","HD","PG","JNJ","ABBV","MRK","LLY","COST","PEP","KO","WMT",
                "BAC","DIS","NFLX","ADBE","CRM","INTC","AMD","CSCO","QCOM","TXN","ORCL"]


def download_extended_data(tickers, period="5y"):
    """Download OHLCV for up to 5 years."""
    import yfinance as yf
    all_dfs = []
    batch_size = 20
    total = len(tickers)
    for i in range(0, total, batch_size):
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
                        if len(tk_data) > 100:
                            all_dfs.append(tk_data)
                    except:
                        pass
            time.sleep(0.15)
        except:
            pass
        done = min(i + batch_size, total)
        if done % 100 == 0 or done == total:
            print(f"  Downloaded {done}/{total} tickers ({len(all_dfs)} OK)")

    result = pd.concat(all_dfs) if all_dfs else pd.DataFrame()
    return result


def add_features(df):
    """Add technical features — exactly like production code."""
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
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        g["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        g["Return_1d"] = c.pct_change() * 100
        g["Return_5d"] = c.pct_change(5) * 100
        g["Return_20d"] = c.pct_change(20) * 100
        g["Return_60d"] = c.pct_change(60) * 100
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        g["ATR_14"] = tr.rolling(14).mean()
        g["ATR_Ratio"] = g["ATR_14"] / c
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        g["BB_Position"] = (c - sma20) / (2 * std20)
        g["BB_Width"] = (4 * std20) / sma20
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        g["MACD"] = ema12 - ema26
        g["MACD_Signal"] = g["MACD"].ewm(span=9).mean()
        g["MACD_Hist"] = g["MACD"] - g["MACD_Signal"]
        g["SMA_50_Dist"] = (c / c.rolling(50).mean() - 1) * 100
        g["SMA_200_Dist"] = (c / c.rolling(200).mean() - 1) * 100
        g["EMA_9_Dist"] = (c / c.ewm(span=9).mean() - 1) * 100
        low14 = l.rolling(14).min()
        high14 = h.rolling(14).max()
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
            if r.iloc[i] == 1:
                r.iloc[i] = r.iloc[i - 1] + 1
        g["Consecutive_Up"] = r
        r2 = (g["Daily_Return_Pct"] < 0).astype(int)
        for i in range(1, len(r2)):
            if r2.iloc[i] == 1:
                r2.iloc[i] = r2.iloc[i - 1] + 1
        g["Consecutive_Down"] = r2
        g["Sector_RS_5d"] = 0
        g["Sector_RS_20d"] = 0
        g["Dist_52w_High"] = (c / c.rolling(252).max() - 1) * 100
        g["Dist_52w_Low"] = (c / c.rolling(252).min() - 1) * 100
        out_dfs.append(g)

    result = pd.concat(out_dfs)
    feature_cols = [
        "Vol_Ratio", "Vol_Trend_5d", "Price_Position_20d", "Vol_Compression",
        "RSI_14", "Return_1d", "Return_5d", "Return_20d", "Return_60d",
        "ATR_14", "ATR_Ratio", "BB_Position", "BB_Width",
        "MACD", "MACD_Signal", "MACD_Hist", "SMA_50_Dist", "SMA_200_Dist",
        "EMA_9_Dist", "Stoch_K", "Stoch_D", "OBV_Slope", "VWAP_Dist",
        "Intraday_Range", "Range_Ratio", "Volatility_20d", "Volatility_Ratio",
        "Gap_Pct", "Body_Range_Ratio", "Consecutive_Up", "Consecutive_Down",
        "Sector_RS_5d", "Sector_RS_20d", "Dist_52w_High", "Dist_52w_Low",
    ]
    for col in feature_cols:
        if col in result.columns:
            result[f"Prev_{col}"] = result.groupby("Ticker")[col].shift(1)

    features = [f"Prev_{c}" for c in feature_cols if f"Prev_{c}" in result.columns]
    return result, features


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(df, features, big_move_threshold=5.0, hold_days=5, with_regressor=False):
    from xgboost import XGBClassifier, XGBRegressor

    df = df.copy()
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= big_move_threshold).astype(int)
    df["Fwd_Return"] = df.groupby("Ticker")["Daily_Return_Pct"].transform(
        lambda x: x.shift(-1).rolling(hold_days, min_periods=1).sum()
    )
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
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(df, features, hold_days=5, max_picks=3, max_positions=3,
                 min_prob=0.50, stop_loss=-7.0, use_ev=False, min_mom=0.0,
                 retrain_every=63, warmup_days=252, start_capital=100000,
                 cost_bps=10, verbose=False):
    """
    Walk-forward backtest with configurable parameters.
    
    cost_bps: total round-trip cost (Alpaca = 0 commission, ~10bps spread)
    """
    dates = sorted(df.index.unique())
    if len(dates) < warmup_days + 20:
        return None

    trade_dates = dates[warmup_days:]
    
    # Initial training
    warmup_data = df[df.index <= dates[warmup_days - 1]]
    model = train_model(warmup_data, features, hold_days=hold_days, with_regressor=use_ev)
    if model is None:
        return None
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
                reason = "stop" if ret <= stop_loss else "hold"
                sell_cost = pos["shares"] * cp * (cost_bps / 2 / 10000)  # half spread on exit
                proceeds = pos["shares"] * cp - sell_cost
                net_ret = (proceeds / pos["cost_basis"] - 1) * 100
                trade_log.append({
                    "Ticker": pos["ticker"], "Entry": pos["entry_date"],
                    "Exit": today, "Return": net_ret,
                    "Days": pos["days_held"], "Reason": reason,
                    "Gross_Return": ret,
                })
                capital += proceeds
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
                        scores["EV"] = scores["Prob"] * pred_ret
                    else:
                        scores["EV"] = scores["Prob"]

                    # Filters
                    mask = (scores["Prob"] >= min_prob) & (scores["Prob"] <= 0.85)
                    if min_mom > 0:
                        mask &= scores["Mom_20d"] > min_mom
                    else:
                        mask &= scores["Mom_20d"] > 0  # always require positive momentum
                    if use_ev:
                        mask &= scores["EV"] > 0

                    scores = scores[mask]

                    sort_col = "EV" if use_ev else "Prob"
                    scores = scores.sort_values(sort_col, ascending=False)

                    held = [p["ticker"] for p in positions]
                    scores = scores[~scores["Ticker"].isin(held)]

                    n_buy = min(open_slots, max_picks, len(scores))
                    for j in range(n_buy):
                        pick = scores.iloc[j]
                        ep = float(pick["Close"])
                        total_eq = capital + sum(p["shares"] * p["entry_price"] for p in positions)
                        pos_val = min(capital, total_eq / max_positions)
                        buy_cost = pos_val * (cost_bps / 2 / 10000)  # half spread on entry
                        shares = int((pos_val - buy_cost) / ep)
                        cost_basis = shares * ep + buy_cost
                        if shares > 0 and capital >= cost_basis:
                            capital -= cost_basis
                            positions.append({
                                "ticker": pick["Ticker"], "entry_date": today,
                                "entry_price": ep, "shares": shares, "days_held": 0,
                                "cost_basis": cost_basis,
                            })

        # Portfolio value
        pv = capital
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            pv += pos["shares"] * (float(td["Close"].iloc[0]) if len(td) > 0 else pos["entry_price"])
        equity_curve.append({"Date": today, "Value": pv})

        if verbose and day_i % 50 == 0:
            ret_pct = (pv / start_capital - 1) * 100
            print(f"    Day {day_i:4d} | {today.date()} | ${pv:>10,.0f} ({ret_pct:+.1f}%) | Trades: {len(trade_log)}")

    # Close remaining
    for pos in positions:
        last = df[df["Ticker"] == pos["ticker"]]
        if len(last) > 0:
            ep2 = float(last.iloc[-1]["Close"])
            ret2 = (ep2 / pos["entry_price"] - 1) * 100
            sell_cost = pos["shares"] * ep2 * (cost_bps / 2 / 10000)
            capital += pos["shares"] * ep2 - sell_cost
            trade_log.append({
                "Ticker": pos["ticker"], "Entry": pos["entry_date"],
                "Exit": trade_dates[-1], "Return": ret2,
                "Days": pos["days_held"], "Reason": "end",
            })

    eq = pd.DataFrame(equity_curve).set_index("Date")
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    
    if len(eq) == 0:
        return None

    final = eq["Value"].iloc[-1]
    total_ret = (final / start_capital - 1) * 100
    days_traded = (eq.index[-1] - eq.index[0]).days
    ann_ret = ((final / start_capital) ** (365.25 / max(days_traded, 1)) - 1) * 100
    daily_ret = eq["Value"].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    max_dd = ((eq["Value"] - eq["Value"].cummax()) / eq["Value"].cummax() * 100).min()
    
    # Sortino (downside deviation)
    neg_ret = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / neg_ret.std()) * np.sqrt(252) if len(neg_ret) > 0 and neg_ret.std() > 0 else 0
    
    # Calmar ratio (annual return / max drawdown)
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    wr = (trades_df["Return"] > 0).mean() * 100 if len(trades_df) > 0 else 0
    avg_win = trades_df.loc[trades_df["Return"] > 0, "Return"].mean() if (trades_df["Return"] > 0).any() else 0
    avg_loss = trades_df.loc[trades_df["Return"] <= 0, "Return"].mean() if (trades_df["Return"] <= 0).any() else 0
    n_trades = len(trades_df)
    avg_ret = trades_df["Return"].mean() if n_trades > 0 else 0
    
    # Profit factor
    gross_win = trades_df.loc[trades_df["Return"] > 0, "Return"].sum() if n_trades > 0 else 0
    gross_loss = abs(trades_df.loc[trades_df["Return"] <= 0, "Return"].sum()) if n_trades > 0 else 1
    pf = gross_win / max(gross_loss, 0.01)

    return {
        "total_return": total_ret,
        "annual_return": ann_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd,
        "win_rate": wr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_trade": avg_ret,
        "profit_factor": pf,
        "n_trades": n_trades,
        "final_value": final,
        "trades_per_month": n_trades / max(days_traded / 30.44, 1),
        "equity": eq,
        "trades": trades_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def run_grid_search(df, features, start_capital=100000):
    """Run comprehensive parameter grid search."""
    
    # Parameter grid
    grid = {
        "hold_days":    [3, 5, 7, 10],
        "max_picks":    [1, 2, 3],
        "min_prob":     [0.45, 0.50, 0.55, 0.60],
        "stop_loss":    [-5.0, -7.0, -10.0, -999.0],  # -999 = no stop
        "use_ev":       [False, True],
    }
    
    combos = list(product(
        grid["hold_days"], grid["max_picks"], grid["min_prob"],
        grid["stop_loss"], grid["use_ev"],
    ))
    
    print(f"\n{'='*80}")
    print(f"  GRID SEARCH: {len(combos)} parameter combinations")
    print(f"{'='*80}")
    print(f"  Hold days: {grid['hold_days']}")
    print(f"  Picks:     {grid['max_picks']}")
    print(f"  Min prob:  {grid['min_prob']}")
    print(f"  Stop loss: {grid['stop_loss']}")
    print(f"  Ranking:   prob, EV")
    print()
    
    results = []
    t0 = time.time()
    
    for i, (hd, mp, mprob, sl, ev) in enumerate(combos):
        sys.stdout.write(f"\r  Running {i+1}/{len(combos)} "
                         f"(hold={hd} picks={mp} prob={mprob:.0%} sl={sl if sl>-100 else 'none':>5} "
                         f"{'EV' if ev else 'Prob'})...")
        sys.stdout.flush()
        
        r = run_backtest(
            df, features,
            hold_days=hd,
            max_picks=mp, max_positions=mp,
            min_prob=mprob,
            stop_loss=sl,
            use_ev=ev,
            cost_bps=10,
            start_capital=start_capital,
            verbose=False,
        )
        
        if r is not None:
            results.append({
                "hold_days": hd,
                "picks": mp,
                "min_prob": mprob,
                "stop_loss": sl if sl > -100 else None,
                "ranking": "EV" if ev else "Prob",
                **{k: v for k, v in r.items() if k not in ("equity", "trades")},
            })
    
    elapsed = time.time() - t0
    print(f"\n\n  Completed {len(results)} backtests in {elapsed:.0f}s")
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_results(results_df):
    """Analyze grid search results and find optimal configuration."""
    
    df = results_df.copy()
    df["stop_loss_label"] = df["stop_loss"].apply(lambda x: f"{x:.0f}%" if x is not None else "none")
    
    print(f"\n{'='*100}")
    print("  TOP 20 STRATEGIES BY SHARPE RATIO")
    print(f"{'='*100}")
    
    top = df.nlargest(20, "sharpe")
    print(f"\n{'Rank':<5} {'Hold':>4} {'Picks':>5} {'Prob':>6} {'SL':>6} {'Rank':>5} "
          f"{'Return':>8} {'Ann.Ret':>8} {'Sharpe':>7} {'MaxDD':>7} {'WR':>5} "
          f"{'AvgTr':>7} {'PF':>5} {'#Tr':>5}")
    print("-" * 100)
    
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        print(f"  {rank:<3} {row['hold_days']:>4}d {row['picks']:>5} {row['min_prob']:>5.0%} "
              f"{row['stop_loss_label']:>6} {row['ranking']:>5} "
              f"{row['total_return']:>+7.1f}% {row['annual_return']:>+7.1f}% "
              f"{row['sharpe']:>6.2f} {row['max_dd']:>6.1f}% {row['win_rate']:>4.0f}% "
              f"{row['avg_trade']:>+6.2f}% {row['profit_factor']:>4.1f} {row['n_trades']:>5.0f}")
    
    print(f"\n{'='*100}")
    print("  TOP 20 STRATEGIES BY TOTAL RETURN")
    print(f"{'='*100}")
    
    top_ret = df.nlargest(20, "total_return")
    print(f"\n{'Rank':<5} {'Hold':>4} {'Picks':>5} {'Prob':>6} {'SL':>6} {'Rank':>5} "
          f"{'Return':>8} {'Ann.Ret':>8} {'Sharpe':>7} {'MaxDD':>7} {'WR':>5} "
          f"{'AvgTr':>7} {'PF':>5} {'#Tr':>5}")
    print("-" * 100)
    
    for rank, (_, row) in enumerate(top_ret.iterrows(), 1):
        print(f"  {rank:<3} {row['hold_days']:>4}d {row['picks']:>5} {row['min_prob']:>5.0%} "
              f"{row['stop_loss_label']:>6} {row['ranking']:>5} "
              f"{row['total_return']:>+7.1f}% {row['annual_return']:>+7.1f}% "
              f"{row['sharpe']:>6.2f} {row['max_dd']:>6.1f}% {row['win_rate']:>4.0f}% "
              f"{row['avg_trade']:>+6.2f}% {row['profit_factor']:>4.1f} {row['n_trades']:>5.0f}")

    # ── Dimension analysis ─────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*100}")
    
    for param in ["hold_days", "picks", "min_prob", "stop_loss_label", "ranking"]:
        print(f"\n  By {param}:")
        grp = df.groupby(param).agg({
            "annual_return": "mean",
            "sharpe": "mean",
            "max_dd": "mean",
            "win_rate": "mean",
            "profit_factor": "mean",
        }).round(2)
        for val, row in grp.iterrows():
            print(f"    {str(val):>8}: AnnRet={row['annual_return']:>+6.1f}%  Sharpe={row['sharpe']:>5.2f}  "
                  f"DD={row['max_dd']:>6.1f}%  WR={row['win_rate']:>4.0f}%  PF={row['profit_factor']:>4.1f}")
    
    # ── Best robust config (Sharpe ≥ top 10 median AND decent return) ──
    print(f"\n{'='*100}")
    print("  RECOMMENDED CONFIGURATION")
    print(f"{'='*100}")
    
    # Score: weighted combination (Sharpe most important for risk-adj returns)
    df["score"] = (
        df["sharpe"].rank(pct=True) * 0.35 +
        df["annual_return"].rank(pct=True) * 0.25 +
        df["sortino"].rank(pct=True) * 0.15 +
        df["calmar"].rank(pct=True) * 0.10 +
        df["profit_factor"].rank(pct=True) * 0.10 +
        df["win_rate"].rank(pct=True) * 0.05
    )
    
    best = df.loc[df["score"].idxmax()]
    print(f"\n  ★ BEST OVERALL (weighted score):")
    print(f"    Hold Days:    {best['hold_days']:.0f}")
    print(f"    Picks:        {best['picks']:.0f}")
    print(f"    Min Prob:     {best['min_prob']:.0%}")
    print(f"    Stop Loss:    {best['stop_loss_label']}")
    print(f"    Ranking:      {best['ranking']}")
    print(f"    ---")
    print(f"    Total Return: {best['total_return']:+.1f}%")
    print(f"    Annual Return:{best['annual_return']:+.1f}%")
    print(f"    Sharpe:       {best['sharpe']:.2f}")
    print(f"    Sortino:      {best['sortino']:.2f}")
    print(f"    Max Drawdown: {best['max_dd']:.1f}%")
    print(f"    Win Rate:     {best['win_rate']:.0f}%")
    print(f"    Profit Factor:{best['profit_factor']:.2f}")
    print(f"    Avg Trade:    {best['avg_trade']:+.2f}%")
    print(f"    Trades:       {best['n_trades']:.0f}")
    
    # Tax analysis
    print(f"\n{'='*100}")
    print("  GERMAN TAX ANALYSIS (Abgeltungsteuer)")
    print(f"{'='*100}")
    
    gross_annual = best["annual_return"]
    tax_rate = 0.26375  # 25% + 5.5% Soli
    sparerpauschbetrag = 1000  # €1,000 tax-free
    
    # Assuming €100k starting capital  
    gross_profit_yr = 100000 * gross_annual / 100
    taxable = max(0, gross_profit_yr - sparerpauschbetrag)
    tax = taxable * tax_rate
    net_profit = gross_profit_yr - tax
    net_annual = net_profit / 100000 * 100
    
    print(f"\n  Assuming €100,000 starting capital:")
    print(f"    Gross annual:    {gross_annual:+.1f}% (€{gross_profit_yr:,.0f})")
    print(f"    Tax-free amount: €{sparerpauschbetrag:,} (Sparerpauschbetrag)")
    print(f"    Taxable:         €{taxable:,.0f}")
    print(f"    Tax (26.375%):   €{tax:,.0f}")
    print(f"    Net annual:      {net_annual:+.1f}% (€{net_profit:,.0f})")
    print(f"\n  NOTE: Abgeltungsteuer applies to ALL realized gains equally.")
    print(f"  No long-term capital gains advantage in Germany.")
    print(f"  Frequent trading = more tax events, but same rate.")
    
    print(f"\n{'='*100}")
    print("  ALPACA TRADING FROM GERMANY — NOTES")
    print(f"{'='*100}")
    print("""
  ✓ Alpaca Markets available for German residents (ALPACA SECURITIES LLC)
  ✓ US stocks traded in USD — currency risk (EUR/USD)
  ✓ Commission: $0 per trade
  ✓ Spread: ~0.02-0.10% for S&P 500 stocks (we use 0.10% conservatively)
  ✓ Tax reporting: YOU must declare to Finanzamt (Anlage KAP)
    Alpaca does NOT withhold German tax automatically
  ✓ US withholding tax on dividends: 15% (reduced by DBA)
    But we're not holding for dividends — short-term trades
  ✓ W-8BEN form required to avoid 30% US withholding on dividends
    """)
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download fresh 5y data")
    parser.add_argument("--quick", action="store_true", help="Use existing 2y cache (faster)")
    args = parser.parse_args()
    
    cache_5y = os.path.join("data", "universe_prices_5y.csv")
    cache_2y = os.path.join("data", "universe_prices.csv")
    
    # ── Data loading ──────────────────────────────────────────────────
    if args.download or not os.path.exists(cache_5y):
        if not args.quick and not os.path.exists(cache_5y):
            print("=" * 80)
            print("  DOWNLOADING 5-YEAR DATA (this takes ~5 minutes)")
            print("=" * 80)
            tickers = get_sp500_tickers()
            print(f"  {len(tickers)} S&P 500 tickers")
            df_raw = download_extended_data(tickers, period="5y")
            if len(df_raw) > 100000:
                df_raw.to_csv(cache_5y)
                print(f"  Saved to {cache_5y}: {len(df_raw):,} rows, {df_raw['Ticker'].nunique()} tickers")
                print(f"  Range: {df_raw.index.min().date()} → {df_raw.index.max().date()}")
            else:
                print("  Download too small, falling back to 2y cache")
                df_raw = pd.read_csv(cache_2y, index_col=0, parse_dates=True)
        else:
            print("Using 2y cached data (--quick mode)")
            df_raw = pd.read_csv(cache_2y, index_col=0, parse_dates=True)
    else:
        print("Loading 5y cached data...")
        df_raw = pd.read_csv(cache_5y, index_col=0, parse_dates=True)
    
    n_tickers = df_raw["Ticker"].nunique()
    n_days = df_raw.index.nunique()
    print(f"\nData: {len(df_raw):,} rows | {n_tickers} tickers | {n_days} trading days")
    print(f"Range: {df_raw.index.min().date()} → {df_raw.index.max().date()}")
    
    # ── Feature engineering ───────────────────────────────────────────
    print("\nEngineering features...")
    df, features = add_features(df_raw)
    gc.collect()
    print(f"  {len(features)} features")
    
    # ── Run grid search ───────────────────────────────────────────────
    results_df = run_grid_search(df, features)
    
    # ── Analyze results ───────────────────────────────────────────────
    full_results = analyze_results(results_df)
    
    # ── Save results ──────────────────────────────────────────────────
    out_path = os.path.join("data", "backtest_grid_results.csv")
    full_results.to_csv(out_path, index=False)
    print(f"\n  Full results saved to {out_path}")
    
    # ── Run the winner in detail ──────────────────────────────────────
    best = full_results.loc[full_results["score"].idxmax()]
    print(f"\n{'='*80}")
    print(f"  DETAILED RUN OF BEST STRATEGY")
    print(f"{'='*80}")
    
    detail = run_backtest(
        df, features,
        hold_days=int(best["hold_days"]),
        max_picks=int(best["picks"]),
        max_positions=int(best["picks"]),
        min_prob=best["min_prob"],
        stop_loss=best["stop_loss"] if best["stop_loss"] is not None else -999.0,
        use_ev=(best["ranking"] == "EV"),
        cost_bps=10,
        verbose=True,
    )
    
    if detail:
        trades = detail["trades"]
        print(f"\n  Monthly trade breakdown:")
        if len(trades) > 0 and "Exit" in trades.columns:
            trades["Month"] = pd.to_datetime(trades["Exit"]).dt.to_period("M")
            monthly = trades.groupby("Month")["Return"].agg(["count", "mean", "sum"])
            monthly.columns = ["Trades", "Avg_Return", "Total_Return"]
            for m, row in monthly.iterrows():
                bar = "█" * max(1, int(row["Total_Return"] / 2)) if row["Total_Return"] > 0 else "▓" * max(1, int(abs(row["Total_Return"]) / 2))
                color = "+" if row["Total_Return"] > 0 else ""
                print(f"    {m}: {row['Trades']:2.0f} trades | Avg {row['Avg_Return']:+.2f}% | Total {color}{row['Total_Return']:.1f}% {bar}")
        
        # Equity curve stats
        eq = detail["equity"]
        print(f"\n  Equity curve:")
        print(f"    Start:  ${eq['Value'].iloc[0]:,.0f}")
        print(f"    End:    ${eq['Value'].iloc[-1]:,.0f}")
        print(f"    High:   ${eq['Value'].max():,.0f}")
        print(f"    Low:    ${eq['Value'].min():,.0f}")
    
    print(f"\n{'='*80}")
    print(f"  DONE — Grid search complete")
    print(f"{'='*80}")
