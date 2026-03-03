"""
REBALANCE vs HOLD backtest — Should we swap positions mid-hold for better picks?
===================================================================================
Tests 4 strategies (all with max_pos=2, 7d hold, -7% SL, Prob ranking):

1. Hold + Equal:   Current strategy (hold 7 days, equal sizing)
2. Hold + Tiered:  Hold 7 days, #1 gets 60% / #2 gets 40%
3. Rebalance + Tiered:  
     Each day, if a new pick has higher prob than weakest holding AND
     the weakest is down or flat, swap. Never sell winners early.
4. Aggressive Rebalance + Tiered:  
     Each day, always swap weakest holding for higher-prob pick,
     regardless of P&L.
"""
import pandas as pd
import numpy as np
import os
import sys
import gc
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from mega_backtest import add_features, train_model

# Fixed params
HOLD_DAYS = 7
STOP_LOSS = -7.0
MIN_PROB = 0.50
MAX_PROB = 0.85
COST_BPS = 10
RETRAIN_EVERY = 63
WARMUP_DAYS = 252
START_CAPITAL = 100000
MAX_POSITIONS = 2
MAX_PICKS = 2


def run_backtest(df, features, strategy="hold_equal", verbose=False):
    """
    Walk-forward backtest with different strategies.
    
    Strategies:
      hold_equal:      Hold 7d, equal weight
      hold_tiered:     Hold 7d, tiered weight (60/40)
      rebal_smart:     Swap losers for better picks (smart rebalance)
      rebal_aggressive: Always swap weakest for better picks
    """
    use_tiered = "tiered" in strategy
    do_rebalance = "rebal" in strategy
    aggressive_rebal = "aggressive" in strategy
    
    dates = sorted(df.index.unique())
    if len(dates) < WARMUP_DAYS + 20:
        return None

    trade_dates = dates[WARMUP_DAYS:]
    
    warmup_data = df[df.index <= dates[WARMUP_DAYS - 1]]
    model = train_model(warmup_data, features, hold_days=HOLD_DAYS, with_regressor=False)
    if model is None:
        return None
    days_since_retrain = 0

    capital = START_CAPITAL
    positions = []  # {ticker, entry_date, entry_price, shares, days_held, cost_basis, prob}
    trade_log = []
    equity_curve = []
    rebalance_count = 0

    for day_i, today in enumerate(trade_dates):
        # Retrain periodically
        if days_since_retrain >= RETRAIN_EVERY:
            past = df[df.index < today]
            new_m = train_model(past, features, hold_days=HOLD_DAYS, with_regressor=False)
            if new_m is not None:
                model = new_m
                days_since_retrain = 0
        days_since_retrain += 1

        # Update positions
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            if len(td) > 0:
                pos["current_price"] = float(td["Close"].iloc[0])
            pos["days_held"] += 1

        # ── Close expired / stopped positions ──
        closed = []
        for pos in positions:
            ret = (pos.get("current_price", pos["entry_price"]) / pos["entry_price"] - 1) * 100
            
            if ret <= STOP_LOSS or pos["days_held"] >= HOLD_DAYS:
                reason = "stop" if ret <= STOP_LOSS else "hold"
                cp = pos.get("current_price", pos["entry_price"])
                sell_cost = pos["shares"] * cp * (COST_BPS / 2 / 10000)
                proceeds = pos["shares"] * cp - sell_cost
                net_ret = (proceeds / pos["cost_basis"] - 1) * 100
                trade_log.append({
                    "Ticker": pos["ticker"], "Entry": pos["entry_date"],
                    "Exit": today, "Return": net_ret, "Days": pos["days_held"],
                    "Reason": reason, "Gross_Return": ret,
                })
                capital += proceeds
                closed.append(pos)
        for c in closed:
            positions.remove(c)

        # ── Score today's candidates ──
        today_df = df[df.index == today].copy()
        scores = pd.DataFrame()
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
                mask = (
                    (scores["Prob"] >= MIN_PROB) & 
                    (scores["Prob"] <= MAX_PROB) &
                    (scores["Mom_20d"] > 0)
                )
                scores = scores[mask].sort_values("Prob", ascending=False)

        # ── Rebalance check (if enabled) ──
        if do_rebalance and len(positions) >= MAX_POSITIONS and len(scores) > 0:
            held_tickers = [p["ticker"] for p in positions]
            candidates = scores[~scores["Ticker"].isin(held_tickers)].head(MAX_PICKS)
            
            if len(candidates) > 0:
                best_candidate_prob = float(candidates.iloc[0]["Prob"])
                
                # Find weakest position
                weakest = min(positions, key=lambda p: p.get("prob", 0))
                weakest_ret = (weakest.get("current_price", weakest["entry_price"]) / weakest["entry_price"] - 1) * 100
                weakest_prob = weakest.get("prob", 0)
                
                # Decide whether to swap
                should_swap = False
                min_prob_improvement = 0.05  # need 5% higher prob
                
                if aggressive_rebal:
                    # Always swap if better pick available
                    should_swap = best_candidate_prob > weakest_prob + min_prob_improvement
                else:
                    # Smart: only swap if weakest is losing/flat AND new pick is better
                    should_swap = (
                        best_candidate_prob > weakest_prob + min_prob_improvement and
                        weakest_ret <= 1.0 and  # only swap if losing or barely positive
                        weakest.get("days_held", 0) >= 2  # give at least 2 days
                    )
                
                if should_swap:
                    # Sell weakest
                    cp = weakest.get("current_price", weakest["entry_price"])
                    sell_cost = weakest["shares"] * cp * (COST_BPS / 2 / 10000)
                    proceeds = weakest["shares"] * cp - sell_cost
                    net_ret = (proceeds / weakest["cost_basis"] - 1) * 100
                    trade_log.append({
                        "Ticker": weakest["ticker"], "Entry": weakest["entry_date"],
                        "Exit": today, "Return": net_ret, "Days": weakest["days_held"],
                        "Reason": "rebalance",
                    })
                    capital += proceeds
                    positions.remove(weakest)
                    rebalance_count += 1

        # ── Open new positions ──
        open_slots = MAX_POSITIONS - len(positions)
        if open_slots > 0 and len(scores) > 0:
            held_tickers = [p["ticker"] for p in positions]
            candidates = scores[~scores["Ticker"].isin(held_tickers)].head(MAX_PICKS)
            
            n_buy = min(open_slots, len(candidates))
            if n_buy > 0:
                total_eq = capital + sum(
                    p["shares"] * p.get("current_price", p["entry_price"]) 
                    for p in positions
                )
                
                # Sizing
                if use_tiered:
                    tiers = [0.60, 0.40]  # #1 gets 60%, #2 gets 40%
                    # Allocate across max_positions slots
                    slot_size = total_eq / MAX_POSITIONS
                    for j in range(n_buy):
                        pick = candidates.iloc[j]
                        tier = tiers[j] if j < len(tiers) else 0.40
                        # Scale tier so total across n_buy picks = n_buy slots
                        alloc = slot_size * (tier / (sum(tiers[:n_buy]) / n_buy))
                        alloc = min(capital, alloc)
                        
                        ep = float(pick["Close"])
                        buy_cost = alloc * (COST_BPS / 2 / 10000)
                        shares = int((alloc - buy_cost) / ep)
                        cost_basis = shares * ep + buy_cost
                        
                        if shares > 0 and capital >= cost_basis:
                            capital -= cost_basis
                            positions.append({
                                "ticker": pick["Ticker"], "entry_date": today,
                                "entry_price": ep, "shares": shares,
                                "days_held": 0, "cost_basis": cost_basis,
                                "prob": float(pick["Prob"]),
                                "current_price": ep,
                            })
                else:
                    # Equal weight
                    pos_size = total_eq / MAX_POSITIONS
                    for j in range(n_buy):
                        pick = candidates.iloc[j]
                        ep = float(pick["Close"])
                        alloc = min(capital, pos_size)
                        buy_cost = alloc * (COST_BPS / 2 / 10000)
                        shares = int((alloc - buy_cost) / ep)
                        cost_basis = shares * ep + buy_cost
                        
                        if shares > 0 and capital >= cost_basis:
                            capital -= cost_basis
                            positions.append({
                                "ticker": pick["Ticker"], "entry_date": today,
                                "entry_price": ep, "shares": shares,
                                "days_held": 0, "cost_basis": cost_basis,
                                "prob": float(pick["Prob"]),
                                "current_price": ep,
                            })

        # Portfolio value
        pv = capital
        for pos in positions:
            pv += pos["shares"] * pos.get("current_price", pos["entry_price"])
        equity_curve.append({"Date": today, "Value": pv})

        if verbose and day_i % 50 == 0:
            ret_pct = (pv / START_CAPITAL - 1) * 100
            print(f"    Day {day_i:4d} | {today.date()} | ${pv:>10,.0f} ({ret_pct:+.1f}%) | "
                  f"Pos: {len(positions)} | Trades: {len(trade_log)} | Rebals: {rebalance_count}")

    # Close remaining
    for pos in positions:
        cp = pos.get("current_price", pos["entry_price"])
        ret = (cp / pos["entry_price"] - 1) * 100
        sell_cost = pos["shares"] * cp * (COST_BPS / 2 / 10000)
        capital += pos["shares"] * cp - sell_cost
        trade_log.append({
            "Ticker": pos["ticker"], "Entry": pos["entry_date"],
            "Exit": trade_dates[-1], "Return": ret, "Days": pos["days_held"],
            "Reason": "end",
        })

    eq = pd.DataFrame(equity_curve).set_index("Date")
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    
    if len(eq) == 0:
        return None

    final = eq["Value"].iloc[-1]
    total_ret = (final / START_CAPITAL - 1) * 100
    days_traded = (eq.index[-1] - eq.index[0]).days
    ann_ret = ((final / START_CAPITAL) ** (365.25 / max(days_traded, 1)) - 1) * 100
    daily_ret = eq["Value"].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    max_dd = ((eq["Value"] - eq["Value"].cummax()) / eq["Value"].cummax() * 100).min()
    
    neg_ret = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / neg_ret.std()) * np.sqrt(252) if len(neg_ret) > 0 and neg_ret.std() > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    wr = (trades_df["Return"] > 0).mean() * 100 if len(trades_df) > 0 else 0
    n_trades = len(trades_df)
    avg_ret = trades_df["Return"].mean() if n_trades > 0 else 0
    
    gross_win = trades_df.loc[trades_df["Return"] > 0, "Return"].sum() if n_trades > 0 else 0
    gross_loss = abs(trades_df.loc[trades_df["Return"] <= 0, "Return"].sum()) if n_trades > 0 else 1
    pf = gross_win / max(gross_loss, 0.01)

    # Rebalance stats
    rebal_trades = trades_df[trades_df["Reason"]=="rebalance"] if "Reason" in trades_df.columns else pd.DataFrame()

    return {
        "strategy": strategy,
        "total_return": total_ret,
        "annual_return": ann_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd,
        "win_rate": wr,
        "avg_trade": avg_ret,
        "profit_factor": pf,
        "n_trades": n_trades,
        "rebalance_count": rebalance_count,
        "rebal_avg_ret": rebal_trades["Return"].mean() if len(rebal_trades) > 0 else 0,
        "final_value": final,
        "equity": eq,
        "trades": trades_df,
    }


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cache_5y = os.path.join("data", "universe_prices_5y.csv")
    
    if not os.path.exists(cache_5y):
        print("ERROR: Need 5y data. Run: python mega_backtest.py --download")
        sys.exit(1)
    
    print("Loading 5y data...")
    df_raw = pd.read_csv(cache_5y, index_col=0, parse_dates=True)
    print(f"Data: {len(df_raw):,} rows | {df_raw['Ticker'].nunique()} tickers | "
          f"{df_raw.index.min().date()} → {df_raw.index.max().date()}")
    
    print("\nEngineering features...")
    df, features = add_features(df_raw)
    gc.collect()
    print(f"  {len(features)} features")
    
    strategies = [
        ("hold_equal",        "Hold 7d + Equal weight"),
        ("hold_tiered",       "Hold 7d + Tiered (60/40)"),
        ("rebal_smart_tiered","Smart Rebalance + Tiered"),
        ("rebal_aggressive_tiered", "Aggressive Rebalance + Tiered"),
    ]
    
    results = []
    for strat_key, strat_name in strategies:
        print(f"\n{'='*80}")
        print(f"  Running: {strat_name}")
        print(f"{'='*80}")
        
        r = run_backtest(df, features, strategy=strat_key, verbose=True)
        if r:
            results.append(r)
            print(f"\n  Result: {r['annual_return']:+.1f}%/yr | Sharpe {r['sharpe']:.2f} | "
                  f"DD {r['max_dd']:.1f}% | {r['n_trades']} trades | "
                  f"Rebals: {r['rebalance_count']}")
    
    # ── Comparison table ──
    print(f"\n{'='*100}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*100}")
    print(f"\n{'Strategy':<35} {'AnnRet':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} "
          f"{'WR':>5} {'PF':>5} {'Trades':>6} {'Rebals':>7} {'Final':>12}")
    print("-" * 100)
    
    for r in results:
        name = r["strategy"]
        for sk, sn in strategies:
            if sk == name:
                name = sn
                break
        print(f"  {name:<33} {r['annual_return']:>+7.1f}% {r['sharpe']:>6.2f} {r['sortino']:>7.2f} "
              f"{r['max_dd']:>6.1f}% {r['win_rate']:>4.0f}% {r['profit_factor']:>4.1f} "
              f"{r['n_trades']:>6} {r['rebalance_count']:>7} ${r['final_value']:>11,.0f}")
    
    # ── Winner ──
    best = max(results, key=lambda r: (
        0.35 * (r["sharpe"] / max(x["sharpe"] for x in results)) +
        0.30 * (r["annual_return"] / max(x["annual_return"] for x in results)) +
        0.15 * (r["sortino"] / max(x["sortino"] for x in results)) +
        0.10 * (r["profit_factor"] / max(x["profit_factor"] for x in results)) +
        0.10 * (1 - abs(r["max_dd"]) / max(abs(x["max_dd"]) for x in results))
    ))
    
    for sk, sn in strategies:
        if sk == best["strategy"]:
            best_name = sn
            break
    
    print(f"\n  ★ WINNER: {best_name}")
    print(f"    Annual Return:  {best['annual_return']:+.1f}%")
    print(f"    Sharpe:         {best['sharpe']:.2f}")
    print(f"    Max Drawdown:   {best['max_dd']:.1f}%")
    print(f"    Final Value:    ${best['final_value']:,.0f}")
    print(f"    Rebalances:     {best['rebalance_count']}")
    
    if best["rebalance_count"] > 0:
        print(f"    Avg rebal trade:{best['rebal_avg_ret']:+.2f}%")
    
    # Tax
    gross = best["annual_return"]
    gross_eur = 100000 * gross / 100
    tax = max(0, gross_eur - 1000) * 0.26375
    net = gross_eur - tax
    print(f"\n    After German tax: ~{net/1000:.1f}%/yr net")
    
    print(f"\n{'='*80}")
    print("  DONE")
    print(f"{'='*80}")
