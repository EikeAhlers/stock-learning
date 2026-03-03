"""
POSITION SIZING BACKTEST — Find optimal max_positions and sizing method.
========================================================================
Tests:
  - max_positions: [2, 4, 6, 8, 10, 14]
  - picks_per_day: [1, 2, 3]
  - sizing method:
      Equal:   equity / max_positions (current)
      Prob:    weight proportional to probability
      Kelly:   fractional Kelly criterion
      Tiered:  top pick gets 1.5x, second gets 1x, third gets 0.75x

Fixed params (from previous grid search winner):
  - hold_days = 7
  - stop_loss = -7%
  - min_prob = 0.50
  - ranking = Prob
  - cost_bps = 10
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

# Reuse data/feature/training code from mega_backtest
from mega_backtest import add_features, train_model, get_sp500_tickers

# Fixed params from grid search winner
HOLD_DAYS = 7
STOP_LOSS = -7.0
MIN_PROB = 0.50
MAX_PROB = 0.85
COST_BPS = 10
USE_EV = False
RETRAIN_EVERY = 63
WARMUP_DAYS = 252
START_CAPITAL = 100000


# ═══════════════════════════════════════════════════════════════════════════
#  SIZING METHODS
# ═══════════════════════════════════════════════════════════════════════════

def size_equal(capital, equity, max_positions, picks_with_probs):
    """Equal weight: equity / max_positions per pick."""
    pos_size = equity / max_positions
    result = []
    for ticker, price, prob in picks_with_probs:
        alloc = min(capital, pos_size)
        result.append((ticker, price, prob, alloc))
        capital -= alloc
        if capital <= 0:
            break
    return result


def size_prob_weighted(capital, equity, max_positions, picks_with_probs):
    """Weight allocation proportional to probability."""
    total_budget = equity / max_positions * len(picks_with_probs)
    total_budget = min(capital, total_budget)
    
    total_prob = sum(p[2] for p in picks_with_probs)
    if total_prob <= 0:
        return size_equal(capital, equity, max_positions, picks_with_probs)
    
    result = []
    remaining = total_budget
    for ticker, price, prob in picks_with_probs:
        weight = prob / total_prob
        alloc = min(remaining, total_budget * weight)
        result.append((ticker, price, prob, alloc))
        remaining -= alloc
        if remaining <= 0:
            break
    return result


def size_kelly(capital, equity, max_positions, picks_with_probs, 
               win_rate=0.55, avg_win=3.5, avg_loss=4.0):
    """
    Fractional Kelly: f* = (p * b - q) / b
    where p = win probability, b = avg_win/avg_loss, q = 1-p
    Use 1/4 Kelly for safety.
    """
    result = []
    for ticker, price, prob in picks_with_probs:
        # Use actual model prob as our edge estimate
        p = min(prob, 0.85)  # cap
        q = 1 - p
        b = avg_win / max(avg_loss, 0.01)
        
        kelly_frac = (p * b - q) / max(b, 0.01)
        kelly_frac = max(0, min(kelly_frac, 0.5))  # cap at 50%
        
        # Quarter Kelly for safety
        frac = kelly_frac * 0.25
        
        # But still respect max_positions slots
        max_per_slot = equity / max_positions
        alloc = min(capital, equity * frac, max_per_slot * 1.5)
        
        if alloc < price:  # can't even buy 1 share
            continue
            
        result.append((ticker, price, prob, alloc))
        capital -= alloc
        if capital <= 0:
            break
    return result


def size_tiered(capital, equity, max_positions, picks_with_probs):
    """
    Tiered: #1 pick gets 1.5x base, #2 gets 1.0x, #3 gets 0.75x.
    Base = equity / (max_positions * adjustment_factor)
    """
    tiers = [1.5, 1.0, 0.75, 0.60, 0.50]  # extend for more picks
    
    # Calculate base so total deployed doesn't exceed available slots
    n_picks = len(picks_with_probs)
    tier_weights = tiers[:n_picks]
    total_weight = sum(tier_weights)
    # Total budget = num_picks slots worth of equity
    total_budget = (equity / max_positions) * n_picks
    base = total_budget / total_weight
    
    result = []
    for i, (ticker, price, prob) in enumerate(picks_with_probs):
        weight = tiers[i] if i < len(tiers) else 0.5
        alloc = min(capital, base * weight)
        result.append((ticker, price, prob, alloc))
        capital -= alloc
        if capital <= 0:
            break
    return result


SIZING_METHODS = {
    "Equal": size_equal,
    "Prob": size_prob_weighted,
    "Kelly": size_kelly,
    "Tiered": size_tiered,
}


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE (with dynamic sizing)
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(df, features, max_picks=2, max_positions=2,
                 sizing_method="Equal", verbose=False):
    """
    Walk-forward backtest with configurable position sizing.
    """
    dates = sorted(df.index.unique())
    if len(dates) < WARMUP_DAYS + 20:
        return None

    trade_dates = dates[WARMUP_DAYS:]
    
    # Initial training
    warmup_data = df[df.index <= dates[WARMUP_DAYS - 1]]
    model = train_model(warmup_data, features, hold_days=HOLD_DAYS, with_regressor=False)
    if model is None:
        return None
    days_since_retrain = 0

    # Running Kelly stats (updated from actual trades)
    kelly_win_rate = 0.55
    kelly_avg_win = 3.5
    kelly_avg_loss = 4.0
    recent_trades = []  # rolling window for Kelly updates

    capital = START_CAPITAL
    positions = []
    trade_log = []
    equity_curve = []
    daily_position_count = []

    sizer = SIZING_METHODS.get(sizing_method, size_equal)

    for day_i, today in enumerate(trade_dates):
        # Retrain periodically
        if days_since_retrain >= RETRAIN_EVERY:
            past = df[df.index < today]
            new_m = train_model(past, features, hold_days=HOLD_DAYS, with_regressor=False)
            if new_m is not None:
                model = new_m
                days_since_retrain = 0
        days_since_retrain += 1

        # ── Close expired / stopped positions ──
        closed = []
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            if len(td) == 0:
                pos["days_held"] += 1
                continue
            cp = float(td["Close"].iloc[0])
            ret = (cp / pos["entry_price"] - 1) * 100
            pos["days_held"] += 1

            if ret <= STOP_LOSS or pos["days_held"] >= HOLD_DAYS:
                reason = "stop" if ret <= STOP_LOSS else "hold"
                sell_cost = pos["shares"] * cp * (COST_BPS / 2 / 10000)
                proceeds = pos["shares"] * cp - sell_cost
                net_ret = (proceeds / pos["cost_basis"] - 1) * 100
                trade_log.append({
                    "Ticker": pos["ticker"], "Entry": pos["entry_date"],
                    "Exit": today, "Return": net_ret,
                    "Days": pos["days_held"], "Reason": reason,
                    "Gross_Return": ret, "Shares": pos["shares"],
                    "Alloc_Pct": pos.get("alloc_pct", 0),
                })
                capital += proceeds
                closed.append(pos)
                
                # Update Kelly stats
                recent_trades.append(net_ret)
                if len(recent_trades) > 100:
                    recent_trades = recent_trades[-100:]
                if len(recent_trades) >= 20:
                    wins = [r for r in recent_trades if r > 0]
                    losses = [r for r in recent_trades if r <= 0]
                    if wins and losses:
                        kelly_win_rate = len(wins) / len(recent_trades)
                        kelly_avg_win = np.mean(wins)
                        kelly_avg_loss = abs(np.mean(losses))

        for c in closed:
            positions.remove(c)

        # ── Open new positions ──
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

                    # Filters
                    mask = (
                        (scores["Prob"] >= MIN_PROB) & 
                        (scores["Prob"] <= MAX_PROB) &
                        (scores["Mom_20d"] > 0)
                    )
                    scores = scores[mask]
                    scores = scores.sort_values("Prob", ascending=False)

                    held = [p["ticker"] for p in positions]
                    scores = scores[~scores["Ticker"].isin(held)]

                    n_buy = min(open_slots, max_picks, len(scores))
                    if n_buy > 0:
                        picks = []
                        for j in range(n_buy):
                            row = scores.iloc[j]
                            picks.append((row["Ticker"], float(row["Close"]), float(row["Prob"])))

                        # Calculate total equity for sizing
                        total_eq = capital + sum(
                            p["shares"] * p["entry_price"] for p in positions
                        )

                        # Apply sizing method
                        if sizing_method == "Kelly":
                            sized = sizer(capital, total_eq, max_positions, picks,
                                         kelly_win_rate, kelly_avg_win, kelly_avg_loss)
                        else:
                            sized = sizer(capital, total_eq, max_positions, picks)

                        for ticker, price, prob, alloc in sized:
                            buy_cost = alloc * (COST_BPS / 2 / 10000)
                            shares = int((alloc - buy_cost) / price)
                            cost_basis = shares * price + buy_cost
                            alloc_pct = (cost_basis / total_eq * 100) if total_eq > 0 else 0

                            if shares > 0 and capital >= cost_basis:
                                capital -= cost_basis
                                positions.append({
                                    "ticker": ticker, "entry_date": today,
                                    "entry_price": price, "shares": shares, 
                                    "days_held": 0, "cost_basis": cost_basis,
                                    "alloc_pct": alloc_pct,
                                })

        # Portfolio value
        pv = capital
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            pv += pos["shares"] * (float(td["Close"].iloc[0]) if len(td) > 0 else pos["entry_price"])
        equity_curve.append({"Date": today, "Value": pv})
        daily_position_count.append(len(positions))

        if verbose and day_i % 50 == 0:
            ret_pct = (pv / START_CAPITAL - 1) * 100
            print(f"    Day {day_i:4d} | {today.date()} | ${pv:>10,.0f} ({ret_pct:+.1f}%) | "
                  f"Pos: {len(positions)} | Trades: {len(trade_log)}")

    # Close remaining
    for pos in positions:
        last = df[df["Ticker"] == pos["ticker"]]
        if len(last) > 0:
            ep2 = float(last.iloc[-1]["Close"])
            ret2 = (ep2 / pos["entry_price"] - 1) * 100
            sell_cost = pos["shares"] * ep2 * (COST_BPS / 2 / 10000)
            capital += pos["shares"] * ep2 - sell_cost
            trade_log.append({
                "Ticker": pos["ticker"], "Entry": pos["entry_date"],
                "Exit": trade_dates[-1], "Return": ret2,
                "Days": pos["days_held"], "Reason": "end",
                "Gross_Return": ret2, "Shares": pos["shares"],
                "Alloc_Pct": pos.get("alloc_pct", 0),
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
    avg_win = trades_df.loc[trades_df["Return"] > 0, "Return"].mean() if len(trades_df) > 0 and (trades_df["Return"] > 0).any() else 0
    avg_loss = trades_df.loc[trades_df["Return"] <= 0, "Return"].mean() if len(trades_df) > 0 and (trades_df["Return"] <= 0).any() else 0
    n_trades = len(trades_df)
    avg_ret = trades_df["Return"].mean() if n_trades > 0 else 0
    
    gross_win = trades_df.loc[trades_df["Return"] > 0, "Return"].sum() if n_trades > 0 else 0
    gross_loss = abs(trades_df.loc[trades_df["Return"] <= 0, "Return"].sum()) if n_trades > 0 else 1
    pf = gross_win / max(gross_loss, 0.01)
    
    avg_positions = np.mean(daily_position_count) if daily_position_count else 0
    max_positions_used = max(daily_position_count) if daily_position_count else 0
    pct_fully_invested = sum(1 for c in daily_position_count if c >= max_positions) / max(len(daily_position_count), 1) * 100

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
        "avg_positions": avg_positions,
        "max_positions_used": max_positions_used,
        "pct_fully_invested": pct_fully_invested,
        "equity": eq,
        "trades": trades_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def run_position_grid(df, features):
    """Test all max_positions × picks × sizing combos."""
    
    grid = {
        "max_positions": [2, 4, 6, 8, 10, 14],
        "picks_per_day": [1, 2, 3],
        "sizing":        ["Equal", "Prob", "Kelly", "Tiered"],
    }
    
    combos = list(product(
        grid["max_positions"], grid["picks_per_day"], grid["sizing"]
    ))
    
    # Filter out impossible combos (picks > positions)
    combos = [(mp, ppd, sz) for mp, ppd, sz in combos if ppd <= mp]
    
    print(f"\n{'='*90}")
    print(f"  POSITION SIZING GRID SEARCH: {len(combos)} combinations")
    print(f"{'='*90}")
    print(f"  Fixed: hold=7d, stop=-7%, min_prob=50%, ranking=Prob, cost=10bps")
    print(f"  Testing: max_positions={grid['max_positions']}")
    print(f"           picks/day={grid['picks_per_day']}")
    print(f"           sizing={grid['sizing']}")
    print()
    
    results = []
    t0 = time.time()
    
    for i, (mp, ppd, sz) in enumerate(combos):
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(combos) - i) if i > 0 else 0
        sys.stdout.write(f"\r  [{i+1}/{len(combos)}] max_pos={mp:>2} picks={ppd} sizing={sz:>6}"
                         f"  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)   ")
        sys.stdout.flush()
        
        r = run_backtest(
            df, features,
            max_picks=ppd,
            max_positions=mp,
            sizing_method=sz,
            verbose=False,
        )
        
        if r is not None:
            results.append({
                "max_positions": mp,
                "picks_per_day": ppd,
                "sizing": sz,
                **{k: v for k, v in r.items() if k not in ("equity", "trades")},
            })
    
    elapsed = time.time() - t0
    print(f"\n\n  Completed {len(results)} backtests in {elapsed:.0f}s")
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_position_results(df):
    """Analyze position sizing results."""
    
    # ── Top 25 by Sharpe ──
    print(f"\n{'='*120}")
    print("  TOP 25 BY SHARPE RATIO")
    print(f"{'='*120}")
    
    top = df.nlargest(25, "sharpe")
    print(f"\n{'Rk':<3} {'MaxPos':>6} {'Picks':>5} {'Sizing':>7} "
          f"{'AnnRet':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} "
          f"{'WR':>5} {'AvgTr':>7} {'PF':>5} {'#Tr':>5} {'AvgPos':>6} {'Full%':>6}")
    print("-" * 120)
    
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        print(f" {rank:<2} {row['max_positions']:>6} {row['picks_per_day']:>5} {row['sizing']:>7} "
              f"{row['annual_return']:>+7.1f}% {row['sharpe']:>6.2f} {row['sortino']:>7.2f} "
              f"{row['max_dd']:>6.1f}% {row['win_rate']:>4.0f}% "
              f"{row['avg_trade']:>+6.2f}% {row['profit_factor']:>4.1f} {row['n_trades']:>5.0f} "
              f"{row['avg_positions']:>5.1f} {row['pct_fully_invested']:>5.1f}%")
    
    # ── Top 25 by Return ──
    print(f"\n{'='*120}")
    print("  TOP 25 BY ANNUAL RETURN")
    print(f"{'='*120}")
    
    top_ret = df.nlargest(25, "annual_return")
    print(f"\n{'Rk':<3} {'MaxPos':>6} {'Picks':>5} {'Sizing':>7} "
          f"{'AnnRet':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} "
          f"{'WR':>5} {'AvgTr':>7} {'PF':>5} {'#Tr':>5} {'AvgPos':>6} {'Full%':>6} {'Final$':>10}")
    print("-" * 120)
    
    for rank, (_, row) in enumerate(top_ret.iterrows(), 1):
        print(f" {rank:<2} {row['max_positions']:>6} {row['picks_per_day']:>5} {row['sizing']:>7} "
              f"{row['annual_return']:>+7.1f}% {row['sharpe']:>6.2f} {row['sortino']:>7.2f} "
              f"{row['max_dd']:>6.1f}% {row['win_rate']:>4.0f}% "
              f"{row['avg_trade']:>+6.2f}% {row['profit_factor']:>4.1f} {row['n_trades']:>5.0f} "
              f"{row['avg_positions']:>5.1f} {row['pct_fully_invested']:>5.1f}% "
              f"${row['final_value']:>10,.0f}")

    # ── Dimension analysis ──
    print(f"\n{'='*120}")
    print("  SENSITIVITY ANALYSIS")
    print(f"{'='*120}")
    
    for param in ["max_positions", "picks_per_day", "sizing"]:
        print(f"\n  By {param}:")
        grp = df.groupby(param).agg({
            "annual_return": "mean",
            "sharpe": "mean",
            "sortino": "mean",
            "max_dd": "mean",
            "win_rate": "mean",
            "profit_factor": "mean",
            "n_trades": "mean",
            "avg_positions": "mean",
            "pct_fully_invested": "mean",
        }).round(2)
        for val, row in grp.iterrows():
            print(f"    {str(val):>8}: AnnRet={row['annual_return']:>+7.1f}%  Sharpe={row['sharpe']:>5.2f}  "
                  f"Sortino={row['sortino']:>5.2f}  DD={row['max_dd']:>6.1f}%  WR={row['win_rate']:>4.0f}%  "
                  f"PF={row['profit_factor']:>4.1f}  Trades={row['n_trades']:>5.0f}  "
                  f"AvgPos={row['avg_positions']:>4.1f}  Full={row['pct_fully_invested']:>4.0f}%")
    
    # ── Cross comparison: max_positions × sizing ──
    print(f"\n  Annual Return by max_positions × sizing:")
    pivot = df.pivot_table(values="annual_return", index="max_positions", 
                           columns="sizing", aggfunc="mean")
    print(pivot.round(1).to_string())
    
    print(f"\n  Sharpe by max_positions × sizing:")
    pivot_s = df.pivot_table(values="sharpe", index="max_positions", 
                              columns="sizing", aggfunc="mean")
    print(pivot_s.round(2).to_string())

    print(f"\n  Max Drawdown by max_positions × sizing:")
    pivot_dd = df.pivot_table(values="max_dd", index="max_positions", 
                               columns="sizing", aggfunc="mean")
    print(pivot_dd.round(1).to_string())

    # ── Composite score ──
    print(f"\n{'='*120}")
    print("  WINNER (composite score)")
    print(f"{'='*120}")
    
    df["score"] = (
        df["sharpe"].rank(pct=True) * 0.30 +
        df["annual_return"].rank(pct=True) * 0.25 +
        df["sortino"].rank(pct=True) * 0.15 +
        df["calmar"].rank(pct=True) * 0.10 +
        df["profit_factor"].rank(pct=True) * 0.10 +
        df["max_dd"].rank(pct=True, ascending=False) * 0.05 +  # less DD = better
        df["win_rate"].rank(pct=True) * 0.05
    )
    
    best = df.loc[df["score"].idxmax()]
    print(f"\n  ★ BEST CONFIGURATION:")
    print(f"    Max Positions:     {best['max_positions']:.0f}")
    print(f"    Picks Per Day:     {best['picks_per_day']:.0f}")
    print(f"    Sizing Method:     {best['sizing']}")
    print(f"    ---")
    print(f"    Annual Return:     {best['annual_return']:+.1f}%")
    print(f"    Sharpe:            {best['sharpe']:.2f}")
    print(f"    Sortino:           {best['sortino']:.2f}")
    print(f"    Max Drawdown:      {best['max_dd']:.1f}%")
    print(f"    Win Rate:          {best['win_rate']:.0f}%")
    print(f"    Profit Factor:     {best['profit_factor']:.2f}")
    print(f"    Avg Trade:         {best['avg_trade']:+.2f}%")
    print(f"    Total Trades:      {best['n_trades']:.0f}")
    print(f"    Avg Positions:     {best['avg_positions']:.1f}")
    print(f"    % Fully Invested:  {best['pct_fully_invested']:.0f}%")
    print(f"    Final Value:       ${best['final_value']:,.0f}  ({best['total_return']:+.1f}%)")
    
    # Compare to OLD config
    print(f"\n  ── vs OLD config (max_pos=2, picks=2, Equal) ──")
    old = df[(df["max_positions"]==2) & (df["picks_per_day"]==2) & (df["sizing"]=="Equal")]
    if len(old) > 0:
        o = old.iloc[0]
        print(f"    OLD:  AnnRet={o['annual_return']:>+7.1f}%  Sharpe={o['sharpe']:.2f}  DD={o['max_dd']:.1f}%  "
              f"Trades={o['n_trades']:.0f}  Final=${o['final_value']:,.0f}")
        print(f"    NEW:  AnnRet={best['annual_return']:>+7.1f}%  Sharpe={best['sharpe']:.2f}  DD={best['max_dd']:.1f}%  "
              f"Trades={best['n_trades']:.0f}  Final=${best['final_value']:,.0f}")
        improvement = best['annual_return'] - o['annual_return']
        print(f"    Improvement: {improvement:+.1f}% annual return")
    
    # German tax
    print(f"\n  ── German Tax (Abgeltungsteuer 26.375%) ──")
    gross = best["annual_return"]
    gross_eur = 100000 * gross / 100
    taxable = max(0, gross_eur - 1000)
    tax = taxable * 0.26375
    net = gross_eur - tax
    print(f"    Gross: {gross:+.1f}%/yr (€{gross_eur:,.0f})")
    print(f"    Tax:   €{tax:,.0f}")
    print(f"    Net:   {net/1000:.1f}%/yr (€{net:,.0f})")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cache_5y = os.path.join("data", "universe_prices_5y.csv")
    
    if not os.path.exists(cache_5y):
        print("ERROR: Need 5y data cache. Run: python mega_backtest.py --download")
        sys.exit(1)
    
    print("Loading 5y cached data...")
    df_raw = pd.read_csv(cache_5y, index_col=0, parse_dates=True)
    n_tickers = df_raw["Ticker"].nunique()
    print(f"Data: {len(df_raw):,} rows | {n_tickers} tickers | "
          f"{df_raw.index.min().date()} → {df_raw.index.max().date()}")
    
    print("\nEngineering features...")
    df, features = add_features(df_raw)
    gc.collect()
    print(f"  {len(features)} features")
    
    # Run grid
    results_df = run_position_grid(df, features)
    
    # Analyze
    full_results = analyze_position_results(results_df)
    
    # Save
    out_path = os.path.join("data", "position_sizing_results.csv")
    cols_to_save = [c for c in full_results.columns if c not in ("equity", "trades")]
    full_results[cols_to_save].to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")
    
    # Detailed run of winner
    best = full_results.loc[full_results["score"].idxmax()]
    print(f"\n{'='*90}")
    print(f"  DETAILED RUN: max_pos={best['max_positions']:.0f} picks={best['picks_per_day']:.0f} sizing={best['sizing']}")
    print(f"{'='*90}")
    
    detail = run_backtest(
        df, features,
        max_picks=int(best["picks_per_day"]),
        max_positions=int(best["max_positions"]),
        sizing_method=best["sizing"],
        verbose=True,
    )
    
    if detail:
        trades = detail["trades"]
        eq = detail["equity"]
        print(f"\n  Equity: ${eq['Value'].iloc[0]:,.0f} → ${eq['Value'].iloc[-1]:,.0f}")
        print(f"  High:   ${eq['Value'].max():,.0f}")
        print(f"  Low:    ${eq['Value'].min():,.0f}")
        
        if len(trades) > 0:
            # Allocation stats
            print(f"\n  Position allocation stats:")
            print(f"    Avg alloc per position:  {trades['Alloc_Pct'].mean():.1f}% of equity")
            print(f"    Min alloc:               {trades['Alloc_Pct'].min():.1f}%")
            print(f"    Max alloc:               {trades['Alloc_Pct'].max():.1f}%")
            
            # By year
            if "Exit" in trades.columns:
                trades["Year"] = pd.to_datetime(trades["Exit"]).dt.year
                yearly = trades.groupby("Year").agg({
                    "Return": ["count", "mean", "sum"],
                }).round(2)
                yearly.columns = ["Trades", "Avg_Return", "Total_Return"]
                print(f"\n  Yearly breakdown:")
                for yr, row in yearly.iterrows():
                    print(f"    {yr}: {row['Trades']:3.0f} trades | "
                          f"Avg {row['Avg_Return']:+.2f}% | Total {row['Total_Return']:+.1f}%")
    
    print(f"\n{'='*90}")
    print("  DONE")
    print(f"{'='*90}")
