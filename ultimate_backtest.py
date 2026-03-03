"""
ULTIMATE BACKTEST -- Optimize EVERYTHING
=========================================
Three-phase tournament on 5 years of S&P 500 data.

PHASE 1 -- Model Hyperparameters (fix trading params to current best):
  big_move_threshold, n_estimators, max_depth, learning_rate,
  min_child_weight, training_lookback, retrain_every
  -> ~36 combos

PHASE 2 -- Filter & Sizing (fix model to Phase 1 winner):
  min_prob, max_prob, min_momentum, tier_ratio
  -> ~135 combos

PHASE 3 -- Dynamic/Adaptive Features (fix to Phase 1+2 winners):
  trailing_stop, vol_adjusted_stop, regime_filter,
  dynamic_hold, partial_profit
  -> ~32 combos

Total: ~203 combos
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
from mega_backtest import add_features, get_sp500_tickers


# ============================================================================
#  CONFIGURABLE MODEL TRAINING
# ============================================================================

def train_model_custom(df, features, big_move_threshold=5.0, hold_days=7,
                       n_estimators=150, max_depth=5, learning_rate=0.05,
                       min_child_weight=20):
    """Train XGBoost with custom hyperparameters."""
    from xgboost import XGBClassifier

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
    scale_pos = (len(y_cls) - y_cls.sum()) / max(y_cls.sum(), 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, min_child_weight=min_child_weight,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=42,
        verbosity=0, tree_method="hist", n_jobs=-1,
    )
    clf.fit(X_scaled, y_cls)

    return {"model": clf, "scaler": scaler, "features": features}


# ============================================================================
#  ULTIMATE BACKTEST ENGINE
# ============================================================================

def run_ultimate_backtest(df, features, params, verbose=False):
    """
    Walk-forward backtest with ALL configurable parameters.

    params dict keys:
      Model: big_move, n_est, depth, lr, mcw, lookback, retrain
      Filters: min_prob, max_prob, min_mom
      Sizing: tier_ratio (tuple like (60,40))
      Stops: stop_loss, trailing_stop, vol_stop
      Dynamic: regime_filter, dynamic_hold, partial_profit
    """
    # Extract params with defaults
    big_move = params.get("big_move", 5.0)
    n_est = params.get("n_est", 150)
    depth = params.get("depth", 5)
    lr = params.get("lr", 0.05)
    mcw = params.get("mcw", 20)
    lookback = params.get("lookback", 252)
    retrain = params.get("retrain", 63)

    min_prob = params.get("min_prob", 0.50)
    max_prob = params.get("max_prob", 0.85)
    min_mom = params.get("min_mom", 0)
    tier_ratio = params.get("tier_ratio", (60, 40))

    stop_loss = params.get("stop_loss", -7.0)
    trailing_stop = params.get("trailing_stop", None)
    vol_stop = params.get("vol_stop", False)

    regime_filter = params.get("regime_filter", None)
    dynamic_hold = params.get("dynamic_hold", False)
    partial_profit = params.get("partial_profit", False)

    hold_days = 7
    max_positions = 2
    max_picks = 2
    cost_bps = 10
    start_capital = 100000

    dates = sorted(df.index.unique())
    if len(dates) < lookback + 20:
        return None

    trade_dates = dates[lookback:]

    # Initial training
    warmup_data = df[df.index <= dates[lookback - 1]]
    model = train_model_custom(warmup_data, features, big_move, hold_days,
                               n_est, depth, lr, mcw)
    if model is None:
        return None
    days_since_retrain = 0

    capital = start_capital
    positions = []
    trade_log = []
    equity_curve = []

    # For regime detection
    recent_equity = []

    for day_i, today in enumerate(trade_dates):
        # Retrain periodically
        if days_since_retrain >= retrain:
            past = df[df.index < today]
            past_dates = sorted(past.index.unique())
            if len(past_dates) > lookback:
                cutoff = past_dates[-lookback]
                past = past[past.index >= cutoff]
            new_m = train_model_custom(past, features, big_move, hold_days,
                                       n_est, depth, lr, mcw)
            if new_m is not None:
                model = new_m
                days_since_retrain = 0
        days_since_retrain += 1

        # Update position prices and track peaks
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            if len(td) > 0:
                cp = float(td["Close"].iloc[0])
                pos["current_price"] = cp
                pos["peak_price"] = max(pos.get("peak_price", pos["entry_price"]), cp)
            pos["days_held"] += 1

        # -- Determine effective stop loss --
        def get_stop(pos):
            base_stop = stop_loss

            # Volatility-adjusted stop
            if vol_stop:
                td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
                if len(td) > 0 and "Prev_ATR_Ratio" in td.columns:
                    atr_ratio = float(td["Prev_ATR_Ratio"].iloc[0]) if not pd.isna(td["Prev_ATR_Ratio"].iloc[0]) else 0.02
                    vol_multiplier = atr_ratio / 0.02 if atr_ratio > 0 else 1.0
                    vol_multiplier = max(0.5, min(vol_multiplier, 2.5))
                    base_stop = stop_loss * vol_multiplier

            return base_stop

        # -- Close positions --
        closed = []
        for pos in positions:
            cp = pos.get("current_price", pos["entry_price"])
            ep = pos["entry_price"]
            ret = (cp / ep - 1) * 100
            eff_stop = get_stop(pos)

            # Trailing stop check
            trail_triggered = False
            if trailing_stop is not None and pos.get("peak_price"):
                peak = pos["peak_price"]
                trail_ret = (cp / peak - 1) * 100
                if trail_ret <= -trailing_stop:
                    trail_triggered = True

            # Dynamic hold: exit early if momentum fades
            dynamic_exit = False
            if dynamic_hold and pos["days_held"] >= 3:
                if ret < -2.0:
                    dynamic_exit = True

            # Partial profit: sell half at +5%
            partial_triggered = False
            if partial_profit and not pos.get("partial_sold", False) and ret >= 5.0:
                partial_triggered = True

            # Determine if we should close
            should_close = False
            reason = "hold"

            if ret <= eff_stop:
                should_close = True
                reason = "stop"
            elif trail_triggered:
                should_close = True
                reason = "trail_stop"
            elif dynamic_exit:
                should_close = True
                reason = "dynamic_exit"
            elif pos["days_held"] >= hold_days:
                should_close = True
                reason = "hold"

            if partial_triggered and not should_close:
                half_shares = pos["shares"] // 2
                if half_shares > 0:
                    sell_cost = half_shares * cp * (cost_bps / 2 / 10000)
                    proceeds = half_shares * cp - sell_cost
                    half_cost_basis = pos["cost_basis"] * (half_shares / pos["shares"])
                    net_ret = (proceeds / half_cost_basis - 1) * 100
                    trade_log.append({
                        "Ticker": pos["ticker"], "Entry": pos["entry_date"],
                        "Exit": today, "Return": net_ret,
                        "Days": pos["days_held"], "Reason": "partial_profit",
                        "Shares_Sold": half_shares,
                    })
                    capital += proceeds
                    pos["shares"] -= half_shares
                    pos["cost_basis"] -= half_cost_basis
                    pos["partial_sold"] = True

            if should_close:
                sell_cost = pos["shares"] * cp * (cost_bps / 2 / 10000)
                proceeds = pos["shares"] * cp - sell_cost
                net_ret = (proceeds / pos["cost_basis"] - 1) * 100
                trade_log.append({
                    "Ticker": pos["ticker"], "Entry": pos["entry_date"],
                    "Exit": today, "Return": net_ret,
                    "Days": pos["days_held"], "Reason": reason,
                })
                capital += proceeds
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        # -- Regime filter --
        skip_trading = False
        if regime_filter:
            pv_now = capital + sum(
                p["shares"] * p.get("current_price", p["entry_price"]) for p in positions
            )
            recent_equity.append(pv_now)

            if regime_filter == "drawdown":
                if len(recent_equity) > 20:
                    peak_eq = max(recent_equity[-60:]) if len(recent_equity) > 60 else max(recent_equity)
                    dd = (pv_now / peak_eq - 1) * 100
                    if dd < -15:
                        skip_trading = True

            elif regime_filter == "momentum":
                today_df_all = df[df.index == today]
                if len(today_df_all) > 50 and "Prev_Return_20d" in today_df_all.columns:
                    mkt_mom = today_df_all["Prev_Return_20d"].median()
                    if not pd.isna(mkt_mom) and mkt_mom < -3:
                        skip_trading = True

        # -- Score today's stocks --
        scores = pd.DataFrame()
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

        # -- Open new positions --
        open_slots = max_positions - len(positions)
        if open_slots > 0 and len(scores) > 0 and not skip_trading:
            mask = (scores["Prob"] >= min_prob) & (scores["Prob"] <= max_prob)
            if min_mom > -900:
                mask &= scores["Mom_20d"] > min_mom
            scores = scores[mask].sort_values("Prob", ascending=False)

            held = [p["ticker"] for p in positions]
            scores = scores[~scores["Ticker"].isin(held)]

            n_buy = min(open_slots, max_picks, len(scores))
            if n_buy > 0:
                total_eq = capital + sum(
                    p["shares"] * p.get("current_price", p["entry_price"])
                    for p in positions
                )
                slot_size = total_eq / max_positions

                t1, t2 = tier_ratio
                tiers = [t1/100, t2/100]
                tier_weights = tiers[:n_buy]
                scale_factor = n_buy / sum(tier_weights) if sum(tier_weights) > 0 else 1

                for j in range(n_buy):
                    pick = scores.iloc[j]
                    ep = float(pick["Close"])
                    tier = tiers[j] if j < len(tiers) else t2/100
                    alloc = min(capital, slot_size * tier * scale_factor)
                    buy_cost = alloc * (cost_bps / 2 / 10000)
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
                            "peak_price": ep,
                            "partial_sold": False,
                        })

        # Portfolio value
        pv = capital
        for pos in positions:
            pv += pos["shares"] * pos.get("current_price", pos["entry_price"])
        equity_curve.append({"Date": today, "Value": pv})

        if verbose and day_i % 100 == 0:
            ret_pct = (pv / start_capital - 1) * 100
            print(f"    Day {day_i:4d} | {today.date()} | ${pv:>10,.0f} ({ret_pct:+.1f}%) | "
                  f"Pos: {len(positions)} | Trades: {len(trade_log)}")

    # Close remaining
    for pos in positions:
        cp = pos.get("current_price", pos["entry_price"])
        ret = (cp / pos["entry_price"] - 1) * 100
        sell_cost = pos["shares"] * cp * (cost_bps / 2 / 10000)
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
    total_ret = (final / start_capital - 1) * 100
    days_traded = (eq.index[-1] - eq.index[0]).days
    ann_ret = ((final / start_capital) ** (365.25 / max(days_traded, 1)) - 1) * 100
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


# ============================================================================
#  PHASE 1: MODEL HYPERPARAMETERS
# ============================================================================

def phase1_model_params(df, features):
    """Test model hyperparameters with structured sampling."""
    print()
    print("=" * 100)
    print("  PHASE 1: MODEL HYPERPARAMETERS")
    print("=" * 100)

    # Key dimensions to test
    combos = [
        # (big_move, n_est, depth, lr, mcw, lookback, retrain)
        # Vary big_move threshold
        (3.0,  150, 5, 0.05, 20, 252, 63),
        (4.0,  150, 5, 0.05, 20, 252, 63),
        (5.0,  150, 5, 0.05, 20, 252, 63),   # current baseline
        (6.0,  150, 5, 0.05, 20, 252, 63),
        (7.0,  150, 5, 0.05, 20, 252, 63),

        # Vary n_estimators
        (5.0,  100, 5, 0.05, 20, 252, 63),
        (5.0,  250, 5, 0.05, 20, 252, 63),
        (5.0,  400, 5, 0.05, 20, 252, 63),

        # Vary max_depth
        (5.0,  150, 3, 0.05, 20, 252, 63),
        (5.0,  150, 7, 0.05, 20, 252, 63),
        (5.0,  150, 9, 0.05, 20, 252, 63),

        # Vary learning_rate
        (5.0,  150, 5, 0.02, 20, 252, 63),
        (5.0,  150, 5, 0.10, 20, 252, 63),
        (5.0,  150, 5, 0.15, 20, 252, 63),

        # Vary min_child_weight
        (5.0,  150, 5, 0.05, 5,  252, 63),
        (5.0,  150, 5, 0.05, 10, 252, 63),
        (5.0,  150, 5, 0.05, 40, 252, 63),
        (5.0,  150, 5, 0.05, 60, 252, 63),

        # Vary lookback
        (5.0,  150, 5, 0.05, 20, 126, 63),
        (5.0,  150, 5, 0.05, 20, 504, 63),
        (5.0,  150, 5, 0.05, 20, 756, 63),

        # Vary retrain frequency
        (5.0,  150, 5, 0.05, 20, 252, 21),
        (5.0,  150, 5, 0.05, 20, 252, 42),
        (5.0,  150, 5, 0.05, 20, 252, 126),

        # Promising combos (mix best from each dimension)
        (4.0,  250, 5, 0.05, 20, 252, 42),
        (3.0,  250, 7, 0.05, 10, 504, 42),
        (4.0,  150, 7, 0.03, 20, 504, 42),
        (5.0,  250, 7, 0.05, 10, 504, 42),
        (4.0,  250, 5, 0.10, 10, 252, 42),
        (3.0,  400, 5, 0.03, 20, 504, 21),
        (6.0,  250, 3, 0.10, 40, 252, 63),
        (4.0,  400, 7, 0.05, 10, 252, 21),
        (5.0,  400, 5, 0.03, 10, 504, 42),
        (3.0,  150, 7, 0.10, 10, 252, 42),
        (4.0,  250, 3, 0.05, 40, 504, 63),
        (5.0,  250, 5, 0.03, 10, 756, 42),
    ]

    print(f"  Testing {len(combos)} model configurations")
    print(f"  Fixed: hold=7d, stop=-7%, picks=2, max_pos=2, tiered 60/40")
    print()

    results = []
    t0 = time.time()

    for i, (bm, ne, dep, lr_val, mcw, lb, rt) in enumerate(combos):
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(combos) - i) if i > 0 else 0
        sys.stdout.write(f"\r  [{i+1}/{len(combos)}] bm={bm} est={ne} d={dep} lr={lr_val} "
                         f"mcw={mcw} lb={lb} rt={rt}  "
                         f"({elapsed:.0f}s, ~{eta:.0f}s left)   ")
        sys.stdout.flush()

        params = {
            "big_move": bm, "n_est": ne, "depth": dep, "lr": lr_val,
            "mcw": mcw, "lookback": lb, "retrain": rt,
            "min_prob": 0.50, "max_prob": 0.85, "min_mom": 0,
            "tier_ratio": (60, 40), "stop_loss": -7.0,
        }

        r = run_ultimate_backtest(df, features, params)
        if r:
            results.append({
                "big_move": bm, "n_est": ne, "depth": dep, "lr": lr_val,
                "mcw": mcw, "lookback": lb, "retrain": rt,
                **{k: v for k, v in r.items() if k not in ("equity", "trades")},
            })

    elapsed = time.time() - t0
    print(f"\n  Done: {len(results)} results in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print()

    res_df = pd.DataFrame(results)

    # Show top 15
    print(f"  TOP 15 BY SHARPE:")
    top = res_df.nlargest(15, "sharpe")
    print(f"  {'Rk':<3} {'BM':>4} {'Est':>4} {'Dep':>3} {'LR':>5} {'MCW':>4} {'LB':>4} {'RT':>3} "
          f"{'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>7} {'WR':>5} {'PF':>5} {'#Tr':>5}")
    print("  " + "-"*90)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        print(f"  {rank:<3} {row['big_move']:>4.0f} {row['n_est']:>4.0f} {row['depth']:>3.0f} "
              f"{row['lr']:>5.2f} {row['mcw']:>4.0f} {row['lookback']:>4.0f} {row['retrain']:>3.0f} "
              f"{row['annual_return']:>+7.1f}% {row['sharpe']:>6.2f} {row['max_dd']:>6.1f}% "
              f"{row['win_rate']:>4.0f}% {row['profit_factor']:>4.1f} {row['n_trades']:>5.0f}")

    # Sensitivity analysis
    print()
    print(f"  SENSITIVITY:")
    for param in ["big_move", "n_est", "depth", "lr", "mcw", "lookback", "retrain"]:
        grp = res_df.groupby(param).agg({"annual_return":"mean","sharpe":"mean","max_dd":"mean"}).round(2)
        vals = []
        for val, row in grp.iterrows():
            vals.append(f"{val}->{row['annual_return']:+.0f}%/{row['sharpe']:.2f}")
        print(f"    {param:>10}: {' | '.join(vals)}")

    # Winner (composite score)
    res_df["score"] = (
        res_df["sharpe"].rank(pct=True) * 0.35 +
        res_df["annual_return"].rank(pct=True) * 0.30 +
        res_df["sortino"].rank(pct=True) * 0.15 +
        res_df["profit_factor"].rank(pct=True) * 0.10 +
        res_df["calmar"].rank(pct=True) * 0.10
    )
    best = res_df.loc[res_df["score"].idxmax()]
    print()
    print(f"  * PHASE 1 WINNER: bm={best['big_move']:.0f} est={best['n_est']:.0f} "
          f"d={best['depth']:.0f} lr={best['lr']:.2f} mcw={best['mcw']:.0f} "
          f"lb={best['lookback']:.0f} rt={best['retrain']:.0f}")
    print(f"    AnnRet={best['annual_return']:+.1f}% Sharpe={best['sharpe']:.2f} "
          f"DD={best['max_dd']:.1f}% WR={best['win_rate']:.0f}% PF={best['profit_factor']:.2f}")

    winner = {
        "big_move": best["big_move"], "n_est": int(best["n_est"]),
        "depth": int(best["depth"]), "lr": best["lr"],
        "mcw": int(best["mcw"]), "lookback": int(best["lookback"]),
        "retrain": int(best["retrain"]),
    }

    return winner, res_df


# ============================================================================
#  PHASE 2: FILTER & SIZING
# ============================================================================

def phase2_filters(df, features, model_winner):
    """Test filter and sizing params using Phase 1 winner model."""
    print()
    print("=" * 100)
    print("  PHASE 2: FILTERS & SIZING")
    print("=" * 100)

    grid = {
        "min_prob":   [0.40, 0.45, 0.50, 0.55, 0.60],
        "max_prob":   [0.85, 0.90, 1.0],
        "min_mom":    [-999, 0, 2],
        "tier_ratio": [(50,50), (60,40), (70,30)],
    }

    combos = list(product(grid["min_prob"], grid["max_prob"],
                          grid["min_mom"], grid["tier_ratio"]))

    # Filter impossible (min_prob >= max_prob)
    combos = [(mp, xp, mm, tr) for mp, xp, mm, tr in combos if mp < xp]

    print(f"  Testing {len(combos)} filter/sizing combos")
    print(f"  Using Phase 1 winner model: bm={model_winner['big_move']} "
          f"est={model_winner['n_est']} d={model_winner['depth']} "
          f"lr={model_winner['lr']} mcw={model_winner['mcw']}")
    print()

    results = []
    t0 = time.time()

    for i, (mp, xp, mm, tr) in enumerate(combos):
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(combos) - i) if i > 0 else 0
        mom_label = 'off' if mm < -900 else f'{mm}%'
        sys.stdout.write(f"\r  [{i+1}/{len(combos)}] prob={mp:.2f}-{xp:.2f} "
                         f"mom={mom_label:>4} tier={tr}  "
                         f"({elapsed:.0f}s, ~{eta:.0f}s left)   ")
        sys.stdout.flush()

        params = {
            **model_winner,
            "min_prob": mp, "max_prob": xp, "min_mom": mm,
            "tier_ratio": tr, "stop_loss": -7.0,
        }

        r = run_ultimate_backtest(df, features, params)
        if r:
            results.append({
                "min_prob": mp, "max_prob": xp, "min_mom": mm,
                "tier_1": tr[0], "tier_2": tr[1],
                **{k: v for k, v in r.items() if k not in ("equity", "trades")},
            })

    elapsed = time.time() - t0
    print(f"\n  Done: {len(results)} results in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print()

    res_df = pd.DataFrame(results)

    # Top 15
    print(f"  TOP 15 BY SHARPE:")
    top = res_df.nlargest(15, "sharpe")
    print(f"  {'Rk':<3} {'MinP':>5} {'MaxP':>5} {'Mom':>4} {'Tier':>7} "
          f"{'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>7} {'WR':>5} {'PF':>5} {'#Tr':>5} {'Final':>10}")
    print("  " + "-"*90)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        mom_str = "off" if row['min_mom'] < -900 else f"{row['min_mom']:.0f}%"
        print(f"  {rank:<3} {row['min_prob']:>5.2f} {row['max_prob']:>5.2f} {mom_str:>4} "
              f"{row['tier_1']:.0f}/{row['tier_2']:.0f}   "
              f"{row['annual_return']:>+7.1f}% {row['sharpe']:>6.2f} {row['max_dd']:>6.1f}% "
              f"{row['win_rate']:>4.0f}% {row['profit_factor']:>4.1f} {row['n_trades']:>5.0f} "
              f"${row['final_value']:>9,.0f}")

    # Sensitivity
    print()
    print(f"  SENSITIVITY:")
    for param in ["min_prob", "max_prob", "min_mom"]:
        grp = res_df.groupby(param).agg({"annual_return":"mean","sharpe":"mean"}).round(2)
        vals = []
        for val, row in grp.iterrows():
            label = "off" if param == "min_mom" and val < -900 else f"{val}"
            vals.append(f"{label}->{row['annual_return']:+.0f}%/{row['sharpe']:.2f}")
        print(f"    {param:>10}: {' | '.join(vals)}")

    # Tier analysis
    res_df["tier_label"] = res_df.apply(lambda r: f"{r['tier_1']:.0f}/{r['tier_2']:.0f}", axis=1)
    grp = res_df.groupby("tier_label").agg({"annual_return":"mean","sharpe":"mean"}).round(2)
    vals = []
    for val, row in grp.iterrows():
        vals.append(f"{val}->{row['annual_return']:+.0f}%/{row['sharpe']:.2f}")
    print(f"    {'tier':>10}: {' | '.join(vals)}")

    # Winner
    res_df["score"] = (
        res_df["sharpe"].rank(pct=True) * 0.35 +
        res_df["annual_return"].rank(pct=True) * 0.30 +
        res_df["sortino"].rank(pct=True) * 0.15 +
        res_df["profit_factor"].rank(pct=True) * 0.10 +
        res_df["calmar"].rank(pct=True) * 0.10
    )
    best = res_df.loc[res_df["score"].idxmax()]
    mom_label = 'off' if best['min_mom'] < -900 else f"{best['min_mom']:.0f}%"
    print()
    print(f"  * PHASE 2 WINNER: prob={best['min_prob']:.2f}-{best['max_prob']:.2f} "
          f"mom={mom_label} tier={best['tier_1']:.0f}/{best['tier_2']:.0f}")
    print(f"    AnnRet={best['annual_return']:+.1f}% Sharpe={best['sharpe']:.2f} "
          f"DD={best['max_dd']:.1f}% WR={best['win_rate']:.0f}% PF={best['profit_factor']:.2f}")

    winner = {
        "min_prob": best["min_prob"], "max_prob": best["max_prob"],
        "min_mom": best["min_mom"], "tier_ratio": (int(best["tier_1"]), int(best["tier_2"])),
    }

    return winner, res_df


# ============================================================================
#  PHASE 3: DYNAMIC / ADAPTIVE FEATURES
# ============================================================================

def phase3_dynamic(df, features, model_winner, filter_winner):
    """Test dynamic/adaptive features using Phase 1+2 winners.

    Strategy: Test each feature independently first, then combine promising ones.
    """
    print()
    print("=" * 100)
    print("  PHASE 3: DYNAMIC & ADAPTIVE FEATURES")
    print("=" * 100)

    # Base params (no dynamic features)
    base = {**model_winner, **filter_winner, "stop_loss": -7.0,
            "trailing_stop": None, "vol_stop": False,
            "regime_filter": None, "dynamic_hold": False, "partial_profit": False}

    combos = []
    labels = []

    # Baseline
    combos.append(dict(base)); labels.append("BASELINE")

    # Stop Loss variations
    for sl in [-3.0, -5.0, -10.0, -15.0]:
        c = dict(base); c["stop_loss"] = sl
        combos.append(c); labels.append(f"SL={sl:.0f}%")

    # Trailing stop (independent)
    for ts in [3.0, 5.0, 7.0, 10.0]:
        c = dict(base); c["trailing_stop"] = ts
        combos.append(c); labels.append(f"Trail={ts:.0f}%")

    # Vol-adjusted stop
    c = dict(base); c["vol_stop"] = True
    combos.append(c); labels.append("VolStop")

    # Vol stop + different base SL
    for sl in [-5.0, -10.0]:
        c = dict(base); c["vol_stop"] = True; c["stop_loss"] = sl
        combos.append(c); labels.append(f"VolStop+SL={sl:.0f}%")

    # Regime filters
    for rf in ["drawdown", "momentum"]:
        c = dict(base); c["regime_filter"] = rf
        combos.append(c); labels.append(f"Regime={rf}")

    # Dynamic hold
    c = dict(base); c["dynamic_hold"] = True
    combos.append(c); labels.append("DynHold")

    # Partial profit
    c = dict(base); c["partial_profit"] = True
    combos.append(c); labels.append("PartialProfit")

    # Trail + vol combos
    for ts in [5.0, 7.0]:
        c = dict(base); c["trailing_stop"] = ts; c["vol_stop"] = True
        combos.append(c); labels.append(f"Trail={ts:.0f}%+Vol")

    # Trail + dynamic hold
    for ts in [5.0, 7.0]:
        c = dict(base); c["trailing_stop"] = ts; c["dynamic_hold"] = True
        combos.append(c); labels.append(f"Trail={ts:.0f}%+DynH")

    # Regime + trail
    for rf in ["drawdown", "momentum"]:
        c = dict(base); c["regime_filter"] = rf; c["trailing_stop"] = 5.0
        combos.append(c); labels.append(f"Regime={rf}+Trail=5%")

    # Kitchen sink combos
    for sl, ts, vs, rf, dh, pp in [
        (-7,  5,  True,  None,       False, False),
        (-7,  5,  True,  "drawdown", False, False),
        (-7,  5,  False, None,       True,  False),
        (-7,  None, True, "drawdown", True,  False),
        (-7,  5,  True,  None,       True,  True),
        (-5,  5,  True,  None,       True,  False),
        (-10, 7,  True,  None,       False, False),
        (-7,  7,  True,  "momentum", True,  False),
        (-5,  3,  True,  None,       True,  True),
        (-10, 10, False, None,       False, False),
    ]:
        c = dict(base)
        c["stop_loss"] = sl; c["trailing_stop"] = ts; c["vol_stop"] = vs
        c["regime_filter"] = rf; c["dynamic_hold"] = dh; c["partial_profit"] = pp
        combos.append(c); labels.append(f"Mix:sl={sl}/t={ts}/v={vs}/r={rf}/d={dh}/p={pp}")

    print(f"  Testing {len(combos)} dynamic feature combos")
    print(f"  Using Phase 1+2 winners as base")
    print()

    results = []
    t0 = time.time()

    for i, params in enumerate(combos):
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(combos) - i) if i > 0 else 0
        label = labels[i] if i < len(labels) else "combo"
        sys.stdout.write(f"\r  [{i+1}/{len(combos)}] {label:<40}  "
                         f"({elapsed:.0f}s, ~{eta:.0f}s left)   ")
        sys.stdout.flush()

        r = run_ultimate_backtest(df, features, params)
        if r:
            results.append({
                "label": label,
                "stop_loss": params.get("stop_loss", -7),
                "trailing_stop": params.get("trailing_stop") if params.get("trailing_stop") else "none",
                "vol_stop": params.get("vol_stop", False),
                "regime_filter": params.get("regime_filter") if params.get("regime_filter") else "none",
                "dynamic_hold": params.get("dynamic_hold", False),
                "partial_profit": params.get("partial_profit", False),
                **{k: v for k, v in r.items() if k not in ("equity", "trades")},
            })

    elapsed = time.time() - t0
    print(f"\n  Done: {len(results)} results in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print()

    res_df = pd.DataFrame(results)

    # Top 20 by Sharpe
    print(f"  TOP 20 BY SHARPE:")
    top = res_df.nlargest(20, "sharpe")
    print(f"  {'Rk':<3} {'Label':<35} "
          f"{'AnnRet':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'WR':>5} {'PF':>5} {'#Tr':>5} {'Final':>10}")
    print("  " + "-"*100)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        print(f"  {rank:<3} {str(row['label']):<35} "
              f"{row['annual_return']:>+7.1f}% {row['sharpe']:>6.2f} {row['sortino']:>7.2f} "
              f"{row['max_dd']:>6.1f}% {row['win_rate']:>4.0f}% {row['profit_factor']:>4.1f} "
              f"{row['n_trades']:>5.0f} ${row['final_value']:>9,.0f}")

    # Top 20 by return
    print()
    print(f"  TOP 20 BY ANNUAL RETURN:")
    top_ret = res_df.nlargest(20, "annual_return")
    print(f"  {'Rk':<3} {'Label':<35} "
          f"{'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>7} {'Final':>10}")
    print("  " + "-"*75)
    for rank, (_, row) in enumerate(top_ret.iterrows(), 1):
        print(f"  {rank:<3} {str(row['label']):<35} "
              f"{row['annual_return']:>+7.1f}% {row['sharpe']:>6.2f} "
              f"{row['max_dd']:>6.1f}% ${row['final_value']:>9,.0f}")

    # Sensitivity
    print()
    print(f"  SENSITIVITY:")
    for param in ["stop_loss", "trailing_stop", "vol_stop", "regime_filter",
                   "dynamic_hold", "partial_profit"]:
        grp = res_df.groupby(param).agg({
            "annual_return":"mean", "sharpe":"mean", "max_dd":"mean"
        }).round(2)
        vals = []
        for val, row in grp.iterrows():
            vals.append(f"{val}->{row['annual_return']:+.0f}%/{row['sharpe']:.2f}/DD{row['max_dd']:.0f}%")
        print(f"    {param:>15}: {' | '.join(vals)}")

    # Winner
    res_df["score"] = (
        res_df["sharpe"].rank(pct=True) * 0.30 +
        res_df["annual_return"].rank(pct=True) * 0.25 +
        res_df["sortino"].rank(pct=True) * 0.15 +
        res_df["calmar"].rank(pct=True) * 0.10 +
        res_df["profit_factor"].rank(pct=True) * 0.10 +
        res_df["max_dd"].rank(pct=True, ascending=False) * 0.05 +
        res_df["win_rate"].rank(pct=True) * 0.05
    )
    best = res_df.loc[res_df["score"].idxmax()]

    print()
    print(f"  * PHASE 3 WINNER: {best['label']}")
    print(f"    Stop Loss:      {best['stop_loss']:.0f}%")
    print(f"    Trailing Stop:  {best['trailing_stop']}")
    print(f"    Vol-Adj Stop:   {best['vol_stop']}")
    print(f"    Regime Filter:  {best['regime_filter']}")
    print(f"    Dynamic Hold:   {best['dynamic_hold']}")
    print(f"    Partial Profit: {best['partial_profit']}")
    print(f"    ---")
    print(f"    AnnRet={best['annual_return']:+.1f}% Sharpe={best['sharpe']:.2f} "
          f"Sortino={best['sortino']:.2f} DD={best['max_dd']:.1f}% "
          f"WR={best['win_rate']:.0f}% PF={best['profit_factor']:.2f}")

    winner = {
        "stop_loss": best["stop_loss"],
        "trailing_stop": best["trailing_stop"] if best["trailing_stop"] != "none" else None,
        "vol_stop": best["vol_stop"],
        "regime_filter": best["regime_filter"] if best["regime_filter"] != "none" else None,
        "dynamic_hold": best["dynamic_hold"],
        "partial_profit": best["partial_profit"],
    }

    return winner, res_df


# ============================================================================
#  FINAL: COMBINED WINNER + DETAILED RUN
# ============================================================================

def final_combined(df, features, model_w, filter_w, dynamic_w):
    """Run the combined winner in detail and compare to old config."""
    print()
    print("=" * 100)
    print("  FINAL: COMBINED WINNER vs OLD CONFIG")
    print("=" * 100)

    # Combined winner
    final_params = {**model_w, **filter_w, **dynamic_w}

    print()
    print(f"  * ULTIMATE CONFIGURATION:")
    print(f"    Model:   bm={model_w['big_move']} est={model_w['n_est']} d={model_w['depth']} "
          f"lr={model_w['lr']} mcw={model_w['mcw']}")
    print(f"    Train:   lookback={model_w['lookback']} retrain={model_w['retrain']}")
    print(f"    Filters: prob={filter_w['min_prob']:.2f}-{filter_w['max_prob']:.2f} "
          f"mom={'off' if filter_w['min_mom']<-900 else str(filter_w['min_mom'])+'%'}")
    print(f"    Sizing:  tier={filter_w['tier_ratio']}")

    ts = dynamic_w.get('trailing_stop')
    rf = dynamic_w.get('regime_filter')
    print(f"    Stop:    fixed={dynamic_w['stop_loss']:.0f}% "
          f"trail={'off' if not ts else str(ts)+'%'} "
          f"vol={'Y' if dynamic_w.get('vol_stop') else 'N'}")
    print(f"    Dynamic: regime={'off' if not rf else rf} "
          f"dynHold={'Y' if dynamic_w.get('dynamic_hold') else 'N'} "
          f"partial={'Y' if dynamic_w.get('partial_profit') else 'N'}")

    # Run winner in detail
    print()
    print(f"  Running ULTIMATE config:")
    r_new = run_ultimate_backtest(df, features, final_params, verbose=True)

    # Run old config for comparison
    print()
    print(f"  Running OLD config (bm=5, est=150, d=5, lr=0.05, equal 50/50):")
    old_params = {
        "big_move": 5.0, "n_est": 150, "depth": 5, "lr": 0.05, "mcw": 20,
        "lookback": 252, "retrain": 63,
        "min_prob": 0.50, "max_prob": 0.85, "min_mom": 0,
        "tier_ratio": (50, 50), "stop_loss": -7.0,
    }
    r_old = run_ultimate_backtest(df, features, old_params, verbose=True)

    # Also run current deployed config
    print()
    print(f"  Running CURRENT DEPLOYED config (same + tiered 60/40):")
    cur_params = {**old_params, "tier_ratio": (60, 40)}
    r_cur = run_ultimate_backtest(df, features, cur_params, verbose=True)

    # Compare
    print()
    print("=" * 100)
    print("  FINAL COMPARISON")
    print("=" * 100)
    print()
    print(f"  {'Config':<30} {'AnnRet':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} "
          f"{'WR':>5} {'PF':>5} {'#Tr':>5} {'Final':>12}")
    print("  " + "-"*95)

    configs = [
        ("OLD (equal 50/50)", r_old),
        ("CURRENT (tiered 60/40)", r_cur),
        ("* ULTIMATE (optimized)", r_new),
    ]

    for name, r in configs:
        if r:
            print(f"  {name:<30} {r['annual_return']:>+7.1f}% {r['sharpe']:>6.2f} "
                  f"{r['sortino']:>7.2f} {r['max_dd']:>6.1f}% {r['win_rate']:>4.0f}% "
                  f"{r['profit_factor']:>4.1f} {r['n_trades']:>5} ${r['final_value']:>11,.0f}")

    if r_new and r_old:
        improvement = r_new['annual_return'] - r_old['annual_return']
        sharpe_imp = r_new['sharpe'] - r_old['sharpe']
        print()
        print(f"  Improvement over OLD:")
        print(f"    Annual Return: {improvement:+.1f}% ({r_old['annual_return']:+.1f}% -> {r_new['annual_return']:+.1f}%)")
        print(f"    Sharpe:        {sharpe_imp:+.2f} ({r_old['sharpe']:.2f} -> {r_new['sharpe']:.2f})")
        print(f"    Max DD:        {r_new['max_dd'] - r_old['max_dd']:+.1f}pp ({r_old['max_dd']:.1f}% -> {r_new['max_dd']:.1f}%)")
        print(f"    Final Value:   ${r_new['final_value'] - r_old['final_value']:+,.0f}")

    # German tax
    if r_new:
        gross = r_new["annual_return"]
        gross_eur = 100000 * gross / 100
        tax = max(0, gross_eur - 1000) * 0.26375
        net = gross_eur - tax
        print()
        print(f"  German Tax (Abgeltungsteuer 26.375%):")
        print(f"    Gross: {gross:+.1f}%/yr (EUR {gross_eur:,.0f})")
        print(f"    Tax:   EUR {tax:,.0f}")
        print(f"    Net:   {net/1000:.1f}%/yr (EUR {net:,.0f})")

    # Yearly breakdown
    if r_new and len(r_new["trades"]) > 0:
        trades = r_new["trades"]
        trades["Year"] = pd.to_datetime(trades["Exit"]).dt.year
        yearly = trades.groupby("Year").agg(
            Trades=("Return", "count"),
            AvgRet=("Return", "mean"),
            WinRate=("Return", lambda x: (x > 0).mean() * 100),
            TotalRet=("Return", "sum"),
        ).round(2)
        print()
        print(f"  Yearly Breakdown:")
        for yr, row in yearly.iterrows():
            if row["TotalRet"] > 0:
                bar = "#" * max(1, int(row["TotalRet"] / 5))
            else:
                bar = "=" * max(1, int(abs(row["TotalRet"]) / 5))
            print(f"    {yr}: {row['Trades']:3.0f} trades | WR={row['WinRate']:.0f}% | "
                  f"Avg {row['AvgRet']:+.2f}% | Total {row['TotalRet']:+.1f}% {bar}")

        # By reason
        if "Reason" in trades.columns:
            print()
            print(f"  Exit Reason Breakdown:")
            reason_grp = trades.groupby("Reason").agg(
                Count=("Return", "count"),
                AvgRet=("Return", "mean"),
                WR=("Return", lambda x: (x > 0).mean() * 100),
            ).round(2)
            for reason, row in reason_grp.iterrows():
                print(f"    {reason:>15}: {row['Count']:4.0f} trades | WR={row['WR']:.0f}% | Avg {row['AvgRet']:+.2f}%")

    # Save ultimate params
    ultimate = {
        "model": {k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v for k, v in model_w.items()},
        "filters": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in filter_w.items() if k != "tier_ratio"},
        "tier_ratio": [int(x) for x in filter_w["tier_ratio"]],
        "dynamic": {k: (v if v is not None else "none") for k, v in dynamic_w.items()},
        "results": {
            "annual_return": round(float(r_new["annual_return"]), 1) if r_new else 0,
            "sharpe": round(float(r_new["sharpe"]), 2) if r_new else 0,
            "max_dd": round(float(r_new["max_dd"]), 1) if r_new else 0,
            "final_value": round(float(r_new["final_value"]), 0) if r_new else 0,
        }
    }
    with open(os.path.join("data", "ultimate_params.json"), "w") as f:
        json.dump(ultimate, f, indent=2)
    print()
    print(f"  Ultimate params saved to data/ultimate_params.json")

    return final_params, r_new


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == "__main__":
    total_start = time.time()

    cache_5y = os.path.join("data", "universe_prices_5y.csv")
    if not os.path.exists(cache_5y):
        print("ERROR: Need 5y data. Run: python mega_backtest.py --download")
        sys.exit(1)

    print("=" * 100)
    print("  ULTIMATE BACKTEST -- Optimizing EVERYTHING")
    print("=" * 100)

    print()
    print("Loading 5y data...")
    df_raw = pd.read_csv(cache_5y, index_col=0, parse_dates=True)
    print(f"Data: {len(df_raw):,} rows | {df_raw['Ticker'].nunique()} tickers | "
          f"{df_raw.index.min().date()} to {df_raw.index.max().date()}")

    print()
    print("Engineering features...")
    df, features = add_features(df_raw)
    gc.collect()
    print(f"  {len(features)} features")

    # -- PHASE 1: Model --
    model_winner, p1_results = phase1_model_params(df, features)
    p1_results_save = p1_results[[c for c in p1_results.columns if c not in ("equity","trades")]].copy()
    p1_results_save.to_csv(os.path.join("data", "ultimate_phase1.csv"), index=False)
    gc.collect()

    # -- PHASE 2: Filters --
    filter_winner, p2_results = phase2_filters(df, features, model_winner)
    p2_results_save = p2_results[[c for c in p2_results.columns if c not in ("equity","trades")]].copy()
    p2_results_save.to_csv(os.path.join("data", "ultimate_phase2.csv"), index=False)
    gc.collect()

    # -- PHASE 3: Dynamic --
    dynamic_winner, p3_results = phase3_dynamic(df, features, model_winner, filter_winner)
    p3_results_save = p3_results[[c for c in p3_results.columns if c not in ("equity","trades")]].copy()
    p3_results_save.to_csv(os.path.join("data", "ultimate_phase3.csv"), index=False)
    gc.collect()

    # -- FINAL --
    final_params, final_result = final_combined(df, features, model_winner, filter_winner, dynamic_winner)

    total_elapsed = time.time() - total_start
    print()
    print("=" * 100)
    print(f"  ULTIMATE BACKTEST COMPLETE -- {total_elapsed/3600:.1f} hours")
    print(f"  Phase 1: {len(p1_results_save)} model combos")
    print(f"  Phase 2: {len(p2_results_save)} filter combos")
    print(f"  Phase 3: {len(p3_results_save)} dynamic combos")
    print(f"  Total:   {len(p1_results_save)+len(p2_results_save)+len(p3_results_save)} configurations tested")
    print("=" * 100)
