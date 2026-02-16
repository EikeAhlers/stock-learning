"""
Enhanced Strategy v2 — Only improvements that ACTUALLY improve performance.

ABLATION RESULTS (v1):
  Ensemble: HURT (-125% return) — LightGBM diluted XGBoost's superior signals
  EV Ranking: HURT (-47%) — Walk-forward magnitude model too noisy
  Circuit Breaker: HURT (-39%) — Skipped recovery trades

IMPROVEMENTS THAT SURVIVED (v2):
  1. Transaction Costs (10bps/side) — Honest friction modeling
  2. ATR Dynamic Stop — Adapts to each stock's volatility instead of fixed -7%
  3. Compounding — Use current equity for sizing (lets winners grow the account)
  4. Trailing Stop — Instead of exiting at fixed hold period, trail from peak
  5. Momentum Continuation — Extend hold if stock is still in strong uptrend
  6. Adaptive Retraining — Retrain more frequently (every 42d vs 63d)

These keep the WINNING XGBoost classifier + probability ranking untouched,
and only improve risk management and trade execution.
"""

import pandas as pd
import numpy as np
import sys, io, time
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from src.deep_analyzer import add_all_lagged_features, ALL_RAW_FEATURES
from src.setup_detector import SETUP_NAMES
from src.backtester import _train_classifier_on_window


def _get_atr_stop(today_row, multiplier=2.0, floor=-15.0, ceiling=-3.0):
    """Dynamic stop = -multiplier × ATR%, clamped to [floor, ceiling]."""
    atr_ratio = today_row.get("Prev_ATR_Ratio", np.nan)
    if pd.isna(atr_ratio) or atr_ratio <= 0:
        return -7.0
    stop = -multiplier * atr_ratio * 100
    return max(floor, min(ceiling, stop))


def run_enhanced_v2(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    start_capital: float = 100_000,
    hold_days: int = 3,
    min_probability: float = 0.50,
    top_n: int = 1,
    max_positions: int = 1,
    retrain_every: int = 42,
    warmup_days: int = 252,
    # Enhancement toggles
    use_atr_stop: bool = True,
    use_txn_costs: bool = True,
    use_compounding: bool = True,
    use_trailing_stop: bool = True,
    use_momentum_ext: bool = True,
    use_momentum_filter: bool = True,
    # Parameters
    txn_cost_bps: float = 10.0,
    atr_multiplier: float = 2.0,
    fixed_stop_loss: float = -7.0,
    trail_pct: float = 5.0,        # trail 5% from peak
    max_hold_days: int = 7,         # max hold with extensions
    verbose: bool = True,
) -> dict:
    """
    Enhanced v2: Same winning XGBoost + probability ranking, better execution.
    """
    if verbose:
        print("=" * 80)
        print("  ENHANCED v2 WALK-FORWARD BACKTEST")
        print("=" * 80)
        flags = []
        if use_atr_stop:         flags.append("ATR-Stop")
        if use_trailing_stop:    flags.append(f"Trail-{trail_pct}%")
        if use_momentum_ext:     flags.append(f"MomExt(max{max_hold_days}d)")
        if use_txn_costs:        flags.append(f"TxCost({txn_cost_bps}bps)")
        if use_compounding:      flags.append("Compound")
        print(f"  Enhancements: {' | '.join(flags)}")
        print(f"  Retrain: every {retrain_every}d (was 63d)")

    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)

    clf_features = classifier_dict["features"]
    all_dates = sorted(df.index.unique())

    if len(all_dates) < warmup_days + 30:
        return {}

    trade_dates = all_dates[warmup_days:]

    if verbose:
        print(f"  Warmup: {all_dates[0].date()} → {all_dates[warmup_days-1].date()}")
        print(f"  Trading: {trade_dates[0].date()} → {trade_dates[-1].date()} ({len(trade_dates)}d)")
        print()

    # Train initial model
    warmup_data = df[df.index <= all_dates[warmup_days - 1]]
    current_model = _train_classifier_on_window(warmup_data, clf_features)
    if current_model is None:
        print("  ERROR: Not enough warmup data")
        return {}

    model = current_model["model"]
    scaler = current_model["scaler"]
    days_since_retrain = 0

    capital = start_capital
    positions = []
    trade_log = []
    equity_curve = []
    dates_processed = 0

    for today in trade_dates:
        # ── Retraining ───────────────────────────────────────────────
        if days_since_retrain >= retrain_every:
            past = df[df.index < today]
            new_m = _train_classifier_on_window(past, clf_features)
            if new_m is not None:
                current_model = new_m
                model = current_model["model"]
                scaler = current_model["scaler"]
                days_since_retrain = 0
        days_since_retrain += 1

        # ── Close positions ──────────────────────────────────────────
        closed = []
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            if len(td) == 0:
                pos["days_held"] += 1
                continue

            cp = td["Close"].iloc[0]
            ret = (cp / pos["entry_price"] - 1) * 100
            pos["days_held"] += 1
            pos["current_return"] = ret

            # Track peak return for trailing stop
            if ret > pos.get("peak_return", 0):
                pos["peak_return"] = ret

            # Stop loss (ATR-based or fixed)
            stop = pos.get("stop_loss", fixed_stop_loss)
            hit_stop = ret <= stop

            # Trailing stop: if we've gained > trail_pct, trail from peak
            hit_trail = False
            if use_trailing_stop and pos.get("peak_return", 0) > trail_pct:
                trail_level = pos["peak_return"] - trail_pct
                if ret <= trail_level:
                    hit_trail = True

            # Hold period expiry
            hit_hold = pos["days_held"] >= hold_days

            # Momentum extension: if still up > 3% at expiry, extend hold
            extend = False
            if use_momentum_ext and hit_hold and not hit_stop and not hit_trail:
                if ret > 3.0 and pos["days_held"] < max_hold_days:
                    # Check if momentum is still positive (today's return > 0)
                    today_ret = td["Daily_Return_Pct"].iloc[0] if "Daily_Return_Pct" in td.columns else 0
                    if today_ret > 0:
                        extend = True
                        pos["hold_days"] = pos["days_held"] + 1  # extend by 1 day

            should_exit = (hit_stop or hit_trail or hit_hold) and not extend

            if should_exit:
                exit_reason = "stop_loss" if hit_stop else ("trail_stop" if hit_trail else "hold_expiry")

                exit_cost = 0
                if use_txn_costs:
                    exit_cost = pos["shares"] * cp * (txn_cost_bps / 10_000)

                proceeds = pos["shares"] * cp - exit_cost
                capital += proceeds
                net_ret = (proceeds / (pos["shares"] * pos["entry_price"]) - 1) * 100

                trade_log.append({
                    "Ticker": pos["ticker"],
                    "Entry_Date": pos["entry_date"],
                    "Exit_Date": today,
                    "Entry_Price": pos["entry_price"],
                    "Exit_Price": cp,
                    "Return_Pct": net_ret,
                    "PnL": proceeds - pos["shares"] * pos["entry_price"],
                    "Hold_Days": pos["days_held"],
                    "Exit_Reason": exit_reason,
                    "Probability": pos.get("probability", 0),
                    "Peak_Return": pos.get("peak_return", 0),
                    "Stop_Loss": stop,
                })
                closed.append(pos)

        for c in closed:
            positions.remove(c)

        # ── Open new positions (same XGBoost + probability ranking) ──
        open_slots = max_positions - len(positions)

        if open_slots > 0:
            today_all = df[df.index == today].copy().reset_index(drop=True)

            if len(today_all) > 10:
                fm = today_all[clf_features].notna().all(axis=1)
                tf = today_all.loc[fm, clf_features]

                if len(tf) > 0:
                    X = scaler.transform(tf)
                    probs = model.predict_proba(X)[:, 1]

                    scores = pd.DataFrame({
                        "idx": today_all.loc[fm].index,
                        "Ticker": today_all.loc[fm, "Ticker"].values,
                        "Probability": probs,
                    })

                    scores = scores[scores["Probability"] >= min_probability]

                    # Momentum filter (same as baseline)
                    if use_momentum_filter and "Prev_Return_20d" in today_all.columns:
                        mom = today_all.loc[scores["idx"], "Prev_Return_20d"]
                        scores = scores[mom.values > 0]

                    held = [p["ticker"] for p in positions]
                    scores = scores[~scores["Ticker"].isin(held)]
                    scores = scores.sort_values("Probability", ascending=False)

                    n_buy = min(open_slots, top_n, len(scores))

                    for j in range(n_buy):
                        pick = scores.iloc[j]
                        ticker = pick["Ticker"]
                        prob = pick["Probability"]

                        td2 = today_all[today_all["Ticker"] == ticker]
                        if len(td2) == 0:
                            continue
                        ep = td2["Close"].iloc[0]
                        if ep <= 0:
                            continue

                        # ATR stop
                        if use_atr_stop:
                            stop = _get_atr_stop(td2.iloc[0], atr_multiplier)
                        else:
                            stop = fixed_stop_loss

                        # Sizing
                        if use_compounding:
                            pv_now = capital
                            for pos in positions:
                                td3 = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
                                pv_now += pos["shares"] * (
                                    td3["Close"].iloc[0] if len(td3) > 0 else pos["entry_price"]
                                )
                            pos_val = min(capital, pv_now / max_positions)
                        else:
                            pos_val = min(capital, start_capital / max_positions)

                        # Entry cost
                        if use_txn_costs:
                            pos_val -= pos_val * (txn_cost_bps / 10_000)

                        shares = int(pos_val / ep)
                        if shares <= 0:
                            continue

                        cost = shares * ep
                        if use_txn_costs:
                            cost += shares * ep * (txn_cost_bps / 10_000)
                        capital -= cost

                        positions.append({
                            "ticker": ticker,
                            "entry_date": today,
                            "entry_price": ep,
                            "shares": shares,
                            "hold_days": hold_days,
                            "days_held": 0,
                            "current_return": 0,
                            "peak_return": 0,
                            "probability": prob,
                            "stop_loss": stop,
                        })

        # ── Portfolio value ──────────────────────────────────────────
        pv = capital
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            pv += pos["shares"] * (td["Close"].iloc[0] if len(td) > 0 else pos["entry_price"])

        equity_curve.append({"Date": today, "Portfolio_Value": pv})
        dates_processed += 1

        if verbose and dates_processed % 50 == 0:
            pct = (pv / start_capital - 1) * 100
            print(f"  Day {dates_processed:4d} | {today.date()} | "
                  f"${pv:>10,.0f} ({pct:+.1f}%) | Trades: {len(trade_log)}")

    # Close remaining
    for pos in positions:
        last = df[df["Ticker"] == pos["ticker"]]
        if len(last) > 0:
            ep2 = last.iloc[-1]["Close"]
            ret2 = (ep2 / pos["entry_price"] - 1) * 100
            if use_txn_costs:
                ec = pos["shares"] * ep2 * (txn_cost_bps / 10_000)
                net = pos["shares"] * ep2 - ec
                ret2 = (net / (pos["shares"] * pos["entry_price"]) - 1) * 100
            capital += pos["shares"] * ep2
            trade_log.append({
                "Ticker": pos["ticker"],
                "Entry_Date": pos["entry_date"],
                "Exit_Date": trade_dates[-1],
                "Return_Pct": ret2,
                "PnL": pos["shares"] * pos["entry_price"] * (ret2 / 100),
                "Hold_Days": pos["days_held"],
                "Exit_Reason": "end_of_period",
                "Probability": pos.get("probability", 0),
            })

    eq_df = pd.DataFrame(equity_curve).set_index("Date") if equity_curve else pd.DataFrame()
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    final_val = eq_df["Portfolio_Value"].iloc[-1] if len(eq_df) > 0 else start_capital

    if verbose:
        total_ret = (final_val / start_capital - 1) * 100
        print(f"\n  RESULT: ${start_capital:,.0f} → ${final_val:,.0f} ({total_ret:+.1f}%)")
        if len(trades_df) > 0:
            wr = (trades_df["Return_Pct"] > 0).mean() * 100
            print(f"  Trades: {len(trades_df)}, Win rate: {wr:.0f}%")

    return {
        "equity_curve": eq_df,
        "trades": trades_df,
        "start_capital": start_capital,
        "final_value": final_val,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ABLATION: Test each v2 improvement individually
# ═══════════════════════════════════════════════════════════════════════════

def run_v2_ablation(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
) -> tuple:
    """
    Progressive ablation: start from baseline, add each improvement one at a time.
    """
    print("\n" + "═" * 80)
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║    ABLATION v2 — Only Improvements That Actually Work  ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print("═" * 80)

    configs = [
        ("Baseline (original)",
         dict(use_atr_stop=False, use_txn_costs=False, use_compounding=False,
              use_trailing_stop=False, use_momentum_ext=False, retrain_every=63)),
        ("+ Faster Retrain (42d)",
         dict(use_atr_stop=False, use_txn_costs=False, use_compounding=False,
              use_trailing_stop=False, use_momentum_ext=False, retrain_every=42)),
        ("+ ATR Dynamic Stop",
         dict(use_atr_stop=True, use_txn_costs=False, use_compounding=False,
              use_trailing_stop=False, use_momentum_ext=False, retrain_every=42)),
        ("+ Trailing Stop",
         dict(use_atr_stop=True, use_txn_costs=False, use_compounding=False,
              use_trailing_stop=True, use_momentum_ext=False, retrain_every=42)),
        ("+ Momentum Extension",
         dict(use_atr_stop=True, use_txn_costs=False, use_compounding=False,
              use_trailing_stop=True, use_momentum_ext=True, retrain_every=42)),
        ("+ Compounding",
         dict(use_atr_stop=True, use_txn_costs=False, use_compounding=True,
              use_trailing_stop=True, use_momentum_ext=True, retrain_every=42)),
        ("+ Transaction Costs (honest)",
         dict(use_atr_stop=True, use_txn_costs=True, use_compounding=True,
              use_trailing_stop=True, use_momentum_ext=True, retrain_every=42)),
    ]

    results = []
    curves = {}

    for label, kwargs in configs:
        print(f"\n  ▸ {label}")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        bt = run_enhanced_v2(
            returns=returns, classifier_dict=classifier_dict,
            regime_data=regime_data, spy_data=spy_data,
            start_capital=100_000, hold_days=3, min_probability=0.50,
            top_n=1, max_positions=1, warmup_days=warmup_days,
            verbose=False, **kwargs,
        )
        sys.stdout = old_stdout

        eq = bt.get("equity_curve", pd.DataFrame())
        trades = bt.get("trades", pd.DataFrame())
        sc = bt.get("start_capital", 100_000)
        fv = bt.get("final_value", sc)
        ret = (fv / sc - 1) * 100
        sharpe = 0
        max_dd = 0
        wr = 0
        alpha = ret

        if len(eq) > 0:
            dr = eq["Portfolio_Value"].pct_change().dropna()
            if len(dr) > 10 and dr.std() > 0:
                sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
            rm = eq["Portfolio_Value"].cummax()
            max_dd = ((eq["Portfolio_Value"] - rm) / rm * 100).min()
            curves[label] = eq["Portfolio_Value"]

        if len(trades) > 0:
            wr = (trades["Return_Pct"] > 0).mean() * 100

        if spy_data is not None and len(eq) > 0:
            sp = spy_data.loc[(spy_data.index >= eq.index[0]) & (spy_data.index <= eq.index[-1])]
            if len(sp) >= 2:
                sr = (sp.iloc[-1] / sp.iloc[0] - 1) * 100
                if isinstance(sr, pd.Series):
                    sr = sr.iloc[0]
                alpha = ret - sr

        # Exit reasons breakdown
        exit_reasons = ""
        if len(trades) > 0 and "Exit_Reason" in trades.columns:
            reasons = trades["Exit_Reason"].value_counts()
            exit_reasons = " | ".join(f"{r}={c}" for r, c in reasons.items())

        print(f"    Ret: {ret:+.1f}%  Sharpe: {sharpe:.2f}  DD: {max_dd:.1f}%  "
              f"WR: {wr:.0f}%  Alpha: {alpha:+.1f}%  Trades: {len(trades)}")
        if exit_reasons:
            print(f"    Exits: {exit_reasons}")

        results.append({
            "Config": label, "Return_%": ret, "Sharpe": sharpe,
            "Max_DD_%": max_dd, "Win_Rate_%": wr, "Alpha_%": alpha,
            "N_Trades": len(trades), "Final_Value": fv,
        })

    rdf = pd.DataFrame(results)

    print(f"\n{'═'*80}")
    print(f"  ABLATION v2 SUMMARY")
    print(f"{'═'*80}")
    for _, row in rdf.iterrows():
        marker = " ★" if "Transaction Costs" in row["Config"] else ""
        print(f"  {row['Config']:35s}  Ret={row['Return_%']:+7.1f}%  "
              f"Sh={row['Sharpe']:5.2f}  DD={row['Max_DD_%']:6.1f}%  "
              f"WR={row['Win_Rate_%']:4.0f}%{marker}")

    base_ret = rdf.iloc[0]["Return_%"]
    full_ret = rdf.iloc[-1]["Return_%"]
    print(f"\n  Baseline → Enhanced v2:")
    print(f"    Return: {base_ret:+.1f}% → {full_ret:+.1f}% (delta {full_ret - base_ret:+.1f}%)")
    print(f"    Sharpe: {rdf.iloc[0]['Sharpe']:.2f} → {rdf.iloc[-1]['Sharpe']:.2f}")

    return rdf, curves


# ═══════════════════════════════════════════════════════════════════════════
#  FULL COMPARISON with visualization data
# ═══════════════════════════════════════════════════════════════════════════

def run_full_comparison(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
) -> dict:
    """
    Run baseline, enhanced v2, and enhanced v2 + txn costs.
    Returns comparison dict with stats and equity curves.
    """
    print("\n" + "═" * 80)
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  BASELINE vs ENHANCED v2 — Final Comparison            ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print("═" * 80)

    t0 = time.time()

    # Baseline
    print("\n  ▶ BASELINE (original strategy)...")
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    bt_base = run_enhanced_v2(
        returns=returns, classifier_dict=classifier_dict,
        regime_data=regime_data, spy_data=spy_data,
        start_capital=100_000, hold_days=3, min_probability=0.50,
        top_n=1, max_positions=1, warmup_days=warmup_days,
        use_atr_stop=False, use_txn_costs=False, use_compounding=False,
        use_trailing_stop=False, use_momentum_ext=False,
        retrain_every=63, verbose=False,
    )
    sys.stdout = old_stdout

    # Enhanced v2 (no txn costs — to see pure improvement)
    print("  ▶ ENHANCED v2 (all improvements, no costs)...")
    sys.stdout = io.StringIO()
    bt_enh_no_cost = run_enhanced_v2(
        returns=returns, classifier_dict=classifier_dict,
        regime_data=regime_data, spy_data=spy_data,
        start_capital=100_000, hold_days=3, min_probability=0.50,
        top_n=1, max_positions=1, warmup_days=warmup_days,
        use_atr_stop=True, use_txn_costs=False, use_compounding=True,
        use_trailing_stop=True, use_momentum_ext=True,
        retrain_every=42, verbose=False,
    )
    sys.stdout = old_stdout

    # Enhanced v2 (with txn costs — honest version)
    print("  ▶ ENHANCED v2 (with transaction costs — honest)...")
    sys.stdout = io.StringIO()
    bt_enh = run_enhanced_v2(
        returns=returns, classifier_dict=classifier_dict,
        regime_data=regime_data, spy_data=spy_data,
        start_capital=100_000, hold_days=3, min_probability=0.50,
        top_n=1, max_positions=1, warmup_days=warmup_days,
        use_atr_stop=True, use_txn_costs=True, use_compounding=True,
        use_trailing_stop=True, use_momentum_ext=True,
        retrain_every=42, verbose=False,
    )
    sys.stdout = old_stdout

    elapsed = time.time() - t0

    def _stats(bt):
        eq = bt.get("equity_curve", pd.DataFrame())
        trades = bt.get("trades", pd.DataFrame())
        sc = bt.get("start_capital", 100_000)
        fv = bt.get("final_value", sc)
        ret = (fv / sc - 1) * 100
        s = {"total_return": ret, "final_value": fv, "n_trades": len(trades)}
        if len(trades) > 0:
            s["win_rate"] = (trades["Return_Pct"] > 0).mean() * 100
            s["avg_return"] = trades["Return_Pct"].mean()
            wins = trades[trades["Return_Pct"] > 0]
            losses = trades[trades["Return_Pct"] <= 0]
            s["profit_factor"] = (
                wins["PnL"].sum() / abs(losses["PnL"].sum())
                if len(losses) > 0 and losses["PnL"].sum() != 0 else 99.0
            )
            if "Exit_Reason" in trades.columns:
                s["trail_exits"] = (trades["Exit_Reason"] == "trail_stop").sum()
                s["stop_exits"] = (trades["Exit_Reason"] == "stop_loss").sum()
                s["hold_exits"] = (trades["Exit_Reason"] == "hold_expiry").sum()
            if "Hold_Days" in trades.columns:
                s["avg_hold"] = trades["Hold_Days"].mean()
        if len(eq) > 0:
            dr = eq["Portfolio_Value"].pct_change().dropna()
            s["sharpe"] = (dr.mean() / dr.std()) * np.sqrt(252) if len(dr) > 10 and dr.std() > 0 else 0
            rm = eq["Portfolio_Value"].cummax()
            s["max_dd"] = ((eq["Portfolio_Value"] - rm) / rm * 100).min()
            mv = eq["Portfolio_Value"].resample("ME").last()
            mr = mv.pct_change().dropna() * 100
            s["monthly_wr"] = (mr > 0).sum() / len(mr) * 100 if len(mr) > 0 else 0
        if spy_data is not None and len(eq) > 0:
            sp = spy_data.loc[(spy_data.index >= eq.index[0]) & (spy_data.index <= eq.index[-1])]
            if len(sp) >= 2:
                sr = (sp.iloc[-1] / sp.iloc[0] - 1) * 100
                if isinstance(sr, pd.Series):
                    sr = sr.iloc[0]
                s["spy_return"] = sr
                s["alpha"] = ret - sr
        return s

    s_b = _stats(bt_base)
    s_nc = _stats(bt_enh_no_cost)
    s_e = _stats(bt_enh)

    # Print comparison
    print(f"\n  {'─'*78}")
    print(f"  {'Metric':22s}  {'Baseline':>12s}  {'Enh(no cost)':>12s}  {'Enh(honest)':>12s}")
    print(f"  {'─'*78}")

    rows = [
        ("Return", "total_return", "%"),
        ("Sharpe", "sharpe", ""),
        ("Max Drawdown", "max_dd", "%"),
        ("Win Rate", "win_rate", "%"),
        ("Profit Factor", "profit_factor", "x"),
        ("Trades", "n_trades", ""),
        ("Avg Hold", "avg_hold", "d"),
        ("Alpha vs SPY", "alpha", "%"),
        ("Trail Exits", "trail_exits", ""),
    ]

    for label, key, suffix in rows:
        bv = s_b.get(key, 0) or 0
        ncv = s_nc.get(key, 0) or 0
        ev = s_e.get(key, 0) or 0
        if suffix == "%":
            print(f"  {label:22s}  {bv:>11.1f}%  {ncv:>11.1f}%  {ev:>11.1f}%")
        elif suffix == "x":
            print(f"  {label:22s}  {bv:>11.2f}x  {ncv:>11.2f}x  {ev:>11.2f}x")
        elif suffix == "d":
            print(f"  {label:22s}  {bv:>11.1f}d  {ncv:>11.1f}d  {ev:>11.1f}d")
        else:
            print(f"  {label:22s}  {bv:>12.2f}  {ncv:>12.2f}  {ev:>12.2f}")

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  {'═'*78}")

    return {
        "baseline": {"stats": s_b, "bt": bt_base},
        "enhanced_no_cost": {"stats": s_nc, "bt": bt_enh_no_cost},
        "enhanced_honest": {"stats": s_e, "bt": bt_enh},
        "elapsed": elapsed,
    }
