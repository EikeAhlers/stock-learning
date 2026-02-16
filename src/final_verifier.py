"""
Final Verifier — The most thorough, honest backtesting suite.

Addresses EVERY known weakness:
  1. Rolling Multi-Window Walk-Forward (12+ independent OOS tests)
  2. Ticker-Capped Backtesting (no single stock dominates)
  3. Regime-Adaptive Position Sizing (scale down in bear markets)
  4. Bootstrap Confidence Intervals (statistical proof of edge)
  5. Year-by-Year & Monthly Alpha Breakdown
  6. Leave-Top-Tickers-Out Test (remove best 3, still profitable?)
  7. Final Combined Scorecard

If the strategy passes ALL of these, it's as real as backtesting can prove.
"""
import pandas as pd
import numpy as np
import io, sys, time
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from src.backtester import _train_classifier_on_window
from src.deep_analyzer import add_all_lagged_features, ALL_RAW_FEATURES
from src.setup_detector import SETUP_NAMES

# ── Module-level defaults for enhanced strategy testing ──────────────────
# Set these before calling run_final_verification to enable compounding/costs
# e.g.:  final_verifier.ENHANCED_DEFAULTS["use_compounding"] = True
ENHANCED_DEFAULTS = {
    "use_compounding": False,   # True → position size scales with portfolio value
    "tx_cost_bps": 0.0,         # Transaction cost per side in basis points (10 = 0.1%)
}


# ═══════════════════════════════════════════════════════════════════════════
#  CORE: Walk-Forward Engine (shared by all tests)
# ═══════════════════════════════════════════════════════════════════════════

def _run_wf_backtest(
    df: pd.DataFrame,
    clf_features: list,
    trade_dates: list,
    all_dates: list,
    warmup_days: int = 252,
    start_capital: float = 100_000,
    hold_days: int = 3,
    min_probability: float = 0.50,
    stop_loss_pct: float = -7.0,
    top_n: int = 1,
    max_positions: int = 1,
    retrain_every: int = 63,
    strategy_type: str = "momentum",
    regime_data: pd.DataFrame = None,
    ticker_cap: int = 0,
    excluded_tickers: set = None,
    position_scale_fn=None,
) -> dict:
    """
    Honest walk-forward backtest with optional:
      - ticker frequency cap (max N trades per ticker)
      - ticker exclusion set
      - dynamic position scaling function
    """
    from sklearn.preprocessing import StandardScaler

    warmup_end_idx = 0
    for i, d in enumerate(all_dates):
        if d >= trade_dates[0]:
            warmup_end_idx = i
            break
    warmup_end_idx = max(warmup_end_idx - 1, warmup_days - 1)

    warmup_data = df[df.index <= all_dates[warmup_end_idx]]
    current_model = _train_classifier_on_window(warmup_data, clf_features)
    if current_model is None:
        return {}

    model = current_model["model"]
    scaler = current_model["scaler"]
    days_since_retrain = 0

    capital = start_capital
    positions = []
    trade_log = []
    equity_curve = []
    ticker_trade_count = {}

    for today in trade_dates:
        if days_since_retrain >= retrain_every:
            past = df[df.index < today]
            new_m = _train_classifier_on_window(past, clf_features)
            if new_m is not None:
                current_model = new_m
                model = current_model["model"]
                scaler = current_model["scaler"]
                days_since_retrain = 0
        days_since_retrain += 1

        # Regime check
        skip_today = False
        if strategy_type in ("momentum", "regime", "combo") and regime_data is not None:
            if today in regime_data.index:
                rv = regime_data.loc[today]
                rs = str(rv.get("Market_Regime", "")) if isinstance(rv, pd.Series) else str(rv)
                if strategy_type == "regime" and "bear" in rs.lower():
                    skip_today = True

        # Close positions
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
            should_exit = ret <= stop_loss_pct or pos["days_held"] >= hold_days
            if should_exit:
                _tx_bps_exit = ENHANCED_DEFAULTS.get("tx_cost_bps", 0.0)
                exit_proceeds = pos["shares"] * cp * (1 - _tx_bps_exit / 10_000)
                capital += exit_proceeds
                trade_log.append({
                    "Ticker": pos["ticker"], "Entry_Date": pos["entry_date"],
                    "Exit_Date": today, "Return_Pct": ret,
                    "PnL": exit_proceeds - pos["shares"] * pos["entry_price"],
                    "Hold_Days": pos["days_held"],
                    "Probability": pos.get("probability", 0),
                })
                closed.append(pos)
        for c in closed:
            positions.remove(c)

        # Open new
        open_slots = max_positions - len(positions)
        if open_slots > 0 and not skip_today:
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

                    # Momentum filter
                    if strategy_type in ("momentum", "combo"):
                        if "Prev_Return_20d" in today_all.columns:
                            mv = today_all.loc[scores["idx"], "Prev_Return_20d"]
                            scores = scores[mv.values > 0]

                    # Exclude tickers
                    if excluded_tickers:
                        scores = scores[~scores["Ticker"].isin(excluded_tickers)]

                    # Ticker frequency cap
                    if ticker_cap > 0:
                        scores = scores[scores["Ticker"].apply(
                            lambda t: ticker_trade_count.get(t, 0) < ticker_cap
                        )]

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

                        # Dynamic position sizing (compounding-aware)
                        _compound = ENHANCED_DEFAULTS.get("use_compounding", False)
                        _tx_bps = ENHANCED_DEFAULTS.get("tx_cost_bps", 0.0)
                        base_size = (capital / max_positions) if _compound else (start_capital / max_positions)
                        if position_scale_fn is not None:
                            scale = position_scale_fn(today, regime_data)
                            base_size *= scale

                        pos_val = min(capital, base_size)
                        shares = int(pos_val / ep)
                        if shares <= 0:
                            continue
                        entry_cost = shares * ep * (1 + _tx_bps / 10_000)
                        capital -= entry_cost
                        positions.append({
                            "ticker": ticker, "entry_date": today,
                            "entry_price": ep, "shares": shares,
                            "hold_days": hold_days, "days_held": 0,
                            "current_return": 0, "probability": prob,
                        })
                        ticker_trade_count[ticker] = ticker_trade_count.get(ticker, 0) + 1

        # Portfolio value
        pv = capital
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            pv += pos["shares"] * (td["Close"].iloc[0] if len(td) > 0 else pos["entry_price"])
        equity_curve.append({"Date": today, "Portfolio_Value": pv})

    # Close remaining
    _tx_bps_final = ENHANCED_DEFAULTS.get("tx_cost_bps", 0.0)
    for pos in positions:
        last = df[df["Ticker"] == pos["ticker"]]
        if len(last) > 0:
            ep2 = last.iloc[-1]["Close"]
            ret2 = (ep2 / pos["entry_price"] - 1) * 100
            final_proceeds = pos["shares"] * ep2 * (1 - _tx_bps_final / 10_000)
            capital += final_proceeds
            trade_log.append({
                "Ticker": pos["ticker"], "Entry_Date": pos["entry_date"],
                "Exit_Date": trade_dates[-1], "Return_Pct": ret2,
                "PnL": final_proceeds - pos["shares"] * pos["entry_price"],
                "Hold_Days": pos["days_held"],
                "Probability": pos.get("probability", 0),
            })

    eq_df = pd.DataFrame(equity_curve).set_index("Date") if equity_curve else pd.DataFrame()
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    final_val = eq_df["Portfolio_Value"].iloc[-1] if len(eq_df) > 0 else start_capital

    return {
        "equity_curve": eq_df, "trades": trades_df,
        "start_capital": start_capital, "final_value": final_val,
        "ticker_trade_count": ticker_trade_count,
    }


def _compute_stats(bt: dict, spy_data=None) -> dict:
    """Compute standard stats from a backtest result."""
    eq = bt.get("equity_curve", pd.DataFrame())
    trades = bt.get("trades", pd.DataFrame())
    sc = bt.get("start_capital", 100_000)
    fv = bt.get("final_value", sc)
    total_ret = (fv / sc - 1) * 100

    stats = {"total_return": total_ret, "final_value": fv, "n_trades": len(trades)}

    if len(trades) > 0:
        stats["win_rate"] = (trades["Return_Pct"] > 0).mean() * 100
        stats["avg_return"] = trades["Return_Pct"].mean()
        wins = trades[trades["Return_Pct"] > 0]
        losses = trades[trades["Return_Pct"] <= 0]
        stats["profit_factor"] = (
            wins["PnL"].sum() / abs(losses["PnL"].sum())
            if len(losses) > 0 and losses["PnL"].sum() != 0 else 99.0
        )

    if len(eq) > 0:
        rm = eq["Portfolio_Value"].cummax()
        dd = ((eq["Portfolio_Value"] - rm) / rm * 100).min()
        stats["max_dd"] = dd
        dr = eq["Portfolio_Value"].pct_change().dropna()
        stats["sharpe"] = (dr.mean() / dr.std()) * np.sqrt(252) if len(dr) > 10 and dr.std() > 0 else 0
        mv = eq["Portfolio_Value"].resample("ME").last()
        mr = mv.pct_change().dropna() * 100
        stats["monthly_win_rate"] = (mr > 0).sum() / len(mr) * 100 if len(mr) > 0 else 0

    if spy_data is not None and len(eq) > 0:
        sp = spy_data.loc[(spy_data.index >= eq.index[0]) & (spy_data.index <= eq.index[-1])]
        if len(sp) >= 2:
            sr = (sp.iloc[-1] / sp.iloc[0] - 1) * 100
            if isinstance(sr, pd.Series):
                sr = sr.iloc[0]
            stats["spy_return"] = sr
            stats["alpha"] = total_ret - sr

    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1: ROLLING MULTI-WINDOW WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════

def rolling_walk_forward(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    train_days: int = 189,
    test_days: int = 63,
    step_days: int = 42,
    warmup_days: int = 126,
) -> dict:
    """
    Rolling walk-forward validation with multiple independent OOS windows.

    For each window:
      - Train on [train_start, train_end] (uses warmup internally)
      - Test on [test_start, test_end]
      - Roll forward by step_days

    This produces 8-15 independent out-of-sample tests.
    The strategy must be profitable in MOST of them.
    """
    print("=" * 80)
    print("  TEST 1: ROLLING MULTI-WINDOW WALK-FORWARD")
    print("  Multiple independent out-of-sample tests across the full timeline")
    print("=" * 80)

    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    all_dates = sorted(df.index.unique())
    n_dates = len(all_dates)

    windows = []
    start = warmup_days
    while start + train_days + test_days <= n_dates:
        train_start = start
        train_end = start + train_days - 1
        test_start = start + train_days
        test_end = min(start + train_days + test_days - 1, n_dates - 1)
        windows.append((train_start, train_end, test_start, test_end))
        start += step_days

    print(f"\n  Data: {n_dates} trading days ({all_dates[0].date()} → {all_dates[-1].date()})")
    print(f"  Windows: {len(windows)} independent OOS tests")
    print(f"  Train={train_days}d, Test={test_days}d, Step={step_days}d, Warmup={warmup_days}d")
    print()

    results = []
    for wi, (ts, te, os_s, os_e) in enumerate(windows):
        test_dates = [all_dates[i] for i in range(os_s, min(os_e + 1, n_dates))]

        # Use the full train period as warmup and let _run_wf_backtest handle training
        # warmup_days = ts means all data up to train_end is used for initial model
        bt = _run_wf_backtest(
            df=df, clf_features=clf_features,
            trade_dates=test_dates, all_dates=all_dates,
            warmup_days=te,  # train on everything up to train_end
            start_capital=100_000,
            hold_days=3, min_probability=0.50,
            stop_loss_pct=-7.0, top_n=1, max_positions=1,
            retrain_every=21,  # retrain every 21 days within window
            strategy_type="momentum", regime_data=regime_data,
        )

        if not bt or len(bt.get("equity_curve", pd.DataFrame())) == 0:
            print(f"  Window {wi+1}: SKIP (no trades)")
            continue

        st = _compute_stats(bt, spy_data)
        period = f"{all_dates[os_s].date()} → {all_dates[min(os_e, n_dates-1)].date()}"
        status = "✓" if st["total_return"] > 0 else "✗"

        print(f"  Window {wi+1:2d}: {status} {period}  "
              f"Ret={st['total_return']:+6.1f}%  Sharpe={st.get('sharpe',0):5.2f}  "
              f"Trades={st['n_trades']:3d}  WR={st.get('win_rate',0):4.0f}%  "
              f"DD={st.get('max_dd',0):5.1f}%")

        results.append({
            "Window": wi + 1,
            "Period": period,
            "Start": all_dates[os_s],
            "End": all_dates[min(os_e, n_dates-1)],
            "Return_Pct": st["total_return"],
            "Sharpe": st.get("sharpe", 0),
            "Max_DD": st.get("max_dd", 0),
            "Win_Rate": st.get("win_rate", 0),
            "N_Trades": st["n_trades"],
            "SPY_Return": st.get("spy_return", 0),
            "Alpha": st.get("alpha", st["total_return"]),
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\n  No valid windows — insufficient data")
        return {"windows": results_df, "pass": False}

    n_profitable = (results_df["Return_Pct"] > 0).sum()
    n_beat_spy = (results_df["Alpha"] > 0).sum()
    n_total = len(results_df)

    print(f"\n  {'─'*70}")
    print(f"  SUMMARY: {n_profitable}/{n_total} windows profitable ({n_profitable/n_total*100:.0f}%)")
    print(f"           {n_beat_spy}/{n_total} windows beat SPY ({n_beat_spy/n_total*100:.0f}%)")
    print(f"           Avg return: {results_df['Return_Pct'].mean():+.1f}%")
    print(f"           Avg Sharpe: {results_df['Sharpe'].mean():.2f}")
    print(f"           Avg alpha:  {results_df['Alpha'].mean():+.1f}%")
    print(f"           Worst window: {results_df['Return_Pct'].min():+.1f}%")
    print(f"           Best window:  {results_df['Return_Pct'].max():+.1f}%")

    passed = n_profitable / n_total >= 0.60 and results_df["Return_Pct"].mean() > 0
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n  VERDICT: {status}")

    return {
        "windows": results_df,
        "n_profitable": n_profitable,
        "n_beat_spy": n_beat_spy,
        "n_total": n_total,
        "avg_return": results_df["Return_Pct"].mean(),
        "avg_sharpe": results_df["Sharpe"].mean(),
        "avg_alpha": results_df["Alpha"].mean(),
        "worst_return": results_df["Return_Pct"].min(),
        "best_return": results_df["Return_Pct"].max(),
        "pass": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2: TICKER-CAPPED BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════

def ticker_capped_backtest(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
    max_trades_per_ticker: int = 20,
) -> dict:
    """
    Run the winning strategy but cap how many times any single ticker
    can be traded. This directly addresses ticker concentration.
    """
    print("\n" + "=" * 80)
    print("  TEST 2: TICKER-CAPPED BACKTEST")
    print(f"  Max {max_trades_per_ticker} trades per ticker — no single stock dominates")
    print("=" * 80)

    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    all_dates = sorted(df.index.unique())
    trade_dates = all_dates[warmup_days:]

    # Uncapped (baseline)
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    bt_uncapped = _run_wf_backtest(
        df=df, clf_features=clf_features,
        trade_dates=trade_dates, all_dates=all_dates,
        warmup_days=warmup_days, hold_days=3, min_probability=0.50,
        stop_loss_pct=-7.0, top_n=1, max_positions=1,
        retrain_every=63, strategy_type="momentum",
        regime_data=regime_data, ticker_cap=0,
    )
    sys.stdout = old_stdout
    st_uncapped = _compute_stats(bt_uncapped, spy_data)

    # Capped
    sys.stdout = io.StringIO()
    bt_capped = _run_wf_backtest(
        df=df, clf_features=clf_features,
        trade_dates=trade_dates, all_dates=all_dates,
        warmup_days=warmup_days, hold_days=3, min_probability=0.50,
        stop_loss_pct=-7.0, top_n=1, max_positions=1,
        retrain_every=63, strategy_type="momentum",
        regime_data=regime_data, ticker_cap=max_trades_per_ticker,
    )
    sys.stdout = old_stdout
    st_capped = _compute_stats(bt_capped, spy_data)

    # Ticker distribution
    tc_uncapped = bt_uncapped.get("ticker_trade_count", {})
    tc_capped = bt_capped.get("ticker_trade_count", {})

    print(f"\n  UNCAPPED (baseline):")
    print(f"    Return: {st_uncapped['total_return']:+.1f}%  Sharpe: {st_uncapped.get('sharpe',0):.2f}  "
          f"Trades: {st_uncapped['n_trades']}  WR: {st_uncapped.get('win_rate',0):.0f}%")
    top5_unc = sorted(tc_uncapped.items(), key=lambda x: -x[1])[:5]
    if top5_unc:
        print(f"    Top tickers: {', '.join(f'{t}({c})' for t,c in top5_unc)}")

    print(f"\n  CAPPED (max {max_trades_per_ticker}/ticker):")
    print(f"    Return: {st_capped['total_return']:+.1f}%  Sharpe: {st_capped.get('sharpe',0):.2f}  "
          f"Trades: {st_capped['n_trades']}  WR: {st_capped.get('win_rate',0):.0f}%")
    top5_cap = sorted(tc_capped.items(), key=lambda x: -x[1])[:5]
    if top5_cap:
        print(f"    Top tickers: {', '.join(f'{t}({c})' for t,c in top5_cap)}")

    # Verdict
    capped_ok = st_capped["total_return"] > 0 and st_capped.get("sharpe", 0) > 0.5
    print(f"\n  Capped still profitable: {'YES ✓' if capped_ok else 'NO ✗'}")
    retention = st_capped["total_return"] / max(st_uncapped["total_return"], 0.01) * 100
    print(f"  Return retention: {retention:.0f}% of uncapped performance")

    passed = capped_ok and retention > 30
    print(f"  VERDICT: {'PASS ✓' if passed else 'FAIL ✗'}")

    return {
        "uncapped": st_uncapped, "capped": st_capped,
        "uncapped_tickers": tc_uncapped, "capped_tickers": tc_capped,
        "retention_pct": retention, "pass": passed,
        "capped_equity": bt_capped.get("equity_curve", pd.DataFrame()),
        "uncapped_equity": bt_uncapped.get("equity_curve", pd.DataFrame()),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3: LEAVE-TOP-TICKERS-OUT
# ═══════════════════════════════════════════════════════════════════════════

def leave_top_tickers_out(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
    n_remove: int = 5,
) -> dict:
    """
    Remove the top N most-traded tickers ENTIRELY and re-run.
    If the edge disappears, it was a few lucky stocks, not skill.
    """
    print("\n" + "=" * 80)
    print(f"  TEST 3: LEAVE-TOP-{n_remove}-TICKERS-OUT")
    print("  Remove the most frequent tickers entirely — is there still alpha?")
    print("=" * 80)

    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    all_dates = sorted(df.index.unique())
    trade_dates = all_dates[warmup_days:]

    # First run uncapped to find top tickers
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    bt_base = _run_wf_backtest(
        df=df, clf_features=clf_features,
        trade_dates=trade_dates, all_dates=all_dates,
        warmup_days=warmup_days, hold_days=3, min_probability=0.50,
        stop_loss_pct=-7.0, top_n=1, max_positions=1,
        retrain_every=63, strategy_type="momentum",
        regime_data=regime_data,
    )
    sys.stdout = old_stdout

    tc = bt_base.get("ticker_trade_count", {})
    top_tickers = sorted(tc.items(), key=lambda x: -x[1])[:n_remove]
    excluded = set(t for t, _ in top_tickers)

    st_base = _compute_stats(bt_base, spy_data)
    print(f"\n  Baseline: {st_base['total_return']:+.1f}% | {st_base['n_trades']} trades")
    print(f"  Removing: {', '.join(f'{t}({c} trades)' for t,c in top_tickers)}")

    # Re-run without those tickers
    sys.stdout = io.StringIO()
    bt_excl = _run_wf_backtest(
        df=df, clf_features=clf_features,
        trade_dates=trade_dates, all_dates=all_dates,
        warmup_days=warmup_days, hold_days=3, min_probability=0.50,
        stop_loss_pct=-7.0, top_n=1, max_positions=1,
        retrain_every=63, strategy_type="momentum",
        regime_data=regime_data, excluded_tickers=excluded,
    )
    sys.stdout = old_stdout
    st_excl = _compute_stats(bt_excl, spy_data)

    print(f"\n  Without top {n_remove}:")
    print(f"    Return: {st_excl['total_return']:+.1f}%  Sharpe: {st_excl.get('sharpe',0):.2f}  "
          f"Trades: {st_excl['n_trades']}  WR: {st_excl.get('win_rate',0):.0f}%")

    passed = st_excl["total_return"] > 0 and st_excl.get("sharpe", 0) > 0
    print(f"\n  Still profitable without top {n_remove}: {'YES ✓' if passed else 'NO ✗'}")
    print(f"  VERDICT: {'PASS ✓' if passed else 'FAIL ✗'}")

    return {
        "baseline": st_base, "excluded": st_excl,
        "removed_tickers": list(excluded),
        "pass": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 4: REGIME-ADAPTIVE POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════

def regime_adaptive_test(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
) -> dict:
    """
    Compare fixed sizing vs adaptive sizing that scales down in bear markets.
    """
    print("\n" + "=" * 80)
    print("  TEST 4: REGIME-ADAPTIVE POSITION SIZING")
    print("  Scale position size based on market regime (smaller bets in bear)")
    print("=" * 80)

    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    all_dates = sorted(df.index.unique())
    trade_dates = all_dates[warmup_days:]

    def adaptive_scale(today, rdata):
        """Return 0.3x in bear, 0.6x in normal_bear, 1.0x in bull."""
        if rdata is None or today not in rdata.index:
            return 1.0
        rv = rdata.loc[today]
        rs = str(rv.get("Market_Regime", "")) if isinstance(rv, pd.Series) else str(rv)
        if "crash" in rs.lower():
            return 0.0
        elif "bear" in rs.lower() and "normal" not in rs.lower():
            return 0.3
        elif "normal_bear" in rs.lower():
            return 0.6
        return 1.0

    # Fixed sizing
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    bt_fixed = _run_wf_backtest(
        df=df, clf_features=clf_features,
        trade_dates=trade_dates, all_dates=all_dates,
        warmup_days=warmup_days, hold_days=3, min_probability=0.50,
        stop_loss_pct=-7.0, top_n=1, max_positions=1,
        retrain_every=63, strategy_type="momentum",
        regime_data=regime_data,
    )
    sys.stdout = old_stdout
    st_fixed = _compute_stats(bt_fixed, spy_data)

    # Adaptive sizing
    sys.stdout = io.StringIO()
    bt_adaptive = _run_wf_backtest(
        df=df, clf_features=clf_features,
        trade_dates=trade_dates, all_dates=all_dates,
        warmup_days=warmup_days, hold_days=3, min_probability=0.50,
        stop_loss_pct=-7.0, top_n=1, max_positions=1,
        retrain_every=63, strategy_type="momentum",
        regime_data=regime_data, position_scale_fn=adaptive_scale,
    )
    sys.stdout = old_stdout
    st_adaptive = _compute_stats(bt_adaptive, spy_data)

    print(f"\n  FIXED (100% always):")
    print(f"    Return: {st_fixed['total_return']:+.1f}%  Sharpe: {st_fixed.get('sharpe',0):.2f}  "
          f"Max DD: {st_fixed.get('max_dd',0):.1f}%  MWR: {st_fixed.get('monthly_win_rate',0):.0f}%")

    print(f"\n  ADAPTIVE (scale by regime):")
    print(f"    Return: {st_adaptive['total_return']:+.1f}%  Sharpe: {st_adaptive.get('sharpe',0):.2f}  "
          f"Max DD: {st_adaptive.get('max_dd',0):.1f}%  MWR: {st_adaptive.get('monthly_win_rate',0):.0f}%")

    # Did adaptive improve risk metrics?
    dd_improved = abs(st_adaptive.get("max_dd", -99)) < abs(st_fixed.get("max_dd", -99))
    sharpe_improved = st_adaptive.get("sharpe", 0) > st_fixed.get("sharpe", 0)

    print(f"\n  Drawdown improved: {'YES ✓' if dd_improved else 'NO'}")
    print(f"  Sharpe improved:   {'YES ✓' if sharpe_improved else 'NO'}")

    passed = st_adaptive["total_return"] > 0 and (dd_improved or sharpe_improved)
    print(f"  VERDICT: {'PASS ✓' if passed else 'FAIL ✗'}")

    return {
        "fixed": st_fixed, "adaptive": st_adaptive,
        "dd_improved": dd_improved, "sharpe_improved": sharpe_improved,
        "pass": passed,
        "adaptive_equity": bt_adaptive.get("equity_curve", pd.DataFrame()),
        "fixed_equity": bt_fixed.get("equity_curve", pd.DataFrame()),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 5: BOOTSTRAP CONFIDENCE INTERVALS ON SHARPE
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_confidence(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Bootstrap confidence intervals on key metrics.
    Resample daily returns WITH replacement 1000 times.
    If 95% CI for Sharpe > 0, the edge is statistically significant.
    """
    print("\n" + "=" * 80)
    print(f"  TEST 5: BOOTSTRAP CONFIDENCE INTERVALS ({n_bootstrap:,} samples)")
    print("  Statistical proof that the edge isn't luck")
    print("=" * 80)

    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    all_dates = sorted(df.index.unique())
    trade_dates = all_dates[warmup_days:]

    # Run the strategy once to get trade returns
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    bt = _run_wf_backtest(
        df=df, clf_features=clf_features,
        trade_dates=trade_dates, all_dates=all_dates,
        warmup_days=warmup_days, hold_days=3, min_probability=0.50,
        stop_loss_pct=-7.0, top_n=1, max_positions=1,
        retrain_every=63, strategy_type="momentum",
        regime_data=regime_data,
    )
    sys.stdout = old_stdout

    trades = bt.get("trades", pd.DataFrame())
    eq = bt.get("equity_curve", pd.DataFrame())

    if len(trades) < 20:
        print("  Not enough trades for bootstrap")
        return {"pass": False}

    trade_returns = trades["Return_Pct"].values
    trade_pnl = trades["PnL"].values
    n_trades = len(trade_returns)
    start_cap = bt["start_capital"]

    boot_total_returns = []
    boot_sharpes = []
    boot_win_rates = []
    boot_max_dds = []

    np.random.seed(42)
    for i in range(n_bootstrap):
        # Resample trades with replacement
        idx = np.random.randint(0, n_trades, size=n_trades)
        sample_pnl = trade_pnl[idx]
        sample_ret = trade_returns[idx]

        # Cumulative equity
        equity = start_cap + np.cumsum(sample_pnl)
        total_ret = (equity[-1] / start_cap - 1) * 100
        boot_total_returns.append(total_ret)

        # Win rate
        boot_win_rates.append((sample_ret > 0).mean() * 100)

        # Crude Sharpe from trade PnL
        daily_equiv = sample_pnl / (start_cap / 1)
        if daily_equiv.std() > 0:
            sr = (daily_equiv.mean() / daily_equiv.std()) * np.sqrt(252 / 3)
            boot_sharpes.append(sr)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = ((equity - peak) / peak * 100).min()
        boot_max_dds.append(dd)

    boot_total_returns = np.array(boot_total_returns)
    boot_sharpes = np.array(boot_sharpes) if boot_sharpes else np.array([0])
    boot_win_rates = np.array(boot_win_rates)
    boot_max_dds = np.array(boot_max_dds)

    # Confidence intervals
    ret_ci = (np.percentile(boot_total_returns, 2.5), np.percentile(boot_total_returns, 97.5))
    sharpe_ci = (np.percentile(boot_sharpes, 2.5), np.percentile(boot_sharpes, 97.5))
    wr_ci = (np.percentile(boot_win_rates, 2.5), np.percentile(boot_win_rates, 97.5))
    dd_ci = (np.percentile(boot_max_dds, 2.5), np.percentile(boot_max_dds, 97.5))

    prob_positive = (boot_total_returns > 0).mean() * 100
    prob_sharpe_gt05 = (boot_sharpes > 0.5).mean() * 100

    print(f"\n  Trades bootstrapped: {n_trades}")
    print(f"\n  RETURN:")
    print(f"    Mean:   {boot_total_returns.mean():+.1f}%")
    print(f"    95% CI: [{ret_ci[0]:+.1f}%, {ret_ci[1]:+.1f}%]")
    print(f"    P(positive): {prob_positive:.0f}%")
    print(f"\n  SHARPE:")
    print(f"    Mean:   {boot_sharpes.mean():.2f}")
    print(f"    95% CI: [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]")
    print(f"    P(Sharpe > 0.5): {prob_sharpe_gt05:.0f}%")
    print(f"\n  WIN RATE:")
    print(f"    95% CI: [{wr_ci[0]:.0f}%, {wr_ci[1]:.0f}%]")
    print(f"\n  MAX DRAWDOWN:")
    print(f"    95% CI: [{dd_ci[0]:.1f}%, {dd_ci[1]:.1f}%]")
    print(f"    Worst-case (bootstrapped): {boot_max_dds.min():.1f}%")

    # Pass if lower bound of Sharpe CI > 0 AND P(positive) > 80%
    passed = sharpe_ci[0] > 0 and prob_positive >= 80
    print(f"\n  Edge statistically significant: {'YES ✓' if passed else 'NO ✗'}")
    print(f"  VERDICT: {'PASS ✓' if passed else 'FAIL ✗'}")

    return {
        "boot_returns": boot_total_returns,
        "boot_sharpes": boot_sharpes,
        "boot_win_rates": boot_win_rates,
        "boot_max_dds": boot_max_dds,
        "return_ci": ret_ci,
        "sharpe_ci": sharpe_ci,
        "wr_ci": wr_ci,
        "dd_ci": dd_ci,
        "prob_positive": prob_positive,
        "prob_sharpe_gt05": prob_sharpe_gt05,
        "pass": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 6: MONTHLY ALPHA BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def monthly_alpha_breakdown(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
) -> dict:
    """
    Month-by-month and year-by-year breakdown of alpha vs SPY.
    """
    print("\n" + "=" * 80)
    print("  TEST 6: MONTHLY & YEARLY ALPHA BREAKDOWN")
    print("  Does the strategy outperform SPY consistently?")
    print("=" * 80)

    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    all_dates = sorted(df.index.unique())
    trade_dates = all_dates[warmup_days:]

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    bt = _run_wf_backtest(
        df=df, clf_features=clf_features,
        trade_dates=trade_dates, all_dates=all_dates,
        warmup_days=warmup_days, hold_days=3, min_probability=0.50,
        stop_loss_pct=-7.0, top_n=1, max_positions=1,
        retrain_every=63, strategy_type="momentum",
        regime_data=regime_data,
    )
    sys.stdout = old_stdout

    eq = bt.get("equity_curve", pd.DataFrame())
    if len(eq) == 0:
        return {"pass": False}

    # Monthly returns
    mv = eq["Portfolio_Value"].resample("ME").last()
    monthly_strat = mv.pct_change().dropna() * 100

    if spy_data is None:
        print("  No SPY data for comparison")
        return {"pass": False}

    sm = spy_data.resample("ME").last()
    monthly_spy = sm.pct_change().dropna() * 100
    common = monthly_strat.index.intersection(monthly_spy.index)

    if len(common) == 0:
        return {"pass": False}

    months_data = []
    print(f"\n  {'Month':>8s}  {'Strategy':>9s}  {'SPY':>8s}  {'Alpha':>8s}  {'Status':>6s}")
    print(f"  {'─'*50}")

    for m in common:
        s_ret = monthly_strat.loc[m]
        sp_ret = monthly_spy.loc[m]
        if isinstance(sp_ret, pd.Series):
            sp_ret = sp_ret.iloc[0]
        alpha = s_ret - sp_ret
        status = "WIN" if alpha > 0 else "LOSE"
        print(f"  {m.strftime('%Y-%m'):>8s}  {s_ret:>+8.1f}%  {sp_ret:>+7.1f}%  {alpha:>+7.1f}%  {status:>6s}")
        months_data.append({
            "Month": m, "Strategy": s_ret, "SPY": sp_ret,
            "Alpha": alpha, "Beat_SPY": alpha > 0,
        })

    mdf = pd.DataFrame(months_data)
    n_beat = mdf["Beat_SPY"].sum()
    n_total = len(mdf)

    print(f"\n  {'─'*50}")
    print(f"  Beat SPY: {n_beat}/{n_total} months ({n_beat/n_total*100:.0f}%)")
    print(f"  Avg monthly alpha: {mdf['Alpha'].mean():+.1f}%")
    print(f"  Total strategy: {mdf['Strategy'].sum():+.1f}%")
    print(f"  Total SPY: {mdf['SPY'].sum():+.1f}%")

    # Year-by-year
    mdf["Year"] = mdf["Month"].dt.year
    for year, ydf in mdf.groupby("Year"):
        yb = ydf["Beat_SPY"].sum()
        yn = len(ydf)
        print(f"\n  {year}: Strategy {ydf['Strategy'].sum():+.1f}% vs SPY {ydf['SPY'].sum():+.1f}% "
              f"| Beat SPY {yb}/{yn} months ({yb/yn*100:.0f}%)")

    passed = n_beat / n_total >= 0.50 and mdf["Alpha"].mean() > 0
    print(f"\n  VERDICT: {'PASS ✓' if passed else 'FAIL ✗'}")

    return {
        "monthly_data": mdf,
        "n_beat_spy": n_beat,
        "n_total_months": n_total,
        "avg_alpha": mdf["Alpha"].mean(),
        "pass": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 7: MULTI-STRATEGY STABILITY
# ═══════════════════════════════════════════════════════════════════════════

def multi_strategy_stability(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
) -> dict:
    """
    Run the strategy with different hold periods + filter combos.
    If most combinations are profitable, the edge is robust.
    """
    print("\n" + "=" * 80)
    print("  TEST 7: MULTI-STRATEGY STABILITY")
    print("  18 different parameter/filter combos — how many are profitable?")
    print("=" * 80)

    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    all_dates = sorted(df.index.unique())
    trade_dates = all_dates[warmup_days:]

    configs = [
        # (hold, min_p, sl, top_n, max_pos, strategy)
        (1, 0.50, -5.0, 1, 1, "momentum"),
        (2, 0.50, -7.0, 1, 1, "momentum"),
        (3, 0.50, -7.0, 1, 1, "momentum"),   # Winning config
        (5, 0.50, -7.0, 1, 1, "momentum"),
        (3, 0.40, -7.0, 1, 1, "momentum"),
        (3, 0.60, -7.0, 1, 1, "momentum"),
        (3, 0.50, -5.0, 1, 1, "momentum"),
        (3, 0.50, -10.0, 1, 1, "momentum"),
        (3, 0.50, -7.0, 3, 3, "momentum"),
        (3, 0.50, -7.0, 1, 1, "base"),
        (3, 0.50, -7.0, 1, 1, "regime"),
        (3, 0.50, -7.0, 1, 1, "combo"),
        (3, 0.50, -7.0, 3, 3, "base"),
        (3, 0.50, -7.0, 3, 3, "regime"),
        (5, 0.50, -7.0, 3, 3, "base"),
        (5, 0.40, -10.0, 3, 3, "momentum"),
        (2, 0.60, -5.0, 1, 1, "combo"),
        (7, 0.50, -10.0, 1, 1, "momentum"),
    ]

    results = []
    for ci, (hd, mp, sl, tn, mpos, strat) in enumerate(configs):
        label = f"{strat}|H{hd}d|P{mp:.0%}|SL{sl}%|T{tn}"
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        bt = _run_wf_backtest(
            df=df, clf_features=clf_features,
            trade_dates=trade_dates, all_dates=all_dates,
            warmup_days=warmup_days, hold_days=hd, min_probability=mp,
            stop_loss_pct=sl, top_n=tn, max_positions=mpos,
            retrain_every=63, strategy_type=strat,
            regime_data=regime_data,
        )
        sys.stdout = old_stdout

        st = _compute_stats(bt, spy_data)
        status = "✓" if st["total_return"] > 0 else "✗"
        beat = st.get("alpha", st["total_return"]) > 0
        spy_sym = ">" if beat else "<"

        print(f"  [{ci+1:2d}/{len(configs)}] {status} {label:40s}  "
              f"Ret={st['total_return']:+6.1f}%  Sh={st.get('sharpe',0):5.2f}  "
              f"WR={st.get('win_rate',0):4.0f}%  {spy_sym}SPY")

        results.append({
            "Config": label, "Return": st["total_return"],
            "Sharpe": st.get("sharpe", 0),
            "Win_Rate": st.get("win_rate", 0),
            "Max_DD": st.get("max_dd", 0),
            "Alpha": st.get("alpha", st["total_return"]),
            "N_Trades": st["n_trades"],
            "Beat_SPY": beat,
        })

    rdf = pd.DataFrame(results)
    n_profitable = (rdf["Return"] > 0).sum()
    n_beat_spy = rdf["Beat_SPY"].sum()
    n_total = len(rdf)

    print(f"\n  {'─'*70}")
    print(f"  Profitable:  {n_profitable}/{n_total} ({n_profitable/n_total*100:.0f}%)")
    print(f"  Beat SPY:    {n_beat_spy}/{n_total} ({n_beat_spy/n_total*100:.0f}%)")
    print(f"  Avg return:  {rdf['Return'].mean():+.1f}%")
    print(f"  Avg Sharpe:  {rdf['Sharpe'].mean():.2f}")
    print(f"  Return range: {rdf['Return'].min():+.1f}% to {rdf['Return'].max():+.1f}%")

    passed = n_profitable / n_total >= 0.65 and rdf["Return"].mean() > 0
    print(f"\n  VERDICT: {'PASS ✓' if passed else 'FAIL ✗'}")

    return {
        "configs": rdf,
        "n_profitable": n_profitable,
        "n_beat_spy": n_beat_spy,
        "n_total": n_total,
        "avg_return": rdf["Return"].mean(),
        "avg_sharpe": rdf["Sharpe"].mean(),
        "pass": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER: Run everything + produce final scorecard
# ═══════════════════════════════════════════════════════════════════════════

def run_final_verification(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
) -> dict:
    """
    Run ALL 7 verification tests and produce a final scorecard.
    This is the single function to call for the most thorough possible test.
    """
    print("\n" + "═" * 80)
    mode_label = ""
    if ENHANCED_DEFAULTS.get("use_compounding"):
        mode_label += " + COMPOUNDING"
    if ENHANCED_DEFAULTS.get("tx_cost_bps", 0) > 0:
        mode_label += f" + {ENHANCED_DEFAULTS['tx_cost_bps']:.0f}bps TX COSTS"
    print("  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║       FINAL VERIFICATION SUITE — 7 TESTS               ║")
    if mode_label:
        line = f"  ║       MODE: {mode_label.strip(' +'):43s}║"
        print(line)
    print("  ║       The most thorough honest backtesting possible     ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print("═" * 80)

    t_start = time.time()
    results = {}

    # Test 1: Rolling walk-forward
    print("\n\n" + "▓" * 80)
    r1 = rolling_walk_forward(
        returns, classifier_dict, regime_data, spy_data,
        train_days=189, test_days=63, step_days=42, warmup_days=126,
    )
    results["rolling_wf"] = r1

    # Test 2: Ticker-capped
    print("\n\n" + "▓" * 80)
    r2 = ticker_capped_backtest(
        returns, classifier_dict, regime_data, spy_data,
        warmup_days, max_trades_per_ticker=20,
    )
    results["ticker_capped"] = r2

    # Test 3: Leave-top-tickers-out
    print("\n\n" + "▓" * 80)
    r3 = leave_top_tickers_out(
        returns, classifier_dict, regime_data, spy_data,
        warmup_days, n_remove=5,
    )
    results["leave_out"] = r3

    # Test 4: Regime-adaptive
    print("\n\n" + "▓" * 80)
    r4 = regime_adaptive_test(
        returns, classifier_dict, regime_data, spy_data,
        warmup_days,
    )
    results["regime_adaptive"] = r4

    # Test 5: Bootstrap confidence
    print("\n\n" + "▓" * 80)
    r5 = bootstrap_confidence(
        returns, classifier_dict, regime_data, spy_data,
        warmup_days, n_bootstrap=1000,
    )
    results["bootstrap"] = r5

    # Test 6: Monthly alpha
    print("\n\n" + "▓" * 80)
    r6 = monthly_alpha_breakdown(
        returns, classifier_dict, regime_data, spy_data,
        warmup_days,
    )
    results["monthly_alpha"] = r6

    # Test 7: Multi-strategy stability
    print("\n\n" + "▓" * 80)
    r7 = multi_strategy_stability(
        returns, classifier_dict, regime_data, spy_data,
        warmup_days,
    )
    results["multi_strategy"] = r7

    elapsed = time.time() - t_start

    # ── FINAL SCORECARD ──────────────────────────────────────────────
    print("\n\n" + "═" * 80)
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║          FINAL SCORECARD — ALL 7 TESTS                 ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print("═" * 80)

    tests = [
        ("1. Rolling Walk-Forward",   r1.get("pass", False),
         f"{r1.get('n_profitable',0)}/{r1.get('n_total',0)} windows profitable, "
         f"avg {r1.get('avg_return',0):+.1f}%"),
        ("2. Ticker-Capped",          r2.get("pass", False),
         f"Capped: {r2.get('capped',{}).get('total_return',0):+.1f}% "
         f"({r2.get('retention_pct',0):.0f}% retention)"),
        ("3. Leave-Top-Tickers-Out",  r3.get("pass", False),
         f"Without top 5: {r3.get('excluded',{}).get('total_return',0):+.1f}%"),
        ("4. Regime-Adaptive Sizing",  r4.get("pass", False),
         f"Adaptive: {r4.get('adaptive',{}).get('total_return',0):+.1f}%, "
         f"DD: {r4.get('adaptive',{}).get('max_dd',0):.1f}%"),
        ("5. Bootstrap Confidence",    r5.get("pass", False),
         f"95% CI Sharpe: [{r5.get('sharpe_ci',(0,0))[0]:.2f}, "
         f"{r5.get('sharpe_ci',(0,0))[1]:.2f}], "
         f"P(profit)={r5.get('prob_positive',0):.0f}%"),
        ("6. Monthly Alpha",          r6.get("pass", False),
         f"Beat SPY {r6.get('n_beat_spy',0)}/{r6.get('n_total_months',0)} months, "
         f"avg alpha={r6.get('avg_alpha',0):+.1f}%"),
        ("7. Multi-Strategy Stability", r7.get("pass", False),
         f"{r7.get('n_profitable',0)}/{r7.get('n_total',0)} configs profitable, "
         f"avg {r7.get('avg_return',0):+.1f}%"),
    ]

    passed = 0
    for name, result, detail in tests:
        icon = "✓" if result else "✗"
        status = "PASS" if result else "FAIL"
        print(f"\n  {icon} {name:35s} {status}")
        print(f"    {detail}")
        if result:
            passed += 1

    print(f"\n  {'═'*70}")
    print(f"  SCORE: {passed}/7 tests passed")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  {'═'*70}")

    if passed == 7:
        grade = "S"
        verdict = "PERFECT — Edge is as proven as backtesting can possibly show"
    elif passed >= 6:
        grade = "A"
        verdict = "EXCELLENT — Strong, robust edge with minor weaknesses"
    elif passed >= 5:
        grade = "B"
        verdict = "GOOD — Genuine edge exists, but some risk areas"
    elif passed >= 4:
        grade = "C"
        verdict = "MODERATE — Some evidence of edge, proceed with caution"
    elif passed >= 3:
        grade = "D"
        verdict = "WEAK — Edge is fragile and likely won't survive live trading"
    else:
        grade = "F"
        verdict = "FAILED — No reliable edge detected"

    print(f"\n  GRADE: {grade}")
    print(f"  {verdict}")
    print(f"\n{'═'*80}")

    results["score"] = passed
    results["grade"] = grade
    results["verdict"] = verdict
    results["elapsed_minutes"] = elapsed / 60

    return results
