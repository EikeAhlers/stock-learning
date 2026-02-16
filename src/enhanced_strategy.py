"""
Enhanced Strategy — 6 improvements over the baseline backtester.

IMPROVEMENTS:
  1. EV-Ranking: Rank by P(move) × E(return) instead of probability alone
     - Trains a magnitude regressor alongside the classifier in walk-forward
     - Picks trades with highest EXPECTED VALUE, not just highest probability
  2. LightGBM Ensemble: Average XGBoost + LightGBM for more robust signals
     - Two independent models vote together; reduces overfitting
  3. ATR Dynamic Stop Loss: Adapt stop to each stock's volatility
     - Stop = -2×ATR%, clamped to [-3%, -15%]
     - Tight stops on calm stocks, wide stops on volatile ones
  4. Transaction Costs: 10bps slippage per side (20bps round trip)
     - Real-world friction that eats into thin edges
  5. Loss Streak Circuit Breaker: Pause 1 day after 3 consecutive losses
     - Prevents tilt-trading during bad streaks
  6. Profit Compounding: Use current equity (not initial capital) for sizing
     - Lets winners grow the account faster (also adds risk)

Run `compare_strategies()` for a head-to-head backtest.
"""

import pandas as pd
import numpy as np
import sys, io, time
from typing import Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from src.deep_analyzer import add_all_lagged_features, ALL_RAW_FEATURES
from src.setup_detector import SETUP_NAMES
from src.backtester import _train_classifier_on_window


# ═══════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 2: Ensemble trainer (XGBoost + LightGBM)
# ═══════════════════════════════════════════════════════════════════════════

def _train_ensemble_on_window(df_train, features, big_move_threshold=5.0):
    """
    Train XGBoost + LightGBM ensemble on past data only.
    Returns both models, scaler, and also a regressor for magnitude.
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        return None
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        has_lgbm = True
    except ImportError:
        has_lgbm = False

    df_t = df_train.copy()
    df_t["Is_Big_Mover"] = (df_t["Daily_Return_Pct"] >= big_move_threshold).astype(int)

    # Forward 3-day return for magnitude (close-to-close, NOT max)
    df_t["Fwd_3d_Return"] = df_t.groupby("Ticker")["Daily_Return_Pct"].transform(
        lambda x: x.shift(-1).rolling(3, min_periods=1).sum()
    )

    model_df = df_t[features + ["Is_Big_Mover", "Fwd_3d_Return"]].dropna()
    if len(model_df) < 500:
        return None

    X = model_df[features]
    y_cls = model_df["Is_Big_Mover"]
    y_reg = model_df["Fwd_3d_Return"]

    pos_count = y_cls.sum()
    neg_count = len(y_cls) - pos_count
    scale_pos = neg_count / max(pos_count, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # XGBoost classifier
    xgb_clf = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=42,
        verbosity=0, eval_metric="logloss",
    )
    xgb_clf.fit(X_scaled, y_cls)

    # LightGBM classifier (if available)
    lgbm_clf = None
    if has_lgbm:
        lgbm_clf = LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, random_state=43,
            verbose=-1,
        )
        lgbm_clf.fit(X_scaled, y_cls)

    # XGBoost regressor for magnitude
    xgb_reg = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    xgb_reg.fit(X_scaled, y_reg)

    return {
        "xgb_clf": xgb_clf,
        "lgbm_clf": lgbm_clf,
        "xgb_reg": xgb_reg,
        "scaler": scaler,
        "features": features,
        "has_lgbm": has_lgbm,
    }


def _ensemble_predict(model_dict, X_scaled):
    """
    Get ensemble probability and magnitude prediction.
    Returns (probabilities, predicted_returns).
    """
    xgb_probs = model_dict["xgb_clf"].predict_proba(X_scaled)[:, 1]

    if model_dict["has_lgbm"] and model_dict["lgbm_clf"] is not None:
        lgbm_probs = model_dict["lgbm_clf"].predict_proba(X_scaled)[:, 1]
        # Weighted average: 55% XGB, 45% LGBM (XGB usually better on this data)
        probs = 0.55 * xgb_probs + 0.45 * lgbm_probs
    else:
        probs = xgb_probs

    pred_returns = model_dict["xgb_reg"].predict(X_scaled)

    return probs, pred_returns


# ═══════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 3: ATR-based dynamic stop loss
# ═══════════════════════════════════════════════════════════════════════════

def _get_atr_stop(today_row, multiplier=2.0, floor=-15.0, ceiling=-3.0):
    """
    Compute a dynamic stop loss based on ATR_Ratio.
    ATR_Ratio = ATR(14) / Close — already computed in advanced_features.
    Stop = -multiplier × ATR_Ratio × 100, clamped to [floor, ceiling].
    """
    atr_ratio = today_row.get("Prev_ATR_Ratio", np.nan)
    if pd.isna(atr_ratio) or atr_ratio <= 0:
        return -7.0  # default fallback
    stop = -multiplier * atr_ratio * 100
    return max(floor, min(ceiling, stop))


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENGINE: Enhanced Walk-Forward Backtest
# ═══════════════════════════════════════════════════════════════════════════

def run_enhanced_backtest(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    start_capital: float = 100_000,
    hold_days: int = 3,
    min_probability: float = 0.50,
    top_n: int = 1,
    max_positions: int = 1,
    retrain_every: int = 63,
    warmup_days: int = 252,
    # Enhancement toggles
    use_ensemble: bool = True,
    use_ev_ranking: bool = True,
    use_atr_stop: bool = True,
    use_txn_costs: bool = True,
    use_circuit_breaker: bool = True,
    use_compounding: bool = True,
    use_momentum_filter: bool = True,
    # Parameters
    txn_cost_bps: float = 10.0,
    breaker_threshold: int = 3,
    atr_multiplier: float = 2.0,
    fixed_stop_loss: float = -7.0,
    verbose: bool = True,
) -> dict:
    """
    Enhanced walk-forward backtest with 6 improvements.

    Returns same format as baseline backtester for easy comparison.
    """
    if verbose:
        print("=" * 80)
        print("  ENHANCED WALK-FORWARD BACKTEST")
        print("=" * 80)
        flags = []
        if use_ensemble:         flags.append("Ensemble(XGB+LGBM)")
        if use_ev_ranking:       flags.append("EV-Ranking(P×E)")
        if use_atr_stop:         flags.append("ATR-Stop")
        if use_txn_costs:        flags.append(f"TxCost({txn_cost_bps}bps)")
        if use_circuit_breaker:  flags.append(f"Breaker({breaker_threshold})")
        if use_compounding:      flags.append("Compounding")
        if use_momentum_filter:  flags.append("Momentum")
        print(f"  Enhancements: {' | '.join(flags)}")

    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)

    clf_features = classifier_dict["features"]
    all_dates = sorted(df.index.unique())

    if len(all_dates) < warmup_days + 30:
        print(f"  Not enough data ({len(all_dates)} days)")
        return {}

    trade_dates = all_dates[warmup_days:]

    if verbose:
        print(f"  Warmup: {all_dates[0].date()} → {all_dates[warmup_days-1].date()} ({warmup_days}d)")
        print(f"  Trading: {trade_dates[0].date()} → {trade_dates[-1].date()} ({len(trade_dates)}d)")
        print(f"  Strategy: Top {top_n}, hold {hold_days}d, min P={min_probability:.0%}")
        print()

    # Train initial model
    warmup_data = df[df.index <= all_dates[warmup_days - 1]]
    if use_ensemble:
        current_model = _train_ensemble_on_window(warmup_data, clf_features)
    else:
        current_model = _train_classifier_on_window(warmup_data, clf_features)

    if current_model is None:
        print("  ERROR: Not enough warmup data")
        return {}

    days_since_retrain = 0

    capital = start_capital
    positions = []
    trade_log = []
    equity_curve = []
    consecutive_losses = 0
    breaker_cooldown = 0
    dates_processed = 0

    for today in trade_dates:
        # ── Periodic retraining ──────────────────────────────────────
        if days_since_retrain >= retrain_every:
            past = df[df.index < today]
            if use_ensemble:
                new_m = _train_ensemble_on_window(past, clf_features)
            else:
                new_m = _train_classifier_on_window(past, clf_features)
            if new_m is not None:
                current_model = new_m
                days_since_retrain = 0
        days_since_retrain += 1

        # ── Circuit breaker check ────────────────────────────────────
        if use_circuit_breaker and breaker_cooldown > 0:
            breaker_cooldown -= 1

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

            # Exit conditions
            stop = pos.get("stop_loss", fixed_stop_loss)
            should_exit = ret <= stop or pos["days_held"] >= hold_days

            if should_exit:
                exit_reason = "stop_loss" if ret <= stop else "target_hold"

                # Transaction cost on exit
                exit_cost = 0
                if use_txn_costs:
                    exit_cost = pos["shares"] * cp * (txn_cost_bps / 10_000)

                proceeds = pos["shares"] * cp - exit_cost
                capital += proceeds

                net_ret = (proceeds / (pos["shares"] * pos["entry_price"]) - 1) * 100
                net_pnl = proceeds - pos["shares"] * pos["entry_price"]

                trade_log.append({
                    "Ticker": pos["ticker"],
                    "Entry_Date": pos["entry_date"],
                    "Exit_Date": today,
                    "Entry_Price": pos["entry_price"],
                    "Exit_Price": cp,
                    "Return_Pct": net_ret,
                    "PnL": net_pnl,
                    "Hold_Days": pos["days_held"],
                    "Exit_Reason": exit_reason,
                    "Probability": pos.get("probability", 0),
                    "EV_Score": pos.get("ev_score", 0),
                    "Stop_Loss": stop,
                })
                closed.append(pos)

                # Track consecutive losses
                if net_ret <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                # Trigger circuit breaker
                if use_circuit_breaker and consecutive_losses >= breaker_threshold:
                    breaker_cooldown = 1  # skip 1 trading day
                    if verbose and dates_processed < 300:
                        pass  # suppress noise

        for c in closed:
            positions.remove(c)

        # ── Open new positions ───────────────────────────────────────
        skip_trading = (use_circuit_breaker and breaker_cooldown > 0)
        open_slots = max_positions - len(positions)

        if open_slots > 0 and not skip_trading:
            today_all = df[df.index == today].copy().reset_index(drop=True)

            if len(today_all) > 10:
                fm = today_all[clf_features].notna().all(axis=1)
                tf = today_all.loc[fm, clf_features]

                if len(tf) > 0:
                    scaler = current_model["scaler"]
                    X = scaler.transform(tf)

                    # Get predictions
                    if use_ensemble and "xgb_clf" in current_model:
                        probs, pred_rets = _ensemble_predict(current_model, X)
                    else:
                        model = current_model["model"]
                        probs = model.predict_proba(X)[:, 1]
                        pred_rets = np.zeros(len(probs))

                    scores = pd.DataFrame({
                        "idx": today_all.loc[fm].index,
                        "Ticker": today_all.loc[fm, "Ticker"].values,
                        "Probability": probs,
                        "Pred_Return": pred_rets,
                    })

                    # Probability filter
                    scores = scores[scores["Probability"] >= min_probability]

                    # Momentum filter
                    if use_momentum_filter and "Prev_Return_20d" in today_all.columns:
                        mom = today_all.loc[scores["idx"], "Prev_Return_20d"]
                        scores = scores[mom.values > 0]

                    # EV ranking: P × E(return)
                    if use_ev_ranking:
                        scores["EV"] = scores["Probability"] * scores["Pred_Return"]
                        scores = scores.sort_values("EV", ascending=False)
                    else:
                        scores["EV"] = scores["Probability"]
                        scores = scores.sort_values("Probability", ascending=False)

                    # Don't buy what we already hold
                    held = [p["ticker"] for p in positions]
                    scores = scores[~scores["Ticker"].isin(held)]

                    n_buy = min(open_slots, top_n, len(scores))

                    for j in range(n_buy):
                        pick = scores.iloc[j]
                        ticker = pick["Ticker"]
                        prob = pick["Probability"]
                        pred_ret = pick["Pred_Return"]
                        ev = pick["EV"]

                        td2 = today_all[today_all["Ticker"] == ticker]
                        if len(td2) == 0:
                            continue
                        ep = td2["Close"].iloc[0]
                        if ep <= 0:
                            continue

                        # Dynamic ATR stop
                        if use_atr_stop:
                            row_data = td2.iloc[0]
                            stop = _get_atr_stop(row_data, atr_multiplier)
                        else:
                            stop = fixed_stop_loss

                        # Position sizing: compounding or fixed
                        current_equity = capital
                        for pos in positions:
                            td3 = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
                            current_equity += pos["shares"] * (
                                td3["Close"].iloc[0] if len(td3) > 0 else pos["entry_price"]
                            )

                        if use_compounding:
                            pos_val = min(capital, current_equity / max_positions)
                        else:
                            pos_val = min(capital, start_capital / max_positions)

                        # Transaction cost on entry
                        if use_txn_costs:
                            entry_cost = pos_val * (txn_cost_bps / 10_000)
                            pos_val -= entry_cost

                        shares = int(pos_val / ep)
                        if shares <= 0:
                            continue

                        capital -= shares * ep
                        if use_txn_costs:
                            capital -= shares * ep * (txn_cost_bps / 10_000)

                        positions.append({
                            "ticker": ticker,
                            "entry_date": today,
                            "entry_price": ep,
                            "shares": shares,
                            "hold_days": hold_days,
                            "days_held": 0,
                            "current_return": 0,
                            "probability": prob,
                            "ev_score": ev,
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

    # Close remaining positions
    for pos in positions:
        last = df[df["Ticker"] == pos["ticker"]]
        if len(last) > 0:
            ep2 = last.iloc[-1]["Close"]
            ret2 = (ep2 / pos["entry_price"] - 1) * 100
            if use_txn_costs:
                exit_cost = pos["shares"] * ep2 * (txn_cost_bps / 10_000)
                ret2 = ((pos["shares"] * ep2 - exit_cost) / (pos["shares"] * pos["entry_price"]) - 1) * 100
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
                "EV_Score": pos.get("ev_score", 0),
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
#  ABLATION STUDY — test each improvement individually
# ═══════════════════════════════════════════════════════════════════════════

def run_ablation_study(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
) -> pd.DataFrame:
    """
    Run the strategy with each improvement toggled on/off to measure
    the individual contribution of each enhancement.
    """
    print("\n" + "═" * 80)
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║       ABLATION STUDY — Value of Each Improvement       ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print("═" * 80)

    configs = [
        ("Baseline (old strategy)",
         dict(use_ensemble=False, use_ev_ranking=False, use_atr_stop=False,
              use_txn_costs=False, use_circuit_breaker=False, use_compounding=False)),
        ("+ Ensemble (XGB+LGBM)",
         dict(use_ensemble=True, use_ev_ranking=False, use_atr_stop=False,
              use_txn_costs=False, use_circuit_breaker=False, use_compounding=False)),
        ("+ EV Ranking (P×E)",
         dict(use_ensemble=True, use_ev_ranking=True, use_atr_stop=False,
              use_txn_costs=False, use_circuit_breaker=False, use_compounding=False)),
        ("+ ATR Dynamic Stop",
         dict(use_ensemble=True, use_ev_ranking=True, use_atr_stop=True,
              use_txn_costs=False, use_circuit_breaker=False, use_compounding=False)),
        ("+ Transaction Costs",
         dict(use_ensemble=True, use_ev_ranking=True, use_atr_stop=True,
              use_txn_costs=True, use_circuit_breaker=False, use_compounding=False)),
        ("+ Circuit Breaker",
         dict(use_ensemble=True, use_ev_ranking=True, use_atr_stop=True,
              use_txn_costs=True, use_circuit_breaker=True, use_compounding=False)),
        ("FULL ENHANCED (all 6)",
         dict(use_ensemble=True, use_ev_ranking=True, use_atr_stop=True,
              use_txn_costs=True, use_circuit_breaker=True, use_compounding=True)),
    ]

    results = []
    equity_curves = {}

    for label, kwargs in configs:
        print(f"\n  {'▸'*3} {label}")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        bt = run_enhanced_backtest(
            returns=returns, classifier_dict=classifier_dict,
            regime_data=regime_data, spy_data=spy_data,
            start_capital=100_000, hold_days=3, min_probability=0.50,
            top_n=1, max_positions=1, retrain_every=63,
            warmup_days=warmup_days, verbose=False,
            **kwargs,
        )
        sys.stdout = old_stdout

        eq = bt.get("equity_curve", pd.DataFrame())
        trades = bt.get("trades", pd.DataFrame())
        sc = bt.get("start_capital", 100_000)
        fv = bt.get("final_value", sc)
        total_ret = (fv / sc - 1) * 100

        sharpe = 0
        max_dd = 0
        wr = 0

        if len(eq) > 0:
            dr = eq["Portfolio_Value"].pct_change().dropna()
            if len(dr) > 10 and dr.std() > 0:
                sharpe = (dr.mean() / dr.std()) * np.sqrt(252)
            rm = eq["Portfolio_Value"].cummax()
            max_dd = ((eq["Portfolio_Value"] - rm) / rm * 100).min()

        if len(trades) > 0:
            wr = (trades["Return_Pct"] > 0).mean() * 100

        # SPY alpha
        alpha = total_ret
        if spy_data is not None and len(eq) > 0:
            sp = spy_data.loc[(spy_data.index >= eq.index[0]) & (spy_data.index <= eq.index[-1])]
            if len(sp) >= 2:
                sr = (sp.iloc[-1] / sp.iloc[0] - 1) * 100
                if isinstance(sr, pd.Series):
                    sr = sr.iloc[0]
                alpha = total_ret - sr

        print(f"    Return: {total_ret:+.1f}%  Sharpe: {sharpe:.2f}  "
              f"DD: {max_dd:.1f}%  WR: {wr:.0f}%  Alpha: {alpha:+.1f}%  "
              f"Trades: {len(trades)}")

        results.append({
            "Config": label,
            "Return_%": total_ret,
            "Sharpe": sharpe,
            "Max_DD_%": max_dd,
            "Win_Rate_%": wr,
            "Alpha_%": alpha,
            "N_Trades": len(trades),
            "Final_Value": fv,
        })

        if len(eq) > 0:
            equity_curves[label] = eq["Portfolio_Value"]

    rdf = pd.DataFrame(results)

    print(f"\n{'═'*80}")
    print(f"  ABLATION SUMMARY")
    print(f"{'═'*80}")
    for _, row in rdf.iterrows():
        marker = " ★" if row["Config"] == "FULL ENHANCED (all 6)" else ""
        print(f"  {row['Config']:35s}  Ret={row['Return_%']:+7.1f}%  "
              f"Sh={row['Sharpe']:5.2f}  DD={row['Max_DD_%']:6.1f}%  "
              f"WR={row['Win_Rate_%']:4.0f}%{marker}")

    # Improvement delta
    base_ret = rdf.iloc[0]["Return_%"]
    full_ret = rdf.iloc[-1]["Return_%"]
    base_sh = rdf.iloc[0]["Sharpe"]
    full_sh = rdf.iloc[-1]["Sharpe"]
    print(f"\n  Baseline → Enhanced:")
    print(f"    Return: {base_ret:+.1f}% → {full_ret:+.1f}% ({full_ret - base_ret:+.1f}%)")
    print(f"    Sharpe: {base_sh:.2f} → {full_sh:.2f} ({full_sh - base_sh:+.2f})")

    return rdf, equity_curves


# ═══════════════════════════════════════════════════════════════════════════
#  HEAD-TO-HEAD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def compare_strategies(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
) -> dict:
    """
    Run baseline vs enhanced strategy and produce a comparison report.
    """
    print("\n" + "═" * 80)
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║     HEAD-TO-HEAD: BASELINE vs ENHANCED STRATEGY        ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print("═" * 80)

    t0 = time.time()

    # 1. Baseline (replicate old strategy exactly)
    print("\n  ▶ Running BASELINE (probability-only, fixed stop, no costs)...")
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    bt_base = run_enhanced_backtest(
        returns=returns, classifier_dict=classifier_dict,
        regime_data=regime_data, spy_data=spy_data,
        start_capital=100_000, hold_days=3, min_probability=0.50,
        top_n=1, max_positions=1, retrain_every=63,
        warmup_days=warmup_days, verbose=False,
        use_ensemble=False, use_ev_ranking=False, use_atr_stop=False,
        use_txn_costs=False, use_circuit_breaker=False, use_compounding=False,
    )
    sys.stdout = old_stdout

    # 2. Enhanced (all 6 improvements)
    print("  ▶ Running ENHANCED (all 6 improvements)...")
    sys.stdout = io.StringIO()
    bt_enh = run_enhanced_backtest(
        returns=returns, classifier_dict=classifier_dict,
        regime_data=regime_data, spy_data=spy_data,
        start_capital=100_000, hold_days=3, min_probability=0.50,
        top_n=1, max_positions=1, retrain_every=63,
        warmup_days=warmup_days, verbose=False,
        use_ensemble=True, use_ev_ranking=True, use_atr_stop=True,
        use_txn_costs=True, use_circuit_breaker=True, use_compounding=True,
    )
    sys.stdout = old_stdout

    elapsed = time.time() - t0

    # Compute stats for both
    def _stats(bt, label):
        eq = bt.get("equity_curve", pd.DataFrame())
        trades = bt.get("trades", pd.DataFrame())
        sc = bt.get("start_capital", 100_000)
        fv = bt.get("final_value", sc)
        ret = (fv / sc - 1) * 100
        s = {"label": label, "total_return": ret, "final_value": fv,
             "n_trades": len(trades)}
        if len(trades) > 0:
            s["win_rate"] = (trades["Return_Pct"] > 0).mean() * 100
            s["avg_return"] = trades["Return_Pct"].mean()
            wins = trades[trades["Return_Pct"] > 0]
            losses = trades[trades["Return_Pct"] <= 0]
            s["profit_factor"] = (
                wins["PnL"].sum() / abs(losses["PnL"].sum())
                if len(losses) > 0 and losses["PnL"].sum() != 0 else 99.0
            )
        if len(eq) > 0:
            dr = eq["Portfolio_Value"].pct_change().dropna()
            s["sharpe"] = (dr.mean() / dr.std()) * np.sqrt(252) if len(dr) > 10 and dr.std() > 0 else 0
            rm = eq["Portfolio_Value"].cummax()
            s["max_dd"] = ((eq["Portfolio_Value"] - rm) / rm * 100).min()
            mv = eq["Portfolio_Value"].resample("ME").last()
            mr = mv.pct_change().dropna() * 100
            s["monthly_win_rate"] = (mr > 0).sum() / len(mr) * 100 if len(mr) > 0 else 0
        if spy_data is not None and len(eq) > 0:
            sp = spy_data.loc[(spy_data.index >= eq.index[0]) & (spy_data.index <= eq.index[-1])]
            if len(sp) >= 2:
                sr = (sp.iloc[-1] / sp.iloc[0] - 1) * 100
                if isinstance(sr, pd.Series):
                    sr = sr.iloc[0]
                s["spy_return"] = sr
                s["alpha"] = ret - sr
        return s

    s_base = _stats(bt_base, "Baseline")
    s_enh = _stats(bt_enh, "Enhanced")

    # Report
    print(f"\n  {'─'*70}")
    print(f"  {'Metric':25s}  {'Baseline':>12s}  {'Enhanced':>12s}  {'Delta':>10s}")
    print(f"  {'─'*70}")

    metrics = [
        ("Total Return", "total_return", "%", ".1f"),
        ("Sharpe Ratio", "sharpe", "", ".2f"),
        ("Max Drawdown", "max_dd", "%", ".1f"),
        ("Win Rate", "win_rate", "%", ".0f"),
        ("Monthly WR", "monthly_win_rate", "%", ".0f"),
        ("Avg Return/Trade", "avg_return", "%", ".2f"),
        ("Profit Factor", "profit_factor", "x", ".2f"),
        ("Trades", "n_trades", "", "d"),
        ("Alpha vs SPY", "alpha", "%", ".1f"),
    ]

    for label, key, suffix, fmt in metrics:
        bv = s_base.get(key, 0)
        ev = s_enh.get(key, 0)
        delta = ev - bv
        bstr = f"{bv:{fmt}}{suffix}"
        estr = f"{ev:{fmt}}{suffix}"
        dstr = f"{delta:+{fmt}}{suffix}" if fmt != "d" else f"{delta:+d}"
        better = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(f"  {label:25s}  {bstr:>12s}  {estr:>12s}  {better} {dstr:>8s}")

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  {'═'*70}")

    return {
        "baseline": {"stats": s_base, "backtest": bt_base},
        "enhanced": {"stats": s_enh, "backtest": bt_enh},
        "elapsed": elapsed,
    }
