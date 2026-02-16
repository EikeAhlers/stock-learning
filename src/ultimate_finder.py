"""
Ultimate Stock Finder — 5 critical improvements to find the stocks with
the MOST LIKELY big upside potential.

IMPROVEMENT 1: PROBABILITY CALIBRATION
  When the model says 70%, does it REALLY mean 70%?  
  We bin all predictions and check actual hit rates to build a calibration 
  curve. Then we adjust raw model probabilities to calibrated ones.

IMPROVEMENT 2: STOCK PREDICTABILITY SCORING
  Not all stocks are equally predictable. Some (like ORCL, MU) the model
  nails repeatedly. Others it always gets wrong. We score each stock by 
  how well the model has predicted it historically.

IMPROVEMENT 3: ENSEMBLE CONSENSUS
  Instead of trusting ONE strategy, we run the top 3 winning strategies 
  and only trade when 2+ agree on the same stock. Consensus = higher conviction.

IMPROVEMENT 4: TRUE FORWARD HOLDOUT TEST  
  We carve out the most recent 3 months and NEVER touch them during
  development. One clean, unbiased look at real performance.

IMPROVEMENT 5: SECTOR-RELATIVE RANKING
  A stock that's strong vs its sector is a better signal than one that's 
  just strong in absolute terms. We rank stocks within their sector before 
  overall ranking.
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
from src.strategy_optimizer import _run_strategy


# ═══════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 1: Probability Calibration
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_probabilities(
    returns: pd.DataFrame,
    classifier_dict: dict,
    warmup_days: int = 252,
    retrain_every: int = 63,
    n_bins: int = 10,
) -> dict:
    """
    Walk-forward calibration: for each probability bin, what % actually moved 5%+?
    
    Returns:
      - bin_edges: probability bin boundaries
      - actual_rates: real hit rate in each bin
      - calibration_map: function to convert raw prob → calibrated prob
      - reliability_score: how well-calibrated the model is (Brier decomposition)
    """
    print("=" * 70)
    print("  IMPROVEMENT 1: PROBABILITY CALIBRATION")
    print("  Does 70% really mean 70%?")
    print("=" * 70)
    
    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    
    dates = sorted(df.index.unique())
    trade_dates = dates[warmup_days:]
    
    # Collect all walk-forward predictions
    all_predictions = []
    
    warmup_data = df[df.index <= dates[warmup_days - 1]]
    current_model = _train_classifier_on_window(warmup_data, clf_features)
    if current_model is None:
        print("  ERROR: Not enough data")
        return {}
    
    model = current_model["model"]
    scaler = current_model["scaler"]
    days_since_retrain = 0
    
    for i, today in enumerate(trade_dates):
        if days_since_retrain >= retrain_every:
            past_data = df[df.index < today]
            new_model = _train_classifier_on_window(past_data, clf_features)
            if new_model is not None:
                current_model = new_model
                model = current_model["model"]
                scaler = current_model["scaler"]
                days_since_retrain = 0
        days_since_retrain += 1
        
        today_df = df[df.index == today].copy().reset_index(drop=True)
        if len(today_df) < 10:
            continue
        
        feat_mask = today_df[clf_features].notna().all(axis=1)
        today_feat = today_df.loc[feat_mask, clf_features]
        
        if len(today_feat) == 0:
            continue
        
        X = scaler.transform(today_feat)
        probs = model.predict_proba(X)[:, 1]
        actuals = today_df.loc[feat_mask, "Is_Big_Mover"].values
        tickers = today_df.loc[feat_mask, "Ticker"].values
        
        for prob, actual, ticker in zip(probs, actuals, tickers):
            all_predictions.append({
                "Date": today, "Ticker": ticker,
                "Predicted_Prob": prob, "Actual": actual,
            })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(trade_dates)} days...")
    
    pred_df = pd.DataFrame(all_predictions)
    print(f"\n  Total predictions: {len(pred_df):,}")
    print(f"  Total big moves: {pred_df['Actual'].sum():,}")
    
    # Bin predictions
    pred_df["Prob_Bin"] = pd.cut(pred_df["Predicted_Prob"], bins=n_bins, labels=False)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    calibration = pred_df.groupby("Prob_Bin").agg(
        Mean_Predicted=("Predicted_Prob", "mean"),
        Actual_Rate=("Actual", "mean"),
        Count=("Actual", "count"),
        Big_Moves=("Actual", "sum"),
    ).reset_index()
    
    print(f"\n  CALIBRATION TABLE:")
    print(f"  {'Predicted':>12s} {'Actual':>10s} {'Count':>8s} {'Moves':>8s} {'Ratio':>8s}")
    print(f"  {'─'*52}")
    
    for _, row in calibration.iterrows():
        ratio = row["Actual_Rate"] / max(row["Mean_Predicted"], 0.001) if row["Mean_Predicted"] > 0 else 0
        bar = "█" * int(row["Actual_Rate"] * 100)
        print(f"  {row['Mean_Predicted']:>10.1%} → {row['Actual_Rate']:>8.1%}  "
              f"{row['Count']:>7.0f}  {row['Big_Moves']:>7.0f}  {ratio:>6.2f}x {bar}")
    
    # Build calibration mapping (isotonic-style from binned data)
    cal_map = {}
    for _, row in calibration.iterrows():
        cal_map[row["Prob_Bin"]] = row["Actual_Rate"]
    
    def calibrate(raw_prob):
        """Convert raw model prob to calibrated prob."""
        bin_idx = min(int(raw_prob * n_bins), n_bins - 1)
        return cal_map.get(bin_idx, raw_prob)
    
    # Brier score
    brier = ((pred_df["Predicted_Prob"] - pred_df["Actual"]) ** 2).mean()
    
    # Reliability diagram assessment
    top_bin = calibration[calibration["Mean_Predicted"] >= 0.5]
    if len(top_bin) > 0:
        top_actual = top_bin["Actual_Rate"].mean()
        top_predicted = top_bin["Mean_Predicted"].mean()
        overconfidence = top_predicted - top_actual
    else:
        overconfidence = 0
    
    print(f"\n  Brier Score: {brier:.4f} (lower is better, <0.1 is good)")
    if overconfidence > 0.1:
        print(f"  ⚠ Model is OVERCONFIDENT by {overconfidence:.1%} in high-probability bins")
    elif overconfidence < -0.05:
        print(f"  ✓ Model is actually UNDERCONFIDENT — real hits > predicted ({abs(overconfidence):.1%})")
    else:
        print(f"  ✓ Model is reasonably well-calibrated (off by {abs(overconfidence):.1%})")
    
    return {
        "predictions_df": pred_df,
        "calibration_table": calibration,
        "calibrate_fn": calibrate,
        "n_bins": n_bins,
        "brier_score": brier,
        "overconfidence": overconfidence,
        "cal_map": cal_map,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 2: Stock Predictability Scoring  
# ═══════════════════════════════════════════════════════════════════════════

def score_stock_predictability(
    calibration_result: dict,
    min_predictions: int = 50,
    min_high_prob_events: int = 3,
) -> pd.DataFrame:
    """
    For each stock, measure how well the model predicts it.
    
    Returns DataFrame with:
      - Ticker
      - Total predictions
      - High-prob predictions (>50%)
      - Actual hit rate when high-prob  
      - Precision score (high-prob correct / high-prob total)
      - Predictability rank
    """
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 2: STOCK PREDICTABILITY SCORING")
    print("  Which stocks does the model predict best?")
    print("=" * 70)
    
    pred_df = calibration_result["predictions_df"]
    
    # Score each stock
    results = []
    for ticker in pred_df["Ticker"].unique():
        t_df = pred_df[pred_df["Ticker"] == ticker]
        
        if len(t_df) < min_predictions:
            continue
        
        total = len(t_df)
        actual_moves = t_df["Actual"].sum()
        base_rate = actual_moves / total
        
        # High-probability predictions (model thinks > 50%)
        high_prob = t_df[t_df["Predicted_Prob"] >= 0.5]
        n_high = len(high_prob)
        
        if n_high < min_high_prob_events:
            precision = 0
            high_hit_rate = 0
        else:
            high_hits = high_prob["Actual"].sum()
            precision = high_hits / n_high
            high_hit_rate = high_hits / max(actual_moves, 1)
        
        # Low-probability should have few actual moves (good negative detection)
        low_prob = t_df[t_df["Predicted_Prob"] < 0.3]
        if len(low_prob) > 0:
            false_calm_rate = low_prob["Actual"].mean()
        else:
            false_calm_rate = 0
        
        # Average return when model is very confident  
        high_prob_tickers_data = t_df[t_df["Predicted_Prob"] >= 0.5]
        
        # Predictability score combines:
        # 1. Precision when confident (+)
        # 2. Few false calms (+)
        # 3. Enough high-prob events (sample size)
        sample_bonus = min(n_high / 10, 1.0)
        score = (
            0.5 * precision + 
            0.3 * (1 - false_calm_rate) + 
            0.2 * sample_bonus
        ) * 100
        
        results.append({
            "Ticker": ticker,
            "Total_Predictions": total,
            "Actual_Big_Moves": actual_moves,
            "Base_Rate": base_rate,
            "High_Prob_Count": n_high,
            "Precision_at_50pct": precision,
            "High_Hit_Rate": high_hit_rate,
            "False_Calm_Rate": false_calm_rate,
            "Predictability_Score": score,
        })
    
    result_df = pd.DataFrame(results).sort_values("Predictability_Score", ascending=False)
    result_df["Rank"] = range(1, len(result_df) + 1)
    
    # Print results
    n_predictable = (result_df["Predictability_Score"] >= 60).sum()
    n_total = len(result_df)
    
    print(f"\n  Stocks analyzed: {n_total}")
    print(f"  Highly predictable (score >= 60): {n_predictable}")
    print(f"\n  TOP 20 MOST PREDICTABLE STOCKS:")
    print(f"  {'Rank':>5s} {'Ticker':>8s} {'Score':>7s} {'Prec':>7s} {'Moves':>6s} "
          f"{'High-P':>7s} {'Base':>6s} {'FalseCm':>7s}")
    print(f"  {'─'*58}")
    
    for _, row in result_df.head(20).iterrows():
        print(f"  {row['Rank']:>5.0f} {row['Ticker']:>8s} {row['Predictability_Score']:>6.1f} "
              f"{row['Precision_at_50pct']:>6.1%} {row['Actual_Big_Moves']:>5.0f} "
              f"{row['High_Prob_Count']:>6.0f} {row['Base_Rate']:>5.1%} "
              f"{row['False_Calm_Rate']:>6.1%}")
    
    print(f"\n  BOTTOM 10 (HARDEST TO PREDICT):")
    for _, row in result_df.tail(10).iterrows():
        print(f"  {row['Rank']:>5.0f} {row['Ticker']:>8s} {row['Predictability_Score']:>6.1f} "
              f"{row['Precision_at_50pct']:>6.1%}")
    
    return result_df


# ═══════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 3: Ensemble Consensus
# ═══════════════════════════════════════════════════════════════════════════

def run_ensemble_backtest(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    ticker_sectors: dict = None,
    spy_data: pd.Series = None,
    warmup_days: int = 252,
    start_capital: float = 100_000,
) -> dict:
    """
    Run the top 3 strategies in parallel and only trade when 2+ agree.
    
    Strategy 1: Momentum concentrated (H3d, P50%, SL-7%, 1 pos)
    Strategy 2: Regime filtered (H5d, P50%, SL-7%, 3 pos)
    Strategy 3: Base (H3d, P50%, SL-7%, 3 pos)
    
    Trade only when ≥2 strategies want the same stock on the same day.
    """
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 3: ENSEMBLE CONSENSUS BACKTEST")
    print("  Only trade when multiple strategies agree")
    print("=" * 70)
    
    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    
    dates = sorted(df.index.unique())
    trade_dates = dates[warmup_days:]
    
    print(f"  Test period: {trade_dates[0].date()} → {trade_dates[-1].date()}")
    
    # Train model once for scoring (same walk-forward)
    warmup_data = df[df.index <= dates[warmup_days - 1]]
    current_model = _train_classifier_on_window(warmup_data, clf_features)
    if current_model is None:
        return {}
    
    model = current_model["model"]
    scaler = current_model["scaler"]
    days_since_retrain = 0
    retrain_every = 63
    
    capital = start_capital
    positions = []
    trade_log = []
    equity_curve = []
    
    hold_days = 3
    stop_loss_pct = -7.0
    max_positions = 3
    min_prob = 0.50
    
    for i, today in enumerate(trade_dates):
        # Retrain
        if days_since_retrain >= retrain_every:
            past_data = df[df.index < today]
            new_model = _train_classifier_on_window(past_data, clf_features)
            if new_model is not None:
                current_model = new_model
                model = current_model["model"]
                scaler = current_model["scaler"]
                days_since_retrain = 0
        days_since_retrain += 1
        
        # Check regime
        in_bear = False
        if regime_data is not None and today in regime_data.index:
            r_val = regime_data.loc[today]
            r_str = str(r_val.get("Market_Regime", "")) if isinstance(r_val, pd.Series) else str(r_val)
            in_bear = "bear" in r_str.lower()
        
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
            
            exit_ = False
            reason = ""
            if ret <= stop_loss_pct:
                exit_ = True; reason = "stop_loss"
            elif pos["days_held"] >= hold_days:
                exit_ = True; reason = "hold_done"
            
            if exit_:
                pnl = pos["shares"] * pos["entry_price"] * (ret / 100)
                capital += pos["shares"] * cp
                trade_log.append({
                    "Ticker": pos["ticker"], "Entry_Date": pos["entry_date"],
                    "Exit_Date": today, "Return_Pct": ret, "PnL": pnl,
                    "Hold_Days": pos["days_held"],
                    "Probability": pos["probability"],
                    "Consensus": pos.get("consensus", 0),
                })
                closed.append(pos)
        for c in closed:
            positions.remove(c)
        
        # Score stocks
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
                        "Prob": probs,
                    })
                    
                    # Strategy 1: Momentum filter
                    s1_picks = set()
                    if "Prev_Return_20d" in today_all.columns:
                        mom = today_all.loc[scores["idx"], "Prev_Return_20d"]
                        s1 = scores[(scores["Prob"] >= min_prob) & (mom.values > 0)]
                        s1 = s1.sort_values("Prob", ascending=False).head(1)
                        s1_picks = set(s1["Ticker"])
                    
                    # Strategy 2: Regime filtered
                    s2_picks = set()
                    if not in_bear:
                        s2 = scores[scores["Prob"] >= min_prob].sort_values("Prob", ascending=False).head(3)
                        s2_picks = set(s2["Ticker"])
                    
                    # Strategy 3: Base (top 3 by prob)
                    s3 = scores[scores["Prob"] >= min_prob].sort_values("Prob", ascending=False).head(3)
                    s3_picks = set(s3["Ticker"])
                    
                    # CONSENSUS: count agreements
                    all_tickers_today = s1_picks | s2_picks | s3_picks
                    consensus_scores = {}
                    for tk in all_tickers_today:
                        count = (tk in s1_picks) + (tk in s2_picks) + (tk in s3_picks)
                        consensus_scores[tk] = count
                    
                    # Only trade consensus >= 2
                    consensus_picks = {tk: c for tk, c in consensus_scores.items() if c >= 2}
                    
                    # Sort by consensus then probability
                    if consensus_picks:
                        held = [p["ticker"] for p in positions]
                        pick_list = []
                        for tk, cons in consensus_picks.items():
                            if tk in held:
                                continue
                            prob_val = scores[scores["Ticker"] == tk]["Prob"].values
                            if len(prob_val) > 0:
                                pick_list.append((tk, cons, prob_val[0]))
                        
                        pick_list.sort(key=lambda x: (-x[1], -x[2]))
                        
                        for tk, cons, prob in pick_list[:open_slots]:
                            td_row = today_all[today_all["Ticker"] == tk]
                            if len(td_row) == 0:
                                continue
                            ep = td_row["Close"].iloc[0]
                            if ep <= 0:
                                continue
                            pv = min(capital, start_capital / max_positions)
                            shares = int(pv / ep)
                            if shares <= 0:
                                continue
                            capital -= shares * ep
                            positions.append({
                                "ticker": tk, "entry_date": today,
                                "entry_price": ep, "shares": shares,
                                "hold_days": hold_days, "days_held": 0,
                                "current_return": 0, "probability": prob,
                                "consensus": cons,
                            })
        
        # Portfolio value
        pv = capital
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            pv += pos["shares"] * (td["Close"].iloc[0] if len(td) > 0 else pos["entry_price"])
        equity_curve.append({"Date": today, "Portfolio_Value": pv})
        
        if (i + 1) % 50 == 0:
            ret_pct = (pv / start_capital - 1) * 100
            print(f"  Day {i+1:4d} | {today.date()} | ${pv:>10,.0f} ({ret_pct:+.1f}%) | Trades: {len(trade_log)}")
    
    # Close remaining
    for pos in positions:
        last = df[df["Ticker"] == pos["ticker"]]
        if len(last) > 0:
            ep = last.iloc[-1]["Close"]
            ret = (ep / pos["entry_price"] - 1) * 100
            capital += pos["shares"] * ep
            trade_log.append({
                "Ticker": pos["ticker"], "Entry_Date": pos["entry_date"],
                "Exit_Date": trade_dates[-1], "Return_Pct": ret,
                "PnL": pos["shares"] * pos["entry_price"] * (ret / 100),
                "Hold_Days": pos["days_held"],
                "Probability": pos.get("probability", 0),
                "Consensus": pos.get("consensus", 0),
            })
    
    eq_df = pd.DataFrame(equity_curve).set_index("Date")
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    final_val = eq_df["Portfolio_Value"].iloc[-1] if len(eq_df) > 0 else start_capital
    
    # Stats
    total_return = (final_val / start_capital - 1) * 100
    
    if len(trades_df) > 0:
        wr = (trades_df["Return_Pct"] > 0).mean() * 100
        avg_ret = trades_df["Return_Pct"].mean()
        wins = trades_df[trades_df["Return_Pct"] > 0]
        losses = trades_df[trades_df["Return_Pct"] <= 0]
        pf = wins["PnL"].sum() / abs(losses["PnL"].sum()) if len(losses) > 0 and losses["PnL"].sum() != 0 else 99
    else:
        wr = avg_ret = pf = 0
    
    rolling_max = eq_df["Portfolio_Value"].cummax()
    dd = ((eq_df["Portfolio_Value"] - rolling_max) / rolling_max * 100).min()
    
    daily_rets = eq_df["Portfolio_Value"].pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if len(daily_rets) > 10 and daily_rets.std() > 0 else 0
    
    # Monthly consistency
    monthly_vals = eq_df["Portfolio_Value"].resample("ME").last()
    monthly_rets = monthly_vals.pct_change().dropna() * 100
    m_wr = (monthly_rets > 0).sum() / len(monthly_rets) * 100 if len(monthly_rets) > 0 else 0
    
    spy_ret = 0
    if spy_data is not None:
        sp = spy_data.loc[(spy_data.index >= eq_df.index[0]) & (spy_data.index <= eq_df.index[-1])]
        if len(sp) >= 2:
            spy_ret = (sp.iloc[-1] / sp.iloc[0] - 1) * 100
            if isinstance(spy_ret, pd.Series):
                spy_ret = spy_ret.iloc[0]
    
    # Consensus analysis
    if len(trades_df) > 0 and "Consensus" in trades_df.columns:
        c3 = trades_df[trades_df["Consensus"] == 3]
        c2 = trades_df[trades_df["Consensus"] == 2]
        print(f"\n  CONSENSUS BREAKDOWN:")
        if len(c3) > 0:
            print(f"  3/3 agree: {len(c3)} trades, WR={( c3['Return_Pct'] > 0).mean()*100:.0f}%, "
                  f"Avg={c3['Return_Pct'].mean():+.1f}%")
        if len(c2) > 0:
            print(f"  2/3 agree: {len(c2)} trades, WR={(c2['Return_Pct'] > 0).mean()*100:.0f}%, "
                  f"Avg={c2['Return_Pct'].mean():+.1f}%")
    
    print(f"\n  ENSEMBLE RESULTS:")
    print(f"  $100K → ${final_val:,.0f} ({total_return:+.1f}%)")
    print(f"  SPY: {spy_ret:+.1f}%  |  Alpha: {total_return - spy_ret:+.1f}%")
    print(f"  Sharpe: {sharpe:.2f}  |  Max DD: {dd:.1f}%")
    print(f"  Win Rate: {wr:.0f}%  |  Profit Factor: {pf:.2f}")
    print(f"  Monthly WR: {m_wr:.0f}%  |  Trades: {len(trades_df)}")
    
    return {
        "equity_curve": eq_df,
        "trades": trades_df,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": dd,
        "win_rate": wr,
        "profit_factor": pf,
        "monthly_win_rate": m_wr,
        "spy_return": spy_ret,
        "alpha": total_return - spy_ret,
        "final_value": final_val,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 4: True Forward Holdout Test
# ═══════════════════════════════════════════════════════════════════════════

def forward_holdout_test(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    holdout_months: int = 3,
    warmup_days: int = 252,
) -> dict:
    """
    True out-of-sample test.
    
    1. Use ALL data except the last 3 months for development
    2. Train a final model on that data
    3. Run the winning strategy on the holdout period ONCE
    4. Report results — this is the truest test we have
    """
    print("\n" + "=" * 70)
    print(f"  IMPROVEMENT 4: TRUE FORWARD HOLDOUT ({holdout_months}mo)")
    print("  Data the model has NEVER seen during development")
    print("=" * 70)
    
    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    
    dates = sorted(df.index.unique())
    
    # Holdout = last N months
    holdout_start_idx = len(dates) - holdout_months * 21  # ~21 trading days/month
    holdout_dates = dates[holdout_start_idx:]
    dev_dates = dates[:holdout_start_idx]
    
    print(f"  Development period: {dates[0].date()} → {dev_dates[-1].date()} ({len(dev_dates)} days)")
    print(f"  Holdout period:     {holdout_dates[0].date()} → {holdout_dates[-1].date()} ({len(holdout_dates)} days)")
    
    # Train model on ALL development data
    dev_data = df[df.index <= dev_dates[-1]]
    final_model = _train_classifier_on_window(dev_data, clf_features)
    if final_model is None:
        print("  ERROR: Cannot train model")
        return {}
    
    model = final_model["model"]
    scaler = final_model["scaler"]
    
    # Run strategy on holdout
    capital = 100_000
    start_capital = capital
    positions = []
    trade_log = []
    equity_curve = []
    
    hold_days = 3
    stop_loss = -7.0
    max_pos = 3
    min_prob = 0.50
    
    for today in holdout_dates:
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
            
            exit_ = False
            reason = ""
            if ret <= stop_loss:
                exit_ = True; reason = "stop_loss"
            elif pos["days_held"] >= hold_days:
                exit_ = True; reason = "hold_done"
            
            if exit_:
                pnl = pos["shares"] * pos["entry_price"] * (ret / 100)
                capital += pos["shares"] * cp
                trade_log.append({
                    "Ticker": pos["ticker"], "Entry_Date": pos["entry_date"],
                    "Exit_Date": today, "Return_Pct": ret, "PnL": pnl,
                    "Hold_Days": pos["days_held"],
                })
                closed.append(pos)
        for c in closed:
            positions.remove(c)
        
        # New positions
        open_slots = max_pos - len(positions)
        if open_slots > 0:
            today_all = df[df.index == today].copy().reset_index(drop=True)
            if len(today_all) > 10:
                fm = today_all[clf_features].notna().all(axis=1)
                tf = today_all.loc[fm, clf_features]
                if len(tf) > 0:
                    X = scaler.transform(tf)
                    probs = model.predict_proba(X)[:, 1]
                    
                    # Momentum filter (winning strategy)
                    scores = pd.DataFrame({
                        "idx": today_all.loc[fm].index,
                        "Ticker": today_all.loc[fm, "Ticker"].values,
                        "Prob": probs,
                    })
                    scores = scores[scores["Prob"] >= min_prob]
                    
                    if "Prev_Return_20d" in today_all.columns:
                        mom = today_all.loc[scores["idx"], "Prev_Return_20d"]
                        scores = scores[mom.values > 0]
                    
                    held = [p["ticker"] for p in positions]
                    scores = scores[~scores["Ticker"].isin(held)]
                    scores = scores.sort_values("Prob", ascending=False)
                    
                    for j in range(min(open_slots, 1, len(scores))):  # concentrated
                        pick = scores.iloc[j]
                        td_row = today_all[today_all["Ticker"] == pick["Ticker"]]
                        if len(td_row) == 0: continue
                        ep = td_row["Close"].iloc[0]
                        if ep <= 0: continue
                        pv = min(capital, start_capital / max_pos)
                        shares = int(pv / ep)
                        if shares <= 0: continue
                        capital -= shares * ep
                        positions.append({
                            "ticker": pick["Ticker"], "entry_date": today,
                            "entry_price": ep, "shares": shares,
                            "hold_days": hold_days, "days_held": 0,
                        })
        
        pv = capital
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            pv += pos["shares"] * (td["Close"].iloc[0] if len(td) > 0 else pos["entry_price"])
        equity_curve.append({"Date": today, "Portfolio_Value": pv})
    
    # Close remaining
    for pos in positions:
        last = df[df["Ticker"] == pos["ticker"]]
        if len(last) > 0:
            ep = last.iloc[-1]["Close"]
            ret = (ep / pos["entry_price"] - 1) * 100
            capital += pos["shares"] * ep
            trade_log.append({
                "Ticker": pos["ticker"], "Entry_Date": pos["entry_date"],
                "Exit_Date": holdout_dates[-1], "Return_Pct": ret,
                "PnL": pos["shares"] * pos["entry_price"] * (ret / 100),
                "Hold_Days": pos["days_held"],
            })
    
    eq_df = pd.DataFrame(equity_curve).set_index("Date")
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    final_val = eq_df["Portfolio_Value"].iloc[-1] if len(eq_df) > 0 else start_capital
    total_return = (final_val / start_capital - 1) * 100
    
    # Stats
    if len(trades_df) > 0:
        wr = (trades_df["Return_Pct"] > 0).mean() * 100
        avg_ret = trades_df["Return_Pct"].mean()
    else:
        wr = avg_ret = 0
    
    daily_rets = eq_df["Portfolio_Value"].pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if len(daily_rets) > 10 and daily_rets.std() > 0 else 0
    
    rolling_max = eq_df["Portfolio_Value"].cummax()
    dd = ((eq_df["Portfolio_Value"] - rolling_max) / rolling_max * 100).min()
    
    spy_ret = 0
    if spy_data is not None:
        sp = spy_data.loc[(spy_data.index >= eq_df.index[0]) & (spy_data.index <= eq_df.index[-1])]
        if len(sp) >= 2:
            spy_ret = (sp.iloc[-1] / sp.iloc[0] - 1) * 100
            if isinstance(spy_ret, pd.Series): spy_ret = spy_ret.iloc[0]
    
    print(f"\n  HOLDOUT RESULTS ({holdout_months} months, NEVER SEEN before):")
    print(f"  $100K → ${final_val:,.0f} ({total_return:+.1f}%)")
    print(f"  SPY same period: {spy_ret:+.1f}%")
    print(f"  Alpha: {total_return - spy_ret:+.1f}%")
    print(f"  Sharpe: {sharpe:.2f}  |  Max DD: {dd:.1f}%")
    print(f"  Win Rate: {wr:.0f}%  |  Avg Return/Trade: {avg_ret:+.1f}%")
    print(f"  Trades: {len(trades_df)}")
    
    if len(trades_df) > 0:
        print(f"\n  HOLDOUT TRADES:")
        for _, t in trades_df.iterrows():
            d = t["Entry_Date"]
            dstr = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            print(f"    {t['Ticker']:6s} {dstr}  {t['Return_Pct']:+6.1f}%  ${t['PnL']:+,.0f}")
    
    # Verdict
    if total_return > spy_ret and total_return > 0:
        verdict = "PASS — Strategy works on truly unseen data"
    elif total_return > 0:
        verdict = "PARTIAL — Profitable but didn't beat SPY in holdout"
    else:
        verdict = "FAIL — Lost money on unseen data"
    
    print(f"\n  VERDICT: {verdict}")
    
    return {
        "equity_curve": eq_df,
        "trades": trades_df,
        "total_return": total_return,
        "spy_return": spy_ret,
        "alpha": total_return - spy_ret,
        "sharpe": sharpe,
        "max_dd": dd,
        "win_rate": wr,
        "verdict": verdict,
        "holdout_months": holdout_months,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 5: Sector-Relative Strength + Combined Ranking
# ═══════════════════════════════════════════════════════════════════════════

def compute_sector_relative_scores(
    returns: pd.DataFrame,
    classifier_dict: dict,
    predictability_df: pd.DataFrame,
    calibration_result: dict,
    ticker_sectors: dict = None,
    warmup_days: int = 252,
) -> pd.DataFrame:
    """
    For TODAY's stocks, compute a combined ranking using:
    1. Model probability (calibrated)
    2. Stock predictability score
    3. Sector-relative strength (stock vs sector average)
    4. Recent momentum quality
    
    Returns ranked DataFrame of today's best opportunities.
    """
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 5: COMBINED RANKING — Today's Best Opportunities")
    print("  Model Prob × Predictability × Sector Strength × Momentum")
    print("=" * 70)
    
    df = add_all_lagged_features(returns.copy())
    clf_features = classifier_dict["features"]
    dates = sorted(df.index.unique())
    today = dates[-1]
    
    # Train on all data up to today
    current_model = _train_classifier_on_window(df[df.index <= today], clf_features)
    if current_model is None:
        print("  ERROR: Cannot train model")
        return pd.DataFrame()
    
    model = current_model["model"]
    scaler = current_model["scaler"]
    
    today_df = df[df.index == today].copy().reset_index(drop=True)
    
    if len(today_df) < 10:
        print("  Not enough data for today")
        return pd.DataFrame()
    
    fm = today_df[clf_features].notna().all(axis=1)
    tf = today_df.loc[fm, clf_features]
    
    if len(tf) == 0:
        return pd.DataFrame()
    
    X = scaler.transform(tf)
    probs = model.predict_proba(X)[:, 1]
    
    results = pd.DataFrame({
        "Ticker": today_df.loc[fm, "Ticker"].values,
        "Raw_Prob": probs,
        "Close": today_df.loc[fm, "Close"].values,
    })
    
    # Add calibrated probability
    cal_fn = calibration_result.get("calibrate_fn")
    if cal_fn:
        results["Cal_Prob"] = results["Raw_Prob"].apply(cal_fn)
    else:
        results["Cal_Prob"] = results["Raw_Prob"]
    
    # Add predictability score
    pred_map = predictability_df.set_index("Ticker")["Predictability_Score"].to_dict()
    results["Predictability"] = results["Ticker"].map(pred_map).fillna(50)
    
    # Add sector info
    if ticker_sectors:
        results["Sector"] = results["Ticker"].map(ticker_sectors).fillna("Unknown")
    else:
        results["Sector"] = "Unknown"
    
    # Sector-relative momentum
    if "Prev_Return_20d" in today_df.columns:
        ret20 = today_df.loc[fm, "Prev_Return_20d"].values
        results["Return_20d"] = ret20
        
        sector_avg = results.groupby("Sector")["Return_20d"].transform("mean")
        results["Sector_Rel_Strength"] = results["Return_20d"] - sector_avg
    else:
        results["Return_20d"] = 0
        results["Sector_Rel_Strength"] = 0
    
    # Additional features
    if "Prev_Vol_Ratio" in today_df.columns:
        results["Vol_Ratio"] = today_df.loc[fm, "Prev_Vol_Ratio"].values
    else:
        results["Vol_Ratio"] = 1.0
    
    if "Prev_RSI_14" in today_df.columns:
        results["RSI"] = today_df.loc[fm, "Prev_RSI_14"].values
    else:
        results["RSI"] = 50
    
    # COMBINED SCORE (weighted)
    # Normalize each component to 0-100
    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(50, index=series.index)
        return (series - mn) / (mx - mn) * 100
    
    results["Prob_Score"] = normalize(results["Cal_Prob"])
    results["Pred_Score"] = normalize(results["Predictability"])
    results["SRS_Score"] = normalize(results["Sector_Rel_Strength"])
    results["Mom_Score"] = normalize(results["Return_20d"].clip(-30, 30))
    
    # Combined = weighted average
    results["Combined_Score"] = (
        0.35 * results["Prob_Score"] +      # Model confidence (most important)
        0.25 * results["Pred_Score"] +       # Stock predictability
        0.20 * results["SRS_Score"] +        # Sector-relative strength
        0.20 * results["Mom_Score"]          # Momentum quality
    )
    
    results = results.sort_values("Combined_Score", ascending=False)
    results["Rank"] = range(1, len(results) + 1)
    
    # Print top picks
    top = results.head(20)
    print(f"\n  Date: {today.date()}")
    print(f"  Stocks scored: {len(results)}")
    print(f"\n  TOP 20 OPPORTUNITIES (Combined Ranking):")
    print(f"  {'Rank':>5s} {'Ticker':>7s} {'RawP':>6s} {'CalP':>6s} {'Pred':>5s} "
          f"{'SRS':>6s} {'Mom20d':>7s} {'Vol':>5s} {'RSI':>4s} {'Score':>6s} {'Sector':>15s}")
    print(f"  {'─'*82}")
    
    for _, r in top.iterrows():
        print(f"  {r['Rank']:>5.0f} {r['Ticker']:>7s} {r['Raw_Prob']:>5.1%} {r['Cal_Prob']:>5.1%} "
              f"{r['Predictability']:>4.0f} {r['Sector_Rel_Strength']:>+5.1f}% "
              f"{r['Return_20d']:>+6.1f}% {r['Vol_Ratio']:>4.1f} {r['RSI']:>3.0f} "
              f"{r['Combined_Score']:>5.1f} {r['Sector']:>15s}")
    
    # Highlight best bets
    best = results[(results["Cal_Prob"] >= 0.4) & (results["Predictability"] >= 55) & (results["Return_20d"] > 0)]
    
    if len(best) > 0:
        print(f"\n  HIGHEST CONVICTION PICKS (high prob + predictable + positive momentum):")
        for _, r in best.head(5).iterrows():
            print(f"    #{r['Rank']:.0f} {r['Ticker']} — Cal Prob: {r['Cal_Prob']:.1%}, "
                  f"Predictability: {r['Predictability']:.0f}/100, "
                  f"Sector Strength: {r['Sector_Rel_Strength']:+.1f}%, "
                  f"Score: {r['Combined_Score']:.1f}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION: Run all 5 improvements
# ═══════════════════════════════════════════════════════════════════════════

def run_all_improvements(
    returns: pd.DataFrame,
    classifier_dict: dict,
    regime_data: pd.DataFrame = None,
    spy_data: pd.Series = None,
    ticker_sectors: dict = None,
    warmup_days: int = 252,
) -> dict:
    """Run all 5 improvements sequentially and return combined results."""
    
    print("\n" + "🔬" * 25)
    print("  RUNNING ALL 5 IMPROVEMENTS")
    print("🔬" * 25 + "\n")
    
    results = {}
    
    # 1. Probability Calibration
    results["calibration"] = calibrate_probabilities(
        returns, classifier_dict, warmup_days=warmup_days)
    
    # 2. Stock Predictability
    results["predictability"] = score_stock_predictability(
        results["calibration"])
    
    # 3. Ensemble Consensus
    results["ensemble"] = run_ensemble_backtest(
        returns, classifier_dict, 
        regime_data=regime_data,
        spy_data=spy_data,
        warmup_days=warmup_days)
    
    # 4. Forward Holdout
    results["holdout"] = forward_holdout_test(
        returns, classifier_dict,
        regime_data=regime_data,
        spy_data=spy_data,
        holdout_months=3,
        warmup_days=warmup_days)
    
    # 5. Combined Ranking (today's picks)
    results["ranked_picks"] = compute_sector_relative_scores(
        returns, classifier_dict,
        results["predictability"],
        results["calibration"],
        ticker_sectors=ticker_sectors,
        warmup_days=warmup_days)
    
    return results
