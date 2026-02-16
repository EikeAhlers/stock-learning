"""
Strategy Optimizer — Exhaustive search for the BEST trading strategy.

Tests 40+ parameter combinations × 6 strategy variants = 240+ configurations.
Each one uses HONEST walk-forward backtesting (no look-ahead bias).

Strategy Variants:
  1. BASE:       Pure probability ranking → buy top N
  2. REGIME:     Only trade in bull/normal markets (skip bear)
  3. MOMENTUM:   Only trade stocks with positive 20d momentum
  4. VOLUME:     Only trade stocks with Vol_Ratio > 1.5
  5. DIVERSIFIED: Max 1 stock per sector
  6. COMBO:      Regime + Momentum + Volume combined

Then finds:
  - Which strategy wins overall
  - Which strategy is most CONSISTENT (wins the most months)
  - Detailed breakdown vs SPY in every market condition
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


# ═══════════════════════════════════════════════════════════════════════════
#  CORE: Enhanced Walk-Forward Engine with Strategy Filters
# ═══════════════════════════════════════════════════════════════════════════

def _run_strategy(
    df: pd.DataFrame,
    clf_features: list,
    trade_dates: list,
    all_dates: list,
    warmup_days: int = 252,
    start_capital: float = 100_000,
    top_n: int = 3,
    hold_days: int = 3,
    min_probability: float = 0.5,
    stop_loss_pct: float = -7.0,
    max_positions: int = 3,
    retrain_every: int = 63,
    take_profit_pct: float = None,
    trailing_stop_pct: float = None,
    strategy_type: str = "base",
    regime_data: pd.DataFrame = None,
    ticker_sectors: dict = None,
    verbose: bool = False,
) -> dict:
    """
    Enhanced walk-forward backtest with multiple strategy types.
    
    strategy_type options:
      'base'       — pure probability ranking
      'regime'     — skip bear market days
      'momentum'   — only trade stocks with positive 20d return
      'volume'     — only trade stocks with high volume ratio
      'diversified'— max 1 per sector
      'combo'      — regime + momentum + volume filters combined
    """
    from sklearn.preprocessing import StandardScaler
    
    # Train initial model
    warmup_data = df[df.index <= all_dates[warmup_days - 1]]
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
    
    for today in trade_dates:
        # Periodic re-training
        if days_since_retrain >= retrain_every:
            past_data = df[df.index < today]
            new_model = _train_classifier_on_window(past_data, clf_features)
            if new_model is not None:
                current_model = new_model
                model = current_model["model"]
                scaler = current_model["scaler"]
                days_since_retrain = 0
        days_since_retrain += 1
        
        # ── Check regime filter ──────────────────────────────────────
        skip_today = False
        if strategy_type in ("regime", "combo") and regime_data is not None:
            if today in regime_data.index:
                regime_val = regime_data.loc[today]
                if isinstance(regime_val, pd.Series):
                    regime_str = str(regime_val.get("Market_Regime", ""))
                elif isinstance(regime_val, str):
                    regime_str = regime_val
                else:
                    regime_str = ""
                if "bear" in regime_str.lower():
                    skip_today = True
        
        # ── Close existing positions ─────────────────────────────────
        closed_today = []
        for pos in positions:
            ticker = pos["ticker"]
            today_data = df[(df.index == today) & (df["Ticker"] == ticker)]
            if len(today_data) == 0:
                pos["days_held"] += 1
                continue
            
            current_price = today_data["Close"].iloc[0]
            current_return = (current_price / pos["entry_price"] - 1) * 100
            pos["days_held"] += 1
            pos["current_return"] = current_return
            
            # Track peak for trailing stop
            if current_price > pos.get("peak_price", pos["entry_price"]):
                pos["peak_price"] = current_price
            
            should_exit = False
            exit_reason = ""
            
            # Stop loss
            if current_return <= stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
            # Take profit
            elif take_profit_pct is not None and current_return >= take_profit_pct:
                should_exit = True
                exit_reason = "take_profit"
            # Trailing stop
            elif trailing_stop_pct is not None and pos.get("peak_price"):
                trail_return = (current_price / pos["peak_price"] - 1) * 100
                if trail_return <= -trailing_stop_pct:
                    should_exit = True
                    exit_reason = "trailing_stop"
            # Hold period expired
            elif pos["days_held"] >= pos["hold_days"]:
                should_exit = True
                exit_reason = "target_hold"
            
            if should_exit:
                trade_pnl = pos["shares"] * pos["entry_price"] * (current_return / 100)
                capital += pos["shares"] * current_price
                trade_log.append({
                    "Ticker": ticker,
                    "Entry_Date": pos["entry_date"],
                    "Exit_Date": today,
                    "Entry_Price": pos["entry_price"],
                    "Exit_Price": current_price,
                    "Return_Pct": current_return,
                    "PnL": trade_pnl,
                    "Hold_Days": pos["days_held"],
                    "Exit_Reason": exit_reason,
                    "Probability": pos.get("probability", 0),
                })
                closed_today.append(pos)
        
        for pos in closed_today:
            positions.remove(pos)
        
        # ── Open new positions (if not skipping today) ───────────────
        open_slots = max_positions - len(positions)
        
        if open_slots > 0 and not skip_today:
            today_all = df[df.index == today].copy().reset_index(drop=True)
            
            if len(today_all) > 10:
                feat_mask = today_all[clf_features].notna().all(axis=1)
                today_features = today_all.loc[feat_mask, clf_features]
                
                if len(today_features) > 0:
                    X = scaler.transform(today_features)
                    probs = model.predict_proba(X)[:, 1]
                    
                    scores = pd.DataFrame({
                        "idx": today_all.loc[feat_mask].index,
                        "Ticker": today_all.loc[feat_mask, "Ticker"].values,
                        "Probability": probs,
                    })
                    
                    scores = scores[scores["Probability"] >= min_probability]
                    
                    # ── Apply strategy-specific filters ───────────────
                    if strategy_type in ("momentum", "combo"):
                        if "Prev_Return_20d" in today_all.columns:
                            mom_vals = today_all.loc[scores["idx"], "Prev_Return_20d"]
                            scores = scores[mom_vals.values > 0]
                    
                    if strategy_type in ("volume", "combo"):
                        if "Prev_Vol_Ratio" in today_all.columns:
                            vol_vals = today_all.loc[scores["idx"], "Prev_Vol_Ratio"]
                            scores = scores[vol_vals.values > 1.5]
                    
                    # Remove already held
                    held_tickers = [p["ticker"] for p in positions]
                    scores = scores[~scores["Ticker"].isin(held_tickers)]
                    scores = scores.sort_values("Probability", ascending=False)
                    
                    # ── Diversification filter ────────────────────────
                    if strategy_type == "diversified" and ticker_sectors:
                        seen_sectors = set()
                        for p in positions:
                            sec = ticker_sectors.get(p["ticker"], "Unknown")
                            seen_sectors.add(sec)
                        
                        diversified_picks = []
                        for _, row in scores.iterrows():
                            sec = ticker_sectors.get(row["Ticker"], "Unknown")
                            if sec not in seen_sectors:
                                diversified_picks.append(row)
                                seen_sectors.add(sec)
                            if len(diversified_picks) >= open_slots:
                                break
                        if diversified_picks:
                            scores = pd.DataFrame(diversified_picks)
                        else:
                            scores = scores.head(0)  # empty
                    
                    n_to_buy = min(open_slots, top_n, len(scores))
                    
                    for j in range(n_to_buy):
                        pick = scores.iloc[j]
                        ticker = pick["Ticker"]
                        prob = pick["Probability"]
                        
                        ticker_today = today_all[today_all["Ticker"] == ticker]
                        if len(ticker_today) == 0:
                            continue
                        
                        entry_price = ticker_today["Close"].iloc[0]
                        if entry_price <= 0:
                            continue
                        
                        position_value = min(capital, start_capital / max_positions)
                        shares = int(position_value / entry_price)
                        if shares <= 0:
                            continue
                        
                        capital -= shares * entry_price
                        positions.append({
                            "ticker": ticker,
                            "entry_date": today,
                            "entry_price": entry_price,
                            "peak_price": entry_price,
                            "shares": shares,
                            "hold_days": hold_days,
                            "days_held": 0,
                            "current_return": 0,
                            "probability": prob,
                        })
        
        # ── Portfolio value ──────────────────────────────────────────
        portfolio_value = capital
        for pos in positions:
            td = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            if len(td) > 0:
                portfolio_value += pos["shares"] * td["Close"].iloc[0]
            else:
                portfolio_value += pos["shares"] * pos["entry_price"]
        
        equity_curve.append({"Date": today, "Portfolio_Value": portfolio_value})
    
    # Close remaining
    for pos in positions:
        last = df[df["Ticker"] == pos["ticker"]]
        if len(last) > 0:
            ep = last.iloc[-1]["Close"]
            ret = (ep / pos["entry_price"] - 1) * 100
            capital += pos["shares"] * ep
            trade_log.append({
                "Ticker": pos["ticker"],
                "Entry_Date": pos["entry_date"],
                "Exit_Date": trade_dates[-1],
                "Entry_Price": pos["entry_price"],
                "Exit_Price": ep,
                "Return_Pct": ret,
                "PnL": pos["shares"] * pos["entry_price"] * (ret / 100),
                "Hold_Days": pos["days_held"],
                "Exit_Reason": "end_of_period",
                "Probability": pos.get("probability", 0),
            })
    
    equity_df = pd.DataFrame(equity_curve).set_index("Date") if equity_curve else pd.DataFrame()
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    
    return {
        "equity_curve": equity_df,
        "trades": trades_df,
        "start_capital": start_capital,
        "final_value": equity_df["Portfolio_Value"].iloc[-1] if len(equity_df) > 0 else start_capital,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  FULL GRID SEARCH: Test all combinations
# ═══════════════════════════════════════════════════════════════════════════

def run_full_optimization(
    returns: pd.DataFrame,
    classifier_dict: dict,
    spy_data: pd.Series = None,
    regime_data: pd.DataFrame = None,
    ticker_sectors: dict = None,
    warmup_days: int = 252,
) -> pd.DataFrame:
    """
    Test 40+ configurations × 6 strategy types.
    Returns DataFrame with results for every single combination.
    """
    print("=" * 80)
    print("  STRATEGY OPTIMIZER — FINDING THE BEST STRATEGY")
    print("  Honest walk-forward backtesting on every combination")
    print("=" * 80)
    
    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    clf_features = classifier_dict["features"]
    
    dates = sorted(df.index.unique())
    trade_dates = dates[warmup_days:]
    
    print(f"\n  Data: {len(dates)} trading days, {len(trade_dates)} in test period")
    print(f"  Test: {trade_dates[0].date()} → {trade_dates[-1].date()}")
    
    # ── Define parameter grid ────────────────────────────────────────
    param_grid = [
        # (hold_days, min_prob, stop_loss, top_n, max_pos, take_profit, trailing_stop)
        (1, 0.50, -5.0,  3, 3, None,  None),    # 1-day scalp
        (1, 0.60, -3.0,  3, 3, 5.0,   None),    # 1-day tight
        (2, 0.40, -5.0,  3, 3, None,  None),    # 2-day loose
        (2, 0.50, -5.0,  3, 3, None,  None),    # 2-day medium
        (2, 0.50, -7.0,  3, 3, None,  None),    # 2-day wider stop
        (2, 0.60, -5.0,  3, 3, 8.0,   None),    # 2-day w/ TP
        (3, 0.40, -5.0,  3, 3, None,  None),    # 3-day loose
        (3, 0.50, -5.0,  3, 3, None,  None),    # 3-day tight stop
        (3, 0.50, -7.0,  3, 3, None,  None),    # 3-day BASELINE
        (3, 0.50, -10.0, 3, 3, None,  None),    # 3-day wide stop
        (3, 0.60, -5.0,  3, 3, None,  None),    # 3-day selective
        (3, 0.60, -7.0,  3, 3, 10.0,  None),    # 3-day w/ TP
        (3, 0.70, -7.0,  3, 3, None,  None),    # 3-day very selective
        (3, 0.50, -7.0,  5, 5, None,  None),    # 3-day more positions
        (3, 0.50, -7.0,  1, 1, None,  None),    # 3-day concentrated
        (3, 0.50, -7.0,  2, 2, None,  None),    # 3-day 2 positions
        (5, 0.40, -7.0,  3, 3, None,  None),    # 5-day loose
        (5, 0.50, -5.0,  3, 3, None,  None),    # 5-day tight
        (5, 0.50, -7.0,  3, 3, None,  None),    # 5-day medium
        (5, 0.50, -10.0, 3, 3, None,  None),    # 5-day wide
        (5, 0.50, -7.0,  5, 5, None,  None),    # 5-day more positions
        (5, 0.60, -7.0,  3, 3, 15.0,  None),    # 5-day w/ TP
        (5, 0.50, -7.0,  3, 3, None,  5.0),     # 5-day trailing stop
        (7, 0.50, -7.0,  3, 3, None,  None),    # 7-day swing
        (7, 0.50, -10.0, 3, 3, None,  5.0),     # 7-day trailing
        (7, 0.40, -10.0, 5, 5, None,  None),    # 7-day wide loose
        (10, 0.50, -10.0, 3, 3, None, None),    # 10-day position
        (10, 0.50, -10.0, 3, 3, None, 7.0),     # 10-day trailing
        (10, 0.40, -15.0, 5, 5, None, None),    # 10-day wide
    ]
    
    strategy_types = ["base", "regime", "momentum", "volume", "diversified", "combo"]
    
    total_configs = len(param_grid) * len(strategy_types)
    print(f"  Testing {len(param_grid)} param configs × {len(strategy_types)} strategy types = {total_configs} backtests")
    print(f"  This will take a while...\n")
    
    results = []
    config_num = 0
    t_start = time.time()
    
    for strat in strategy_types:
        print(f"\n  ── Strategy: {strat.upper()} ──────────────────────────────────")
        
        for hd, mp, sl, tn, mpos, tp, ts in param_grid:
            config_num += 1
            label = f"{strat}|H{hd}d|P{mp:.0%}|SL{sl}%|T{tn}|M{mpos}"
            if tp: label += f"|TP{tp}%"
            if ts: label += f"|TS{ts}%"
            
            # Suppress output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                bt = _run_strategy(
                    df=df, clf_features=clf_features,
                    trade_dates=trade_dates, all_dates=dates,
                    warmup_days=warmup_days, start_capital=100_000,
                    top_n=tn, hold_days=hd, min_probability=mp,
                    stop_loss_pct=sl, max_positions=mpos,
                    retrain_every=63,
                    take_profit_pct=tp, trailing_stop_pct=ts,
                    strategy_type=strat,
                    regime_data=regime_data, ticker_sectors=ticker_sectors,
                    verbose=False,
                )
            except Exception as e:
                sys.stdout = old_stdout
                print(f"  [{config_num:3d}/{total_configs}] {label} → ERROR: {e}")
                continue
            finally:
                sys.stdout = old_stdout
            
            if not bt or "equity_curve" not in bt or len(bt.get("equity_curve", pd.DataFrame())) == 0:
                continue
            
            eq = bt["equity_curve"]
            trades_df = bt["trades"]
            final_val = bt["final_value"]
            total_return = (final_val / 100_000 - 1) * 100
            
            # Stats
            n_trades = len(trades_df)
            if n_trades > 0:
                win_rate = (trades_df["Return_Pct"] > 0).mean() * 100
                avg_return = trades_df["Return_Pct"].mean()
                wins = trades_df[trades_df["Return_Pct"] > 0]
                losses = trades_df[trades_df["Return_Pct"] <= 0]
                profit_factor = (wins["PnL"].sum() / abs(losses["PnL"].sum())) if len(losses) > 0 and losses["PnL"].sum() != 0 else 99.0
                avg_hold = trades_df["Hold_Days"].mean()
            else:
                win_rate = avg_return = profit_factor = avg_hold = 0
            
            # Drawdown
            rolling_max = eq["Portfolio_Value"].cummax()
            drawdown = (eq["Portfolio_Value"] - rolling_max) / rolling_max * 100
            max_dd = drawdown.min()
            
            # Sharpe
            daily_rets = eq["Portfolio_Value"].pct_change().dropna()
            sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if len(daily_rets) > 10 and daily_rets.std() > 0 else 0
            
            # Calmar ratio (return / max drawdown)
            calmar = total_return / abs(max_dd) if max_dd != 0 else 0
            
            # SPY comparison
            spy_ret = 0
            if spy_data is not None:
                spy_period = spy_data.loc[(spy_data.index >= eq.index[0]) & (spy_data.index <= eq.index[-1])]
                if len(spy_period) >= 2:
                    spy_ret = (spy_period.iloc[-1] / spy_period.iloc[0] - 1) * 100
                    if isinstance(spy_ret, pd.Series):
                        spy_ret = spy_ret.iloc[0]
            
            alpha = total_return - spy_ret
            
            # Monthly consistency
            monthly_vals = eq["Portfolio_Value"].resample("ME").last()
            monthly_rets = monthly_vals.pct_change().dropna() * 100
            positive_months = (monthly_rets > 0).sum()
            total_months = len(monthly_rets)
            monthly_win_rate = positive_months / total_months * 100 if total_months > 0 else 0
            
            # Monthly beat SPY
            if spy_data is not None and total_months > 0:
                spy_monthly = spy_data.resample("ME").last()
                spy_monthly_rets = spy_monthly.pct_change().dropna() * 100
                common_months = monthly_rets.index.intersection(spy_monthly_rets.index)
                if len(common_months) > 0:
                    beats_spy_months = (monthly_rets.loc[common_months] > spy_monthly_rets.loc[common_months]).sum()
                    beat_spy_pct = beats_spy_months / len(common_months) * 100
                else:
                    beat_spy_pct = 0
            else:
                beat_spy_pct = 0
            
            # Composite score: balances return, risk, consistency
            composite = (
                0.25 * min(total_return / 100, 2.0) +   # return (capped at 200%)
                0.20 * min(sharpe / 2, 1.0) +            # risk-adjusted
                0.15 * (1 + max_dd / 100) +              # drawdown penalty
                0.15 * (monthly_win_rate / 100) +        # consistency
                0.15 * (beat_spy_pct / 100) +            # beats benchmark
                0.10 * min(profit_factor / 3, 1.0)       # profitability
            ) * 100
            
            elapsed = time.time() - t_start
            eta = elapsed / config_num * (total_configs - config_num)
            
            status = "✓" if total_return > 0 else "✗"
            print(f"  [{config_num:3d}/{total_configs}] {status} {label:50s} "
                  f"Ret={total_return:+6.1f}% Sh={sharpe:.2f} DD={max_dd:5.1f}% "
                  f"MWR={monthly_win_rate:.0f}% Score={composite:.1f} "
                  f"[ETA {eta/60:.0f}m]")
            
            results.append({
                "Strategy": strat,
                "Hold_Days": hd, "Min_Prob": mp, "Stop_Loss": sl,
                "Top_N": tn, "Max_Pos": mpos,
                "Take_Profit": tp, "Trailing_Stop": ts,
                "Return_Pct": total_return,
                "Alpha_Pct": alpha,
                "Sharpe": sharpe,
                "Calmar": calmar,
                "Max_DD_Pct": max_dd,
                "Trades": n_trades,
                "Win_Rate": win_rate,
                "Avg_Return": avg_return,
                "Profit_Factor": profit_factor,
                "Avg_Hold": avg_hold,
                "Monthly_Win_Rate": monthly_win_rate,
                "Beat_SPY_Months_Pct": beat_spy_pct,
                "Positive_Months": positive_months,
                "Total_Months": total_months,
                "Composite_Score": composite,
                "Label": label,
                "SPY_Return": spy_ret,
                "Final_Value": final_val,
                "equity_curve": eq,          # keep for deeper analysis
                "trades_df": trades_df,      # keep for deeper analysis
            })
    
    total_time = time.time() - t_start
    print(f"\n  Completed {config_num} backtests in {total_time/60:.1f} minutes")
    
    results_df = pd.DataFrame(results)
    return results_df


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS: Find the best strategy
# ═══════════════════════════════════════════════════════════════════════════

def analyze_optimization_results(results_df: pd.DataFrame) -> dict:
    """
    Deep analysis of all tested strategies.
    Returns the best strategy by multiple criteria.
    """
    if len(results_df) == 0:
        return {}
    
    # Clean up — remove equity/trades columns for display
    display_cols = [c for c in results_df.columns if c not in ("equity_curve", "trades_df")]
    df = results_df[display_cols].copy()
    
    # Only consider strategies with enough trades
    valid = df[df["Trades"] >= 20].copy()
    if len(valid) == 0:
        valid = df[df["Trades"] >= 5].copy()
    
    print("\n" + "=" * 80)
    print("  OPTIMIZATION RESULTS — STRATEGY COMPARISON")
    print("=" * 80)
    
    # ── Overview ─────────────────────────────────────────────────────
    profitable = (valid["Return_Pct"] > 0).sum()
    beats_spy = (valid["Alpha_Pct"] > 0).sum()
    print(f"\n  Total configs tested:    {len(df)}")
    print(f"  With enough trades:      {len(valid)}")
    print(f"  Profitable:              {profitable} ({profitable/len(valid)*100:.0f}%)")
    print(f"  Beat SPY:                {beats_spy} ({beats_spy/len(valid)*100:.0f}%)")
    
    # ── Best by each metric ──────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  BEST BY EACH METRIC")
    print(f"  {'─'*70}")
    
    metrics = {
        "Highest Return": ("Return_Pct", True),
        "Best Sharpe": ("Sharpe", True),
        "Lowest Drawdown": ("Max_DD_Pct", False),  # higher (less negative) is better
        "Best Win Rate": ("Win_Rate", True),
        "Most Consistent (Monthly)": ("Monthly_Win_Rate", True),
        "Best Composite Score": ("Composite_Score", True),
        "Best Profit Factor": ("Profit_Factor", True),
        "Most Months Beating SPY": ("Beat_SPY_Months_Pct", True),
    }
    
    best_by = {}
    for name, (col, ascending) in metrics.items():
        if ascending:
            idx = valid[col].idxmax()
        else:
            idx = valid[col].idxmax()  # max of max_dd (least negative)
        row = valid.loc[idx]
        best_by[name] = row
        print(f"\n  {name}:")
        print(f"    Strategy: {row['Label']}")
        print(f"    Return: {row['Return_Pct']:+.1f}% | Sharpe: {row['Sharpe']:.2f} | "
              f"DD: {row['Max_DD_Pct']:.1f}% | WR: {row['Win_Rate']:.0f}% | "
              f"Monthly WR: {row['Monthly_Win_Rate']:.0f}% | Score: {row['Composite_Score']:.1f}")
    
    # ── Strategy type comparison ─────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  STRATEGY TYPE AVERAGES")
    print(f"  {'─'*70}")
    
    type_avg = valid.groupby("Strategy").agg({
        "Return_Pct": "mean",
        "Sharpe": "mean",
        "Max_DD_Pct": "mean",
        "Win_Rate": "mean",
        "Monthly_Win_Rate": "mean",
        "Composite_Score": "mean",
        "Trades": "mean",
    }).round(2)
    
    type_avg = type_avg.sort_values("Composite_Score", ascending=False)
    
    for strat, row in type_avg.iterrows():
        print(f"  {strat:12s}: Ret={row['Return_Pct']:+6.1f}% Sh={row['Sharpe']:.2f} "
              f"DD={row['Max_DD_Pct']:5.1f}% WR={row['Win_Rate']:.0f}% "
              f"MWR={row['Monthly_Win_Rate']:.0f}% Score={row['Composite_Score']:.1f} "
              f"Trades={row['Trades']:.0f}")
    
    # ── Top 10 overall ───────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  TOP 10 STRATEGIES (by Composite Score)")
    print(f"  {'─'*70}")
    
    top10 = valid.nlargest(10, "Composite_Score")
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        marker = "👑" if rank == 1 else f"#{rank}"
        print(f"  {marker:3s} {row['Label']:50s}")
        print(f"       Ret={row['Return_Pct']:+6.1f}% Alpha={row['Alpha_Pct']:+5.1f}% "
              f"Sh={row['Sharpe']:.2f} DD={row['Max_DD_Pct']:5.1f}% "
              f"WR={row['Win_Rate']:.0f}% PF={row['Profit_Factor']:.2f} "
              f"MWR={row['Monthly_Win_Rate']:.0f}% Beat SPY={row['Beat_SPY_Months_Pct']:.0f}% "
              f"Score={row['Composite_Score']:.1f}")
    
    # ── Hold period analysis ─────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  HOLD PERIOD ANALYSIS")
    print(f"  {'─'*70}")
    
    hold_avg = valid.groupby("Hold_Days").agg({
        "Return_Pct": ["mean", "std", "max"],
        "Sharpe": "mean",
        "Max_DD_Pct": "mean",
        "Monthly_Win_Rate": "mean",
    }).round(2)
    
    for hd in sorted(valid["Hold_Days"].unique()):
        subset = valid[valid["Hold_Days"] == hd]
        print(f"  {hd:2d}-day hold: Avg Ret={subset['Return_Pct'].mean():+6.1f}% "
              f"(±{subset['Return_Pct'].std():.1f}%) "
              f"Best={subset['Return_Pct'].max():+.1f}% "
              f"Avg Sharpe={subset['Sharpe'].mean():.2f} "
              f"Avg MWR={subset['Monthly_Win_Rate'].mean():.0f}%")
    
    # ── Winner ───────────────────────────────────────────────────────
    best_idx = valid["Composite_Score"].idxmax()
    best = valid.loc[best_idx]
    
    print(f"\n  {'='*70}")
    print(f"  THE WINNER")
    print(f"  {'='*70}")
    print(f"  {best['Label']}")
    print(f"  Return:          {best['Return_Pct']:+.1f}%")
    print(f"  Alpha vs SPY:    {best['Alpha_Pct']:+.1f}%")
    print(f"  Sharpe Ratio:    {best['Sharpe']:.2f}")
    print(f"  Max Drawdown:    {best['Max_DD_Pct']:.1f}%")
    print(f"  Win Rate:        {best['Win_Rate']:.0f}%")
    print(f"  Profit Factor:   {best['Profit_Factor']:.2f}")
    print(f"  Monthly WR:      {best['Monthly_Win_Rate']:.0f}%")
    print(f"  Beat SPY:        {best['Beat_SPY_Months_Pct']:.0f}% of months")
    print(f"  Composite Score: {best['Composite_Score']:.1f}")
    print(f"  Trades:          {best['Trades']:.0f}")
    print(f"  {'='*70}")
    
    return {
        "best_overall": best,
        "best_by_metric": best_by,
        "type_averages": type_avg,
        "top10": top10,
        "all_valid": valid,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  DEEP VERIFICATION: Stress test the winner
# ═══════════════════════════════════════════════════════════════════════════

def deep_verify_winner(
    winner_row: pd.Series,
    results_df: pd.DataFrame,
    returns: pd.DataFrame,
    classifier_dict: dict,
    spy_data: pd.Series = None,
    regime_data: pd.DataFrame = None,
    ticker_sectors: dict = None,
) -> dict:
    """
    Run extra verification on the winning strategy:
    1. Rolling 3-month window analysis
    2. Year-over-year breakdown
    3. Per-sector performance
    4. Drawdown analysis
    5. Trade quality distribution
    6. Win/loss streak analysis
    """
    print("\n" + "=" * 80)
    print(f"  DEEP VERIFICATION OF WINNING STRATEGY")
    print(f"  {winner_row['Label']}")
    print("=" * 80)
    
    # Find the full result with equity curve
    match = results_df[results_df["Label"] == winner_row["Label"]]
    if len(match) == 0:
        print("  ERROR: Could not find equity curve for winner")
        return {}
    
    eq = match.iloc[0]["equity_curve"]
    trades = match.iloc[0]["trades_df"]
    
    verification = {}
    
    # ── 1. Rolling 3-month performance ───────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  1. ROLLING 3-MONTH WINDOWS")
    print(f"  {'─'*60}")
    
    monthly_vals = eq["Portfolio_Value"].resample("ME").last()
    rolling_returns = []
    for i in range(3, len(monthly_vals)):
        start_val = monthly_vals.iloc[i-3]
        end_val = monthly_vals.iloc[i]
        ret_3m = (end_val / start_val - 1) * 100
        rolling_returns.append({
            "End_Month": monthly_vals.index[i],
            "Return_3M": ret_3m,
        })
    
    if rolling_returns:
        roll_df = pd.DataFrame(rolling_returns)
        positive_windows = (roll_df["Return_3M"] > 0).sum()
        total_windows = len(roll_df)
        verification["rolling_3m_win_rate"] = positive_windows / total_windows * 100
        verification["rolling_3m_avg"] = roll_df["Return_3M"].mean()
        verification["rolling_3m_worst"] = roll_df["Return_3M"].min()
        verification["rolling_3m_best"] = roll_df["Return_3M"].max()
        
        print(f"  Windows tested:    {total_windows}")
        print(f"  Positive windows:  {positive_windows} ({verification['rolling_3m_win_rate']:.0f}%)")
        print(f"  Avg 3-month ret:   {verification['rolling_3m_avg']:+.1f}%")
        print(f"  Best 3 months:     {verification['rolling_3m_best']:+.1f}%")
        print(f"  Worst 3 months:    {verification['rolling_3m_worst']:+.1f}%")
    
    # ── 2. Per-quarter breakdown ─────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  2. QUARTERLY BREAKDOWN")
    print(f"  {'─'*60}")
    
    quarterly_vals = eq["Portfolio_Value"].resample("QE").last()
    quarterly_rets = quarterly_vals.pct_change().dropna() * 100
    
    for date, ret in quarterly_rets.items():
        marker = "+" if ret > 0 else " "
        bar = "█" * int(abs(ret) / 2)
        q_label = f"{date.year} Q{date.quarter}" if hasattr(date, 'quarter') else str(date)[:7]
        print(f"  {q_label:>10s}: {marker}{ret:+6.1f}% {'|' + bar}")
    
    verification["quarterly_win_rate"] = (quarterly_rets > 0).sum() / len(quarterly_rets) * 100 if len(quarterly_rets) > 0 else 0
    
    # ── 3. Trade quality ─────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  3. TRADE QUALITY DISTRIBUTION")
    print(f"  {'─'*60}")
    
    if len(trades) > 0:
        bins = [-100, -10, -5, -2, 0, 2, 5, 10, 20, 50, 200]
        labels = ["<-10%", "-10 to -5%", "-5 to -2%", "-2 to 0%", 
                  "0 to 2%", "2 to 5%", "5 to 10%", "10 to 20%", "20 to 50%", ">50%"]
        trades["Bucket"] = pd.cut(trades["Return_Pct"], bins=bins, labels=labels)
        bucket_counts = trades["Bucket"].value_counts().sort_index()
        
        total = len(trades)
        for bucket, count in bucket_counts.items():
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {str(bucket):>14s}: {count:4d} ({pct:4.1f}%) {bar}")
        
        # Big winner ratio
        big_winners = (trades["Return_Pct"] >= 10).sum()
        big_losers = (trades["Return_Pct"] <= -5).sum()
        verification["big_winner_count"] = big_winners
        verification["big_loser_count"] = big_losers
        verification["big_winner_ratio"] = big_winners / max(big_losers, 1)
        
        print(f"\n  Big winners (>=10%): {big_winners}")
        print(f"  Big losers (<=-5%):  {big_losers}")
        print(f"  Ratio:               {verification['big_winner_ratio']:.2f}x")
    
    # ── 4. Win/Loss streaks ──────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  4. WIN/LOSS STREAKS")
    print(f"  {'─'*60}")
    
    if len(trades) > 0:
        wins_losses = (trades["Return_Pct"] > 0).astype(int).values
        
        max_win_streak = max_loss_streak = 0
        current_streak = 0
        for i, wl in enumerate(wins_losses):
            if i == 0:
                current_streak = 1
            elif wl == wins_losses[i-1]:
                current_streak += 1
            else:
                if wins_losses[i-1] == 1:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
                current_streak = 1
        # Final streak
        if len(wins_losses) > 0:
            if wins_losses[-1] == 1:
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)
        
        verification["max_win_streak"] = max_win_streak
        verification["max_loss_streak"] = max_loss_streak
        
        print(f"  Longest win streak:  {max_win_streak} trades")
        print(f"  Longest loss streak: {max_loss_streak} trades")
    
    # ── 5. Drawdown analysis ─────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  5. DRAWDOWN DEEP DIVE")
    print(f"  {'─'*60}")
    
    rolling_max = eq["Portfolio_Value"].cummax()
    drawdown = (eq["Portfolio_Value"] - rolling_max) / rolling_max * 100
    
    # Find drawdown periods
    in_dd = drawdown < -1  # at least 1% drawdown
    dd_periods = []
    dd_start = None
    
    for date, is_dd in in_dd.items():
        if is_dd and dd_start is None:
            dd_start = date
        elif not is_dd and dd_start is not None:
            dd_depth = drawdown.loc[dd_start:date].min()
            dd_length = len(drawdown.loc[dd_start:date])
            dd_periods.append({
                "Start": dd_start,
                "End": date,
                "Depth": dd_depth,
                "Length_Days": dd_length,
            })
            dd_start = None
    
    if dd_start is not None:
        dd_depth = drawdown.loc[dd_start:].min()
        dd_length = len(drawdown.loc[dd_start:])
        dd_periods.append({
            "Start": dd_start,
            "End": drawdown.index[-1],
            "Depth": dd_depth,
            "Length_Days": dd_length,
        })
    
    if dd_periods:
        dd_df = pd.DataFrame(dd_periods).sort_values("Depth")
        print(f"  Total drawdown periods (>1%): {len(dd_df)}")
        print(f"\n  Top 5 deepest drawdowns:")
        for _, dd in dd_df.head(5).iterrows():
            print(f"    {dd['Start'].date()} → {dd['End'].date()}: "
                  f"{dd['Depth']:.1f}% over {dd['Length_Days']} days")
        
        verification["n_drawdowns"] = len(dd_df)
        verification["avg_dd_depth"] = dd_df["Depth"].mean()
        verification["avg_dd_length"] = dd_df["Length_Days"].mean()
    
    # ── 6. Per-sector performance ────────────────────────────────────
    if ticker_sectors and len(trades) > 0:
        print(f"\n  {'─'*60}")
        print(f"  6. SECTOR BREAKDOWN")
        print(f"  {'─'*60}")
        
        trades["Sector"] = trades["Ticker"].map(ticker_sectors).fillna("Unknown")
        sector_stats = trades.groupby("Sector").agg({
            "Return_Pct": ["mean", "count"],
            "PnL": "sum",
        })
        sector_stats.columns = ["Avg_Return", "Trades", "Total_PnL"]
        sector_stats = sector_stats.sort_values("Total_PnL", ascending=False)
        
        for sec, row in sector_stats.iterrows():
            marker = "+" if row["Total_PnL"] > 0 else " "
            print(f"  {sec:30s}: {row['Trades']:3.0f} trades, "
                  f"Avg={row['Avg_Return']:+5.1f}%, "
                  f"PnL=${row['Total_PnL']:+,.0f}")
        
        verification["sector_stats"] = sector_stats
    
    # ── Final Verdict ────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  VERIFICATION VERDICT")
    print(f"  {'='*60}")
    
    checks = 0
    total_checks = 6
    
    # Check 1: Rolling 3-month positive > 60%
    if verification.get("rolling_3m_win_rate", 0) >= 60:
        checks += 1
        print(f"  ✓ Rolling 3-month win rate: {verification['rolling_3m_win_rate']:.0f}% (>=60%)")
    else:
        print(f"  ✗ Rolling 3-month win rate: {verification.get('rolling_3m_win_rate', 0):.0f}% (<60%)")
    
    # Check 2: Max loss streak < 10
    if verification.get("max_loss_streak", 99) < 10:
        checks += 1
        print(f"  ✓ Max loss streak: {verification.get('max_loss_streak', 0)} (<10)")
    else:
        print(f"  ✗ Max loss streak: {verification.get('max_loss_streak', 0)} (>=10)")
    
    # Check 3: Big winner ratio > 0.5
    if verification.get("big_winner_ratio", 0) >= 0.5:
        checks += 1
        print(f"  ✓ Big winner/loser ratio: {verification.get('big_winner_ratio', 0):.2f} (>=0.5)")
    else:
        print(f"  ✗ Big winner/loser ratio: {verification.get('big_winner_ratio', 0):.2f} (<0.5)")
    
    # Check 4: Max DD less than -50%
    max_dd = winner_row.get("Max_DD_Pct", -100)
    if max_dd > -50:
        checks += 1
        print(f"  ✓ Max drawdown: {max_dd:.1f}% (>-50%)")
    else:
        print(f"  ✗ Max drawdown: {max_dd:.1f}% (<=-50%)")
    
    # Check 5: Profit factor > 1.2
    pf = winner_row.get("Profit_Factor", 0)
    if pf > 1.2:
        checks += 1
        print(f"  ✓ Profit factor: {pf:.2f} (>1.2)")
    else:
        print(f"  ✗ Profit factor: {pf:.2f} (<=1.2)")
    
    # Check 6: Sharpe > 0.5
    sharpe = winner_row.get("Sharpe", 0)
    if sharpe > 0.5:
        checks += 1
        print(f"  ✓ Sharpe ratio: {sharpe:.2f} (>0.5)")
    else:
        print(f"  ✗ Sharpe ratio: {sharpe:.2f} (<=0.5)")
    
    stars = "★" * checks + "☆" * (total_checks - checks)
    grade = ["F", "D", "C", "B", "B+", "A", "A+"][min(checks, 6)]
    
    print(f"\n  Score: {checks}/{total_checks} {stars}")
    print(f"  Grade: {grade}")
    print(f"  {'='*60}\n")
    
    verification["checks_passed"] = checks
    verification["total_checks"] = total_checks
    verification["grade"] = grade
    
    return verification


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION helpers
# ═══════════════════════════════════════════════════════════════════════════

def plot_optimization_results(results_df, top_n=10):
    """Create comprehensive visualization of strategy comparison."""
    import matplotlib.pyplot as plt
    
    display_cols = [c for c in results_df.columns if c not in ("equity_curve", "trades_df")]
    df = results_df[display_cols].copy()
    valid = df[df["Trades"] >= 20].copy()
    if len(valid) == 0:
        valid = df[df["Trades"] >= 5].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Strategy Optimization Results", fontsize=16, fontweight="bold")
    
    # 1. Return distribution by strategy type
    ax = axes[0, 0]
    for strat in valid["Strategy"].unique():
        subset = valid[valid["Strategy"] == strat]
        ax.scatter(subset["Sharpe"], subset["Return_Pct"], label=strat, alpha=0.7, s=60)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Total Return %")
    ax.set_title("Return vs Sharpe by Strategy Type")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Return vs Drawdown
    ax = axes[0, 1]
    scatter = ax.scatter(valid["Max_DD_Pct"], valid["Return_Pct"], 
                         c=valid["Composite_Score"], cmap="RdYlGn", s=60, alpha=0.7)
    ax.set_xlabel("Max Drawdown %")
    ax.set_ylabel("Total Return %")
    ax.set_title("Return vs Risk (color=Score)")
    plt.colorbar(scatter, ax=ax, label="Composite Score")
    ax.grid(True, alpha=0.3)
    
    # 3. Monthly win rate vs Return
    ax = axes[0, 2]
    ax.scatter(valid["Monthly_Win_Rate"], valid["Return_Pct"],
               c=valid["Strategy"].astype("category").cat.codes, cmap="tab10", s=60, alpha=0.7)
    ax.set_xlabel("Monthly Win Rate %")
    ax.set_ylabel("Total Return %")
    ax.set_title("Consistency vs Return")
    ax.axvline(x=50, color="red", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # 4. Top 10 strategies bar chart
    ax = axes[1, 0]
    top10 = valid.nlargest(top_n, "Composite_Score")
    short_labels = [f"{r['Strategy'][:4]}|H{r['Hold_Days']}|P{r['Min_Prob']:.0%}" 
                    for _, r in top10.iterrows()]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10)))
    bars = ax.barh(range(len(top10)), top10["Composite_Score"], color=colors)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Composite Score")
    ax.set_title(f"Top {top_n} Strategies")
    ax.invert_yaxis()
    
    # 5. Hold period boxplot
    ax = axes[1, 1]
    hold_groups = []
    hold_labels = []
    for hd in sorted(valid["Hold_Days"].unique()):
        data = valid[valid["Hold_Days"] == hd]["Return_Pct"].values
        if len(data) > 0:
            hold_groups.append(data)
            hold_labels.append(f"{hd}d")
    if hold_groups:
        bp = ax.boxplot(hold_groups, labels=hold_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], plt.cm.Blues(np.linspace(0.3, 0.9, len(hold_groups)))):
            patch.set_facecolor(color)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Hold Period")
    ax.set_ylabel("Return %")
    ax.set_title("Return by Hold Period")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Strategy type comparison
    ax = axes[1, 2]
    type_avg = valid.groupby("Strategy")["Composite_Score"].mean().sort_values(ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(type_avg)))
    ax.barh(type_avg.index, type_avg.values, color=colors)
    ax.set_xlabel("Avg Composite Score")
    ax.set_title("Strategy Type Comparison")
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_top_equity_curves(results_df, spy_data=None, top_n=5):
    """Plot equity curves of top strategies overlaid with SPY."""
    import matplotlib.pyplot as plt
    
    display_cols = [c for c in results_df.columns if c not in ("equity_curve", "trades_df")]
    df = results_df[display_cols].copy()
    valid_idx = df[df["Trades"] >= 20].nlargest(top_n, "Composite_Score").index
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    for i, idx in enumerate(valid_idx):
        eq = results_df.loc[idx, "equity_curve"]
        label = results_df.loc[idx, "Label"]
        score = results_df.loc[idx, "Composite_Score"]
        ret = results_df.loc[idx, "Return_Pct"]
        
        normalized = eq["Portfolio_Value"] / eq["Portfolio_Value"].iloc[0] * 100
        short_label = f"#{i+1} {label.split('|')[0]}|{label.split('|')[1]} ({ret:+.0f}%)"
        ax.plot(normalized.index, normalized.values, color=colors[i], 
                linewidth=2 if i == 0 else 1.2, label=short_label, alpha=0.9 if i == 0 else 0.7)
    
    # SPY baseline
    if spy_data is not None:
        eq_first = results_df.loc[valid_idx[0], "equity_curve"]
        spy_period = spy_data.loc[(spy_data.index >= eq_first.index[0]) & (spy_data.index <= eq_first.index[-1])]
        if len(spy_period) > 0:
            spy_norm = spy_period / spy_period.iloc[0] * 100
            ax.plot(spy_norm.index, spy_norm.values, color="gray", linewidth=2, 
                    linestyle="--", label=f"SPY ({((spy_period.iloc[-1]/spy_period.iloc[0]-1)*100):+.0f}%)", alpha=0.8)
    
    ax.axhline(y=100, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (normalized to 100)")
    ax.set_title(f"Top {top_n} Strategies vs SPY", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig
