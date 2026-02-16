"""
Backtester — HONEST walk-forward simulation with NO look-ahead bias.

CRITICAL FIX: The previous version used a model trained on ALL data
(including the test period). That's cheating. The +2730% was fake.

This version does TRUE walk-forward:
  1. Train model ONLY on data before the current date
  2. Re-train periodically as new data arrives
  3. Score stocks and trade
  4. No information from the future ever leaks into predictions

This is the only honest way to evaluate a trading strategy.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from src.deep_analyzer import add_all_lagged_features, ALL_RAW_FEATURES
from src.setup_detector import SETUP_NAMES


def _train_classifier_on_window(df_train, features, big_move_threshold=5.0):
    """Train a fresh classifier on only past data — NO future leakage."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

    df_train = df_train.copy()
    df_train["Is_Big_Mover"] = (df_train["Daily_Return_Pct"] >= big_move_threshold).astype(int)
    
    model_df = df_train[features + ["Is_Big_Mover"]].dropna()
    if len(model_df) < 500:
        return None
    
    X = model_df[features]
    y = model_df["Is_Big_Mover"]
    
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos = neg_count / max(pos_count, 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=42,
        verbosity=0, eval_metric="logloss",
    )
    
    clf.fit(X_scaled, y)
    return {"model": clf, "scaler": scaler, "features": features}


def run_backtest(
    returns: pd.DataFrame,
    classifier_dict: dict,
    magnitude_models: dict,
    start_capital: float = 100_000,
    top_n: int = 3,
    hold_days: int = 3,
    min_probability: float = 0.5,
    stop_loss_pct: float = -7.0,
    max_positions: int = 3,
    retrain_every: int = 63,
    warmup_days: int = 252,
    honest_mode: bool = True,
) -> dict:
    """
    HONEST walk-forward backtesting engine.
    
    When honest_mode=True:
      - Retrains the classifier every `retrain_every` days using ONLY past data
      - Uses classifier probability only for ranking (no magnitude model in backtest)
      - Position sizing is fixed (no compounding bias)
    """
    print("=" * 80)
    label = "(HONEST — no look-ahead)" if honest_mode else "(WARNING: pre-trained model)"
    print(f"  WALK-FORWARD BACKTEST {label}")
    print("=" * 80)
    
    df = add_all_lagged_features(returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= 5.0).astype(int)
    
    clf_features = classifier_dict["features"]
    dates = sorted(df.index.unique())
    
    if len(dates) < warmup_days + 30:
        print(f"Not enough dates ({len(dates)}). Need {warmup_days + 30}.")
        return {}
    
    trade_dates = dates[warmup_days:]
    
    print(f"  Training warmup:  {dates[0].date()} to {dates[warmup_days-1].date()} ({warmup_days} days)")
    print(f"  Trading period:   {trade_dates[0].date()} to {trade_dates[-1].date()} ({len(trade_dates)} days)")
    print(f"  Capital: ${start_capital:,.0f}")
    print(f"  Strategy: Top {top_n}, hold {hold_days}d, stop {stop_loss_pct}%, min P={min_probability:.0%}")
    if honest_mode:
        print(f"  Retrain every {retrain_every} days on past-only data")
    print()
    
    if honest_mode:
        print("  Training initial model on warmup period only...")
        warmup_data = df[df.index <= dates[warmup_days - 1]]
        current_model = _train_classifier_on_window(warmup_data, clf_features)
        if current_model is None:
            print("  ERROR: Not enough warmup data")
            return {}
        days_since_retrain = 0
    else:
        current_model = classifier_dict
    
    model = current_model["model"]
    scaler = current_model["scaler"]
    
    capital = start_capital
    positions = []
    trade_log = []
    equity_curve = []
    dates_processed = 0
    
    for i, today in enumerate(trade_dates):
        # ── Periodic re-training (honest mode only) ──────────────────
        if honest_mode and days_since_retrain >= retrain_every:
            past_data = df[df.index < today]
            new_model = _train_classifier_on_window(past_data, clf_features)
            if new_model is not None:
                current_model = new_model
                model = current_model["model"]
                scaler = current_model["scaler"]
                days_since_retrain = 0
        
        if honest_mode:
            days_since_retrain += 1
        
        # ── 1. Check existing positions ──────────────────────────────
        closed_today = []
        for pos in positions:
            ticker = pos["ticker"]
            entry_price = pos["entry_price"]
            
            today_data = df[(df.index == today) & (df["Ticker"] == ticker)]
            if len(today_data) == 0:
                pos["days_held"] += 1
                continue
            
            current_price = today_data["Close"].iloc[0]
            current_return = (current_price / entry_price - 1) * 100
            pos["days_held"] += 1
            pos["current_return"] = current_return
            
            should_exit = False
            exit_reason = ""
            
            if current_return <= stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
            elif pos["days_held"] >= pos["hold_days"]:
                should_exit = True
                exit_reason = "target_hold"
            
            if should_exit:
                trade_pnl = pos["shares"] * entry_price * (current_return / 100)
                capital += pos["shares"] * current_price
                trade_log.append({
                    "Ticker": ticker,
                    "Entry_Date": pos["entry_date"],
                    "Exit_Date": today,
                    "Entry_Price": entry_price,
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
        
        # ── 2. Score stocks and open new positions ───────────────────
        open_slots = max_positions - len(positions)
        
        if open_slots > 0:
            today_all = df[df.index == today].copy().reset_index(drop=True)
            
            if len(today_all) > 10:
                feat_mask = today_all[clf_features].notna().all(axis=1)
                today_features = today_all.loc[feat_mask, clf_features]
                
                if len(today_features) > 0:
                    X = scaler.transform(today_features)
                    probs = model.predict_proba(X)[:, 1]
                    
                    scores = pd.DataFrame({
                        "Ticker": today_all.loc[feat_mask, "Ticker"].values,
                        "Probability": probs,
                    })
                    
                    scores = scores[scores["Probability"] >= min_probability]
                    
                    held_tickers = [p["ticker"] for p in positions]
                    scores = scores[~scores["Ticker"].isin(held_tickers)]
                    scores = scores.sort_values("Probability", ascending=False)
                    
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
                        
                        # FIXED position sizing — no compounding
                        position_value = min(capital, start_capital / max_positions)
                        shares = int(position_value / entry_price)
                        
                        if shares <= 0:
                            continue
                        
                        capital -= shares * entry_price
                        
                        positions.append({
                            "ticker": ticker,
                            "entry_date": today,
                            "entry_price": entry_price,
                            "shares": shares,
                            "hold_days": hold_days,
                            "days_held": 0,
                            "current_return": 0,
                            "probability": prob,
                        })
        
        # ── 3. Calculate daily portfolio value ───────────────────────
        portfolio_value = capital
        for pos in positions:
            ticker_today = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
            if len(ticker_today) > 0:
                portfolio_value += pos["shares"] * ticker_today["Close"].iloc[0]
            else:
                portfolio_value += pos["shares"] * pos["entry_price"]
        
        equity_curve.append({
            "Date": today,
            "Portfolio_Value": portfolio_value,
            "Capital": capital,
            "Positions": len(positions),
            "Total_Trades": len(trade_log),
        })
        
        dates_processed += 1
        if dates_processed % 50 == 0:
            pct_return = (portfolio_value / start_capital - 1) * 100
            print(f"  Day {dates_processed:4d} | {today.date()} | "
                  f"${portfolio_value:>10,.0f} ({pct_return:+.1f}%) | "
                  f"Trades: {len(trade_log)}")
    
    # Close remaining positions
    for pos in positions:
        last_data = df[df["Ticker"] == pos["ticker"]]
        if len(last_data) > 0:
            exit_price = last_data.iloc[-1]["Close"]
            current_return = (exit_price / pos["entry_price"] - 1) * 100
            trade_pnl = pos["shares"] * pos["entry_price"] * (current_return / 100)
            capital += pos["shares"] * exit_price
            trade_log.append({
                "Ticker": pos["ticker"],
                "Entry_Date": pos["entry_date"],
                "Exit_Date": trade_dates[-1],
                "Entry_Price": pos["entry_price"],
                "Exit_Price": exit_price,
                "Return_Pct": current_return,
                "PnL": trade_pnl,
                "Hold_Days": pos["days_held"],
                "Exit_Reason": "end_of_period",
                "Probability": pos.get("probability", 0),
            })
    
    equity_df = pd.DataFrame(equity_curve).set_index("Date")
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    
    return {
        "equity_curve": equity_df,
        "trades": trades_df,
        "start_capital": start_capital,
        "final_value": equity_df["Portfolio_Value"].iloc[-1] if len(equity_df) > 0 else start_capital,
    }


def analyze_backtest(result: dict, spy_data: pd.DataFrame = None) -> dict:
    """Compute comprehensive backtest statistics."""
    equity = result["equity_curve"]
    trades = result["trades"]
    start_cap = result["start_capital"]
    final_val = result["final_value"]
    total_return = (final_val / start_cap - 1) * 100
    
    stats = {"total_return_pct": total_return, "final_value": final_val}
    
    if len(trades) > 0:
        wins = trades[trades["Return_Pct"] > 0]
        losses = trades[trades["Return_Pct"] <= 0]
        
        stats["total_trades"] = len(trades)
        stats["win_rate"] = len(wins) / len(trades) * 100
        stats["avg_win_pct"] = wins["Return_Pct"].mean() if len(wins) > 0 else 0
        stats["avg_loss_pct"] = losses["Return_Pct"].mean() if len(losses) > 0 else 0
        stats["avg_return_pct"] = trades["Return_Pct"].mean()
        stats["median_return_pct"] = trades["Return_Pct"].median()
        stats["best_trade_pct"] = trades["Return_Pct"].max()
        stats["worst_trade_pct"] = trades["Return_Pct"].min()
        stats["total_pnl"] = trades["PnL"].sum()
        stats["avg_hold_days"] = trades["Hold_Days"].mean()
        stats["profit_factor"] = (
            wins["PnL"].sum() / abs(losses["PnL"].sum())
            if len(losses) > 0 and losses["PnL"].sum() != 0 else float("inf")
        )
        stats["top_winners"] = trades.nlargest(5, "Return_Pct")[
            ["Ticker", "Entry_Date", "Return_Pct", "PnL"]
        ].to_dict("records")
        stats["top_losers"] = trades.nsmallest(5, "Return_Pct")[
            ["Ticker", "Entry_Date", "Return_Pct", "PnL"]
        ].to_dict("records")
    
    if len(equity) > 0:
        rolling_max = equity["Portfolio_Value"].cummax()
        drawdown = (equity["Portfolio_Value"] - rolling_max) / rolling_max * 100
        stats["max_drawdown_pct"] = drawdown.min()
        
        daily_returns = equity["Portfolio_Value"].pct_change().dropna()
        if len(daily_returns) > 10:
            stats["sharpe_ratio"] = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            stats["sharpe_ratio"] = 0
        
        monthly = equity["Portfolio_Value"].resample("ME").last()
        monthly_returns = monthly.pct_change().dropna() * 100
        stats["monthly_returns"] = monthly_returns
    
    if spy_data is not None and len(equity) > 0:
        spy_period = spy_data.loc[
            (spy_data.index >= equity.index[0]) & (spy_data.index <= equity.index[-1])
        ]
        if len(spy_period) >= 2:
            spy_return = (spy_period.iloc[-1] / spy_period.iloc[0] - 1) * 100
            if isinstance(spy_return, pd.Series):
                spy_return = spy_return.iloc[0]
            stats["spy_return_pct"] = spy_return
            stats["alpha"] = total_return - spy_return
    
    return stats


def print_backtest_report(stats: dict, result: dict):
    """Pretty-print the backtest report."""
    print("\n" + "=" * 80)
    print("  BACKTEST REPORT — HISTORICAL PERFORMANCE")
    print("=" * 80)
    
    print(f"\n  Starting capital:    ${result['start_capital']:>12,.0f}")
    print(f"  Final value:         ${stats['final_value']:>12,.0f}")
    print(f"  Total return:        {stats['total_return_pct']:>11.1f}%")
    
    if "spy_return_pct" in stats:
        print(f"  SPY return (same):   {stats['spy_return_pct']:>11.1f}%")
        print(f"  Alpha (vs SPY):      {stats['alpha']:>11.1f}%")
    
    if "sharpe_ratio" in stats:
        print(f"  Sharpe ratio:        {stats['sharpe_ratio']:>11.2f}")
    if "max_drawdown_pct" in stats:
        print(f"  Max drawdown:        {stats['max_drawdown_pct']:>11.1f}%")
    
    print(f"\n  {'─'*70}")
    print(f"  TRADE STATISTICS")
    print(f"  {'─'*70}")
    
    if "total_trades" in stats:
        print(f"  Total trades:        {stats['total_trades']:>8d}")
        print(f"  Win rate:            {stats['win_rate']:>7.1f}%")
        print(f"  Avg winner:          {stats['avg_win_pct']:>+7.2f}%")
        print(f"  Avg loser:           {stats['avg_loss_pct']:>+7.2f}%")
        print(f"  Avg return/trade:    {stats['avg_return_pct']:>+7.2f}%")
        print(f"  Median return:       {stats['median_return_pct']:>+7.2f}%")
        print(f"  Profit factor:       {stats['profit_factor']:>7.2f}")
        print(f"  Avg hold period:     {stats['avg_hold_days']:>7.1f} days")
        print(f"  Best trade:          {stats['best_trade_pct']:>+7.1f}%")
        print(f"  Worst trade:         {stats['worst_trade_pct']:>+7.1f}%")
    
    if "top_winners" in stats and stats["top_winners"]:
        print(f"\n  TOP 5 WINNERS:")
        for t in stats["top_winners"]:
            d = t["Entry_Date"]
            dstr = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            print(f"    {t['Ticker']:6s} {dstr}  {t['Return_Pct']:+6.1f}%  (${t['PnL']:+,.0f})")
    
    if "top_losers" in stats and stats["top_losers"]:
        print(f"\n  TOP 5 LOSERS:")
        for t in stats["top_losers"]:
            d = t["Entry_Date"]
            dstr = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            print(f"    {t['Ticker']:6s} {dstr}  {t['Return_Pct']:+6.1f}%  (${t['PnL']:+,.0f})")
    
    if "monthly_returns" in stats and len(stats["monthly_returns"]) > 0:
        print(f"\n  MONTHLY RETURNS:")
        for date, ret in stats["monthly_returns"].items():
            color = "+" if ret > 0 else ""
            print(f"    {date.strftime('%Y-%m'):>7s}:  {color}{ret:.1f}%")
    
    print("\n" + "=" * 80)
