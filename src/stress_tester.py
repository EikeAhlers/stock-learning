"""
Stress Tester — Insane backtesting to prove (or disprove) the edge is real.

7 independent tests that each try to BREAK the strategy:

  1. RANDOM BASELINE       — Does the model beat randomly picking stocks?
  2. PARAMETER SENSITIVITY  — Is the result fragile to parameter changes?
  3. MONTE CARLO BOOTSTRAP  — Confidence intervals on the return via trade shuffling
  4. TICKER CONCENTRATION   — Is it just 3 lucky stocks carrying everything?
  5. REGIME STABILITY       — Does it work in both bull and bear months?
  6. TRANSACTION COSTS      — Does the edge survive realistic friction?
  7. PERMUTATION TEST       — If we scramble the model's rankings, does skill matter?

If the strategy survives ALL 7 tests, it's real. If it fails even one, be cautious.
"""
import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from src.backtester import run_backtest, analyze_backtest, _train_classifier_on_window
from src.deep_analyzer import add_all_lagged_features


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1: RANDOM BASELINE
# ═══════════════════════════════════════════════════════════════════════════

def random_baseline_test(
    returns: pd.DataFrame,
    classifier_dict: dict,
    magnitude_models: dict,
    n_simulations: int = 50,
    hold_days: int = 3,
    max_positions: int = 3,
    warmup_days: int = 252,
    start_capital: float = 100_000,
    stop_loss_pct: float = -7.0,
) -> dict:
    """
    Run N simulations where we pick stocks RANDOMLY instead of using the model.
    If the model can't beat random, it has no edge.
    """
    print(f"\n{'='*80}")
    print(f"  TEST 1: RANDOM BASELINE — {n_simulations} simulations")
    print(f"  Can the model beat a monkey throwing darts?")
    print(f"{'='*80}\n")

    df = add_all_lagged_features(returns.copy())
    dates = sorted(df.index.unique())
    trade_dates = dates[warmup_days:]

    if len(trade_dates) < 30:
        print("  Not enough data for random baseline test.")
        return {}

    random_returns = []

    for sim in range(n_simulations):
        np.random.seed(sim + 1000)
        capital = start_capital
        positions = []
        trade_log = []

        for today in trade_dates:
            # Close positions
            closed = []
            for pos in positions:
                today_data = df[(df.index == today) & (df["Ticker"] == pos["ticker"])]
                if len(today_data) == 0:
                    pos["days_held"] += 1
                    continue
                current_price = today_data["Close"].iloc[0]
                ret = (current_price / pos["entry_price"] - 1) * 100
                pos["days_held"] += 1
                should_exit = ret <= stop_loss_pct or pos["days_held"] >= hold_days
                if should_exit:
                    capital += pos["shares"] * current_price
                    trade_log.append({"Return_Pct": ret,
                                      "PnL": pos["shares"] * pos["entry_price"] * (ret / 100)})
                    closed.append(pos)
            for c in closed:
                positions.remove(c)

            # Open random positions
            open_slots = max_positions - len(positions)
            if open_slots > 0:
                today_stocks = df[df.index == today]
                if len(today_stocks) > 10:
                    held = [p["ticker"] for p in positions]
                    available = today_stocks[~today_stocks["Ticker"].isin(held)]
                    if len(available) > 0:
                        picks = available.sample(min(open_slots, len(available)))
                        for _, row in picks.iterrows():
                            price = row["Close"]
                            if price <= 0:
                                continue
                            pos_val = min(capital, start_capital / max_positions)
                            shares = int(pos_val / price)
                            if shares <= 0:
                                continue
                            capital -= shares * price
                            positions.append({
                                "ticker": row["Ticker"], "entry_price": price,
                                "shares": shares, "days_held": 0,
                            })

        # Close remaining
        for pos in positions:
            last = df[df["Ticker"] == pos["ticker"]]
            if len(last) > 0:
                ep = last.iloc[-1]["Close"]
                ret = (ep / pos["entry_price"] - 1) * 100
                capital += pos["shares"] * ep
                trade_log.append({"Return_Pct": ret,
                                  "PnL": pos["shares"] * pos["entry_price"] * (ret / 100)})

        final_val = capital
        for pos in positions:
            pass  # already closed above
        total_ret = (final_val / start_capital - 1) * 100
        random_returns.append(total_ret)

        if (sim + 1) % 10 == 0:
            print(f"  Simulation {sim+1:3d}/{n_simulations}: {total_ret:+.1f}%")

    random_returns = np.array(random_returns)
    return {
        "random_returns": random_returns,
        "random_mean": random_returns.mean(),
        "random_median": np.median(random_returns),
        "random_std": random_returns.std(),
        "random_5th_pctile": np.percentile(random_returns, 5),
        "random_95th_pctile": np.percentile(random_returns, 95),
        "random_best": random_returns.max(),
        "random_worst": random_returns.min(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2: PARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════

def parameter_sensitivity_test(
    returns: pd.DataFrame,
    classifier_dict: dict,
    magnitude_models: dict,
    spy_data=None,
) -> pd.DataFrame:
    """
    Run backtest across many parameter combinations.
    If only ONE specific combo works, the result is fragile (overfit to params).
    """
    print(f"\n{'='*80}")
    print(f"  TEST 2: PARAMETER SENSITIVITY SWEEP")
    print(f"  Does the strategy survive different parameter choices?")
    print(f"{'='*80}\n")

    param_grid = [
        # (hold_days, min_prob, stop_loss, top_n, max_pos)
        (2, 0.40, -5.0, 3, 3),
        (2, 0.50, -7.0, 3, 3),
        (2, 0.60, -10.0, 3, 3),
        (3, 0.40, -5.0, 3, 3),
        (3, 0.50, -7.0, 3, 3),   # baseline
        (3, 0.60, -10.0, 3, 3),
        (5, 0.40, -5.0, 3, 3),
        (5, 0.50, -7.0, 3, 3),
        (5, 0.60, -10.0, 3, 3),
        (3, 0.50, -7.0, 5, 5),   # more positions
        (3, 0.50, -7.0, 1, 1),   # concentrated
        (3, 0.30, -7.0, 3, 3),   # lower threshold
        (3, 0.70, -7.0, 3, 3),   # higher threshold
        (5, 0.50, -5.0, 5, 5),   # longer hold, more positions
        (1, 0.50, -5.0, 3, 3),   # day-trade
    ]

    results = []
    for i, (hd, mp, sl, tn, mpos) in enumerate(param_grid):
        label = f"Hold={hd}d P>={mp:.0%} SL={sl}% Top{tn} Max{mpos}"
        print(f"  [{i+1:2d}/{len(param_grid)}] {label}", end="")

        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            bt = run_backtest(
                returns=returns, classifier_dict=classifier_dict,
                magnitude_models=magnitude_models,
                start_capital=100_000, top_n=tn, hold_days=hd,
                min_probability=mp, stop_loss_pct=sl,
                max_positions=mpos, retrain_every=63,
                warmup_days=252, honest_mode=True,
            )
        finally:
            sys.stdout = old_stdout

        if not bt or "equity_curve" not in bt:
            print(f"  → FAILED")
            continue

        st = analyze_backtest(bt, spy_data=spy_data)
        n_trades = st.get("total_trades", 0)
        ret = st.get("total_return_pct", 0)
        sharpe = st.get("sharpe_ratio", 0)
        mdd = st.get("max_drawdown_pct", 0)
        wr = st.get("win_rate", 0)
        pf = st.get("profit_factor", 0)
        alpha = st.get("alpha", ret)

        print(f"  → Ret={ret:+.1f}% Sharpe={sharpe:.2f} DD={mdd:.1f}% "
              f"WR={wr:.0f}% PF={pf:.2f} Trades={n_trades}")

        results.append({
            "Hold_Days": hd, "Min_Prob": mp, "Stop_Loss": sl,
            "Top_N": tn, "Max_Pos": mpos,
            "Return_Pct": ret, "Sharpe": sharpe, "Max_DD": mdd,
            "Win_Rate": wr, "Profit_Factor": pf, "Alpha": alpha,
            "Trades": n_trades, "Label": label,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3: MONTE CARLO BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════

def monte_carlo_bootstrap(
    trades_df: pd.DataFrame,
    start_capital: float = 100_000,
    n_simulations: int = 1000,
) -> dict:
    """
    Shuffle the actual trade returns 1000 times to get confidence intervals.
    This tells you: "Given these trades, how much of the result is due to ORDER vs EDGE?"
    """
    print(f"\n{'='*80}")
    print(f"  TEST 3: MONTE CARLO BOOTSTRAP — {n_simulations:,} simulations")
    print(f"  Shuffling trade order to get confidence intervals")
    print(f"{'='*80}\n")

    if len(trades_df) == 0:
        print("  No trades to bootstrap.")
        return {}

    returns_array = trades_df["Return_Pct"].values
    pnl_array = trades_df["PnL"].values
    n_trades = len(returns_array)

    sim_total_returns = []
    sim_max_drawdowns = []
    sim_sharpe = []

    for sim in range(n_simulations):
        np.random.seed(sim)
        shuffled_pnl = np.random.choice(pnl_array, size=n_trades, replace=True)
        equity = start_capital + np.cumsum(shuffled_pnl)
        total_ret = (equity[-1] / start_capital - 1) * 100
        sim_total_returns.append(total_ret)

        # Running max drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100
        sim_max_drawdowns.append(dd.min())

        # Crude Sharpe: mean daily vs std daily from trade pnl
        daily_equiv = shuffled_pnl / (start_capital / 3)  # rough daily return
        if daily_equiv.std() > 0:
            sr = (daily_equiv.mean() / daily_equiv.std()) * np.sqrt(252 / 3)
            sim_sharpe.append(sr)

    sim_total_returns = np.array(sim_total_returns)
    sim_max_drawdowns = np.array(sim_max_drawdowns)

    result = {
        "mc_returns": sim_total_returns,
        "mc_drawdowns": sim_max_drawdowns,
        "mc_mean_return": sim_total_returns.mean(),
        "mc_median_return": np.median(sim_total_returns),
        "mc_5th_pctile": np.percentile(sim_total_returns, 5),
        "mc_25th_pctile": np.percentile(sim_total_returns, 25),
        "mc_75th_pctile": np.percentile(sim_total_returns, 75),
        "mc_95th_pctile": np.percentile(sim_total_returns, 95),
        "mc_prob_positive": (sim_total_returns > 0).mean() * 100,
        "mc_prob_beat_spy": None,  # filled by caller
        "mc_worst_case": sim_total_returns.min(),
        "mc_best_case": sim_total_returns.max(),
        "mc_mean_drawdown": sim_max_drawdowns.mean(),
        "mc_worst_drawdown": sim_max_drawdowns.min(),
    }

    print(f"  Mean return:        {result['mc_mean_return']:+.1f}%")
    print(f"  Median return:      {result['mc_median_return']:+.1f}%")
    print(f"  5th percentile:     {result['mc_5th_pctile']:+.1f}% (worst realistic)")
    print(f"  95th percentile:    {result['mc_95th_pctile']:+.1f}% (best realistic)")
    print(f"  P(positive return): {result['mc_prob_positive']:.0f}%")
    print(f"  Worst case:         {result['mc_worst_case']:+.1f}%")
    print(f"  Best case:          {result['mc_best_case']:+.1f}%")
    print(f"  Avg max drawdown:   {result['mc_mean_drawdown']:.1f}%")
    print(f"  Worst drawdown:     {result['mc_worst_drawdown']:.1f}%")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 4: TICKER CONCENTRATION
# ═══════════════════════════════════════════════════════════════════════════

def ticker_concentration_test(trades_df: pd.DataFrame) -> dict:
    """
    Check if returns are dominated by a few lucky picks.
    If removing the top 3 tickers kills the profit, the edge is fragile.
    """
    print(f"\n{'='*80}")
    print(f"  TEST 4: TICKER CONCENTRATION")
    print(f"  Are profits from skill or a few lucky tickers?")
    print(f"{'='*80}\n")

    if len(trades_df) == 0:
        return {}

    total_pnl = trades_df["PnL"].sum()
    ticker_pnl = trades_df.groupby("Ticker")["PnL"].sum().sort_values(ascending=False)
    ticker_count = trades_df.groupby("Ticker").size().sort_values(ascending=False)

    n_tickers = len(ticker_pnl)
    top3_tickers = ticker_pnl.head(3)
    top3_pnl = top3_tickers.sum()
    bottom3_tickers = ticker_pnl.tail(3)
    bottom3_pnl = bottom3_tickers.sum()

    # PnL without top 3 winners
    pnl_without_top3 = total_pnl - top3_pnl
    # PnL without bottom 3 losers
    pnl_without_bottom3 = total_pnl - bottom3_pnl

    # "Leave one ticker out" — remove each ticker and check if still profitable
    still_profitable_count = 0
    for ticker in ticker_pnl.index:
        remaining_pnl = total_pnl - ticker_pnl[ticker]
        if remaining_pnl > 0:
            still_profitable_count += 1

    result = {
        "total_pnl": total_pnl,
        "n_tickers_traded": n_tickers,
        "top3_pnl": top3_pnl,
        "top3_pct_of_total": (top3_pnl / total_pnl * 100) if total_pnl > 0 else 0,
        "pnl_without_top3": pnl_without_top3,
        "still_profitable_without_top3": pnl_without_top3 > 0,
        "bottom3_pnl": bottom3_pnl,
        "still_profitable_without_any_one": still_profitable_count,
        "ticker_pnl_table": ticker_pnl,
        "ticker_count_table": ticker_count,
    }

    print(f"  Total tickers traded:       {n_tickers}")
    print(f"  Total P&L:                  ${total_pnl:+,.0f}")
    print(f"\n  Top 3 contributors:")
    for t, p in top3_tickers.items():
        cnt = ticker_count.get(t, 0)
        print(f"    {t:6s}: ${p:+,.0f} ({cnt} trades)")
    print(f"  Top 3 = ${top3_pnl:+,.0f} ({result['top3_pct_of_total']:.0f}% of total)")
    print(f"  P&L WITHOUT top 3:          ${pnl_without_top3:+,.0f} "
          f"({'STILL PROFITABLE' if pnl_without_top3 > 0 else 'GOES NEGATIVE'})")

    print(f"\n  Bottom 3 (biggest losers):")
    for t, p in bottom3_tickers.items():
        cnt = ticker_count.get(t, 0)
        print(f"    {t:6s}: ${p:+,.0f} ({cnt} trades)")

    print(f"\n  Removing ANY single ticker: still profitable for "
          f"{still_profitable_count}/{n_tickers} tickers")

    # Herfindahl index of PnL concentration
    if total_pnl > 0:
        pnl_shares = (ticker_pnl[ticker_pnl > 0] / ticker_pnl[ticker_pnl > 0].sum()) ** 2
        hhi = pnl_shares.sum()
        result["herfindahl_index"] = hhi
        print(f"  Herfindahl Index (HHI):     {hhi:.3f} "
              f"({'CONCENTRATED' if hhi > 0.25 else 'DIVERSIFIED'})")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 5: REGIME STABILITY
# ═══════════════════════════════════════════════════════════════════════════

def regime_stability_test(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    spy_data: pd.DataFrame = None,
) -> dict:
    """
    Split performance into up-market and down-market months.
    A real edge should work in BOTH — not just ride the bull.
    """
    print(f"\n{'='*80}")
    print(f"  TEST 5: REGIME STABILITY")
    print(f"  Does the strategy work in BOTH bull and bear markets?")
    print(f"{'='*80}\n")

    if len(trades_df) == 0 or spy_data is None:
        print("  Need trades and SPY data for this test.")
        return {}

    # Monthly SPY returns
    spy_monthly = spy_data.resample("ME").last().pct_change().dropna() * 100

    # Monthly strategy P&L
    trades_df = trades_df.copy()
    trades_df["Exit_Date"] = pd.to_datetime(trades_df["Exit_Date"])
    trades_df["Month"] = trades_df["Exit_Date"].dt.to_period("M")
    monthly_pnl = trades_df.groupby("Month").agg(
        PnL=("PnL", "sum"),
        Trades=("PnL", "count"),
        Win_Rate=("Return_Pct", lambda x: (x > 0).mean() * 100),
        Avg_Return=("Return_Pct", "mean"),
    )

    bull_months = []
    bear_months = []

    for period, row in monthly_pnl.iterrows():
        month_end = period.to_timestamp("M")
        # Find nearest SPY monthly return
        spy_match = spy_monthly.index[spy_monthly.index <= month_end]
        if len(spy_match) == 0:
            continue
        closest = spy_match[-1]
        spy_ret = spy_monthly.loc[closest]
        if isinstance(spy_ret, pd.Series):
            spy_ret = spy_ret.iloc[0]

        entry = {
            "Month": str(period),
            "PnL": row["PnL"],
            "Trades": row["Trades"],
            "Win_Rate": row["Win_Rate"],
            "Avg_Return": row["Avg_Return"],
            "SPY_Return": spy_ret,
        }

        if spy_ret >= 0:
            bull_months.append(entry)
        else:
            bear_months.append(entry)

    result = {
        "bull_months": pd.DataFrame(bull_months),
        "bear_months": pd.DataFrame(bear_months),
    }

    print(f"  BULL MONTHS (SPY >= 0%):")
    if len(bull_months) > 0:
        bdf = result["bull_months"]
        print(f"    Count:           {len(bdf)}")
        print(f"    Total P&L:       ${bdf['PnL'].sum():+,.0f}")
        print(f"    Avg monthly P&L: ${bdf['PnL'].mean():+,.0f}")
        print(f"    Avg win rate:    {bdf['Win_Rate'].mean():.0f}%")
        result["bull_total_pnl"] = bdf["PnL"].sum()
        result["bull_avg_pnl"] = bdf["PnL"].mean()
        result["bull_profitable"] = (bdf["PnL"] > 0).sum()
    else:
        print("    (no bull months)")

    print(f"\n  BEAR MONTHS (SPY < 0%):")
    if len(bear_months) > 0:
        bdf = result["bear_months"]
        print(f"    Count:           {len(bdf)}")
        print(f"    Total P&L:       ${bdf['PnL'].sum():+,.0f}")
        print(f"    Avg monthly P&L: ${bdf['PnL'].mean():+,.0f}")
        print(f"    Avg win rate:    {bdf['Win_Rate'].mean():.0f}%")
        result["bear_total_pnl"] = bdf["PnL"].sum()
        result["bear_avg_pnl"] = bdf["PnL"].mean()
        result["bear_profitable"] = (bdf["PnL"] > 0).sum()
    else:
        print("    (no bear months)")

    # Check if strategy alpha exists in bear months too
    if len(bear_months) > 0:
        bear_pnl = pd.DataFrame(bear_months)["PnL"].sum()
        if bear_pnl > 0:
            result["bear_alpha"] = True
            print(f"\n  ✓ Strategy is PROFITABLE even in bear months — real alpha")
        else:
            result["bear_alpha"] = False
            print(f"\n  ⚠️  Strategy LOSES money in bear months — may just be riding the bull")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 6: TRANSACTION COST IMPACT
# ═══════════════════════════════════════════════════════════════════════════

def transaction_cost_test(
    trades_df: pd.DataFrame,
    start_capital: float = 100_000,
) -> dict:
    """
    Re-calculate returns with realistic trading friction:
    - 0.05% commission per trade (round trip)
    - 0.1% slippage per trade (bid-ask + market impact)
    - Total: ~0.15% per round trip (0.075% per leg)
    """
    print(f"\n{'='*80}")
    print(f"  TEST 6: TRANSACTION COST IMPACT")
    print(f"  Does the edge survive after realistic trading friction?")
    print(f"{'='*80}\n")

    if len(trades_df) == 0:
        return {}

    cost_levels = [
        ("Zero cost (ideal)", 0.0),
        ("Low (0.05% rt)", 0.05),
        ("Medium (0.15% rt)", 0.15),
        ("High (0.30% rt)", 0.30),
        ("Extreme (0.50% rt)", 0.50),
    ]

    results = []
    for label, cost_pct in cost_levels:
        adjusted_returns = trades_df["Return_Pct"] - cost_pct
        adjusted_pnl = trades_df["PnL"] - (trades_df["Entry_Price"] * trades_df.get("shares", 1) * cost_pct / 100)

        # Simpler: just remove cost_pct from each trade return
        n = len(adjusted_returns)
        wins = (adjusted_returns > 0).sum()
        wr = wins / n * 100
        avg_ret = adjusted_returns.mean()
        total_ret = adjusted_returns.sum()
        # Approximate total portfolio return
        approx_total_ret = total_ret / (start_capital / 100)

        results.append({
            "Label": label,
            "Cost_Pct": cost_pct,
            "Win_Rate": wr,
            "Avg_Return": avg_ret,
            "Sum_Returns": total_ret,
            "Approx_Portfolio_Ret": approx_total_ret,
        })

        profit_loss = "PROFITABLE" if avg_ret > 0 else "UNPROFITABLE"
        print(f"  {label:25s}: WR={wr:.0f}%  Avg={avg_ret:+.2f}%  "
              f"Sum={total_ret:+.0f}%  → {profit_loss}")

    return {"cost_results": pd.DataFrame(results)}


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 7: PERMUTATION TEST (Statistical Significance)
# ═══════════════════════════════════════════════════════════════════════════

def permutation_test(
    returns: pd.DataFrame,
    classifier_dict: dict,
    magnitude_models: dict,
    actual_return: float,
    n_permutations: int = 30,
    hold_days: int = 3,
    spy_data=None,
) -> dict:
    """
    Scramble the model's probability rankings and re-run the backtest N times.
    If scrambled results are just as good → the model has NO skill, it's luck.
    
    p-value < 0.05 means the model's skill is statistically significant.
    """
    print(f"\n{'='*80}")
    print(f"  TEST 7: PERMUTATION TEST — {n_permutations} scrambled runs")
    print(f"  Scrambling model predictions to test if skill is real")
    print(f"{'='*80}\n")

    import io, sys

    # We create a fake classifier that returns RANDOM probabilities
    clf_features = classifier_dict["features"]
    scrambled_returns = []

    for perm in range(n_permutations):
        print(f"  Permutation {perm+1:2d}/{n_permutations}", end="")

        # Monkey-patch: create a wrapper that scrambles predictions
        orig_classifier = dict(classifier_dict)

        class ScrambledModel:
            """Model that returns random probabilities."""
            def __init__(self, seed):
                self.rng = np.random.RandomState(seed + 2000)
                self.classes_ = np.array([0, 1])
            def predict_proba(self, X):
                n = len(X)
                probs = self.rng.uniform(0, 1, size=n)
                return np.column_stack([1 - probs, probs])

        scrambled_clf = {
            "model": ScrambledModel(perm),
            "scaler": classifier_dict["scaler"],
            "features": clf_features,
        }

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            bt = run_backtest(
                returns=returns, classifier_dict=scrambled_clf,
                magnitude_models=magnitude_models,
                start_capital=100_000, top_n=3, hold_days=hold_days,
                min_probability=0.5, stop_loss_pct=-7.0,
                max_positions=3, retrain_every=63,
                warmup_days=252, honest_mode=False,  # Use scrambled model directly
            )
        finally:
            sys.stdout = old_stdout

        if bt and "equity_curve" in bt:
            st = analyze_backtest(bt, spy_data=spy_data)
            perm_ret = st.get("total_return_pct", 0)
        else:
            perm_ret = 0

        scrambled_returns.append(perm_ret)
        print(f"  → {perm_ret:+.1f}%")

    scrambled_returns = np.array(scrambled_returns)
    n_better = (scrambled_returns >= actual_return).sum()
    p_value = (n_better + 1) / (n_permutations + 1)  # +1 for the actual result

    result = {
        "scrambled_returns": scrambled_returns,
        "scrambled_mean": scrambled_returns.mean(),
        "scrambled_std": scrambled_returns.std(),
        "actual_return": actual_return,
        "n_scrambled_beat_actual": int(n_better),
        "p_value": p_value,
    }

    print(f"\n  Actual strategy return:       {actual_return:+.1f}%")
    print(f"  Scrambled mean return:        {scrambled_returns.mean():+.1f}%")
    print(f"  Scrambled std:                {scrambled_returns.std():.1f}%")
    print(f"  Scrambled beat actual:        {n_better}/{n_permutations}")
    print(f"  p-value:                      {p_value:.4f}")

    if p_value < 0.01:
        print(f"\n  ✓ p < 0.01 — HIGHLY SIGNIFICANT. The model has real skill.")
    elif p_value < 0.05:
        print(f"\n  ✓ p < 0.05 — SIGNIFICANT. The model likely has skill.")
    elif p_value < 0.10:
        print(f"\n  ⚠️  p < 0.10 — MARGINAL. Suggestive but not conclusive.")
    else:
        print(f"\n  ✗ p >= 0.10 — NOT SIGNIFICANT. Could be luck.")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_stress_test_verdict(
    actual_return: float,
    random_result: dict,
    sensitivity_df: pd.DataFrame,
    mc_result: dict,
    concentration_result: dict,
    regime_result: dict,
    cost_result: dict,
    permutation_result: dict,
):
    """Print the final all-in-one stress test verdict."""
    print(f"\n\n{'═'*90}")
    print(f"  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║     STRESS TEST FINAL VERDICT — ALL 7 TESTS            ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")
    print(f"{'═'*90}\n")

    tests_passed = 0
    total_tests = 7

    # Test 1: Random baseline
    if random_result:
        random_mean = random_result.get("random_mean", 0)
        beats_random = actual_return > random_result.get("random_95th_pctile", actual_return)
        if beats_random:
            tests_passed += 1
            print(f"  ✓ TEST 1 — RANDOM BASELINE:     PASS")
        elif actual_return > random_mean:
            tests_passed += 0.5
            print(f"  ~ TEST 1 — RANDOM BASELINE:     PARTIAL (beats mean but not 95th pctile)")
        else:
            print(f"  ✗ TEST 1 — RANDOM BASELINE:     FAIL")
        print(f"           Strategy={actual_return:+.1f}% vs Random mean={random_mean:+.1f}%, "
              f"95th={random_result.get('random_95th_pctile',0):+.1f}%")
    else:
        print(f"  ? TEST 1 — RANDOM BASELINE:     SKIPPED")

    # Test 2: Parameter sensitivity
    if len(sensitivity_df) > 0:
        profitable_configs = (sensitivity_df["Return_Pct"] > 0).sum()
        total_configs = len(sensitivity_df)
        pct_profitable = profitable_configs / total_configs * 100
        if pct_profitable >= 70:
            tests_passed += 1
            print(f"  ✓ TEST 2 — PARAMETER SENSITIVITY: PASS")
        elif pct_profitable >= 50:
            tests_passed += 0.5
            print(f"  ~ TEST 2 — PARAMETER SENSITIVITY: PARTIAL")
        else:
            print(f"  ✗ TEST 2 — PARAMETER SENSITIVITY: FAIL")
        print(f"           {profitable_configs}/{total_configs} configs profitable ({pct_profitable:.0f}%). "
              f"Return range: {sensitivity_df['Return_Pct'].min():+.1f}% to "
              f"{sensitivity_df['Return_Pct'].max():+.1f}%")
    else:
        print(f"  ? TEST 2 — PARAMETER SENSITIVITY: SKIPPED")

    # Test 3: Monte Carlo
    if mc_result:
        prob_positive = mc_result.get("mc_prob_positive", 0)
        if prob_positive >= 90:
            tests_passed += 1
            print(f"  ✓ TEST 3 — MONTE CARLO:         PASS")
        elif prob_positive >= 70:
            tests_passed += 0.5
            print(f"  ~ TEST 3 — MONTE CARLO:         PARTIAL")
        else:
            print(f"  ✗ TEST 3 — MONTE CARLO:         FAIL")
        print(f"           P(profit)={prob_positive:.0f}%, "
              f"95% CI: [{mc_result.get('mc_5th_pctile',0):+.1f}%, "
              f"{mc_result.get('mc_95th_pctile',0):+.1f}%]")
    else:
        print(f"  ? TEST 3 — MONTE CARLO:         SKIPPED")

    # Test 4: Concentration
    if concentration_result:
        still_ok = concentration_result.get("still_profitable_without_top3", False)
        if still_ok:
            tests_passed += 1
            print(f"  ✓ TEST 4 — TICKER CONCENTRATION: PASS")
        else:
            print(f"  ✗ TEST 4 — TICKER CONCENTRATION: FAIL (top 3 carry everything)")
        pct = concentration_result.get("top3_pct_of_total", 0)
        print(f"           Top 3 tickers = {pct:.0f}% of profits. "
              f"Without them: ${concentration_result.get('pnl_without_top3',0):+,.0f}")
    else:
        print(f"  ? TEST 4 — TICKER CONCENTRATION: SKIPPED")

    # Test 5: Regime
    if regime_result:
        bear_alpha = regime_result.get("bear_alpha", False)
        if bear_alpha:
            tests_passed += 1
            print(f"  ✓ TEST 5 — REGIME STABILITY:    PASS (profitable in bear months)")
        else:
            print(f"  ✗ TEST 5 — REGIME STABILITY:    FAIL (loses in bear months)")
        bull_pnl = regime_result.get("bull_total_pnl", 0)
        bear_pnl = regime_result.get("bear_total_pnl", 0)
        print(f"           Bull P&L=${bull_pnl:+,.0f}, Bear P&L=${bear_pnl:+,.0f}")
    else:
        print(f"  ? TEST 5 — REGIME STABILITY:    SKIPPED")

    # Test 6: Transaction costs
    if cost_result and "cost_results" in cost_result:
        cdf = cost_result["cost_results"]
        medium_row = cdf[cdf["Cost_Pct"] == 0.15]
        if len(medium_row) > 0 and medium_row.iloc[0]["Avg_Return"] > 0:
            tests_passed += 1
            print(f"  ✓ TEST 6 — TRANSACTION COSTS:   PASS (survives 0.15% friction)")
        elif len(medium_row) > 0:
            print(f"  ✗ TEST 6 — TRANSACTION COSTS:   FAIL (edge eaten by friction)")
        print(f"           Avg return at 0.15% cost: "
              f"{medium_row.iloc[0]['Avg_Return']:+.2f}% per trade" if len(medium_row) > 0 else "")
    else:
        print(f"  ? TEST 6 — TRANSACTION COSTS:   SKIPPED")

    # Test 7: Permutation
    if permutation_result:
        p_val = permutation_result.get("p_value", 1.0)
        if p_val < 0.05:
            tests_passed += 1
            print(f"  ✓ TEST 7 — PERMUTATION TEST:    PASS (p={p_val:.3f})")
        elif p_val < 0.10:
            tests_passed += 0.5
            print(f"  ~ TEST 7 — PERMUTATION TEST:    MARGINAL (p={p_val:.3f})")
        else:
            print(f"  ✗ TEST 7 — PERMUTATION TEST:    FAIL (p={p_val:.3f})")
        print(f"           Scrambled mean={permutation_result.get('scrambled_mean',0):+.1f}%, "
              f"Actual={actual_return:+.1f}%")
    else:
        print(f"  ? TEST 7 — PERMUTATION TEST:    SKIPPED")

    # Final score
    print(f"\n  {'─'*80}")
    print(f"  SCORE: {tests_passed:.1f} / {total_tests} tests passed")
    print(f"  {'─'*80}")

    if tests_passed >= 6:
        verdict = "BATTLE-TESTED — The strategy has a REAL, robust edge"
        emoji = "★★★★★"
    elif tests_passed >= 5:
        verdict = "STRONG — Edge is likely real but has some weaknesses"
        emoji = "★★★★"
    elif tests_passed >= 4:
        verdict = "MODERATE — Some evidence of edge, proceed with caution"
        emoji = "★★★"
    elif tests_passed >= 3:
        verdict = "WEAK — Edge is fragile and may not survive live trading"
        emoji = "★★"
    else:
        verdict = "FAILED — No reliable edge detected. DO NOT TRADE."
        emoji = "★"

    print(f"\n  {emoji}  {verdict}")
    print(f"\n{'═'*90}")

    return {"tests_passed": tests_passed, "total_tests": total_tests, "verdict": verdict}
