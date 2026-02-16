"""
Expected Value Engine — The core decision engine.

This is where everything comes together:
  P(big move) × E(return | move) = Expected Value

But we go further:
  - Conviction score: how many independent signals agree
  - Risk adjustment: penalize high-volatility names
  - Market regime filter: sit out when conditions are hostile
  - Optimal horizon: which holding period maximizes the edge
  - Position sizing hint: Kelly criterion approximation

The output is ONE list: ranked by expected profit per trade.
This is what you trade.
"""
import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from src.deep_analyzer import add_all_lagged_features, ALL_RAW_FEATURES
from src.setup_detector import SETUP_NAMES


def compute_expected_value(
    returns: pd.DataFrame,
    classifier_dict: dict,
    magnitude_models: dict,
    regime_data: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    min_conviction: float = 0.3,
) -> pd.DataFrame:
    """
    THE master scoring function.
    
    For each stock, computes:
      1. P(big move)           — from classifier
      2. E(return)             — from magnitude regressor (multiple horizons)
      3. Expected Value        — P × E
      4. Conviction Score      — agreement of independent signals
      5. Risk-Adjusted Score   — EV / realized_vol
      6. Optimal Horizon       — which holding period is best for this stock
      7. Final Rank            — sorted by risk-adjusted EV
    
    Returns the trading decision list.
    """
    if not classifier_dict or not magnitude_models:
        print("ERROR: Need both classifier and magnitude models")
        return pd.DataFrame()
    
    # ── Step 1: Get classifier probabilities ─────────────────────────
    clf_model = classifier_dict["model"]
    clf_scaler = classifier_dict["scaler"]
    clf_features = classifier_dict["features"]
    
    df = add_all_lagged_features(returns.copy())
    
    if "Ticker" in df.columns:
        latest = df.groupby("Ticker").last()
    else:
        latest = df.copy()
    
    available = latest[clf_features].dropna()
    if len(available) == 0:
        print("No stocks with complete data")
        return pd.DataFrame()
    
    X_clf = clf_scaler.transform(available[clf_features])
    probabilities = clf_model.predict_proba(X_clf)[:, 1]
    
    scored = pd.DataFrame({
        "Ticker": available.index,
        "P_BigMove": probabilities,
    })
    scored = scored.set_index("Ticker")
    
    # ── Step 2: Get magnitude predictions for each horizon ───────────
    for horizon, mag_dict in magnitude_models.items():
        mag_features = mag_dict["features"]
        mag_available = latest[mag_features].dropna()
        
        common_tickers = scored.index.intersection(mag_available.index)
        if len(common_tickers) == 0:
            scored[f"E_Return_{horizon}d"] = np.nan
            continue
        
        X_mag = mag_dict["scaler"].transform(mag_available.loc[common_tickers, mag_features])
        mag_preds = mag_dict["model"].predict(X_mag)
        scored.loc[common_tickers, f"E_Return_{horizon}d"] = mag_preds
    
    # ── Step 3: Expected Value = P × E ───────────────────────────────
    for horizon in magnitude_models:
        ret_col = f"E_Return_{horizon}d"
        if ret_col in scored.columns:
            scored[f"EV_{horizon}d"] = scored["P_BigMove"] * scored[ret_col]
    
    # Find optimal horizon (max EV) for each stock
    ev_cols = [c for c in scored.columns if c.startswith("EV_")]
    if ev_cols:
        scored["Best_EV"] = scored[ev_cols].max(axis=1)
        scored["Best_Horizon"] = scored[ev_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(float)
        scored["Best_E_Return"] = scored.apply(
            lambda row: row.get(f"E_Return_{int(row['Best_Horizon'])}d", np.nan)
            if pd.notna(row.get("Best_Horizon")) else np.nan, axis=1
        )
    else:
        scored["Best_EV"] = scored["P_BigMove"] * 5.0  # fallback: assume 5% if no regressor
        scored["Best_Horizon"] = 1
        scored["Best_E_Return"] = 5.0
    
    # ── Step 4: Conviction Score ─────────────────────────────────────
    # Count how many independent signal classes agree this stock will move
    conviction_components = []
    
    # 4a. Classifier confidence (rescaled 0-1)
    denom = max(scored["P_BigMove"].quantile(0.95) - scored["P_BigMove"].quantile(0.5), 0.01)
    clf_signal = (scored["P_BigMove"] - scored["P_BigMove"].quantile(0.5)) / denom
    clf_signal = clf_signal.clip(0, 1)
    conviction_components.append(clf_signal)
    
    # 4b. Regressor agreement (are multiple horizons positive?)
    e_ret_cols = [c for c in scored.columns if c.startswith("E_Return_")]
    if e_ret_cols:
        positive_horizons = (scored[e_ret_cols] > 0).sum(axis=1) / len(e_ret_cols)
        conviction_components.append(positive_horizons)
    
    # 4c. Setup count from latest data
    setup_cols = [s for s in SETUP_NAMES if s in latest.columns]
    if setup_cols:
        setup_score = latest.loc[scored.index, setup_cols].sum(axis=1).clip(0, 3) / 3
        scored["Setup_Count"] = latest.loc[scored.index, setup_cols].sum(axis=1)
        conviction_components.append(setup_score)
    
    # 4d. Volume confirmation (recent unusual volume)
    if "Vol_ZScore" in latest.columns:
        vol_signal = (latest.loc[scored.index, "Vol_ZScore"] > 1.0).astype(float)
        conviction_components.append(vol_signal)
    
    # 4e. Momentum alignment
    if "MTF_Bullish_Count" in latest.columns:
        mtf = latest.loc[scored.index, "MTF_Bullish_Count"] / 5
        conviction_components.append(mtf)
    
    if conviction_components:
        scored["Conviction"] = pd.concat(conviction_components, axis=1).mean(axis=1)
    else:
        scored["Conviction"] = scored["P_BigMove"]
    
    # ── Step 5: Risk adjustment ──────────────────────────────────────
    if "Volatility_20d" in latest.columns:
        vol = latest.loc[scored.index, "Volatility_20d"].clip(lower=0.5)
        scored["Volatility"] = vol
        scored["Risk_Adjusted_EV"] = scored["Best_EV"] / vol
    else:
        scored["Volatility"] = 2.0
        scored["Risk_Adjusted_EV"] = scored["Best_EV"]
    
    # ── Step 6: Market regime filter ─────────────────────────────────
    # In hostile regimes, raise the bar
    if "Market_Regime" in latest.columns:
        current_regime = latest["Market_Regime"].mode()
        if len(current_regime) > 0:
            current_regime = current_regime.iloc[0]
            scored["Market_Regime"] = current_regime
            
            regime_multiplier = {
                "low_vol_bull": 1.0,
                "normal_bull": 0.9,
                "high_vol_bull": 0.7,
                "normal_bear": 0.6,
                "high_vol_bear": 0.4,
            }
            mult = regime_multiplier.get(current_regime, 0.8)
            scored["Regime_Mult"] = mult
            scored["Final_Score"] = scored["Risk_Adjusted_EV"] * mult * scored["Conviction"]
        else:
            scored["Final_Score"] = scored["Risk_Adjusted_EV"] * scored["Conviction"]
    else:
        scored["Final_Score"] = scored["Risk_Adjusted_EV"] * scored["Conviction"]
    
    # ── Step 7: Kelly criterion position sizing hint ─────────────────
    # Kelly = (p * b - q) / b, where p=win prob, b=avg win/avg loss, q=1-p
    # Simplified: fraction of capital to risk
    scored["Kelly_Fraction"] = (
        scored["P_BigMove"] - (1 - scored["P_BigMove"]) / scored["Best_E_Return"].clip(lower=1).abs()
    ).clip(lower=0, upper=0.25)  # Cap at 25% of capital
    
    # ── Step 8: Add context columns for display ──────────────────────
    for col in ["RSI_14", "Vol_Ratio", "Return_5d", "Dist_SMA20_Pct",
                "ATR_Ratio", "BB_Width", "OBV_Bullish_Div",
                "Dist_52w_High_Pct", "Close"]:
        if col in latest.columns:
            scored[col] = latest.loc[scored.index, col]
    
    # Active setups string
    if setup_cols:
        scored["Active_Setups"] = latest.loc[scored.index, setup_cols].apply(
            lambda row: ", ".join([c.replace("Setup_", "") for c in setup_cols if row[c] == 1]) or "none",
            axis=1,
        )
    
    # ── Step 9: SHAP explanations for top picks ──────────────────────
    if classifier_dict.get("shap_explainer") is not None:
        try:
            import shap
            explainer = classifier_dict["shap_explainer"]
            scored_sorted = scored.sort_values("Final_Score", ascending=False)
            explanations = {}
            
            for ticker in scored_sorted.head(top_n).index:
                if ticker not in available.index:
                    continue
                x = available.loc[[ticker]][clf_features]
                x_scaled = clf_scaler.transform(x)
                sv = explainer.shap_values(x_scaled)
                if isinstance(sv, list):
                    sv = sv[1]
                
                feature_impacts = pd.DataFrame({
                    "Feature": clf_features,
                    "SHAP": sv[0],
                }).sort_values("SHAP", key=abs, ascending=False)
                
                top5 = feature_impacts.head(5)
                why = " | ".join([
                    f"{r['Feature'].replace('Prev_', '')} ({'+' if r['SHAP']>0 else ''}{r['SHAP']:.3f})"
                    for _, r in top5.iterrows()
                ])
                explanations[ticker] = why
            
            scored["Why"] = scored.index.map(explanations).fillna("")
        except Exception as e:
            scored["Why"] = ""
    else:
        scored["Why"] = ""
    
    # ── Final sort & filter ──────────────────────────────────────────
    scored = scored.sort_values("Final_Score", ascending=False)
    
    # Filter by minimum conviction
    scored = scored[scored["Conviction"] >= min_conviction]
    
    result = scored.head(top_n).reset_index()
    result.index = range(1, len(result) + 1)
    result.index.name = "Rank"
    
    return result


def print_trading_decisions(decisions: pd.DataFrame, show_detail: bool = True):
    """
    Pretty-print the trading decision list.
    This is THE output — what to buy and why.
    """
    if len(decisions) == 0:
        print("No stocks meet the conviction threshold.")
        return
    
    regime = decisions.get("Market_Regime", pd.Series(["unknown"])).iloc[0]
    
    print("=" * 110)
    print("  TRADING DECISIONS — STOCKS TO BUY")
    print("=" * 110)
    print(f"  Market Regime: {regime}")
    print(f"  Stocks scored: many → filtered to {len(decisions)} by conviction\n")
    
    for _, row in decisions.iterrows():
        ticker = row["Ticker"]
        prob = row["P_BigMove"]
        ev = row.get("Best_EV", 0)
        e_ret = row.get("Best_E_Return", 0)
        horizon = int(row.get("Best_Horizon", 1))
        conviction = row.get("Conviction", 0)
        score = row.get("Final_Score", 0)
        setups = row.get("Active_Setups", "none")
        kelly = row.get("Kelly_Fraction", 0)
        price = row.get("Close", 0)
        rsi = row.get("RSI_14", None)
        
        rsi_str = f"RSI:{rsi:.0f}" if pd.notna(rsi) else ""
        
        star = "★★★" if conviction > 0.6 else ("★★" if conviction > 0.4 else "★")
        
        print(f"  {row.name:2d}. {star} {ticker:6s}  "
              f"Score:{score:.3f}  P:{prob:.0%}  E[ret]:{e_ret:+.1f}%  "
              f"Hold:{horizon}d  Conv:{conviction:.0%}  {rsi_str}")
        
        if show_detail:
            if setups and setups != "none":
                print(f"      Setups: {setups}")
            why = row.get("Why", "")
            if why:
                print(f"      Why:    {why}")
            print(f"      Size hint: {kelly:.0%} of capital  |  "
                  f"Price: ${price:.2f}" if pd.notna(price) else "")
        print()
    
    # Summary
    avg_ev = decisions["Best_EV"].mean()
    avg_prob = decisions["P_BigMove"].mean()
    avg_ret = decisions["Best_E_Return"].mean()
    print(f"  {'─'*100}")
    print(f"  Average:  P={avg_prob:.0%}  E[return]={avg_ret:+.1f}%  "
          f"EV={avg_ev:.2f}%")
    print("=" * 110)
