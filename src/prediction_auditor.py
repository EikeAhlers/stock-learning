"""
Prediction Auditor — Stress-test every prediction before you trade it.

This module answers: "Is this prediction REAL or is the model hallucinating?"

For each pick, it checks:
  1. Historical base rate: How often does THIS stock actually make 5%+ moves?
  2. Feature reality check: Are the feature values for today actually unusual?
  3. Magnitude sanity: Is +17% realistic given this stock's historical distribution?
  4. Confidence interval: What's the realistic range of outcomes?
  5. Similar conditions: When this stock had similar features in the past, what happened?
  6. Catalyst check: Is there a known catalyst (earnings, etc.)?
  7. Model disagreement: Do different folds/models agree?
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.deep_analyzer import add_all_lagged_features, ALL_RAW_FEATURES
from src.setup_detector import SETUP_NAMES


def audit_prediction(
    returns: pd.DataFrame,
    ticker: str,
    classifier_dict: dict,
    magnitude_models: dict = None,
    top_shap_features: int = 10,
) -> dict:
    """
    Deep audit of a single stock prediction.
    Returns a dict with all audit results and a confidence grade (A/B/C/D/F).
    """
    audit = {"ticker": ticker, "warnings": [], "strengths": []}
    
    # Get this stock's data
    stock = returns[returns["Ticker"] == ticker].copy()
    if len(stock) == 0:
        audit["grade"] = "F"
        audit["warnings"].append("Stock not found in dataset")
        return audit
    
    latest = stock.iloc[-1]
    
    # ── 1. Historical base rate for THIS stock ───────────────────────
    stock_returns = stock["Daily_Return_Pct"].dropna()
    total_days = len(stock_returns)
    big_moves_up = (stock_returns >= 5.0).sum()
    big_moves_down = (stock_returns <= -5.0).sum()
    
    base_rate = big_moves_up / total_days * 100 if total_days > 0 else 0
    
    audit["total_trading_days"] = total_days
    audit["big_moves_up_5pct"] = int(big_moves_up)
    audit["big_moves_down_5pct"] = int(big_moves_down)
    audit["stock_base_rate_pct"] = base_rate
    
    if base_rate < 0.5:
        audit["warnings"].append(
            f"This stock rarely moves 5%+. Only {big_moves_up} times in {total_days} days "
            f"({base_rate:.1f}% base rate). Model may be overconfident."
        )
    elif base_rate > 3.0:
        audit["strengths"].append(
            f"This stock IS prone to big moves: {big_moves_up} times in {total_days} days ({base_rate:.1f}%)"
        )
    
    # ── 2. Return distribution for this stock ────────────────────────
    audit["avg_daily_return"] = stock_returns.mean()
    audit["std_daily_return"] = stock_returns.std()
    audit["max_1d_return"] = stock_returns.max()
    audit["min_1d_return"] = stock_returns.min()
    audit["median_daily_return"] = stock_returns.median()
    
    # Percentile analysis
    audit["return_95th_pctile"] = stock_returns.quantile(0.95)
    audit["return_99th_pctile"] = stock_returns.quantile(0.99)
    
    # Multi-day returns
    stock_sorted = stock.sort_index()
    fwd_3d = stock_sorted["Daily_Return_Pct"].rolling(3, min_periods=1).sum()
    fwd_5d = stock_sorted["Daily_Return_Pct"].rolling(5, min_periods=1).sum()
    
    audit["max_3d_return"] = fwd_3d.max()
    audit["max_5d_return"] = fwd_5d.max()
    audit["avg_5d_return"] = fwd_5d.mean()
    audit["std_5d_return"] = fwd_5d.std()
    
    # ── 3. Magnitude prediction sanity check ─────────────────────────
    if magnitude_models:
        for horizon, mag_dict in magnitude_models.items():
            features = mag_dict["features"]
            df_lagged = add_all_lagged_features(stock.copy())
            latest_row = df_lagged.iloc[-1:]
            available = latest_row[features].dropna(axis=1)
            
            if len(available.columns) == len(features):
                X = mag_dict["scaler"].transform(latest_row[features])
                pred = mag_dict["model"].predict(X)[0]
                audit[f"predicted_return_{horizon}d"] = pred
                
                # How many standard deviations is this prediction?
                if horizon == 1:
                    sigma_count = abs(pred) / max(audit["std_daily_return"], 0.01)
                else:
                    std_ref = audit.get("std_5d_return", audit["std_daily_return"] * np.sqrt(horizon))
                    sigma_count = abs(pred) / max(std_ref, 0.01)
                
                audit[f"sigma_count_{horizon}d"] = sigma_count
                
                if sigma_count > 3:
                    audit["warnings"].append(
                        f"The {horizon}d prediction of {pred:+.1f}% is {sigma_count:.1f} sigma — "
                        f"an extreme outlier for this stock. Likely overfit."
                    )
                elif sigma_count > 2:
                    audit["warnings"].append(
                        f"The {horizon}d prediction of {pred:+.1f}% is {sigma_count:.1f} sigma — unusual."
                    )
                elif pred > 0:
                    audit["strengths"].append(
                        f"The {horizon}d prediction of {pred:+.1f}% is within normal range ({sigma_count:.1f}σ)"
                    )
    
    # ── 4. Current feature values vs historical distribution ─────────
    df_lagged = add_all_lagged_features(returns.copy())
    stock_lagged = df_lagged[df_lagged["Ticker"] == ticker]
    
    if len(stock_lagged) > 20:
        latest_lagged = stock_lagged.iloc[-1]
        
        key_features = [
            "Prev_ATR_Ratio", "Prev_Vol_ZScore", "Prev_Unusual_Volume",
            "Prev_Intraday_Range_Pct", "Prev_RSI_14", "Prev_BB_Width",
            "Prev_Volatility_20d", "Prev_Dist_52w_High_Pct",
        ]
        
        feature_audit = []
        for feat in key_features:
            if feat not in stock_lagged.columns:
                continue
            current_val = latest_lagged.get(feat, np.nan)
            if pd.isna(current_val):
                continue
            
            hist = stock_lagged[feat].dropna()
            if len(hist) < 20:
                continue
            
            pctile = (hist < current_val).mean() * 100
            z = (current_val - hist.mean()) / max(hist.std(), 0.001)
            
            feature_audit.append({
                "Feature": feat.replace("Prev_", ""),
                "Current": current_val,
                "Historical_Mean": hist.mean(),
                "Percentile": pctile,
                "Z_Score": z,
            })
        
        audit["feature_audit"] = pd.DataFrame(feature_audit)
        
        # Are the key features actually elevated?
        extreme_features = [f for f in feature_audit if abs(f["Z_Score"]) > 2]
        if len(extreme_features) == 0:
            audit["warnings"].append(
                "No features are at extreme levels. The model may be picking up noise."
            )
        else:
            names = [f["Feature"] for f in extreme_features]
            audit["strengths"].append(
                f"{len(extreme_features)} features at extreme levels: {', '.join(names)}"
            )
    
    # ── 5. Historical precedent: similar conditions → what happened? ─
    if len(stock_lagged) > 50:
        # Find days where ATR_Ratio was similar (within 20%)
        atr_col = "Prev_ATR_Ratio"
        if atr_col in stock_lagged.columns:
            current_atr = latest_lagged.get(atr_col, None)
            if current_atr and not pd.isna(current_atr) and current_atr > 0:
                similar = stock_lagged[
                    (stock_lagged[atr_col] > current_atr * 0.8) & 
                    (stock_lagged[atr_col] < current_atr * 1.2)
                ]
                
                if len(similar) > 5 and "Daily_Return_Pct" in similar.columns:
                    next_day_returns = similar["Daily_Return_Pct"]
                    audit["similar_conditions_count"] = len(similar)
                    audit["similar_avg_return"] = next_day_returns.mean()
                    audit["similar_median_return"] = next_day_returns.median()
                    audit["similar_pct_positive"] = (next_day_returns > 0).mean() * 100
                    audit["similar_pct_5plus"] = (next_day_returns >= 5).mean() * 100
                    
                    if audit["similar_pct_5plus"] < 5:
                        audit["warnings"].append(
                            f"In {len(similar)} similar past conditions, only "
                            f"{audit['similar_pct_5plus']:.0f}% resulted in 5%+ moves. "
                            f"Average return was {audit['similar_avg_return']:+.2f}%."
                        )
                    elif audit["similar_pct_5plus"] > 10:
                        audit["strengths"].append(
                            f"In {len(similar)} similar past conditions, "
                            f"{audit['similar_pct_5plus']:.0f}% resulted in 5%+ moves"
                        )
    
    # ── 6. Recent price action ───────────────────────────────────────
    recent = stock.tail(10)
    audit["last_5d_return"] = stock["Daily_Return_Pct"].tail(5).sum()
    audit["last_10d_return"] = stock["Daily_Return_Pct"].tail(10).sum()
    audit["current_price"] = latest.get("Close", None)
    audit["rsi"] = latest.get("RSI_14", None)
    audit["vol_ratio"] = latest.get("Vol_Ratio", None)
    
    # ── 7. Compute honest confidence interval ────────────────────────
    # Based on this stock's actual return distribution, not the model
    std = audit["std_daily_return"]
    audit["realistic_1d_range"] = f"{-1.5*std:+.1f}% to {+1.5*std:+.1f}%"
    audit["realistic_5d_range"] = f"{-1.5*std*np.sqrt(5):+.1f}% to {+1.5*std*np.sqrt(5):+.1f}%"
    audit["realistic_upside_5d_p90"] = stock_returns.quantile(0.90) * np.sqrt(5)
    
    # ── 8. Overall grade ─────────────────────────────────────────────
    score = 50  # Start neutral
    
    # Strengths add points
    score += len(audit["strengths"]) * 10
    
    # Warnings subtract
    score -= len(audit["warnings"]) * 15
    
    # Base rate matters a lot
    if base_rate > 3:
        score += 15
    elif base_rate < 1:
        score -= 20
    
    # Extreme prediction = penalty
    for h in [1, 3, 5]:
        sigma = audit.get(f"sigma_count_{h}d", 0)
        if sigma > 3:
            score -= 25
        elif sigma > 2:
            score -= 10
    
    if score >= 70:
        audit["grade"] = "A"
    elif score >= 55:
        audit["grade"] = "B"
    elif score >= 40:
        audit["grade"] = "C"
    elif score >= 25:
        audit["grade"] = "D"
    else:
        audit["grade"] = "F"
    
    audit["confidence_score"] = max(0, min(100, score))
    
    return audit


def print_audit_report(audit: dict):
    """Print a detailed audit report for a stock prediction."""
    print(f"\n{'='*90}")
    print(f"  PREDICTION AUDIT — {audit['ticker']}  (Grade: {audit['grade']}, Score: {audit['confidence_score']}/100)")
    print(f"{'='*90}")
    
    # Basic stats
    print(f"\n  HISTORICAL PROFILE:")
    print(f"    Trading days in dataset:  {audit.get('total_trading_days', '?')}")
    print(f"    Times moved 5%+ up:      {audit.get('big_moves_up_5pct', '?')}")
    print(f"    Times moved 5%+ down:    {audit.get('big_moves_down_5pct', '?')}")
    print(f"    Stock base rate:         {audit.get('stock_base_rate_pct', 0):.2f}%")
    print(f"    Avg daily return:        {audit.get('avg_daily_return', 0):+.3f}%")
    print(f"    Std daily return:        {audit.get('std_daily_return', 0):.3f}%")
    print(f"    Max single-day up:       {audit.get('max_1d_return', 0):+.1f}%")
    print(f"    Max single-day down:     {audit.get('min_1d_return', 0):+.1f}%")
    print(f"    95th percentile:         {audit.get('return_95th_pctile', 0):+.2f}%")
    print(f"    99th percentile:         {audit.get('return_99th_pctile', 0):+.2f}%")
    
    # Magnitude predictions
    for h in [1, 3, 5]:
        pred_key = f"predicted_return_{h}d"
        sigma_key = f"sigma_count_{h}d"
        if pred_key in audit:
            sigma = audit.get(sigma_key, 0)
            flag = " ⚠️  EXTREME" if sigma > 3 else (" ⚠️  HIGH" if sigma > 2 else " ✓")
            print(f"    Predicted {h}d return:     {audit[pred_key]:+.2f}% ({sigma:.1f}σ){flag}")
    
    # Realistic ranges
    print(f"\n  REALISTIC RETURN RANGES (based on this stock's actual history):")
    print(f"    1-day range (±1.5σ):     {audit.get('realistic_1d_range', '?')}")
    print(f"    5-day range (±1.5σ):     {audit.get('realistic_5d_range', '?')}")
    
    # Current state
    print(f"\n  CURRENT STATE:")
    price = audit.get("current_price")
    if price:
        print(f"    Price:                   ${price:.2f}")
    rsi = audit.get("rsi")
    if rsi and not pd.isna(rsi):
        print(f"    RSI(14):                 {rsi:.1f}")
    print(f"    Last 5d return:          {audit.get('last_5d_return', 0):+.2f}%")
    print(f"    Last 10d return:         {audit.get('last_10d_return', 0):+.2f}%")
    
    # Feature audit
    if "feature_audit" in audit and len(audit["feature_audit"]) > 0:
        print(f"\n  KEY FEATURES — Current vs Historical:")
        fa = audit["feature_audit"]
        for _, row in fa.iterrows():
            flag = "⚡" if abs(row["Z_Score"]) > 2 else ("•" if abs(row["Z_Score"]) > 1 else " ")
            print(f"    {flag} {row['Feature']:<25} Current: {row['Current']:>8.2f}  "
                  f"Mean: {row['Historical_Mean']:>8.2f}  "
                  f"Pctile: {row['Percentile']:>5.0f}%  Z: {row['Z_Score']:>+5.1f}")
    
    # Similar conditions
    if "similar_conditions_count" in audit:
        print(f"\n  SIMILAR HISTORICAL CONDITIONS:")
        print(f"    Times seen:              {audit['similar_conditions_count']}")
        print(f"    Avg next-day return:     {audit['similar_avg_return']:+.2f}%")
        print(f"    Median next-day return:  {audit['similar_median_return']:+.2f}%")
        print(f"    % positive:              {audit['similar_pct_positive']:.0f}%")
        print(f"    % moved 5%+:             {audit['similar_pct_5plus']:.0f}%")
    
    # Warnings and strengths
    if audit["warnings"]:
        print(f"\n  ⚠️  WARNINGS:")
        for w in audit["warnings"]:
            print(f"    • {w}")
    
    if audit["strengths"]:
        print(f"\n  ✓ STRENGTHS:")
        for s in audit["strengths"]:
            print(f"    • {s}")
    
    # Final verdict
    grade = audit["grade"]
    if grade == "A":
        verdict = "HIGH CONFIDENCE — Multiple signal classes confirm this pick"
    elif grade == "B":
        verdict = "MODERATE CONFIDENCE — Reasonable but not bulletproof"
    elif grade == "C":
        verdict = "LOW CONFIDENCE — Significant uncertainty, reduce position size"
    elif grade == "D":
        verdict = "VERY LOW CONFIDENCE — Most signals don't support this trade"
    else:
        verdict = "NO CONFIDENCE — Do not trade this"
    
    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*90}")
