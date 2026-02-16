"""
Deep Analyzer — Advanced ML model with full explainability.

Uses XGBoost (if available, else GradientBoosting) with SHAP values
to explain EXACTLY why each stock is flagged. No more black box.

Also includes:
  - Forensic case study analysis of past big moves
  - Multi-factor setup scoring
  - Historical analog matching
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from src.advanced_features import ADVANCED_RAW_FEATURES
from src.external_data import EXTERNAL_RAW_FEATURES
from src.setup_detector import SETUP_NAMES


# ═══════════════════════════════════════════════════════════════════════════
# Combined feature list — all previous + new features
# ═══════════════════════════════════════════════════════════════════════════

BASE_RAW_FEATURES = [
    "Vol_Ratio", "Vol_Trend_5d", "Price_Position_20d", "Vol_Compression",
    "RSI_14", "Return_5d", "Return_10d", "Return_20d",
    "Dist_SMA20_Pct", "Dist_SMA50_Pct", "Volatility_10d", "Volatility_20d",
    "Up_Streak", "Down_Streak",
]

ALL_RAW_FEATURES = BASE_RAW_FEATURES + ADVANCED_RAW_FEATURES + EXTERNAL_RAW_FEATURES


def get_all_lagged_feature_names(available_cols: list) -> list:
    """Get Prev_ prefixed feature names for all available raw features."""
    return [f"Prev_{f}" for f in ALL_RAW_FEATURES if f in available_cols or f"Prev_{f}" in available_cols]


def add_all_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Prev_ versions of ALL features (basic + advanced + external)."""
    df = df.sort_values(["Ticker", df.index.name or "Date"]).copy()
    for col in ALL_RAW_FEATURES:
        if col in df.columns:
            df[f"Prev_{col}"] = df.groupby("Ticker")[col].shift(1)
    # Also lag setup indicators
    for col in SETUP_NAMES:
        if col in df.columns:
            df[f"Prev_{col}"] = df.groupby("Ticker")[col].shift(1)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# XGBoost model with SHAP
# ═══════════════════════════════════════════════════════════════════════════

def build_advanced_model(
    all_returns: pd.DataFrame,
    big_move_threshold: float = 5.0,
    n_splits: int = 5,
    use_setups: bool = True,
) -> dict:
    """
    Build an advanced prediction model using all available features.
    
    Steps:
    1. Create lagged features (no look-ahead)
    2. Train XGBoost (or GBM fallback) with time-series CV
    3. Compute SHAP values for explainability
    4. Return model + explanations
    """
    # Try XGBoost, fall back to sklearn GBM
    try:
        from xgboost import XGBClassifier
        use_xgb = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        use_xgb = False
        print("XGBoost not available, using sklearn GradientBoosting")
    
    df = add_all_lagged_features(all_returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= big_move_threshold).astype(int)
    
    # Build feature list from what's actually available
    lagged_features = [f"Prev_{f}" for f in ALL_RAW_FEATURES if f"Prev_{f}" in df.columns]
    if use_setups:
        setup_features = [f"Prev_{s}" for s in SETUP_NAMES if f"Prev_{s}" in df.columns]
        lagged_features += setup_features
    
    if not lagged_features:
        print("ERROR: No lagged features available!")
        return {}
    
    model_df = df[lagged_features + ["Is_Big_Mover", "Ticker"]].dropna()
    
    if len(model_df) < 500:
        print(f"Not enough data ({len(model_df)} rows). Need at least 500.")
        return {}
    
    X = model_df[lagged_features]
    y = model_df["Is_Big_Mover"]
    
    print(f"Training with {len(lagged_features)} features on {len(X)} samples "
          f"({y.sum()} big movers = {y.mean()*100:.2f}%)")
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    all_predictions = []
    
    print(f"\nTime-Series Cross-Validation ({n_splits} folds)")
    print("=" * 70)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        if use_xgb:
            # Calculate scale_pos_weight for class imbalance
            neg = (y_train == 0).sum()
            pos = max((y_train == 1).sum(), 1)
            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=20,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=neg / pos,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=20,
                subsample=0.8,
                random_state=42,
            )
        
        model.fit(X_train_s, y_train)
        y_pred_proba = model.predict_proba(X_test_s)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = 0.5
        
        # Precision at various thresholds
        for threshold in [0.2, 0.3, 0.5, 0.7]:
            preds = (y_pred_proba >= threshold).astype(int)
            tp = ((preds == 1) & (y_test == 1)).sum()
            fp = ((preds == 1) & (y_test == 0)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(y_test.sum(), 1)
            
            fold_results.append({
                "Fold": fold + 1, "Threshold": threshold,
                "AUC": auc, "Precision": prec, "Recall": rec,
                "True_Positives": tp, "Predictions": tp + fp,
            })
        
        print(f"  Fold {fold+1}: AUC={auc:.3f}, Test={len(y_test)}, "
              f"BigMovers={y_test.sum()}")
        
        preds_df = pd.DataFrame({
            "Probability": y_pred_proba,
            "Actual": y_test.values,
        }, index=y_test.index)
        all_predictions.append(preds_df)
    
    results_df = pd.DataFrame(fold_results)
    predictions_df = pd.concat(all_predictions)
    
    # Train final model on ALL data
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    if use_xgb:
        neg = (y == 0).sum()
        pos = max((y == 1).sum(), 1)
        final_model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / pos, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
    else:
        final_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42,
        )
    final_model.fit(X_scaled, y)
    
    # Feature importance
    fi = pd.DataFrame({
        "Feature": lagged_features,
        "Importance": final_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    
    # SHAP values (if available)
    shap_values = None
    shap_explainer = None
    try:
        import shap
        if use_xgb:
            shap_explainer = shap.TreeExplainer(final_model)
        else:
            # For GBM, use a subset for speed
            sample_idx = np.random.choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)
            shap_explainer = shap.TreeExplainer(final_model)
        
        # Compute SHAP on a sample
        sample_size = min(5000, len(X_scaled))
        sample_X = X_scaled[:sample_size]
        shap_values = shap_explainer.shap_values(sample_X)
        print(f"\nSHAP values computed on {sample_size} samples")
    except ImportError:
        print("\nInstall 'shap' package for model explainability: pip install shap")
    except Exception as e:
        print(f"\nSHAP computation failed: {e}")
    
    # Summary stats
    avg_auc = results_df.groupby("Threshold")["AUC"].mean().iloc[0]
    print(f"\n{'='*70}")
    print(f"Average AUC: {avg_auc:.3f}")
    print(f"Features used: {len(lagged_features)}")
    
    summary = results_df.groupby("Threshold").agg({
        "AUC": "mean", "Precision": "mean", "Recall": "mean",
        "True_Positives": "sum", "Predictions": "sum",
    }).round(3)
    print(f"\nPerformance by Threshold:")
    print(summary)
    print("=" * 70)
    
    return {
        "model": final_model,
        "scaler": scaler_final,
        "features": lagged_features,
        "results": results_df,
        "predictions": predictions_df,
        "feature_importance": fi,
        "shap_explainer": shap_explainer,
        "shap_values": shap_values,
        "use_xgb": use_xgb,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Score today's stocks with SHAP explanations
# ═══════════════════════════════════════════════════════════════════════════

def score_stocks_with_explanations(
    returns: pd.DataFrame,
    model_dict: dict,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Score all stocks and provide SHAP-based explanations for WHY
    each stock is flagged. No more black box.
    """
    if not model_dict:
        return pd.DataFrame()
    
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    features = model_dict["features"]
    
    data = add_all_lagged_features(returns.copy())
    
    if "Ticker" in data.columns:
        latest = data.groupby("Ticker").last()
    else:
        latest = data.copy()
    
    available = latest[features].dropna()
    if len(available) == 0:
        print("No stocks with complete data")
        return pd.DataFrame()
    
    X_scaled = scaler.transform(available[features])
    probs = model.predict_proba(X_scaled)[:, 1]
    
    scored = pd.DataFrame({
        "Ticker": available.index,
        "Breakout_Probability": probs,
    }).sort_values("Breakout_Probability", ascending=False)
    
    # Add raw feature values for context
    for f in BASE_RAW_FEATURES + ADVANCED_RAW_FEATURES[:10]:
        if f in latest.columns:
            scored[f] = latest.loc[available.index, f].values
    
    # Add setup indicators
    for s in SETUP_NAMES:
        if s in latest.columns:
            scored[s] = latest.loc[available.index, s].values
    
    # Build setup summary string
    setup_cols_available = [s for s in SETUP_NAMES if s in scored.columns]
    if setup_cols_available:
        scored["Active_Setups"] = scored[setup_cols_available].apply(
            lambda row: ", ".join([c.replace("Setup_", "") for c in setup_cols_available if row[c] == 1]) or "none",
            axis=1
        )
    
    # SHAP explanations for top picks
    if model_dict.get("shap_explainer") is not None:
        try:
            import shap
            explainer = model_dict["shap_explainer"]
            top_indices = scored.head(top_n).index
            
            explanations = []
            for i, (_, row) in enumerate(scored.head(top_n).iterrows()):
                ticker = row["Ticker"]
                if ticker not in available.index:
                    continue
                
                x = available.loc[[ticker]][features]
                x_scaled = scaler.transform(x)
                sv = explainer.shap_values(x_scaled)
                
                if isinstance(sv, list):
                    sv = sv[1]  # For binary classification
                
                # Top 5 drivers for this prediction
                feature_impacts = pd.DataFrame({
                    "Feature": features,
                    "SHAP_Value": sv[0],
                }).sort_values("SHAP_Value", key=abs, ascending=False)
                
                top_drivers = feature_impacts.head(5)
                driver_str = " | ".join([
                    f"{r['Feature'].replace('Prev_', '')} ({'+' if r['SHAP_Value'] > 0 else ''}{r['SHAP_Value']:.3f})"
                    for _, r in top_drivers.iterrows()
                ])
                explanations.append(driver_str)
            
            scored = scored.head(top_n).copy()
            scored["Why"] = explanations[:len(scored)]
        except Exception as e:
            scored = scored.head(top_n).copy()
            scored["Why"] = "SHAP unavailable"
    else:
        scored = scored.head(top_n).copy()
        scored["Why"] = "Install shap for explanations"
    
    scored = scored.reset_index(drop=True)
    scored.index = range(1, len(scored) + 1)
    scored.index.name = "Rank"
    
    return scored


# ═══════════════════════════════════════════════════════════════════════════
# Forensic Case Study — Deep dive into specific past big moves
# ═══════════════════════════════════════════════════════════════════════════

def forensic_case_study(
    returns: pd.DataFrame,
    ticker: str,
    move_date: str,
    lookback_days: int = 10,
) -> dict:
    """
    Deep forensic analysis of a single big move event.
    Shows exactly what was observable in the days leading up to it.
    
    Returns a detailed dict with day-by-day pre-move conditions.
    """
    move_date = pd.Timestamp(move_date)
    
    ticker_data = returns[returns["Ticker"] == ticker].sort_index()
    
    # Get the move day
    move_idx = ticker_data.index.get_indexer([move_date], method="nearest")[0]
    
    if move_idx < lookback_days:
        print(f"Not enough history before {move_date}")
        return {}
    
    move_row = ticker_data.iloc[move_idx]
    pre_move = ticker_data.iloc[move_idx - lookback_days:move_idx]
    
    # Key features to track day-by-day
    key_features = [
        "Close", "Volume", "Vol_Ratio", "RSI_14", "Daily_Return_Pct",
        "OBV_Slope_10d", "AD_Line_Slope_10d", "BB_Width", "MFI",
        "Vol_ZScore", "Quiet_Accum_Days_10", "Close_Location",
    ]
    available = [f for f in key_features if f in ticker_data.columns]
    
    # Day-by-day summary
    daily = pre_move[available].copy()
    daily["Day"] = range(-lookback_days, 0)
    
    # Setups active
    setup_cols = [s for s in SETUP_NAMES if s in ticker_data.columns]
    active_setups_pre = {}
    for s in setup_cols:
        if pre_move[s].sum() > 0:
            active_setups_pre[s.replace("Setup_", "")] = int(pre_move[s].sum())
    
    return {
        "ticker": ticker,
        "move_date": str(move_date.date()),
        "move_return": float(move_row.get("Daily_Return_Pct", 0)),
        "move_volume_ratio": float(move_row.get("Vol_Ratio", 1)),
        "gap_pct": float(move_row.get("Gap_Pct", 0)),
        "pre_move_daily": daily,
        "setups_active_before": active_setups_pre,
        "pre_move_avg_vol_ratio": float(pre_move.get("Vol_Ratio", pd.Series(1)).mean()),
        "pre_move_avg_rsi": float(pre_move.get("RSI_14", pd.Series(50)).mean()),
        "pre_move_obv_trend": "RISING" if pre_move.get("OBV_Slope_10d", pd.Series(0)).iloc[-1] > 0 else "FALLING",
        "pre_move_ad_trend": "ACCUMULATION" if pre_move.get("AD_Line_Slope_10d", pd.Series(0)).iloc[-1] > 0 else "DISTRIBUTION",
        "pre_move_mfi_avg": float(pre_move.get("MFI", pd.Series(50)).mean()),
    }


def print_case_study(case: dict):
    """Pretty-print a forensic case study."""
    if not case:
        print("No data")
        return
    
    print(f"\n{'='*70}")
    print(f"FORENSIC ANALYSIS: {case['ticker']} on {case['move_date']}")
    print(f"{'='*70}")
    print(f"  Move: +{case['move_return']:.1f}%  |  Gap: {case['gap_pct']:.1f}%  |  "
          f"Volume: {case['move_volume_ratio']:.1f}x avg")
    
    print(f"\n  PRE-MOVE FINGERPRINT (10 days before):")
    print(f"    Avg Volume Ratio: {case['pre_move_avg_vol_ratio']:.2f}x")
    print(f"    Avg RSI:          {case['pre_move_avg_rsi']:.0f}")
    print(f"    OBV Trend:        {case['pre_move_obv_trend']}")
    print(f"    A/D Line:         {case['pre_move_ad_trend']}")
    print(f"    Avg MFI:          {case['pre_move_mfi_avg']:.0f}")
    
    if case['setups_active_before']:
        print(f"\n  SETUPS DETECTED BEFORE MOVE:")
        for setup, count in case['setups_active_before'].items():
            print(f"    - {setup} (active on {count} of 10 pre-move days)")
    else:
        print(f"\n  NO NAMED SETUPS DETECTED (move came without typical setup)")
    
    if "pre_move_daily" in case and len(case["pre_move_daily"]) > 0:
        print(f"\n  DAY-BY-DAY LEADING UP TO MOVE:")
        daily = case["pre_move_daily"]
        display = daily[["Day"] + [c for c in ["Close", "Volume", "Vol_Ratio", "RSI_14", "Daily_Return_Pct"] if c in daily.columns]]
        for _, row in display.iterrows():
            day = int(row["Day"])
            parts = [f"  Day {day:+3d}:"]
            if "Daily_Return_Pct" in row:
                parts.append(f"Ret={row['Daily_Return_Pct']:+5.1f}%")
            if "Vol_Ratio" in row:
                parts.append(f"Vol={row['Vol_Ratio']:.2f}x")
            if "RSI_14" in row:
                parts.append(f"RSI={row['RSI_14']:.0f}")
            print("  ".join(parts))
    
    print(f"{'='*70}")
