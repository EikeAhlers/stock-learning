"""
Pattern Discovery — Find repeating patterns that precede big stock moves.

This module uses statistical analysis and machine learning to:
1. Find which pre-move signals are most predictive
2. Build a scoring model to rank today's stocks by "breakout potential"
3. Backtest: would the signals have predicted past movers?
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature columns used for prediction
# IMPORTANT: These must all be LAGGED features (known before market open).
# We explicitly exclude Gap_Pct (same-day data = look-ahead bias).
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "Prev_Vol_Ratio",
    "Prev_Vol_Trend_5d",
    "Prev_Price_Position_20d",
    "Prev_Vol_Compression",
    "Prev_RSI_14",
    "Prev_Return_5d",
    "Prev_Return_10d",
    "Prev_Return_20d",
    "Prev_Dist_SMA20_Pct",
    "Prev_Dist_SMA50_Pct",
    "Prev_Volatility_10d",
    "Prev_Volatility_20d",
    "Prev_Up_Streak",
    "Prev_Down_Streak",
]

# Raw (unlagged) feature names — used to create the lagged versions
RAW_FEATURE_COLS = [
    "Vol_Ratio",
    "Vol_Trend_5d",
    "Price_Position_20d",
    "Vol_Compression",
    "RSI_14",
    "Return_5d",
    "Return_10d",
    "Return_20d",
    "Dist_SMA20_Pct",
    "Dist_SMA50_Pct",
    "Volatility_10d",
    "Volatility_20d",
    "Up_Streak",
    "Down_Streak",
]


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lagged (previous-day) versions of all features.
    This eliminates look-ahead bias: we only use data known
    BEFORE market open on the prediction day.
    """
    df = df.sort_values(["Ticker", df.index.name or "Date"]).copy()
    available_raw = [c for c in RAW_FEATURE_COLS if c in df.columns]
    for col in available_raw:
        df[f"Prev_{col}"] = df.groupby("Ticker")[col].shift(1)
    return df


# ---------------------------------------------------------------------------
# 1. Feature importance analysis (which signals matter most?)
# ---------------------------------------------------------------------------

def analyze_feature_importance(
    all_returns: pd.DataFrame,
    big_move_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Train a model to distinguish "big move days" from "normal days"
    and extract which features are most important.
    
    Uses LAGGED features only (previous day's values) to avoid look-ahead bias.
    """
    df = add_lagged_features(all_returns.copy())
    
    # Label: was this day a big mover?
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= big_move_threshold).astype(int)
    
    # Get available features (lagged only)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    
    # Drop rows with NaN in features
    model_df = df[available_features + ["Is_Big_Mover"]].dropna()
    
    if len(model_df) < 100:
        print("Not enough data for feature importance analysis")
        return pd.DataFrame()
    
    X = model_df[available_features]
    y = model_df["Is_Big_Mover"]
    
    print(f"Dataset: {len(X)} samples, {y.sum()} big movers ({y.mean()*100:.1f}%)")
    
    # Random Forest for feature importance
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",  # Handle class imbalance
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    
    importance = pd.DataFrame({
        "Feature": available_features,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)
    
    importance["Cumulative_Importance"] = importance["Importance"].cumsum()
    importance["Rank"] = range(1, len(importance) + 1)
    
    print("\nFeature Importance Ranking:")
    print("=" * 50)
    for _, row in importance.iterrows():
        bar = "█" * int(row["Importance"] * 50)
        print(f"  {row['Rank']:2d}. {row['Feature']:25s} {row['Importance']:.3f} {bar}")
    
    return importance


# ---------------------------------------------------------------------------
# 2. Backtested prediction model
# ---------------------------------------------------------------------------

def build_prediction_model(
    all_returns: pd.DataFrame,
    big_move_threshold: float = 5.0,
    n_splits: int = 5,
) -> dict:
    """
    Build and backtest a model that predicts big stock moves.
    
    Uses time-series cross-validation to prevent look-ahead bias.
    
    Returns:
        dict with model, performance metrics, and feature importances
    """
    df = add_lagged_features(all_returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= big_move_threshold).astype(int)
    
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    model_df = df[available_features + ["Is_Big_Mover", "Ticker"]].dropna()
    
    X = model_df[available_features]
    y = model_df["Is_Big_Mover"]
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_results = []
    all_predictions = []
    
    print(f"\nTime-Series Cross-Validation ({n_splits} folds)")
    print("=" * 60)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Gradient Boosting model
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = 0.5
        
        accuracy = (y_pred == y_test).mean()
        
        # Precision at different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            preds_at_thresh = (y_pred_proba >= threshold).astype(int)
            tp = ((preds_at_thresh == 1) & (y_test == 1)).sum()
            fp = ((preds_at_thresh == 1) & (y_test == 0)).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(y_test.sum(), 1)
            
            fold_results.append({
                "Fold": fold + 1,
                "Threshold": threshold,
                "AUC": auc,
                "Precision": precision,
                "Recall": recall,
                "Predictions": tp + fp,
                "True_Positives": tp,
            })
        
        print(f"  Fold {fold+1}: AUC={auc:.3f}, "
              f"Test size={len(y_test)}, "
              f"Big movers in test={y_test.sum()}")
        
        # Save predictions 
        preds_df = pd.DataFrame({
            "Probability": y_pred_proba,
            "Actual": y_test.values,
        }, index=y_test.index)
        all_predictions.append(preds_df)
    
    results_df = pd.DataFrame(fold_results)
    predictions_df = pd.concat(all_predictions)
    
    # Train final model on all data
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    final_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    final_model.fit(X_scaled, y)
    
    # Summary
    print("\n\nPerformance Summary by Threshold:")
    print("=" * 60)
    summary = results_df.groupby("Threshold").agg({
        "AUC": "mean",
        "Precision": "mean",
        "Recall": "mean",
        "True_Positives": "sum",
        "Predictions": "sum",
    }).round(3)
    print(summary)
    
    return {
        "model": final_model,
        "scaler": scaler_final,
        "features": available_features,
        "results": results_df,
        "predictions": predictions_df,
        "feature_importance": pd.DataFrame({
            "Feature": available_features,
            "Importance": final_model.feature_importances_,
        }).sort_values("Importance", ascending=False),
    }


# ---------------------------------------------------------------------------
# 3. Score today's stocks (rank by breakout potential)
# ---------------------------------------------------------------------------

def score_current_stocks(
    latest_data: pd.DataFrame,
    model_dict: dict,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Score all stocks using the trained model to identify
    which ones have the highest probability of a big move tomorrow.
    
    Parameters:
        latest_data: Most recent row of data for each stock
        model_dict: Output from build_prediction_model()
        top_n: Number of top candidates to return
    """
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    features = model_dict["features"]
    
    # Add lagged features so we use yesterday's data to predict tomorrow
    data_with_lags = add_lagged_features(latest_data.copy())
    
    # Get latest data for each ticker
    if "Ticker" in data_with_lags.columns:
        latest = data_with_lags.groupby("Ticker").last()
    else:
        latest = data_with_lags.copy()
    
    # Prepare features
    available = latest[features].dropna()
    
    if len(available) == 0:
        print("No stocks with complete data to score")
        return pd.DataFrame()
    
    # Score
    X_scaled = scaler.transform(available[features])
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    scored = pd.DataFrame({
        "Ticker": available.index,
        "Breakout_Probability": probabilities,
    }).sort_values("Breakout_Probability", ascending=False)
    
    # Add lagged feature values (used by model)
    for feat in features:
        scored[feat] = available[feat].values
    
    # Also add raw (latest) feature values for display
    for raw_feat in RAW_FEATURE_COLS:
        if raw_feat in latest.columns:
            scored[raw_feat] = latest.loc[available.index, raw_feat].values
    
    scored = scored.head(top_n).reset_index(drop=True)
    scored.index = range(1, len(scored) + 1)
    scored.index.name = "Rank"
    
    return scored


# ---------------------------------------------------------------------------
# 4. Pattern correlation analysis
# ---------------------------------------------------------------------------

def find_correlated_signals(
    all_returns: pd.DataFrame,
    big_move_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Find which pairs of signals tend to appear together
    before big moves. This reveals multi-factor patterns.
    """
    df = add_lagged_features(all_returns.copy())
    df["Is_Big_Mover"] = (df["Daily_Return_Pct"] >= big_move_threshold).astype(int)
    
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    
    # Correlation of features among big movers
    big_mover_data = df[df["Is_Big_Mover"] == 1][available_features].dropna()
    normal_data = df[df["Is_Big_Mover"] == 0][available_features].dropna()
    
    # Compare distributions
    comparison = pd.DataFrame({
        "Feature": available_features,
        "Big_Mover_Mean": [big_mover_data[c].mean() for c in available_features],
        "Normal_Mean": [normal_data[c].mean() for c in available_features],
        "Big_Mover_Median": [big_mover_data[c].median() for c in available_features],
        "Normal_Median": [normal_data[c].median() for c in available_features],
    })
    
    comparison["Mean_Diff_Pct"] = (
        (comparison["Big_Mover_Mean"] - comparison["Normal_Mean"]) 
        / comparison["Normal_Mean"].abs().clip(lower=0.001) * 100
    ).round(1)
    
    comparison = comparison.sort_values("Mean_Diff_Pct", ascending=False)
    
    print("\nSignal Comparison: Big Movers vs Normal Days")
    print("=" * 70)
    print(f"{'Feature':25s} {'Big Mover Avg':>14s} {'Normal Avg':>12s} {'Diff %':>8s}")
    print("-" * 70)
    for _, row in comparison.iterrows():
        print(f"  {row['Feature']:25s} {row['Big_Mover_Mean']:12.2f} {row['Normal_Mean']:12.2f} {row['Mean_Diff_Pct']:+8.1f}%")
    
    return comparison
