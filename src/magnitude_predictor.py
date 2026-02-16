"""
Magnitude Predictor — Predict HOW MUCH a stock will move, not just IF.

This is the missing piece. The classifier says "this stock might move 5%+".
The magnitude predictor says "this stock will likely move +8 to +15%".

Combined: Expected Value = P(move) × E(return | move)

Uses XGBoost Regressor trained on:
  - Same 68+ lagged features as the classifier
  - Target: actual next-day return (continuous)
  - Also predicts max return over 1d, 3d, 5d horizons

Key insight: We train SEPARATE models for different holding periods to find
the optimal horizon for each stock.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from src.deep_analyzer import ALL_RAW_FEATURES, add_all_lagged_features
from src.setup_detector import SETUP_NAMES


def _prepare_regression_data(returns: pd.DataFrame, horizon: int = 1):
    """
    Prepare data for regression: predict forward returns at given horizon.
    
    Horizons:
      1 = next day return
      3 = max return over next 3 days
      5 = max return over next 5 days
    """
    df = add_all_lagged_features(returns.copy())
    
    # Create forward return targets per ticker
    if horizon == 1:
        df["Target_Return"] = df.groupby("Ticker")["Daily_Return_Pct"].shift(-1)
    else:
        # Max return over next N days (what you'd capture with perfect exit)
        df["Target_Return"] = df.groupby("Ticker")["Daily_Return_Pct"].transform(
            lambda x: x.shift(-1).rolling(horizon, min_periods=1).max()
        )
    
    # Build feature list
    lagged_features = [f"Prev_{f}" for f in ALL_RAW_FEATURES if f"Prev_{f}" in df.columns]
    setup_features = [f"Prev_{s}" for s in SETUP_NAMES if f"Prev_{s}" in df.columns]
    all_features = lagged_features + setup_features
    
    return df, all_features


def build_magnitude_models(
    returns: pd.DataFrame,
    horizons: list = [1, 3, 5],
    n_splits: int = 5,
) -> dict:
    """
    Build regression models for multiple holding horizons.
    
    Returns dict with models for each horizon + performance metrics.
    This tells us: for a stock flagged today, what's the expected return
    if we hold for 1, 3, or 5 days?
    """
    try:
        from xgboost import XGBRegressor
        use_xgb = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        use_xgb = False
    
    results = {}
    
    for horizon in horizons:
        print(f"\n{'='*70}")
        print(f"MAGNITUDE MODEL — {horizon}-day horizon")
        print(f"{'='*70}")
        
        df, features = _prepare_regression_data(returns, horizon)
        model_df = df[features + ["Target_Return", "Ticker"]].dropna()
        
        if len(model_df) < 500:
            print(f"  Not enough data ({len(model_df)} rows)")
            continue
        
        X = model_df[features]
        y = model_df["Target_Return"]
        
        print(f"  Training on {len(X)} samples, {len(features)} features")
        print(f"  Target stats: mean={y.mean():.3f}%, median={y.median():.3f}%, "
              f"std={y.std():.3f}%")
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_maes = []
        fold_r2s = []
        all_preds = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            if use_xgb:
                model = XGBRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0,
                )
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    min_samples_leaf=20, subsample=0.8, random_state=42,
                )
            
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            fold_maes.append(mae)
            fold_r2s.append(r2)
            
            preds_df = pd.DataFrame({
                "Predicted": y_pred, "Actual": y_test.values,
            }, index=y_test.index)
            all_preds.append(preds_df)
            
            print(f"  Fold {fold+1}: MAE={mae:.3f}%, R²={r2:.3f}")
        
        # Train final model on all data
        scaler_final = StandardScaler()
        X_scaled = scaler_final.fit_transform(X)
        
        if use_xgb:
            final_model = XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            )
        else:
            final_model = GradientBoostingRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=20, subsample=0.8, random_state=42,
            )
        final_model.fit(X_scaled, y)
        
        predictions_df = pd.concat(all_preds)
        avg_mae = np.mean(fold_maes)
        avg_r2 = np.mean(fold_r2s)
        
        # Analyze prediction quality for BIG predicted moves
        big_pred = predictions_df[predictions_df["Predicted"] >= 3.0]
        if len(big_pred) > 0:
            big_accuracy = (big_pred["Actual"] > 0).mean()
            big_avg_actual = big_pred["Actual"].mean()
            print(f"\n  When model predicts ≥3% return:")
            print(f"    {len(big_pred)} predictions, {big_accuracy:.1%} direction correct")
            print(f"    Average actual return: {big_avg_actual:.2f}%")
        
        fi = pd.DataFrame({
            "Feature": features,
            "Importance": final_model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        
        print(f"\n  Average MAE: {avg_mae:.3f}%")
        print(f"  Average R²:  {avg_r2:.3f}")
        
        results[horizon] = {
            "model": final_model,
            "scaler": scaler_final,
            "features": features,
            "mae": avg_mae,
            "r2": avg_r2,
            "predictions": predictions_df,
            "feature_importance": fi,
        }
    
    return results


def predict_returns(
    returns: pd.DataFrame,
    magnitude_models: dict,
) -> pd.DataFrame:
    """
    For each stock, predict expected return at each horizon.
    Returns a DataFrame with columns: Ticker, Pred_1d, Pred_3d, Pred_5d
    """
    df = add_all_lagged_features(returns.copy())
    
    if "Ticker" in df.columns:
        latest = df.groupby("Ticker").last()
    else:
        latest = df.copy()
    
    scored = pd.DataFrame({"Ticker": latest.index})
    scored = scored.set_index("Ticker")
    
    for horizon, model_dict in magnitude_models.items():
        features = model_dict["features"]
        available = latest[features].dropna()
        
        if len(available) == 0:
            scored[f"Pred_{horizon}d"] = np.nan
            continue
        
        X_scaled = model_dict["scaler"].transform(available[features])
        preds = model_dict["model"].predict(X_scaled)
        
        scored.loc[available.index, f"Pred_{horizon}d"] = preds
    
    return scored.reset_index()


def find_optimal_horizon(magnitude_models: dict) -> dict:
    """
    For each stock prediction, determine which holding period maximizes return.
    Returns analysis of which horizon tends to be most profitable.
    """
    analysis = {}
    for horizon, model_dict in magnitude_models.items():
        preds = model_dict["predictions"]
        # When predictions are high, which horizon delivers best?
        analysis[horizon] = {
            "mae": model_dict["mae"],
            "r2": model_dict["r2"],
            "big_pred_accuracy": None,
            "big_pred_avg_return": None,
        }
        big = preds[preds["Predicted"] >= 3.0]
        if len(big) > 10:
            analysis[horizon]["big_pred_accuracy"] = (big["Actual"] > 0).mean()
            analysis[horizon]["big_pred_avg_return"] = big["Actual"].mean()
    
    return analysis
