import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
DATA_DIR = Path('../data')
MODELS_DIR = Path('../models')
ENRICHED_CSV_PATH = DATA_DIR / 'oak_wilt_cluster_enriched.csv'

# Ensure models dir exists
MODELS_DIR.mkdir(exist_ok=True)

def main():
    print("Loading data...")
    df = pd.read_csv(ENRICHED_CSV_PATH)
    
    # Feature Engineering
    print("Feature Engineering...")
    df['log_density'] = np.log1p(df['point_density'])
    df['log_duration'] = np.log1p(df['duration_days'])
    
    feature_cols = [
        'log_duration',
        'log_density',
        'avg_precip',
        'avg_humidity',
        'avg_wind'
    ]
    
    # Determine target column
    target_col = 'radius_ft' 
    if target_col not in df.columns:
        if 'radius_m' in df.columns:
            target_col = 'radius_m'
        else:
            raise ValueError("Could not find radius target column (radius_ft or radius_m)")
            
    print(f"Using target column: {target_col}")

    X = df[feature_cols]
    y = df[target_col]
    
    # Handle NaNs
    print("Handling NaNs...")
    data_combined = pd.concat([X, y], axis=1)
    initial_len = len(data_combined)
    data_combined = data_combined.dropna()
    print(f"Dropped {initial_len - len(data_combined)} rows with missing values.")
    
    X = data_combined[feature_cols]
    y = data_combined[target_col]

    # Split for Conformal Prediction (Train vs Calibration)
    X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Gradient Boosting Models (Train: {len(X_train)}, Calib: {len(X_calib)})...")
    
    # 1. Lower Quantile (10th percentile)
    model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=100, random_state=42)
    model_lower.fit(X_train, y_train)
    
    # 2. Median (50th percentile - Expected)
    model_median = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100, random_state=42)
    model_median.fit(X_train, y_train)
    
    # 3. Upper Quantile (90th percentile)
    model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=100, random_state=42)
    model_upper.fit(X_train, y_train)
    
    print("Calculating Conformal Prediction Scores (CQR)...")
    pred_lower_calib = model_lower.predict(X_calib)
    pred_upper_calib = model_upper.predict(X_calib)
    
    scores = np.maximum(pred_lower_calib - y_calib, y_calib - pred_upper_calib)
    
    alpha = 0.2 # 20% allowed error
    q_hat = np.quantile(scores, 1 - alpha)
    
    print(f"Conformal Correction Factor (q_hat): {q_hat:.4f}")
    
    # Save Models and q_hat
    print("Saving models...")
    joblib.dump(model_lower, MODELS_DIR / 'gbr_lower.pkl')
    joblib.dump(model_median, MODELS_DIR / 'gbr_median.pkl')
    joblib.dump(model_upper, MODELS_DIR / 'gbr_upper.pkl')
    joblib.dump(q_hat, MODELS_DIR / 'conformal_q.pkl')
    
    print("Done.")

if __name__ == "__main__":
    main()
