import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

# Paths
DATA_DIR = Path('../data')
MODELS_DIR = Path('../models')
VISUALS_DIR = Path('../visuals')
CLUSTERS_JS_PATH = DATA_DIR / 'oak_wilt_clusters.js'
ENRICHED_CSV_PATH = DATA_DIR / 'oak_wilt_cluster_enriched.csv'
OUTPUT_JS_PATH = VISUALS_DIR / 'dashboard_data.js'

# Model Paths
MODEL_LOWER_PATH = MODELS_DIR / 'gbr_lower.pkl'
MODEL_MEDIAN_PATH = MODELS_DIR / 'gbr_median.pkl'
MODEL_UPPER_PATH = MODELS_DIR / 'gbr_upper.pkl'
CONFORMAL_Q_PATH = MODELS_DIR / 'conformal_q.pkl'

def load_clusters_js(path):
    with open(path, 'r') as f:
        content = f.read()
        # Find the start of the JSON object
        start_idx = content.find('{')
        if start_idx == -1:
            raise ValueError("Could not find JSON object in file")
        
        # Find the last '}'
        end_idx = content.rfind('}')
        if end_idx == -1:
            raise ValueError("Could not find end of JSON object in file")
            
        json_str = content[start_idx:end_idx+1]
        return json.loads(json_str)

def main():
    print("Loading data...")
    clusters_data = load_clusters_js(CLUSTERS_JS_PATH)
    enriched_df = pd.read_csv(ENRICHED_CSV_PATH)
    
    print("Loading models...")
    model_lower = joblib.load(MODEL_LOWER_PATH)
    model_median = joblib.load(MODEL_MEDIAN_PATH)
    model_upper = joblib.load(MODEL_UPPER_PATH)
    q_hat = joblib.load(CONFORMAL_Q_PATH)
    
    print(f"Loaded Conformal Correction Factor (q_hat): {q_hat:.4f}")
    
    # Prepare features for prediction
    print("Preparing features...")
    enriched_df['log_density'] = np.log1p(enriched_df['point_density'])
    enriched_df['log_duration'] = np.log1p(enriched_df['duration_days'])
    
    feature_cols = [
        'log_duration',
        'log_density',
        'avg_precip',
        'avg_humidity',
        'avg_wind'
    ]
    
    # Handle NaNs in features for prediction (fill with mean or 0, or drop?)
    # Since we dropped them in training, we should probably drop them here or fill.
    # For dashboard generation, we want predictions for all clusters if possible.
    # Let's fill with mean of the column to avoid dropping clusters from the dashboard.
    X = enriched_df[feature_cols].copy()
    for col in X.columns:
        if X[col].isnull().any():
            print(f"Warning: Filling NaNs in {col} with mean")
            X[col] = X[col].fillna(X[col].mean())
            
    print("Running inference...")
    pred_lower = model_lower.predict(X)
    pred_median = model_median.predict(X)
    pred_upper = model_upper.predict(X)
    
    # Apply Conformal Prediction Correction
    # Interval: [lower - q, upper + q]
    # We clamp lower bound to 0
    final_lower = np.maximum(0, pred_lower - q_hat)
    final_upper = pred_upper + q_hat
    final_median = pred_median # Median stays as point estimate
    
    # Add predictions to dataframe
    enriched_df['pred_radius_10'] = final_lower
    enriched_df['pred_radius_50'] = final_median
    enriched_df['pred_radius_90'] = final_upper
    
    # Create a lookup dictionary for enriched data by cluster_id
    enriched_lookup = enriched_df.set_index('cluster_id').to_dict('index')
    
    print("Merging data...")
    # Update the clusters_data structure
    for cluster in clusters_data['clusters']:
        c_id = cluster['cluster_id']
        if c_id in enriched_lookup:
            data = enriched_lookup[c_id]
            
            # Add Environmental Data
            cluster['environment'] = {
                'avg_precip': round(data['avg_precip'], 4) if pd.notnull(data['avg_precip']) else 0,
                'avg_humidity': round(data['avg_humidity'], 2) if pd.notnull(data['avg_humidity']) else 0,
                'avg_wind': round(data['avg_wind'], 2) if pd.notnull(data['avg_wind']) else 0
            }
            
            # Add Predictions
            # Check for NaNs in predictions or actuals
            actual_radius = data.get('radius_ft')
            if pd.isna(actual_radius):
                actual_radius = 0 # Or handle appropriately
            
            cluster['predictions'] = {
                'radius_10_conservative': round(float(data['pred_radius_10']), 2),
                'radius_50_expected': round(float(data['pred_radius_50']), 2),
                'radius_90_severe': round(float(data['pred_radius_90']), 2),
                'actual_radius': round(float(actual_radius), 2)
            }
            
            # Add derived metrics
            cluster['metrics'] = {
                'density': round(data['point_density'], 6) if pd.notnull(data['point_density']) else 0,
                'duration_days': int(data['duration_days']) if pd.notnull(data['duration_days']) else 0
            }
        else:
            print(f"Warning: Cluster {c_id} not found in enriched CSV")
            cluster['environment'] = None
            cluster['predictions'] = None

    print(f"Saving to {OUTPUT_JS_PATH}...")
    with open(OUTPUT_JS_PATH, 'w') as f:
        f.write('var dashboardData = ')
        json.dump(clusters_data, f, indent=2)
        f.write(';')
        
    print("Done!")

if __name__ == "__main__":
    main()
