import pandas as pd
import numpy as np
import joblib
import math
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Set random seed for reproducibility in synthetic data generation
np.random.seed(42)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MEMBERS_FILE = DATA_DIR / 'oak_wilt_cluster_members.csv'
ENRICHED_FILE = DATA_DIR / 'oak_wilt_cluster_enriched.csv'

def haversine_vectorized(lat1, lon1, lat_arr, lon_arr):
    # Vectorized haversine for one point vs many points
    R = 3959 * 5280  # ft
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat_arr)
    dphi = np.radians(lat_arr - lat1)
    dlambda = np.radians(lon_arr - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def prepare_advanced_features():
    print("Loading data...")
    df = pd.read_csv(MEMBERS_FILE)
    df['date'] = pd.to_datetime(df['INSPECTION_DATE'])

    # Load enriched weather data
    if ENRICHED_FILE.exists():
        weather_df = pd.read_csv(ENRICHED_FILE)
        # Map weather from cluster ID
        weather_map = weather_df.set_index('cluster_id')[['avg_temp', 'avg_precip', 'avg_humidity', 'avg_wind']].to_dict('index')
    else:
        print("Warning: Enriched weather data not found. Using defaults.")
        weather_map = {}
    
    X_rows = []
    y_rows = []
    
    print("Generating 'Infection Pressure' dataset (Gravity Model)...")
    
    # Process each cluster
    for cid, cluster in df.groupby('cluster_id'):
        cluster = cluster.sort_values('date')
        if len(cluster) < 5: continue
        
        # Get cluster weather
        c_weather = weather_map.get(cid, {'avg_temp': 20.0, 'avg_precip': 0.0, 'avg_humidity': 50.0, 'avg_wind': 5.0})

        # We simulate the timeline
        # For each tree T at time t, what was the pressure from all trees existing before t?
        
        # 1. Positive Samples (Actual Infections)
        # We look at each tree and calculate pressure from OLDER trees
        for i in range(1, len(cluster)):
            target = cluster.iloc[i]
            sources = cluster.iloc[:i] # Trees infected before target
            
            # Simple time filters
            dists = haversine_vectorized(
                target['LATITUDE'], target['LONGITUDE'],
                sources['LATITUDE'].values, sources['LONGITUDE'].values
            )
            
            # Feature 1: Gravity Pressure
            dists = np.maximum(dists, 1.0) 
            pressure = np.sum(1000 / (dists ** 2))
            
            # Feature 2: Distance to nearest
            min_dist = np.min(dists)
            
            # Feature 3: Local Density (within 100ft)
            density = np.sum(dists < 100)
            
            # Feature 4: Seasonality
            month = target['date'].month
            
            X_rows.append({
                'log_pressure': np.log1p(pressure),
                'log_min_dist': np.log1p(min_dist),
                'local_density': density,
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'avg_temp': c_weather['avg_temp'],
                'avg_precip': c_weather['avg_precip'],
                'avg_humidity': c_weather['avg_humidity'],
                'avg_wind': c_weather['avg_wind']
            })
            y_rows.append(1) # Infected
            
        # 2. Negative Samples (Synthetic Healthy Trees)
        lat_min, lat_max = cluster['LATITUDE'].min(), cluster['LATITUDE'].max()
        lon_min, lon_max = cluster['LONGITUDE'].min(), cluster['LONGITUDE'].max()
        buff = 0.002
        
        for i in range(1, len(cluster)):
            for _ in range(2): # 2 negatives per positive
                r_lat = np.random.uniform(lat_min-buff, lat_max+buff)
                r_lon = np.random.uniform(lon_min-buff, lon_max+buff)
                
                # Check metrics against sources
                sources = cluster.iloc[:i]
                dists = haversine_vectorized(
                    r_lat, r_lon,
                    sources['LATITUDE'].values, sources['LONGITUDE'].values
                )
                
                dists = np.maximum(dists, 1.0)
                pressure = np.sum(1000 / (dists ** 2))
                min_dist = np.min(dists)
                
                density = np.sum(dists < 100)
                month = cluster.iloc[i]['date'].month
                
                X_rows.append({
                    'log_pressure': np.log1p(pressure),
                    'log_min_dist': np.log1p(min_dist),
                    'local_density': density,
                    'month_sin': np.sin(2 * np.pi * month / 12),
                    'month_cos': np.cos(2 * np.pi * month / 12),
                    'avg_temp': c_weather['avg_temp'],
                    'avg_precip': c_weather['avg_precip'],
                    'avg_humidity': c_weather['avg_humidity'],
                    'avg_wind': c_weather['avg_wind']
                })
                y_rows.append(0) # Healthy
                
    return pd.DataFrame(X_rows), np.array(y_rows)

def train():
    X, y = prepare_advanced_features()
    print(f"Dataset compiled. {len(X)} samples.")
    print(f"Class Balance: {np.mean(y):.2f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    print("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    print("\n--- Advanced Model Results ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
    
    # Feature Importance
    print("\nFeature Importance:")
    for name, imp in zip(X.columns, model.feature_importances_):
        print(f"{name}: {imp:.4f}")
        
    # Save
    path = MODELS_DIR / 'graph_transmission_model_pressure.pkl'
    joblib.dump(model, path)
    print(f"\nSaved improved model to {path}")

if __name__ == "__main__":
    train()
