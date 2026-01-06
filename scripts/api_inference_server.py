import pandas as pd
import numpy as np
import joblib
import uvicorn
import httpx
import math
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# --- Schemas ---
class ForecastInput(BaseModel):
    current_radius_ft: float
    current_points: int
    lat: float
    lon: float
    date: str # YYYY-MM-DD

class TreePoint(BaseModel):
    lat: float
    lon: float
    type: str # 'infected' or 'healthy'

class NetworkInput(BaseModel):
    trees: List[TreePoint]
    date: str

class SimulationRequest(BaseModel):
    trees: List[TreePoint]
    start_date: str
    months: int = 24

class WeatherInput(BaseModel):
    lat: float
    lon: float
    start_date: str 
    end_date: str

app = FastAPI(title="Oak Wilt Risk API")

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
VISUALS_DIR = BASE_DIR / 'visuals'
DATA_DIR = BASE_DIR / 'data'
CLUSTER_MEMBERS_PATH = DATA_DIR / 'oak_wilt_cluster_members.csv'

# --- Load Resources ---
models = {}
cluster_members_df = pd.DataFrame()
cluster_features_df = pd.DataFrame()

print("Initializing Server Resources...")

# 1. Models
try:
    # Exclusive use of the Spatio-Temporal Infection Pressure Model
    if (MODELS_DIR / 'graph_transmission_model_pressure.pkl').exists():
        models['graph'] = joblib.load(MODELS_DIR / 'graph_transmission_model_pressure.pkl')
        models['type'] = 'pressure' 
        print("✓ Loaded Model: Spatio-Temporal Infection Pressure (Gravity).")
    else:
        raise FileNotFoundError("Critical: Infection Pressure Model not found.")
        
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load model: {e}")
    # We maintain a 'models' dict so the server can start, but endpoints will fail gracefully


# 2. Data
if CLUSTER_MEMBERS_PATH.exists():
    cluster_members_df = pd.read_csv(CLUSTER_MEMBERS_PATH)
    cluster_members_df['date'] = pd.to_datetime(cluster_members_df['INSPECTION_DATE']) # Pre-parse
    print(f"✓ Cluster Members loaded: {len(cluster_members_df)} rows")
else:
    print(f"⚠ Warning: {CLUSTER_MEMBERS_PATH} not found.")

features_path = DATA_DIR / 'oak_wilt_cluster_features.csv'
if features_path.exists():
    cluster_features_df = pd.read_csv(features_path)
    # Ensure numeric
    cluster_features_df['year_span'] = pd.to_numeric(cluster_features_df['year_span'], errors='coerce')
    cluster_features_df['point_count'] = pd.to_numeric(cluster_features_df['point_count'], errors='coerce')
    print(f"✓ Cluster Features loaded: {len(cluster_features_df)} rows")
else:
    print(f"⚠ Warning: {features_path} not found.")


# --- Helpers ---
def haversine_dist(lat1, lon1, lat2, lon2):
    R = 3959 * 5280  # Earth radius in ft
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

async def fetch_recent_weather(lat, lon, end_date_str, days=60):
    # Mock/Simulated for speed if external API fails or for quick analysis
    # In production, use cached real data
    # Here we return defaults based on seasonality for Austin approx
    dt = datetime.strptime(end_date_str, '%Y-%m-%d')
    month = dt.month
    
    # Summer (hot/dry), Spring/Fall (wet/mild)
    if 5 <= month <= 9:
        temp = 30.0 # C
        precip = 50.0 # mm total
    else:
        temp = 15.0
        precip = 80.0
        
    return {
        'recent_precip_60d': precip,
        'recent_temp_60d': temp
    }


# --- Endpoints ---

@app.get("/api/historical_scenario")
def get_historical_scenario():
    global cluster_features_df, cluster_members_df
    
    if cluster_features_df.empty or cluster_members_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded on server.")

    # 1. Filter candidates
    candidates = cluster_features_df[
        (cluster_features_df['year_span'] >= 1) & 
        (cluster_features_df['point_count'] >= 10) # Enough points for meaningful visualization
    ]
    
    if candidates.empty:
         candidates = cluster_features_df # fallback

    # 2. Select Random
    selected = candidates.sample(1).iloc[0]
    cid = selected['cluster_id']
    
    # 3. Get Members
    members = cluster_members_df[cluster_members_df['cluster_id'] == cid].sort_values('date')
    
    # 4. Determine Cutoff
    # We want to see growth. Let's find a date that splits the cluster well.
    # Ideally halfway through its lifespan.
    start = members['date'].min()
    end = members['date'].max()
    midpoint = start + (end - start) / 2
    
    # Ensure there are points before and after
    past = members[members['date'] <= midpoint]
    future = members[members['date'] > midpoint]
    
    # Retry logic if bad split
    if len(future) < 3: 
        midpoint = members.iloc[int(len(members)*0.6)]['date']
        past = members[members['date'] <= midpoint]
        future = members[members['date'] > midpoint]

    # 5. Noise / Candidates
    # We need to provide "Healthy" trees that *MIGHT* get infected.
    # In reality, these are the uninfected trees in the forest.
    # We simulate them by scattering points, plus including the 'future' points as unlabelled candidates.
    
    lat_min, lat_max = members['LATITUDE'].min(), members['LATITUDE'].max()
    lon_min, lon_max = members['LONGITUDE'].min(), members['LONGITUDE'].max()
    buff = 0.002
    
    candidates_list = []
    
    # A. The actual future infections (ground truth positives)
    for _, row in future.iterrows():
        candidates_list.append({
            "lat": row['LATITUDE'],
            "lon": row['LONGITUDE'],
            "type": "healthy", # Masked as healthy
            "is_future_infection": True,
            "infection_date": row['date'].strftime('%Y-%m-%d')
        })
        
    # B. Distractors (ground truth negatives)
    for _ in range(50):
        candidates_list.append({
            "lat": np.random.uniform(lat_min-buff, lat_max+buff),
            "lon": np.random.uniform(lon_min-buff, lon_max+buff),
            "type": "healthy",
            "is_future_infection": False
        })
        
    return {
        "cluster_id": int(cid),
        "cutoff_date": midpoint.strftime('%Y-%m-%d'),
        "past_infection": past[['LATITUDE', 'LONGITUDE']].rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'}).to_dict('records'),
        "candidates": candidates_list,
        "center": { "lat": selected['centroid_lat'], "lon": selected['centroid_lon'] }
    }


@app.post("/api/network_simulation")
async def run_network_simulation(data: SimulationRequest):
    """
    Run a multi-step simulation of infection spread.
    Returns the state of the forest at each month.
    """
    if 'graph' not in models:
         raise HTTPException(status_code=500, detail="Graph model missing")

    model = models['graph']
    
    # Setup State
    current_date = datetime.strptime(data.start_date, '%Y-%m-%d')
    forest = []
    for i, t in enumerate(data.trees):
        forest.append({
            "id": i,
            "lat": t.lat, 
            "lon": t.lon, 
            "status": t.type,
            "infection_month": 0 if t.type == 'infected' else -1,
            "prob_history": []
        })
        
    timeline_events = [] # List of {month: int, new_infections: [ids]}
    
    # Simulation Loop
    for month in range(1, data.months + 1):
        step_date = current_date + timedelta(days=30 * month)
        m_sin = np.sin(2 * np.pi * step_date.month / 12)
        m_cos = np.cos(2 * np.pi * step_date.month / 12)
        
        # INFECTIOUS SET SELECTION (Biological Latency)
        # A tree infected today does not immediately transmit 'pressure' tomorrow.
        # It takes time for the fungal mats to form or root grafts to transfer.
        # We introduce a 'Incubation Period' of 3 months before a tree becomes a vector.
        # Exception: Trees infected at start (month 0) are assumed to be established/infectious sources.
        
        infectious_indices = [
            idx for idx, t in enumerate(forest) 
            if t['status'] == 'infected' and (t['infection_month'] == 0 or (month - t['infection_month']) >= 3)
        ]
        
        healthy_indices = [idx for idx, t in enumerate(forest) if t['status'] == 'healthy']
        
        newly_infected = []
        
        # If no infectious trees yet (e.g. all original infections cured, or new ones latent)
        if not infectious_indices:
            # Check if we have *any* infected (just latent ones)
            total_infected = len([t for t in forest if t['status'] == 'infected'])
            if total_infected == 0:
                print("No infected trees remaining.")
                break 
            # If we have latent infected but no infectious, we just wait this month out.
            continue
            
        # Check every healthy tree against disease pressure
        for h_idx in healthy_indices:
            h_tree = forest[h_idx]
            
            # Find nearest INFECTIOUS tree
            min_dist = float('inf')
            pressure = 0.0
            nearby_count = 0
            
            for i_idx in infectious_indices:
                i_tree = forest[i_idx]
                d = haversine_dist(h_tree['lat'], h_tree['lon'], i_tree['lat'], i_tree['lon'])
                
                # Pressure calculation (Inverse Square Law)
                # Pressure = Sum(1000 / distance^2) from all infected sources
                d_safe = max(d, 1.0)
                pressure += 1000.0 / (d_safe ** 2)
                
                if d < min_dist: min_dist = d
                if d < 100: nearby_count += 1
                
            # --- Biological Guardrail (Root Graft Limit) ---
            # The model is trained to detect local spread. 
            # Biologically, root graft transmission is physically impossible beyond ~100-150ft.
            # We strictly enforce this to prevent "Teleporting" infections (out-of-distribution errors).
            if min_dist > 150:
                continue # Skip prediction, too far for root transmission.

            # Feature Vector
            # Spatio-Temporal Features
            feats = pd.DataFrame([{
                'log_pressure': np.log1p(pressure),
                'log_min_dist': np.log1p(min_dist),
                'local_density': nearby_count,
                'month_sin': m_sin,
                'month_cos': m_cos
            }])
            
            # Predict
            prob = model.predict_proba(feats)[0][1] # Probability of CLASS 1 (Infected)
            
            # DEBUG: Print high probabilities to see if anything is triggering
            if prob > 0.1:
                print(f"Tree {h_idx}: Dist={min_dist:.1f}ft, Pressure={pressure:.1f}, Prob={prob:.4f}")

            forest[h_idx]['prob_history'].append(float(prob))
            
            # --- Deterministic Decision Support ---
            # Arborists need reliability, not randomness.
            # We use a calibrated confidence threshold.
            if prob > 0.50: 
                newly_infected.append(h_idx)
        
        # Apply updates
        if newly_infected:
            for idx in newly_infected:
                forest[idx]['status'] = 'infected'
                forest[idx]['infection_month'] = month
            
            timeline_events.append({
                "month": month,
                "date": step_date.strftime('%Y-%m'),
                "new_cases": newly_infected
            })
            
    return {
        "timeline": timeline_events,
        "total_months": data.months
    }

@app.post("/api/forecast")
async def get_forecast(data: ForecastInput):
    # Legacy wrapper for Growth Model
    if 'main' not in models: return {}
    # ... logic skipped for brevity, keeping existing ...
    # Re-implementing simplified version for completeness of file
    
    weather = await fetch_recent_weather(data.lat, data.lon, data.date)
    dt = datetime.strptime(data.date, '%Y-%m-%d')
    
    features = pd.DataFrame([{
        'log_radius': np.log1p(data.current_radius_ft),
        'log_points': np.log1p(data.current_points),
        'recent_precip_60d': weather['recent_precip_60d'],
        'recent_temp_60d': weather['recent_temp_60d'],
        'month_sin': np.sin(2 * np.pi * dt.month / 12),
        'month_cos': np.cos(2 * np.pi * dt.month / 12)
    }])
    
    pred = models['main'].predict(features)[0]
    return {
        "forecast_growth_90d_ft": float(max(0, pred)),
        "predicted_radius_90d_ft": float(data.current_radius_ft + max(0, pred)),
        "risk_level": "High" if pred > 15 else "Low",
        "weather_context": weather
    }

# Static Files
app.mount("/visuals", StaticFiles(directory=VISUALS_DIR), name="visuals")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
