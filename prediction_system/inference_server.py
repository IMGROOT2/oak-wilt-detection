import pandas as pd
import numpy as np
import joblib
import uvicorn
import httpx
import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

# Schemas
class ForecastInput(BaseModel):
    current_radius_ft: float
    current_points: int
    lat: float
    lon: float
    date: str

class TreePoint(BaseModel):
    lat: float
    lon: float
    type: str

class NetworkInput(BaseModel):
    trees: List[TreePoint]
    date: str

class SimulationRequest(BaseModel):
    trees: List[TreePoint]
    start_date: str
    months: int = 24
    custom_temp: Optional[float] = None
    custom_precip: Optional[float] = None
    custom_humidity: Optional[float] = None
    custom_wind_speed: Optional[float] = None

class WeatherInput(BaseModel):
    lat: float
    lon: float
    start_date: str 
    end_date: str

app = FastAPI(title="Oak Wilt Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
VISUALS_DIR = BASE_DIR / 'prediction_system' / 'web_interface'
DATA_DIR = BASE_DIR / 'data'
CLUSTER_MEMBERS_PATH = DATA_DIR / 'oak_wilt_cluster_members.csv'

# Load resources on startup
models = {}
cluster_members_df = pd.DataFrame()
cluster_features_df = pd.DataFrame()

print("Initializing server...")

# 1. Load model
try:
    if (MODELS_DIR / 'graph_transmission_model_pressure.pkl').exists():
        models['graph'] = joblib.load(MODELS_DIR / 'graph_transmission_model_pressure.pkl')
        models['type'] = 'pressure' 
        print("Loaded infection pressure model.")
    else:
        raise FileNotFoundError("Infection pressure model not found.")
        
except Exception as e:
    print(f"Model load error: {e}")


# 2. Load data
if CLUSTER_MEMBERS_PATH.exists():
    cluster_members_df = pd.read_csv(CLUSTER_MEMBERS_PATH)
    cluster_members_df['date'] = pd.to_datetime(cluster_members_df['INSPECTION_DATE'])
    print(f"Cluster members: {len(cluster_members_df)} rows")
else:
    print(f"Warning: {CLUSTER_MEMBERS_PATH} not found.")

features_path = DATA_DIR / 'oak_wilt_cluster_features.csv'
if features_path.exists():
    cluster_features_df = pd.read_csv(features_path)
    cluster_features_df['year_span'] = pd.to_numeric(cluster_features_df['year_span'], errors='coerce')
    cluster_features_df['point_count'] = pd.to_numeric(cluster_features_df['point_count'], errors='coerce')
    print(f"Cluster features: {len(cluster_features_df)} rows")
else:
    print(f"Warning: {features_path} not found.")


# Helpers
def haversine_dist(lat1, lon1, lat2, lon2):
    """Great-circle distance in feet."""
    R = 3959 * 5280
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

async def fetch_recent_weather(lat, lon, end_date_str, days=60):
    """Seasonal weather defaults for Austin (fallback)."""
    dt = datetime.strptime(end_date_str, '%Y-%m-%d')
    month = dt.month
    if 5 <= month <= 9:
        temp = 30.0
        precip = 50.0
    else:
        temp = 15.0
        precip = 80.0
        
    return {
        'recent_precip_60d': precip,
        'recent_temp_60d': temp
    }


async def fetch_real_nasa_weather(lat, lon, start_date):
    """Fetch 30-day averaged weather from NASA POWER (async)."""
    try:
        # look back 35 days from simulation start (5-day lag buffer)
        sim_start = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = sim_start - timedelta(days=5)
        start_dt = end_dt - timedelta(days=30)
        
        start_str = start_dt.strftime('%Y%m%d')
        end_str = end_dt.strftime('%Y%m%d')
        
        params = {
            'parameters': 'T2M,PRECTOTCORR,RH2M,WS2M',
            'community': 'AG',
            'longitude': lon,
            'latitude': lat,
            'start': start_str,
            'end': end_str,
            'format': 'JSON'
        }
        
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
        properties = data.get('properties', {}).get('parameter', {})
        t2m = properties.get('T2M', {})
        precip = properties.get('PRECTOTCORR', {})
        rh2m = properties.get('RH2M', {})
        ws2m = properties.get('WS2M', {})
        
        # average valid values
        def get_avg(d):
            vals = [v for v in d.values() if v != -999]
            return sum(vals)/len(vals) if vals else None

        avg_temp = get_avg(t2m)
        avg_precip = get_avg(precip)
        avg_humid = get_avg(rh2m)
        avg_wind = get_avg(ws2m)
        
        return {
            "temp": avg_temp,
            "precip": avg_precip,
            "humidity": avg_humid,
            "wind": avg_wind
        }
        
    except Exception as e:
        print(f"NASA API fetch failed: {e}")
        return None

# Endpoints

@app.get("/api/historical_scenario")
def get_historical_scenario(cluster_id: Optional[int] = None):
    global cluster_features_df, cluster_members_df
    
    if cluster_features_df.empty or cluster_members_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded on server.")

    # filter clusters with enough data for a meaningful scenario
    candidates = cluster_features_df[
        (cluster_features_df['year_span'] >= 1) & 
        (cluster_features_df['point_count'] >= 10)
    ]
    
    if candidates.empty:
         candidates = cluster_features_df

    if cluster_id is not None:
        match = candidates[candidates['cluster_id'] == cluster_id]
        if match.empty:
            # If the requested cluster is not eligible, return 404 so the client can retry
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not available for historical scenario")
        selected = match.iloc[0]
    else:
        selected = candidates.sample(1).iloc[0]
    cid = selected['cluster_id']
    
    # get cluster members and split at the midpoint
    members = cluster_members_df[cluster_members_df['cluster_id'] == cid].sort_values('date')
    
    start = members['date'].min()
    end = members['date'].max()
    midpoint = start + (end - start) / 2
    past = members[members['date'] <= midpoint]
    future = members[members['date'] > midpoint]
    
    # re-split if the initial split doesn't leave enough future data
    if len(future) < 3: 
        midpoint = members.iloc[int(len(members)*0.6)]['date']
        past = members[members['date'] <= midpoint]
        future = members[members['date'] > midpoint]

    # build candidate list: future infections (masked as healthy) + distractors
    lat_min, lat_max = members['LATITUDE'].min(), members['LATITUDE'].max()
    lon_min, lon_max = members['LONGITUDE'].min(), members['LONGITUDE'].max()
    buff = 0.002
    
    candidates_list = []
    
    # actual future infections, masked as healthy for backtest
    for _, row in future.iterrows():
        candidates_list.append({
            "lat": row['LATITUDE'],
            "lon": row['LONGITUDE'],
            "type": "healthy",
            "is_future_infection": True,
            "infection_date": row['date'].strftime('%Y-%m-%d')
        })
        
    # distractor points (true negatives)
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


@app.get("/api/eligible_clusters")
def get_eligible_clusters():
    """Return cluster IDs with spread rates in the 20-200 ft/yr range."""
    sim_path = DATA_DIR / 'simulated_spread_rates.csv'
    eligible = []
    excluded = []
    source = None

    try:
        if sim_path.exists():
            df = pd.read_csv(sim_path)
            # Expect columns: cluster_id, spread_ft_per_yr
            if 'cluster_id' in df.columns and 'spread_ft_per_yr' in df.columns and not df.empty:
                source = 'simulated'
                df['cluster_id'] = pd.to_numeric(df['cluster_id'], errors='coerce')
                df['spread_ft_per_yr'] = pd.to_numeric(df['spread_ft_per_yr'], errors='coerce')
                df = df.dropna(subset=['cluster_id', 'spread_ft_per_yr'])
                eligible_df = df[(df['spread_ft_per_yr'] >= 20) & (df['spread_ft_per_yr'] <= 200)]
                eligible = [int(x) for x in eligible_df['cluster_id'].tolist()]
                excluded = [int(x) for x in df[~df.index.isin(eligible_df.index)]['cluster_id'].tolist()]

        # If simulated not available or empty, fallback to cluster_features_df
        if not eligible:
            if cluster_features_df is None or cluster_features_df.empty:
                raise HTTPException(status_code=503, detail='Cluster features not available')
            source = source or 'features'
            df2 = cluster_features_df.copy()
            if 'cluster_id' in df2.columns and 'spread_rate_km_per_year' in df2.columns:
                df2['cluster_id'] = pd.to_numeric(df2['cluster_id'], errors='coerce')
                df2['spread_rate_km_per_year'] = pd.to_numeric(df2['spread_rate_km_per_year'], errors='coerce')
                df2 = df2.dropna(subset=['cluster_id', 'spread_rate_km_per_year'])
                df2['spread_ft_per_yr'] = df2['spread_rate_km_per_year'] * 3280.84
                eligible_df2 = df2[(df2['spread_ft_per_yr'] >= 20) & (df2['spread_ft_per_yr'] <= 200)]
                eligible = [int(x) for x in eligible_df2['cluster_id'].tolist()]
                excluded = [int(x) for x in df2[~df2.index.isin(eligible_df2.index)]['cluster_id'].tolist()]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute eligible clusters: {e}")

    return {
        'source': source,
        'eligible': eligible,
        'excluded': excluded
    }


@app.post("/api/network_simulation")
async def run_network_simulation(data: SimulationRequest):
    """Run a month-by-month simulation of infection spread."""
    if 'graph' not in models:
         raise HTTPException(status_code=500, detail="Graph model missing")

    model = models['graph']
    current_date = datetime.strptime(data.start_date, '%Y-%m-%d')
    
    # fill in any missing weather values from NASA POWER
    user_overrides = {
        "temp": data.custom_temp,
        "precip": data.custom_precip,
        "humidity": data.custom_humidity,
        "wind": data.custom_wind_speed
    }
    
    if None in user_overrides.values():
        print("Fetching missing weather data from NASA POWER...")
        
        if data.trees:
             lats = [t.lat for t in data.trees]
             lons = [t.lon for t in data.trees]
             c_lat = sum(lats) / len(lats)
             c_lon = sum(lons) / len(lons)
             
             nasa_data = await fetch_real_nasa_weather(c_lat, c_lon, data.start_date)
             
             if nasa_data:
                 print(f"NASA Data Fetched: {nasa_data}")
                 if user_overrides['temp'] is None and nasa_data['temp']: 
                     user_overrides['temp'] = round(nasa_data['temp'], 1)
                 if user_overrides['precip'] is None and nasa_data['precip']: 
                     user_overrides['precip'] = round(nasa_data['precip'] * 30, 1) # Monthly Est (mm/day * 30)
                 if user_overrides['humidity'] is None and nasa_data['humidity']: 
                     user_overrides['humidity'] = round(nasa_data['humidity'], 1)
                 if user_overrides['wind'] is None and nasa_data['wind']: 
                     user_overrides['wind'] = round(nasa_data['wind'], 1)

    # weather defaults (metric: C, mm/month, %, m/s)
    c_temp = user_overrides['temp'] if user_overrides['temp'] is not None else 25.0
    c_precip = user_overrides['precip'] if user_overrides['precip'] is not None else 50.0
    c_humidity = user_overrides['humidity'] if user_overrides['humidity'] is not None else 65.0
    c_wind = user_overrides['wind'] if user_overrides['wind'] is not None else 3.0
    
    print(f"Weather context: T={c_temp}C, P={c_precip}mm, H={c_humidity}%, W={c_wind}m/s")
    
    if data.custom_temp is not None or data.custom_precip is not None:
        print(f"Using user overrides: Temp={data.custom_temp}, Precip={data.custom_precip}")
    
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
        
    timeline_events = []
    
    # month-by-month simulation loop
    for month in range(1, data.months + 1):
        step_date = current_date + timedelta(days=30 * month)
        m_sin = np.sin(2 * np.pi * step_date.month / 12)
        m_cos = np.cos(2 * np.pi * step_date.month / 12)
        
        # 3-month incubation: newly infected trees do not transmit immediately
        infectious_indices = [
            idx for idx, t in enumerate(forest) 
            if t['status'] == 'infected' and (t['infection_month'] == 0 or (month - t['infection_month']) >= 3)
        ]
        
        healthy_indices = [idx for idx, t in enumerate(forest) if t['status'] == 'healthy']
        
        newly_infected = []
        
        if not infectious_indices:
            total_infected = len([t for t in forest if t['status'] == 'infected'])
            if total_infected == 0:
                break 
            continue
            
        # Check every healthy tree against disease pressure
        for h_idx in healthy_indices:
            h_tree = forest[h_idx]
            
            min_dist = float('inf')
            pressure = 0.0
            nearby_count = 0
            
            for i_idx in infectious_indices:
                i_tree = forest[i_idx]
                d = haversine_dist(h_tree['lat'], h_tree['lon'], i_tree['lat'], i_tree['lon'])
                
                # inverse-square pressure model
                d_safe = max(d, 1.0)
                pressure += 1000.0 / (d_safe ** 2)
                
                if d < min_dist: min_dist = d
                if d < 100: nearby_count += 1
                
            # root graft transmission caps out around 150ft
            if min_dist > 150:
                continue

            # build feature vector
            feats = pd.DataFrame([{
                'log_pressure': np.log1p(pressure),
                'log_min_dist': np.log1p(min_dist),
                'local_density': nearby_count,
                'month_sin': m_sin,
                'month_cos': m_cos,
                'avg_temp': c_temp,
                'avg_precip': c_precip / 30.0,
                'avg_humidity': c_humidity,
                'avg_wind': c_wind
            }])
            
            # Predict
            prob = model.predict_proba(feats)[0][1]
            if prob > 0.1:
                print(f"Tree {h_idx}: Dist={min_dist:.1f}ft, Pressure={pressure:.1f}, Prob={prob:.4f}")

            forest[h_idx]['prob_history'].append(float(prob))
            
            # threshold at 0.50 for deterministic predictions
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
        "total_months": data.months,
        "environment": {
            "temp": c_temp, 
            "precip": c_precip, 
            "humidity": c_humidity, 
            "wind": c_wind 
        }
    }

@app.get("/health")
def health_check():
    return {"status": "online"}

@app.post("/api/forecast")
async def get_forecast(data: ForecastInput):
    """Legacy growth model endpoint."""
    if 'main' not in models: return {}
    
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

# Static file serving
app.mount("/visuals", StaticFiles(directory=VISUALS_DIR), name="visuals")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
