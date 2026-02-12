import pandas as pd
import numpy as np
import requests
import time
import json
from pathlib import Path
from datetime import datetime
import math

MEMBERS_FILE = Path('data/oak_wilt_cluster_members.csv')
FEATURES_FILE = Path('data/oak_wilt_cluster_features.csv')
OUTPUT_FILE = Path('data/oak_wilt_cluster_enriched.csv')
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def haversine_distance_ft(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in feet."""
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 3956
    return c * r * 5280

def get_nasa_weather(lat, lon, start_date, end_date):
    """Fetch temperature, precipitation, humidity, and wind from NASA POWER."""
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    params = {
        'parameters': 'T2M,PRECTOTCORR,RH2M,WS2M',
        'community': 'AG',
        'longitude': lon,
        'latitude': lat,
        'start': start_str,
        'end': end_str,
        'format': 'JSON'
    }
    
    try:
        response = requests.get(NASA_POWER_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        properties = data.get('properties', {}).get('parameter', {})
        t2m = properties.get('T2M', {})
        precip = properties.get('PRECTOTCORR', {})
        humidity = properties.get('RH2M', {})
        wind = properties.get('WS2M', {})
        
        # filter out NASA fill values (-999)
        temp_vals = [v for v in t2m.values() if v != -999]
        precip_vals = [v for v in precip.values() if v != -999]
        humidity_vals = [v for v in humidity.values() if v != -999]
        wind_vals = [v for v in wind.values() if v != -999]
        
        return {
            'avg_temp': np.mean(temp_vals) if temp_vals else np.nan,
            'avg_precip': np.mean(precip_vals) if precip_vals else np.nan,
            'total_precip': np.sum(precip_vals) if precip_vals else np.nan,
            'avg_humidity': np.mean(humidity_vals) if humidity_vals else np.nan,
            'avg_wind': np.mean(wind_vals) if wind_vals else np.nan
        }
        
    except Exception as e:
        print(f"Error fetching weather for {lat}, {lon}: {e}")
        return None

def main():
    print("Loading data...")
    if not MEMBERS_FILE.exists() or not FEATURES_FILE.exists():
        print("Data files not found.")
        return

    df_members = pd.read_csv(MEMBERS_FILE)
    df_features = pd.read_csv(FEATURES_FILE)
    
    df_members['INSPECTION_DATE'] = pd.to_datetime(df_members['INSPECTION_DATE'], errors='coerce')
    
    enriched_data = []
    print(f"Processing {len(df_features)} clusters...")
    
    for _, cluster in df_features.iterrows():
        cid = cluster['cluster_id']
        members = df_members[df_members['cluster_id'] == cid]
        
        if members.empty:
            continue

        # earliest inspection = patient zero
        members = members.sort_values('INSPECTION_DATE')
        patient_zero = members.iloc[0]
        
        pz_date = patient_zero['INSPECTION_DATE']
        if pd.isna(pz_date):
            pz_date = datetime(int(patient_zero['INSPECTION_YEAR']), 1, 1)
            
        pz_lat = patient_zero['LATITUDE']
        pz_lon = patient_zero['LONGITUDE']

        last_date = members.iloc[-1]['INSPECTION_DATE']
        if pd.isna(last_date):
            last_date = datetime(int(members.iloc[-1]['INSPECTION_YEAR']), 12, 31)

        print(f"Cluster {cid}: Fetching weather from {pz_date.date()} to {last_date.date()}...")
        
        # max radial extent from patient zero
        max_dist = 0
        for _, member in members.iterrows():
            dist = haversine_distance_ft(pz_lat, pz_lon, member['LATITUDE'], member['LONGITUDE'])
            if dist > max_dist:
                max_dist = dist
        
        weather = get_nasa_weather(pz_lat, pz_lon, pz_date, last_date)
        
        if weather:
            record = {
                'cluster_id': cid,
                'patient_zero_lat': pz_lat,
                'patient_zero_lon': pz_lon,
                'patient_zero_date': pz_date,
                'duration_days': (last_date - pz_date).days,
                'radius_ft': max_dist,
                'point_count': cluster['point_count'],
                'point_density': cluster['point_density_per_km2'],
                'avg_temp': weather['avg_temp'],
                'avg_precip': weather['avg_precip'],
                'total_precip': weather['total_precip'],
                'avg_humidity': weather['avg_humidity'],
                'avg_wind': weather['avg_wind']
            }
            enriched_data.append(record)
        
        # rate-limit the API
        time.sleep(0.5)
        
    df_enriched = pd.DataFrame(enriched_data)
    df_enriched.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved enriched data to {OUTPUT_FILE}")
    print(df_enriched.head())

if __name__ == "__main__":
    main()
