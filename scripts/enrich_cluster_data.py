import pandas as pd
import numpy as np
import requests
import time
import json
from pathlib import Path
from datetime import datetime
import math

# Constants
MEMBERS_FILE = Path('data/oak_wilt_cluster_members.csv')
FEATURES_FILE = Path('data/oak_wilt_cluster_features.csv')
OUTPUT_FILE = Path('data/oak_wilt_cluster_enriched.csv')
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def haversine_distance_ft(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in feet.
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 3956 # Radius of earth in miles. Use 6371 for km
    return c * r * 5280 # Convert to feet

def get_nasa_weather(lat, lon, start_date, end_date):
    """
    Fetch weather data from NASA POWER API.
    Parameters: PRECTOTCORR (Precipitation), RH2M (Humidity), WS2M (Wind Speed)
    """
    # Format dates as YYYYMMDD
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    params = {
        'parameters': 'PRECTOTCORR,RH2M,WS2M',
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
        
        # Extract values
        properties = data.get('properties', {}).get('parameter', {})
        precip = properties.get('PRECTOTCORR', {})
        humidity = properties.get('RH2M', {})
        wind = properties.get('WS2M', {})
        
        # Filter out fill values (-999)
        precip_vals = [v for v in precip.values() if v != -999]
        humidity_vals = [v for v in humidity.values() if v != -999]
        wind_vals = [v for v in wind.values() if v != -999]
        
        return {
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
    
    # Convert dates
    df_members['INSPECTION_DATE'] = pd.to_datetime(df_members['INSPECTION_DATE'], errors='coerce')
    
    enriched_data = []
    
    print(f"Processing {len(df_features)} clusters...")
    
    for _, cluster in df_features.iterrows():
        cid = cluster['cluster_id']
        members = df_members[df_members['cluster_id'] == cid]
        
        if members.empty:
            continue
            
        # Identify Patient Zero (earliest inspection)
        # Note: If multiple trees have the same earliest date, we pick the first one (or centroid of them)
        # For simplicity, we pick the first one sorted by date
        members = members.sort_values('INSPECTION_DATE')
        patient_zero = members.iloc[0]
        
        pz_date = patient_zero['INSPECTION_DATE']
        if pd.isna(pz_date):
            # Fallback to Jan 1st of the year if date is missing
            pz_date = datetime(int(patient_zero['INSPECTION_YEAR']), 1, 1)
            
        pz_lat = patient_zero['LATITUDE']
        pz_lon = patient_zero['LONGITUDE']
        
        # End date is the last inspection in the cluster
        last_date = members.iloc[-1]['INSPECTION_DATE']
        if pd.isna(last_date):
            last_date = datetime(int(members.iloc[-1]['INSPECTION_YEAR']), 12, 31)
            
        # Fetch Weather Data
        # We fetch from Patient Zero date to End Date
        # If duration is too short (e.g. same day), we fetch at least a week?
        # Let's stick to the actual range. If same day, it returns that day's weather.
        
        print(f"Cluster {cid}: Fetching weather from {pz_date.date()} to {last_date.date()}...")
        
        # Calculate Radius from Patient Zero
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
                'radius_ft': max_dist, # Calculated from Patient Zero
                'point_count': cluster['point_count'],
                'point_density': cluster['point_density_per_km2'],
                'avg_precip': weather['avg_precip'],
                'total_precip': weather['total_precip'],
                'avg_humidity': weather['avg_humidity'],
                'avg_wind': weather['avg_wind']
            }
            enriched_data.append(record)
        
        # Be nice to the API
        time.sleep(0.5)
        
    df_enriched = pd.DataFrame(enriched_data)
    df_enriched.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved enriched data to {OUTPUT_FILE}")
    print(df_enriched.head())

if __name__ == "__main__":
    main()
