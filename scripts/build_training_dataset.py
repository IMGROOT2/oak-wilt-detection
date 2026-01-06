import pandas as pd
import numpy as np
import math
from pathlib import Path
from datetime import datetime, timedelta

# Constants
DATA_DIR = Path('data')
MEMBERS_FILE = DATA_DIR / 'oak_wilt_cluster_members.csv'
OUTPUT_FILE = DATA_DIR / 'oak_wilt_graph_data.csv'

# Configuration
NEGATIVE_SAMPLE_RATIO = 2  # Generate 2 phantom healthy trees for every real infected tree
MAX_INFLUENCE_DIST_FT = 300 # Don't consider trees further than this as "parents"

def haversine_distance_ft(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 3956 
    return c * r * 5280

def generate_phantom_point(center_lat, center_lon, radius_ft):
    # Random angle
    theta = np.random.uniform(0, 2 * math.pi)
    # Random radius (square root to distribute evenly in circle)
    r_ft = math.sqrt(np.random.uniform(0, 1)) * radius_ft
    
    # Approx conversion ft to degrees
    r_deg = r_ft / 364000.0 # Rough approx
    
    new_lat = center_lat + r_deg * math.cos(theta)
    new_lon = center_lon + r_deg * math.sin(theta)
    return new_lat, new_lon

def main():
    print("Loading cluster members...")
    if not MEMBERS_FILE.exists():
        print("Error: Cluster members file not found.")
        return

    df = pd.read_csv(MEMBERS_FILE)
    df['INSPECTION_DATE'] = pd.to_datetime(df['INSPECTION_DATE'])
    
    graph_data = []
    
    clusters = df.groupby('cluster_id')
    print(f"Processing {len(clusters)} clusters...")
    
    for cluster_id, group in clusters:
        group = group.sort_values('INSPECTION_DATE')
        
        # We start looking after the first few trees have established a core
        if len(group) < 3:
            continue
            
        points = group.to_dict('records')
        
        # Iterate through the timeline of the cluster
        # Start from index 1 (the second tree)
        for i in range(1, len(points)):
            current_tree = points[i]
            event_date = current_tree['INSPECTION_DATE']
            
            # The "Past" is everyone before this index
            prior_trees = points[:i]
            
            # 1. POSITIVE SAMPLE (The tree that actually got infected)
            # Find closest infected neighbor (most likely parent)
            min_dist = float('inf')
            nearby_count = 0
            
            for parent in prior_trees:
                dist = haversine_distance_ft(
                    current_tree['LATITUDE'], current_tree['LONGITUDE'],
                    parent['LATITUDE'], parent['LONGITUDE']
                )
                if dist < min_dist:
                    min_dist = dist
                if dist < 100: # Density within 100ft
                    nearby_count += 1
            
            graph_data.append({
                'label': 1, # Infected
                'dist_to_nearest_infected': min_dist,
                'nearby_infection_count': nearby_count,
                'month': event_date.month
            })
            
            # 2. NEGATIVE SAMPLES (Phantom healthy trees that stayed healthy)
            # We generate points around the CURRENT cluster boundary
            # Calculate current cluster centroid and radius
            lats = [p['LATITUDE'] for p in prior_trees]
            lons = [p['LONGITUDE'] for p in prior_trees]
            cent_lat = sum(lats) / len(lats)
            cent_lon = sum(lons) / len(lons)
            
            # To be fair, negatives should be somewhat close (at risk) but not infected
            # Let's generate them within [MinDist, MaxDist] of the cluster center
            # Find max extent of current cluster
            current_max_radius = 0
            for p in prior_trees:
                d = haversine_distance_ft(cent_lat, cent_lon, p['LATITUDE'], p['LONGITUDE'])
                if d > current_max_radius:
                    current_max_radius = d
            
            # Generate phantoms just outside the current radius (The "Front")
            search_radius = max(current_max_radius + 50, 100) # At least 100ft
            
            for _ in range(NEGATIVE_SAMPLE_RATIO):
                phant_lat, phant_lon = generate_phantom_point(cent_lat, cent_lon, search_radius)
                
                # Calculate features for phantom
                p_min_dist = float('inf')
                p_nearby_count = 0
                for parent in prior_trees:
                    dist = haversine_distance_ft(
                        phant_lat, phant_lon,
                        parent['LATITUDE'], parent['LONGITUDE']
                    )
                    if dist < p_min_dist:
                        p_min_dist = dist
                    if dist < 100:
                        p_nearby_count += 1
                
                graph_data.append({
                    'label': 0, # Healthy
                    'dist_to_nearest_infected': p_min_dist,
                    'nearby_infection_count': p_nearby_count,
                    'month': event_date.month
                })

    # Save
    out_df = pd.DataFrame(graph_data)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Graph dataset saved to {OUTPUT_FILE} with {len(out_df)} samples.")
    print(out_df['label'].value_counts())

if __name__ == "__main__":
    main()
