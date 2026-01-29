import csv
from geopy.distance import geodesic
import numpy as np
API_BASE_URL = "http://localhost:8000"

# Prefer importing server functions directly to avoid HTTP when running locally
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

LOCAL_API = False
try:
    from prediction_system.inference_server import get_historical_scenario as _get_historical_scenario
    from prediction_system.inference_server import run_network_simulation as _run_network_simulation
    from prediction_system.inference_server import SimulationRequest
    import asyncio
    LOCAL_API = True
except Exception:
    LOCAL_API = False
    import requests

# Load cluster members (tree locations per cluster)
def load_cluster_members(path):
    clusters = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Support both 'lat'/'lon' and 'LATITUDE'/'LONGITUDE'
            lat_val = row.get('lat') or row.get('LATITUDE')
            lon_val = row.get('lon') or row.get('LONGITUDE')
            try:
                lat = float(lat_val)
                lon = float(lon_val)
            except (TypeError, ValueError):
                print(f"Warning: Skipping row with invalid lat/lon: {row}")
                continue
            cid = row.get('cluster_id')
            if not cid:
                print(f"Warning: Skipping row with missing cluster_id: {row}")
                continue
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append({
                'lat': lat,
                'lon': lon,
                'type': row.get('type', row.get('SPECIES', 'unknown')),
                'id': row.get('tree_id', row.get('original_label', ''))
            })
    return clusters

# Get scenario for a cluster from the backend
def get_scenario(cluster_id):
    if LOCAL_API:
        # call the server function directly
        return _get_historical_scenario(cluster_id)
    else:
        resp = requests.get(f"{API_BASE_URL}/api/historical_scenario?cluster_id={cluster_id}")
        resp.raise_for_status()
        return resp.json()

# Run simulation for a cluster
def run_simulation(scenario):
    trees = []
    for t in scenario['past_infection']:
        trees.append({
            'lat': t.get('lat') or t.get('LATITUDE'),
            'lon': t.get('lon') or t.get('LONGITUDE'),
            'type': 'infected'
        })
    for t in scenario['candidates']:
        trees.append({
            'lat': t.get('lat') or t.get('LATITUDE'),
            'lon': t.get('lon') or t.get('LONGITUDE'),
            'type': 'healthy'
        })
    payload = {
        'trees': trees,
        'start_date': scenario['cutoff_date'],
        'months': 24,  # match frontend logic
        'custom_temp': None,
        'custom_precip': None,
        'custom_humidity': None,
        'custom_wind_speed': None
    }
    if LOCAL_API:
        # Build SimulationRequest (Pydantic) and call the async function
        req = SimulationRequest(**payload)
        return asyncio.get_event_loop().run_until_complete(_run_network_simulation(req))
    else:
        resp = requests.post(f"{API_BASE_URL}/api/network_simulation", json=payload)
        resp.raise_for_status()
        return resp.json()

# Calculate centroid
def centroid(coords):
    lat = np.mean([c[0] for c in coords])
    lon = np.mean([c[1] for c in coords])
    return (lat, lon)

# Calculate 90th percentile distance from centroid
def effective_radius(centroid, coords):
    if not coords:
        return 0
    dists = [geodesic(centroid, c).meters for c in coords]
    dists.sort()
    # JS logic: k = Math.floor(length * 0.9), clamp to length-1
    k = int(np.floor(len(dists) * 0.9))
    if k >= len(dists):
        k = len(dists) - 1
    return dists[k]

# Main logic
def main():
    clusters = load_cluster_members("data/oak_wilt_cluster_members.csv")
    results = []
    for idx, cid in enumerate(clusters):
        try:
            scenario = get_scenario(cid)
            sim = run_simulation(scenario)
            # Helper functions for ID and coordinates
            def get_id(t):
                return t.get('id') or t.get('tree_id') or t.get('original_label')
            def get_lat(t):
                return float(t.get('lat') or t.get('LATITUDE'))
            def get_lon(t):
                return float(t.get('lon') or t.get('LONGITUDE'))

            # 1. Centroid from original infections only
            origin = scenario['past_infection']
            origin_coords = [(get_lat(t), get_lon(t)) for t in origin]
            if not origin_coords:
                print(f"Cluster {cid}: ERROR - No original infections")
                continue
            c = centroid(origin_coords)

            # 2. Initial indices (original infections are first in the payload)
            all_trees = scenario['past_infection'] + scenario['candidates']
            origin_count = len(origin)
            initial_indices = set(range(origin_count))

            # 3. Final indices (initial + new_cases indices returned by simulation)
            final_indices = set(initial_indices)
            for ev in sim.get('timeline', []):
                for nc in ev.get('new_cases', []):
                    try:
                        final_indices.add(int(nc))
                    except (TypeError, ValueError):
                        pass

            # Debug print for first cluster only
            if idx == 0:
                print(f"[DEBUG] Cluster {cid}")
                print(f"  origin_count: {origin_count}")
                print(f"  initial_indices sample: {list(initial_indices)[:5]}")
                print(f"  final_indices sample: {list(final_indices)[:5]} (total {len(final_indices)})")
                print(f"  sim keys: {list(sim.keys())}")
                if sim.get('timeline'):
                    print(f"  first timeline event new_cases (sample): {sim['timeline'][0].get('new_cases')[:10]}")
                print(f"  first 5 candidate coords: {[ (get_lat(t), get_lon(t)) for t in scenario['candidates'][:5]]}")

            # 4. Get coordinates for initial and final sets by index
            initial_coords = [ (float(t.get('lat') or t.get('LATITUDE')), float(t.get('lon') or t.get('LONGITUDE'))) for i, t in enumerate(all_trees) if i in initial_indices ]
            final_coords = [ (float(t.get('lat') or t.get('LATITUDE')), float(t.get('lon') or t.get('LONGITUDE'))) for i, t in enumerate(all_trees) if i in final_indices and i < len(all_trees) ]

            # 5. Calculate effective radii (90th percentile)
            r0 = effective_radius(c, initial_coords)
            r1 = effective_radius(c, final_coords)

            # 6. Calculate growth in feet and annualize
            growth_ft = (r1 - r0) * 3.28084
            months = sim.get('total_months', 24)
            deltaMonths = max(1, months)  # Avoid div/0
            if growth_ft > 0:
                yearly_rate = (growth_ft / deltaMonths) * 12
            else:
                yearly_rate = 0
            results.append((cid, round(yearly_rate, 2)))
            print(f"Cluster {cid}: {yearly_rate:.2f} ft/yr")
        except Exception as e:
            print(f"Cluster {cid}: ERROR - {e}")
    # Output table
    print("\nClusterID,SpreadRate_ft_per_yr")
    for cid, rate in results:
        print(f"{cid},{rate}")

    # Save results to CSV
    out_path = 'data/simulated_spread_rates.csv'
    try:
        with open(out_path, 'w', newline='') as outf:
            writer = csv.writer(outf)
            writer.writerow(['cluster_id', 'spread_ft_per_yr'])
            for cid, rate in results:
                writer.writerow([cid, rate])
        print(f"Saved results to {out_path}")
    except Exception as e:
        print(f"Failed to write CSV: {e}")

    # Print average spread rate
    if results:
        avg = sum(r for _, r in results) / len(results)
        print(f"Average spread rate: {avg:.2f} ft/yr")
    else:
        print("No results to average.")

if __name__ == "__main__":
    main()
