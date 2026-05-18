import csv
from geopy.distance import geodesic
import numpy as np
API_BASE_URL = "http://localhost:8000"

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Try to import the server's functions directly; fall back to HTTP calls if the import fails
# (e.g. running this script against a remotely deployed server).
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

def load_cluster_members(path):
    """Load cluster member tree locations from CSV."""
    clusters = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # CSVs use either lowercase or the original ArcGIS column names.
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

def get_scenario(cluster_id):
    """Fetch a historical scenario for the given cluster."""
    if LOCAL_API:
        return _get_historical_scenario(cluster_id)
    else:
        resp = requests.get(f"{API_BASE_URL}/api/historical_scenario?cluster_id={cluster_id}")
        resp.raise_for_status()
        return resp.json()

def run_simulation(scenario):
    """Run the spread simulation on a scenario's trees."""
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
        'months': 24,
        'custom_temp': None,
        'custom_precip': None,
        'custom_humidity': None,
        'custom_wind_speed': None
    }
    if LOCAL_API:
        req = SimulationRequest(**payload)
        return asyncio.get_event_loop().run_until_complete(_run_network_simulation(req))
    else:
        resp = requests.post(f"{API_BASE_URL}/api/network_simulation", json=payload)
        resp.raise_for_status()
        return resp.json()

def centroid(coords):
    lat = np.mean([c[0] for c in coords])
    lon = np.mean([c[1] for c in coords])
    return (lat, lon)

def effective_radius(centroid, coords):
    """90th percentile distance from centroid."""
    if not coords:
        return 0
    dists = [geodesic(centroid, c).meters for c in coords]
    dists.sort()
    k = int(np.floor(len(dists) * 0.9))
    if k >= len(dists):
        k = len(dists) - 1
    return dists[k]

def main():
    clusters = load_cluster_members("data/oak_wilt_cluster_members.csv")
    results = []
    for idx, cid in enumerate(clusters):
        try:
            scenario = get_scenario(cid)
            sim = run_simulation(scenario)

            def get_lat(t):
                return float(t.get('lat') or t.get('LATITUDE'))
            def get_lon(t):
                return float(t.get('lon') or t.get('LONGITUDE'))

            origin = scenario['past_infection']
            origin_coords = [(get_lat(t), get_lon(t)) for t in origin]
            if not origin_coords:
                print(f"Cluster {cid}: ERROR - No original infections")
                continue
            c = centroid(origin_coords)

            all_trees = scenario['past_infection'] + scenario['candidates']
            origin_count = len(origin)
            initial_indices = set(range(origin_count))

            final_indices = set(initial_indices)
            for ev in sim.get('timeline', []):
                for nc in ev.get('new_cases', []):
                    try:
                        final_indices.add(int(nc))
                    except (TypeError, ValueError):
                        pass

            initial_coords = [(get_lat(t), get_lon(t)) for i, t in enumerate(all_trees) if i in initial_indices]
            final_coords = [(get_lat(t), get_lon(t)) for i, t in enumerate(all_trees) if i in final_indices and i < len(all_trees)]

            r0 = effective_radius(c, initial_coords)
            r1 = effective_radius(c, final_coords)

            # Annualized radial spread rate. effective_radius returns meters; convert to ft.
            growth_ft = (r1 - r0) * 3.28084
            months = sim.get('total_months', 24)
            delta_months = max(1, months)
            if growth_ft > 0:
                yearly_rate = (growth_ft / delta_months) * 12
            else:
                yearly_rate = 0
            results.append((cid, round(yearly_rate, 2)))
            print(f"Cluster {cid}: {yearly_rate:.2f} ft/yr")
        except Exception as e:
            print(f"Cluster {cid}: ERROR - {e}")
    print("\nClusterID,SpreadRate_ft_per_yr")
    for cid, rate in results:
        print(f"{cid},{rate}")

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

    if results:
        avg = sum(r for _, r in results) / len(results)
        print(f"Average spread rate: {avg:.2f} ft/yr")
    else:
        print("No results to average.")

if __name__ == "__main__":
    main()
