import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
sim_path = DATA_DIR / 'simulated_spread_rates.csv'
feat_path = DATA_DIR / 'oak_wilt_cluster_features.csv'

def calculate_averages():
    # 1. Simulation-based rates (with weather + gravity model)
    if sim_path.exists():
        try:
            df = pd.read_csv(sim_path)
            if 'spread_ft_per_yr' in df.columns:
                print("--- Current Simulation ---")
                
                # All
                all_avg = df['spread_ft_per_yr'].mean()
                print(f"All Clusters Average: {all_avg:.2f} ft/yr (Count: {len(df)})")

                # Filtered
                filtered_df = df[(df['spread_ft_per_yr'] >= 20) & (df['spread_ft_per_yr'] <= 200)]
                filtered_avg = filtered_df['spread_ft_per_yr'].mean()
                print(f"Filtered (20-200 ft/yr) Average: {filtered_avg:.2f} ft/yr (Count: {len(filtered_df)})")
            else:
                print("Error: 'spread_ft_per_yr' column not found in simulated file.")
        except Exception as e:
            print(f"Error reading simulated results: {e}")
    else:
        print(f"Warning: {sim_path} does not exist.")

    print("") 

    # 2. Static cluster geometry rates (baseline, no simulation)
    if feat_path.exists():
        try:
            df_feat = pd.read_csv(feat_path)
            if 'spread_rate_km_per_year' in df_feat.columns:
                # Convert km/yr to ft/yr
                df_feat['spread_ft_per_yr'] = df_feat['spread_rate_km_per_year'] * 3280.84
                
                print("--- Historical Baseline ---")
                
                # All
                all_avg_h = df_feat['spread_ft_per_yr'].mean()
                print(f"All Clusters Average: {all_avg_h:.2f} ft/yr (Count: {len(df_feat)})")
                
                # Filtered
                filtered_h = df_feat[(df_feat['spread_ft_per_yr'] >= 20) & (df_feat['spread_ft_per_yr'] <= 200)]
                filtered_avg_h = filtered_h['spread_ft_per_yr'].mean()
                print(f"Filtered (20-200 ft/yr) Average: {filtered_avg_h:.2f} ft/yr (Count: {len(filtered_h)})")
                
        except Exception as e:
            print(f"Error reading features file: {e}")
    else:
        print(f"Warning: {feat_path} does not exist.")

if __name__ == "__main__":
    calculate_averages()
