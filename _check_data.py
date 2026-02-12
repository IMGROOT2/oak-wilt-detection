import pandas as pd
import numpy as np

members = pd.read_csv('data/oak_wilt_cluster_members.csv')
features = pd.read_csv('data/oak_wilt_cluster_features.csv')

sizes = members.groupby('cluster_id').size().sort_values(ascending=False)
print('Top 10 clusters by size:')
print(sizes.head(10))

rates_all = []
rates_plausible = []
for _, row in features.iterrows():
    yr_span = row['year_span']
    radius_km = row['radius_km']
    if yr_span > 0 and radius_km > 0:
        rate_ft_yr = (radius_km * 3280.84) / yr_span
        rates_all.append(rate_ft_yr)
        if 10 <= rate_ft_yr <= 300:
            rates_plausible.append(rate_ft_yr)

print(f'\nAll clusters mean: {np.mean(rates_all):.2f} ft/yr (n={len(rates_all)})')
print(f'Plausible (10-300) mean: {np.mean(rates_plausible):.2f} ft/yr (n={len(rates_plausible)})')
plaus20 = [r for r in rates_all if 20 <= r <= 300]
print(f'Plausible (20-300) mean: {np.mean(plaus20):.2f} ft/yr (n={len(plaus20)})')

mid = sizes[(sizes >= 10) & (sizes <= 25)]
print(f'\nMid-size clusters (10-25 trees):')
for cid in mid.index[:8]:
    c = members[members['cluster_id']==cid]
    years = c['INSPECTION_YEAR']
    print(f'  Cluster {cid}: {len(c)} trees, {years.min()}-{years.max()}')
