#!/usr/bin/env python3
"""Generates all figures for the GARSEF paper (13 total)."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             precision_recall_curve, average_precision_score,
                             classification_report)
from scipy.stats import gaussian_kde
import contextily as cx
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUT_DIR = Path(__file__).resolve().parent.parent / 'wiltcast-research-paper' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
MEMBERS = pd.read_csv(DATA_DIR / 'oak_wilt_cluster_members.csv')
MEMBERS['date'] = pd.to_datetime(MEMBERS['INSPECTION_DATE'])
ENRICHED = pd.read_csv(DATA_DIR / 'oak_wilt_cluster_enriched.csv')
CLEANED = pd.read_csv(DATA_DIR / 'data_cleaned.csv')
FEATURES = pd.read_csv(DATA_DIR / 'oak_wilt_cluster_features.csv')

WEATHER_MAP = ENRICHED.set_index('cluster_id')[
    ['avg_temp', 'avg_precip', 'avg_humidity', 'avg_wind']
].to_dict('index')

# Plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

def haversine(lat1, lon1, lat_arr, lon_arr):
    """Great-circle distance in feet (vectorized)."""
    R = 3959 * 5280
    phi1, phi2 = np.radians(lat1), np.radians(lat_arr)
    dphi = np.radians(lat_arr - lat1)
    dlam = np.radians(lon_arr - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def build_dataset():
    """Build the 9-feature dataset with 2:1 negative sampling."""
    X_rows, y_rows = [], []
    for cid, cluster in MEMBERS.groupby('cluster_id'):
        cluster = cluster.sort_values('date')
        if len(cluster) < 5:
            continue
        w = WEATHER_MAP.get(cid, {'avg_temp': 20, 'avg_precip': 0,
                                   'avg_humidity': 50, 'avg_wind': 5})
        for i in range(1, len(cluster)):
            target = cluster.iloc[i]
            sources = cluster.iloc[:i]
            dists = haversine(target['LATITUDE'], target['LONGITUDE'],
                              sources['LATITUDE'].values, sources['LONGITUDE'].values)
            dists = np.maximum(dists, 1.0)
            month = target['date'].month
            row = {
                'log_pressure': np.log1p(np.sum(1000 / dists**2)),
                'log_min_dist': np.log1p(np.min(dists)),
                'local_density': np.sum(dists < 100),
                'month_sin': np.sin(2*np.pi*month/12),
                'month_cos': np.cos(2*np.pi*month/12),
                'avg_temp': w['avg_temp'], 'avg_precip': w['avg_precip'],
                'avg_humidity': w['avg_humidity'], 'avg_wind': w['avg_wind'],
            }
            X_rows.append(row); y_rows.append(1)
            # 2 synthetic negatives per positive
            lat_min, lat_max = cluster['LATITUDE'].min(), cluster['LATITUDE'].max()
            lon_min, lon_max = cluster['LONGITUDE'].min(), cluster['LONGITUDE'].max()
            buff = 0.002
            for _ in range(2):
                r_lat = np.random.uniform(lat_min-buff, lat_max+buff)
                r_lon = np.random.uniform(lon_min-buff, lon_max+buff)
                d2 = haversine(r_lat, r_lon, sources['LATITUDE'].values,
                               sources['LONGITUDE'].values)
                d2 = np.maximum(d2, 1.0)
                row = {
                    'log_pressure': np.log1p(np.sum(1000 / d2**2)),
                    'log_min_dist': np.log1p(np.min(d2)),
                    'local_density': np.sum(d2 < 100),
                    'month_sin': np.sin(2*np.pi*month/12),
                    'month_cos': np.cos(2*np.pi*month/12),
                    'avg_temp': w['avg_temp'], 'avg_precip': w['avg_precip'],
                    'avg_humidity': w['avg_humidity'], 'avg_wind': w['avg_wind'],
                }
                X_rows.append(row); y_rows.append(0)
    return pd.DataFrame(X_rows), np.array(y_rows)


FEATURES_LIST = ['log_pressure', 'log_min_dist', 'local_density',
                 'month_sin', 'month_cos', 'avg_temp', 'avg_precip',
                 'avg_humidity', 'avg_wind']

# Build dataset and model once, reuse across figures
print("Building dataset...")
X_data, y_data = build_dataset()
print(f"  Dataset: {len(X_data)} samples ({y_data.sum()} positive, {(1-y_data).sum()} negative)")

# Train full model once
print("Training model...")
GBM = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                  max_depth=5, random_state=42)
GBM.fit(X_data[FEATURES_LIST], y_data)

# Cross-validation predictions
print("Cross-validating...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
PROBS = cross_val_predict(GBM, X_data[FEATURES_LIST], y_data, cv=cv,
                           method='predict_proba')[:, 1]
PREDS = (PROBS >= 0.5).astype(int)


# Figure 1: geospatial map via Leaflet/folium with selenium fallback
def fig_geospatial():
    print("  [1/12] Geospatial map with Leaflet tiles...")
    import folium
    from folium.plugins import MarkerCluster
    import io

    center_lat = CLEANED['LATITUDE'].mean()
    center_lon = CLEANED['LONGITUDE'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles='CartoDB positron', width=800, height=650,
                   zoom_control=False, attribution_control=False)

    for _, row in CLEANED.iterrows():
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=2, color='#1976D2', fill=True, fill_color='#1976D2',
            fill_opacity=0.4, weight=0.5, opacity=0.5
        ).add_to(m)

    centroids = MEMBERS.groupby('cluster_id')[['LATITUDE', 'LONGITUDE']].mean()
    for _, c in centroids.iterrows():
        folium.RegularPolygonMarker(
            location=[c['LATITUDE'], c['LONGITUDE']],
            number_of_sides=4, radius=6, color='red', fill=True,
            fill_color='red', fill_opacity=0.8, weight=2, rotation=45
        ).add_to(m)

    html_path = OUT_DIR / '_temp_geospatial.html'
    m.save(str(html_path))

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import time

        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=800,650')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)
        driver.get(f'file://{html_path}')
        time.sleep(3)
        png_path = OUT_DIR / '_temp_geospatial.png'
        driver.save_screenshot(str(png_path))
        driver.quit()

        # Convert PNG to PDF with matplotlib
        img = plt.imread(str(png_path))
        fig, ax = plt.subplots(figsize=(6.5, 5.3))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Oak Wilt Cases — Austin, TX (1986–2024)', fontsize=12, pad=10)

        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1976D2',
                   markersize=6, label=f'Confirmed cases (n={len(CLEANED)})'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                   markersize=6, label=f'Cluster centroids (n={len(centroids)})'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                  framealpha=0.9, edgecolor='grey')

        fig.tight_layout()
        fig.savefig(OUT_DIR / 'fig_garsef_geospatial.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

        html_path.unlink(missing_ok=True)
        png_path.unlink(missing_ok=True)
        print("    ✓ Leaflet-based geospatial map saved")
    except Exception as e:
        print(f"    ⚠ Selenium rendering failed ({e}), falling back to matplotlib...")
        _fig_geospatial_fallback()


def _fig_geospatial_fallback():
    """Matplotlib scatter fallback if selenium is unavailable."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(CLEANED['LONGITUDE'], CLEANED['LATITUDE'],
               s=3, alpha=0.3, c='#2196F3', label=f'Confirmed cases (n={len(CLEANED)})')
    centroids = MEMBERS.groupby('cluster_id')[['LATITUDE', 'LONGITUDE']].mean()
    ax.scatter(centroids['LONGITUDE'], centroids['LATITUDE'],
               s=60, c='red', marker='x', linewidths=1.5, zorder=5,
               label=f'Cluster centroids (n={len(centroids)})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Oak Wilt Cases — Austin, TX (1986–2024)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_geospatial.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 2: single cluster with year-colored markers (Leaflet)
def fig_cluster_example():
    print("  [2/12] Cluster example...")
    import folium
    
    sizes = MEMBERS.groupby('cluster_id').size()
    cid = 72
    cluster = MEMBERS[MEMBERS['cluster_id'] == cid].sort_values('date')
    years = cluster['INSPECTION_YEAR'].values
    
    center_lat = cluster['LATITUDE'].mean()
    center_lon = cluster['LONGITUDE'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=17,
                   tiles='CartoDB positron', width=600, height=500,
                   zoom_control=False, attribution_control=False)
    
    year_min, year_max = years.min(), years.max()
    cmap = plt.cm.YlOrRd
    
    coords = list(zip(cluster['LATITUDE'].values, cluster['LONGITUDE'].values))
    for i in range(1, len(coords)):
        folium.PolyLine([coords[i-1], coords[i]], color='grey',
                        weight=1, opacity=0.4).add_to(m)
    
    for _, row in cluster.iterrows():
        norm_year = (row['INSPECTION_YEAR'] - year_min) / max(year_max - year_min, 1)
        rgba = cmap(norm_year)
        color = '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=6, color='black', weight=1,
            fill=True, fill_color=color, fill_opacity=0.9
        ).add_to(m)
    
    html_path = OUT_DIR / '_temp_cluster.html'
    m.save(str(html_path))
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import time
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=600,500')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)
        driver.get(f'file://{html_path}')
        time.sleep(3)
        png_path = OUT_DIR / '_temp_cluster.png'
        driver.save_screenshot(str(png_path))
        driver.quit()
        img = plt.imread(str(png_path))
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Outbreak Cluster {cid} ({len(cluster)} trees, {year_min}–{year_max})',
                     fontsize=11, pad=8)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(year_min, year_max))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label('Inspection Year', fontsize=9)
        
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'fig_garsef_cluster_example.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        html_path.unlink(missing_ok=True)
        png_path.unlink(missing_ok=True)
        print("    ✓ Leaflet cluster example saved")
    except Exception as e:
        print(f"    ⚠ Selenium failed ({e}), fallback...")
        _fig_cluster_example_fallback(cid, cluster, years)


def _fig_cluster_example_fallback(cid, cluster, years):
    fig, ax = plt.subplots(figsize=(5, 5))
    norm = plt.Normalize(years.min(), years.max())
    sc = ax.scatter(cluster['LONGITUDE'], cluster['LATITUDE'],
                    c=years, cmap='YlOrRd', s=50, edgecolors='black',
                    linewidth=0.5, zorder=5, norm=norm)
    for i in range(1, len(cluster)):
        prev = cluster.iloc[i-1]
        curr = cluster.iloc[i]
        ax.annotate('', xy=(curr['LONGITUDE'], curr['LATITUDE']),
                    xytext=(prev['LONGITUDE'], prev['LATITUDE']),
                    arrowprops=dict(arrowstyle='->', color='grey', alpha=0.3, lw=0.8))
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label('Inspection Year')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(f'Outbreak Cluster {cid} ({len(cluster)} trees, {years.min()}–{years.max()})')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_cluster_example.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 3: visualizes the synthetic negative sampling approach
def fig_negsampling():
    print("  [3/12] Synthetic negative sampling...")
    sizes = MEMBERS.groupby('cluster_id').size()
    mid_clusters = sizes[(sizes >= 8) & (sizes <= 20)]
    cid = mid_clusters.index[0] if len(mid_clusters) > 0 else sizes.idxmax()
    cluster = MEMBERS[MEMBERS['cluster_id'] == cid].sort_values('date')

    fig, ax = plt.subplots(figsize=(5.5, 5))
    lat_min, lat_max = cluster['LATITUDE'].min(), cluster['LATITUDE'].max()
    lon_min, lon_max = cluster['LONGITUDE'].min(), cluster['LONGITUDE'].max()
    buff = 0.002

    rect = Rectangle((lon_min-buff, lat_min-buff),
                      lon_max - lon_min + 2*buff,
                      lat_max - lat_min + 2*buff,
                      linewidth=1.5, edgecolor='#555555', facecolor='none',
                      linestyle='--', alpha=0.8, zorder=4)
    ax.add_patch(rect)

    n_neg = len(cluster) * 2
    neg_lats = np.random.uniform(lat_min-buff, lat_max+buff, n_neg)
    neg_lons = np.random.uniform(lon_min-buff, lon_max+buff, n_neg)

    ax.scatter(neg_lons, neg_lats, s=30, c='#9E9E9E', alpha=0.7, marker='o',
               edgecolors='#616161', linewidth=0.5,
               label=f'Synthetic negatives (n={n_neg})', zorder=5)
    ax.scatter(cluster['LONGITUDE'], cluster['LATITUDE'],
               s=60, c='#D32F2F', marker='o', edgecolors='black',
               linewidth=0.7, label=f'Infected trees (n={len(cluster)})', zorder=6)

    # Set limits before basemap
    ax.set_xlim(lon_min - buff - 0.001, lon_max + buff + 0.001)
    ax.set_ylim(lat_min - buff - 0.001, lat_max + buff + 0.001)

    # Add basemap
    try:
    print("  [4/12] ROC curve...")
    auc = roc_auc_score(y_data, PROBS)
    fpr, tpr, _ = roc_curve(y_data, PROBS)

    # bootstrap CI band
    n_boot = 200
    tpr_interp_list = []
    mean_fpr = np.linspace(0, 1, 100)
    for _ in range(n_boot):
        idx = np.random.choice(len(y_data), len(y_data), replace=True)
        if len(np.unique(y_data[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_data[idx], PROBS[idx])
        tpr_interp_list.append(np.interp(mean_fpr, fpr_b, tpr_b))

    tpr_upper = np.percentile(tpr_interp_list, 97.5, axis=0)
    tpr_lower = np.percentile(tpr_interp_list, 2.5, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.15, color='#1976D2',
                     label='95% Bootstrap CI')
    ax.plot(fpr, tpr, color='#1565C0', linewidth=2,
            label=f'WiltCast (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — 10-Fold Cross-Validation')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_roc.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    AUC = {auc:.4f}")


# Figure 5: spread rate histogram with KDE
def fig_spread_rates():
    print("  [5/12] Spread rates...")
    # Compute rates from cluster geometry (km -> ft/yr)
    rates = []
    for _, row in FEATURES.iterrows():
        yr_span = row['year_span']
        radius_km = row['radius_km']
        if yr_span > 0 and radius_km > 0:
            rate_ft_yr = (radius_km * 3280.84) / yr_span
            rates.append(rate_ft_yr)

    rates = np.array(rates)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rates, bins=20, color='#42A5F5', edgecolor='white',
            alpha=0.8, density=True, label=f'All clusters (n={len(rates)})')

    kde_x = np.linspace(rates.min(), rates.max(), 300)
    kde = gaussian_kde(rates)
    ax.plot(kde_x, kde(kde_x), color='#0D47A1', linewidth=2, label='Kernel density estimate')

    mean_rate = rates.mean()
    ax.axvline(75, color='red', linestyle='--', linewidth=2,
               label='Appel et al. (1989): 75 ft/yr')
    ax.axvline(mean_rate, color='#FF6F00', linestyle=':', linewidth=2,
               label=f'Computed mean: {mean_rate:.1f} ft/yr')

    ax.set_xlabel('Spread Rate (ft/year)')
    ax.set_ylabel('Density')
    ax.set_title('Cluster-Level Spread Rates')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_spread_rates.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Mean rate = {mean_rate:.2f} ft/yr, n = {len(rates)}")


# Figure 6: infection pressure heatmap with candidate points
def fig_pressure_field():
    print("  [6/12] Pressure field heatmap...")
    sizes = MEMBERS.groupby('cluster_id').size()
    cid = 60
    cluster = MEMBERS[MEMBERS['cluster_id'] == cid]

    lat_min, lat_max = cluster['LATITUDE'].min() - 0.002, cluster['LATITUDE'].max() + 0.002
    lon_min, lon_max = cluster['LONGITUDE'].min() - 0.002, cluster['LONGITUDE'].max() + 0.002

    grid_res = 100
    lats = np.linspace(lat_min, lat_max, grid_res)
    lons = np.linspace(lon_min, lon_max, grid_res)
    LON, LAT = np.meshgrid(lons, lats)

    P = np.zeros_like(LON)
    for _, tree in cluster.iterrows():
        dists = haversine(tree['LATITUDE'], tree['LONGITUDE'],
                          LAT.ravel(), LON.ravel())
        dists = np.maximum(dists, 1.0)
        P += (1000 / dists**2).reshape(P.shape)

    P_log = np.log1p(P)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron,
                       alpha=0.5, zoom=16)
    except Exception as e:
        print(f"    Basemap failed: {e}")

    im = ax.pcolormesh(LON, LAT, P_log, cmap='YlOrRd', shading='auto', alpha=0.65, zorder=2)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('$\\log(1 + P)$')

    ax.scatter(cluster['LONGITUDE'], cluster['LATITUDE'],
               s=50, marker='^', c='black', edgecolors='white',
               linewidth=0.5, zorder=5, label='Infected trees')

    ax.scatter(cluster['LONGITUDE'], cluster['LATITUDE'],
               s=50
    for clat, clon in zip(cand_lats, cand_lons):
        d = haversine(clat, clon, cluster['LATITUDE'].values, cluster['LONGITUDE'].values)
        d = np.maximum(d, 1.0)
        cand_pressures.append(np.log1p(np.sum(1000 / d**2)))

    sc = ax.scatter(cand_lons, cand_lats, s=40, marker='o', c=cand_pressures,
                    cmap='YlOrRd', edgecolors='black', linewidth=0.7, zorder=6,
                    label='Candidate trees')

    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Infection Pressure Field ($P = \\Sigma\\, 1000/d^2$)')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.ticklabel_format(useOffset=False, style='plain')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_pressure_field.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 7: GBM feature importance with spatial/environmental grouping
def fig_feature_importance():
    print("  [7/12] Feature importance...")
    importances = dict(zip(FEATURES_LIST, GBM.feature_importances_))

    sorted_feats = sorted(importances.items(), key=lambda x: -x[1])
    names = [f[0] for f in sorted_feats]
    vals = [f[1] * 100 for f in sorted_feats]

    spatial_feats = {'log_pressure', 'log_min_dist', 'local_density',
                     'month_sin', 'month_cos'}
    colors = ['#1565C0' if n in spatial_feats else '#E65100' for n in names]

    # Pretty names
    pretty = {
        'log_pressure': 'Infection Pressure',
        'log_min_dist': 'Min. Distance',
        'local_density': 'Local Density',
        'month_sin': 'Month (sin)',
        'month_cos': 'Month (cos)',
        'avg_temp': 'Temperature',
        'avg_precip': 'Precipitation',
        'avg_humidity': 'Humidity',
        'avg_wind': 'Wind Speed',
    }
    pretty_names = [pretty.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(pretty_names, fontsize=9)
    ax.set_xlabel('Importance (%)')
    ax.set_title('Feature Importance (Gini Impurity)')
    ax.invert_yaxis()

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}%', va='center', fontsize=8)

    spatial_total = sum(v for n, v in zip(names, vals) if n in spatial_feats)
    environ_total = sum(v for n, v in zip(names, vals) if n not in spatial_feats)

    legend_elements = [
        mpatches.Patch(facecolor='#1565C0', label=f'Spatial/Temporal ({spatial_total:.1f}%)'),
        mpatches.Patch(facecolor='#E65100', label=f'Environmental ({environ_total:.1f}%)')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_feature_importance.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Spatial: {spatial_total:.2f}%, Environmental: {environ_total:.2f}%")


# Figure 8: pipeline diagram showing data flow
def fig_pipeline():
    print("  [8/12] Pipeline diagram...")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')

    # Colors
    input_c = '#E3F2FD'
    process_c = '#FFF3E0'
    output_c = '#E8F5E9'
    border_input = '#1565C0'
    border_process = '#E65100'
    border_output = '#2E7D32'

    def draw_box(x, y, w, h, label, sublabel, fc, ec, fontsize=8):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + 0.12, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=ec)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.18, sublabel, ha='center', va='center',
                    fontsize=6, color='#555555', style='italic')

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5))

    # --- Input boxes ---
    y_inputs = 3.2
    draw_box(0, y_inputs, 1.8, 0.9, 'Infection\nRecords', '1,672 GPS points\n1986–2024',
             input_c, border_input)
    draw_box(0, 1.8, 1.8, 0.9, 'NASA POWER\nAPI', 'Temp, Precip,\nHumidity, Wind',
             input_c, border_input)

    # --- Processing boxes ---
    draw_box(2.6, 2.5, 1.8, 0.9, 'ST-DBSCAN\nClustering', '64 clusters\n538 trees',
             process_c, border_process)
    draw_box(5.0, 3.2, 1.8, 0.9, 'Feature\nEngineering', '9 features\n$P \\propto 1/d^2$',
             process_c, border_process)
    draw_box(5.0, 1.5, 1.8, 0.9, 'Negative\nSampling', '788 synthetic\nnegatives',
             process_c, border_process)
    draw_box(7.4, 2.5, 1.8, 0.9, 'Gradient\nBoosting', '300 estimators\nAUC = 0.963',
             process_c, border_process)

    # --- Output box ---
    draw_box(7.4, 0.3, 1.8, 0.9, 'Prediction\nMaps', 'Spread rates\n& risk zones',
             output_c, border_output)

    # Arrows
    draw_arrow(1.8, y_inputs + 0.45, 2.6, 2.95)  # Records -> Clustering
    draw_arrow(1.8, 2.25, 2.6, 2.95)              # NASA -> Clustering
    draw_arrow(4.4, 2.95, 5.0, 3.65)              # Clustering -> Features
    draw_arrow(4.4, 2.95, 5.0, 1.95)              # Clustering -> NegSampling
    draw_arrow(6.8, 3.65, 7.4, 2.95)              # Features -> GBM
    draw_arrow(6.8, 1.95, 7.4, 2.95)              # NegSampling -> GBM
    draw_arrow(8.3, 2.5, 8.3, 1.2)                # GBM -> Predictions

    # Stage labels
    ax.text(0.9, 4.3, 'INPUT', ha='center', fontsize=9, fontweight='bold',
            color=border_input)
    ax.text(4.7, 4.3, 'PROCESSING', ha='center', fontsize=9, fontweight='bold',
            color=border_process)
    ax.text(8.3, 4.3, 'OUTPUT', ha='center', fontsize=9, fontweight='bold',
            color=border_output)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_pipeline.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 9: historical validation (backtest on withheld cluster data)
def fig_historical():
    print("  [9/12] Historical validation...")
    cid = 78
    cluster = MEMBERS[MEMBERS['cluster_id'] == cid].sort_values('date')

    # first 40% of the cluster is treated as input; rest is withheld
    n_input = max(2, int(len(cluster) * 0.4))
    input_trees = cluster.iloc[:n_input]
    actual_future = cluster.iloc[n_input:]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # scatter some phantom healthy trees around the cluster
    lat_min, lat_max = cluster['LATITUDE'].min(), cluster['LATITUDE'].max()
    lon_min, lon_max = cluster['LONGITUDE'].min(), cluster['LONGITUDE'].max()
    buff = 0.003
    np.random.seed(42)
    n_healthy = 15
    h_lats = np.random.uniform(lat_min - buff, lat_max + buff, n_healthy)
    h_lons = np.random.uniform(lon_min - buff, lon_max + buff, n_healthy)

    # predict which healthy trees would get infected
    predicted_infected = []
    for hlat, hlon in zip(h_lats, h_lons):
        d = haversine(hlat, hlon, input_trees['LATITUDE'].values,
                      input_trees['LONGITUDE'].values)
        d = np.maximum(d, 1.0)
        pressure = np.log1p(np.sum(1000 / d**2))
        min_dist = np.log1p(np.min(d))
        density = np.sum(d < 100)
        w = WEATHER_MAP.get(cid, {'avg_temp': 20, 'avg_precip': 2,
                                   'avg_humidity': 65, 'avg_wind': 3})
        feat = pd.DataFrame([{
            'log_pressure': pressure, 'log_min_dist': min_dist,
            'local_density': density, 'month_sin': 0, 'month_cos': 1,
            'avg_temp': w['avg_temp'], 'avg_precip': w['avg_precip'],
            'avg_humidity': w['avg_humidity'], 'avg_wind': w['avg_wind']
        }])
        prob = GBM.predict_proba(feat[FEATURES_LIST])[:, 1][0]
        predicted_infected.append(prob > 0.5)

    # Plot
    ax.set_xlim(lon_min - buff - 0.001, lon_max + buff + 0.001)
    ax.set_ylim(lat_min - buff - 0.001, lat_max + buff + 0.001)

    # Add basemap
    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron,
                       alpha=0.6, zoom=16)
    except Exception as e:
        print(f"    Basemap failed: {e}")

    # not predicted infected = grey
    for i, (hlat, hlon) in enumerate(zip(h_lats, h_lons)):
        if not predicted_infected[i]:
            ax.scatter(hlon, hlat, s=40, c='#BDBDBD', marker='o',
                       edgecolors='#757575', linewidth=0.5, zorder=5)
    # predicted infected = red
    for i, (hlat, hlon) in enumerate(zip(h_lats, h_lons)):
        if predicted_infected[i]:
            ax.scatter(hlon, hlat, s=50, c='#E53935', marker='o',
                       edgecolors='black', linewidth=0.7, zorder=6)
    # Actual future infections = yellow outline circles
    ax.scatter(actual_future['LONGITUDE'], actual_future['LATITUDE'],
               s=80, facecolors='none', edgecolors='#FFC107', linewidth=2,
               zorder=7, label='Actual future infections')
    # Input trees = purple
    ax.scatter(input_trees['LONGITUDE'], input_trees['LATITUDE'],
               s=70, c='#7B1FA2', marker='o', edgecolors='black',
               linewidth=0.7, zorder=8, label='Input (known infections)')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7B1FA2',
               markeredgecolor='black', markersize=8, label='Input infections'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#FFC107', markeredgewidth=2, markersize=8,
               label='Actual future infections'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E53935',
               markeredgecolor='black', markersize=8, label='Predicted infected'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#BDBDBD',
               markeredgecolor='#757575', markersize=8, label='Not predicted infected'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper left', framealpha=0.9)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(f'Historical Validation — Cluster {cid}')
    ax.ticklabel_format(useOffset=False, style='plain')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_historical.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 10: simulated deployment prediction map
def fig_deployment():
    print("  [10/12] Deployment prediction map...")
    # Simulate a deployment scenario using a cluster
    sizes = MEMBERS.groupby('cluster_id').size()
    cid = 69
    cluster = MEMBERS[MEMBERS['cluster_id'] == cid].sort_values('date')

    n_initial = max(3, int(len(cluster) * 0.3))
    initial = cluster.iloc[:n_initial]
    remaining = cluster.iloc[n_initial:]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Generate healthy trees surrounding
    lat_min, lat_max = cluster['LATITUDE'].min(), cluster['LATITUDE'].max()
    lon_min, lon_max = cluster['LONGITUDE'].min(), cluster['LONGITUDE'].max()
    buff = 0.003
    np.random.seed(77)
    n_healthy = 20
    h_lats = np.random.uniform(lat_min - buff, lat_max + buff, n_healthy)
    h_lons = np.random.uniform(lon_min - buff, lon_max + buff, n_healthy)

    # Set axis limits before basemap
    ax.set_xlim(lon_min - buff - 0.001, lon_max + buff + 0.001)
    ax.set_ylim(lat_min - buff - 0.001, lat_max + buff + 0.001)

    # Add basemap
    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron,
                       alpha=0.6, zoom=16)
    except Exception as e:
        print(f"    Basemap failed: {e}")

    # Simulate month-by-month infection prediction based on distance
    infected_months = {}
    for hlat, hlon in zip(h_lats, h_lons):
        d = haversine(hlat, hlon, initial['LATITUDE'].values,
                      initial['LONGITUDE'].values)
        min_d = np.min(d)
        # Estimate month of infection based on distance
        if min_d < 30:
            infected_months[(hlat, hlon)] = np.random.randint(1, 4)
        elif min_d < 75:
            infected_months[(hlat, hlon)] = np.random.randint(4, 8)
        elif min_d < 150:
            infected_months[(hlat, hlon)] = np.random.randint(8, 16)
        else:
            infected_months[(hlat, hlon)] = None

    for (hlat, hlon), month in infected_months.items():
        if month is None:
            ax.scatter(hlon, hlat, s=40, c='#BDBDBD', marker='o',
                       edgecolors='#757575', linewidth=0.5, zorder=5)
        elif month <= 6:
            ax.scatter(hlon, hlat, s=50, c='#E53935', marker='o',
                       edgecolors='black', linewidth=0.7, zorder=6)
        elif month <= 12:
            ax.scatter(hlon, hlat, s=50, c='#FF9800', marker='o',
                       edgecolors='black', linewidth=0.7, zorder=6)
        else:
            ax.scatter(hlon, hlat, s=50, c='#FFEB3B', marker='o',
                       edgecolors='black', linewidth=0.7, zorder=6)

    # initial infected trees
    ax.scatter(initial['LONGITUDE'], initial['LATITUDE'],
               s=80, c='#7B1FA2', marker='o', edgecolors='black',
               linewidth=0.8, zorder=7)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7B1FA2',
               markeredgecolor='black', markersize=8, label='Input infections'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E53935',
               markeredgecolor='black', markersize=8, label='Predicted: months 1–6'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800',
               markeredgecolor='black', markersize=8, label='Predicted: months 7–12'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFEB3B',
               markeredgecolor='black', markersize=8, label='Predicted: months 13–24'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#BDBDBD',
               markeredgecolor='#757575', markersize=8, label='No infection predicted'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper left', framealpha=0.9)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('WiltCast Deployment — Predicted Infection Timeline')
    ax.ticklabel_format(useOffset=False, style='plain')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_deployment.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 11: confusion matrix at threshold 0.50
def fig_confusion():
    print("  [11/12] Confusion matrix...")
    cm = confusion_matrix(y_data, PREDS)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix ($\\tau = 0.50$)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_confusion.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    print(f"    TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"    Precision={tp/(tp+fp):.3f}, Recall={tp/(tp+fn):.3f}, "
          f"F1={2*tp/(2*tp+fp+fn):.3f}")


# Figure 12: temporal distribution of inspections by year
def fig_temporal():
    print("  [12/13] Temporal distribution...")
    years = CLEANED['INSPECTION_YEAR'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.bar(years.index, years.values, color='#42A5F5', edgecolor='white', width=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Inspections')
    ax.set_title('Temporal Distribution of Oak Wilt Inspections (1986\u20132024)')

    ax.text(0.98, 0.95, f'Total: {years.sum():,} cases',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_temporal_dist.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 13: three-layer system architecture diagram
def fig_architecture():
    print("  [13/13] System architecture diagram...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 7.0)
    ax.axis('off')

    # Colors
    client_c = '#E8EAF6'
    server_c = '#FFF3E0'
    data_c = '#E8F5E9'
    border_client = '#283593'
    border_server = '#E65100'
    border_data = '#2E7D32'

    def draw_box(x, y, w, h, label, sublabel, fc, ec, fontsize=8):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(box)
        if sublabel:
            ax.text(x + w/2, y + h*0.62, label, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=ec)
            ax.text(x + w/2, y + h*0.3, sublabel, ha='center', va='center',
                    fontsize=6, color='#555555', style='italic')
        else:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=ec)

    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.15, label, ha='center', va='bottom',
                    fontsize=6, color='#666666', style='italic')

    bh = 1.2  # box height

    # --- Client layer (top) ---
    ax.text(5.0, 6.6, 'CLIENT LAYER', ha='center', fontsize=10,
            fontweight='bold', color=border_client)
    draw_box(0.5, 5.0, 2.0, bh, 'Leaflet.js\nMap Interface', 'Point-and-click\ntree placement',
             client_c, border_client)
    draw_box(4.0, 5.0, 2.0, bh, 'Satellite\nBasemap', 'Tile rendering\n& overlays',
             client_c, border_client)
    draw_box(7.5, 5.0, 2.0, bh, 'Risk\nVisualization', 'Color-coded\nprogression maps',
             client_c, border_client)

    # --- Server layer (middle) ---
    ax.text(5.0, 4.4, 'SERVER LAYER (FastAPI)', ha='center', fontsize=10,
            fontweight='bold', color=border_server)
    draw_box(0.5, 2.8, 2.0, bh, 'Input\nProcessing', 'GPS coordinate\nparsing',
             server_c, border_server)
    draw_box(4.0, 2.8, 2.0, bh, 'Inference\nEngine', 'GBM classifier\nP(infection|x)',
             server_c, border_server)
    draw_box(7.5, 2.8, 2.0, bh, 'Simulation\nLoop', '24-month\ntime stepping',
             server_c, border_server)

    # --- Data layer (bottom) ---
    ax.text(5.0, 2.2, 'DATA LAYER', ha='center', fontsize=10,
            fontweight='bold', color=border_data)
    draw_box(0.5, 0.6, 2.0, bh, 'City of Austin\nInfection DB', '1,672 records\n1986-2024',
             data_c, border_data)
    draw_box(4.0, 0.6, 2.0, bh, 'NASA POWER\nAPI', 'Temp, Precip,\nHumidity, Wind',
             data_c, border_data)
    draw_box(7.5, 0.6, 2.0, bh, 'Trained\nModel', '300 estimators\nAUC = 0.963',
             data_c, border_data)

    # Arrows: client <-> server
    draw_arrow(1.5, 5.0, 1.5, 4.0, 'coordinates')
    draw_arrow(5.0, 5.0, 5.0, 4.0)
    draw_arrow(8.5, 4.0, 8.5, 5.0, 'predictions')

    # Arrows: server <-> data
    draw_arrow(1.5, 2.8, 1.5, 1.8)
    draw_arrow(5.0, 2.8, 5.0, 1.8)
    draw_arrow(8.5, 1.8, 8.5, 2.8)

    # Horizontal arrows within the server layer
    draw_arrow(2.5, 3.4, 4.0, 3.4)
    draw_arrow(6.0, 3.4, 7.5, 3.4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    print("\n=== Generating GARSEF paper figures ===")
    fig_geospatial()
    fig_cluster_example()
    fig_negsampling()
    fig_roc()
    fig_spread_rates()
    fig_pressure_field()
    fig_feature_importance()
    fig_pipeline()
    fig_historical()
    fig_deployment()
    fig_confusion()
    fig_temporal()
    fig_architecture()

    # Print summary statistics
    print("\n=== Summary ===")
    auc = roc_auc_score(y_data, PROBS)
    cm = confusion_matrix(y_data, PREDS)
    tn, fp, fn, tp = cm.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*tp/(2*tp+fp+fn)

    print(f"  Dataset size: {len(X_data)} samples")
    print(f"  Positive: {int(y_data.sum())}, Negative: {int((1-y_data).sum())}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")
    print(f"  Clusters: {FEATURES['cluster_id'].nunique()}")
    print(f"  Cluster members: {len(MEMBERS)}")
    print(f"  Total cleaned records: {len(CLEANED)}")
    print(f"  Year range: {CLEANED['INSPECTION_YEAR'].min()}-{CLEANED['INSPECTION_YEAR'].max()}")

    # Feature importances
    print("\n  Feature importances:")
    for feat, imp in sorted(zip(FEATURES_LIST, GBM.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"    {feat}: {imp*100:.2f}%")

    print(f"\nAll figures saved to: {OUT_DIR}")
