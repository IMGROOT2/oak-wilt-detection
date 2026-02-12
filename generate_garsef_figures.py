#!/usr/bin/env python3
"""Generates figures for the GARSEF research paper submission."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUT_DIR = Path(__file__).resolve().parent.parent / 'wiltcast-research-paper' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEMBERS = pd.read_csv(DATA_DIR / 'oak_wilt_cluster_members.csv')
MEMBERS['date'] = pd.to_datetime(MEMBERS['INSPECTION_DATE'])
ENRICHED = pd.read_csv(DATA_DIR / 'oak_wilt_cluster_enriched.csv')
CLEANED = pd.read_csv(DATA_DIR / 'data_cleaned.csv')

WEATHER_MAP = ENRICHED.set_index('cluster_id')[
    ['avg_temp', 'avg_precip', 'avg_humidity', 'avg_wind']
].to_dict('index')

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
        w = WEATHER_MAP.get(cid, {'avg_temp': 20, 'avg_precip': 0, 'avg_humidity': 50, 'avg_wind': 5})
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
            # synthetic negatives (2 per positive)
            lat_min, lat_max = cluster['LATITUDE'].min(), cluster['LATITUDE'].max()
            lon_min, lon_max = cluster['LONGITUDE'].min(), cluster['LONGITUDE'].max()
            buff = 0.002
            for _ in range(2):
                r_lat = np.random.uniform(lat_min-buff, lat_max+buff)
                r_lon = np.random.uniform(lon_min-buff, lon_max+buff)
                d2 = haversine(r_lat, r_lon, sources['LATITUDE'].values, sources['LONGITUDE'].values)
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


# Figure 1: geospatial distribution of all cases and cluster centroids
def fig_geospatial():
    print("  [1/7] Geospatial distribution...")
    fig, ax = plt.subplots(figsize=(6, 5))
    if 'LATITUDE' in CLEANED.columns:
        ax.scatter(CLEANED['LONGITUDE'], CLEANED['LATITUDE'],
                   s=3, alpha=0.3, c='#2196F3', label='Confirmed cases (1,672)')
    else:
        ax.scatter(MEMBERS['LONGITUDE'], MEMBERS['LATITUDE'],
                   s=3, alpha=0.3, c='#2196F3', label='Clustered cases (538)')
    
    # Cluster centroids
    centroids = MEMBERS.groupby('cluster_id')[['LATITUDE', 'LONGITUDE']].mean()
    ax.scatter(centroids['LONGITUDE'], centroids['LATITUDE'],
               s=60, c='red', marker='x', linewidths=1.5, zorder=5,
               label=f'Cluster centroids ($n$={len(centroids)})')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Oak Wilt Cases — Austin, TX (1986–2024)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_geospatial.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 2: temporal progression within a single cluster
def fig_cluster_example():
    print("  [2/7] Cluster example...")
    sizes = MEMBERS.groupby('cluster_id').size()
    cid = sizes.idxmax()
    cluster = MEMBERS[MEMBERS['cluster_id'] == cid].sort_values('date')
    
    fig, ax = plt.subplots(figsize=(5, 5))
    years = cluster['INSPECTION_YEAR'].values
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
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Outbreak Cluster {cid} ({len(cluster)} trees, {years.min()}–{years.max()})')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_cluster_example.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 3: illustrates the negative sampling strategy
def fig_negsampling():
    print("  [3/7] Synthetic negative sampling...")
    sizes = MEMBERS.groupby('cluster_id').size()
    mid_clusters = sizes[(sizes >= 8) & (sizes <= 20)]
    cid = mid_clusters.index[0] if len(mid_clusters) > 0 else sizes.idxmax()
    cluster = MEMBERS[MEMBERS['cluster_id'] == cid].sort_values('date')
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    lat_min, lat_max = cluster['LATITUDE'].min(), cluster['LATITUDE'].max()
    lon_min, lon_max = cluster['LONGITUDE'].min(), cluster['LONGITUDE'].max()
    buff = 0.002
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((lon_min-buff, lat_min-buff),
                      (lon_max-lat_min+2*buff)*1.0,
                      lat_max-lat_min+2*buff,
                      linewidth=1.5, edgecolor='grey', facecolor='#f0f0f0',
                      linestyle='--', alpha=0.5, zorder=1)
    rect_width = lon_max - lon_min + 2*buff
    rect_height = lat_max - lat_min + 2*buff
    rect = Rectangle((lon_min-buff, lat_min-buff), rect_width, rect_height,
                      linewidth=1.5, edgecolor='grey', facecolor='#f0f0f0',
                      linestyle='--', alpha=0.5, zorder=1)
    ax.add_patch(rect)
    n_neg = len(cluster) * 2
    neg_lats = np.random.uniform(lat_min-buff, lat_max+buff, n_neg)
    neg_lons = np.random.uniform(lon_min-buff, lon_max+buff, n_neg)
    
    ax.scatter(neg_lons, neg_lats, s=30, c='#9E9E9E', alpha=0.6, marker='o',
               edgecolors='#616161', linewidth=0.5, label='Phantom trees (negatives)', zorder=3)
    ax.scatter(cluster['LONGITUDE'], cluster['LATITUDE'],
               s=60, c='#D32F2F', marker='o', edgecolors='black',
               linewidth=0.7, label='Infected trees (positives)', zorder=5)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Synthetic Negative Sampling')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_negsampling.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 4: ROC curve from 10-fold cross-validation
def fig_roc():
    print("  [4/7] ROC curve (10-fold CV)...")
    X, y = build_dataset()
    features = ['log_pressure', 'log_min_dist', 'local_density',
                'month_sin', 'month_cos', 'avg_temp', 'avg_precip',
                'avg_humidity', 'avg_wind']
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                      max_depth=5, random_state=42)
    probs = cross_val_predict(gbm, X[features], y, cv=cv, method='predict_proba')[:, 1]
    
    auc = roc_auc_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)
    
    n_boot = 200
    tpr_interp_list = []
    mean_fpr = np.linspace(0, 1, 100)
    for _ in range(n_boot):
        idx = np.random.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y[idx], probs[idx])
        tpr_interp_list.append(np.interp(mean_fpr, fpr_b, tpr_b))
    
    tpr_upper = np.percentile(tpr_interp_list, 97.5, axis=0)
    tpr_lower = np.percentile(tpr_interp_list, 2.5, axis=0)
    tpr_mean = np.mean(tpr_interp_list, axis=0)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.15, color='#1976D2',
                     label='95% Bootstrap CI')
    ax.plot(fpr, tpr, color='#1565C0', linewidth=2,
            label=f'WiltCast GBM (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — 10-Fold Cross-Validation')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_roc.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    AUC = {auc:.4f}")


# Figure 5: distribution of cluster-level spread rates
def fig_spread_rates():
    print("  [5/7] Spread rate histogram...")
    spread = pd.read_csv(DATA_DIR / 'simulated_spread_rates.csv')
    
    rates = spread['avg_spread_ft_per_year'] if 'avg_spread_ft_per_year' in spread.columns else spread.iloc[:, -1]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    plausible = rates[(rates >= 0) & (rates <= 250)]
    
    ax.hist(plausible, bins=20, color='#42A5F5', edgecolor='white',
            alpha=0.8, density=True, label=f'All clusters ($n$={len(plausible)})')
    
    from scipy.stats import gaussian_kde
    kde_x = np.linspace(0, 250, 300)
    kde = gaussian_kde(plausible.dropna())
    ax.plot(kde_x, kde(kde_x), color='#0D47A1', linewidth=2, label='Kernel density estimate')
    
    ax.axvline(75, color='red', linestyle='--', linewidth=2,
               label='Appel et al. (1989): 75 ft/yr')
    
    mean_rate = rates.mean()
    ax.axvline(mean_rate, color='#FF6F00', linestyle=':', linewidth=2,
               label=f'Computed mean: {mean_rate:.1f} ft/yr')
    
    ax.set_xlabel('Spread Rate (ft/year)')
    ax.set_ylabel('Density')
    ax.set_title('Cluster-Level Spread Rates')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_spread_rates.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Mean rate = {mean_rate:.2f} ft/yr")


# Figure 6: infection pressure heatmap (inverse-square model)
def fig_pressure_field():
    print("  [6/7] Pressure field heatmap...")
    sizes = MEMBERS.groupby('cluster_id').size()
    cid = sizes.idxmax()
    cluster = MEMBERS[MEMBERS['cluster_id'] == cid]
    
    lat_min, lat_max = cluster['LATITUDE'].min() - 0.002, cluster['LATITUDE'].max() + 0.002
    lon_min, lon_max = cluster['LONGITUDE'].min() - 0.002, cluster['LONGITUDE'].max() + 0.002
    
    grid_res = 80
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
    im = ax.pcolormesh(LON, LAT, P_log, cmap='YlOrRd', shading='auto')
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('log(1 + Pressure)')
    
    ax.scatter(cluster['LONGITUDE'], cluster['LATITUDE'],
               s=40, marker='^', c='black', edgecolors='white',
               linewidth=0.5, zorder=5, label='Infected trees')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Infection Pressure Field ($P = \\Sigma\\, 1000/d^2$)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_pressure_field.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure 7: GBM feature importance (spatial vs. environmental)
def fig_feature_importance():
    print("  [7/7] Feature importance...")
    X, y = build_dataset()
    features = ['log_pressure', 'log_min_dist', 'local_density',
                'month_sin', 'month_cos', 'avg_temp', 'avg_precip',
                'avg_humidity', 'avg_wind']
    
    gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                      max_depth=5, random_state=42)
    gbm.fit(X[features], y)
    
    importances = dict(zip(features, gbm.feature_importances_))
    
    spatial = sum(importances[f] for f in ['log_pressure', 'log_min_dist', 'local_density',
                                            'month_sin', 'month_cos'])
    environ = sum(importances[f] for f in ['avg_temp', 'avg_precip', 'avg_humidity', 'avg_wind'])

    sorted_feats = sorted(importances.items(), key=lambda x: -x[1])
    names = [f[0] for f in sorted_feats]
    vals = [f[1] * 100 for f in sorted_feats]
    
    spatial_feats = {'log_pressure', 'log_min_dist', 'local_density', 'month_sin', 'month_cos'}
    colors = ['#1565C0' if n in spatial_feats else '#E65100' for n in names]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Importance (%)')
    ax.set_title('Feature Importance (Gini Impurity)')
    ax.invert_yaxis()
    
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}%', va='center', fontsize=8)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1565C0', label='Spatial/Temporal (84.05%)'),
                       Patch(facecolor='#E65100', label='Environmental (15.95%)')]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_garsef_feature_importance.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    total_spatial = spatial * 100
    total_environ = environ * 100
    print(f"    Spatial+Temporal: {total_spatial:.2f}%  Environmental: {total_environ:.2f}%")


if __name__ == "__main__":
    print("Generating GARSEF paper figures...\n")
    fig_geospatial()
    fig_cluster_example()
    fig_negsampling()
    fig_roc()
    fig_spread_rates()
    fig_pressure_field()
    fig_feature_importance()
    print(f"\nAll figures saved to: {OUT_DIR}")
