# WiltCast

Oak wilt prediction system for Austin, TX. Uses city inspection records (1986-2024) and a gradient-boosted model to forecast where oak wilt is likely to spread next.

For more information, visit [wiltcast.ruhangupta.com](https://wiltcast.ruhangupta.com).

## Overview

WiltCast models oak wilt spread using an inverse-square gravity pressure function over spatiotemporal clusters of confirmed infections. The pipeline clusters 1,672 historical infection records with ST-DBSCAN, generates negative samples via phantom point placement, enriches each sample with NASA POWER weather data, and trains a GradientBoostingClassifier on 9 features. The trained model achieves an AUC of 0.963 on held-out data.

A FastAPI server exposes the model for live inference, and a browser-based interface lets users place trees on a map and run month-by-month spread simulations with real weather conditions.

The mean radial spread rate computed from the 64 longitudinal Austin clusters is **75.34 ft/year** (71.26 ft/year after filtering outliers to the 20-200 ft/year biologically plausible range), which aligns with the 75 ft/year reference value from Appel et al. (1989).

## Repository structure

```
prediction_system/
    build_dataset.py        # graph dataset with distance features and negative sampling
    enrich_data.py          # NASA POWER weather enrichment per cluster
    train_model.py          # GBM training and evaluation
    run_simulation.py       # annualized spread rate computation across clusters
    inference_server.py     # FastAPI backend for live and historical inference
    web_interface/          # Leaflet + Tailwind frontend

validation_study/
    verify_spread_rate_75ft.py
    notebooks/analysis/
        data_cleaning.ipynb   # raw -> data_cleaned.csv
        clustering.ipynb      # ST-DBSCAN clusters and per-cluster spread rates
        spread_analysis.ipynb # publication figures and the 75 ft/yr verification

data/                       # cleaned records, cluster features, simulation outputs, TAMU polygons
models/                     # saved model artifacts (committed for one-command startup)
visuals/                    # figure-generation scripts; outputs land in visuals/output/ (gitignored)
```

## Quickstart

```bash
pip install -r requirements.txt
```

Run the backend and frontend in two terminals:

```bash
npm run backend    # FastAPI on http://localhost:8000
npm run frontend   # static server on http://localhost:8080
```

Then open [http://localhost:8080/live_inference.html](http://localhost:8080/live_inference.html).

The trained model is committed under `models/`, so the backend boots without a training step. To retrain from scratch:

```bash
python prediction_system/train_model.py
```

To re-run the spread-rate simulation across clusters:

```bash
python prediction_system/run_simulation.py
```

## Replication

The full pipeline can be replicated end-to-end from `data/data_original.csv`:

1. `validation_study/notebooks/analysis/data_cleaning.ipynb` produces `data/data_cleaned.csv`.
2. `validation_study/notebooks/analysis/clustering.ipynb` produces `data/oak_wilt_cluster_members.csv` and `data/oak_wilt_cluster_features.csv`.
3. `python prediction_system/enrich_data.py` adds NASA POWER weather to the clusters.
4. `python prediction_system/train_model.py` fits the GBM and writes `models/graph_transmission_model_pressure.pkl`.
5. `python validation_study/verify_spread_rate_75ft.py` prints the historical baseline and simulation spread rates for direct comparison against the 75 ft/year literature value.
6. `validation_study/notebooks/analysis/spread_analysis.ipynb` regenerates the publication figures.

## Data

Source data comes from the City of Austin's oak wilt inspection records. Weather variables are pulled from NASA POWER. Texas A&M's statewide oak wilt detection polygons (2020-2026) are stored under `data/OW_2020_2026/` for cross-region validation.

## License

MIT
