# WiltCast

Oak wilt prediction system for Austin, TX. Uses city inspection records (1986-2024) and a gradient-boosted model to forecast where oak wilt is likely to spread next.

For more information, visit [wiltcast.ruhangupta.com](https://wiltcast.ruhangupta.com).

## Overview

WiltCast models oak wilt spread using an inverse-square gravity pressure function over spatiotemporal clusters of confirmed infections. The pipeline clusters 1,672 historical infection records with ST-DBSCAN, generates negative samples via phantom point placement, enriches each sample with NASA POWER weather data, and trains a GradientBoostingClassifier on 9 features. The trained model achieves an AUC of 0.963 on held-out data.

A FastAPI server exposes the model for live inference, and a browser-based interface lets users place trees on a map and run month-by-month spread simulations with real weather conditions.

## Repository Structure

```
prediction_system/
    build_dataset.py      # graph dataset with distance features and negative sampling
    enrich_data.py        # NASA POWER weather enrichment per cluster
    train_model.py        # GBM training and evaluation
    run_simulation.py     # annualized spread rate computation across clusters
    inference_server.py   # FastAPI backend for live and historical inference
    web_interface/        # Leaflet + Tailwind frontend

validation_study/
    verify_spread_rate_75ft.py
    notebooks/analysis/   # exploratory analysis and visualization notebooks

data/                     # cleaned records, cluster features, simulation outputs
models/                   # saved model artifacts
visuals/                  # generated figures

generate_garsef_figures.py      # figures for the GARSEF research paper
generate_all_garsef_figures.py  # extended figure set (13 figures)
```

## Quickstart

```bash
pip install -r requirements.txt
```

**Train the model:**

```bash
python prediction_system/train_model.py
```

**Run the inference server:**

```bash
python prediction_system/inference_server.py
```

Then open `prediction_system/web_interface/live_inference.html` in a browser.
## Data

Source data comes from the City of Austin's oak wilt inspection records. Weather variables are pulled from NASA POWER.

## License

MIT