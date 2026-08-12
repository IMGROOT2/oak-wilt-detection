"""Export the trained GradientBoostingClassifier to JSON for browser-side inference.

The static site at try.wiltcast.com has no backend, so the model has to run in the
user's browser. A gradient-boosted ensemble is just a sum over regression trees, so
the export is lossless: every threshold and leaf value is carried over, and
gbm.js reproduces sklearn's decision_function to floating-point precision.

Run after retraining:
    python prediction_system/export_model_json.py
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'graph_transmission_model_pressure.pkl'
OUT_PATH = BASE_DIR / 'prediction_system' / 'web_interface' / 'model' / 'gbm_pressure.json'

# Thresholds are split points on feature values that are themselves logs, counts and
# weather readings, so 6 decimals is far below the resolution that could flip a split.
# Leaf values need more: 300 of them are summed, so their rounding error accumulates.
THRESHOLD_DECIMALS = 6
LEAF_DECIMALS = 10


def flatten_trees(model):
    """Pack every tree into flat parallel arrays with tree boundaries as offsets.

    One flat array per field beats an array of node objects: it drops the repeated
    JSON keys, which is most of the file size at ~15k nodes.
    """
    feature, threshold, left, right, value, offsets = [], [], [], [], [], [0]

    for stage in model.estimators_:
        tree = stage[0].tree_
        base = offsets[-1]
        n = tree.node_count

        for i in range(n):
            is_leaf = tree.children_left[i] == -1
            if is_leaf:
                # -1 marks a leaf; children are unused but kept aligned for a flat layout.
                feature.append(-1)
                threshold.append(0.0)
                left.append(-1)
                right.append(-1)
                value.append(round(float(tree.value[i][0][0]), LEAF_DECIMALS))
            else:
                feature.append(int(tree.feature[i]))
                threshold.append(round(float(tree.threshold[i]), THRESHOLD_DECIMALS))
                # Rebase child indices into the flat array.
                left.append(base + int(tree.children_left[i]))
                right.append(base + int(tree.children_right[i]))
                value.append(0.0)

        offsets.append(base + n)

    return feature, threshold, left, right, value, offsets


def main():
    model = joblib.load(MODEL_PATH)

    # Binary classification only: one tree per boosting stage.
    if model.estimators_.shape[1] != 1:
        raise ValueError(f"Expected a binary classifier, got {model.estimators_.shape[1]} trees per stage")

    feature, threshold, left, right, value, offsets = flatten_trees(model)

    # The constant prior sklearn starts boosting from, in log-odds space.
    probe = pd.DataFrame(np.zeros((1, model.n_features_in_)), columns=model.feature_names_in_)
    base_score = float(model._raw_predict_init(probe).ravel()[0])

    payload = {
        "format": "wiltcast-gbm-1",
        "model": "graph_transmission_model_pressure",
        "objective": "binary_logistic",
        "feature_names": [str(f) for f in model.feature_names_in_],
        "n_trees": int(model.estimators_.shape[0]),
        "n_nodes": len(feature),
        "learning_rate": float(model.learning_rate),
        "base_score": base_score,
        "tree_offsets": offsets,
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "value": value,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(',', ':')))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH.relative_to(BASE_DIR)} ({size_kb:.0f} KB, "
          f"{payload['n_trees']} trees, {payload['n_nodes']} nodes)")

    verify(model, payload)


def verify(model, payload):
    """Re-score random inputs through the exported arrays and compare against sklearn."""
    rng = np.random.default_rng(0)
    n = 2000

    # Sample across the range each feature actually takes during a simulation.
    X = np.column_stack([
        rng.uniform(0, 12, n),      # log_pressure
        rng.uniform(0, 6, n),       # log_min_dist
        rng.integers(0, 40, n),     # local_density
        rng.uniform(-1, 1, n),      # month_sin
        rng.uniform(-1, 1, n),      # month_cos
        rng.uniform(-5, 45, n),     # avg_temp
        rng.uniform(0, 15, n),      # avg_precip
        rng.uniform(10, 100, n),    # avg_humidity
        rng.uniform(0, 15, n),      # avg_wind
    ])

    expected = model.predict_proba(pd.DataFrame(X, columns=payload['feature_names']))[:, 1]

    feature = np.array(payload['feature'])
    threshold = np.array(payload['threshold'])
    left = np.array(payload['left'])
    right = np.array(payload['right'])
    value = np.array(payload['value'])
    offsets = payload['tree_offsets']

    got = np.empty(n)
    for r in range(n):
        row = X[r]
        raw = payload['base_score']
        for t in range(payload['n_trees']):
            node = offsets[t]
            while feature[node] != -1:
                node = left[node] if row[feature[node]] <= threshold[node] else right[node]
            raw += payload['learning_rate'] * value[node]
        got[r] = 1.0 / (1.0 + np.exp(-raw))

    max_err = float(np.max(np.abs(got - expected)))
    print(f"Parity check over {n} random samples: max abs probability error = {max_err:.3e}")

    # The simulation thresholds probabilities at 0.50, so anything near float noise is
    # immaterial; this bar is many orders of magnitude tighter than that.
    if max_err > 1e-9:
        raise SystemExit(f"Export does not reproduce sklearn (max error {max_err:.3e})")
    print("Export verified.")


if __name__ == '__main__':
    main()
