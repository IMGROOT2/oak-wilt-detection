// Browser-side scorer for the gradient-boosted transmission model.
//
// Reproduces sklearn's GradientBoostingClassifier.predict_proba on the JSON written by
// prediction_system/export_model_json.py. A boosted ensemble is a plain sum over regression
// trees, so this is the same arithmetic sklearn does, not an approximation — the exporter
// asserts agreement to ~1e-11 before it will write the file.

class GBMModel {
    constructor(spec) {
        this.featureNames = spec.feature_names;
        this.nTrees = spec.n_trees;
        this.learningRate = spec.learning_rate;
        this.baseScore = spec.base_score;

        // Typed arrays so traversal stays in a monomorphic fast path; the simulation
        // scores a few thousand trees per healthy tree per month.
        this.treeOffsets = Int32Array.from(spec.tree_offsets);
        this.feature = Int32Array.from(spec.feature);
        this.threshold = Float64Array.from(spec.threshold);
        this.left = Int32Array.from(spec.left);
        this.right = Int32Array.from(spec.right);
        this.value = Float64Array.from(spec.value);
    }

    static async load(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Could not load model (HTTP ${res.status})`);
        const spec = await res.json();
        if (spec.format !== 'wiltcast-gbm-1') {
            throw new Error(`Unsupported model format: ${spec.format}`);
        }
        return new GBMModel(spec);
    }

    /** Sum of leaf values across all trees, in log-odds space. */
    decisionFunction(row) {
        const { feature, threshold, left, right, value, treeOffsets } = this;
        let raw = this.baseScore;

        for (let t = 0; t < this.nTrees; t++) {
            let node = treeOffsets[t];
            // -1 in the feature array marks a leaf.
            while (feature[node] !== -1) {
                node = row[feature[node]] <= threshold[node] ? left[node] : right[node];
            }
            raw += this.learningRate * value[node];
        }
        return raw;
    }

    /** Probability of the positive (infected) class. */
    predictProba(row) {
        return 1 / (1 + Math.exp(-this.decisionFunction(row)));
    }
}

window.GBMModel = GBMModel;
