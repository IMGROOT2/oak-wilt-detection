# Spatio-Temporal Modeling of Oak Wilt Transmission
## Using Physics-Based Infection Pressure and Gradient Boosting

### Project Overview
This research project develops a novel **Spatio-Temporal Infection Pressure Model** to predict the transmission of Oak Wilt (*Bretziella fagacearum*). Unlike traditional models that often rely on simple radius buffers (the "Cone of Uncertainty"), this approach applies physics-based principles—specifically the **Inverse Square Law**—to model the cumulative "Infection Pressure" aimed at healthy trees from all nearby infection sources.

Using a machine learning classifier (**Gradient Boosting**), the model integrates:
1.  **Infection Pressure:** $\sum \frac{k}{d^2}$ (Cumulative risk from all neighbors).
2.  **Local Density:** Clustering intensity of the disease.
3.  **Seasonality:** Time-of-year vectors (Month Sine/Cosine).

### Repository Structure

#### 📂 `notebooks/`
*   `01_Train_Infection_Pressure_Model.ipynb`: The primary research notebook. Contains the logic for feature engineering, training the Gradient Boosting Classifier, and evaluating performance using ROC-AUC and Confusion Matrices.

#### 📂 `scripts/`
*   `build_training_dataset.py`: Preprocessing script that converts raw geospatial data (Clusters) into a graph-based edge list suitable for training.
*   `train_infection_pressure_model.py`: Production training pipeline to generate the `.pkl` model artifacts.
*   `api_inference_server.py`: FastAPI backend that serves the live prediction engine. It performs real-time feature extraction on incoming geospatial queries.

#### 📂 `visuals/`
*   `live_inference.html`: Interactive Decision Support System (DSS). A web-based dashboard allowing arborists to run "Network Simulations" and "Historical Validations" to see how the disease might spread over 12 months.

### Methodology & Metrics
*   **Synthetic Negatives:** To train the classifier, we generate random "healthy" points in the forest to contrast against confirmed "infected" trees.
*   **Performance:** The model achieves an **ROC-AUC of ~0.84**, indicating strong discriminative ability.
*   **Precision vs. Recall:** The system prioritizes **Recall** (Sensitivity) to minimize "Missed Cases" (False Negatives), accepting a lower Precision (False Positives) which effectively serves as a safety buffer for high-risk zones.

### Usage
1.  **Start API:** `python scripts/api_inference_server.py`
2.  **Launch Dashboard:** Open `visuals/live_inference.html` in a browser.
3.  **Simulate:** Select "Network Simulation" to predict future spread.

---
*Developed for the detection and management of Oak Wilt in Central Texas.*
