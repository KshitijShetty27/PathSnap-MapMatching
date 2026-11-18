# PathSnap: Distributed Map-Matching & Route Reconstruction

**Course:** UE24CS352A — Machine Learning  
**Department of Computer Science and Engineering (AI/ML), PES University**  
**Team Members:**  
* **Kshitij Satish Shetty** — PES1UG24AM143  
* **Krrish Singla** — PES1UG24AM141  
* **Kota Videesha** — PES1UG24AM140  
* **Kshama** — PES1UG24AM142  

PathSnap is an end-to-end distributed machine learning system that reconstructs true vehicle trajectories by matching noisy GPS telemetry observations to road network graphs.

---

## System Architecture

PathSnap uses a decoupled client-server architecture designed for a two-laptop evaluation setup:

```
+-------------------------------------------------------------+
|                 LAPTOP 2: Evaluator UI (Streamlit)          |
|  - Dropdown of 10 Reserved Evaluation Trips                 |
|  - Submits Trip ID only (No ground-truth labels exposed)    |
|  - Renders Top-1 Accuracy, Route Continuity & Folium Map    |
+-------------------------------------------------------------+
                               |
                               | HTTP POST /evaluate {"trip_id": "00000015"}
                               v
+-------------------------------------------------------------+
|                 LAPTOP 1: Model Backend (FastAPI)           |
|  - Hosts trained Tuned HistGradientBoosting model weights   |
|  - Precomputes cKDTree spatial arc indexing (6.2 µs/query)  |
|  - Extracts 9 geometric & kinematic motion features         |
|  - Executes HMM Viterbi Sequential Route Reconstruction     |
|  - Evaluates server-side accuracy against internal .route   |
|  - Streams live request execution logs to console           |
+-------------------------------------------------------------+
```

---

## Project Structure

```
PathSnap/
├── data/
│   └── data_loader.py             # Loads .track, .nodes, .arcs, .route; metric projection
├── preprocessing/
│   └── preprocessing.py          # Arc discretization & cKDTree spatial retrieval
├── feature_engineering/
│   └── features.py               # 9 geometric and kinematic motion features
├── models/
│   ├── model_1/                  # Model 1: Logistic Regression
│   │   └── train.py
│   ├── model_2/                  # Model 2: K-Nearest Neighbors
│   │   └── train.py
│   ├── model_3/                  # Model 3: Decision Tree
│   │   └── train.py
│   ├── model_4/                  # Model 4: Random Forest
│   │   └── train.py
│   ├── model_5/                  # Model 5: HistGradientBoosting (Selected)
│   │   └── train.py
│   └── models.py                 # Candidate models definition
├── evaluation/
│   └── evaluation.py             # Accuracy, continuity, Macro F1, log loss, and latency
├── hyperparameter_optimization/
│   └── bayesian_optuna.py        # Optuna TPE Bayesian hyperparameter optimization
├── hmm_viterbi/
│   └── hmm_decoder.py            # HMM transition matrices & Viterbi route decoding
├── backend/
│   └── app.py                    # FastAPI model & evaluation server (Laptop 1)
├── frontend/
│   └── app.py                    # Streamlit evaluation dashboard (Laptop 2)
├── visualizations/
│   ├── visualizer.py             # Folium HTML map generator & plotting routines
│   └── folium_trip_00000000.html # Multi-layer interactive trajectory map
├── results/
│   ├── best_model.joblib         # Trained Bayesian-tuned model weights
│   ├── model_comparison.csv      # 5-model benchmark scorecard
│   ├── bayesian_trials.csv       # Optuna optimization trial history
│   ├── tuned_vs_untuned.csv      # Before vs after hyperparameter tuning
│   ├── hmm_evaluation.csv        # Classifier-only vs HMM decoding metrics
│   ├── evaluator_test_results.csv # Metrics across all 10 reserved evaluator trips
│   ├── model_comparison.png      # 4-panel multi-metric model comparison plot
│   ├── hmm_improvement.png       # Route continuity improvement bar chart
│   ├── route_reconstruction_00000000.png # Snapped trajectory visualization
│   ├── cmd_api_response.jpg      # Live CMD terminal response screenshot
│   └── ui_evaluator_dashboard.jpg # Live Streamlit UI evaluator screenshot
├── report/
│   └── PathSnap_Technical_Report.pdf # Consolidated 7-section Technical Report (PDF)
├── requirements.txt
└── README.md
```

---

## Installation & Dependencies

Requires Python 3.10+. Install dependencies with:

```bash
cd PathSnap_Project
pip install -r requirements.txt
```

---

## Running the Pipeline

To execute the complete pipeline (data loading, feature extraction, 5-model training, Bayesian tuning, HMM evaluation, and artifacts export):

```bash
python run_pipeline.py
```

To run individual model experiments:
```bash
python experiments/train_model1_logistic_regression.py
python experiments/train_model2_knn.py
python experiments/train_model3_decision_tree.py
python experiments/train_model4_random_forest.py
python experiments/train_model5_boosting.py
```

To test specifically on the 10 evaluator trips:
```bash
python test_evaluator_set.py
```

---

## Two-Laptop Evaluation Setup

### Step 1: Start the API on Laptop 1 (Server / Backend)

On **Laptop 1**, open terminal in `PathSnap_Project` and run:

```bash
start_laptop_1.bat
```
*(or run: `python -m uvicorn api.app:app --host 0.0.0.0 --port 8000`)*

The terminal will print Laptop 1's local IPv4 address (e.g. `10.8.120.208` or `192.168.x.x`). Keep this window open.

### Step 2: Start the UI on Laptop 2 (Client / Frontend)

On **Laptop 2**, copy the `Laptop_2_Client` folder and double-click:

```bash
start_laptop_2.bat
```
*(or run: `streamlit run ui/app.py`)*

1. In the sidebar on Laptop 2, enter Laptop 1's API URL: `http://<LAPTOP_1_IP>:8000`.
2. The status indicator will turn green: **Laptop 1 API Online**.
3. Select any trip from the dropdown (e.g., `00000015`, `00000049`, `00000087`).
4. Click **Run Map-Matching Evaluation**.
5. Live request execution logs stream to the Laptop 1 console. Top-1 Accuracy, Route Continuity, and the Folium interactive map render on Laptop 2.

---

## Experimental Results

### 1. Model Comparison Across 5 Algorithm Families (Student Test Set)

| Model Name | Algorithm Family | Student-Test Top-1 Acc (%) | Route Continuity (%) | Macro F1 | Log Loss | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | 71.42% | 87.39% | 0.7911 | 0.1426 | 0.007 ms/obs |
| **K-Nearest Neighbors** | Instance-Based | 69.00% | 84.93% | 0.7962 | 0.2224 | 2.496 ms/obs |
| **Decision Tree** | Tree-Based | 70.62% | 85.92% | 0.8091 | 0.1734 | 0.002 ms/obs |
| **Random Forest** | Ensemble (Bagging) | 73.03% | 88.52% | 0.8192 | 0.1254 | 0.222 ms/obs |
| **HistGradientBoosting** | Gradient Boosting | **73.28%** | **88.19%** | **0.8202** | **0.1243** | **0.019 ms/obs** |

*Selected Model:* **HistGradientBoosting** — highest accuracy, lowest log loss, and 11.6x faster inference than Random Forest.

### 2. Bayesian Hyperparameter Optimization (Optuna TPE)

- **Search Space:** `max_iter` [40, 160], `learning_rate` [0.03, 0.2], `max_leaf_nodes` [15, 63], `max_depth` [5, 15], `min_samples_leaf` [10, 50], `l2_regularization` [0.0, 5.0].
- **Best Hyperparameters:** `max_iter=140`, `learning_rate=0.0456`, `max_leaf_nodes=17`, `max_depth=7`, `min_samples_leaf=18`, `l2_regularization=4.1341`.
- **Validation Accuracy (GroupKFold):** 93.97%.
- **Tuned Test Performance:** Observation Top-1 Accuracy: **73.41%**, Route Continuity: **88.41%**, Macro F1: **0.8218**, Log Loss: **0.1239**.

### 3. Reserved Evaluator Test Set Performance (10 Trips)

| Trip ID | Observations | Classifier Acc (%) | HMM Top-1 Acc (%) | HMM Continuity (%) | Sequence Recovery (Jaccard) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `00000015` | 6,854 | 55.11% | 59.53% | **100.00%** | 41.72% |
| `00000017` | 1,346 | 85.14% | 82.76% | 96.65% | 79.33% |
| `00000023` | 458 | 84.06% | 86.03% | 97.59% | 60.22% |
| `00000030` | 654 | 75.99% | 76.91% | 97.40% | 61.36% |
| `00000049` | 2,642 | 94.47% | **95.99%** | **99.47%** | **87.69%** |
| `00000059` | 1,031 | 88.75% | 86.61% | 98.25% | 78.99% |
| `00000067` | 916 | 70.74% | 74.34% | **100.00%** | 75.70% |
| `00000073` | 1,144 | 88.55% | **94.58%** | **100.00%** | **87.78%** |
| `00000087` | 1,399 | 90.71% | **97.36%** | **99.36%** | **97.39%** |
| `00000094` | 3,230 | 73.50% | 80.53% | **99.85%** | 79.57% |
| **Average** | **1,967** | **80.70%** | **83.46%** | **98.86%** | **74.97%** |

### 4. Sequential Route Reconstruction Summary

| Decoding Strategy | Observation Top-1 Acc (%) | Route Continuity (%) | Sequence Recovery (Jaccard) |
| :--- | :---: | :---: | :---: |
| **Classifier-Only (Argmax Top-1)** | 80.70% | 87.62% | Discontinuous |
| **Classifier + HMM (Viterbi Decoding)** | **83.46%** *(up to 97.4%)* | **98.86%** *(up to 100%)* | **74.97%** *(up to 97.4%)* |

HMM Viterbi decoding enforces physical road network graph adjacency, eliminating disconnected candidate jumps and achieving **98.86% average route continuity** across all 10 evaluator trips.
