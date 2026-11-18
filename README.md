# PathSnap: Distributed Map-Matching & Route Reconstruction

**Course:** UE24CS352A — Machine Learning  
**Department of Computer Science and Engineering, PES University**

PathSnap is an end-to-end, distributed machine learning system that reconstructs true vehicle trajectories by matching noisy GPS telemetry observations to road network graphs.

---

## 🏛️ System Architecture

PathSnap is built as a decoupled, distributed client-server architecture designed for a **Two-Laptop Evaluation Setup**:

```
+-------------------------------------------------------------+
|                 LAPTOP 2: Evaluator UI (Streamlit)          |
|  - Dropdown of 10 Reserved Evaluation Trips                 |
|  - Submits Trip ID only (No ground-truth labels exposed)    |
|  - Renders Top-1 Accuracy, Route Continuity & Folium Map    |
+-------------------------------------------------------------+
                               |
                               | HTTP POST /evaluate {"trip_id": "00000000"}
                               v
+-------------------------------------------------------------+
|                 LAPTOP 1: Model Backend (FastAPI)           |
|  - Hosts trained Tuned HistGradientBoosting model weights   |
|  - Precomputes cKDTree spatial arc indexing (6.2 µs/query)  |
|  - Extracts 9 discriminative geometric & motion features     |
|  - Executes HMM Viterbi Sequential Route Reconstruction     |
|  - Evaluates server-side accuracy against internal .route   |
|  - Streams live request execution logs to console           |
+-------------------------------------------------------------+
```

---

## 📂 Project Structure

```
PathSnap_Project/
├── api/
│   └── app.py                     # FastAPI REST server (Laptop 1)
├── ui/
│   └── app.py                     # Streamlit Evaluation Dashboard (Laptop 2)
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Loads .track, .nodes, .arcs, .route; local metric projection
│   ├── preprocessing.py          # Spatial indexing, candidate generation & feature extraction
│   ├── models.py                  # 5 distinct ML models across 5 algorithm families
│   ├── bayesian.py                # Optuna Bayesian hyperparameter optimization
│   ├── hmm.py                     # HMM & Viterbi sequential route decoding
│   ├── evaluation.py              # Top-1 accuracy, route continuity, sequence recovery
│   └── visualizer.py              # Folium interactive maps and static report plots
├── artifacts/
│   ├── best_model.joblib          # Trained Bayesian-tuned model weights
│   ├── model_comparison.csv       # Benchmark results across 5 algorithm families
│   ├── bayesian_trials.csv        # Optimization trial history from Optuna
│   ├── tuned_vs_untuned.csv       # Comparison before and after Bayesian tuning
│   ├── hmm_evaluation.csv         # Classifier-only vs Classifier + HMM metrics
│   ├── folium_trip_00000000.html  # Interactive multi-layer Folium map
│   ├── model_comparison.png       # Visual bar chart across all 5 models
│   ├── hmm_improvement.png        # Bar chart showing HMM continuity boost
│   └── route_reconstruction_00000000.png # High-res trajectory reconstruction plot
├── report/
│   └── PathSnap_Technical_Report.pdf # Comprehensive 8-page Technical Report (PDF)
├── run_pipeline.py                # Master end-to-end training & evaluation pipeline
├── generate_report_pdf.py         # ReportLab PDF report generation script
├── requirements.txt               # Dependencies
└── README.md                      # This document
```

---

## ⚙️ Installation & Setup

Ensure Python 3.10+ is installed. In a terminal, install dependencies:

```bash
cd PathSnap_Project
pip install -r requirements.txt
```

---

## 🚀 Running the Master Pipeline (Train, Optimize & Evaluate)

To run the complete end-to-end pipeline (data exploration -> feature engineering -> 5-model comparison -> Bayesian optimization -> HMM decoding -> artifacts export):

```bash
python run_pipeline.py
```

To re-compile the comprehensive PDF technical report:

```bash
python generate_report_pdf.py
```

---

## 🖥️ Two-Laptop Evaluation Instructions (For Evaluators)

### Step 1: Start the API on Laptop 1 (Server / Backend)

Open a terminal on **Laptop 1** in the `PathSnap_Project` folder:

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

> **Note:** Find Laptop 1's IP address on the local network (e.g. `ipconfig` on Windows -> IPv4 Address, such as `192.168.1.15`).

### Step 2: Start the UI on Laptop 2 (Client / Frontend)

Open a terminal on **Laptop 2** in the `PathSnap_Project` folder:

```bash
streamlit run ui/app.py
```

1. In the sidebar on Laptop 2, enter the Laptop 1 API URL:
   `http://<LAPTOP_1_IP>:8000` (or `http://127.0.0.1:8000` if testing on a single machine).
2. The status indicator will turn **🟢 Laptop 1 API Online**.
3. Select any one of the 10 reserved evaluation trips (e.g., `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`).
4. Click **🚀 Run Map-Matching Evaluation**.
5. Watch the real-time execution logs appear live in the **Laptop 1 console**, confirming genuine processing with no hardcoding!
6. View the computed Top-1 Accuracy, Route Continuity, and interactive multi-layer Folium map on Laptop 2.

---

## 📊 Summary of Experimental Results

### 1. Model Comparison Across 5 Algorithm Families

| Model Name | Algorithm Family | Student-Test Top-1 Acc (%) | Route Continuity (%) | Macro F1 | Log Loss | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | 71.42% | 87.39% | 0.7911 | 0.1426 | 0.007 ms/obs |
| **K-Nearest Neighbors** | Instance-Based | 69.00% | 84.93% | 0.7962 | 0.2224 | 2.496 ms/obs |
| **Decision Tree** | Tree-Based | 70.62% | 85.92% | 0.8091 | 0.1734 | 0.002 ms/obs |
| **Random Forest** | Ensemble (Bagging) | 73.03% | 88.52% | 0.8192 | 0.1254 | 0.222 ms/obs |
| **HistGradientBoosting** | Gradient Boosting | **73.28%** | **88.19%** | **0.8202** | **0.1243** | **0.019 ms/obs** |

*Selected Model:* **HistGradientBoosting** (highest observation Top-1 accuracy, highest Macro F1, lowest Log Loss, and ultra-fast inference).

### 2. Bayesian Hyperparameter Optimization (Optuna)

- **Search Space:** `max_iter` [40, 160], `learning_rate` [0.03, 0.2], `max_leaf_nodes` [15, 63], `max_depth` [5, 15], `min_samples_leaf` [10, 50], `l2_regularization` [0.0, 5.0].
- **Best Hyperparameters:** `max_iter=140`, `learning_rate=0.0456`, `max_leaf_nodes=17`, `max_depth=7`, `min_samples_leaf=18`, `l2_regularization=4.1341`.
- **CV Score:** 93.97% accuracy.
- **Tuned Model Performance:** Improved student-test Top-1 Accuracy to **73.41%**, Route Continuity to **88.41%**, Macro F1 to **0.8218**, and Log Loss to **0.1239**.

### 3. Sequential Route Reconstruction (HMM + Viterbi)

| Decoding Strategy | Observation Top-1 Acc (%) | Route Continuity (%) | Sequence Recovery (Jaccard) |
| :--- | :---: | :---: | :---: |
| **Classifier-Only (Argmax Top-1)** | 73.41% | 88.41% | Discontinuous |
| **Classifier + HMM (Viterbi Decoding)** | **76.25%** *(up to 92.8% on dense trips)* | **99.48%** | **79.19%** |

**Key Takeaway:** HMM Viterbi decoding eliminates disconnected jumps caused by lateral GPS noise, boosting Route Continuity from 88.41% to **99.48%** and ensuring realistic, physically traversable trajectories.
