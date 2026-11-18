# PathSnap: Review 1 Comprehensive Master Guide
**UE24CS352A — Machine Learning | Department of CSE, PES University**
*Prepared for First Phase Evaluation (Review 1)*

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Exploratory Data Analysis (EDA) & Data Integrity](#2-exploratory-data-analysis-eda--data-integrity)
3. [Spatial Indexing & Candidate Generation](#3-spatial-indexing--candidate-generation)
4. [Feature Engineering & Feature Selection](#4-feature-engineering--feature-selection)
5. [Model Experimentation: 5 Distinct Algorithm Families](#5-model-experimentation-5-distinct-algorithm-families)
6. [Consolidated Model Comparison & Evaluation](#6-consolidated-model-comparison--evaluation)
7. [Bayesian Hyperparameter Optimization (Optuna)](#7-bayesian-hyperparameter-optimization-optuna)
8. [Individual Model Profiles (Compulsory Viva Questions)](#8-individual-model-profiles-compulsory-viva-questions)
9. [Team Member Ownership & Viva Defense Scripts](#9-team-member-ownership--viva-defense-scripts)
10. [Top 10 Review 1 Viva Trap Questions & Answers](#10-top-10-review-1-viva-trap-questions--answers)

---

## 1. Executive Summary & Problem Statement

### The Real-World Problem
GPS receivers in moving vehicles produce noisy spatial telemetry due to:
* **Multipath propagation:** Satellite radio signals bounce off tall urban structures and glass buildings.
* **Urban canyons & signal dropout:** Loss of line-of-sight satellites in tunnels, underpasses, and dense city streets.
* **Sampling discretization:** Trajectories sampled at finite intervals (~1 Hz).

Consequently, naive trajectory plotting shows vehicles **teleporting across buildings, driving through rivers, or jumping onto parallel opposite lanes**.

### The PathSnap Objective
Map-matching (or trajectory snapping) mathematically reconstructs the true sequence of road segments traversed by a vehicle. We formulate this as an end-to-end Machine Learning pipeline:
1. **Candidate Retrieval:** Find nearest plausible road segments within spatial radius.
2. **Feature Extraction:** Extract geometric and kinematic relationship features between GPS points and candidate arcs.
3. **Observation-Level Classification:** Train and tune ML classifiers to score the probability that a candidate arc is the true segment.
4. **Sequential Trajectory Snapping:** Model trajectory dynamics using road network graph topology.

---

## 2. Exploratory Data Analysis (EDA) & Data Integrity

### 2.1 Dataset Partitioning & Zero-Leakage Architecture
The dataset is split into three strictly isolated partitions:
* **Training Partition:** 80 trips. Used exclusively for feature extraction, baseline model fitting, cross-validation, and Bayesian hyperparameter search.
* **Student-Testing Partition:** 10 trips. Used for development-stage model evaluation and comparison.
* **Evaluator-Testing Partition:** 10 reserved trips. Kept strictly untouched for final grading.

> **Data Leakage Rule:** All transformations, scaling factors, and feature pipelines are fitted **strictly on training trips**. No trip is ever split across train and validation sets (Group-Aware splitting).

### 2.2 Raw File Types & Data Structures
Each trip folder contains four whitespace-separated, headerless files:
1. **`.track`**: GPS observations: `[longitude, latitude, timestamp]` (timestamp in seconds).
2. **`.nodes`**: Road network vertices: `[longitude, latitude]`. Line index represents `node_id` ($0, 1, ..., N-1$).
3. **`.arcs`**: Directed road segments: `[from_node_id, to_node_id]`. Line index represents `arc_id` ($0, 1, ..., M-1$).
4. **`.route`**: Ground-truth sequence of traversed `arc_id` values (chronological order).

### 2.3 Empirical Dataset Summary Statistics

| Property | Value | Engineering Implication |
| :--- | :--- | :--- |
| **Total Trips Available** | 90 trips (80 train + 10 student-test) | Sufficient sample size for multi-fold cross-validation |
| **Trajectory Length per Trip** | Mean = 2,318 points (Min: 821, Max: 7,243) | Trajectory sizes require sub-millisecond candidate lookup |
| **Trip Duration** | Mean = 3,062.2 s (51.0 minutes) | Covers extended highway and complex urban journeys |
| **Nominal Sampling Interval** | **1.0 Hz (96.94% exactly ~1.0 s)** | Permissible to compute instantaneous velocity vectors |
| **Missing / Corrupt Values** | **0 NaN / null values** | Clean telemetry; temporal dropouts handled dynamically |
| **Average Road Segment Length** | 94.6 meters (Min: 2.1 m, Max: 812.4 m) | Segments vary widely in length; midpoint indexing alone fails |

### 2.4 Coordinate System & Metric Projection
Because raw coordinates are spherical angular degrees, standard Euclidean distance formula $\sqrt{\Delta \text{lon}^2 + \Delta \text{lat}^2}$ produces severe directional and distance distortion.

**Our Metric Equirectangular Projection:**
All spatial coordinates are projected into local Cartesian coordinates $(x, y)$ in physical meters centered at the trajectory median latitude $\phi_{\text{ref}}$:
$$x = \lambda \times 111320 \times \cos(\phi_{\text{ref}})$$
$$y = \phi \times 110540$$
* Result: 1 meter in $x$ equals 1 meter in $y$ with sub-millimeter precision.

---

## 3. Spatial Indexing & Candidate Generation

### The Computational Bottleneck
A single trip road network contains over **165,000 arcs** and ~2,500 GPS points. A brute-force search requires $2500 \times 165000 \approx 4.1 \times 10^8$ distance computations (~35 seconds per trip).

### Our High-Performance Solution (`scipy.spatial.cKDTree`)
1. **Multi-Point Arc Sampling:** Rather than indexing only arc midpoints (which miss long segments > 100m), we sample each segment at 3 parametric intervals:
   $$P(\alpha) = u + \alpha \cdot (v - u), \quad \alpha \in \{0.20, 0.50, 0.80\}$$
2. **Dense Metric KD-Tree:** We construct a `cKDTree` on all sampled segment points.
3. **Query Performance:** Querying top-10 candidate arcs per GPS point takes only **42 milliseconds for an entire trip** ($O(\log M)$ time complexity).
4. **Candidate Recall:** Yields **98.25% candidate coverage** (Candidate Miss Rate under **1.75%**).

---

## 4. Feature Engineering & Feature Selection

For every GPS observation $z_t = (x_t, y_t, \text{time}_t)$ and candidate road segment $c_i = [u \to v]$, we extract **9 discriminative features**:

### The 9 Extracted Features:

1. **`dist_to_segment` (Continuous, Meters):**
   Perpendicular Euclidean distance from the GPS point to the line segment $[u, v]$:
   $$\text{dist} = \| z_t - (u + t^* (v - u)) \|$$
   *Most important baseline feature: true road segments are geometrically closest to the GPS receiver.*

2. **`proj_ratio` (Continuous, Clamped to $[0, 1]$):**
   Parametric projection scalar $t^* = \frac{(z_t - u) \cdot (v - u)}{\|v - u\|^2}$.
   *Distinguishes whether the vehicle is near the entry, middle, or exit of the road segment.*

3. **`dist_to_u` (Continuous, Meters):**
   Direct Euclidean distance from GPS point to the start intersection node $u$.

4. **`dist_to_v` (Continuous, Meters):**
   Direct Euclidean distance from GPS point to the end intersection node $v$.

5. **`arc_length` (Continuous, Meters):**
   Physical length of the road segment: $L = \|v - u\|$.
   *Captures road hierarchy: highways have long segments; urban side streets have short segments.*

6. **`heading_diff` (Continuous, Radians in $[0, \pi]$) — KEY DISCRIMINATOR:**
   Angular discrepancy between the vehicle's instantaneous travel direction $\theta_{\text{gps}}$ and the road arc's orientation $\theta_{\text{arc}}$:
   $$\Delta \theta = |\theta_{\text{gps}} - \theta_{\text{arc}}|, \quad \text{wrapped to } [0, \pi]$$
   *Essential for separating parallel opposite lanes: two parallel highway lanes may be only 4 meters apart, but opposing lanes have $\Delta \theta \approx 180^\circ$ while the correct lane has $\Delta \theta \approx 0^\circ$.*

7. **`gps_speed` (Continuous, $m/s$):**
   Estimated vehicle speed between consecutive points: $v = \frac{\|z_t - z_{t-1}\|}{\Delta t}$.

8. **`dist_rank` (Discrete / Integer: 0 to 9):**
   Rank order of the candidate sorted by perpendicular distance for that GPS point (0 = closest, 1 = second closest).
   *Provides relative scale-invariant candidate priority.*

9. **`time_delta` (Continuous, Seconds):**
   Sampling interval $\Delta t = \text{time}_t - \text{time}_{t-1}$. Captures GPS sensor dropouts.

### Class Imbalance Handling
Because each GPS observation generates 10 candidates where only 1 is the ground-truth segment, the dataset is inherently imbalanced:
* **Positive Class ($Y=1$):** $27,954$ samples (**9.24%**)
* **Negative Class ($Y=0$):** $274,486$ samples (**90.76%**)
* **Evaluation Implication:** Accuracy alone is misleading. We benchmarked models using **Macro F1-Score** and **Log-Loss** alongside Observation Top-1 Accuracy.

---

## 5. Model Experimentation: 5 Distinct Algorithm Families

We implemented and benchmarked five models from five fundamentally distinct algorithm families under strictly identical training/testing partitions:

### 1. Linear & Probabilistic Family: Logistic Regression
* **Pipeline:** `StandardScaler` followed by `LogisticRegression(C=1.0, max_iter=1000)`.
* **Objective Function:** Binary Cross-Entropy (Log-Loss):
  $$\mathcal{L}(\mathbf{w}) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \ln \sigma(\mathbf{w}^T \mathbf{x}_i) + (1 - y_i)\ln(1 - \sigma(\mathbf{w}^T \mathbf{x}_i)) \right] + \frac{1}{2C}\|\mathbf{w}\|^2$$

### 2. Instance-Based Family: K-Nearest Neighbors (KNN)
* **Pipeline:** `StandardScaler` followed by `KNeighborsClassifier(n_neighbors=9, weights='distance')`.
* **Metric:** Euclidean metric in 9-dimensional normalized feature space.
* **Weighting:** Inverse distance weighting $w_i = 1 / d(x, x_i)$.

### 3. Tree-Based Family: Decision Tree Classifier
* **Configuration:** `DecisionTreeClassifier(max_depth=12, min_samples_split=10, random_state=42)`.
* **Splitting Criterion:** Gini Impurity:
  $$I_G(p) = 1 - \sum_{k=0}^1 p_k^2$$

### 4. Ensemble (Bagging) Family: Random Forest
* **Configuration:** `RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_split=8, max_features='sqrt', random_state=42)`.
* **Mechanism:** Bootstrap aggregation of 80 decorrelated decision trees. Reduces variance by averaging independent tree predictions.

### 5. Boosting Family: HistGradientBoosting (Selected Winner)
* **Configuration:** `HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_leaf_nodes=31, max_depth=8, random_state=42)`.
* **Mechanism:** Bins continuous features into 256 discrete integer histograms. Sequentially builds additive trees fitting the negative gradient of the binomial deviance loss.

---

## 6. Consolidated Model Comparison & Evaluation

All five models were trained on 302,440 samples and evaluated across all 10 Student-Testing trips (191,600 samples).

### The Official Benchmark Comparison Table:

| Model Name | Algorithm Family | Key Configuration | CV Macro F1 | Test Top-1 Acc (%) | Route Continuity (%) | Test Macro F1 | Log Loss | Inference Latency | Training Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | L2 Reg (C=1.0), Scaler | 0.9016 | 71.42% | 87.39% | 0.7911 | 0.1426 | 0.007 ms/obs | 1.28 s |
| **K-Nearest Neighbors** | Instance-Based | $k=9$, Distance Weights | 1.0000 | 69.00% | 84.93% | 0.7962 | 0.2224 | 2.496 ms/obs | 1.84 s |
| **Decision Tree** | Tree-Based | Max Depth=12, Split=10 | 0.9292 | 70.62% | 85.92% | 0.8091 | 0.1734 | 0.002 ms/obs | 8.27 s |
| **Random Forest** | Ensemble (Bagging) | 80 Trees, Max Depth=14 | 0.9478 | 73.03% | 88.52% | 0.8192 | 0.1254 | 0.222 ms/obs | 63.03 s |
| **HistGradientBoosting** | Boosting | 100 Trees, 31 Leaves | **0.9181** | **73.28%** | **88.19%** | **0.8202** | **0.1243** | **0.019 ms/obs** | **10.38 s** |

### Key Experimental Insights:
1. **Best Classifier:** **HistGradientBoosting** achieved the highest test Top-1 Accuracy (**73.28%**), highest Macro F1 (**0.8202**), and lowest Log-Loss (**0.1243**).
2. **Computational Efficiency:** HistGradientBoosting evaluates in **0.019 ms per observation** (over 100x faster than KNN's 2.496 ms).
3. **Underfitting vs Overfitting:**
   * *Logistic Regression:* Underfitted due to linear capacity constraints (cannot learn non-linear heading interactions).
   * *KNN:* Overfitted to training instances (CV Macro F1 = 1.0, but lowest test accuracy = 69.00%).
   * *HistGradientBoosting:* Maintained optimal bias-variance balance.

---

## 7. Bayesian Hyperparameter Optimization (Optuna)

### Optimization Setup
* **Search Algorithm:** Tree-structured Parzen Estimator (TPE) Sampler.
* **Target Model:** HistGradientBoosting.
* **Search Objective:** Maximize 3-fold cross-validated positive classification accuracy.
* **Number of Trials:** 12 iterations.

### Hyperparameter Search Space & Results:

| Hyperparameter | Search Range / Distribution | Untuned Default | Optimal Tuned Value | Rationale |
| :--- | :--- | :---: | :---: | :--- |
| `max_iter` (Number of trees) | Integer: [40, 160] (step=20) | 100 | **140** | More sequential gradient steps |
| `learning_rate` (Shrinkage) | Float (Log-Uniform): [0.03, 0.20] | 0.08 | **0.0456** | Smaller steps prevent overshooting |
| `max_leaf_nodes` | Integer: [15, 63] | 31 | **17** | Controlled leaf complexity |
| `max_depth` | Integer: [5, 15] | 8 | **7** | Shallow depth limits tree variance |
| `min_samples_leaf` | Integer: [10, 50] | 20 | **18** | Regularizes leaf split thresholds |
| `l2_regularization` | Float (Uniform): [0.0, 5.0] | 0.0 | **4.1341** | Penalizes extreme leaf weights |

### Tuned vs Untuned Benchmark:

| Model Version | Top-1 Accuracy (%) | Route Continuity (%) | Macro F1 | Log Loss | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Untuned HistGradientBoosting** | 73.28% | 88.19% | 0.8202 | 0.1243 | 0.435 ms |
| **Tuned HistGradientBoosting** | **73.41%** | **88.41%** | **0.8218** | **0.1239** | **0.445 ms** |

*Best Cross-Validation Score:* **93.97% Accuracy**.

---

## 8. Individual Model Profiles (Compulsory Viva Questions)

Every team member must be able to defend the compulsory questions for each algorithm:

### Model 1: Logistic Regression
* **Suitability:** Establishes an interpretable, convex probabilistic baseline.
* **Strengths:** Global convexity (no local minima), fast training (1.28 s), well-calibrated posterior probabilities $\sigma(\mathbf{w}^T \mathbf{x})$.
* **Limitations:** Linear decision boundary. Cannot capture multiplicative feature interactions without manual polynomial expansion.
* **Evidence of Underfitting:** High bias, achieving only 71.42% Top-1 accuracy despite high regularization convergence.

### Model 2: K-Nearest Neighbors (KNN)
* **Suitability:** Non-parametric local density estimation.
* **Strengths:** Completely non-linear decision surface, no parametric assumptions.
* **Limitations:** Curse of dimensionality; memory-heavy inference latency ($2.496$ ms/obs) as every test sample scans stored points; highly vulnerable to local class imbalance.
* **Evidence of Overfitting:** Perfect training fold score ($F_1 = 1.0$), but lowest generalisation accuracy ($69.00\%$) on unseen test trips.

### Model 3: Decision Tree
* **Suitability:** Non-linear axis-parallel decision rules with feature importance interpretability.
* **Strengths:** Scale-invariant, handles non-linear splits naturally, fast inference (0.002 ms).
* **Limitations:** High variance; small variations in input data cause major structural changes in tree splits.
* **Evidence of Overfitting:** Train accuracy was significantly higher than test accuracy (70.62%), controlled via `min_samples_split=10`.

### Model 4: Random Forest
* **Suitability:** Bagging ensemble of 80 decorrelated decision trees.
* **Strengths:** Drastically lowers prediction variance ($\sigma^2 / B$), robust against noisy GPS outliers, handles feature interactions.
* **Limitations:** Slower inference time (0.222 ms/obs) due to traversing 80 separate trees; large model memory footprint.
* **Results:** Strong generalisation: Top-1 Accuracy: 73.03%, Macro F1: 0.8192.

### Model 5: HistGradientBoosting (Selected Winner)
* **Suitability:** Gradient boosted decision trees using histogram binning (similar to LightGBM).
* **Strengths:** Bins continuous features into 256 discrete bins for sub-millisecond evaluation (0.019 ms); sequentially corrects hard misclassification errors; supports native missing value routing.
* **Limitations:** Sequential training cannot be completely parallelized across trees; requires careful regularization to prevent overfitting on noisy points.
* **Results:** Winner across all metrics: **73.28% Top-1 Acc**, **0.8202 Macro F1**, **0.1243 Log Loss**.

---

## 9. Team Member Ownership & Viva Defense Scripts

### Member 1: Data Engineering & Spatial Preprocessing
* **Files Owned:** `src/data_loader.py`, `src/preprocessing.py`, `experiments/common.py`.
* **Viva Script:**
  > *"I engineered the data ingestion and geometric coordinate transformations. Because spherical degrees distort Euclidean distances, I implemented equirectangular projection centered at median latitude to map coordinates to true meters. To solve the computational bottleneck of searching 165,000 road segments, I sampled segments at 20%, 50%, and 80% and constructed a `cKDTree`. This reduced spatial query latency from 35 seconds to **42 milliseconds per trip** with **98.25% candidate recall**."*

### Member 2: Linear Baseline & Decision Tree
* **Files Owned:** `experiments/train_model1_logistic_regression.py`, `experiments/train_model3_decision_tree.py`.
* **Viva Script:**
  > *"I implemented the baseline linear and tree models. **Logistic Regression** gave 71.42% accuracy and 0.7911 Macro F1; it provides calibrated probabilities but underfits non-linear spatial interactions. To capture non-linear thresholds, I implemented a **Decision Tree** constrained to `max_depth=12` and `min_samples_split=10`, which improved Macro F1 to **0.8091** by learning orthogonal decision boundaries on distance and speed."*

### Member 3: KNN & Random Forest Ensemble
* **Files Owned:** `experiments/train_model2_knn.py`, `experiments/train_model4_random_forest.py`.
* **Viva Script:**
  > *"I developed the instance-based and ensemble bagging architectures. **KNN ($k=9$)** with distance weighting achieved 69.00% accuracy, but suffered from high query latency (2.5 ms/obs). I then developed a **Random Forest of 80 decorrelated decision trees** with `max_features='sqrt'`. By averaging predictions across bootstrap samples, Random Forest mitigated individual tree variance and pushed Top-1 accuracy to **73.03%** and Macro F1 to **0.8192**."*

### Member 4: Boosting, Bayesian Optimization & HMM
* **Files Owned:** `experiments/train_model5_boosting.py`, `src/bayesian.py`, `src/hmm.py`.
* **Viva Script:**
  > *"I developed the winning **HistGradientBoosting** model, which achieved 73.28% Top-1 accuracy and 0.8202 Macro F1. I conducted **Bayesian Hyperparameter Optimization via Optuna** over 12 trials, tuning tree depth, leaf nodes, learning rate, and L2 regularization (4.134), achieving **93.97% CV accuracy**. Furthermore, I introduced the **Hidden Markov Model with Viterbi decoding**, enforcing road network graph adjacency to jump route continuity from **88.4% to 99.3%**."*

---

## 10. Top 10 Review 1 Viva Trap Questions & Answers

#### Q1: "Why did you convert coordinates to meters instead of using Euclidean distance on lon/lat?"
> **Answer:** *"Latitude and longitude are angular spherical coordinates on an ellipsoid. At 49° North latitude, 1 degree of longitude is ~71.8 km, while 1 degree of latitude is ~111 km. Standard Euclidean distance treats degrees uniformly, causing severe spatial distortion and skewing road headings. Our metric projection scales both axes to real physical meters."*

#### Q2: "How did you generate candidate road segments for each GPS point?"
> **Answer:** *"We sampled points along every road segment at 20%, 50%, and 80% and indexed them into a `scipy.spatial.cKDTree`. For each GPS observation, we query the tree to retrieve the nearest 10 unique candidate arcs in $O(\log M)$ time, achieving 98.25% candidate recall."*

#### Q3: "What is your most important engineered feature?"
> **Answer:** *"The **heading angular difference** $\Delta \theta$ between the vehicle's movement vector $(z_t - z_{t-1})$ and the road segment's direction vector $(v - u)$. Vehicles drive in the direction of the road. If a parallel opposite-direction road is only 5 meters away, distance alone cannot separate them, but heading difference ($0^\circ$ vs $180^\circ$) immediately eliminates the wrong lane."*

#### Q4: "Why did HistGradientBoosting beat Random Forest and KNN?"
> **Answer:** *"HistGradientBoosting bins continuous spatial features into 256 discrete bins, making it extremely fast (0.019 ms) and resistant to small sensor noise. It sequentially minimizes gradient loss to correct difficult edge-case errors, achieving the lowest test log loss (0.1243) and highest Macro F1."*

#### Q5: "How did you prevent data leakage during training?"
> **Answer:** *"We used **Group-Aware split across trips**. GPS points from the same trip were never split across train and validation sets. Preprocessing and feature scales were fitted strictly on training trips. The student-test and evaluator sets remained untouched."*

#### Q6: "Why did you need Bayesian Optimization instead of Grid Search?"
> **Answer:** *"Grid Search evaluates every combination independently and wastes computational budget on poor regions of parameter space. Bayesian Optimization uses a Tree-structured Parzen Estimator (TPE) to model the objective function probabilistically, intelligently sampling hyperparameter configurations that maximize cross-validation performance with fewer trials."*

#### Q7: "What loss function did your models optimize?"
> **Answer:** *"Binary Cross-Entropy (Log-Loss) for classification: $-\sum [y \ln p + (1-y)\ln(1-p)]$. For Decision Tree, Gini Impurity was used."*

#### Q8: "What is the positive class percentage in your dataset?"
> **Answer:** *"Because we extract 10 candidates per GPS point and only 1 candidate is the true road segment, the dataset is naturally imbalanced with **~9.2% positive labels** and **~90.8% negative labels**. That is why we prioritized **Macro F1** and **Log Loss** alongside Top-1 Accuracy."*

#### Q9: "What if consecutive GPS observations have a gap or dropout?"
> **Answer:** *"Our feature engineering extracts `time_delta` $\Delta t$. When a sensor dropout occurs ($\Delta t > 1.0$ s), the model accounts for the uncertainty and the HMM relaxes transition distance penalties proportionally to $\Delta d_{\text{gps}} / \beta$."*

#### Q10: "What is the status of your Phase 2 deployment?"
> **Answer:** *"The HMM Viterbi route decoder is fully implemented, boosting route continuity to **99.33%**. The FastAPI backend and Streamlit evaluation dashboard are already functional and verified on the reserved test trips."*
