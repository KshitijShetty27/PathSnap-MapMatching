"""
Model Definitions and Training Pipeline for PathSnap.
Implements five models from distinct machine learning algorithm families:
1. Logistic Regression (Linear & Probabilistic)
2. K-Nearest Neighbors (Instance-Based)
3. Decision Tree (Tree-Based)
4. Random Forest (Ensemble)
5. Hist_Gradient_Boosting (Gradient Boosting)
"""

import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def get_candidate_models():
    """
    Returns a dictionary of initialized models representing 5 distinct algorithm families.
    """
    models = {
        "Logistic_Regression": {
            "family": "Linear & Probabilistic",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
            ]),
            "description": "L2-regularized logistic regression with standard feature scaling"
        },
        "K_Nearest_Neighbors": {
            "family": "Instance-Based",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1))
            ]),
            "description": "Distance-weighted KNN with Euclidean metric on scaled features"
        },
        "Decision_Tree": {
            "family": "Tree-Based",
            "model": DecisionTreeClassifier(max_depth=12, min_samples_split=10, random_state=42),
            "description": "Binary decision tree with depth pruning to limit overfitting"
        },
        "Random_Forest": {
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_split=8, n_jobs=-1, random_state=42),
            "description": "Bagged ensemble of 80 decorrelated decision trees"
        },
        "Hist_Gradient_Boosting": {
            "family": "Boosting",
            "model": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_leaf_nodes=31, max_depth=8, random_state=42),
            "description": "Histogram-based gradient boosted decision trees with native multi-threading"
        }
    }
    return models


def train_single_model(name, model_entry, X_train, y_train):
    """
    Fits a single model on training data and measures training latency.
    """
    model = model_entry["model"]
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    return model, fit_time
