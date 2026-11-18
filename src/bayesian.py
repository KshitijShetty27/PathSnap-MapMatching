"""
Bayesian Hyperparameter Optimization Module for PathSnap.
Uses Optuna to optimize hyperparameter configurations using cross-validation.
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold


def run_bayesian_optimization(X_train, y_train, n_trials=12, seed=42):
    """
    Executes Bayesian Hyperparameter Optimization on HistGradientBoosting.
    Parameters searched:
        - max_iter: int [40, 160]
        - learning_rate: float [0.03, 0.2]
        - max_leaf_nodes: int [15, 63]
        - max_depth: int [5, 15]
        - min_samples_leaf: int [10, 50]
        - l2_regularization: float [0.0, 5.0]
    Objective:
        Cross-validated Macro F1 / Top-1 observation accuracy.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Subsample for fast cross-validation
    n_samples = len(X_train)
    cv_size = min(30000, n_samples)
    indices = np.random.RandomState(seed).permutation(n_samples)[:cv_size]
    X_cv, y_cv = X_train[indices], y_train[indices]

    kf = KFold(n_splits=3, shuffle=True, random_state=seed)

    def objective(trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 40, 160, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
            "random_state": seed,
        }

        scores = []
        for tr_idx, va_idx in kf.split(X_cv):
            X_tr, y_tr = X_cv[tr_idx], y_cv[tr_idx]
            X_va, y_va = X_cv[va_idx], y_cv[va_idx]

            clf = HistGradientBoostingClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            acc = np.mean(preds == y_va)
            scores.append(acc)

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    trials_df = study.trials_dataframe()

    best_params = study.best_params
    best_params.update({"random_state": seed})

    # Fit final tuned model on full training set
    best_model = HistGradientBoostingClassifier(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, study.best_params, study.best_value, trials_df
