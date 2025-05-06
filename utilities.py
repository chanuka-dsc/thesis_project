import os
from typing import List, Dict, Optional, Any
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer


def evaluate_with_cv_seeds_and_feature_logging(
    model,
    model_name: str,
    desc: str,
    X: pd.DataFrame,
    y: np.ndarray,
    seeds: List[int] = [14159, 26535, 89793, 23846],
    n_splits: int = 5,
) -> pd.DataFrame:

    # === Impute missing values before splitting ===
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    feature_names = list(X_imputed.columns)
    X_array = X_imputed.values

    records = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_array, y), start=1):
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # === nested feature-selection ===
            selector = SequentialFeatureSelector(
                model,
                direction=desc,
                scoring="f1_weighted",
                n_jobs=-1,
                cv=2,  # Minimum for valid CV
                n_features_to_select="auto",
            )
            selector.fit(X_train, y_train)
            mask = selector.get_support()

            X_train_sel = X_train[:, mask]
            X_val_sel = X_val[:, mask]
            selected = [feature_names[i] for i, keep in enumerate(mask) if keep]

            # === per-fold scaling ===
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_val_scaled = scaler.transform(X_val_sel)

            # === train & predict ===
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            if y_pred.dtype.kind in {"f", "c"}:
                y_pred = np.round(y_pred).astype(int)

            # === compute F1 metrics ===
            f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_micro = f1_score(y_val, y_pred, average="micro", zero_division=0)
            f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

            records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "description": desc,
                    "fold": fold_idx,
                    "features": selected,
                    "f1_macro": f1_macro,
                    "f1_micro": f1_micro,
                    "f1_weighted": f1_weighted,
                }
            )

    results_df = pd.DataFrame(records)

    # === print summary ===
    for metric in ["f1_macro", "f1_micro", "f1_weighted"]:
        m_mean = results_df[metric].mean()
        m_std = results_df[metric].std()
        print(
            f"{model_name} [{desc}] {metric.replace('f1_','').capitalize()} F1: "
            f"{m_mean:.4f} ± {m_std:.4f}"
        )

    return results_df


def evaluate_with_cv_seeds_and_feature_logging_with_hyper(
    model,
    model_name: str,
    desc: str,
    X: pd.DataFrame,
    y: np.ndarray,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    seeds: List[int] = [14159, 26535, 89793, 23846],
    n_splits: int = 5,
) -> pd.DataFrame:

    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    feature_names = list(X_imputed.columns)
    X_array = X_imputed.values

    records = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_array, y), start=1):
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # === Feature selection using base model ===
            selector = SequentialFeatureSelector(
                model,
                direction=desc,
                scoring="f1_weighted",
                n_jobs=-1,
                cv=2,
                n_features_to_select="auto",
            )
            selector.fit(X_train, y_train)
            mask = selector.get_support()
            X_train_sel = X_train[:, mask]
            X_val_sel = X_val[:, mask]
            selected = [feature_names[i] for i, keep in enumerate(mask) if keep]

            # === Scaling ===
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_val_scaled = scaler.transform(X_val_sel)

            # === Hyperparameter tuning on selected features ===
            if param_grid:
                grid_search = GridSearchCV(
                    model,
                    param_grid=param_grid,
                    scoring="f1_weighted",
                    n_jobs=-1,
                    cv=3,
                )
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = model

            # === Final Training and Prediction ===
            best_model.fit(X_train_scaled, y_train)
            y_pred = best_model.predict(X_val_scaled)
            if y_pred.dtype.kind in {"f", "c"}:
                y_pred = np.round(y_pred).astype(int)

            # === F1 Metrics ===
            f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_micro = f1_score(y_val, y_pred, average="micro", zero_division=0)
            f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

            records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "description": desc,
                    "fold": fold_idx,
                    "features": selected,
                    "f1_macro": f1_macro,
                    "f1_micro": f1_micro,
                    "f1_weighted": f1_weighted,
                }
            )

    results_df = pd.DataFrame(records)
    for metric in ["f1_macro", "f1_micro", "f1_weighted"]:
        m_mean = results_df[metric].mean()
        m_std = results_df[metric].std()
        print(
            f"{model_name} [{desc}] {metric.replace('f1_','').capitalize()} F1: "
            f"{m_mean:.4f} ± {m_std:.4f}"
        )

    return results_df


def evaluate_with_cv_seeds_taxonomy_fixed_features(
    model,
    model_name: str,
    desc: str,
    X: pd.DataFrame,
    y: np.ndarray,
    seeds: List[int] = [14159, 26535, 89793, 23846],
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Cross-validation evaluation with pre-fixed features (no feature selection inside).
    """
    feature_names = list(X.columns)
    X_array = X.values

    records = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_array, y), start=1):
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # === No feature selection ===
            selected = feature_names
            X_train_sel = X_train
            X_val_sel = X_val

            # === per-fold scaling ===
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_val_scaled = scaler.transform(X_val_sel)

            # === train & predict ===
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            if y_pred.dtype.kind in {"f", "c"}:
                y_pred = np.round(y_pred).astype(int)

            # === compute F1 metrics ===
            f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_micro = f1_score(y_val, y_pred, average="micro", zero_division=0)
            f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

            records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "description": desc,
                    "fold": fold_idx,
                    "features": selected,
                    "f1_macro": f1_macro,
                    "f1_micro": f1_micro,
                    "f1_weighted": f1_weighted,
                }
            )

    results_df = pd.DataFrame(records)

    # print summary
    for metric in ["f1_macro", "f1_micro", "f1_weighted"]:
        m_mean = results_df[metric].mean()
        m_std = results_df[metric].std()
        print(
            f"{model_name} [{desc}] {metric.replace('f1_','').capitalize()} F1: "
            f"{m_mean:.4f} ± {m_std:.4f}"
        )

    return results_df
