import os
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def evaluate_with_cv_seeds_and_feature_logging(
    model,
    model_name: str,
    desc:str,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    seeds=[14159, 26535, 89793, 23846],
    n_splits=5,
) -> pd.DataFrame:

    records = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]



            # per-fold scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # fit & predict
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            if y_pred.dtype.kind in {"f", "c"}:
                y_pred = np.round(y_pred).astype(int)

            # compute F1s
            f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_micro = f1_score(y_val, y_pred, average="micro", zero_division=0)
            f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

            # append record
            records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "description": desc,
                    "fold": fold_idx,
                    "features": feature_names,
                    "f1_macro": f1_macro,
                    "f1_micro": f1_micro,
                    "f1_weighted": f1_weighted,
                }
            )

    # build DataFrame
    results_df = pd.DataFrame(records)

    # print overall stats
    for col in ["f1_macro", "f1_micro", "f1_weighted"]:
        mean = results_df[col].mean()
        std = results_df[col].std()
        print(
            f"{model_name} {col.replace('f1_','').capitalize()} F1: {mean:.4f} ± {std:.4f}"
        )

    return results_df


def apply_feature_selection(
    X_train, y_train, X_scaled, model, method="forward", k="auto", scoring="f1_micro"
):

    direction = "forward" if method == "forward" else "backward"
    selector = SequentialFeatureSelector(
        model,
        direction=direction,
        scoring=scoring,
        n_jobs=-1,
        cv=5,
        n_features_to_select=k,
        tol=None,
    )
    selector.fit(X_train, y_train)
    selected_indices = selector.get_support()
    X_selected = X_scaled[:, selected_indices]
    return X_selected, selected_indices
