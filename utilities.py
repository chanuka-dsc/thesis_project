import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd


def evaluate_with_cv_seeds_and_boxplot(
    model, model_name, X, y, seeds=[42, 52, 62, 72], n_splits=5, save_path=None
):
    f1_macro_scores = []
    f1_micro_scores = []
    f1_weighted_scores = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Per-fold scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)

            if y_pred.dtype.kind in {"f", "c"}:
                y_pred = np.round(y_pred).astype(int)

            f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_micro = f1_score(y_val, y_pred, average="micro", zero_division=0)
            f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

            f1_macro_scores.append(f1_macro)
            f1_micro_scores.append(f1_micro)
            f1_weighted_scores.append(f1_weighted)

    print(
        f"{model_name} Macro F1: {np.mean(f1_macro_scores):.4f} ± {np.std(f1_macro_scores):.4f}"
    )
    print(
        f"{model_name} Micro F1: {np.mean(f1_micro_scores):.4f} ± {np.std(f1_micro_scores):.4f}"
    )
    print(
        f"{model_name} Weighted F1: {np.mean(f1_weighted_scores):.4f} ± {np.std(f1_weighted_scores):.4f}"
    )

    return {
        "macro": f1_macro_scores,
        "micro": f1_micro_scores,
        "weighted": f1_weighted_scores,
    }


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


def save_all_f1_scores_to_csv(
    f1_scores_macro_forward,
    f1_scores_micro_forward,
    f1_scores_macro_backward,
    f1_scores_micro_backward,
    save_path,
    filename="all_f1_scores.csv",
):
   
    os.makedirs(save_path, exist_ok=True)


    df_mf = pd.DataFrame(f1_scores_macro_forward).add_prefix("forward_macro_")
    df_mi = pd.DataFrame(f1_scores_micro_forward).add_prefix("forward_micro_")
    df_bm = pd.DataFrame(f1_scores_macro_backward).add_prefix("backward_macro_")
    df_bi = pd.DataFrame(f1_scores_micro_backward).add_prefix("backward_micro_")


    df_all = pd.concat([df_mf, df_mi, df_bm, df_bi], axis=1)


    out_file = os.path.join(save_path, filename)
    df_all.to_csv(out_file, index=False)
