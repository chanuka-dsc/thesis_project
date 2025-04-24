from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector


def evaluate_with_cv_seeds_and_boxplot(
    model, model_name, X, y, seeds=[42, 52, 62, 72], n_splits=5, save_path=None
):
    """
    Evaluates a model using 5-fold Stratified CV over multiple seeds and returns F1 scores.

    Parameters:
        model: scikit-learn classifier
        model_name: str, name to display
        X: unscaled features
        y: encoded labels
        seeds: list of int, random seeds for StratifiedKFold
        n_splits: int, number of folds (default=5)
        save_path: str, optional path to save the plot
    """
    f1_macro_scores = []
    f1_micro_scores = []

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

            f1_macro_scores.append(f1_macro)
            f1_micro_scores.append(f1_micro)

    # 📊 Plot both F1 scores
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [f1_macro_scores, f1_micro_scores],
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="coral"),
        labels=["Macro F1", "Micro F1"],
    )
    plt.title(f"{model_name} F1 Score Distribution (4 Seeds × 5-Fold CV)")
    plt.xlabel("F1 Score")
    plt.grid(axis="x")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(
        f"{model_name} Macro F1: {np.mean(f1_macro_scores):.4f} ± {np.std(f1_macro_scores):.4f}"
    )
    print(
        f"{model_name} Micro F1: {np.mean(f1_micro_scores):.4f} ± {np.std(f1_micro_scores):.4f}"
    )

    return {"macro": f1_macro_scores, "micro": f1_micro_scores}


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
