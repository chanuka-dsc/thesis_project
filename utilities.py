from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np


def evaluate_with_repeated_cv_and_boxplot(
    model, model_name, X_train, y_train, n_splits=5, n_repeats=20, save_path=None
):
    """
    Evaluates a classification model using Repeated Stratified K-Fold CV
    with per-fold scaling and shows boxplots of macro and micro F1 scores.

    Parameters:
        model: scikit-learn classifier (e.g., RandomForestClassifier())
        model_name: str, name to use for plot title and file
        X_train: numpy array or DataFrame (unscaled features)
        y_train: numpy array, encoded labels
        n_splits: int, number of folds (default=5)
        n_repeats: int, number of repetitions (default=20)
        save_path: str, if provided, saves boxplot to this file
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    f1_macro_scores = []
    f1_micro_scores = []

    for train_idx, val_idx in rskf.split(X_train, y_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

        # Per-fold scaling
        scaler = StandardScaler()
        X_cv_train_scaled = scaler.fit_transform(X_cv_train)
        X_cv_val_scaled = scaler.transform(X_cv_val)

        model.fit(X_cv_train_scaled, y_cv_train)
        y_cv_pred = model.predict(X_cv_val_scaled)

        # Fix for regression-style predictions
        if y_cv_pred.dtype.kind in {"f", "c"}:
            y_cv_pred = np.round(y_cv_pred).astype(int)

        f1_macro = f1_score(y_cv_val, y_cv_pred, average="macro", zero_division=0)
        f1_micro = f1_score(y_cv_val, y_cv_pred, average="micro", zero_division=0)

        f1_macro_scores.append(f1_macro)
        f1_micro_scores.append(f1_micro)

    # 📊 Plot both F1 scores side-by-side
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [f1_macro_scores, f1_micro_scores],
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="coral"),
        labels=["Macro F1", "Micro F1"],
    )
    plt.title(f"{model_name} F1 Score Distribution (5-Fold CV × 20 Repeats)")
    plt.xlabel("F1 Score")
    plt.grid(axis="x")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # 📢 Print summary
    print(
        f"{model_name} CV Macro F1: {np.mean(f1_macro_scores):.4f} ± {np.std(f1_macro_scores):.4f}"
    )
    print(
        f"{model_name} CV Micro F1: {np.mean(f1_micro_scores):.4f} ± {np.std(f1_micro_scores):.4f}"
    )

    return {"macro": f1_macro_scores, "micro": f1_micro_scores}
