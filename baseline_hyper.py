import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from utilities import evaluate_with_cv_seeds_and_feature_logging_with_hyper
import os

# === Load and Prepare Dataset ===
df = pd.read_csv("datasets/hurricane_balanced_top5_classes.csv")

X = df.drop(columns=["tid", "label"], errors="ignore")
y = LabelEncoder().fit_transform(df["label"])

# === Define models ===
models = {
    "Logistic Regression": LogisticRegression(
        solver="liblinear", random_state=42, max_iter=10000
    ),
    "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
    # "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss", n_jobs=-1),
    # "MLP": MLPClassifier(
    #     hidden_layer_sizes=(10, 5),
    #     activation="relu",
    #     solver="adam",
    #     max_iter=15000,
    #     random_state=42,
    # ),
}

# === Define hyperparameter grids ===
param_grids = {
    "Logistic Regression": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    },
    "Random Forest": {
        "n_estimators": [100, 500, 1000],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2", 16],
    },
    "XGBoost": {
        "n_estimators": [100, 500, 1000],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 1e-3, 1e-2],
    },
}

# === Run tuned evaluation with selection ===
for name, model in models.items():
    param_grid = param_grids.get(name)

    result_forward = evaluate_with_cv_seeds_and_feature_logging_with_hyper(
        model=model,
        model_name=f"{name} + Tuning",
        desc="forward",
        X=X,
        y=y,
        param_grid=param_grid,
    )

    result_backward = evaluate_with_cv_seeds_and_feature_logging_with_hyper(
        model=model,
        model_name=f"{name} + Tuning",
        desc="backward",
        X=X,
        y=y,
        param_grid=param_grid,
    )

    df_all = pd.concat([result_forward, result_backward], ignore_index=True)
    csv_path = f"results/csv/base/hurricane/{name.lower().replace(' ', '_')}_tuned_selection_f1_scores.csv"
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_all.to_csv(csv_path, index=False)
    print(f"Saved tuned selection CSV for {name}: {csv_path}")
