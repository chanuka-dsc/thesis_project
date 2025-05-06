import pandas as pd
import ast
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv("datasets/fox-point-feats-extracted.csv")
X = df.drop(columns=["tid", "label"], errors="ignore")
y = LabelEncoder().fit_transform(df["label"])

# Define models with paths to their evaluated result CSVs
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=10000, random_state=42),
        "param_grid": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"],
        },
        "combination_file": "results/csv/taxonomy/fox/partial/logistic_regression_combo_distance_geometry+angles.csv",
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_jobs=-1, random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 500, 1000],
            "clf__max_depth": [None, 10, 20],
            "clf__max_features": ["sqrt", "log2", 16],
        },
        "combination_file": "results/csv/taxonomy/fox/partial/random_forest_combo_acceleration.csv",
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="mlogloss", n_jobs=-1, random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 500, 1000],
            "clf__max_depth": [3, 6, 10],
            "clf__learning_rate": [0.01, 0.1],
            "clf__subsample": [0.8, 1.0],
        },
        "combination_file": "results/csv/taxonomy/fox/partial/xgboost_combo_distance_geometry+speed.csv",
    },
    "MLP": {
        "model": MLPClassifier(max_iter=15000, random_state=42),
        "param_grid": {
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "clf__alpha": [1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-4, 1e-3, 1e-2],
        },
        "combination_file": "results/csv/taxonomy/fox/partial/mlp_combo_distance_geometry+angles.csv",
    },
}

# Run tuning
results = []

for model_name, config in models.items():
    print(f"\n📌 Processing model: {model_name}")

    df_combo = pd.read_csv(config["combination_file"])

    for _, row in df_combo.iterrows():
        try:
            features = ast.literal_eval(row["features"])
        except (ValueError, SyntaxError):
            print(f"⚠️ Skipping malformed feature list at row {row}")
            continue

        selected_features = [f for f in features if f in X.columns]
        if not selected_features:
            continue

        X_selected = X[selected_features]

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", config["model"]),
            ]
        )

        grid_search = GridSearchCV(
            pipeline,
            config["param_grid"],
            scoring=make_scorer(f1_score, average="weighted"),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1,
        )

        grid_search.fit(X_selected, y)

        results.append(
            {
                "model": model_name,
                "seed": row.get("seed"),
                "fold": row.get("fold"),
                "description": row.get("description"),
                "n_features": len(selected_features),
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_,
            }
        )

# Save results
results_df = pd.DataFrame(results)
output_dir = "results/csv/taxonomy/fox/partial/tuned"
os.makedirs(output_dir, exist_ok=True)

for model_name in results_df["model"].unique():
    model_df = results_df[results_df["model"] == model_name]
    file_name = f"tuning_per_fold_{model_name.lower().replace(' ', '_')}.csv"
    model_df.to_csv(os.path.join(output_dir, file_name), index=False)
    print(f"✅ Saved: {file_name}")
