import pandas as pd
import ast
import os
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Load your full dataset
df = pd.read_csv("datasets/fox-point-feats-extracted.csv")
X = df.drop(columns=["tid", "label"], errors="ignore")
y = LabelEncoder().fit_transform(df["label"])

# Define models and their hyperparameter grids
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=10000, solver="liblinear", random_state=42),
        {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["liblinear", "lbfgs"],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(n_jobs=-1, random_state=42),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20]},
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="mlogloss", n_jobs=-1, random_state=42),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 6],
            "clf__learning_rate": [0.1, 0.01],
        },
    ),
    "MLP": (
        MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=15000, random_state=42),
        {"clf__alpha": [0.0001, 0.001], "clf__learning_rate_init": [0.001, 0.01]},
    ),
}

# Map filenames to their selection model
selection_files = {
    "Logistic Regression": "results/csv/base/fox/logistic_regression_selection_f1_scores.csv",
    "Random Forest": "results/csv/base/fox/random_forest_selection_f1_scores.csv",
    "XGBoost": "results/csv/base/fox/xgboost_selection_f1_scores.csv",
    "MLP": "results/csv/base/fox/mlp_selection_f1_scores.csv",
}

results = []

# Loop over selection files
for selection_model, file_path in selection_files.items():
    print(f"\nProcessing feature sets from: {selection_model}")
    selection_df = pd.read_csv(file_path)
    feature_sets = selection_df[["description", "seed", "features"]].drop_duplicates()

    for _, row in feature_sets.iterrows():
        features = ast.literal_eval(row["features"])
        valid_features = [f for f in features if f in X.columns]
        if not valid_features:
            continue

        X_selected = X[valid_features]

        # Loop over evaluation models
        for eval_model_name, (eval_model, param_grid) in models.items():
            print(
                f" - Tuning {eval_model_name} on features from {selection_model} | {row['description']} (seed {row['seed']})"
            )

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", eval_model),
                ]
            )

            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                scoring=make_scorer(f1_score, average="weighted"),
                cv=5,
                n_jobs=-1,
            )

            grid_search.fit(X_selected, y)

            results.append(
                {
                    "selection_model": selection_model,
                    "eval_model": eval_model_name,
                    "description": row["description"],
                    "seed": row["seed"],
                    "n_features": len(valid_features),
                    "best_score": grid_search.best_score_,
                    "best_params": grid_search.best_params_,
                }
            )

# Save results separately for each eval_model
results_df = pd.DataFrame(results)

output_dir = "results/csv/base/fox"
os.makedirs(output_dir, exist_ok=True)

for model_name in results_df["eval_model"].unique():
    model_df = results_df[results_df["eval_model"] == model_name]
    file_name = f"tuning_{model_name.lower().replace(' ', '_')}.csv"
    file_path = os.path.join(output_dir, file_name)
    model_df.to_csv(file_path, index=False)
    print(f"✅ Saved: {file_path}")
