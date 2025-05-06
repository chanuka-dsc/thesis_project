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


models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=10000, random_state=42),
        "param_grid": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"],
        },
        "selection_file": "results/csv/base/fox/logistic_regression_selection_f1_scores.csv",
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_jobs=-1, random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 500, 1000],
            "clf__max_depth": [None, 10, 20],
            "clf__max_features": ["sqrt", "log2", 16],
        },
        "selection_file": "results/csv/base/fox/random_forest_selection_f1_scores.csv",
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="mlogloss", n_jobs=-1, random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 500, 1000],
            "clf__max_depth": [3, 6, 10],
            "clf__learning_rate": [0.01, 0.1],
            "clf__subsample": [0.8, 1.0],
        },
        "selection_file": "results/csv/base/fox/xgboost_selection_f1_scores.csv",
    },
    "MLP": {
        "model": MLPClassifier(max_iter=15000, random_state=42),
        "param_grid": {
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "clf__alpha": [1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-4, 1e-3, 1e-2],
        },
        "selection_file": "results/csv/base/fox/mlp_selection_f1_scores.csv",
    },
}


results = []
for model_name, config in models.items():
    print(f"\nProcessing model: {model_name}")
    selection_df = pd.read_csv(config["selection_file"])
    feature_sets = selection_df[["description", "seed", "features"]].drop_duplicates()

    for _, row in feature_sets.iterrows():
        features = ast.literal_eval(row["features"])
        valid_features = [f for f in features if f in X.columns]
        if not valid_features:
            continue

        X_selected = X[valid_features]

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
            cv=5,
            n_jobs=-1,
        )

        grid_search.fit(X_selected, y)

        results.append(
            {
                "selection_model": model_name,
                "eval_model": model_name,
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
