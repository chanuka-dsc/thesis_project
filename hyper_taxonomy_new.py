import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from utilities import evaluate_with_cv_seeds_taxonomy_fixed_features_hyper
import os
from itertools import combinations

# === Load and Prepare Dataset ===
df = pd.read_csv("datasets/hurricane_balanced_top5_classes.csv")
X = df.drop(columns=["tid", "label"], errors="ignore")
y = LabelEncoder().fit_transform(df["label"])

# === Define Models with Hyperparameter Grids ===
models = {
    # "Logistic Regression": {
    #     "model": LogisticRegression(solver="liblinear", random_state=42, max_iter=10000),
    #     "param_grid": {
    #         "C": [0.1, 1.0, 10.0],
    #         "penalty": ["l1", "l2"],
    #         "solver": ["liblinear", "saga"],
    #     },
    # },
    # "Random Forest": {
    #     "model": RandomForestClassifier(random_state=42),
    #     "param_grid": {
    #         "n_estimators": [100, 500, 1000],
    #         "max_depth": [None, 10, 20],
    #         "max_features": ["sqrt", "log2", 16],
    #     },
    # },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric="mlogloss"),
        "param_grid": {
            "n_estimators": [100, 500, 1000],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
        },
    },
    "MLP": {
        "model": MLPClassifier(
            hidden_layer_sizes=(10, 5),
            activation="relu",
            solver="adam",
            max_iter=15000,
            random_state=42,
        ),
        "param_grid": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-4, 1e-3, 1e-2],
        },
    },
}

# === Load Taxonomy ===
taxonomy_df = pd.read_csv("results/taxonomy_simple.csv")
mid_levels = taxonomy_df["mid_level"].unique().tolist()

# === Get all non-empty combinations of mid-levels ===
combination_list = []
for r in range(1, len(mid_levels) + 1):
    combination_list.extend(combinations(mid_levels, r))

for name, config in models.items():
    all_results = []
    for combo in combination_list:
        selected_features = taxonomy_df.loc[
            taxonomy_df["mid_level"].isin(combo), "feature_name"
        ].tolist()

        if not selected_features:
            continue

        X_subset = X[selected_features]
        desc = "+".join(combo)

        # Evaluate model with hyperparameter tuning
        result_combo = evaluate_with_cv_seeds_taxonomy_fixed_features_hyper(
            model=config["model"],
            model_name=name,
            desc=f"combo_{desc}",
            X=X_subset,
            y=y,
            param_grid=config["param_grid"],
            cv_inner=3,
            scoring="f1_weighted",
        )

        # Save detailed results per combination
        model_safe = name.lower().replace(" ", "_")
        combo_safe = desc.replace(" ", "_")
        csv_name = f"results/csv/taxonomy/hurricane/partial/tuned/{model_safe}_combo_tuned_{combo_safe}.csv"
        result_combo.to_csv(csv_name, index=False)
        print(f"Saved: {csv_name}")

        # Store summary for this combo
        f1_mean = result_combo["f1_weighted"].mean()
        all_results.append(
            {"model": name, "combination": desc, "mean_f1_weighted": f1_mean}
        )

    # Save summary for this model
    summary_df = pd.DataFrame(all_results).sort_values(
        by="mean_f1_weighted", ascending=False
    )
    summary_path = f"results/csv/taxonomy/hurricane/{model_safe}_combination_tuned_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary for {name}: {summary_path}")

    # Print the best-performing combination
    best_row = summary_df.iloc[0]
    best_df = pd.DataFrame([best_row])
    print(
        f"Best for {name}: {best_row['combination']} "
        f"with mean F1 (weighted): {best_row['mean_f1_weighted']:.4f}"
    )
