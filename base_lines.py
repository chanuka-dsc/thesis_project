import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from utilities import evaluate_with_cv_seeds_and_feature_logging


# === Load and Prepare Dataset ===
df = pd.read_csv(
    "datasets/ais_point_feats_extracted_datasets_top_4_classes_dataset.csv"
)

# Separate features and label
X = df.drop(columns=["tid", "label"], errors="ignore")
y = LabelEncoder().fit_transform(df["label"])


models = {
    "Logistic Regression": LogisticRegression(
        solver="liblinear", random_state=42, max_iter=10000
    ),
    "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss", n_jobs=-1),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(10, 5),
        activation="relu",
        solver="adam",
        max_iter=15000,
        random_state=42,
    ),
}

for name, model in models.items():
    # === Forward Selection ===

    result_forward = evaluate_with_cv_seeds_and_feature_logging(
        model=model,
        model_name=f"{name}",
        desc="forward",
        X=X,
        y=y,
    )

    # === Backward Selection ===

    results_backward = evaluate_with_cv_seeds_and_feature_logging(
        model=model,
        model_name=f"{name}",
        desc="backward",
        X=X,
        y=y,
    )

    df_all = pd.concat([result_forward, results_backward], ignore_index=True)
    csv_path = f"results/csv/{name.lower().replace(' ', '_')}_selection_f1_scores.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"Saved combined CSV for {name}: {csv_path}")