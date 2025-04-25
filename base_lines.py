import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier 
from utilities import evaluate_with_cv_seeds_and_feature_logging
from utilities import apply_feature_selection


# === Load and Prepare Dataset ===
df = pd.read_csv("datasets/fox-point-feats-extracted.csv")

# Separate features and label
X = df.drop(columns=["tid", "label"], errors="ignore")
y = LabelEncoder().fit_transform(df["label"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(
        solver="liblinear", random_state=42, max_iter=1000
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss"),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(10, 5),
        activation="relu",
        solver="adam",
        max_iter=5000,
        random_state=42,
    ),
}

for name, model in models.items():
    # === Forward Selection ===
    X_selected_forward, support_mask_forward = apply_feature_selection(
        X_train, y_train, X_scaled, model, method="forward"
    )

    selected_features_forward = X.columns[support_mask_forward].tolist()

    result_forward = evaluate_with_cv_seeds_and_feature_logging(
        model=model,
        model_name=f"{name}",
        desc="Forward Selection",
        feature_names=selected_features_forward,
        X=X_selected_forward,
        y=y,
    )

    # === Backward Selection ===
    X_selected_backward, support_mask_backward = apply_feature_selection(
        X_train, y_train, X_scaled, model, method="backward"
    )

    selected_features_backward = X.columns[support_mask_backward].tolist()

    results_backward = evaluate_with_cv_seeds_and_feature_logging(
        model=model,
        model_name=f"{name}",
        desc="Backward Selection",
        feature_names=selected_features_backward,
        X=X_selected_backward,
        y=y,
    )

    df_all = pd.concat([result_forward, results_backward], ignore_index=True)
    csv_path = f"results/csv/{name.lower().replace(' ', '_')}_selection_f1_scores.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"Saved combined CSV for {name}: {csv_path}")
