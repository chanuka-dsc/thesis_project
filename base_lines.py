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
from log_results_in_csv import log_result
from utilities import evaluate_with_cv_seeds_and_boxplot
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

# Dictionary to store F1 scores
f1_scores_macro_backward = {}
f1_scores_micro_backward = {}
f1_scores_macro_forward = {}
f1_scores_micro_forward = {}

models = {
    "Logistic Regression": LogisticRegression(
        solver="liblinear", random_state=42, max_iter=1000
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss"),
    "Linear Regression": LinearRegression(),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(10, 5),
        activation="relu",
        solver="adam",
        max_iter=5000,
        random_state=42,
    ),
}

for name, model in models.items():
    # Forward selection
    X_selected_forward, _ = apply_feature_selection(
        X_train, y_train, X_scaled, model, method="forward"
    )
    scores_forward = evaluate_with_cv_seeds_and_boxplot(
        model=model,
        model_name=f"{name} (Forward Selection)",
        X=X_selected_forward,
        y=y,
        save_path=f"results/figures/{name.lower().replace(' ', '_')}_forward_f1_boxplot.png",
    )
    f1_scores_macro_forward[name] = scores_forward["macro"]
    f1_scores_micro_forward[name] = scores_forward["micro"]

    # Backward selection
    X_selected_backward, _ = apply_feature_selection(
        X_train, y_train, X_scaled, model, method="backward"
    )
    scores_backward = evaluate_with_cv_seeds_and_boxplot(
        model=model,
        model_name=f"{name} (Backward Selection)",
        X=X_selected_backward,
        y=y,
        save_path=f"results/figures/{name.lower().replace(' ', '_')}_backward_f1_boxplot.png",
    )
    f1_scores_macro_backward[name] = scores_backward["macro"]
    f1_scores_micro_backward[name] = scores_backward["micro"]

# Plotting helper


def plot_f1_scores(score_dict, title, filename, color):
    plt.figure(figsize=(12, 8))
    plt.boxplot(
        score_dict.values(),
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor=color),
        labels=score_dict.keys(),
    )
    plt.title(title)
    plt.xlabel("F1 Score")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# Plot all F1 comparisons
plot_f1_scores(
    f1_scores_macro_forward,
    "Macro F1 - Forward Selection",
    "results/figures/macro_f1_forward_all.png",
    "coral",
)
plot_f1_scores(
    f1_scores_micro_forward,
    "Micro F1 - Forward Selection",
    "results/figures/micro_f1_forward_all.png",
    "skyblue",
)
plot_f1_scores(
    f1_scores_macro_backward,
    "Macro F1 - Backward Selection",
    "results/figures/macro_f1_backward_all.png",
    "coral",
)
plot_f1_scores(
    f1_scores_micro_backward,
    "Micro F1 - Backward Selection",
    "results/figures/micro_f1_backward_all.png",
    "skyblue",
)
