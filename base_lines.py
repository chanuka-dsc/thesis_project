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


# Dictionary to store model names and their corresponding accuracy scores
f1_scores_macro = {}
f1_scores_micro = {}

# # 1. Logistic Regression: A simple linear model that can be used as a baseline for binary classification tasks.

# logistic_model = LogisticRegression(solver="liblinear", random_state=42, max_iter=1000)

# # Forward selection
# X_selected_forward, selected_indices_ = apply_feature_selection(
#     X_train, y_train, X_scaled, logistic_model, method="forward"
# )
# evaluate_with_cv_seeds_and_boxplot(
#     model=logistic_model,
#     model_name="Logistic Regression (Forward Selection)",
#     X=X_selected_forward,
#     y=y,
#     save_path="results/figures/logistic_forward_f1_boxplot.png",
# )

# # Backward selection
# X_selected_backward, selected_indices = apply_feature_selection(
#     X_train, y_train, X_scaled, logistic_model, method="backward"
# )
# evaluate_with_cv_seeds_and_boxplot(
#     model=logistic_model,
#     model_name="Logistic Regression (Backward Selection)",
#     X=X_selected_backward,
#     y=y,
#     save_path="results/figures/logistic_backward_f1_boxplot.png",
# )


# # 2. Decision tree: A simple decision tree model that can be used as a baseline for classification tasks.

# decision_tree = DecisionTreeClassifier(random_state=42)

# # Forward selection
# X_selected_forward, selected_indices_ = apply_feature_selection(
#     X_train, y_train, X_scaled, decision_tree, method="forward"
# )
# evaluate_with_cv_seeds_and_boxplot(
#     model=decision_tree,
#     model_name="Decision Tree (Forward Selection)",
#     X=X_selected_forward,
#     y=y,
#     save_path="results/figures/decision_tree_forward_f1_boxplot.png",
# )

# # Backward selection
# X_selected_backward, selected_indices = apply_feature_selection(
#     X_train, y_train, X_scaled, decision_tree, method="backward"
# )
# evaluate_with_cv_seeds_and_boxplot(
#     model=decision_tree,
#     model_name="Decision Tree (Backward Selection)",
#     X=X_selected_backward,
#     y=y,
#     save_path="results/figures/decision_tree_backward_f1_boxplot.png",
# )


# 3. Random Forest Classifier: A more complex ensemble model that can be used as a baseline for classification tasks.

random_forest = RandomForestClassifier(random_state=42)

# Forward selection
X_selected_forward, selected_indices_ = apply_feature_selection(
    X_train, y_train, X_scaled, random_forest, method="forward"
)
evaluate_with_cv_seeds_and_boxplot(
    model=random_forest,
    model_name="Random Forest (Forward Selection)",
    X=X_selected_forward,
    y=y,
    save_path="results/figures/random_forest_forward_f1_boxplot.png",
)

# Backward selection
X_selected_backward, selected_indices = apply_feature_selection(
    X_train, y_train, X_scaled, random_forest, method="backward"
)
evaluate_with_cv_seeds_and_boxplot(
    model=random_forest,
    model_name="Random Forest  (Backward Selection)",
    X=X_selected_backward,
    y=y,
    save_path="results/figures/random_forest_backward_f1_boxplot.png",
)

# # 4. XGBoost Classifier: A powerful gradient boosting model that can be used as a baseline for classification tasks.

# xgb_model = XGBClassifier(random_state=42, eval_metric="mlogloss")

# # Forward selection
# X_selected_forward, selected_indices_ = apply_feature_selection(
#     X_train, y_train, X_scaled, xgb_model, method="forward"
# )
# evaluate_with_cv_seeds_and_boxplot(
#     model=xgb_model,
#     model_name="XGBoost (Forward Selection)",
#     X=X_selected_forward,
#     y=y,
#     save_path="results/figures/xgb_model_forward_f1_boxplot.png",
# )

# # Backward selection
# X_selected_backward, selected_indices = apply_feature_selection(
#     X_train, y_train, X_scaled, xgb_model, method="backward"
# )
# evaluate_with_cv_seeds_and_boxplot(
#     model=xgb_model,
#     model_name="XGBoost  (Backward Selection)",
#     X=X_selected_backward,
#     y=y,
#     save_path="results/figures/xgb_model_backward_f1_boxplot.png",
# )


# 5. Regression Classifier: A regression model that can be used as a baseline for regression tasks.

# # Linear Regression baseline model
# linear_model = LinearRegression()

# # Forward selection
# X_selected_forward, selected_indices_ = apply_feature_selection(
#     X_train, y_train, X_scaled, linear_model, method="forward"
# )
# evaluate_with_cv_seeds_and_boxplot(
#     model=linear_model,
#     model_name="Linear regression (Forward Selection)",
#     X=X_selected_forward,
#     y=y,
#     save_path="results/figures/linear_model_forward_f1_boxplot.png",
# )

# # Backward selection
# X_selected_backward, selected_indices = apply_feature_selection(
#     X_train, y_train, X_scaled, linear_model, method="backward"
# )
# evaluate_with_cv_seeds_and_boxplot(
#     model=linear_model,
#     model_name="Linear regression  (Backward Selection)",
#     X=X_selected_backward,
#     y=y,
#     save_path="results/figures/linear_model_backward_f1_boxplot.png",
# )

# 6. MLP

# mlp = MLPClassifier(
#     hidden_layer_sizes=(10, 5),
#     activation="relu",
#     solver="adam",
#     max_iter=1000,
#     random_state=42,
# )

# # Forward selection
# X_selected_forward, selected_indices_ = apply_feature_selection(
#     X_train, y_train, X_scaled, mlp, method="forward"
# )
# evaluate_with_cv_seeds_and_boxplot(
#     model=mlp,
#     model_name="Linear regression (Forward Selection)",
#     X=X_selected_forward,
#     y=y,
#     save_path="results/figures/mlp_forward_f1_boxplot.png",
# )

# # Backward selection
# X_selected_backward, selected_indices = apply_feature_selection(
#     X_train, y_train, X_scaled, mlp, method="backward"
# )
# evaluate_with_cv_seeds_and_boxplot(
#     model=mlp,
#     model_name="Linear regression  (Backward Selection)",
#     X=X_selected_backward,
#     y=y,
#     save_path="results/figures/mlp_backward_f1_boxplot.png",
# )
