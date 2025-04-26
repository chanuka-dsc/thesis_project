from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from result_logger_helper import evaluate_and_log
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from utilities import evaluate_with_cv_seeds_and_boxplot


# Load data
ffoxi = pd.read_csv("datasets/fox-point-feats-extracted.csv")

# Prepare data
X = ffoxi.drop(columns=['tid', 'label'], errors='ignore')
y = LabelEncoder().fit_transform(ffoxi['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Dictionary to store model names and their corresponding accuracy scores
f1_scores_macro = {}
f1_scores_micro = {}

# 1. Logistic Regression: A simple linear model that can be used as a baseline for binary classification tasks.

# Logistic Regression baseline model
logistic_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
logistic_scores = evaluate_with_cv_seeds_and_boxplot(
    model=logistic_model,
    model_name="Logistic regression",
    X=X_scaled,
    y=y,
    save_path="results/figures/logistic_regression_boxplot.png",
)

f1_scores_macro["Logistic Regression"] = logistic_scores["macro"]
f1_scores_micro["Logistic Regression"] = logistic_scores["micro"]


# 2. Decision Tree Classifier: A simple decision tree model that can be used as a baseline for classification tasks.

decision_tree = DecisionTreeClassifier(random_state=42)
dt_scores = evaluate_with_cv_seeds_and_boxplot(
    model=decision_tree,
    model_name="Decision Tree",
    X=X_scaled,
    y=y,
    save_path="results/figures/decision_boxplot.png",
)

f1_scores_macro["Decision Tree"] = dt_scores["macro"]
f1_scores_micro["Decision Tree"] = dt_scores["micro"]

# 3. Random Forest Classifier: A more complex ensemble model that can be used as a baseline for classification tasks.
# without scaling
random_forest = RandomForestClassifier(random_state=42)
random_forest_scores = evaluate_with_cv_seeds_and_boxplot(
    model=random_forest,
    model_name="Random Forest",
    X=X_scaled,
    y=y,
    save_path="results/figures/random_forest_boxplot.png",
)

f1_scores_macro["Random Forest"] = random_forest_scores["macro"]
f1_scores_micro["Random Forest"] = random_forest_scores["micro"]

# 4. XGBoost Classifier: A powerful gradient boosting model that can be used as a baseline for classification tasks.
# without scaling
xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
xgb_model_scores = evaluate_with_cv_seeds_and_boxplot(
    model=xgb_model,
    model_name="XGBoost",
    X=X_scaled,
    y=y,
    save_path="results/figures/xg_boxplot.png",
)

f1_scores_macro["XGBoost"] = xgb_model_scores["macro"]
f1_scores_micro["XGBoost"] = xgb_model_scores["micro"]

# 5. Regression Classifier: A regression model that can be used as a baseline for regression tasks.

# Linear Regression baseline model (scaled)
linear_model = LinearRegression()
linear_model_scores = evaluate_with_cv_seeds_and_boxplot(
    model=linear_model,
    model_name="Linear_model",
    X=X_scaled,
    y=y,
    save_path="results/figures/linear_model_boxplot.png",
)

f1_scores_macro["Linear_model"] = linear_model_scores["macro"]
f1_scores_micro["Linear_model"] = linear_model_scores["micro"]

# 6. MLP
# Initialize MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
)

mlp_scores = evaluate_with_cv_seeds_and_boxplot(
    model=mlp,
    model_name="MLP",
    X=X_scaled,
    y=y,
    save_path="results/figures/mlp_boxplot.png",
)

f1_scores_macro["MLP"] = mlp_scores["macro"]
f1_scores_micro["MLP"] = mlp_scores["micro"]


# Plot all the accuracy scores for the models

# Macro
plt.figure(figsize=(12, 8))
plt.boxplot(
    f1_scores_macro.values(),
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="coral"),
    labels=f1_scores_macro.keys(),
)
plt.title("Macro F1 Score Comparison (5-Fold CV × 20 Repeats)")
plt.xlabel("F1 Score (macro)")
plt.grid(axis="x")
plt.tight_layout()
plt.savefig(
    "results/figures/all_models_macro_f1_boxplot.png", dpi=300, bbox_inches="tight"
)
plt.show()


# Micro

plt.figure(figsize=(12, 8))
plt.boxplot(
    f1_scores_micro.values(),
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="skyblue"),
    labels=f1_scores_micro.keys(),
)
plt.title("Micro F1 Score Comparison (5-Fold CV × 20 Repeats)")
plt.xlabel("F1 Score (micro)")
plt.grid(axis="x")
plt.tight_layout()
plt.savefig(
    "results/figures/all_models_micro_f1_boxplot.png", dpi=300, bbox_inches="tight"
)
plt.show()

# # F1 score dictionary
# f1_scores = {
#     "Dummy (Most Frequent)": f1_score(y_test, y_pred_dummy, average='micro', zero_division=0),
#     "Dummy (Uniform)": f1_score(y_test, y_pred_dummy_uniform, average='micro', zero_division=0),
#     "Dummy (Stratified)": f1_score(y_test, y_pred_dummy_stratified, average='micro', zero_division=0),
#     "Logistic Regression": f1_score(y_test, y_pred_logistic_unscaled, average='micro', zero_division=0),
#     "Logistic Regression (Scaled)": f1_score(y_test, y_pred_logistic, average='micro', zero_division=0),
#     "Decision Tree": f1_score(y_test, y_pred_tree, average='micro', zero_division=0),
#     "Random Forest": f1_score(y_test, y_pred_forest, average='micro', zero_division=0),
#     "XGBoost": f1_score(y_test, y_pred_xgb, average='micro', zero_division=0),
#     "KNN": f1_score(y_test, y_pred_knn, average='micro', zero_division=0),
#     "KNN (Scaled)": f1_score(y_test, y_pred_knn_scaled, average='micro', zero_division=0),
#     "SVM": f1_score(y_test, y_pred_svm, average='micro', zero_division=0),
#     "SVM (Scaled)": f1_score(y_test, y_pred_svm_scaled, average='micro', zero_division=0),
#     "Linear Regression": f1_score(y_test, y_pred_linear_class, average='micro', zero_division=0),
#     "Linear Regression (Scaled)": f1_score(y_test, y_pred_linear_scaled_class, average='micro', zero_division=0)
# }


# # Plot F1 scores
# plt.figure(figsize=(18, 10))
# plt.title("Baseline Model F1 Score Comparison")
# plt.barh(list(f1_scores.keys()), list(f1_scores.values()), color='coral')
# plt.xlabel("F1 Score")
# plt.xlim(0, 1)
# plt.grid(axis='x')
# plt.tight_layout()
# plt.show()

# Log results in CSV
# evaluate_and_log(
#     model=LogisticRegression(solver='liblinear', random_state=42),
#     model_name="Logistic Regression",
#     X_train=X_train,
#     X_test=X_test_scaled,
#     y_train=y_train,
#     y_test=y_test,
#     csv_path="/results/csv/baseline_fox_logistic_regression.csv",
#     description="Logistic Regression (Scaled)",
#     feature_vector=X.columns.tolist(),  # This is essential for row-wise
#     technique="None"
# )

# evaluate_and_log(
#     model=LogisticRegression(solver='liblinear', random_state=42),
#     model_name="Logistic Regression",
#     X_train=X_train_scaled,
#     X_test=X_test_scaled,
#     y_train=y_train,
#     y_test=y_test,
#     csv_path="results/baseline_fox.csv"
# )
