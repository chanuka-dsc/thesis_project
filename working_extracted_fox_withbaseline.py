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
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store model names and their corresponding accuracy scores
accuracies = {}

# 1. Logistic Regression: A simple linear model that can be used as a baseline for binary classification tasks.

# Logistic Regression baseline model
logistic_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)
accuracies["Logistic Regression (Scaled)"] = accuracy_score(y_test, logistic_model.predict(X_test_scaled))

# Evaluate Baseline Performance
print("🔹 Baseline (Logistic Regression) Classification Report:")
print(classification_report(y_test, y_pred_logistic))

# 2. Decision Tree Classifier: A simple decision tree model that can be used as a baseline for classification tasks.

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_scaled, y_train)
y_pred_tree = decision_tree.predict(X_test_scaled)
accuracies["Decision Tree"] = accuracy_score(
    y_test, decision_tree.predict(X_test_scaled)
)

# Evaluate Baseline Performance
print("🔹 Baseline (Decision Tree) Classification Report:")
print(classification_report(y_test, y_pred_tree))

# 3. Random Forest Classifier: A more complex ensemble model that can be used as a baseline for classification tasks.
# without scaling
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train_scaled, y_train)
y_pred_forest = random_forest.predict(X_test_scaled)
accuracies["Random Forest"] = accuracy_score(
    y_test, random_forest.predict(X_test_scaled)
)

# Evaluate Baseline Performance
print("🔹 Baseline (Random Forest) Classification Report:")
print(classification_report(y_test, y_pred_forest))


# 4. XGBoost Classifier: A powerful gradient boosting model that can be used as a baseline for classification tasks.
# without scaling
xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
accuracies["XGBoost"] = accuracy_score(y_test, xgb_model.predict(X_test_scaled))
# Evaluate Baseline Performance
print("🔹 Baseline (XGBoost) Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# 5. Regression Classifier: A regression model that can be used as a baseline for regression tasks.


# Linear Regression baseline model (scaled)
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_linear_class = np.round(y_pred_linear).astype(int)
accuracies["Linear Regression (Scaled)"] = accuracy_score(y_test, y_pred_linear_class)

print("🔹 Baseline (Linear Regression Scaled) Classification Report:")
print(classification_report(y_test, y_pred_linear_class, zero_division=0))

# Plot all the accuracy scores for the models
plt.figure(figsize=(18, 10))
plt.title("Baseline Model Accuracy Comparison")
plt.barh(list(accuracies.keys()), list(accuracies.values()), color='coral')
plt.xlabel("Accuracy Score")
plt.xlim(0, 1)
plt.grid(axis='x')
plt.tight_layout()
plt.savefig("results/figures/fox_baseline_comparison.png", dpi=300, bbox_inches='tight')
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
evaluate_and_log(
    model=LogisticRegression(solver='liblinear', random_state=42),
    model_name="Logistic Regression",
    X_train=X_train_scaled,
    X_test=X_test_scaled,
    y_train=y_train,
    y_test=y_test,
    csv_path="/results/csv/baseline_fox_logistic_regression.csv",
    description="Logistic Regression (Scaled)",
    feature_vector=X.columns.tolist(),  # This is essential for row-wise
    technique="None"
)

# evaluate_and_log(
#     model=LogisticRegression(solver='liblinear', random_state=42),
#     model_name="Logistic Regression",
#     X_train=X_train_scaled,
#     X_test=X_test_scaled,
#     y_train=y_train,
#     y_test=y_test,
#     csv_path="results/baseline_fox.csv"
# )
