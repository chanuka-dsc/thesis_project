from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from log_results_in_csv import log_result
from sklearn.metrics import accuracy_score, f1_score
from result_logger_helper import evaluate_and_log
from sklearn.ensemble import RandomForestClassifier

# Load data
ffoxi = pd.read_csv('C:/Users/dhruv/Desktop/Thesis work/Datasets/trajectory-feats/fox-point-feats-extracted.csv')

# Prepare data
X = ffoxi.drop(columns=['tid', 'label'], errors='ignore')
y = LabelEncoder().fit_transform(ffoxi['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Dictionary to store model names and their corresponding accuracy scores
accuracies = {}

# Baseline Model1: Dummy Classifier (most frequent class)
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
accuracies["Dummy (Most Frequent)"] = accuracy_score(y_test, dummy.predict(X_test))

# Evaluate Baseline Performance
print("🔹 Baseline (Dummy Classifier) Classification Report:")
print(classification_report(y_test, y_pred_dummy, zero_division=0))
# added to suppress warnings for zero division

# as this baseline model favors the most frequent class, it will not be very informative.
# SO other baseline models can be used like:

# 2. Dummy Classifier (uniform): This classifier generates predictions uniformly at random, regardless of the training set's class distribution.
#    It can be useful for balanced datasets.

dummy_uniform = DummyClassifier(strategy='uniform', random_state=42)
dummy_uniform.fit(X_train, y_train)
y_pred_dummy_uniform = dummy_uniform.predict(X_test)
accuracies["Dummy (Uniform)"] = accuracy_score(y_test, dummy_uniform.predict(X_test))

# Evaluate Baseline Performance
print("🔹 Baseline (Dummy Classifier Uniform) Classification Report:")
print(classification_report(y_test, y_pred_dummy_uniform))

# 3. Dummy Classifier (stratified): This classifier generates predictions by respecting the training set's class distribution.
#    It can be useful for imbalanced datasets.
dummy_stratified = DummyClassifier(strategy='stratified', random_state=42)
dummy_stratified.fit(X_train, y_train)
y_pred_dummy_stratified = dummy_stratified.predict(X_test)
accuracies["Dummy (Stratified)"] = accuracy_score(y_test, dummy_stratified.predict(X_test))

# Evaluate Baseline Performance
print("🔹 Baseline (Dummy Classifier Stratified) Classification Report:")
print(classification_report(y_test, y_pred_dummy_stratified))

# as this baseline model favors the both classes but still accuracy is not good.

# SO other baseline models can be used like:

# 4. Logistic Regression: A simple linear model that can be used as a baseline for binary classification tasks.
# without scaling
logistic_model_us = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
logistic_model_us.fit(X_train, y_train)
y_pred_logistic_unscaled = logistic_model_us.predict(X_test)
accuracies["Logistic Regression"] = accuracy_score(y_test, logistic_model_us.predict(X_test))

# Evaluate Baseline Performance
print("🔹 Baseline (Logistic Regression Unscaled) Classification Report:")
print(classification_report(y_test, y_pred_logistic_unscaled))

# Scaling for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression baseline model (scaled)
logistic_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)
accuracies["Logistic Regression (Scaled)"] = accuracy_score(y_test, logistic_model.predict(X_test_scaled))

# Evaluate Baseline Performance
print("🔹 Baseline (Logistic Regression) Classification Report:")
print(classification_report(y_test, y_pred_logistic))

#5. Decision Tree Classifier: A simple decision tree model that can be used as a baseline for classification tasks.
# without scaling
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
accuracies["Decision Tree"] = accuracy_score(y_test, decision_tree.predict(X_test))

# Evaluate Baseline Performance
print("🔹 Baseline (Decision Tree) Classification Report:")
print(classification_report(y_test, y_pred_tree))

# 6. Random Forest Classifier: A more complex ensemble model that can be used as a baseline for classification tasks.
# without scaling
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
y_pred_forest = random_forest.predict(X_test)
accuracies["Random Forest"] = accuracy_score(y_test, random_forest.predict(X_test))

# Evaluate Baseline Performance
print("🔹 Baseline (Random Forest) Classification Report:")
print(classification_report(y_test, y_pred_forest))


# 7. XGBoost Classifier: A powerful gradient boosting model that can be used as a baseline for classification tasks.
# without scaling
xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracies["XGBoost"] = accuracy_score(y_test, xgb_model.predict(X_test))
# Evaluate Baseline Performance
print("🔹 Baseline (XGBoost) Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# 8. KNN Classifier: A simple K-Nearest Neighbors model that can be used as a baseline for classification tasks.
# without scaling

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracies["KNN"] = accuracy_score(y_test, knn_model.predict(X_test))
# Evaluate Baseline Performance
print("🔹 Baseline (KNN) Classification Report:")
print(classification_report(y_test, y_pred_knn))

# with scaling
scaler = StandardScaler()
X_train_scaled_knn = scaler.fit_transform(X_train)
X_test_scaled_knn = scaler.transform(X_test)

knn_model_scaled = KNeighborsClassifier()
knn_model_scaled.fit(X_train_scaled_knn, y_train)
y_pred_knn_scaled = knn_model_scaled.predict(X_test_scaled_knn)
accuracies["KNN (Scaled)"] = accuracy_score(y_test, knn_model_scaled.predict(X_test_scaled_knn))

print("🔹 Baseline (KNN Scaled) Classification Report:")
print(classification_report(y_test, y_pred_knn_scaled))

# 9. Support Vector Classifier: A powerful model that can be used as a baseline for classification tasks.
# without scaling
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
accuracies["SVM"] = accuracy_score(y_test, svm_model.predict(X_test))
# probs = svm_model.predict_proba(X_test)
# print("Probabilities:------>", probs)
y_pred_svm = svm_model.predict(X_test)
# Evaluate Baseline Performance
print("🔹 Baseline (SVM) Classification Report:")
print(classification_report(y_test, y_pred_svm, zero_division=0))

# with scaling
scaler = StandardScaler()
X_train_scaled_svm = scaler.fit_transform(X_train)
X_test_scaled_svm = scaler.transform(X_test)

svm_model_scaled = SVC(random_state=42)
svm_model_scaled.fit(X_train_scaled_svm, y_train)
y_pred_svm_scaled = svm_model_scaled.predict(X_test_scaled_svm)
accuracies["SVM (Scaled)"] = accuracy_score(y_test, svm_model_scaled.predict(X_test_scaled_svm))


print("🔹 Baseline (SVM Scaled) Classification Report:")
print(classification_report(y_test, y_pred_svm_scaled, zero_division=0))

# 10. Regression Classifier: A regression model that can be used as a baseline for regression tasks.
# without scaling
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Convert continuous predictions to binary class (0 or 1)
y_pred_linear_class = np.round(y_pred_linear).astype(int)
accuracies["Linear Regression"] = accuracy_score(y_test, y_pred_linear_class)

# Evaluate Baseline Performance
print("🔹 Baseline (Linear Regression) Classification Report:")
print(classification_report(y_test, y_pred_linear_class, zero_division=0))

# with scaling
scaler = StandardScaler()
X_train_scaled_linear = scaler.fit_transform(X_train)
X_test_scaled_linear = scaler.transform(X_test)

# Linear Regression baseline model (scaled)
linear_model_scaled = LinearRegression()
linear_model_scaled.fit(X_train_scaled_linear, y_train)
y_pred_linear_scaled = linear_model_scaled.predict(X_test_scaled_linear)
y_pred_linear_scaled_class = np.round(y_pred_linear_scaled).astype(int)
accuracies["Linear Regression (Scaled)"] = accuracy_score(y_test, y_pred_linear_scaled_class)

print("🔹 Baseline (Linear Regression Scaled) Classification Report:")
print(classification_report(y_test, y_pred_linear_scaled_class, zero_division=0))

# Plot all the accuracy scores for the models
plt.figure(figsize=(18, 10))
plt.title("Baseline Model Accuracy Comparison")
plt.barh(list(accuracies.keys()), list(accuracies.values()), color='coral')
plt.xlabel("Accuracy Score")
plt.xlim(0, 1)
plt.grid(axis='x')
plt.tight_layout()
plt.savefig("C:/Users/dhruv/Desktop/Thesis work/plots/fox_baseline_comparison.png", dpi=300, bbox_inches='tight')
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
    csv_path="C:/Users/dhruv/Desktop/Thesis work/results/baseline_fox_logistic_regression.csv",
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
