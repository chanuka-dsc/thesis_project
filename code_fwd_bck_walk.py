import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from log_results_in_csv import log_result  

# === Load and Prepare Dataset ===
df = pd.read_csv("C:/Users/dhruv/Desktop/Thesis work/Datasets/trajectory-feats/amended/fox-point-feats-extracted.csv")

# Separate features and label
X = df.drop(columns=['tid', 'label'], errors='ignore')
y = LabelEncoder().fit_transform(df['label'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Base Model ===
model = LogisticRegression(solver='liblinear', random_state=42)

# === Forward Feature Selection ===
forward_selector = SequentialFeatureSelector(
    model,
    direction='forward',
    scoring='f1_micro',
    n_jobs=-1,
    cv=5,
    n_features_to_select='auto',
    # n_features_to_select=25,  # Set to a specific number of features for demonstration
    tol=None
)
forward_selector.fit(X_train, y_train)
forward_features = X.columns[forward_selector.get_support()].tolist()

# === Backward Feature Selection ===
backward_selector = SequentialFeatureSelector(
    model,
    direction='backward',
    scoring='f1_micro',
    n_jobs=-1,
    cv=5,
    n_features_to_select='auto',
    # n_features_to_select=25,
    tol=None
)
backward_selector.fit(X_train, y_train)
backward_features = X.columns[backward_selector.get_support()].tolist()

# === Evaluate Forward Model ===
X_train_forward = X_train[:, forward_selector.get_support()]
X_test_forward = X_test[:, forward_selector.get_support()]
model_forward = LogisticRegression(solver='liblinear', random_state=42)
model_forward.fit(X_train_forward, y_train)
y_pred_forward = model_forward.predict(X_test_forward)

# === Evaluate Backward Model ===
X_train_backward = X_train[:, backward_selector.get_support()]
X_test_backward = X_test[:, backward_selector.get_support()]
model_backward = LogisticRegression(solver='liblinear', random_state=42)
model_backward.fit(X_train_backward, y_train)
y_pred_backward = model_backward.predict(X_test_backward)

# === Print and Compare Results ===
results = {
    "Forward Selection": {
        "features": forward_features,
        "accuracy": accuracy_score(y_test, y_pred_forward),
        "f1_micro": f1_score(y_test, y_pred_forward, average='micro')
    },
    "Backward Selection": {
        "features": backward_features,
        "accuracy": accuracy_score(y_test, y_pred_backward),
        "f1_micro": f1_score(y_test, y_pred_backward, average='micro')
    }
}

results_df = pd.DataFrame(results).T
print("Feature Selection Summary:\n")
# print(results_df)

# === Nicely print selected features for both methods ===
print("\n🔹 Forward Selection Chose {} Features:".format(len(forward_features)))
for feat in forward_features:
    print("   -", feat)

print("\n🔹 Backward Selection Chose {} Features:".format(len(backward_features)))
for feat in backward_features:
    print("   -", feat)


# === Plot ===
# === Improved Visualization ===
fig, ax = plt.subplots(figsize=(10, 6))

bars = results_df[["accuracy", "f1_micro"]].plot(
    kind='barh',
    ax=ax,
    color=['#1f77b4', '#ff7f0e'],  # blue for accuracy, orange for f1
    edgecolor='black'
)

ax.set_title("Logistic Regression: Forward vs Backward Feature Selection", fontsize=14, weight='bold', pad=15)
ax.set_xlabel("Performance Score", fontsize=12)
ax.set_ylabel("Feature Selection Method", fontsize=12)
ax.set_xlim(0, 1)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(axis='x', linestyle='--', alpha=0.6)
ax.legend(["Accuracy", "F1 Score"], loc='lower right', frameon=True)

# Label exact values on the bars
for container in bars.containers:
    for bar in container:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig("C:/Users/dhruv/Desktop/Thesis work/plots/fox_feature_selection_comparison.png", dpi=300, bbox_inches='tight')
plt.show()


# === Log Results to CSV ===
log_result(
    csv_path="C:/Users/dhruv/Desktop/Thesis work/results/fox_feature_selection_log_forwardxgboost.csv",
    seed=42,
    model_name="Logistic Regression",
    description="Forward Selection",
    features=forward_features,
    technique="Forward Selection",
    accuracy=results["Forward Selection"]["accuracy"],
    f1=results["Forward Selection"]["f1_micro"]
)

log_result(
    csv_path="C:/Users/dhruv/Desktop/Thesis work/results/fox_feature_selection_log_backwardxgboost.csv",
    seed=42,
    model_name="Logistic Regression",
    description="Backward Selection",
    features=backward_features,
    technique="Backward Selection",
    accuracy=results["Backward Selection"]["accuracy"],
    f1=results["Backward Selection"]["f1_micro"]
)
