import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
ffoxi = pd.read_csv('C:/Users/dhruv/Desktop/Thesis work/Datasets/trajectory-feats/fox-point-feats-extracted.csv')

# Prepare data
X = ffoxi.drop(columns=['tid', 'label'], errors='ignore')
y = LabelEncoder().fit_transform(ffoxi['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cross-validation function
def cross_val_report(model, X, y, model_name):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n🔹 {model_name} Cross-validation accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

# -------- Random Forest --------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)
print("🔹 Random Forest Classification Report:")
print(classification_report(y_test, rf_preds))
cross_val_report(rf, X_train_scaled, y_train, "Random Forest")

# Feature Importance (Random Forest)
rf_importance = rf.feature_importances_

indices_rf = np.argsort(rf_importance)[::-1]
plt.figure(figsize=(12, 8))
plt.title("Top 15 Feature Importances (Random Forest)")
plt.bar(range(15), rf_importance[indices_rf][:15])
plt.xticks(range(15), X.columns[indices_rf][:15], rotation=90)
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# -------- XGBoost --------
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train_scaled, y_train)
xgb_preds = xgb.predict(X_test_scaled)
print("\n🔹 XGBoost Classification Report:")
print(classification_report(y_test, xgb_preds))
cross_val_report(xgb, X_train_scaled, y_train, "XGBoost")

# Feature Importance (XGBoost)
xgb_importance = xgb.feature_importances_

indices_xgb = np.argsort(xgb_importance)[::-1]
plt.figure(figsize=(12, 8))
plt.title("Top 15 Feature Importances (XGBoost)")
plt.bar(range(15), xgb_importance[indices_xgb][:15])
plt.xticks(range(15), X.columns[indices_xgb][:15], rotation=90)
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# -------- Logistic Regression --------
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
print("\n🔹 Logistic Regression Classification Report:")
print(classification_report(y_test, lr_preds))
cross_val_report(lr, X_train_scaled, y_train, "Logistic Regression")

# Feature Importance (Logistic Regression coefficients)
lr_coef = np.abs(lr.coef_[0])

indices_lr = np.argsort(lr_coef)[::-1]
plt.figure(figsize=(12, 8))
plt.title("Top 15 Feature Coefficients (Logistic Regression)")
plt.bar(range(15), lr_coef[indices_lr][:15])
plt.xticks(range(15), X.columns[indices_lr][:15], rotation=90)
plt.ylabel("Coefficient Magnitude")
plt.tight_layout()
plt.show()
