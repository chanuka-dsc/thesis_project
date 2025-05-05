from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


# Load extracted features
ffoxi = pd.read_csv('datasets/fox-point-feats-extracted.csv')

# Display the first few rows to inspect the data
print(ffoxi.head())

# Exploring Dataset
# Print all column names to see what features we have
print("Columns in the dataset:")
print(ffoxi.columns.tolist())

# Get a summary of the dataset to see statistics (mean, std, etc.)
print("\nDataset summary:")
print(ffoxi.describe())

# Check for missing values:
print("\nMissing values per column:")
print(ffoxi.isnull().sum())

# Create groups based on column name keywords
feature_groups = {
    "Trajectory ID": [col for col in ffoxi.columns if col.lower() == "tid"],
    "Distance Geometry": [col for col in ffoxi.columns if "distance_geometry" in col],
    "Angles": [col for col in ffoxi.columns if "angles" in col],
    "Speed": [col for col in ffoxi.columns if "speed" in col],
    "Acceleration": [col for col in ffoxi.columns if "acceleration" in col],
    "Label": [col for col in ffoxi.columns if col.lower() == "label"]
}

# Print out the feature groups
for group, features in feature_groups.items():
    print(f"{group}:")
    print(features)
    # print(features.__len__())
    print(f"Number of features: {len(features)}")
    print("-----")
    
print('-' * 100)


# # Prepare data
# X = ffoxi.drop(columns=['tid', 'label'], errors='ignore')  # Features
# y = ffoxi['label']  # Labels

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Random Forest
# print("Random Forest Results:")
# rf_clf = RandomForestClassifier()
# rf_clf.fit(X_train, y_train)
# rf_y_pred = rf_clf.predict(X_test)
# print(classification_report(y_test, rf_y_pred))

# # XGBoost
# print("\nXGBoost Results:")
# xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# xgb_clf.fit(X_train, y_train)
# xgb_y_pred = xgb_clf.predict(X_test)
# print(classification_report(y_test, xgb_y_pred))

# # Logistic Regression
# print("\nLogistic Regression Results:")
# lr_clf = LogisticRegression(max_iter=1000)
# lr_clf.fit(X_train, y_train)
# lr_y_pred = lr_clf.predict(X_test)
# print(classification_report(y_test, lr_y_pred))
