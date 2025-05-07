import pandas as pd
import numpy as np
from collections import Counter
from sklearn.calibration import LabelEncoder

# === Load and Prepare Dataset ===
df = pd.read_csv("datasets/ais-point-feats-extracted.csv")
X = df.drop(columns=["tid", "label"], errors="ignore")
y = LabelEncoder().fit_transform(df["label"])

# Assume X and y are your original DataFrame and target array
min_samples = 5  # or set to n_splits if using StratifiedKFold(n_splits=5)

# Count occurrences of each class
class_counts = Counter(y)

# Identify classes that meet the minimum sample requirement
valid_classes = {cls for cls, count in class_counts.items() if count >= min_samples}

# Create a mask for keeping only those rows
mask = np.isin(y, list(valid_classes))

# Apply the mask to filter X and y
X_filtered = X[mask]
y_filtered = y[mask]

# Optional: confirm result
print("Original class distribution:", class_counts)
print("Filtered class distribution:", Counter(y_filtered))
