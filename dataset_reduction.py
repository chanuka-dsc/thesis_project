import pandas as pd
from sklearn.utils import resample

# df = pd.read_csv("datasets/hurricanes-point-feats-extracted.csv")

# Get the counts of each class
# class_counts = df["label"].value_counts()

# Display the class counts
# print(class_counts)

# # Define the target classes
# target_classes = [30.0, 52.0, 60.0, 70.0]


# # Filter for only the target classes
# filtered_df = df[df["label"].isin(target_classes)]

# # Save the filtered data to a new CSV file
# filtered_file_path = "datasets/ais_filtered_classes.csv"
# filtered_df.to_csv(filtered_file_path, index=False


# Hurricane classes
# Load the full dataset
df = pd.read_csv("datasets/hurricanes-point-feats-extracted.csv")

# Define the top 5 classes
top_5_classes = ["WP", "SI", "NI", "EP", "SP"]

# Filter for only the target classes
df_top5 = df[df["label"].isin(top_5_classes)]

# Check the class distribution before balancing
print("Initial class distribution:")
print(df_top5["label"].value_counts())

# Create a balanced dataset
balanced_samples = []

for cls in top_5_classes:
    # Filter each class
    cls_subset = df_top5[df_top5["label"] == cls]

    # Resample to exactly 200 samples per class
    balanced_cls = resample(
        cls_subset,
        replace=len(cls_subset) < 200,  # Allow oversampling if fewer than 200
        n_samples=200,
        random_state=42,
    )

    balanced_samples.append(balanced_cls)

# Combine all balanced samples into one dataframe
balanced_df = pd.concat(balanced_samples, ignore_index=True)

# Check the final class distribution
print("\nBalanced class distribution:")
print(balanced_df["label"].value_counts())

# Check the total number of samples
print(f"\nTotal samples: {len(balanced_df)}")

# Save the balanced dataset to a CSV file
output_path = "datasets/hurricane_balanced_top5_classes.csv"
balanced_df.to_csv(output_path, index=False)
print(f"\nBalanced dataset saved to '{output_path}'")
