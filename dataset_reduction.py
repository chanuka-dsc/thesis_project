import pandas as pd

df = pd.read_csv("datasets/ais-point-feats-extracted.csv")

# Get the counts of each class
class_counts = df["label"].value_counts()

# Display the class counts
print(class_counts)

# Define the target classes
target_classes = [30.0, 52.0, 60.0, 70.0]


# Filter for only the target classes
filtered_df = df[df["label"].isin(target_classes)]

# Save the filtered data to a new CSV file
filtered_file_path = "datasets/ais_filtered_classes.csv"
filtered_df.to_csv(filtered_file_path, index=False)
