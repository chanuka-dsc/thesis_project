import pandas as pd

df = pd.read_csv("datasets/hurricanes-point-feats-extracted.csv")

top_labels = df["label"].value_counts().nlargest(4).index

filtered_df = df[df["label"].isin(top_labels)]

print(filtered_df["label"].value_counts())

filtered_df.to_csv(
    "datasets/hurricanes_point_feats_extracted_top_4_classes_dataset.csv",
    index=False,
)


