import pandas as pd

# Step 1: Read your file
df = pd.read_csv("datasets/fox-point-feats-extracted.csv")
feature_names = df.columns.tolist()

# Step 2: Create simple 2-level taxonomy
taxonomy_rows = []

for feature in feature_names:
    if feature.startswith("distance_geometry"):
        taxonomy_rows.append(
            {
                "top_level": "geometric",
                "mid_level": "distance_geometry",
                "feature_name": feature,
            }
        )
    elif feature.startswith("angles"):
        taxonomy_rows.append(
            {"top_level": "geometric", "mid_level": "angles", "feature_name": feature}
        )
    elif feature.startswith("speed"):
        taxonomy_rows.append(
            {"top_level": "kinematic", "mid_level": "speed", "feature_name": feature}
        )
    elif feature.startswith("acceleration"):
        taxonomy_rows.append(
            {
                "top_level": "kinematic",
                "mid_level": "acceleration",
                "feature_name": feature,
            }
        )
    else:
        taxonomy_rows.append(
            {"top_level": "unknown", "mid_level": "unknown", "feature_name": feature}
        )

# Step 3: Save to CSV
taxonomy_df = pd.DataFrame(taxonomy_rows)
taxonomy_df.to_csv("taxonomy_simple.csv", index=False)
