import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
# file_paths_logistic_fox = [
#     "results/csv/base/fox/logistic_regression_selection_f1_scores.csv",
#     "results/csv/base/fox/logistic_regression_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/fox/partial/logistic_regression_combo_distance_geometry+angles.csv",
#     "results/csv/taxonomy/fox/partial/tuned/logistic_regression_combo_tuned_distance_geometry+angles.csv",
# ]

# file_paths_random_fox = [
#     "results/csv/base/fox/random_forest_selection_f1_scores.csv",
#     "results/csv/base/fox/random_forest_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/fox/partial/random_forest_combo_acceleration.csv",
#     "results/csv/taxonomy/fox/partial/tuned/random_forest_combo_tuned_acceleration.csv",
# ]


# file_paths_xg_fox = [
#     "results/csv/base/fox/xgboost_selection_f1_scores.csv",
#     "results/csv/base/fox/xgboost_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/fox/partial/xgboost_combo_distance_geometry+speed.csv",
#     "results/csv/taxonomy/fox/partial/tuned/xgboost_combo_tuned_acceleration.csv",
# ]

# file_paths_fox = [
#     "results/csv/base/fox/mlp_selection_f1_scores.csv",
#     "results/csv/base/fox/mlp_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/fox/partial/mlp_combo_distance_geometry+angles.csv",
#     "results/csv/taxonomy/fox/partial/tuned/mlp_combo_tuned_angles.csv",
# ]

# file_paths_logistic_ais = [
#     "results/csv/base/ais/logistic_regression_selection_f1_scores.csv",
#     "results/csv/base/ais/logistic_regression_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/ais/partial/logistic_regression_combo_angles+speed+acceleration.csv",
#     "results/csv/taxonomy/ais/partial/tuned/logistic_regression_combo_tuned_distance_geometry+angles+speed.csv",
# ]

# file_paths_mlp_ais   = [
#     "results/csv/base/ais/mlp_selection_f1_scores.csv",
#     "results/csv/base/ais/mlp_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/ais/partial/mlp_combo_speed+acceleration.csv",
#     "results/csv/taxonomy/ais/partial/tuned/mlp_combo_tuned_angles+speed+acceleration.csv",
# ]

# file_paths_random_ais = [
#     "results/csv/base/ais/random_forest_selection_f1_scores.csv",
#     "results/csv/base/ais/random_forest_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/ais/partial/random_forest_combo_distance_geometry+speed+acceleration.csv",
#     "results/csv/taxonomy/ais/partial/tuned/random_forest_combo_tuned_distance_geometry+angles+speed.csv",
# ]

# file_paths_xg_ais = [
#     "results/csv/base/ais/xgboost_selection_f1_scores.csv",
#     "results/csv/base/ais/xgboost_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/ais/partial/xgboost_combo_distance_geometry+angles+speed+acceleration.csv",
#     "results/csv/taxonomy/ais/partial/tuned/xgboost_combo_tuned_distance_geometry+angles+speed+acceleration.csv",
# ]

# file_paths_logistic_hurricane = [
#     "results/csv/base/hurricane/logistic_regression_selection_f1_scores.csv",
#     "results/csv/base/hurricane/logistic_regression_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/hurricane/partial/logistic_regression_combo_distance_geometry+angles+speed+acceleration.csv",
#     "results/csv/taxonomy/hurricane/partial/tuned/logistic_regression_combo_tuned_distance_geometry+angles+speed+acceleration.csv",
# ]

# file_paths_mlp_hurricane = [
#     "results/csv/base/hurricane/mlp_selection_f1_scores.csv",
#     "results/csv/base/hurricane/mlp_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/hurricane/partial/mlp_combo_angles+speed+acceleration.csv",
#     "results/csv/taxonomy/hurricane/partial/tuned/mlp_combo_tuned_speed+acceleration.csv",
# ]

# file_paths_random_hurricane = [
#     "results/csv/base/hurricane/random_forest_selection_f1_scores.csv",
#     "results/csv/base/hurricane/random_forest_tuned_selection_f1_scores.csv",
#     "results/csv/taxonomy/hurricane/partial/random_forest_combo_distance_geometry+angles+speed+acceleration.csv",
#     "results/csv/taxonomy/hurricane/partial/tuned/random_forest_combo_tuned_distance_geometry+angles+speed+acceleration.csv",
# ]

file_paths = [
    "results/csv/base/hurricane/xgboost_selection_f1_scores.csv",
    "results/csv/base/hurricane/xgboost_tuned_selection_f1_scores.csv",
    "results/csv/taxonomy/hurricane/partial/tuned/xgboost_combo_tuned_distance_geometry+angles+speed.csv",
    "results/csv/taxonomy/hurricane/partial/tuned/xgboost_combo_tuned_distance_geometry+angles+speed+acceleration.csv",
]

# Labels for the different sources
labels = [
    "Base - Non-tuned (Backward)",
    "Base - Non-tuned (Forward)",
    "Base - Tuned (Backward)",
    "Base - Tuned (Forward)",
    "Taxonomy - Non-tuned",
    "Taxonomy - Tuned",
]

# Prepare an empty dataframe to consolidate all the results
all_results = pd.DataFrame()

# Load and process each file
for i, path in enumerate(file_paths):
    df = pd.read_csv(path)
    if i < 2:
        # Separate backward and forward for the first two files
        backward = df[df["description"].str.contains("backward")].copy()
        backward["Source"] = labels[i * 2]
        forward = df[df["description"].str.contains("forward")].copy()
        forward["Source"] = labels[i * 2 + 1]
        all_results = pd.concat([all_results, backward, forward], ignore_index=True)
    else:
        # Use the single label for taxonomy data
        df["Source"] = labels[i + 2]
        all_results = pd.concat([all_results, df], ignore_index=True)

# Create the boxplot
plt.figure(figsize=(15, 8))
sns.boxplot(data=all_results, x="Source", y="f1_weighted")
plt.xticks(rotation=10, fontsize=10)
plt.title(
    "F1 Weighted Scores for XGBoost - Tuned and Non-Tuned Models",
    fontsize=14,
)
plt.ylabel("F1 Weighted Score", fontsize=10)
plt.xlabel("Feature selection Variant", fontsize=10)
plt.savefig("results/figures/hurricane/xgboost.png")
plt.show()
