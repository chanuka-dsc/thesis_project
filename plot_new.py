import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


#  labels
common_labels = [
    "Base - Non-tuned (Backward)",
    "Base - Non-tuned (Forward)",
    "Base - Tuned (Backward)",
    "Base - Tuned (Forward)",
    "Taxonomy - Non-tuned",
    "Taxonomy - Tuned",
]

# Dictionary with only dataset/model specific info
result_sets = {
    "fox_logistic": {
        "file_paths": [
            "results/csv/base/fox/logistic_regression_selection_f1_scores.csv",
            "results/csv/base/fox/logistic_regression_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/fox/partial/logistic_regression_combo_distance_geometry+angles.csv",
            "results/csv/taxonomy/fox/partial/tuned/logistic_regression_combo_tuned_distance_geometry+angles.csv",
        ],
        "plot_title": "F1 Weighted Scores for Logistic Regression - Fox",
        "output_path": "results/figures/fox/logistic_regression.png",
        "YMIN": 0.15,
        "YMAX": 0.95,
    },
    "fox_random_forest": {
        "file_paths": [
            "results/csv/base/fox/random_forest_selection_f1_scores.csv",
            "results/csv/base/fox/random_forest_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/fox/partial/random_forest_combo_acceleration.csv",
            "results/csv/taxonomy/fox/partial/tuned/random_forest_combo_tuned_acceleration.csv",
        ],
        "plot_title": "F1 Weighted Scores for Random Forest - Fox",
        "output_path": "results/figures/fox/random_forest.png",
        "YMIN": 0.15,
        "YMAX": 0.95,
    },
    "fox_xgboost": {
        "file_paths": [
            "results/csv/base/fox/xgboost_selection_f1_scores.csv",
            "results/csv/base/fox/xgboost_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/fox/partial/xgboost_combo_distance_geometry+speed.csv",
            "results/csv/taxonomy/fox/partial/tuned/xgboost_combo_tuned_acceleration.csv",
        ],
        "plot_title": "F1 Weighted Scores for XGBoost - Fox",
        "output_path": "results/figures/fox/xgboost.png",
        "YMIN": 0.15,
        "YMAX": 0.95,
    },
    "fox_mlp": {
        "file_paths": [
            "results/csv/base/fox/mlp_selection_f1_scores.csv",
            "results/csv/base/fox/mlp_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/fox/partial/mlp_combo_distance_geometry+angles.csv",
            "results/csv/taxonomy/fox/partial/tuned/mlp_combo_tuned_angles.csv",
        ],
        "plot_title": "F1 Weighted Scores for MLP - Fox",
        "output_path": "results/figures/fox/mlp.png",
        "YMIN": 0.15,
        "YMAX": 0.95,
    },
    "ais_logistic": {
        "file_paths": [
            "results/csv/base/ais/logistic_regression_selection_f1_scores.csv",
            "results/csv/base/ais/logistic_regression_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/ais/partial/logistic_regression_combo_angles+speed+acceleration.csv",
            "results/csv/taxonomy/ais/partial/tuned/logistic_regression_combo_tuned_distance_geometry+angles+speed.csv",
        ],
        "plot_title": "F1 Weighted Scores for Logistic Regression - AIS",
        "output_path": "results/figures/ais/logistic_regression.png",
        "YMIN": 0.5,
        "YMAX": 0.9,
    },
    "ais_mlp": {
        "file_paths": [
            "results/csv/base/ais/mlp_selection_f1_scores.csv",
            "results/csv/base/ais/mlp_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/ais/partial/mlp_combo_speed+acceleration.csv",
            "results/csv/taxonomy/ais/partial/tuned/mlp_combo_tuned_angles+speed+acceleration.csv",
        ],
        "plot_title": "F1 Weighted Scores for MLP - AIS",
        "output_path": "results/figures/ais/mlp.png",
        "YMIN": 0.5,
        "YMAX": 0.9,
    },
    "ais_random_forest": {
        "file_paths": [
            "results/csv/base/ais/random_forest_selection_f1_scores.csv",
            "results/csv/base/ais/random_forest_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/ais/partial/random_forest_combo_distance_geometry+speed+acceleration.csv",
            "results/csv/taxonomy/ais/partial/tuned/random_forest_combo_tuned_distance_geometry+angles+speed.csv",
        ],
        "plot_title": "F1 Weighted Scores for Random Forest - AIS",
        "output_path": "results/figures/ais/random_forest.png",
        "YMIN": 0.5,
        "YMAX": 0.9,
    },
    "ais_xgboost": {
        "file_paths": [
            "results/csv/base/ais/xgboost_selection_f1_scores.csv",
            "results/csv/base/ais/xgboost_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/ais/partial/xgboost_combo_distance_geometry+angles+speed+acceleration.csv",
            "results/csv/taxonomy/ais/partial/tuned/xgboost_combo_tuned_distance_geometry+angles+speed+acceleration.csv",
        ],
        "plot_title": "F1 Weighted Scores for XGBoost - AIS",
        "output_path": "results/figures/ais/xgboost.png",
        "YMIN": 0.5,
        "YMAX": 0.9,
    },
    "hurricane_logistic": {
        "file_paths": [
            "results/csv/base/hurricane/logistic_regression_selection_f1_scores.csv",
            "results/csv/base/hurricane/logistic_regression_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/hurricane/partial/logistic_regression_combo_distance_geometry+angles+speed+acceleration.csv",
            "results/csv/taxonomy/hurricane/partial/tuned/logistic_regression_combo_tuned_distance_geometry+angles+speed+acceleration.csv",
        ],
        "plot_title": "F1 Weighted Scores for Logistic Regression - Hurricane",
        "output_path": "results/figures/hurricane/logistic_regression.png",
        "YMIN": 0.35,
        "YMAX": 0.6,
    },
    "hurricane_mlp": {
        "file_paths": [
            "results/csv/base/hurricane/mlp_selection_f1_scores.csv",
            "results/csv/base/hurricane/mlp_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/hurricane/partial/mlp_combo_angles+speed+acceleration.csv",
            "results/csv/taxonomy/hurricane/partial/tuned/mlp_combo_tuned_speed+acceleration.csv",
        ],
        "plot_title": "F1 Weighted Scores for MLP - Hurricane",
        "output_path": "results/figures/hurricane/mlp.png",
        "YMIN": 0.35,
        "YMAX": 0.6,
    },
    "hurricane_random_forest": {
        "file_paths": [
            "results/csv/base/hurricane/random_forest_selection_f1_scores.csv",
            "results/csv/base/hurricane/random_forest_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/hurricane/partial/random_forest_combo_distance_geometry+angles+speed+acceleration.csv",
            "results/csv/taxonomy/hurricane/partial/tuned/random_forest_combo_tuned_distance_geometry+angles+speed+acceleration.csv",
        ],
        "plot_title": "F1 Weighted Scores for Random Forest - Hurricane",
        "output_path": "results/figures/hurricane/random_forest.png",
        "YMIN": 0.35,
        "YMAX": 0.6,
    },
    "hurricane_xgboost": {
        "file_paths": [
            "results/csv/base/hurricane/xgboost_selection_f1_scores.csv",
            "results/csv/base/hurricane/xgboost_tuned_selection_f1_scores.csv",
            "results/csv/taxonomy/hurricane/partial/tuned/xgboost_combo_tuned_distance_geometry+angles+speed.csv",
            "results/csv/taxonomy/hurricane/partial/tuned/xgboost_combo_tuned_distance_geometry+angles+speed+acceleration.csv",
        ],
        "plot_title": "F1 Weighted Scores for XGBoost - Hurricane",
        "output_path": "results/figures/hurricane/xgboost.png",
        "YMIN": 0.35,
        "YMAX": 0.6,
    },
}

palette = sns.color_palette("Set1", n_colors=6)

for name, config in result_sets.items():
    all_results = pd.DataFrame()
    file_paths = config["file_paths"]
    labels = common_labels

    for i, path in enumerate(file_paths):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        df = pd.read_csv(path)
        if i < 2:
            backward = df[df["description"].str.contains("backward", case=False)].copy()
            backward["Source"] = labels[i * 2]
            forward = df[df["description"].str.contains("forward", case=False)].copy()
            forward["Source"] = labels[i * 2 + 1]
            all_results = pd.concat([all_results, backward, forward], ignore_index=True)
        else:
            label_index = i + 2 if len(labels) > len(file_paths) else i
            df["Source"] = labels[label_index]
            all_results = pd.concat([all_results, df], ignore_index=True)

    if all_results.empty:
        print(f"No data to plot for {name}.")
        continue

    plt.figure(figsize=(15, 8))
    sns.boxplot(
        data=all_results,
        x="Source",
        y="f1_weighted",
        hue="Source", 
        palette=palette,
        dodge=False,
        legend=False,
    )
    plt.xticks(rotation=10, fontsize=10)
    plt.title(config["plot_title"], fontsize=14)
    plt.ylabel("F1 Weighted Score", fontsize=10)
    plt.xlabel("Feature selection Variant", fontsize=10)
    plt.ylim(config["YMIN"], config["YMAX"])
    plt.tight_layout()
    output_dir = os.path.dirname(config["output_path"])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(config["output_path"])
    plt.show()
    print(f"Saved plot: {config['output_path']}")
    plt.close()
