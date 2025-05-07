import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Parameters
results_folder = "results/csv"
save_folder = "results/figures"
models = [
    "logistic_regression",
    "random_forest",
    "mlp",
    "xgboost",
]  # extend later if needed
mid_levels = ["distance_geometry", "angles", "speed", "acceleration"]
metric_to_plot = "f1_weighted"

# Collect all fold results
all_records = []

for model in models:
    for mid_level in mid_levels:
        file_path = os.path.join(
            results_folder, f"{model}_taxonomy_{mid_level}_f1_scores.csv"
        )
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        for f1 in df[metric_to_plot]:
            all_records.append(
                {
                    "Model": model.replace("_", " ").title(),
                    "Mid Level": mid_level.replace("_", " ").title(),
                    "F1 Score": f1,
                }
            )

# Create a combined DataFrame
full_df = pd.DataFrame(all_records)

# === Create one plot per model ===
for model_name in full_df["Model"].unique():
    model_df = full_df[full_df["Model"] == model_name]

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=model_df,
        x="Mid Level",
        y="F1 Score",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "6",
        },
    )

    plt.title(f"{model_name} - Boxplot of F1 Scores by Mid-Level ({metric_to_plot})")
    plt.xlabel("Feature Group (Mid Level)")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save figure
    safe_model_name = model_name.lower().replace(" ", "_")
    plot_filename = os.path.join(
        save_folder, f"boxplot_{safe_model_name}_{metric_to_plot}.png"
    )
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved boxplot for {model_name}: {plot_filename}")

    plt.close()  # Close figure to avoid memory issues
