import pandas as pd

# Load the uploaded CSV file
file_path = "results/csv/taxonomy/fox/partial/tuned/mlp_combo_tuned_angles.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand its structure
df.head()


# Group by model and selection description (e.g., forward/backward)
for (model_name, desc), group in df.groupby(["model", "description"]):
    for metric in ["f1_macro", "f1_micro", "f1_weighted"]:
        m_mean = group[metric].mean()
        m_std = group[metric].std()
        print(
            f"{model_name} [{desc}] {metric.replace('f1_', '').capitalize():<8} F1: {m_mean:.4f} ± {m_std:.4f}"
        )
