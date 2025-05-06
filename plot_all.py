import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSVs
file1 = pd.read_csv("results/csv/base/fox/random_forest_selection_f1_scores.csv")
file2 = pd.read_csv("results/csv/base/fox/tuning_random_forest.csv")
file3 = pd.read_csv(
    "results/csv/taxonomy/fox/partial/random_forest_combo_acceleration.csv"
)
file4 = pd.read_csv(
    "results/csv/taxonomy/fox/partial/tuned/tuning_per_fold_random_forest.csv"
)

# --- Filter Backward and Forward ---

# Non-tuned base
backward_f1 = file1[file1["description"].str.contains("backward", case=False)]
forward_f1 = file1[file1["description"].str.contains("forward", case=False)]

# Tuned base
backward_tuned = file2[file2["description"].str.contains("backward", case=False)]
forward_tuned = file2[file2["description"].str.contains("forward", case=False)]

# Taxonomy
taxonomy_f1 = file3.copy()
taxonomy_tuned = file4.copy()

# --- Set up plotting ---
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Comparison of Weighted F1 and Best Scores Across Selection Methods Random Forest", fontsize=16
)

# Plot 1: Backward (Non-Tuned)
sns.boxplot(data=backward_f1, y="f1_weighted", ax=axes[0, 0])
axes[0, 0].set_title("Backward (Non-Tuned)")
axes[0, 0].set_ylabel("Weighted F1")

# Plot 2: Forward (Non-Tuned)
sns.boxplot(data=forward_f1, y="f1_weighted", ax=axes[0, 1])
axes[0, 1].set_title("Forward (Non-Tuned)")
axes[0, 1].set_ylabel("Weighted F1")

# Plot 3: Taxonomy (Non-Tuned)
sns.boxplot(data=taxonomy_f1, y="f1_weighted", ax=axes[0, 2])
axes[0, 2].set_title("Taxonomy (Non-Tuned)")
axes[0, 2].set_ylabel("Weighted F1")

# Plot 4: Backward (Tuned)
sns.boxplot(data=backward_tuned, y="best_score", ax=axes[1, 0])
axes[1, 0].set_title("Backward (Tuned)")
axes[1, 0].set_ylabel("Best Score")

# Plot 5: Forward (Tuned)
sns.boxplot(data=forward_tuned, y="best_score", ax=axes[1, 1])
axes[1, 1].set_title("Forward (Tuned)")
axes[1, 1].set_ylabel("Best Score")

# Plot 6: Taxonomy (Tuned)
sns.boxplot(data=taxonomy_tuned, y="best_score", ax=axes[1, 2])
axes[1, 2].set_title("Taxonomy (Tuned)")
axes[1, 2].set_ylabel("Best Score")

# Layout adjustment
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("random_forest_boxplots.png", dpi=300)
plt.show()
