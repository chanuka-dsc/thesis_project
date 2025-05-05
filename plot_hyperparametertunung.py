import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure output folder exists
os.makedirs("figures", exist_ok=True)

# Load the tuning results
df = pd.read_csv("results/hyperparam_grid_5fold.csv")

# Models and metrics
models = df['model'].unique().tolist()
metrics = ['cv_best_score', 'test_f1_score']
metric_labels = ['CV Best Score', 'Test Set F1 Score']
colors = ['lightblue', 'lightgreen']  # colors for the two metrics

# Create side-by-side subplots for each model
n_models = len(models)
fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 6), sharey=True)

for ax, model in zip(axes, models):
    # Collect the two metric values for this model
    values = [df.loc[df['model'] == model, m].values for m in metrics]
    
    # Draw boxplot on this axis
    bp = ax.boxplot(
        values,
        patch_artist=True,
        showmeans=True,
        widths=0.6
    )
    
    # Color each box
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # Style medians and means
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    for mean in bp['means']:
        mean.set(marker='D', markeredgecolor='black', markerfacecolor='yellow')
    
    # Set labels and title for this subplot
    ax.set_title(model)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Common y-label
fig.text(0.04, 0.5, 'F1 Score (weighted)', va='center', rotation='vertical', fontsize=12)
fig.suptitle('CV vs Test F1 Scores by Model', fontsize=14)

plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
out_path = "figures/all_models_f1_boxplots.png"
fig.savefig(out_path, dpi=300)
print(f"Saved combined boxplots to {out_path}")
plt.show()
