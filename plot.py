import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load your data
df = pd.read_csv("results/csv/foxxgboost_selection_f1_scores.csv")

# Define metrics & techniques
metrics    = ['f1_macro', 'f1_micro', 'f1_weighted']
techniques = ['forward', 'backward']
colors     = {'forward':'skyblue', 'backward':'lightgreen'}

# Prepare data arrays per technique
data = {
    tech: [df[df['description']==tech][m].values for m in metrics]
    for tech in techniques
}

# Positions for grouped boxes
x = [1, 2, 3]  # one slot per metric
offset = 0.2   # half the box width
positions = {
    'forward': [xi - offset for xi in x],
    'backward':[xi + offset for xi in x]
}

fig, ax = plt.subplots(figsize=(8,5))

for tech in techniques:
    bp = ax.boxplot(
        data[tech],
        positions=positions[tech],
        widths=0.35,
        patch_artist=True,
        medianprops=dict(color='red')
    )
    for box in bp['boxes']:
        box.set_facecolor(colors[tech])

# Format axes
ax.set_xticks(x)
ax.set_xticklabels([m.replace('f1_','').capitalize() for m in metrics])
ax.set_xlabel('F1 Metric')
ax.set_ylabel('Score')
ax.set_title('F1 Score Comparison: Forward vs. Backward Selection')

# Legend
patch_f = mpatches.Patch(color=colors['forward'],  label='Forward')
patch_b = mpatches.Patch(color=colors['backward'], label='Backward')
ax.legend(handles=[patch_f, patch_b], loc='upper left')

plt.tight_layout()
fig.savefig('boxplot_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()
