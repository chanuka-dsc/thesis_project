import pandas as pd
from utilities import plot_selection_boxplots


# === logistic_regression ===
df_lgr = pd.read_csv("results/csv/logistic_regression_selection_f1_scores.csv")

plot_selection_boxplots(
    df_lgr,
    title="Forward vs Backward Selection F1 comparison Logistic Regression",
    save_path="results/figures/logistic_regression.png",
)


# === decision_tree ===
df_dt = pd.read_csv("results/csv/decision_tree_selection_f1_scores.csv")
plot_selection_boxplots(
    df_dt,
    title="Forward vs Backward Selection F1 comparison Decision Tree",
    save_path="results/figures/decision_tree.png",
)
