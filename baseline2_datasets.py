import os
import pandas as pd
from sklearn.linear_model         import LogisticRegression
from sklearn.tree                 import DecisionTreeClassifier
from sklearn.ensemble             import RandomForestClassifier
from xgboost                      import XGBClassifier
from sklearn.neural_network       import MLPClassifier
from sklearn.preprocessing        import LabelEncoder
from sklearn.impute               import SimpleImputer
# ← CHANGED: import both evaluators
from utilities                    import (
    evaluate_with_cv_seeds_taxonomy_fixed_features,
    evaluate_with_cv_seeds_and_feature_logging,
)

# datasets to run
datasets = {
    "hurricane": "datasets/hurricanes-point-feats-extracted.csv",
    "ais":       "datasets/ais-point-feats-extracted.csv",
}

models = {
    "LogisticRegression": LogisticRegression(
        solver="liblinear", max_iter=2000, random_state=42
    ),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="mlogloss", use_label_encoder=False, n_jobs=-1, random_state=42
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(50,),
        activation='relu',
        solver='adam',
        max_iter=15000,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    ),
}

# make result dirs
os.makedirs("results/baseline", exist_ok=True)
os.makedirs("results/feature_selection", exist_ok=True)

for name, path in datasets.items():
    print(f"\n=== {name.upper()} ===")
    df = pd.read_csv(path).dropna(subset=["label"])

    # determine n_splits dynamically
    counts    = df.label.value_counts()
    min_count = counts.min()
    # at least 2 folds, at most 5, no more than your rarest class can support
    n_splits  = min(5, max(2, min_count))                                # ← CHANGED
    print(f"using n_splits={n_splits} (min label count = {min_count})")

    # drop labels with too few examples
    valid = counts[counts >= n_splits].index
    df    = df[df.label.isin(valid)].copy()                             # ← CHANGED

    # prepare X, y
    X = df.drop(columns=["tid","label"], errors="ignore")
    X = pd.DataFrame(
        SimpleImputer(strategy="mean").fit_transform(X),               # ← CHANGED
        columns=X.columns
    )
    y = LabelEncoder().fit_transform(df.label)

    # --- 1) Baseline (all features) ---
    for mname, model in models.items():
        print(f"-- BASELINE: {mname}")
        res_base = evaluate_with_cv_seeds_taxonomy_fixed_features(
            model=model,
            model_name=mname,
            desc="baseline",
            X=X, y=y,
            n_splits=n_splits                                           # ← CHANGED
        )
        fn = f"results/baseline/{name}_{mname}_baseline.csv"
        res_base.to_csv(fn, index=False)
        print(f"   saved {fn}")

    # --- 2) Forward & Backward selection ---
    for mname, model in models.items():
        for direction in ("forward", "backward"):
            print(f"-- SELECTION: {mname} ({direction})")
            res_sel = evaluate_with_cv_seeds_and_feature_logging(
                model=model,
                model_name=mname,
                desc=direction,
                X=X, y=y,
                n_splits=n_splits                                       # ← CHANGED
            )
            fn = f"results/feature_selection/{name}_{mname}_{direction}.csv"
            res_sel.to_csv(fn, index=False)
            print(f"   saved {fn}")
