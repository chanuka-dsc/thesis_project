#!/usr/bin/env python3
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Suppress convergence & XGBoost warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")


# 1) Load & encode the Fox dataset
df = pd.read_csv("datasets/fox-point-feats-extracted.csv")
X  = df.drop(columns=["tid", "label"], errors="ignore")
y  = LabelEncoder().fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2) Define base estimators (all will be wrapped in a Pipeline for scaling)
models = {
    "LogisticRegression": LogisticRegression(max_iter=10000, random_state=42),
    "RandomForest"      : RandomForestClassifier(n_jobs=-1, random_state=42),
    "XGBoost"           : XGBClassifier(            # no more use_label_encoder
                              eval_metric="logloss",
                              n_jobs=-1,
                              random_state=42
                          ),
    "MLP"               : MLPClassifier(
                              max_iter=10000,
                              tol=1e-4,
                              early_stopping=True,
                              validation_fraction=0.1,
                              n_iter_no_change=10,
                              random_state=42
                          ),
}


# 3) Compact, explicit parameter grids (3–5 choices each)
param_grids = {
    "LogisticRegression": {
        "est__C"      : [0.01, 0.1, 1.0, 10.0],
        "est__penalty": ["l1", "l2"],
        "est__solver" : ["liblinear", "saga"]
    },
    "RandomForest": {
        "est__n_estimators": [100, 300, 500, 1000],
        "est__max_depth"   : [None, 10, 20, 30],
        "est__max_features": ["sqrt", "log2", 8, 16]   # powers of two
    },
    "XGBoost": {
        "est__n_estimators"  : [100, 300, 500, 1000],
        "est__max_depth"     : [3, 6, 10, 15],
        "est__learning_rate" : [0.01, 0.05, 0.1, 0.2],
        "est__subsample"     : [0.5, 0.7, 0.9, 1.0]
    },
    "MLP": {
        "est__hidden_layer_sizes": [(50,), (100,), (50,50), (100,50)],
        "est__alpha"             : [1e-5, 1e-4, 1e-3, 1e-2],
        "est__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3]
    }
}

# 4) Shared 5-fold Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

os.makedirs("results", exist_ok=True)
records = []

for name, estimator in models.items():
    print(f"\n▶ Tuning {name} via GridSearchCV (5-fold)…")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("est",    estimator)
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[name],
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score=np.nan
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f" ✔ {name} — CV f1_weighted: {grid.best_score_:.4f}, "
          f"test f1_weighted: {test_f1:.4f}")

    records.append({
        "model"          : name,
        "best_params"    : grid.best_params_,
        "cv_best_score" : grid.best_score_,
        "test_f1_score" : test_f1
    })

# 5) Save all results
out_df = pd.DataFrame(records)
out_df.to_csv("results/hyperparam_grid_5fold.csv", index=False)
print("\n✅ All results written to results/hyperparam_grid_5fold.csv")
