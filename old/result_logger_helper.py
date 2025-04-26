from log_results_in_csv import log_result
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def evaluate_and_log(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    csv_path,
    seed=42,
    description="Default sklearn",
    feature_vector=None,
    technique="None"
):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute both macro and micro F1 scores
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

    # Default to all features if not passed
    features_used = feature_vector if feature_vector is not None else X_train.columns.tolist()

    # Removed accuracy, passed f1_macro and f1_micro instead
    log_result(
        csv_path=csv_path,
        seed=seed,
        model_name=model_name,
        description=description,
        features=features_used,
        technique=technique,
        f1_macro=f1_macro,
        f1_micro=f1_micro
    )

    print(f"\n🔹 {model_name} Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
