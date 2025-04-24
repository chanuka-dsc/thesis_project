import os
import pandas as pd

# # Step 1: Always clean up the existing CSV before working
# def clean_existing_csv(csv_path, expected_cols):
#     """Ensure the CSV file exists and only contains valid columns and rows."""
#     if os.path.exists(csv_path):
#         try:
#             df = pd.read_csv(csv_path)

#             # Keep only expected columns and drop malformed rows
#             df = df[expected_cols]
#             df = df.dropna(subset=['feature_name'])  # Ensure no blank rows

#             # Drop duplicates based on key columns
#             df.drop_duplicates(subset=['seed', 'model', 'feature_name', 'technique'], keep='last', inplace=True)

#         except Exception as e:
#             print(f"CSV malformed or broken: {e}. Resetting file.")
#             df = pd.DataFrame(columns=expected_cols)
#     else:
#         df = pd.DataFrame(columns=expected_cols)

#     # Overwrite the CSV with cleaned data (always)
#     df.to_csv(csv_path, index=False)

# Step 2: Use this cleaned CSV file before logging new results
def log_result(csv_path, seed, model_name, description, features, technique, accuracy, f1):
    """
    Logs results per feature (one row per feature), ensuring:
    - Overwrites existing matching rows (based on key)
    - Removes malformed or duplicate rows
    - Cleans CSV file on every run to keep it consistent
    - Avoids duplicate insertions, even on 2nd run
    """

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    expected_cols = ['seed', 'model', 'description', 'feature_name', 'technique', 'accuracy', 'f1_score']
    key_cols = ['seed', 'model', 'feature_name', 'technique']

    def clean_csv(csv_path, expected_cols):
        """Load, clean, and return the existing CSV data"""
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df = df[expected_cols]
                df = df.dropna(subset=['feature_name'])
            except Exception as e:
                print(f"Malformed CSV, reinitializing. Reason: {e}")
                df = pd.DataFrame(columns=expected_cols)
        else:
            df = pd.DataFrame(columns=expected_cols)
        df.to_csv(csv_path, index=False)
        return df

    def normalize_keys(df, key_cols):
        """Ensure all key columns are strings and stripped for safe comparison"""
        for col in key_cols:
            df[col] = df[col].fillna("None").astype(str).str.strip()
        return df

    # Step 1: Clean and load
    existing_df = clean_csv(csv_path, expected_cols)

    # Step 2: Normalize key columns
    existing_df = normalize_keys(existing_df, key_cols)

    # Step 3: Create new DataFrame
    new_df = pd.DataFrame([{
        'seed': seed,
        'model': model_name,
        'description': description,
        'feature_name': feat,
        'technique': technique,
        'accuracy': round(accuracy, 4),
        'f1_score': round(f1, 4)
    } for feat in features])
    new_df = normalize_keys(new_df, key_cols)

    # Step 4: Remove existing rows that match keys in the new data
    filtered_df = existing_df[
        ~existing_df[key_cols].apply(tuple, axis=1).isin(
            new_df[key_cols].apply(tuple, axis=1)
        )
    ]

    # Step 5: Combine and clean
    final_df = pd.concat([filtered_df, new_df], ignore_index=True)
    final_df = final_df[expected_cols].drop_duplicates(subset=key_cols, keep='last')
    final_df.to_csv(csv_path, index=False)