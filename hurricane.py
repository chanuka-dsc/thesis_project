import pandas as pd

# Load the data
df = pd.read_csv('datasets/old/hurricanes-point-feats-extracted.csv')

# Extract year from the 'id' field (first 4 digits)
df['year'] = df['tid'].astype(str).str[:4].astype(int)

# Filter for the last 20 years (2005 to 2025 inclusive)
df_recent = df[(df['year'] >= 2014) & (df['year'] <= 2024)]

# Drop the 'year' column if you don't need it
df_recent = df_recent.drop(columns=['year'])

# Save or use the filtered dataframe
df_recent.to_csv('trajectories_last20years.csv', index=False)
