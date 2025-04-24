import pandas as pd
import extract_features as ef

# Replace with your actual CSV file path
df1 = pd.read_csv("C:/Users/dhruv/Desktop/Thesis work/Datasets/trajectory-feats/fox-point-feats.csv")
# df2 = pd.read_csv("C:/Users/dhruv/Desktop/Thesis work/Datasets/trajectory-feats/hurricanes-point-feats.csv")
# df3 = pd.read_csv("C:/Users/dhruv/Desktop/Thesis work/Datasets/trajectory-feats/ais-point-feats.csv")


# Convert time column to datetime
df1['time'] = pd.to_datetime(df1['time'])

# running the feature extraction function
features_foxi = ef.extract_trajectory_features(df1)

print(features_foxi.head())


# Save the extracted features to a CSV file
features_foxi.to_csv("C:/Users/dhruv/Desktop/Thesis work/Datasets/trajectory-feats/amended/fox-point-feats-extracted.csv", index=False)

# df2 = pd.read_csv("C:/Users/dhruv/Desktop/Thesis work/Datasets/trajectory-feats/hurricanes-point-feats.csv")


# # Convert time column to datetime
# df2['time'] = pd.to_datetime(df2['time'])

# # running the feature extraction function
# features_hurri = ef.extract_trajectory_features(df2)

# print(features_hurri.head())


# # Save the extracted features to a CSV file
# features_hurri.to_csv("C:/Users/dhruv/Desktop/Thesis work/Datasets/trajectory-feats/amended/hurricanes-point-feats-extracted.csv", index=False)


# # Convert time column to datetime
# df3['time'] = pd.to_datetime(df3['time'])

# # running the feature extraction function
# features_ais = ef.extract_trajectory_features(df3)

# print(features_ais.head())


# # Save the extracted features to a CSV file
# features_ais.to_csv("C:/Users/dhruv/Desktop/Thesis work/Datasets/trajectory-feats/amended/ais-point-feats-extracted.csv", index=False)