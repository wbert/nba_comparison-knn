import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import os

# Load your dataset
df = pd.read_csv("dataset.csv")

# Save the player names and year
player_names = df[["Player", "Year"]]

# Feature columns to keep
features = [
    "Age",
    "FG",
    "FGA",
    "FG%",
    "3P",
    "3PA",
    "3P%",
    "2P",
    "2PA",
    "2P%",
    "eFG%",
    "FT",
    "FTA",
    "FT%",
    "ORB",
    "DRB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
]

# Normalize the stats
scaler = StandardScaler()
normalized_stats = scaler.fit_transform(df[features])

# Fit KNN model on normalized stats (n=4 because we exclude self in evaluation)
knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
knn.fit(normalized_stats)

# Evaluation: collect distances to neighbors for each player (excluding self)
all_mean_distances = []
all_std_distances = []

print("\nSimilarity Evaluation on Training Data:")
for i in range(len(df)):
    distances, indices = knn.kneighbors([normalized_stats[i]])

    # Exclude self (first index)
    neighbor_distances = distances[0][1:]

    mean_dist = np.mean(neighbor_distances)
    std_dist = np.std(neighbor_distances)

    all_mean_distances.append(mean_dist)
    all_std_distances.append(std_dist)

    player = player_names.iloc[i]
    print(
        f"{player['Player']} ({player['Year']}): Mean Dist = {mean_dist:.3f}, Std = {std_dist:.3f}"
    )

# Overall evaluation summary
overall_mean = np.mean(all_mean_distances)
overall_std = np.mean(all_std_distances)

print("\nOverall Similarity Metrics:")
print(f"Average Mean Distance to Neighbors: {overall_mean:.3f}")
print(f"Average Std of Neighbor Distances: {overall_std:.3f}")

# Save model and scaler
try:
    os.makedirs("models", exist_ok=True)
    joblib.dump(knn, "models/knn_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("\n✅ KNN model and scaler saved successfully to 'models/' folder.")
except Exception as e:
    print(f"\n❌ Failed to save models: {e}")
