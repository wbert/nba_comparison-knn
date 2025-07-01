import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Features (must match training order exactly)
FEATURES = [
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


# Load model, scaler, and player names
@st.cache_resource
def load_model_and_scaler():
    try:
        knn = joblib.load("models/knn_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        player_data = pd.read_csv("dataset.csv")[["Player", "Year"]]
        return knn, scaler, player_data
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None, None


knn, scaler, player_names = load_model_and_scaler()

st.title("🏀 NBA Player Similarity Search")
st.write("Enter player stats to find the 3 most similar NBA players from your dataset.")

if knn and scaler and player_names is not None:
    # Get input for each feature
    user_input = {}
    for feature in FEATURES:
        if "%" in feature:
            user_input[feature] = st.number_input(
                f"{feature} (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5
            )
        else:
            user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1)

    if st.button("Find Similar Players"):
        try:
            # Prepare and normalize input
            input_df = pd.DataFrame([user_input])
            normalized_input = scaler.transform(input_df)

            # Predict using loaded KNN
            distances, indices = knn.kneighbors(normalized_input)

            # Show top 3 results
            st.subheader("🔍 Top 3 Similar Players:")
            for i, idx in enumerate(indices[0]):
                player = player_names.iloc[idx]
                st.markdown(
                    f"**{i+1}. {player['Player']} ({player['Year']})** - Distance: `{distances[0][i]:.3f}`"
                )

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.stop()
