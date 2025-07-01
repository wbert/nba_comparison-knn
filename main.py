import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Feature keys and full names for display
FEATURES = {
    "Age": "Age",
    "FG": "FG – Field Goals Made",
    "FGA": "FGA – Field Goals Attempted",
    "FG%": "FG% – Field Goal Percentage",
    "3P": "3P – Three-Point Field Goals Made",
    "3PA": "3PA – Three-Point FGA",
    "3P%": "3P% – Three-Point FG Percentage",
    "2P": "2P – Two-Point Field Goals Made",
    "2PA": "2PA – Two-Point FGA",
    "2P%": "2P% – Two-Point FG Percentage",
    "eFG%": "eFG% – Effective Field Goal %",
    "FT": "FT – Free Throws Made",
    "FTA": "FTA – Free Throws Attempted",
    "FT%": "FT% – Free Throw Percentage",
    "ORB": "ORB – Offensive Rebounds",
    "DRB": "DRB – Defensive Rebounds",
    "TRB": "TRB – Total Rebounds",
    "AST": "AST – Assists",
    "STL": "STL – Steals",
    "BLK": "BLK – Blocks",
    "TOV": "TOV – Turnovers",
    "PF": "PF – Personal Fouls",
    "PTS": "PTS – Points",
}


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

st.title("🏀 NBA Player Similarity Finder")
st.write(
    "Enter player stats to find the 3 most similar NBA players based on career metrics."
)

if knn and scaler and player_names is not None:
    st.markdown("### 📊 Player Stats Input")

    user_input = {}

    def input_row(features_row):
        cols = st.columns(3)
        for i, key in enumerate(features_row):
            label = FEATURES[key]
            if "%" in key:
                user_input[key] = cols[i].number_input(
                    label, min_value=0.0, max_value=1.0, value=0.5
                )
            else:
                user_input[key] = cols[i].number_input(label, min_value=0.0, step=0.1)

    feature_keys = list(FEATURES.keys())
    for i in range(0, len(feature_keys), 3):
        input_row(feature_keys[i : i + 3])

    st.markdown("---")

    if st.button("🔍 Find Similar Players"):
        try:
            input_df = pd.DataFrame([user_input])
            normalized_input = scaler.transform(input_df)
            distances, indices = knn.kneighbors(normalized_input)

            st.subheader("👥 Top 3 Similar Players")
            for i, idx in enumerate(indices[0]):
                player = player_names.iloc[idx]
                st.markdown(
                    f"**{i+1}. {player['Player']} ({player['Year']})** – Distance: `{distances[0][i]:.3f}`"
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.stop()
