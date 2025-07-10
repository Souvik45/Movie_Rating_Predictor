import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load("best_imdb_model(8).pkl")
scaler = joblib.load("scaler(8).pkl")
feature_columns = joblib.load("feature_columns(3).pkl")
actor_encoder = joblib.load("actor_encoder(2).pkl")
director_encoder = joblib.load("director_encoder(2).pkl")

def encode_actor(name):
    return int(actor_encoder.transform([name])[0]) if name in actor_encoder.classes_ else -1

def encode_director(name):
    return int(director_encoder.transform([name])[0]) if name in director_encoder.classes_ else -1

st.set_page_config(page_title="IMDb Movie Rating Predictor", layout="centered")
st.title("🎬 IMDb Movie Rating Predictor")

st.markdown("""
This app uses a machine learning model (SVR) trained on the IMDb dataset to predict a movie's IMDb rating.
Fill in the form below with your movie's details to get a predicted rating.
""")

# --- Input Form ---
with st.form("movie_form"):
    genre_input = st.text_input("Genres (separated by |)", "Action|Adventure|Sci-Fi")
    actor_1 = st.text_input("Actor 1 Name", "Leonardo DiCaprio")
    actor_2 = st.text_input("Actor 2 Name", "Joseph Gordon-Levitt")
    actor_3 = st.text_input("Actor 3 Name", "Elliot Page")
    director = st.text_input("Director Name", "Christopher Nolan")
    plot_keywords = st.text_input("Plot Keywords (comma-separated)", "dream,subconscious,heist")
    duration = st.slider("Duration (in minutes)", 60, 240, 120)
    budget = st.number_input("Budget in USD", value=10000000, step=1000000)

    submitted = st.form_submit_button("🎯 Predict IMDb Rating")

# --- Define necessary transformations based on training ---
genre_columns = ['Action', 'Adventure', 'Sci-Fi', 'Comedy', 'Drama', 'Romance', 'Horror', 'Thriller', 'Fantasy', 'Animation']
keyword_columns = ['dream', 'subconscious', 'heist', 'love', 'war', 'future', 'space', 'murder', 'death', 'family',
                   'friendship', 'school', 'revenge', 'violence', 'music', 'battle', 'escape', 'based', 'robot', 'monster']

# Sample Label Encoders (you should ideally save and load them from training)
sample_names = {name: idx for idx, name in enumerate([
    "Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page", "Christopher Nolan"
])}
def encode_name(name):
    return sample_names.get(name, 0)

# --- On Submission ---
if submitted:
    # Genre one-hot
    genre_values = [1 if g in genre_input.split("|") else 0 for g in genre_columns]

    # Keyword one-hot
    kw_list = plot_keywords.lower().replace(" ", "").split(",")
    keyword_values = [1 if kw in kw_list else 0 for kw in keyword_columns]

    # Encode names (you should save your real LabelEncoders in practice)
    actor_1_encoded = encode_name(actor_1)
    actor_2_encoded = encode_name(actor_2)
    actor_3_encoded = encode_name(actor_3)
    director_encoded = encode_name(director)

    # Create final input
    # Create dictionary for inputs
    user_input_dict = {
        "duration": duration,
        "budget": budget,
        "actor_1_name": encode_name(actor_1),
        "actor_2_name": encode_name(actor_2),
        "actor_3_name": encode_name(actor_3),
        "director_name": encode_name(director),
    }

    # Add genres
    for g in genre_columns:
        user_input_dict[g] = 1 if g in genre_input.split("|") else 0

    # Add plot keywords
    kw_list = plot_keywords.lower().replace(" ", "").split(",")
    for kw in keyword_columns:
        user_input_dict[kw] = 1 if kw in kw_list else 0

    # 👇 After building user_input_dict (actors, director, genres, keywords)

    # 🔐 Permanently fix missing column error
    for col in feature_columns:
        if col not in user_input_dict:
            user_input_dict[col] = 0

    # ➡ Convert to DataFrame and reorder
    input_data = pd.DataFrame([user_input_dict])
    input_data = input_data[feature_columns]

    # 🔄 Scale and predict
    input_scaled = scaler.transform(input_data.values)
    prediction = model.predict(input_scaled)[0]


    # Convert to DataFrame
    input_data = pd.DataFrame([user_input_dict])

    # Reorder columns to match training data
    input_data = input_data[feature_columns]

    # Scale and predict
    input_scaled = scaler.transform(input_data.values)
    prediction = model.predict(input_scaled)[0]

    # Display the result
    st.success(f"🎬 Predicted IMDb Rating: {round(prediction, 2)} / 10")

    # Optional: Show raw input
    with st.expander("🔍 View Input Data"):
        st.dataframe(input_data)




# --- Footer ---
st.markdown("---")
st.markdown("✅ Model: Support Vector Regression (SVR)  \n📊 Trained on IMDb 5000 Movie Dataset\n💻 Built by Souvik Samanta")
