# 🎬 IMDb Movie Rating Predictor

This is a machine learning project that predicts the IMDb rating of a movie based on its metadata — such as director, main actors, budget, duration, and more. The app is built with Streamlit and uses a trained Support Vector Regression (SVR) model under the hood.

🚀 Live Demo

> Coming Soon — You can deploy this on Streamlit Cloud or Hugging Face Spaces.



📌 Features

Predicts IMDb rating of any movie using:

Director and actor names
Budget, duration, and gross revenue
Facebook likes (actors, movie, director)
Language, country, content rating, and more
Categorical encoding using pre-trained LabelEncoders
Feature scaling using StandardScaler
Clean and simple user interface with Streamlit


📁 Project Structure

📦 imdb-rating-predictor/
├── app.py                        # Streamlit app
├── requirements.txt             # Required Python packages
├── README.md
│
├── data/
│   └── movie_metadata.csv       # Dataset (cleaned)
│
├── models/
│   ├── model.pkl                # Trained SVR model
│   ├── scaler.pkl               # Scaler used for numeric features
│   ├── actor_encoder.pkl        # LabelEncoder for actors
│   ├── director_encoder.pkl     # LabelEncoder for directors
│   └── feature_columns.pkl      # List of features used during training

⚙️ How to Run Locally

1. Clone the repository
git clone https://github.com/Souvik45/Movie_Rating_Predictor.git
cd imdb-rating-predictor
2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run app.py

🧠 Model Info
Algorithm: Support Vector Regressor (SVR)

Preprocessing:

Numerical features scaled using StandardScaler
Categorical features (like actor and director names) encoded using LabelEncoder
Trained on IMDb movie dataset with 28 features


📊 Sample Input Fields

Director Name
Lead Actor Name
Duration
Budget
Gross
Facebook Likes (actors, movie, director)
Genre, Language, Country, Content Rating


📌 Requirements

See requirements.txt or install manually:
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn

🙌 Acknowledgements

Dataset Source: IMDb movie metadata
Streamlit for app interface
scikit-learn for model training


🧑‍💻 Author
Souvik Samanta
Feel free to connect or give a ⭐️ if you like the project!


---
