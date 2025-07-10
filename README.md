# 🎬 Movie Rating Predictor

A machine learning project that predicts IMDB ratings for movies based on metadata like cast, director, genres, and other features.

## 📂 Project Structure

- Project.ipynb - Main Jupyter notebook with data cleaning, EDA & model building  
- app.py - Backend deployment script (e.g., Flask or Streamlit)  
- movie_metadata.csv - Dataset used for training and prediction  
- best_imdb_model(8).pkl - Trained ML model  
- director_encoder(2).pkl - LabelEncoder/OneHotEncoder for director names  
- encoder(2).pkl - Encoders for other categorical features  
- feature_columns(3).pkl - Final selected feature columns used for model  
- scaler(8).pkl - Scaler (e.g., StandardScaler) used in preprocessing  
- README.md - Project documentation  

## 🚀 Features

- Predicts movie IMDB ratings with good accuracy  
- Handles feature preprocessing using encoders and scalers  
- Deployable via app.py (Flask or Streamlit)  
- Easily extendable with more features or models  

## 🧠 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib/Seaborn (for EDA)  
- Flask / Streamlit (for deployment)  
- Jupyter Notebook  

## 📊 Dataset

Dataset: movie_metadata.csv  
Originally sourced from IMDB/Kaggle.  

*Features used*:
- Title, Director  
- Cast  
- Genres  
- Budget, Gross  
- Duration, etc.  

*Target*:
- IMDB Rating
