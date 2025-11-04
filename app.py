# Libraries
import streamlit as st
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
model.load_model('youtube_trending_model.json')  

# Streamlit App UI
st.title("YouTube Views Prediction")

# User inputs for features
likes = st.number_input("Likes", min_value=0, value=1000)
dislikes = st.number_input("Dislikes", min_value=0, value=100)
comment_count = st.number_input("Comment Count", min_value=0, value=100)
title_length = st.number_input("Title Length", min_value=1, value=10)

# DataFrame for the input data
input_data = pd.DataFrame({
    'likes': [likes],
    'dislikes': [dislikes],
    'comment_count': [comment_count],
    'title_length': [title_length]
})

# Preprocess the input
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Model Prediction
if st.button("Predict Views"):
    predicted_log_views = model.predict(input_data_scaled)
    predicted_views = np.expm1(predicted_log_views)

    st.write(f"Predicted Views: {predicted_views[0]:,.0f}")
