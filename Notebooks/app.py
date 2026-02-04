import streamlit as st
import joblib
import os
st.set_page_config(page_title="Crop Advisor", page_icon="🌱")

# Your title with the spelling fix
st.title("Crop Recommendation System")

st.write("Welcome! This app helps you find the best crops for your land.")
model_path = os.path.join('..', 'Models', 'crop_recomender.joblib')

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()
st.sidebar()

  


