import streamlit as st
import joblib
import os
from datetime import datetime

st.set_page_config(page_title="Crop Advisor", page_icon="🌱")
st.title("Crop Recommendation System")

@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    model_path_local = os.path.join(root_dir, 'Models', 'crop_recomender.joblib')
    model_path_cloud_fallback = os.path.join('Models', 'crop_recomender.joblib')
    if os.path.exists(model_path_local):
        return joblib.load(model_path_local)
    elif os.path.exists(model_path_cloud_fallback):
        return joblib.load(model_path_cloud_fallback)
    else:
        st.error(f"Model file not found! Looked in: {model_path_local} and {model_path_cloud_fallback}")
        return None

model = load_model()
# fetching season
current_month = datetime.now().strftime("%B")

with st.sidebar:
    st.header("Input Parameters")
    choice = st.radio("Preferred version", ["Most preferred crop", "Top 5 crop"])
    
    





