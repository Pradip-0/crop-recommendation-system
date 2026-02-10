import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime, date

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AgriSmart Advisor", page_icon="🌱", layout="wide")

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
CACHE_FILE_PATH = os.path.join(root_dir, "daily_weather_cache.csv")
MODELS_DIR = os.path.join(root_dir, "Models")

HA_TO_SQFT = 107639.0 

# --- CROP MAPPING (The Fix) ---
# This list MUST be sorted alphabetically to match the model's training
CROP_LIST = [
    'Arhar/Tur', 'Banana', 'Black pepper', 'Cardamom', 'Cashewnut', 
    'Coconut', 'Coriander', 'Cowpea(Lobia)', 'Dry chillies', 'Garlic', 
    'Ginger', 'Gram', 'Groundnut', 'Masoor', 'Moong(Green Gram)', 
    'Onion', 'Peas & beans (Pulses)', 'Potato', 'Sweet potato', 
    'Tapioca', 'Turmeric', 'Urad'
]

# --- EXACT DATA LISTS ---

# 1. STATES (30 States)
UI_STATES = sorted([
    'Assam', 'Karnataka', 'Kerala', 'Meghalaya', 'West Bengal',
    'Puducherry', 'Goa', 'Andhra Pradesh', 'Tamil Nadu', 'Odisha',
    'Bihar', 'Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Mizoram',
    'Punjab', 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh',
    'Tripura', 'Nagaland', 'Chhattisgarh', 'Uttarakhand', 'Jharkhand',
    'Delhi', 'Manipur', 'Jammu and Kashmir', 'Telangana',
    'Arunachal Pradesh', 'Sikkim'
])

# 2. SEASONS (6 Seasons with exact spaces)
SEASON_MAPPING = {
    "Autumn": "Autumn     ",
    "Kharif": "Kharif     ",
    "Rabi": "Rabi       ",
    "Summer": "Summer     ",
    "Whole Year": "Whole Year ",
    "Winter": "Winter     "
}

MONTH_TO_SEASON = {
    "July": "Kharif", "August": "Kharif", "September": "Kharif", "October": "Kharif",
    "November": "Rabi", "December": "Rabi",
    "January": "Winter", "February": "Winter",
    "March": "Summer", "April": "Summer", "May": "Summer", "June": "Summer"
}

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    model_path = os.path.join(MODELS_DIR, 'crop_recomender.joblib')
    # We don't strictly need label_encoder anymore since we hardcoded the list
    
    assets = {}
    
    if os.path.exists(model_path):
        try:
            assets['model'] = joblib.load(model_path)
        except Exception as e:
             st.error(f"🚨 Error loading model: {e}")
             return None
    else:
        # Fallback for local testing
        local_path = os.path.join(current_dir, 'crop_recomender.joblib')
        if os.path.exists(local_path):
             assets['model'] = joblib.load(local_path)
        else:
            st.error(f"🚨 Model not found at: `{model_path}`")
            return None
    
    return assets

# --- HELPER FUNCTIONS ---
def get_season_clean(month_name):
    return MONTH_TO_SEASON.get(month_name, "Summer")

def calculate_vpd(temp_c, rh_pct):
    svp = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    avp = svp * (rh_pct / 100)
    return max(0, svp - avp)

def get_weather_data():
    if os.path.exists(CACHE_FILE_PATH):
        try:
            return pd.read_csv(CACHE_FILE_PATH)
        except:
            return None
    return None

def generate_manual_features():
    """
    Constructs the 44 feature names in the exact order the model expects.
    Order: Numerical -> Sorted Seasons -> Sorted States
    """
    # 1. Numerical Columns (8 cols)
    num_cols = [
        'Area', 'Annual_Rainfall', 'Avg_Temperature', 'humidity_pct', 
        'soil_N_kg_sq_foot', 'soil_P_kg_sq_foot', 'soil_K_kg_sq_foot', 'soil_pH'
    ]
    
    # 2. Season Columns
    raw_seasons = list(SEASON_MAPPING.values())
    season_cols = [f"Season_{s}" for s in sorted(raw_seasons)]
    
    # 3. State Columns
    state_cols = [f"State_{s}" for s in sorted(UI_STATES)]
    
    return num_cols + season_cols + state_cols

# --- MAIN APP UI ---

assets = load_assets()
weather_df = get_weather_data()

st.title("🌾 Smart Kitchen Garden Advisor")
st.markdown("Your AI-powered guide for home farming.")

with st.sidebar:
    st.header("1. Location & Area")
    state_input = st.selectbox("Select State", UI_STATES)
    area_sqft = st.number_input("Garden Area (sq ft)", min_value=10, value=100)

    st.divider()
    st.header("2. Soil Details")
    
    with st.expander("ℹ️ How to find N, P, K values?"):
        st.markdown("**Enter values in kg/hectare.** We will convert them for you.")
    
    N_ha = st.number_input("Nitrogen (N)", value=140)
    P_ha = st.number_input("Phosphorus (P)", value=40)
    K_ha = st.number_input("Potassium (K)", value=40)
    ph = st.slider("Soil pH", 4.0, 9.5, 6.5)

    predict_btn = st.button("Get Recommendation", type="primary")

# --- MAIN LOGIC ---

if weather_df is not None and state_input in weather_df['State'].values:
    state_data = weather_df[weather_df['State'] == state_input].iloc[0]
    
    st.subheader(f"Current Climate Analysis: {state_input}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{state_data['Avg_Temperature']:.1f}°C")
    c2.metric("Rainfall", f"{state_data['Annual_Rainfall']:.1f}mm")
    c3.metric("Humidity", f"{state_data['humidity_pct']:.0f}%")
    
    current_vpd = calculate_vpd(state_data['Avg_Temperature'], state_data['humidity_pct'])
    c4.metric("VPD", f"{current_vpd:.2f} kPa")

    if predict_btn and assets and 'model' in assets:
        model = assets['model']

        # --- 1. Get Features ---
        model_cols = generate_manual_features()

        # --- 2. Create Data Frame ---
        data_array = np.zeros((1, len(model_cols)))
        input_df = pd.DataFrame(data_array, columns=model_cols)

        # --- 3. Prepare Values ---
        current_month = datetime.now().strftime("%B")
        clean_season = get_season_clean(current_month)
        messy_season = SEASON_MAPPING.get(clean_season, "Whole Year ") 
        
        # Conversions
        n_sq = N_ha / HA_TO_SQFT
        p_sq = P_ha / HA_TO_SQFT
        k_sq = K_ha / HA_TO_SQFT

        # --- 4. Fill Numerical Columns ---
        input_df['Area'] = area_sqft
        input_df['Annual_Rainfall'] = state_data['Annual_Rainfall']
        input_df['Avg_Temperature'] = state_data['Avg_Temperature']
        input_df['humidity_pct'] = state_data['humidity_pct']
        input_df['soil_N_kg_sq_foot'] = n_sq
        input_df['soil_P_kg_sq_foot'] = p_sq
        input_df['soil_K_kg_sq_foot'] = k_sq
        input_df['soil_pH'] = ph

        # --- 5. Fill Categorical (One-Hot) ---
        
        # A. Fill Season
        season_col_name = f"Season_{messy_season}"
        if season_col_name in input_df.columns:
            input_df[season_col_name] = 1
        else:
            st.error(f"❌ Error: Could not find column '{season_col_name}'.")
            st.stop()

        # B. Fill State
        state_col_name = f"State_{state_input}"
        if state_col_name in input_df.columns:
            input_df[state_col_name] = 1
        else:
             st.error(f"❌ Error: Model missing column for '{state_input}'.")
             st.stop()

        # --- 6. Predict ---
        try:
            # Get the index (e.g., 17)
            prediction_idx = int(model.predict(input_df)[0])
            
            # Map index to Name using our manual list
            if 0 <= prediction_idx < len(CROP_LIST):
                crop_name = CROP_LIST[prediction_idx]
                st.success(f"🌱 Recommended Crop: **{crop_name}**")
            else:
                st.error(f"⚠️ Prediction Index {prediction_idx} is out of range.")

            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

elif weather_df is None:
    st.info("Loading weather data... (Please ensure the daily update script has run)")
