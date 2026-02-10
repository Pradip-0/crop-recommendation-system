import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime, date

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AgriSmart Advisor", page_icon="🌱", layout="wide")

# --- CONSTANTS ---
# Path is relative to this script (inside Notebooks/ folder)
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_weather_cache.csv")
HA_TO_SQFT = 107639.0  # Conversion factor

# --- STATE COORDINATES ---
state_coords = {
    'Andhra Pradesh': (15.91, 79.74), 'Arunachal Pradesh': (28.21, 94.72),
    'Assam': (26.20, 92.93), 'Bihar': (25.09, 85.31), 'Chhattisgarh': (21.27, 81.86),
    'Delhi': (28.61, 77.20), 'Goa': (15.29, 74.12), 'Gujarat': (22.25, 71.19),
    'Haryana': (29.05, 76.08), 'Himachal Pradesh': (31.10, 77.17),
    'Jammu And Kashmir': (33.77, 76.57), 'Jharkhand': (23.61, 85.27),
    'Karnataka': (15.31, 75.71), 'Kerala': (10.85, 76.27),
    'Madhya Pradesh': (23.47, 77.94), 'Maharashtra': (19.75, 75.71),
    'Manipur': (24.66, 93.90), 'Meghalaya': (25.46, 91.36),
    'Mizoram': (23.16, 92.93), 'Nagaland': (26.15, 94.56),
    'Odisha': (20.95, 85.09), 'Puducherry': (11.94, 79.80),
    'Punjab': (31.14, 75.34), 'Rajasthan': (27.02, 74.21),
    'Sikkim': (27.53, 88.51), 'Tamil Nadu': (11.12, 78.65),
    'Telangana': (18.11, 79.01), 'Tripura': (23.94, 91.98),
    'Uttar Pradesh': (26.84, 80.94), 'Uttarakhand': (30.06, 79.01),
    'West Bengal': (22.98, 87.85)
}

# --- SEASON MAPPING ---
season_months = {
    "Kharif": ["July", "August", "September", "October"],
    "Rabi": ["November", "December"],
    "Winter": ["January", "February"],
    "Summer": ["March", "April", "May", "June"],
    "Autumn": [],
    "Whole Year": []
}

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    """Loads the Model and the Target Label Encoder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'Models')

    model_path = os.path.join(models_dir, 'crop_recomender.joblib')
    # The notebook saves the target encoder with this name
    le_path = os.path.join(models_dir, 'label_encoder.joblib')

    assets = {}

    # Load Model
    if os.path.exists(model_path):
        try:
            assets['model'] = joblib.load(model_path)
        except Exception as e:
             st.error(f"🚨 Error loading model: {e}")
             return None
    else:
        st.error(f"🚨 Model not found at {model_path}")
        return None

    # Load Target Label Encoder (to convert prediction back to crop name)
    if os.path.exists(le_path):
        try:
            assets['target_le'] = joblib.load(le_path)
        except Exception as e:
            st.warning(f"⚠️ Could not load label encoder: {e}. Prediction will be a number.")
    else:
        st.warning("⚠️ 'label_encoder.joblib' not found. Prediction will be a number.")

    return assets

# --- HELPER FUNCTIONS ---
def get_season_clean(month_name):
    for season, months in season_months.items():
        if month_name in months:
            return season
    return "Summer"

def calculate_vpd(temp_c, rh_pct):
    # VPD calculation from the notebook
    svp = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    avp = svp * (rh_pct / 100)
    return max(0, svp - avp)

def get_weather_data():
    """Reads the pre-fetched weather data from CSV."""
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE)
            # Optional: Check if data is recent
            # if pd.to_datetime(df['Date_Updated'].iloc[0]).date() == date.today():
            return df
        except Exception as e:
            st.error(f"Error reading weather cache: {e}")
            return None
    else:
        st.warning("⚠️ Weather data cache not found. Please run the data update script.")
        return None

# --- MAIN APP UI ---

assets = load_assets()
weather_df = get_weather_data()

st.title("🌾 Smart Kitchen Garden Advisor")
st.markdown("Your AI-powered guide for home farming.")

with st.sidebar:
    st.header("1. Location & Area")
    state_input = st.selectbox("Select State", sorted(list(state_coords.keys())))
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
    # Use correct column names from the cache file
    c1.metric("Temperature", f"{state_data['Avg_Temperature']:.1f}°C")
    c2.metric("Rainfall", f"{state_data['Annual_Rainfall']:.1f}mm")
    c3.metric("Humidity", f"{state_data['humidity_pct']:.0f}%")
    # Calculate VPD here as it might not be in the cache or needs recalculation
    current_vpd = calculate_vpd(state_data['Avg_Temperature'], state_data['humidity_pct'])
    c4.metric("VPD", f"{current_vpd:.2f} kPa")

    if predict_btn and assets and 'model' in assets:
        model = assets['model']
        target_le = assets.get('target_le')

        # --- 1. Get Model's Expected Columns ---
        model_cols = None
        try:
            if hasattr(model, 'feature_names_in_'):
                model_cols = model.feature_names_in_
            elif hasattr(model, 'get_booster'):
                 model_cols = model.get_booster().feature_names
        except Exception:
            pass

        if model_cols is None:
            # FALLBACK: Hardcoded list based on the provided notebook's training data.
            # This is necessary if the saved model object doesn't have feature names.
            model_cols = [
                'Area', 'N', 'P', 'K', 'pH', 'rain', 'temp', 'humidity', 'VPD',
                'Season_Kharif', 'Season_Rabi', 'Season_Summer', 'Season_Whole Year ', 'Season_Winter',
                'State_Arunachal Pradesh', 'State_Assam', 'State_Bihar', 'State_Chhattisgarh', 'State_Delhi',
                'State_Goa', 'State_Gujarat', 'State_Haryana', 'State_Himachal Pradesh', 'State_Jammu and Kashmir',
                'State_Jharkhand', 'State_Karnataka', 'State_Kerala', 'State_Madhya Pradesh', 'State_Maharashtra',
                'State_Manipur', 'State_Meghalaya', 'State_Mizoram', 'State_Nagaland', 'State_Odisha',
                'State_Puducherry', 'State_Punjab', 'State_Rajasthan', 'State_Sikkim', 'State_Tamil Nadu',
                'State_Telangana', 'State_Tripura', 'State_Uttar Pradesh', 'State_Uttarakhand', 'State_West Bengal'
            ]
            # st.warning("⚠️ Using hardcoded feature list as fallback.")

        # --- 2. Create a "Zero" DataFrame ---
        # Initialize a DataFrame with one row of zeros and the correct column names
        input_df = pd.DataFrame(np.zeros((1, len(model_cols))), columns=model_cols)

        # --- 3. Prepare Data Values ---
        current_month = datetime.now().strftime("%B")
        clean_season = get_season_clean(current_month)
        
        # Unit conversions from kg/ha to kg/sq ft
        n_sq = N_ha / HA_TO_SQFT
        p_sq = P_ha / HA_TO_SQFT
        k_sq = K_ha / HA_TO_SQFT

        # --- 4. Fill Numerical Features ---
        # Use the SHORT names from the notebook ('N', 'P', 'K', 'temp', etc.)
        def safe_set(col_name, value):
            if col_name in input_df.columns:
                input_df[col_name] = value
        
        safe_set('Area', area_sqft)
        safe_set('N', n_sq)
        safe_set('P', p_sq)
        safe_set('K', k_sq)
        safe_set('pH', ph)
        safe_set('rain', state_data['Annual_Rainfall'])
        safe_set('temp', state_data['Avg_Temperature'])
        safe_set('humidity', state_data['humidity_pct'])
        safe_set('VPD', current_vpd)

        # --- 5. Fill Categorical (One-Hot) Features ---
        # Find and set the correct Season column to 1
        clean_season_str = clean_season.strip().lower()
        season_found = False
        for col in model_cols:
            # Fuzzy match: check if column contains "season" and our season name
            if "season" in col.lower() and clean_season_str in col.lower():
                input_df[col] = 1
                season_found = True
                break
        if not season_found:
             st.warning(f"⚠️ Could not find a model column for season '{clean_season}'.")

        # Find and set the correct State column to 1
        clean_state_str = state_input.strip().lower()
        state_found = False
        for col in model_cols:
            # Fuzzy match: check if column contains "state" and our state name
            if "state" in col.lower() and clean_state_str in col.lower():
                input_df[col] = 1
                state_found = True
                break
        if not state_found:
             st.error(f"❌ Matching Error: The model does not have a column for '{state_input}'.")
             st.stop()

        # --- 6. Predict and Decode ---
        try:
            # Make prediction (returns an array with one numerical label)
            prediction_idx = model.predict(input_df)[0]
            
            # Convert numerical label back to crop name using the target encoder
            if target_le:
                prediction_name = target_le.inverse_transform([prediction_idx])[0]
                st.success(f"🌱 Recommended Crop: **{prediction_name}**")
            else:
                # Fallback if label encoder is missing
                st.success(f"🌱 Recommended Crop Index: **{prediction_idx}**")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction Failed: {e}")

elif weather_df is None:
    st.info("Please ensure 'daily_weather_cache.csv' is present in the Notebooks folder.")
