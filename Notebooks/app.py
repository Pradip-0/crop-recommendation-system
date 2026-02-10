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
root_dir = os.path.dirname(current_dir) # Go up one level
CACHE_FILE_PATH = os.path.join(root_dir, "daily_weather_cache.csv")
MODELS_DIR = os.path.join(root_dir, "Models")

HA_TO_SQFT = 107639.0 

# --- DATA DEFINITIONS ---

# 1. STATES (Sorted Alphabetically - Critical for One-Hot Encoding order)
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

# 2. SEASONS (Sorted Alphabetically - Critical for One-Hot Encoding order)
season_months = {
    "Autumn": [], 
    "Kharif": ["July", "August", "September", "October"], 
    "Rabi": ["November", "December"],
    "Summer": ["March", "April", "May", "June"],
    "Whole Year": [],
    "Winter": ["January", "February"]
}

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    model_path = os.path.join(MODELS_DIR, 'crop_recomender.joblib')
    le_path = os.path.join(MODELS_DIR, 'label_encoder.joblib') 

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

    if os.path.exists(le_path):
        try:
            assets['target_le'] = joblib.load(le_path)
        except:
            pass
    
    return assets

# --- HELPER FUNCTIONS ---
def get_season_clean(month_name):
    for season, months in season_months.items():
        if month_name in months:
            return season
    return "Summer"

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
    Manually constructs the 44 feature names in the exact order 
    Pandas get_dummies() would produce.
    Order: Numerical Columns + Sorted Categorical Columns
    """
    # 1. Numerical Columns (8 cols)
    # MUST match the user provided list order or typical dataframe order
    num_cols = [
        'Area', 'Annual_Rainfall', 'Avg_Temperature', 'humidity_pct', 
        'soil_N_kg_sq_foot', 'soil_P_kg_sq_foot', 'soil_K_kg_sq_foot', 'soil_pH'
    ]
    
    # 2. Season Columns (Sorted Alphabetically)
    # Expected: Season_Autumn, Season_Kharif, Season_Rabi...
    season_cols = [f"Season_{s}" for s in sorted(season_months.keys())]
    
    # 3. State Columns (Sorted Alphabetically)
    # Expected: State_Andhra Pradesh, State_Arunachal Pradesh...
    state_cols = [f"State_{s}" for s in sorted(state_coords.keys())]
    
    # Combined List (44 cols total)
    return num_cols + season_cols + state_cols

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
    c1.metric("Temperature", f"{state_data['Avg_Temperature']:.1f}°C")
    c2.metric("Rainfall", f"{state_data['Annual_Rainfall']:.1f}mm")
    c3.metric("Humidity", f"{state_data['humidity_pct']:.0f}%")
    
    current_vpd = calculate_vpd(state_data['Avg_Temperature'], state_data['humidity_pct'])
    c4.metric("VPD", f"{current_vpd:.2f} kPa")

    if predict_btn and assets and 'model' in assets:
        model = assets['model']
        target_le = assets.get('target_le')

        # --- 1. Get Features (MANUAL MODE) ---
        # We assume the user's list implies these 44 columns
        model_cols = generate_manual_features()

        # --- 2. Create Data Frame ---
        # Explicitly create zero-filled array first
        data_array = np.zeros((1, len(model_cols)))
        input_df = pd.DataFrame(data_array, columns=model_cols)

        # --- 3. Prepare Values ---
        current_month = datetime.now().strftime("%B")
        clean_season = get_season_clean(current_month)
        
        # Conversions
        n_sq = N_ha / HA_TO_SQFT
        p_sq = P_ha / HA_TO_SQFT
        k_sq = K_ha / HA_TO_SQFT

        # --- 4. Fill Numerical Columns ---
        # We fill the 8 numerical columns we know exist
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
        # Logic: Find column named "Season_Winter" and set to 1
        clean_season_str = clean_season.strip() 
        season_col_name = f"Season_{clean_season_str}"
        
        # Fuzzy match to handle "Whole Year " spaces if they exist in column names
        season_set = False
        for col in input_df.columns:
            if col.strip() == season_col_name:
                input_df[col] = 1
                season_set = True
                break
        
        if not season_set:
            # Try finding it with fuzzy matching (ignoring case/spaces)
            for col in input_df.columns:
                if f"season_{clean_season_str.lower()}" in col.lower():
                    input_df[col] = 1
                    season_set = True
                    break
        
        if not season_set:
            st.warning(f"⚠️ Warning: Season '{clean_season}' not found in model columns. Using default.")

        # B. Fill State
        state_col_name = f"State_{state_input}"
        state_set = False
        for col in input_df.columns:
            if col == state_col_name:
                input_df[col] = 1
                state_set = True
                break
        
        if not state_set:
             st.error(f"❌ Error: Model missing column for '{state_input}'.")
             st.stop()

        # --- 6. Predict ---
        try:
            prediction_idx = model.predict(input_df)[0]
            
            if target_le:
                try:
                    prediction_name = target_le.inverse_transform([prediction_idx])[0]
                    st.success(f"🌱 Recommended Crop: **{prediction_name}**")
                except:
                    st.success(f"🌱 Recommended Crop Index: **{prediction_idx}**")
            else:
                st.success(f"🌱 Recommended Crop Index: **{prediction_idx}**")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.write("Debug: Model expects", len(model_cols), "columns.")
            st.write("First 10 columns generated:", model_cols[:10])

elif weather_df is None:
    st.info("Loading weather data... (If this takes too long, check if daily_weather_cache.csv exists in the root folder)")
