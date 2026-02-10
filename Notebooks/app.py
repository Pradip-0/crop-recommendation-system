import streamlit as st
import joblib
import os
import requests
import numpy as np
import pandas as pd
import imdlib as imd
from datetime import datetime, date

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AgriSmart Advisor", page_icon="🌱", layout="wide")

# --- CONSTANTS ---
CACHE_FILE = "daily_weather_cache.csv"
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
def load_model():
    """Loads only the Model (Encoders are not needed for One-Hot logic)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_dirs = [
        os.path.join(base_dir, 'Models'), 
        os.path.join(os.path.dirname(base_dir), 'Models'), 
        'Models' 
    ]
    
    model_path = None

    for d in possible_dirs:
        m_p = os.path.join(d, 'crop_recomender.joblib')
        if os.path.exists(m_p): 
            model_path = m_p
            break
    
    if model_path:
        return joblib.load(model_path)
    else:
        st.error("🚨 Critical Error: 'crop_recomender.joblib' not found.")
        return None

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

# --- WEATHER DATA FETCHING ---
def fetch_all_states_batch():
    with st.spinner("🔄 Updating climate database for today..."):
        end_year = datetime.now().year - 1
        start_year = end_year - 4
        month_num = datetime.now().month

        try:
            rain_data = imd.get_data('rain', start_year, end_year, fn_format='yearwise')
            tmax_data = imd.get_data('tmax', start_year, end_year, fn_format='yearwise')
            tmin_data = imd.get_data('tmin', start_year, end_year, fn_format='yearwise')
            
            ds_rain = rain_data.get_xarray()
            ds_tmax = tmax_data.get_xarray()
            ds_tmin = tmin_data.get_xarray()
        except Exception as e:
            st.error(f"Failed to fetch IMD data: {e}")
            return None

        all_data = []
        progress_bar = st.progress(0)
        total_states = len(state_coords)
        
        for i, (state, (lat, lon)) in enumerate(state_coords.items()):
            try:
                m_rain = ds_rain.sel(time=ds_rain.time.dt.month == month_num).sel(lat=lat, lon=lon, method='nearest')
                m_tmax = ds_tmax.sel(time=ds_tmax.time.dt.month == month_num).sel(lat=lat, lon=lon, method='nearest')
                m_tmin = ds_tmin.sel(time=ds_tmin.time.dt.month == month_num).sel(lat=lat, lon=lon, method='nearest')

                yearly_means = []
                for y in range(start_year, end_year + 1):
                    val = (m_tmax.sel(time=m_tmax.time.dt.year == y).tmax.mean() + 
                           m_tmin.sel(time=m_tmin.time.dt.year == y).tmin.mean()) / 2
                    yearly_means.append(float(val))
                weighted_temp = np.dot(yearly_means, [0.05, 0.10, 0.15, 0.30, 0.40])

                total_rain_5yr = float(m_rain.rain.sum())
                avg_monthly_rain = total_rain_5yr / 5
                rainy_days = (m_rain.rain > 2.5).sum().item() / 5

                avg_humidity = 60.0 
                vpd = calculate_vpd(weighted_temp, avg_humidity)

                all_data.append({
                    "State": state,
                    "Avg_Temperature": weighted_temp,
                    "Annual_Rainfall": avg_monthly_rain,
                    "humidity_pct": avg_humidity,
                    "Rainy_Days": rainy_days,
                    "VPD": vpd,
                    "Date_Updated": date.today()
                })
            except:
                pass
            progress_bar.progress((i + 1) / total_states)

        df_cache = pd.DataFrame(all_data)
        df_cache.to_csv(CACHE_FILE, index=False)
        progress_bar.empty()
        return df_cache

def get_weather_data():
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE)
            if pd.to_datetime(df['Date_Updated'].iloc[0]).date() == date.today():
                return df
        except:
            pass
    st.warning("⚠️ First run of the day: Fetching fresh climate data...")
    return fetch_all_states_batch()

# --- MAIN APP UI ---

model = load_model()
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
    c4.metric("VPD", f"{state_data['VPD']:.2f} kPa")

    if predict_btn and model:
        # --- 1. Get Model's Expected Columns ---
        # The model knows what 44 columns it was trained on.
        try:
            if hasattr(model, 'feature_names_in_'):
                model_cols = model.feature_names_in_
            else:
                model_cols = model.get_booster().feature_names
        except:
            st.error("🚨 Critical Error: Could not retrieve feature names from the model.")
            st.stop()

        # --- 2. Create a "Zero" DataFrame ---
        # Create a single row with all 44 columns initialized to 0
        input_df = pd.DataFrame(0, index=[0], columns=model_cols)

        # --- 3. Fill Numerical Values ---
        current_month = datetime.now().strftime("%B")
        clean_season = get_season_clean(current_month)
        
        # Conversions
        n_sq = N_ha / HA_TO_SQFT
        p_sq = P_ha / HA_TO_SQFT
        k_sq = K_ha / HA_TO_SQFT

        # Helper to safely set column if it exists
        def safe_set(col_name, value):
            if col_name in input_df.columns:
                input_df[col_name] = value

        # Fill known numerical columns
        safe_set('Area', area_sqft)
        safe_set('Annual_Rainfall', state_data['Annual_Rainfall'])
        safe_set('Avg_Temperature', state_data['Avg_Temperature'])
        safe_set('humidity_pct', state_data['humidity_pct'])
        safe_set('soil_N_kg_sq_foot', n_sq)
        safe_set('soil_P_kg_sq_foot', p_sq)
        safe_set('soil_K_kg_sq_foot', k_sq)
        safe_set('soil_pH', ph)

        # --- 4. Fill Categorical (One-Hot) Values ---
        
        # A. Handle Season (Fuzzy Match to handle spaces)
        clean_season_str = clean_season.strip().lower()
        season_found = False
        
        for col in model_cols:
            # Check if column is a Season column AND matches our season
            if "season" in col.lower() and clean_season_str in col.lower():
                input_df[col] = 1
                season_found = True
                break 
        
        if not season_found:
             st.warning(f"⚠️ Warning: Could not find a model column for season '{clean_season}'.")

        # B. Handle State (Fuzzy Match)
        clean_state_str = state_input.strip().lower()
        state_found = False
        
        for col in model_cols:
            if "state" in col.lower() and clean_state_str in col.lower():
                input_df[col] = 1
                state_found = True
                break
                
        if not state_found:
             st.error(f"❌ Matching Error: The model does not have a column for '{state_input}'.")
             st.write("Debug: Model expects these state columns:", [c for c in model_cols if 'state' in c.lower()])
             st.stop()

        # --- 5. Predict ---
        try:
            prediction = model.predict(input_df)
            st.success(f"🌱 Recommended Crop: **{prediction[0]}**")
            st.balloons()
        except Exception as e:
            st.error(f"Prediction Failed: {e}")
            st.write("Debug - Input Shape:", input_df.shape)

else:
    st.info("Loading weather data...")
