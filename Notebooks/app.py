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
# We use clean names here. The code later adds the spaces to match your encoder.
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_dirs = [
        os.path.join(base_dir, 'Models'), 
        os.path.join(os.path.dirname(base_dir), 'Models'), 
        'Models' 
    ]
    
    model_path = None
    encoder_path = None

    for d in possible_dirs:
        m_p = os.path.join(d, 'crop_recomender.joblib')
        e_p = os.path.join(d, 'all_encoders.joblib')
        if os.path.exists(m_p): model_path = m_p
        if os.path.exists(e_p): encoder_path = e_p
    
    assets = {}
    
    if model_path:
        assets['model'] = joblib.load(model_path)
    else:
        st.error("🚨 Critical Error: 'crop_recomender.joblib' not found.")
        return None

    if encoder_path:
        assets['encoders'] = joblib.load(encoder_path)
    else:
        st.error("🚨 Critical Error: 'all_encoders.joblib' not found.")
        return None
            
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
    c4.metric("VPD", f"{state_data['VPD']:.2f} kPa")

    if predict_btn and assets:
        model = assets['model']
        encoders = assets['encoders']

        current_month = datetime.now().strftime("%B")
        
        # 1. Get Clean Season Name
        clean_season = get_season_clean(current_month)
        
        n_sq = N_ha / HA_TO_SQFT
        p_sq = P_ha / HA_TO_SQFT
        k_sq = K_ha / HA_TO_SQFT
        
        try:
            # --- AUTO-SPACER LOGIC (Season) ---
            valid_seasons = encoders['Season'].classes_
            
            # Create map: {'winter': 'Winter     ', 'kharif': 'Kharif     '}
            season_map = {s.strip().lower(): s for s in valid_seasons}
            
            # Convert our clean input to the spaced version
            clean_key = clean_season.strip().lower()
            
            if clean_key in season_map:
                exact_season_label = season_map[clean_key]
                season_encoded = encoders['Season'].transform([exact_season_label])[0]
            else:
                st.error(f"❌ Season Error: The season '{clean_season}' is not in your training data.")
                st.write("Available seasons:", valid_seasons)
                st.stop()

            # --- AUTO-SPACER LOGIC (State) ---
            valid_states = encoders['State'].classes_
            state_map = {s.strip().lower(): s for s in valid_states}
            
            if state_input.strip().lower() in state_map:
                exact_state_label = state_map[state_input.strip().lower()]
                state_encoded = encoders['State'].transform([exact_state_label])[0]
            else:
                st.error(f"❌ State Error: '{state_input}' not found in training data.")
                st.write("Available states:", valid_states)
                st.stop()

            # Create DataFrame
            input_df = pd.DataFrame({
                'Season': [season_encoded],
                'State': [state_encoded],
                'Area': [area_sqft],
                'Annual_Rainfall': [state_data['Annual_Rainfall']],
                'Avg_Temperature': [state_data['Avg_Temperature']],
                'humidity_pct': [state_data['humidity_pct']],
                'soil_N_kg_sq_foot': [n_sq],
                'soil_P_kg_sq_foot': [p_sq],
                'soil_K_kg_sq_foot': [k_sq],
                'soil_pH': [ph]
            })
            
            prediction = model.predict(input_df)
            st.success(f"🌱 Recommended Crop: **{prediction[0]}**")
            st.balloons()

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("Loading weather data...")
