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
HA_TO_SQFT = 107639.0  # Conversion factor (Hectare to Sq Ft)

# --- STATE COORDINATES ---
# Ensure these names match EXACTLY with your training data (spelling/capitalization)
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
    "Kharif": ["July", "August"],        
    "Autumn": ["September", "October"], 
    "Rabi": ["November", "December"],    
    "Winter": ["January", "February"],    
    "Summer": ["March", "April", "May", "June"] 
}

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    """Loads the Model and Encoders from the Models folder."""
    # Handle different path structures (Cloud vs Local)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try finding the Models folder
    possible_dirs = [
        os.path.join(base_dir, 'Models'), # Nested in Notebooks/
        os.path.join(os.path.dirname(base_dir), 'Models'), # In Root
        'Models' # Relative
    ]
    
    model_path = None
    encoder_path = None

    for d in possible_dirs:
        m_p = os.path.join(d, 'crop_recomender.joblib')
        e_p = os.path.join(d, 'all_encoders.joblib')
        if os.path.exists(m_p): model_path = m_p
        if os.path.exists(e_p): encoder_path = e_p
    
    assets = {}
    
    # Load Model
    if model_path:
        assets['model'] = joblib.load(model_path)
    else:
        st.error("🚨 Critical Error: 'crop_recomender.joblib' not found in Models folder.")
        return None

    # Load Encoders
    if encoder_path:
        assets['encoders'] = joblib.load(encoder_path)
    else:
        st.error("🚨 Critical Error: 'all_encoders.joblib' not found in Models folder.")
        return None
            
    return assets

# --- HELPER FUNCTIONS ---
def get_season(month_name):
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
    """Fetches weather data for all states and saves to CSV."""
    with st.spinner("🔄 Updating climate database for today... (This happens once daily)"):
        end_year = datetime.now().year - 1
        start_year = end_year - 4
        month_num = datetime.now().month

        try:
            # Fetch IMD data
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
                # IMD Data Slicing
                m_rain = ds_rain.sel(time=ds_rain.time.dt.month == month_num).sel(lat=lat, lon=lon, method='nearest')
                m_tmax = ds_tmax.sel(time=ds_tmax.time.dt.month == month_num).sel(lat=lat, lon=lon, method='nearest')
                m_tmin = ds_tmin.sel(time=ds_tmin.time.dt.month == month_num).sel(lat=lat, lon=lon, method='nearest')

                # Calculate Weighted Temp
                yearly_means = []
                for y in range(start_year, end_year + 1):
                    val = (m_tmax.sel(time=m_tmax.time.dt.year == y).tmax.mean() + 
                           m_tmin.sel(time=m_tmin.time.dt.year == y).tmin.mean()) / 2
                    yearly_means.append(float(val))
                weighted_temp = np.dot(yearly_means, [0.05, 0.10, 0.15, 0.30, 0.40])

                # Calculate Rain Stats
                total_rain_5yr = float(m_rain.rain.sum())
                avg_monthly_rain = total_rain_5yr / 5
                rainy_days = (m_rain.rain > 2.5).sum().item() / 5

                # Humidity (Using default or API if needed)
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
                pass # Skip state if data error
            
            progress_bar.progress((i + 1) / total_states)

        # Save to Cache
        df_cache = pd.DataFrame(all_data)
        df_cache.to_csv(CACHE_FILE, index=False)
        progress_bar.empty()
        return df_cache

def get_weather_data():
    """Smart loader: Checks cache first."""
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE)
            # Check if cache is from today
            if pd.to_datetime(df['Date_Updated'].iloc[0]).date() == date.today():
                return df
        except:
            pass # Cache corrupted
    
    # If no cache or old cache, run batch fetch
    st.warning("⚠️ First run of the day: Fetching fresh climate data...")
    return fetch_all_states_batch()

# --- MAIN APP UI ---

# Load Assets
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
    
    # Help Guide
    with st.expander("ℹ️ How to find N, P, K values?"):
        st.markdown("""
        **Option 1: Search Online**
        * Look for values in **kg/hectare** (standard agricultural unit).
        """)

    # User inputs kg/ha
    N_ha = st.number_input("Nitrogen (N)", value=140, help="Enter value in kg/ha")
    P_ha = st.number_input("Phosphorus (P)", value=40, help="Enter value in kg/ha")
    K_ha = st.number_input("Potassium (K)", value=40, help="Enter value in kg/ha")
    ph = st.slider("Soil pH", 4.0, 9.5, 6.5)

    predict_btn = st.button("Get Recommendation", type="primary")

# --- MAIN DASHBOARD LOGIC ---

if weather_df is not None and state_input in weather_df['State'].values:
    # 1. Get Climate Data
    state_data = weather_df[weather_df['State'] == state_input].iloc[0]
    
    # 2. Display Metrics
    st.subheader(f"Current Climate Analysis: {state_input}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{state_data['Avg_Temperature']:.1f}°C")
    c2.metric("Rainfall", f"{state_data['Annual_Rainfall']:.1f}mm")
    c3.metric("Humidity", f"{state_data['humidity_pct']:.0f}%")
    c4.metric("VPD", f"{state_data['VPD']:.2f} kPa", delta="High Stress" if state_data['VPD'] > 1.2 else "Normal", delta_color="inverse")

    # 3. PREDICTION PIPELINE
    if predict_btn and assets:
        model = assets['model']
        encoders = assets['encoders'] # Expected format: {'Season': LeObject, 'State': LeObject}

        current_month = datetime.now().strftime("%B")
        current_season = get_season(current_month)
        
        # A. Unit Conversion (kg/ha -> kg/sq_ft)
        n_sq = N_ha / HA_TO_SQFT
        p_sq = P_ha / HA_TO_SQFT
        k_sq = K_ha / HA_TO_SQFT
        
        try:
            # B. Encode Categorical Columns
            # We use the loaded encoders to transform user input into the number the model expects
            
            # Check 1: Does the encoder exist?
            if 'State' not in encoders or 'Season' not in encoders:
                st.error("The loaded 'all_encoders.joblib' file does not contain 'State' or 'Season' keys.")
                st.stop()
                
            # Check 2: Transform
            # .transform() expects a list, e.g. ['Punjab']
            state_encoded = encoders['State'].transform([state_input])[0]
            season_encoded = encoders['Season'].transform([current_season])[0]
            
            # C. Create DataFrame
            # MUST match the exact column order used during training
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
            
            # D. Predict
            prediction = model.predict(input_df)
            
            # E. Output
            st.success(f"🌱 Recommended Crop: **{prediction[0]}**")
            st.balloons()

        except ValueError as e:
            st.error(f"Encoding Error: The state '{state_input}' was not found in the training data. Please check spelling.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("Loading weather data...")
