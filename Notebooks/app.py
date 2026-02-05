import streamlit as st
import joblib
import os
import requests
import numpy as np
from datetime import datetime
import imdlib as imd

st.set_page_config(page_title="Crop Advisor", page_icon="🌱")
st.title("Climate-Aware Crop Recommendation")

# --- MODEL LOADING ---
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
    return None

model = load_model()

# --- SEASON LOGIC ---
current_month = datetime.now().strftime("%B")
season_months = {
    "Kharif": ["July", "August"],        
    "Autumn": ["September", "October"], 
    "Rabi": ["November", "December"],   
    "Winter": ["January", "February"],   
    "Summer": ["March", "April", "May", "June"] 
}

def get_season(month_name, mapping):
    for season, months in mapping.items():
        if month_name in months:
            return season
    return "Unknown"

# --- CLIMATE CALCULATION FUNCTIONS ---

def calculate_vpd(temp_c, rh_pct):
    """Calculates Vapor Pressure Deficit (kPa)"""
    svp = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    avp = svp * (rh_pct / 100)
    return svp - avp

@st.cache_data
def get_climate_metrics(lat, lon, target_month_name):
    # 1. Setup Timeframe (Last 5 Years)
    end_year = datetime.now().year - 1
    start_year = end_year - 4
    month_num = datetime.strptime(target_month_name, "%B").month

    # 2. Fetch IMD Data (Rain & Temp)
    # Using 'yearwise' to get gridded data
    rain_data = imd.get_data('rain', start_year, end_year, fn_format='yearwise')
    tmax_data = imd.get_data('tmax', start_year, end_year, fn_format='yearwise')
    tmin_data = imd.get_data('tmin', start_year, end_year, fn_format='yearwise')

    ds_rain = rain_data.get_xarray()
    ds_tmax = tmax_data.get_xarray()
    ds_tmin = tmin_data.get_xarray()

    # Filter for the specific month across all 5 years
    m_rain = ds_rain.sel(time=ds_rain.time.dt.month == month_num).sel(lat=lat, lon=lon, method='nearest')
    m_tmax = ds_tmax.sel(time=ds_tmax.time.dt.month == month_num).sel(lat=lat, lon=lon, method='nearest')
    m_tmin = ds_tmin.sel(time=ds_tmin.time.dt.month == month_num).sel(lat=lat, lon=lon, method='nearest')

    # Logic 1: Weighted Temperature (Last 2 years get 70% weight)
    yearly_means = []
    for y in range(start_year, end_year + 1):
        y_t = (m_tmax.sel(time=m_tmax.time.dt.year == y).tmax.mean() + 
               m_tmin.sel(time=m_tmin.time.dt.year == y).tmin.mean()) / 2
        yearly_means.append(float(y_t))
    
    weights = [0.05, 0.10, 0.15, 0.30, 0.40] # Weighted towards recent years
    weighted_temp = np.dot(yearly_means, weights)

    # Logic 2: Rainfall Reliability (Daily > 2.5mm)
    total_rain_5yr = float(m_rain.rain.sum())
    avg_monthly_rain = total_rain_5yr / 5
    # Counting consistent rainy days
    rainy_days_avg = (m_rain.rain > 2.5).sum().item() / 5

    # 3. Fetch Humidity via Open-Meteo API
    yearly_rh = []
    for year in range(start_year, end_year + 1):
        url = "https://archive-api.open-meteo.com"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{year}-{month_num:02d}-01",
            "end_date": f"{year}-{month_num:02d}-28",
            "hourly": "relative_humidity_2m"
        }
        res = requests.get(url, params=params).json()
        if "hourly" in res:
            yearly_rh.append(np.mean(res["hourly"]["relative_humidity_2m"]))
    
    avg_humidity = np.mean(yearly_rh) if yearly_rh else 50.0
    vpd = calculate_vpd(weighted_temp, avg_humidity)

    return weighted_temp, avg_monthly_rain, avg_humidity, rainy_days_avg, vpd

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


# --- SIDEBAR UI ---
with st.sidebar:
    st.header("Input Parameters")
    
    # 1. Version Selection
    choice = st.radio("Preferred version", ["Most preferred crop", "Top 5 crop"])
    
    # 2. State & Area Selection
    state_input = st.selectbox("Choose your state", list(state_coords.keys()))
    area = st.number_input("Enter area in square foot", min_value=0, value=1000)
    
    st.divider()
    st.subheader("🧪 Soil Nutrients (Manual)")
    # Manual NPK inputs as requested
    N = st.number_input("Nitrogen (N)", min_value=0, value=70)
    P = st.number_input("Phosphorus (P)", min_value=0, value=40)
    K = st.number_input("Potassium (K)", min_value=0, value=40)
#    ph = st.slider("Soil pH Level", 4.0, 9.5, 6.5)


# --- MAIN LOGIC ---
if state_input in state_coords:
    lat, lon = state_coords[state_input]
    
    with st.spinner(f"Analyzing climate trends for {state_input}..."):
        temp, rain, humidity, r_days, vpd = get_climate_metrics(lat, lon, current_month)

    # UI Display of Climate Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Weighted Temp", f"{temp:.1f}°C")
    col2.metric("Avg Rainfall", f"{rain:.1f}mm")
    col3.metric("Humidity", f"{humidity:.1f}%")

    st.info(f"**Season:** {get_season(current_month, season_months)} | **Rainy Days:** ~{r_days:.1f} days/month")
    
    if vpd > 1.2:
        st.warning(f"High Vapor Pressure Deficit ({vpd:.2f} kPa). High plant water stress detected.")





