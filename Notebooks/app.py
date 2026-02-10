import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import imdlib as imd
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AgriSmart Advisor", 
    page_icon="🌱", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .success-box h2 {
        margin: 0 0 1rem 0;
        font-size: 2rem;
    }
    
    .success-box p {
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
    
    /* Info message styling */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
CACHE_FILE = "daily_weather_cache.csv"
HA_TO_SQFT = 107639.1  # Conversion factor (hectare to sq ft)

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

# --- CROP INFORMATION ---
crop_info = {
    'Onion': {'icon': '🧅', 'season': 'Rabi', 'days': '100-120'},
    'Potato': {'icon': '🥔', 'season': 'Rabi', 'days': '90-120'},
    'Sweet potato': {'icon': '🍠', 'season': 'Kharif', 'days': '120-150'},
    'Garlic': {'icon': '🧄', 'season': 'Rabi', 'days': '150-180'},
    'Tapioca': {'icon': '🌾', 'season': 'Kharif', 'days': '240-300'},
    'Dry chillies': {'icon': '🌶️', 'season': 'Kharif', 'days': '120-150'},
    'Ginger': {'icon': '🫚', 'season': 'Kharif', 'days': '240-270'},
    'Turmeric': {'icon': '🌿', 'season': 'Kharif', 'days': '240-300'},
    'Coriander': {'icon': '🌱', 'season': 'Rabi', 'days': '40-50'},
    'Black pepper': {'icon': '⚫', 'season': 'Whole Year', 'days': '365+'},
    'Cardamom': {'icon': '🌿', 'season': 'Whole Year', 'days': '365+'},
    'Gram': {'icon': '🫘', 'season': 'Rabi', 'days': '100-120'},
    'Moong(Green Gram)': {'icon': '💚', 'season': 'Kharif', 'days': '60-90'},
    'Urad': {'icon': '⚫', 'season': 'Kharif', 'days': '75-90'},
    'Cowpea(Lobia)': {'icon': '🫘', 'season': 'Kharif', 'days': '70-90'},
    'Masoor': {'icon': '🔴', 'season': 'Rabi', 'days': '110-130'},
    'Peas & beans (Pulses)': {'icon': '🫛', 'season': 'Rabi', 'days': '60-90'},
    'Arhar/Tur': {'icon': '🟡', 'season': 'Kharif', 'days': '180-200'},
    'Banana': {'icon': '🍌', 'season': 'Whole Year', 'days': '365+'},
    'Coconut ': {'icon': '🥥', 'season': 'Whole Year', 'days': '365+'},
    'Cashewnut': {'icon': '🌰', 'season': 'Whole Year', 'days': '365+'},
    'Groundnut': {'icon': '🥜', 'season': 'Kharif', 'days': '100-120'}
}

# --- ASSET LOADING ---
@st.cache_resource
def load_model():
    """Loads the trained model"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_dirs = [
        os.path.join(base_dir, 'Models'), 
        os.path.join(os.path.dirname(base_dir), 'Models'), 
        'Models',
        base_dir  # Check current directory too
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
        st.error("🚨 Critical Error: 'crop_recomender.joblib' not found in any expected directory.")
        st.info("📁 Please ensure the model file is in a 'Models' folder or the same directory as app.py")
        return None

@st.cache_resource
def load_training_data():
    """Loads the training data to get feature names"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_files = [
        os.path.join(base_dir, 'Crop_data.csv'),
        os.path.join(base_dir, 'Models', 'Crop_data.csv'),
        'Crop_data.csv'
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
    
    st.warning("⚠️ Training data file 'Crop_data.csv' not found. Using model's feature names.")
    return None

# --- HELPER FUNCTIONS ---
def get_season_clean(month_name):
    """Returns the season for a given month"""
    for season, months in season_months.items():
        if month_name in months:
            return season
    return "Summer"

def calculate_vpd(temp_c, rh_pct):
    """Calculate Vapor Pressure Deficit"""
    svp = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    avp = svp * (rh_pct / 100)
    return max(0, svp - avp)

# --- WEATHER DATA FETCHING ---
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_all_states_batch():
    """Fetch weather data for all states using IMD data"""
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
            st.error(f"❌ Failed to fetch IMD data: {e}")
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
            except Exception as e:
                st.warning(f"⚠️ Could not fetch data for {state}: {str(e)}")
                pass
            
            progress_bar.progress((i + 1) / total_states)

        df_cache = pd.DataFrame(all_data)
        df_cache.to_csv(CACHE_FILE, index=False)
        progress_bar.empty()
        return df_cache

def get_weather_data():
    """Get weather data from cache or fetch new data"""
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE)
            if pd.to_datetime(df['Date_Updated'].iloc[0]).date() == date.today():
                return df
        except:
            pass
    
    st.info("ℹ️ First run of the day: Fetching fresh climate data. This may take a few minutes...")
    return fetch_all_states_batch()

def prepare_input_features(state_input, area_sqft, N_ha, P_ha, K_ha, ph, weather_data, model, training_data):
    """
    Prepare input features matching the model's expected format
    """
    # Get state weather data
    if state_input not in weather_data['State'].values:
        raise ValueError(f"Weather data not available for {state_input}")
    
    state_data = weather_data[weather_data['State'] == state_input].iloc[0]
    
    # Get current season
    current_month = datetime.now().strftime("%B")
    clean_season = get_season_clean(current_month)
    
    # Convert NPK from kg/hectare to kg/sq_foot
    n_sq = N_ha / HA_TO_SQFT
    p_sq = P_ha / HA_TO_SQFT
    k_sq = K_ha / HA_TO_SQFT
    
    # Create base input data
    input_data = {
        'Area': area_sqft,
        'Annual_Rainfall': state_data['Annual_Rainfall'],
        'Avg_Temperature': state_data['Avg_Temperature'],
        'humidity_pct': state_data['humidity_pct'],
        'soil_N_kg_sq_foot': n_sq,
        'soil_P_kg_sq_foot': p_sq,
        'soil_K_kg_sq_foot': k_sq,
        'soil_pH': ph,
        'Season': clean_season,
        'State': state_input
    }
    
    # Get model's expected features
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    elif hasattr(model, 'get_booster'):
        try:
            expected_features = model.get_booster().feature_names
        except:
            expected_features = None
    else:
        expected_features = None
    
    # If we have training data, use it to get the correct feature format
    if training_data is not None:
        # Create DataFrame with categorical and numerical columns
        input_df = pd.DataFrame([input_data])
        
        # Get categorical columns from training data
        cat_cols = training_data.select_dtypes(include=['object']).columns.tolist()
        if 'Crop' in cat_cols:
            cat_cols.remove('Crop')
        
        # One-hot encode categorical variables
        input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=False)
        
        # Get all columns from training data (excluding target)
        train_cols = [col for col in training_data.columns if col != 'Crop']
        train_df = training_data[train_cols]
        train_encoded = pd.get_dummies(train_df, columns=cat_cols, drop_first=False)
        
        # Ensure input has all columns from training data
        for col in train_encoded.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[train_encoded.columns]
        
    elif expected_features is not None:
        # Use model's feature names directly
        input_df = pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)
        
        # Set numerical values
        for key, value in input_data.items():
            if key in input_df.columns:
                input_df[key] = value
        
        # Set one-hot encoded categorical values
        season_col = f'Season_{clean_season}'
        state_col = f'State_{state_input}'
        
        if season_col in input_df.columns:
            input_df[season_col] = 1
        if state_col in input_df.columns:
            input_df[state_col] = 1
    else:
        raise ValueError("Cannot determine model's expected features")
    
    return input_df

# --- MAIN APP UI ---
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🌾 AgriSmart Kitchen Garden Advisor</h1>
            <p>Your AI-powered guide for sustainable home farming</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model = load_model()
    training_data = load_training_data()
    weather_df = get_weather_data()
    
    if model is None:
        st.stop()
    
    # Sidebar - Input Section
    with st.sidebar:
        st.markdown("### 📍 Location & Garden Details")
        
        col1, col2 = st.columns(2)
        with col1:
            state_input = st.selectbox(
                "Select Your State",
                sorted(list(state_coords.keys())),
                help="Choose your state for location-based recommendations"
            )
        
        with col2:
            area_sqft = st.number_input(
                "Garden Area (sq ft)",
                min_value=10,
                max_value=10000,
                value=100,
                step=10,
                help="Enter your garden area in square feet"
            )
        
        st.markdown("---")
        st.markdown("### 🧪 Soil Nutrient Levels")
        
        with st.expander("ℹ️ How to measure soil nutrients?", expanded=False):
            st.markdown("""
            **Quick Guide:**
            - Get a soil test kit from a local agricultural store
            - Send samples to a soil testing lab
            - Use NPK values in **kg/hectare** (we'll convert them)
            - Typical ranges:
              - **N (Nitrogen):** 100-200 kg/ha
              - **P (Phosphorus):** 30-60 kg/ha
              - **K (Potassium):** 30-60 kg/ha
              - **pH:** 6.0-7.5 (neutral)
            """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            N_ha = st.number_input(
                "N (kg/ha)",
                min_value=0,
                max_value=500,
                value=140,
                help="Nitrogen content"
            )
        with col2:
            P_ha = st.number_input(
                "P (kg/ha)",
                min_value=0,
                max_value=200,
                value=40,
                help="Phosphorus content"
            )
        with col3:
            K_ha = st.number_input(
                "K (kg/ha)",
                min_value=0,
                max_value=200,
                value=40,
                help="Potassium content"
            )
        
        ph = st.slider(
            "Soil pH Level",
            min_value=4.0,
            max_value=9.5,
            value=6.5,
            step=0.1,
            help="7.0 is neutral, <7 is acidic, >7 is alkaline"
        )
        
        st.markdown("---")
        predict_btn = st.button("🔍 Get Crop Recommendation", type="primary", use_container_width=True)
    
    # Main Content Area
    if weather_df is not None and state_input in weather_df['State'].values:
        state_data = weather_df[weather_df['State'] == state_input].iloc[0]
        
        # Current Climate Analysis
        st.markdown("### 🌤️ Current Climate Analysis")
        st.markdown(f"**Location:** {state_input} | **Month:** {datetime.now().strftime('%B %Y')}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🌡️ Temperature",
                value=f"{state_data['Avg_Temperature']:.1f}°C",
                help="Average temperature for this month"
            )
        
        with col2:
            st.metric(
                label="🌧️ Rainfall",
                value=f"{state_data['Annual_Rainfall']:.1f}mm",
                help="Average monthly rainfall"
            )
        
        with col3:
            st.metric(
                label="💧 Humidity",
                value=f"{state_data['humidity_pct']:.0f}%",
                help="Relative humidity"
            )
        
        with col4:
            st.metric(
                label="📊 VPD",
                value=f"{state_data['VPD']:.2f} kPa",
                help="Vapor Pressure Deficit"
            )
        
        st.markdown("---")
        
        # Prediction Logic
        if predict_btn:
            try:
                with st.spinner("🔄 Analyzing soil and climate data..."):
                    # Prepare input features
                    input_df = prepare_input_features(
                        state_input, area_sqft, N_ha, P_ha, K_ha, ph,
                        weather_df, model, training_data
                    )
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    # Get prediction probabilities if available
                    try:
                        probabilities = model.predict_proba(input_df)[0]
                        confidence = max(probabilities) * 100
                    except:
                        confidence = None
                    
                    # Display result
                    crop_icon = crop_info.get(prediction, {}).get('icon', '🌱')
                    crop_season = crop_info.get(prediction, {}).get('season', 'N/A')
                    crop_days = crop_info.get(prediction, {}).get('days', 'N/A')
                    
                    st.markdown(f"""
                        <div class="success-box">
                            <h2>{crop_icon} Recommended Crop: {prediction}</h2>
                            <p><strong>Best Season:</strong> {crop_season}</p>
                            <p><strong>Growing Period:</strong> {crop_days} days</p>
                            {f'<p><strong>Confidence:</strong> {confidence:.1f}%</p>' if confidence else ''}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional Information
                    st.markdown("### 📋 Cultivation Tips")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div class="info-card">
                            <h4>🌱 Soil Preparation</h4>
                            <ul>
                                <li>Ensure proper drainage</li>
                                <li>Add organic compost</li>
                                <li>Maintain optimal pH level</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="info-card">
                            <h4>💧 Watering Schedule</h4>
                            <ul>
                                <li>Water in early morning or evening</li>
                                <li>Avoid waterlogging</li>
                                <li>Adjust based on rainfall</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Nutrient Analysis
                    st.markdown("### 🧪 Your Soil Nutrient Status")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        n_status = "✅ Good" if 100 <= N_ha <= 200 else "⚠️ Adjust"
                        st.metric("Nitrogen (N)", f"{N_ha} kg/ha", n_status)
                    
                    with col2:
                        p_status = "✅ Good" if 30 <= P_ha <= 60 else "⚠️ Adjust"
                        st.metric("Phosphorus (P)", f"{P_ha} kg/ha", p_status)
                    
                    with col3:
                        k_status = "✅ Good" if 30 <= K_ha <= 60 else "⚠️ Adjust"
                        st.metric("Potassium (K)", f"{K_ha} kg/ha", k_status)
                    
                    with col4:
                        ph_status = "✅ Good" if 6.0 <= ph <= 7.5 else "⚠️ Adjust"
                        st.metric("pH Level", f"{ph}", ph_status)
                    
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("💡 Please check your input values and try again.")
                
                # Debug information (optional - can be removed in production)
                with st.expander("🔧 Debug Information"):
                    st.write("Error details:", str(e))
                    st.write("Input data:", {
                        'State': state_input,
                        'Area': area_sqft,
                        'N': N_ha,
                        'P': P_ha,
                        'K': K_ha,
                        'pH': ph
                    })
        
        else:
            # Initial state - show helpful information
            st.markdown("### 🌾 Welcome to AgriSmart Advisor!")
            
            st.info("""
            👈 **Get Started:**
            1. Select your state from the sidebar
            2. Enter your garden area
            3. Input soil nutrient values (N, P, K, pH)
            4. Click **Get Crop Recommendation**
            
            Our AI will analyze your soil and climate data to recommend the best crop for your kitchen garden!
            """)
            
            # Show sample crops
            st.markdown("### 🌱 Crops We Can Recommend")
            
            crop_cols = st.columns(5)
            crop_list = list(crop_info.keys())
            
            for i, crop in enumerate(crop_list[:20]):  # Show first 20 crops
                col_idx = i % 5
                with crop_cols[col_idx]:
                    icon = crop_info[crop]['icon']
                    st.markdown(f"{icon} **{crop}**")
    
    else:
        st.error("❌ Weather data unavailable. Please try again later.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🌱 AgriSmart Advisor | Powered by Machine Learning | Data from IMD India</p>
            <p style="font-size: 0.9rem;">💡 For best results, get your soil tested at a local agricultural lab</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
