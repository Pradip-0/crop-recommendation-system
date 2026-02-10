import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime, date

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AgriSmart Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR BEAUTIFICATION ---
st.markdown("""
<style>
    /* Main Background and Font */
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2E7D32; /* Green shade */
        font-weight: 600;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
    }
    div[data-testid="stMetricLabel"] {
        color: #666;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #2E7D32;
        font-size: 1.5rem;
    }
    
    /* Result Cards */
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #2E7D32;
    }
    .result-title {
        color: #1B5E20;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .result-subtitle {
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 20px;
        font-style: italic;
    }
    
    /* Advice Cards */
    .advice-box {
        padding: 15px;
        border-radius: 10px;
        height: 100%;
    }
    .fert-box { background-color: #E8F5E9; border: 1px solid #C8E6C9; color: #1B5E20; }
    .pest-box { background-color: #FFF3E0; border: 1px solid #FFE0B2; color: #E65100; }
    .yield-box { background-color: #E3F2FD; border: 1px solid #BBDEFB; color: #0D47A1; }
    
    .advice-header {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Button */
    div.stButton > button {
        background-color: #2E7D32;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1B5E20;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
CACHE_FILE_PATH = os.path.join(root_dir, "daily_weather_cache.csv")
MODELS_DIR = os.path.join(root_dir, "Models")

HA_TO_SQFT = 107639.0 

# --- CROP KNOWLEDGE BASE ---
CROP_INFO = {
    'Arhar/Tur': {'yield': 1.5, 'fert': '20kg N, 50kg P per ha. Sulfur ensures high protein.', 'pest': 'Pod Borer: Spray Indoxacarb 14.5 SC.'},
    'Banana': {'yield': 40.0, 'fert': '200g N, 60g P, 300g K per plant. Needs heavy Potash.', 'pest': 'Rhizome Weevil: Apply Carbofuran granules.'},
    'Black pepper': {'yield': 0.6, 'fert': 'Organic manure + 100g NPK per vine per year.', 'pest': 'Pollu Beetle: Spray Quinalphos.'},
    'Cardamom': {'yield': 0.3, 'fert': '75kg N, 75kg P, 150kg K per ha.', 'pest': 'Thrips: Spray Diafenthiuron.'},
    'Cashewnut': {'yield': 1.0, 'fert': '500g N, 125g P, 125g K per tree.', 'pest': 'Tea Mosquito Bug: Spray Lambda-cyhalothrin.'},
    'Coconut': {'yield': 12.0, 'fert': '500g N, 320g P, 1200g K per tree/year.', 'pest': 'Rhinoceros Beetle: Use Naphthalene balls in leaf axils.'},
    'Coriander': {'yield': 1.2, 'fert': '15kg N, 40kg P, 20kg K per ha.', 'pest': 'Aphids: Spray Dimethoate.'},
    'Cowpea(Lobia)': {'yield': 1.5, 'fert': '20kg N, 60kg P, 20kg K per ha.', 'pest': 'Pod Borer: Spray Neem oil.'},
    'Dry chillies': {'yield': 2.5, 'fert': '120kg N, 60kg P, 60kg K per ha.', 'pest': 'Thrips/Mites: Spray Imidacloprid.'},
    'Garlic': {'yield': 6.0, 'fert': '100kg N, 50kg P, 50kg K per ha.', 'pest': 'Thrips: Spray Fipronil.'},
    'Ginger': {'yield': 15.0, 'fert': '75kg N, 50kg P, 50kg K per ha. Heavy mulching required.', 'pest': 'Shoot Borer: Spray Chlorpyrifos.'},
    'Gram': {'yield': 1.2, 'fert': '20kg N, 60kg P per ha. Avoid excess Nitrogen.', 'pest': 'Pod Borer: Spray Emamectin benzoate.'},
    'Groundnut': {'yield': 2.5, 'fert': '20kg N, 60kg P, 40kg K per ha + Gypsum.', 'pest': 'Leaf Miner: Spray Monocrotophos.'},
    'Masoor': {'yield': 1.2, 'fert': '20kg N, 40kg P per ha.', 'pest': 'Aphids: Spray Dimethoate.'},
    'Moong(Green Gram)': {'yield': 1.0, 'fert': '20kg N, 50kg P per ha.', 'pest': 'Whitefly: Spray Triazophos.'},
    'Onion': {'yield': 25.0, 'fert': '100kg N, 50kg P, 50kg K per ha.', 'pest': 'Thrips: Spray Fipronil.'},
    'Peas & beans (Pulses)': {'yield': 8.0, 'fert': '50kg N, 60kg P, 60kg K per ha.', 'pest': 'Pod Borer: Spray Neem oil.'},
    'Potato': {'yield': 25.0, 'fert': '120kg N, 60kg P, 100kg K per ha. Soil must be loose.', 'pest': 'Late Blight: Spray Mancozeb.'},
    'Sweet potato': {'yield': 20.0, 'fert': '50kg N, 25kg P, 50kg K per ha.', 'pest': 'Sweet Potato Weevil: Use Pheromone traps.'},
    'Tapioca': {'yield': 30.0, 'fert': '100kg N, 50kg P, 100kg K per ha.', 'pest': 'Scale insects: Spray Neem oil.'},
    'Turmeric': {'yield': 25.0, 'fert': '120kg N, 60kg P, 60kg K per ha.', 'pest': 'Rhizome Scale: Dip seed in Quinalphos.'},
    'Urad': {'yield': 1.0, 'fert': '20kg N, 50kg P per ha.', 'pest': 'Pod Borer: Spray Indoxacarb.'}
}

DEFAULT_INFO = {'yield': 2.0, 'fert': 'Standard NPK 100:50:50', 'pest': 'Use Neem Oil as preventive.'}
CROP_LIST = sorted(list(CROP_INFO.keys()))

# --- DATA LISTS ---
UI_STATES = sorted([
    'Assam', 'Karnataka', 'Kerala', 'Meghalaya', 'West Bengal',
    'Puducherry', 'Goa', 'Andhra Pradesh', 'Tamil Nadu', 'Odisha',
    'Bihar', 'Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Mizoram',
    'Punjab', 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh',
    'Tripura', 'Nagaland', 'Chhattisgarh', 'Uttarakhand', 'Jharkhand',
    'Delhi', 'Manipur', 'Jammu and Kashmir', 'Telangana',
    'Arunachal Pradesh', 'Sikkim'
])

SEASON_MAPPING = {
    "Autumn": "Autumn     ", "Kharif": "Kharif     ", "Rabi": "Rabi       ",
    "Summer": "Summer     ", "Whole Year": "Whole Year ", "Winter": "Winter     "
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
    assets = {}
    if os.path.exists(model_path):
        try:
            assets['model'] = joblib.load(model_path)
        except Exception as e:
             st.error(f"🚨 Error loading model: {e}")
             return None
    else:
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
    num_cols = [
        'Area', 'Annual_Rainfall', 'Avg_Temperature', 'humidity_pct', 
        'soil_N_kg_sq_foot', 'soil_P_kg_sq_foot', 'soil_K_kg_sq_foot', 'soil_pH'
    ]
    season_cols = [f"Season_{s}" for s in sorted(list(SEASON_MAPPING.values()))]
    state_cols = [f"State_{s}" for s in sorted(UI_STATES)]
    return num_cols + season_cols + state_cols

def preprocess_input(state, area_sqft, n_ha, p_ha, k_ha, ph, weather_row, season_override=None):
    model_cols = generate_manual_features()
    data_array = np.zeros((1, len(model_cols)))
    input_df = pd.DataFrame(data_array, columns=model_cols)

    if season_override:
        messy_season = SEASON_MAPPING.get(season_override, "Whole Year ")
    else:
        current_month = datetime.now().strftime("%B")
        clean_season = get_season_clean(current_month)
        messy_season = SEASON_MAPPING.get(clean_season, "Whole Year ") 
    
    n_sq = n_ha / HA_TO_SQFT
    p_sq = p_ha / HA_TO_SQFT
    k_sq = k_ha / HA_TO_SQFT

    input_df['Area'] = area_sqft
    input_df['Annual_Rainfall'] = weather_row['Annual_Rainfall']
    input_df['Avg_Temperature'] = weather_row['Avg_Temperature']
    input_df['humidity_pct'] = weather_row['humidity_pct']
    input_df['soil_N_kg_sq_foot'] = n_sq
    input_df['soil_P_kg_sq_foot'] = p_sq
    input_df['soil_K_kg_sq_foot'] = k_sq
    input_df['soil_pH'] = ph

    season_col = f"Season_{messy_season}"
    if season_col in input_df.columns:
        input_df[season_col] = 1
    
    state_col = f"State_{state}"
    if state_col in input_df.columns:
        input_df[state_col] = 1
        
    return input_df

# --- MAIN APP UI ---

assets = load_assets()
weather_df = get_weather_data()

# Hero Section
st.title("🌾 AgriSmart Advisor")
st.markdown("### Your Intelligent Companion for Home Farming")
st.markdown("---")

# Layout: Sidebar for controls, Main for results
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.header("🏡 Garden Profile")
    
    state_input = st.selectbox("📍 Select State", UI_STATES)
    area_sqft = st.number_input("📐 Garden Area (sq ft)", min_value=10, value=100)

    st.markdown("---")
    st.header("🧪 Soil Composition")
    st.caption("Enter nutrient values in **kg/hectare**.")
    
    col1, col2 = st.columns(2)
    with col1:
        N_ha = st.number_input("Nitrogen (N)", value=140)
        K_ha = st.number_input("Potassium (K)", value=40)
    with col2:
        P_ha = st.number_input("Phosphorus (P)", value=40)
        ph = st.slider("Soil pH", 4.0, 9.5, 6.5)

    st.markdown("---")
    predict_btn = st.button("🌱 Get Recommendation", type="primary")

# --- MAIN LOGIC ---

if weather_df is not None and state_input in weather_df['State'].values:
    state_data = weather_df[weather_df['State'] == state_input].iloc[0]
    
    # Weather Dashboard
    st.subheader(f"🌤️ Live Climate Analysis: {state_input}")
    
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        st.metric("Temperature", f"{state_data['Avg_Temperature']:.1f}°C")
    with w_col2:
        st.metric("Rainfall", f"{state_data['Annual_Rainfall']:.1f} mm")
    with w_col3:
        st.metric("Humidity", f"{state_data['humidity_pct']:.0f}%")
    with w_col4:
        current_vpd = calculate_vpd(state_data['Avg_Temperature'], state_data['humidity_pct'])
        st.metric("VPD", f"{current_vpd:.2f} kPa")

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn and assets and 'model' in assets:
        with st.spinner("🤖 Analyzing soil and climate data..."):
            try:
                model = assets['model']

                # --- DUAL PREDICTION LOGIC ---
                df_seasonal = preprocess_input(state_input, area_sqft, N_ha, P_ha, K_ha, ph, state_data)
                pred_idx_seasonal = int(model.predict(df_seasonal)[0])
                try: prob_seasonal = np.max(model.predict_proba(df_seasonal))
                except: prob_seasonal = 0.0 
                
                df_whole = preprocess_input(state_input, area_sqft, N_ha, P_ha, K_ha, ph, state_data, season_override="Whole Year")
                pred_idx_whole = int(model.predict(df_whole)[0])
                try: prob_whole = np.max(model.predict_proba(df_whole))
                except: prob_whole = 0.0

                if prob_whole > prob_seasonal:
                    final_idx = pred_idx_whole
                    note = "Suitable Year-Round"
                else:
                    final_idx = pred_idx_seasonal
                    note = f"Best for {datetime.now().strftime('%B')}"

                # --- DISPLAY RESULTS ---
                if 0 <= final_idx < len(CROP_LIST):
                    crop_name = CROP_LIST[final_idx]
                    info = CROP_INFO.get(crop_name, DEFAULT_INFO)
                    
                    yield_ton_ha = info['yield']
                    yield_kg = (yield_ton_ha * 1000) * (area_sqft / HA_TO_SQFT)
                    
                    # Custom Result Card using HTML
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">🌱 Recommended Crop: {crop_name}</div>
                        <div class="result-subtitle">Match Type: {note}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed Advice Grid
                    col_A, col_B, col_C = st.columns(3)
                    
                    with col_A:
                        st.markdown(f"""
                        <div class="advice-box fert-box">
                            <div class="advice-header">🧪 Fertilizer</div>
                            {info['fert']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_B:
                        st.markdown(f"""
                        <div class="advice-box pest-box">
                            <div class="advice-header">🐛 Pesticide</div>
                            {info['pest']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_C:
                        st.markdown(f"""
                        <div class="advice-box yield-box">
                            <div class="advice-header">📉 Exp. Yield</div>
                            <b>{yield_kg:.2f} kg</b><br>
                            <small>(Based on garden area)</small>
                        </div>
                        """, unsafe_allow_html=True)

                    st.balloons()
                else:
                    st.error(f"⚠️ Prediction Index {final_idx} is out of range.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    
    elif not predict_btn:
        st.info("👈 Please enter your soil details in the sidebar and click 'Get Recommendation'")

elif weather_df is None:
    st.warning("⚠️ Weather data is loading. Please ensure 'daily_weather_cache.csv' is in the root directory.")
