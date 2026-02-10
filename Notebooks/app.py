import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AgriSmart Labs",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- SCIENTIFIC DARK MODE CSS ---
st.markdown("""
<style>
    /* 1. MAIN BACKGROUND: Deep Scientific Black with Grid Pattern */
    .stApp {
        background-color: #050505;
        background-image: linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
    }

    /* 2. TYPOGRAPHY: Scientific & High Contrast */
    h1, h2, h3 {
        color: #00e5ff !important; /* Cyan Neon */
        font-family: 'Segoe UI', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stMarkdown p, .stMarkdown label, .stMarkdown div {
        color: #e0e0e0 !important;
    }
    
    /* 3. METRIC CARDS: HUD Style */
    div[data-testid="stMetric"] {
        background-color: rgba(0, 229, 255, 0.05); /* Cyan Tint */
        border: 1px solid rgba(0, 229, 255, 0.2);
        padding: 15px;
        border-radius: 5px;
        color: #e0e0e0;
    }
    div[data-testid="stMetricLabel"] {
        color: #00e5ff;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }

    /* 4. INPUT CARDS: Glassmorphism */
    .input-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* 5. RESULT CARDS: Holographic Style */
    .result-card {
        background: linear-gradient(135deg, rgba(0, 255, 65, 0.1), rgba(0, 0, 0, 0.5));
        border-left: 4px solid #00ff41; /* Neon Green */
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
    }
    .result-title {
        color: #00ff41;
        font-size: 1.8rem;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        margin-bottom: 5px;
    }
    .result-subtitle {
        color: #b0bec5;
        font-size: 0.9rem;
        font-family: 'Courier New', monospace;
    }

    /* 6. ADVICE GRIDS */
    .advice-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 5px;
        height: 100%;
        color: #cfd8dc;
        font-size: 0.9rem;
    }
    .advice-header {
        color: #00e5ff;
        font-weight: bold;
        margin-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 5px;
        font-family: 'Courier New', monospace;
    }

    /* 7. BUTTONS: Cyberpunk Style */
    div.stButton > button {
        background: transparent;
        color: #00ff41;
        border: 1px solid #00ff41;
        font-family: 'Courier New', monospace;
        text-transform: uppercase;
        font-weight: bold;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: #00ff41;
        color: black;
        box-shadow: 0 0 10px #00ff41;
    }
    
    /* Force white text on sliders/inputs */
    .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #e0e0e0 !important;
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

# --- HERO SECTION ---
# Clean scientific header without sidebar
assets = load_assets()
weather_df = get_weather_data()

c1, c2 = st.columns([1, 4])
with c1:
    # Scientific Icon (Microscope/Plant)
    st.image("https://cdn-icons-png.flaticon.com/512/3058/3058995.png", width=80)
with c2:
    st.title("AGRI-SMART LABS")
    st.caption("AI-POWERED HYDROPONICS & SOIL ANALYTICS")

st.markdown("---")

# --- MAIN INPUT GRID (Centered) ---
st.subheader("📡 FIELD DATA ENTRY")

# Row 1: Location & Area
col1, col2 = st.columns(2)
with col1:
    state_input = st.selectbox("LOCATION MONITORING", UI_STATES)
with col2:
    area_sqft = st.number_input("CULTIVATION AREA (SQ FT)", min_value=10, value=100)

# Row 2: Soil Sensors
st.markdown("<br>", unsafe_allow_html=True)
col3, col4, col5, col6 = st.columns(4)

with col3:
    N_ha = st.number_input("NITROGEN (N) [kg/ha]", value=140)
with col4:
    P_ha = st.number_input("PHOSPHORUS (P) [kg/ha]", value=40)
with col5:
    K_ha = st.number_input("POTASSIUM (K) [kg/ha]", value=40)
with col6:
    ph = st.slider("SOIL pH LEVEL", 4.0, 9.5, 6.5)

st.markdown("---")

# --- LIVE METRICS DASHBOARD ---
if weather_df is not None and state_input in weather_df['State'].values:
    state_data = weather_df[weather_df['State'] == state_input].iloc[0]
    
    st.subheader(f"📊 ENVIRONMENTAL TELEMETRY: {state_input.upper()}")
    
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        st.metric("AVG TEMP", f"{state_data['Avg_Temperature']:.1f}°C")
    with w_col2:
        st.metric("PRECIPITATION", f"{state_data['Annual_Rainfall']:.1f} mm")
    with w_col3:
        st.metric("REL. HUMIDITY", f"{state_data['humidity_pct']:.0f}%")
    with w_col4:
        current_vpd = calculate_vpd(state_data['Avg_Temperature'], state_data['humidity_pct'])
        st.metric("VAPOR PRESSURE DEFICIT", f"{current_vpd:.2f} kPa")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # PREDICT BUTTON
    col_centered = st.columns([1, 2, 1])
    with col_centered[1]:
        predict_btn = st.button("INITIATE CROP ANALYSIS SEQUENCE")

    if predict_btn and assets and 'model' in assets:
        with st.spinner("PROCESSING ALGORITHMS..."):
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
                    note = "SUITABLE YEAR-ROUND"
                else:
                    final_idx = pred_idx_seasonal
                    note = f"OPTIMAL FOR {datetime.now().strftime('%B').upper()}"

                # --- DISPLAY RESULTS ---
                if 0 <= final_idx < len(CROP_LIST):
                    crop_name = CROP_LIST[final_idx]
                    info = CROP_INFO.get(crop_name, DEFAULT_INFO)
                    
                    yield_ton_ha = info['yield']
                    yield_kg = (yield_ton_ha * 1000) * (area_sqft / HA_TO_SQFT)
                    
                    # Result Card HTML
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">IDENTIFIED CROP: {crop_name.upper()}</div>
                        <div class="result-subtitle">ANALYSIS: {note}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Advice Grid
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_A, col_B, col_C = st.columns(3)
                    
                    with col_A:
                        st.markdown(f"""
                        <div class="advice-box">
                            <div class="advice-header">⚗️ NUTRIENT PROTOCOL</div>
                            {info['fert']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_B:
                        st.markdown(f"""
                        <div class="advice-box">
                            <div class="advice-header">🛡️ PROTECTION PROTOCOL</div>
                            {info['pest']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_C:
                        st.markdown(f"""
                        <div class="advice-box">
                            <div class="advice-header">📉 YIELD PROJECTION</div>
                            <span style="font-size: 1.5rem; color: #00e5ff; font-weight: bold;">{yield_kg:.2f} KG</span><br>
                            <span style="font-size: 0.8rem;">ESTIMATED FOR {area_sqft} SQ FT</span>
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    st.error(f"⚠️ SYSTEM ERROR: Index {final_idx} out of bounds.")

            except Exception as e:
                st.error(f"COMPUTATION ERROR: {e}")

elif weather_df is None:
    st.warning("⚠️ WEATHER TELEMETRY OFFLINE. CHECK ROOT DIRECTORY FOR CSV CACHE.")
