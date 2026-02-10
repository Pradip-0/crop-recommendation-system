import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import base64
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AgriSmart Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARK DASHBOARD CSS ---
st.markdown("""
<style>
    /* Global Dark Background */
    .stApp {
        background-color: #0e1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Container Style */
    .css-card {
        background-color: #181b21;
        border: 1px solid #303339;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #181b21;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #303339;
    }
    div[data-testid="stMetricLabel"] {
        color: #9ca3af; /* Muted text */
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #13151a;
        border-right: 1px solid #303339;
    }
    
    /* Custom Result Banner - Default Style (Overridden by Python if image exists) */
    .result-banner {
        color: white;
        padding: 40px 25px; /* Increased padding for better image visibility */
        border-radius: 12px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #43a047;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        position: relative;
        overflow: hidden;
    }
    
    .crop-name {
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(0,0,0,0.8);
        z-index: 2;
    }
    .result-label {
        font-size: 0.9rem; 
        opacity: 0.9;
        text-shadow: 0 1px 4px rgba(0,0,0,0.8);
        z-index: 2;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .match-tag {
        background-color: rgba(0,0,0,0.6);
        backdrop-filter: blur(5px);
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1px solid rgba(255,255,255,0.4);
        z-index: 2;
        font-weight: bold;
    }

    /* Advice Cards */
    .info-card {
        background-color: #181b21;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #555;
        height: 100%;
        border: 1px solid #303339;
    }
    .card-fert { border-left-color: #00e676; } /* Bright Green */
    .card-pest { border-left-color: #ff9100; } /* Bright Orange */
    .card-yield { border-left-color: #2979ff; } /* Bright Blue */

    .card-header {
        font-weight: 700;
        font-size: 1.1rem;
        color: #e0e0e0;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #303339;
        padding-bottom: 8px;
    }
    
    .card-content {
        color: #b0bec5;
        line-height: 1.5;
    }
    
    /* Button Override */
    div.stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #1b5e20;
        box-shadow: 0 0 10px rgba(46, 125, 50, 0.5);
    }

    /* Force text colors for inputs */
    label, .stMarkdown p {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
CACHE_FILE_PATH = os.path.join(root_dir, "daily_weather_cache.csv")
MODELS_DIR = os.path.join(root_dir, "Models")
IMAGES_DIR = os.path.join(root_dir, "data", "Images") # Path to images

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

def get_img_as_base64(file_path):
    """Converts a binary file to base64 string for HTML embedding."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
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

# --- MAIN LAYOUT ---

assets = load_assets()
weather_df = get_weather_data()

# Hero Section
st.title("🌾 AgriSmart Analytics")
st.markdown("### Professional Crop Intelligence Dashboard")
st.markdown("---")

# Layout
with st.sidebar:
    st.markdown("### 🛠️ Input Parameters")
    
    st.markdown("**1. Location Profile**")
    state_input = st.selectbox("Select State", UI_STATES)
    area_sqft = st.number_input("Garden Area (sq ft)", min_value=10, value=100)

    st.markdown("---")
    st.markdown("**2. Soil Chemistry**")
    st.caption("Values in kg/hectare")
    
    col1, col2 = st.columns(2)
    with col1:
        N_ha = st.number_input("Nitrogen (N)", value=140)
        K_ha = st.number_input("Potassium (K)", value=40)
    with col2:
        P_ha = st.number_input("Phosphorus (P)", value=40)
        ph = st.slider("Soil pH", 4.0, 9.5, 6.5)

    st.markdown("---")
    predict_btn = st.button("RUN ANALYSIS", type="primary")

# --- MAIN DASHBOARD LOGIC ---

if weather_df is not None and state_input in weather_df['State'].values:
    state_data = weather_df[weather_df['State'] == state_input].iloc[0]
    
    st.subheader(f"📍 Climate Telemetry: {state_input}")
    
    # 4-Column Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{state_data['Avg_Temperature']:.1f}°C", "Live")
    c2.metric("Rainfall", f"{state_data['Annual_Rainfall']:.1f} mm", "Annual Avg")
    c3.metric("Humidity", f"{state_data['humidity_pct']:.0f}%", "Avg")
    
    current_vpd = calculate_vpd(state_data['Avg_Temperature'], state_data['humidity_pct'])
    c4.metric("VPD", f"{current_vpd:.2f} kPa", "Moisture Stress")
    
    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn and assets and 'model' in assets:
        with st.spinner("Processing agricultural models..."):
            try:
                model = assets['model']

                # --- DUAL PREDICTION ---
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
                    note = "YEAR-ROUND VARIETY"
                else:
                    final_idx = pred_idx_seasonal
                    note = f"SEASONAL: {datetime.now().strftime('%B').upper()}"

                # --- RESULT DISPLAY ---
                if 0 <= final_idx < len(CROP_LIST):
                    crop_name = CROP_LIST[final_idx]
                    info = CROP_INFO.get(crop_name, DEFAULT_INFO)
                    
                    yield_ton_ha = info['yield']
                    yield_kg = (yield_ton_ha * 1000) * (area_sqft / HA_TO_SQFT)
                    
                    # --- DYNAMIC BACKGROUND IMAGE LOGIC (ROBUST VERSION) ---
                    
                    # 1. Generate variations of filename (Arhar/Tur -> Arhar Tur, Arhar_Tur, etc)
                    variations = [
                        crop_name, 
                        crop_name.replace("/", " "),
                        crop_name.replace("/", "_"),
                        crop_name.replace("/", "-"),
                        crop_name.replace(" ", "_"),
                        crop_name.replace(" ", "")
                    ]
                    # Add lowercase versions
                    variations.extend([v.lower() for v in variations])
                    # Unique list
                    variations = list(set(variations))
                    
                    img_path = None
                    # 2. Robust Search
                    if os.path.exists(IMAGES_DIR):
                        for name in variations:
                            for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"]:
                                possible_path = os.path.join(IMAGES_DIR, f"{name}{ext}")
                                if os.path.exists(possible_path):
                                    img_path = possible_path
                                    break
                            if img_path: break
                    else:
                        print(f"Warning: Images directory not found at {IMAGES_DIR}")

                    # 3. Construct CSS
                    banner_style = "background: linear-gradient(90deg, #1b5e20 0%, #2e7d32 100%);" # Default
                    
                    if img_path:
                        img_b64 = get_img_as_base64(img_path)
                        if img_b64:
                            # Determine mime type roughly
                            mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
                            banner_style = f"""
                                background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.8)), 
                                url("data:{mime};base64,{img_b64}");
                                background-size: cover;
                                background-position: center;
                            """

                    # 4. Render Banner
                    st.markdown(f"""
                    <div class="result-banner" style='{banner_style}'>
                        <div>
                            <div class="result-label">OPTIMAL CROP IDENTIFIED</div>
                            <div class="crop-name">{crop_name}</div>
                        </div>
                        <div class="match-tag">{note}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 5. Detailed Cards
                    col_A, col_B, col_C = st.columns(3)
                    
                    with col_A:
                        st.markdown(f"""
                        <div class="info-card card-fert">
                            <div class="card-header">🧪 Fertilizer Protocol</div>
                            <div class="card-content">{info['fert']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_B:
                        st.markdown(f"""
                        <div class="info-card card-pest">
                            <div class="card-header">🛡️ Pest Management</div>
                            <div class="card-content">{info['pest']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_C:
                        st.markdown(f"""
                        <div class="info-card card-yield">
                            <div class="card-header">📉 Yield Forecast</div>
                            <div style="font-size: 2rem; font-weight: 800; color: #1565C0;">{yield_kg:.2f} kg</div>
                            <div style="color: #666; font-size: 0.8rem;">Estimated for {area_sqft} sq ft</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.balloons()
                else:
                    st.error(f"Prediction Error: Index {final_idx} out of range.")

            except Exception as e:
                st.error(f"System Error: {e}")

    elif not predict_btn:
        st.info("👋 Ready to analyze. Adjust parameters in the sidebar and click RUN ANALYSIS.")

elif weather_df is None:
    st.warning("⚠️ WEATHER DATA OFFLINE. Please check repository configuration.")
