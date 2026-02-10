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
root_dir = os.path.dirname(current_dir)
CACHE_FILE_PATH = os.path.join(root_dir, "daily_weather_cache.csv")
MODELS_DIR = os.path.join(root_dir, "Models")

HA_TO_SQFT = 107639.0 

# --- CROP KNOWLEDGE BASE ---
# Maps Crop Name -> Yield (tons/ha), Fertilizer, Pesticide
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

# CRITICAL: This list matches the model's alphabetical training order
CROP_LIST = sorted(list(CROP_INFO.keys()))

# --- EXACT DATA LISTS ---

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
    "Autumn": "Autumn     ",
    "Kharif": "Kharif     ",
    "Rabi": "Rabi       ",
    "Summer": "Summer     ",
    "Whole Year": "Whole Year ",
    "Winter": "Winter     "
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
        # Fallback for local testing
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
    """Generates the 44 feature names in Model's exact order."""
    num_cols = [
        'Area', 'Annual_Rainfall', 'Avg_Temperature', 'humidity_pct', 
        'soil_N_kg_sq_foot', 'soil_P_kg_sq_foot', 'soil_K_kg_sq_foot', 'soil_pH'
    ]
    season_cols = [f"Season_{s}" for s in sorted(list(SEASON_MAPPING.values()))]
    state_cols = [f"State_{s}" for s in sorted(UI_STATES)]
    return num_cols + season_cols + state_cols

def preprocess_input(state, area_sqft, n_ha, p_ha, k_ha, ph, weather_row, season_override=None):
    """
    Creates the 44-column input vector.
    If 'season_override' is provided, it forces that season (for Whole Year logic).
    """
    model_cols = generate_manual_features()
    data_array = np.zeros((1, len(model_cols)))
    input_df = pd.DataFrame(data_array, columns=model_cols)

    # 1. Determine Season
    if season_override:
        messy_season = SEASON_MAPPING.get(season_override, "Whole Year ")
    else:
        current_month = datetime.now().strftime("%B")
        clean_season = get_season_clean(current_month)
        messy_season = SEASON_MAPPING.get(clean_season, "Whole Year ") 
    
    # 2. Conversions
    n_sq = n_ha / HA_TO_SQFT
    p_sq = p_ha / HA_TO_SQFT
    k_sq = k_ha / HA_TO_SQFT

    # 3. Fill Numerical
    input_df['Area'] = area_sqft
    input_df['Annual_Rainfall'] = weather_row['Annual_Rainfall']
    input_df['Avg_Temperature'] = weather_row['Avg_Temperature']
    input_df['humidity_pct'] = weather_row['humidity_pct']
    input_df['soil_N_kg_sq_foot'] = n_sq
    input_df['soil_P_kg_sq_foot'] = p_sq
    input_df['soil_K_kg_sq_foot'] = k_sq
    input_df['soil_pH'] = ph

    # 4. Fill One-Hot Season
    season_col = f"Season_{messy_season}"
    if season_col in input_df.columns:
        input_df[season_col] = 1
    
    # 5. Fill One-Hot State
    state_col = f"State_{state}"
    if state_col in input_df.columns:
        input_df[state_col] = 1
        
    return input_df

# --- MAIN APP UI ---

assets = load_assets()
weather_df = get_weather_data()

st.title("🌾 Smart Kitchen Garden Advisor")
st.markdown("Your AI-powered guide for home farming.")

with st.sidebar:
    st.header("1. Location & Area")
    state_input = st.selectbox("Select State", UI_STATES)
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
        try:
            model = assets['model']

            # --- DUAL PREDICTION LOGIC ---
            # 1. Ask model: "What grows best in the CURRENT season?"
            df_seasonal = preprocess_input(state_input, area_sqft, N_ha, P_ha, K_ha, ph, state_data)
            pred_idx_seasonal = int(model.predict(df_seasonal)[0])
            try:
                prob_seasonal = np.max(model.predict_proba(df_seasonal))
            except:
                prob_seasonal = 0.0 # Fallback if model doesn't support probability
            
            # 2. Ask model: "What grows best if we consider WHOLE YEAR crops?"
            df_whole = preprocess_input(state_input, area_sqft, N_ha, P_ha, K_ha, ph, state_data, season_override="Whole Year")
            pred_idx_whole = int(model.predict(df_whole)[0])
            try:
                prob_whole = np.max(model.predict_proba(df_whole))
            except:
                prob_whole = 0.0

            # 3. Decision: Who wins?
            # If the "Whole Year" confidence is higher than "Seasonal", or if they are equal, pick Whole Year.
            # This handles cases where seasonal conditions are bad, but a perennial crop thrives.
            if prob_whole > prob_seasonal:
                final_idx = pred_idx_whole
                note = " (Suitable Year-Round)"
            else:
                final_idx = pred_idx_seasonal
                note = f" (Best for {datetime.now().strftime('%B')})"

            # --- DISPLAY RESULTS ---
            if 0 <= final_idx < len(CROP_LIST):
                crop_name = CROP_LIST[final_idx]
                info = CROP_INFO.get(crop_name, DEFAULT_INFO)
                
                # Yield Calc: (Tons/Ha -> Kg/Garden)
                yield_ton_ha = info['yield']
                yield_kg = (yield_ton_ha * 1000) * (area_sqft / HA_TO_SQFT)
                
                st.success(f"🌱 Recommended Crop: **{crop_name}**{note}")
                
                # Detailed Advice Columns
                col_A, col_B, col_C = st.columns(3)
                
                with col_A:
                    st.info(f"**🧪 Fertilizer Advice**\n\n{info['fert']}")
                    
                with col_B:
                    st.warning(f"**🐛 Pesticide Advice**\n\n{info['pest']}")
                    
                with col_C:
                    st.metric(
                        label="Est. Yield (in your garden)", 
                        value=f"{yield_kg:.2f} kg",
                        delta=f"{yield_ton_ha} Tons/Ha avg"
                    )

                st.balloons()
            else:
                st.error(f"⚠️ Prediction Index {final_idx} is out of range.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif weather_df is None:
    st.info("Loading weather data... (Please ensure the daily update script has run)")
