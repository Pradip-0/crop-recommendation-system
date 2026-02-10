import pandas as pd
import numpy as np
import imdlib as imd
from datetime import datetime, date
import os

# --- CONFIG ---
# This saves the file to the MAIN folder (one level up from Notebooks)
# So your app.py (which is likely in the main folder or Notebooks) can find it.
CACHE_FILE = "daily_weather_cache.csv" 

# Coordinates
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

def calculate_vpd(temp_c, rh_pct):
    svp = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    avp = svp * (rh_pct / 100)
    return max(0, svp - avp)

def fetch_data():
    print("Starting Daily Weather Fetch...")
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
        print(f"Error downloading IMD data: {e}")
        return

    all_data = []
    
    for state, (lat, lon) in state_coords.items():
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
            total_rain = float(m_rain.rain.sum())
            avg_rain = total_rain / 5
            rainy_days = (m_rain.rain > 2.5).sum().item() / 5
            
            avg_humidity = 60.0 
            vpd = calculate_vpd(weighted_temp, avg_humidity)

            all_data.append({
                "State": state,
                "Avg
