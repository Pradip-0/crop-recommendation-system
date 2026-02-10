# 🌾 AgriSmart (Crop Recommendation System)

## Your Intelligent, AI-Powered Companion for Precision Home Farming.

**Live Demo:**  ([*Click here to launch App*](https://crop-recommendation-system-8xvxxwfmzyrwkutnvhbvbz.streamlit.app/))

## 📖 Project Description

This is a professional data-driven system designed to bridge the gap between complex agricultural data and home gardening decisions. By integrating historical climate telemetry with advanced machine learning, it recommends the optimal crop for a specific Indian state and garden size.

Unlike generic crop recommenders, AgriSmart features a Dual-Prediction Engine. This logic simultaneously evaluates "Seasonal" crops against "Year-Round" varieties (like Coconut or Banana) to identify the highest confidence match, ensuring users don't miss out on perennial opportunities just because of the current month.

**It provides a complete cultivation protocol, including:**

Scientific Recommendations: Based on soil NPK values and pH.

Yield Estimation: Converts industrial "Tons/Hectare" metrics into practical "Kg/Garden Area".

Care Guides: Specific fertilizer and pesticide protocols for the predicted crop.

## ✨ Key Features

📍 Location-Aware Telemetry: Automatically fetches and caches historical weather data (Temperature, Rainfall, Humidity) for 30+ Indian states/UTs using IMDLib.

🧠 Dual-Prediction AI: A custom logic layer that prevents seasonal bias by comparing seasonal vs. perennial crop probabilities.

🧪 Precision Soil Analysis: Tailors advice based on Nitrogen, Phosphorus, Potassium, and pH inputs.

📉 Smart Yield Calculator: Automatically calculates expected harvest weight based on your specific garden square footage.

🌑 Pro Dashboard UI: A dark-mode, high-contrast interface designed for clarity and professional aesthetics.

## ⚙️ Tech Stack

***Frontend***: Streamlit (Python-based web framework)

***Machine Learning Algorithm***: XGBoost Classifier

***Data Processing***: Pandas, NumPy

***Climate Data***: IMDLib (India Meteorological Department Data)

***Automation***: GitHub Actions (Daily weather data fetching)

## 📂 Dataset & Model Logic

The model was trained on a comprehensive agricultural dataset containing records from across India.

**Algorithm**: XGBoost Classifier (Extreme Gradient Boosting).

**Input Features (10 Total)**: Area, N, P, K, pH, Rainfall, Temperature, Humidity, States, and Seasons.

**Target Classes**: 22 unique crop varieties (e.g., Rice, Banana, Coffee, Cotton, etc.).

**Performance**: The model achieves ~56% accuracy on the test set.

[**Disclaimer:** The model is not a well-performing model due to a few reasons. 
1. The data available is not for kitchen gardening; it is for industrial purposes.
2. There are more features for predicting a crop category, such as the previous year yeild on that area, etc. Those aren't considered in this project.
This project just provides an outline of how an approach can be taken to build such systems.]

## 🚀 Quick Start

Follow these steps to set up the project locally.

### 1. Clone the Repository

git clone [https://github.com/your-username/crop-recommendation-system.git](https://github.com/your-username/crop-recommendation-system.git)
cd crop-recommendation-system


### 2. Create a Virtual Environment

It is highly recommended to use Python 3.10 or 3.11 to avoid compilation issues with XGBoost.

*Windows*
python -m venv venv
venv\Scripts\activate

*Mac/Linux*
python3 -m venv venv
source venv/bin/activate


### 3. Install Dependencies

pip install -r requirements.txt


### 4. Project Structure

Ensure your directory looks like this:
``` text
crop-recommendation-system/
├── Notebooks/
│   └── app.py                 # Main Application Logic
│   └── daily_update.py        # Weather Fetching Script
├── Models/
│   └── crop_recomender.joblib # Trained XGBoost Model
│   └── label_encoder.joblib   # Target Decoder
├── .github/workflows/
│   └── weather_cron.yml       # Automation Workflow
├── daily_weather_cache.csv    # Climate Data (Generated via script)
└── requirements.txt
```

### 5. Run the App

streamlit run Notebooks/app.py


## 🤖 Automation (GitHub Actions)

This project uses a Cron Job via GitHub Actions to keep weather data fresh without slowing down the user experience.

***Workflow File***: .github/workflows/weather_cron.yml

***Schedule***: Runs daily at 00:00 UTC.

***Action***: Executes Notebooks/daily_update.py, fetches the latest climate data from IMD, saves it to daily_weather_cache.csv, and commits the file back to the repository.

***Benefit***: The Streamlit app loads instantly by reading the static CSV instead of querying APIs in real-time.

## 📸 Visuals

<img width="1895" height="862" alt="App image" src="https://github.com/user-attachments/assets/e4716dd8-2714-4fb0-a8d9-98d02a5098bb" />

1. The Professional Dashboard
Real-time climate telemetry and soil input grid in Dark Mode.

2. Prediction Results
The Result Card shows the recommended crop, expected yield (in kg), and specific care instructions.p




