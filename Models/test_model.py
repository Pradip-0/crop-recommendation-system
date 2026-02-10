#!/usr/bin/env python3
"""
Model Testing Script for AgriSmart Advisor
Tests the model with sample inputs to verify it works correctly
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def test_model():
    """Test the crop recommendation model"""
    
    print("="*60)
    print("  Model Testing Script")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    model_paths = [
        Path('Models/crop_recomender.joblib'),
        Path('crop_recomender.joblib')
    ]
    
    model = None
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            model = joblib.load(path)
            print(f"✅ Model loaded from: {path}")
            break
    
    if model is None:
        print("❌ Model file not found!")
        return False
    
    # Check model type and attributes
    print(f"\n2. Model Information:")
    print(f"   Type: {type(model).__name__}")
    print(f"   Has feature_names_in_: {hasattr(model, 'feature_names_in_')}")
    print(f"   Has get_booster: {hasattr(model, 'get_booster')}")
    
    # Try to get feature names
    print(f"\n3. Checking feature names...")
    features = None
    
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
        print(f"✅ Found {len(features)} features via feature_names_in_")
    elif hasattr(model, 'get_booster'):
        try:
            booster = model.get_booster()
            if hasattr(booster, 'feature_names'):
                features = booster.feature_names
                print(f"✅ Found {len(features)} features via get_booster().feature_names")
            elif hasattr(booster, 'feature_name'):
                features = booster.feature_name
                print(f"✅ Found {len(features)} features via get_booster().feature_name")
        except Exception as e:
            print(f"⚠️  Could not get features from booster: {e}")
    
    if features:
        print(f"\n   First 10 features: {features[:10]}")
        print(f"   Last 10 features: {features[-10:]}")
    else:
        print("⚠️  Model does not preserve feature names")
        print("   This is OK - we'll construct features manually")
    
    # Load training data if available
    print(f"\n4. Checking for training data...")
    training_data = None
    data_paths = [
        Path('Data/Crop_data.csv'),  # Data folder (primary)
        Path('Models/Crop_data.csv'),  # Models folder
        Path('Crop_data.csv')  # Current directory
    ]
    
    for path in data_paths:
        if path.exists():
            training_data = pd.read_csv(path)
            print(f"✅ Training data loaded from: {path}")
            print(f"   Shape: {training_data.shape}")
            print(f"   Columns: {list(training_data.columns)}")
            break
    
    if training_data is None:
        print("⚠️  Training data not found (optional)")
        print("   Looked in: Data/, Models/, and current directory")
    
    # Create test input
    print(f"\n5. Creating test input...")
    
    # All states (sorted alphabetically as get_dummies would do)
    all_states = sorted([
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
        'Jammu And Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
        'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
        'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab',
        'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
        'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
    ])
    
    # All seasons (sorted alphabetically)
    all_seasons = sorted(['Kharif', 'Rabi', 'Winter', 'Summer', 'Autumn', 'Whole Year'])
    
    # Test with West Bengal, Winter season
    test_state = 'West Bengal'
    test_season = 'Winter'
    
    # Create features in order
    test_features = {}
    
    # Numerical features FIRST
    test_features['Area'] = 100.0
    test_features['Annual_Rainfall'] = 150.0
    test_features['Avg_Temperature'] = 25.0
    test_features['humidity_pct'] = 60.0
    test_features['soil_N_kg_sq_foot'] = 0.0013  # 140 kg/ha converted
    test_features['soil_P_kg_sq_foot'] = 0.00037  # 40 kg/ha converted
    test_features['soil_K_kg_sq_foot'] = 0.00037  # 40 kg/ha converted
    test_features['soil_pH'] = 7.2
    
    # Season one-hot encoding (alphabetically)
    for season in all_seasons:
        test_features[f'Season_{season}'] = 1 if season == test_season else 0
    
    # State one-hot encoding (alphabetically)
    for state in all_states:
        test_features[f'State_{state}'] = 1 if state == test_state else 0
    
    # Create DataFrame
    test_df = pd.DataFrame([test_features])
    
    print(f"✅ Created test input with {test_df.shape[1]} features")
    print(f"   Test State: {test_state}")
    print(f"   Test Season: {test_season}")
    print(f"   Sample features: {list(test_df.columns[:5])} ... {list(test_df.columns[-5:])}")
    
    # Try prediction
    print(f"\n6. Testing prediction...")
    try:
        prediction = model.predict(test_df)
        print(f"✅ Prediction successful!")
        print(f"   Recommended crop: {prediction[0]}")
        
        # Try probability if available
        try:
            probabilities = model.predict_proba(test_df)[0]
            confidence = max(probabilities) * 100
            print(f"   Confidence: {confidence:.1f}%")
        except:
            print(f"   (Probability not available)")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed!")
        print(f"   Error: {str(e)}")
        print(f"\n   Model might expect different features.")
        print(f"   Try adding Crop_data.csv to the Models/ folder.")
        return False

if __name__ == "__main__":
    success = test_model()
    
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed! Model is working correctly.")
        print("\nYou can now run the Streamlit app:")
        print("   streamlit run app.py")
    else:
        print("❌ Tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure crop_recomender.joblib is in Models/ folder")
        print("2. Add Crop_data.csv to Models/ folder (optional but recommended)")
        print("3. Verify the model was trained correctly")
    print("="*60)
    
    sys.exit(0 if success else 1)
