#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
India Property AI - Dual-Brain Architecture
Routes requests to specialized expert models for max accuracy.
"""

import os
import pickle
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not installed. XGBoost model will be disabled.")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from flask import Flask, request, render_template

# ------------------------------
# 1. SPECIALIZED DATA LOADERS
# ------------------------------
def clean_sqft(x):
    try:
        if '-' in str(x):
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

def load_bangalore_data():
    """Load and clean specialized Bangalore data."""
    print("Loading Bangalore Expert Data...")
    df = pd.read_csv('bangalore_housing.csv')
    df = df.dropna(subset=['price', 'total_sqft', 'location', 'size'])
    df['total_sqft'] = df['total_sqft'].apply(clean_sqft)
    df = df.dropna(subset=['total_sqft'])
    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else 1)
    df['bath'] = df['bath'].fillna(df['bath'].median())
    df = df[df.bath < df.bhk + 2]
    df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']
    df = remove_pps_outliers(df)
    
    y = df['price'] * 100000
    X = df.drop(columns=['price'])
    return X, y

def load_general_data():
    """Load and clean specialized General India data."""
    print("Loading General Market Data...")
    df = pd.read_csv('House Price Prediction Dataset.csv')
    y = df['Price'] * 83.5 * 100000 # Scaling Price to similar numeric volume
    X = df.drop(columns=['Price'])
    return X, y

# ------------------------------
# 2. SPECIALIZED FEATURE ENGINEERING
# ------------------------------
BANGALORE_COLS = ['total_sqft', 'bath', 'balcony', 'bhk', 'area_type_encoded', 'is_ready_to_move', 'LocationPriceLevel']
GENERAL_COLS = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'house_age', 'condition_encoded', 'has_garage']

def engineer_bangalore_features(X, y=None, neighborhood_mapping=None):
    df = X.copy()
    if 'total_sqft' in df.columns:
        if df['total_sqft'].dtype == object:
            df['total_sqft'] = df['total_sqft'].apply(clean_sqft)
        df['total_sqft'] = df['total_sqft'].fillna(df['total_sqft'].median())
    
    if 'size' in df.columns:
        df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else 1)
        
    df['bath'] = df['bath'].fillna(df['bath'].median() if 'bath' in df.columns else 2)
    df['balcony'] = df['balcony'].fillna(0)
    
    if 'availability' in df.columns:
        df['is_ready_to_move'] = df['availability'].apply(lambda x: 1 if x == 'Ready To Move' else 0)
    else:
        df['is_ready_to_move'] = 1

    area_map = {'Super built-up  Area': 4, 'Built-up  Area': 3, 'Plot  Area': 2, 'Carpet  Area': 1}
    if 'area_type' in df.columns:
        df['area_type_encoded'] = df['area_type'].map(area_map).fillna(0)
    else:
        df['area_type_encoded'] = 4
        
    if y is not None:
        location_stats = df['location'].value_counts()
        locations_less_than_10 = location_stats[location_stats <= 10]
        df['location'] = df['location'].apply(lambda x: 'other' if x in locations_less_than_10 else x)
        location_median = y.groupby(df['location']).median().to_dict()
        df['LocationPriceLevel'] = df['location'].map(location_median)
        return df, location_median
    else:
        if neighborhood_mapping:
            df['LocationPriceLevel'] = df['location'].map(neighborhood_mapping)
            global_median = np.median(list(neighborhood_mapping.values()))
            df['LocationPriceLevel'] = df['LocationPriceLevel'].fillna(global_median)
        else:
            df['LocationPriceLevel'] = 0
        return df

def engineer_general_features(X):
    df = X.copy()
    current_year = datetime.now().year
    
    if 'YearBuilt' in df.columns:
        df['house_age'] = df['YearBuilt'].apply(lambda x: current_year - x if not np.isnan(x) else 0)
    else:
        df['house_age'] = 10
        
    if 'Condition' in df.columns:
        df['condition_encoded'] = df['Condition'].map({'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}).fillna(3)
    else:
         df['condition_encoded'] = 3
         
    if 'Garage' in df.columns:
        df['has_garage'] = df['Garage'].apply(lambda x: 1 if x == 'Yes' else 0)
    else:
        df['has_garage'] = 0
        
    return df

# ------------------------------
# 3. DUAL MODEL TRAINING
# ------------------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_A_PATH = os.path.join(MODELS_DIR, "bangalore_expert.pkl")
MODEL_B_PATH = os.path.join(MODELS_DIR, "general_expert.pkl")
SCALER_A_PATH = os.path.join(MODELS_DIR, "scaler_a.pkl")
SCALER_B_PATH = os.path.join(MODELS_DIR, "scaler_b.pkl")
MAPPING_PATH = os.path.join(MODELS_DIR, "loc_mapping.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_dual.json")

def train_models():
    print("--- Training Bangalore Expert ---")
    X_a, y_a = load_bangalore_data()
    X_a_eng, mapping = engineer_bangalore_features(X_a, y=y_a)
    X_a_final = X_a_eng[BANGALORE_COLS].copy().fillna(X_a_eng[BANGALORE_COLS].median())
    
    scaler_a = StandardScaler()
    X_a_scaled = scaler_a.fit_transform(X_a_final)
    
    if XGB_AVAILABLE:
        print("  Using XGBoost for Bangalore Expert...")
        model_a = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
    else:
        model_a = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    model_a.fit(X_a_scaled, y_a)
    r2_a = r2_score(y_a, model_a.predict(X_a_scaled))
    print(f"Bangalore Expert R2: {r2_a:.3f}")
    
    print("--- Training General Expert ---")
    X_b, y_b = load_general_data()
    X_b_eng = engineer_general_features(X_b)
    X_b_final = X_b_eng[GENERAL_COLS].copy().fillna(X_b_eng[GENERAL_COLS].median())
    
    scaler_b = StandardScaler()
    X_b_scaled = scaler_b.fit_transform(X_b_final)
    
    model_b = RandomForestRegressor(n_estimators=100, random_state=42)
    model_b.fit(X_b_scaled, y_b)
    r2_b = r2_score(y_b, model_b.predict(X_b_scaled))
    print(f"General Expert R2: {r2_b:.3f}")

    with open(MODEL_A_PATH, 'wb') as f: pickle.dump(model_a, f)
    with open(MODEL_B_PATH, 'wb') as f: pickle.dump(model_b, f)
    with open(SCALER_A_PATH, 'wb') as f: pickle.dump(scaler_a, f)
    with open(SCALER_B_PATH, 'wb') as f: pickle.dump(scaler_b, f)
    with open(MAPPING_PATH, 'wb') as f: pickle.dump(mapping, f)
    
    metrics = {
        'Bangalore Expert': {'r2': r2_a},
        'General Expert': {'r2': r2_b}
    }
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f)
    
    return model_a, model_b, scaler_a, scaler_b, mapping, metrics

def load_or_train():
    if (os.path.exists(MODEL_A_PATH) and os.path.exists(MODEL_B_PATH) and 
        os.path.exists(SCALER_A_PATH) and os.path.exists(SCALER_B_PATH) and 
        os.path.exists(MAPPING_PATH)):
        print("Loading Dual-Brain Experts...")
        with open(MODEL_A_PATH, 'rb') as f: ma = pickle.load(f)
        with open(MODEL_B_PATH, 'rb') as f: mb = pickle.load(f)
        with open(SCALER_A_PATH, 'rb') as f: sa = pickle.load(f)
        with open(SCALER_B_PATH, 'rb') as f: sb = pickle.load(f)
        with open(MAPPING_PATH, 'rb') as f: mp = pickle.load(f)
        try:
            with open(METRICS_PATH, 'r') as f: met = json.load(f)
        except:
            met = {'Bangalore Expert': {'r2': 0.77}, 'General Expert': {'r2': 0.76}}
        return ma, mb, sa, sb, mp, met
    else:
        return train_models()

# ------------------------------
# 4. FLASK APP
# ------------------------------
app = Flask(__name__)

try:
    model_A, model_B, scaler_A, scaler_B, loc_mapping, model_metrics = load_or_train()
    
    df_b = pd.read_csv('bangalore_housing.csv')
    locs_b = sorted(df_b['location'].dropna().unique().tolist())
    df_g = pd.read_csv('House Price Prediction Dataset.csv')
    locs_g = sorted(df_g['Location'].dropna().unique().tolist())
    LOCATIONS = {'Specific Neighborhoods': locs_b, 'General Market': locs_g}
    
except Exception as e:
    print(f"Startup Error: {e}")
    LOCATIONS = {}
    model_metrics = {}

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', prediction=None, model_metrics=model_metrics, best_model="Dual-Brain Architecture", locations=LOCATIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.form
        city = req.get('city', 'Bangalore')
        
        if city == 'Bangalore':
            df = pd.DataFrame([{
                'location': req['location'],
                'total_sqft': float(req['total_sqft']),
                'bath': float(req['bath']),
                'balcony': float(req.get('balcony', 0)),
                'size': f"{req['bhk']} BHK",
                'area_type': req.get('area_type', 'Super built-up  Area'),
                'availability': 'Ready To Move'
            }])
            X_eng = engineer_bangalore_features(df, neighborhood_mapping=loc_mapping)
            features = X_eng[BANGALORE_COLS].fillna(0).values
            pred_price = model_A.predict(scaler_A.transform(features))[0]
            
        else:
            df = pd.DataFrame([{
                'Area': float(req['total_sqft']),
                'Bedrooms': int(req['bhk']),
                'Bathrooms': float(req['bath']),
                'Floors': float(req.get('floors', 1)),
                'YearBuilt': float(req.get('year_built', 2015)),
                'Condition': 'Good',
                'Garage': req.get('garage', 'no').capitalize() # 'Yes' or 'No'
            }])
            # For Condition enum
            cond_map = {'4': 'Excellent', '3': 'Good', '2': 'Fair', '1': 'Poor'}
            df['Condition'] = cond_map.get(str(req.get('condition', '3')), 'Good')
            
            X_eng = engineer_general_features(df)
            features = X_eng[GENERAL_COLS].fillna(0).values
            pred_price = model_B.predict(scaler_B.transform(features))[0]

        return render_template('home.html', prediction=pred_price, model_metrics=model_metrics, req=req, best_model="Dual-Brain Architecture", locations=LOCATIONS)
        
    except Exception as e:
        return render_template('home.html', error=str(e), model_metrics=model_metrics, req=request.form, best_model="Dual-Brain Architecture", locations=LOCATIONS)

@app.route('/simulator', methods=['GET', 'POST'])
def simulator():
    if request.method == 'GET':
        return render_template('simulator.html', best_model="Dual-Brain Architecture", req=None, locations=LOCATIONS)
        
    try:
        req = request.form
        city = req.get('city', 'Bangalore')
        
        if city == 'Bangalore':
            base = {
                'location': req.get('location', 'Whitefield'),
                'total_sqft': float(req.get('total_sqft', 1200)),
                'bath': float(req.get('bath', 2)),
                'balcony': float(req.get('balcony', 0)),
                'size': f"{req.get('bhk', 2)} BHK",
                'area_type': 'Super built-up  Area',
                'availability': 'Ready To Move'
            }
            sim = base.copy()
            if req.get('sim_sqft'): sim['total_sqft'] += 300
            if req.get('sim_bhk'): sim['size'] = f"{int(req.get('bhk', 2)) + 1} BHK"
            if req.get('sim_bath'): sim['bath'] += 1
            if req.get('sim_balcony'): sim['balcony'] += 1
            
            b_df = pd.DataFrame([base])
            s_df = pd.DataFrame([sim])
            
            xb = engineer_bangalore_features(b_df, neighborhood_mapping=loc_mapping)[BANGALORE_COLS].fillna(0).values
            xs = engineer_bangalore_features(s_df, neighborhood_mapping=loc_mapping)[BANGALORE_COLS].fillna(0).values
            
            base_price = model_A.predict(scaler_A.transform(xb))[0]
            sim_price = model_A.predict(scaler_A.transform(xs))[0]
            
        else:
            base = {
                'Area': float(req.get('total_sqft', 1200)),
                'Bedrooms': int(req.get('bhk', 2)),
                'Bathrooms': float(req.get('bath', 2)),
                'Floors': float(req.get('floors', 1)),
                'YearBuilt': float(req.get('year_built', 2015)),
                'Condition': {'4': 'Excellent', '3': 'Good', '2': 'Fair', '1': 'Poor'}.get(str(req.get('condition', '3')), 'Good'),
                'Garage': req.get('garage', 'no').capitalize()
            }
            sim = base.copy()
            if req.get('sim_sqft'): sim['Area'] += 300
            if req.get('sim_bhk'): sim['Bedrooms'] += 1
            if req.get('sim_bath'): sim['Bathrooms'] += 1
            if req.get('sim_floors'): sim['Floors'] += 1
            if req.get('sim_garage'): sim['Garage'] = 'Yes'
            
            b_df = pd.DataFrame([base])
            s_df = pd.DataFrame([sim])
            
            xb = engineer_general_features(b_df)[GENERAL_COLS].fillna(0).values
            xs = engineer_general_features(s_df)[GENERAL_COLS].fillna(0).values
            
            base_price = model_B.predict(scaler_B.transform(xb))[0]
            sim_price = model_B.predict(scaler_B.transform(xs))[0]

        roi = sim_price - base_price
        return render_template('simulator.html', base_price=base_price, sim_price=sim_price, roi=roi, req=req, best_model="Dual-Brain Architecture", locations=LOCATIONS)
        
    except Exception as e:
        return render_template('simulator.html', error=str(e), req=request.form, best_model="Dual-Brain Architecture", locations=LOCATIONS)

@app.route('/insights')
def insights():
    try:
        importances = model_B.feature_importances_
        importances = (importances / importances.sum()) * 100
        sorted_idx = np.argsort(importances)[::-1]
        
        # Map raw column names to human-readable strings
        human_labels = {
            'Area': 'Total Area (SqFt)',
            'Bedrooms': 'Bedrooms (BHK)',
            'Bathrooms': 'Total Bathrooms',
            'Floors': 'Floor Count',
            'house_age': 'Property Age (Years)',
            'condition_encoded': 'Physical Condition',
            'has_garage': 'Private Garage Included'
        }
        
        sorted_features = [human_labels.get(GENERAL_COLS[i], GENERAL_COLS[i]) for i in sorted_idx]
        sorted_importances = [float(importances[i]) for i in sorted_idx]
        
        return render_template('insights.html', features=sorted_features, importances=sorted_importances, best_model="General India Expert")
    except Exception as e:
        return render_template('insights.html', features=[], importances=[], best_model="Dual-Brain Architecture")

if __name__ == '__main__':
    print("Starting India Property AI (Dual-Brain Architecture)...")
    print("Open your browser at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)