# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:05:40 2025

@author: nfpm5
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
import xgboost as xgb

# Define constants
OUTPUT_PATH = r'C:\Users\nfpm5\Downloads\Safa & Yusuf_updated\Results'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load the dataset
file_path = r'C:\Users\nfpm5\Downloads\Safa & Yusuf_updated\Electrode_data_new.xlsx'
sheet_name = 'CC-Negative electrode'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Define scan rates to process
scan_rates = [2, 5, 10, 20, 50]

# Initialize storage for all data
all_data = {}

for rate in scan_rates:
    potential_col = f'for {rate} mV/s'
    current_col = f'Unnamed: {2 * (scan_rates.index(rate)) + 1}'

    potential = pd.to_numeric(data[potential_col], errors='coerce')
    current = pd.to_numeric(data[current_col], errors='coerce')
    valid_data = ~potential.isna() & ~current.isna()
    potential = potential[valid_data]
    current = current[valid_data]

    filtered_data = pd.DataFrame({'Potential (V)': potential, 'Current (A)': current})
    filtered_data = filtered_data[(filtered_data['Potential (V)'] >= -1.0) & (filtered_data['Potential (V)'] <= 0.0) &
                                  (filtered_data['Current (A)'] >= -0.008) & (filtered_data['Current (A)'] <= 0.004)]
    
    X = filtered_data[['Potential (V)']]
    y = filtered_data['Current (A)']
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    all_data[rate] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

print("Data preparation complete.")

# Train and evaluate models
rf_results = {}
xgb_results = {}
dl_results = {}

for rate, data in all_data.items():
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    
    # Train Deep Learning Model
    dl_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    dl_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    dl_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    dl_predictions = dl_model.predict(X_test).flatten()
    
    # Sort data for plotting
    sorted_indices = X_test[:, 0].argsort()
    X_test_sorted = X_test[sorted_indices, 0]
    y_test_sorted = y_test.iloc[sorted_indices]
    rf_predictions_sorted = rf_predictions[sorted_indices]
    xgb_predictions_sorted = xgb_predictions[sorted_indices]
    dl_predictions_sorted = dl_predictions[sorted_indices]
    
    # Compute metrics
    rf_results[rate] = {'mse': mean_squared_error(y_test, rf_predictions), 'mae': mean_absolute_error(y_test, rf_predictions), 'r2': r2_score(y_test, rf_predictions)}
    
    xgb_results[rate] = {'mse': mean_squared_error(y_test, xgb_predictions), 'mae': mean_absolute_error(y_test, xgb_predictions), 'r2': r2_score(y_test, xgb_predictions)}
    
    dl_results[rate] = {'mse': mean_squared_error(y_test, dl_predictions), 'mae': mean_absolute_error(y_test, dl_predictions), 'r2': r2_score(y_test, dl_predictions)}

# Print model evaluation metrics
print("\nModel Evaluation Metrics:")
print("Scan Rate (mV/s) | Model  | MSE        | MAE        | R²")
print("-----------------------------------------------------------")
for rate in scan_rates:
    print(f"{rate:<17} | RF    | {rf_results[rate]['mse']:.6f} | {rf_results[rate]['mae']:.6f} | {rf_results[rate]['r2']:.6f}")
    print(f"{rate:<17} | XGB   | {xgb_results[rate]['mse']:.6f} | {xgb_results[rate]['mae']:.6f} | {xgb_results[rate]['r2']:.6f}")
    print(f"{rate:<17} | DL    | {dl_results[rate]['mse']:.6f} | {dl_results[rate]['mae']:.6f} | {dl_results[rate]['r2']:.6f}")
    print("-----------------------------------------------------------")

# Adjust font size of plots
plt.rc('axes', labelsize=14)  # Set x and y labels font size
plt.rc('legend', fontsize=14)  # Set legend font size
