# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:03:33 2025

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
OUTPUT_PATH = r'C:\Users\nfpm5\Downloads\Safa & Yusuf_updated_2\Results'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load the dataset
file_path = r'C:\Users\nfpm5\Downloads\Safa & Yusuf_updated_2\Electrode_data_new_V10.xlsx'
sheet_name = 'MMO-Positive electrode'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Define scan rates to process
scan_rates = [2, 5, 20, 50]

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

    df = pd.DataFrame({'Potential (V)': potential, 'Current (A)': current})
    
    X = df[['Potential (V)']]
    y = df['Current (A)']
    
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
    rf_results[rate] = {'sorted_X_test': X_test_sorted, 'sorted_y_test': y_test_sorted, 'sorted_predictions': rf_predictions_sorted,
                         'mse': mean_squared_error(y_test, rf_predictions), 'mae': mean_absolute_error(y_test, rf_predictions), 'r2': r2_score(y_test, rf_predictions)}
    
    xgb_results[rate] = {'sorted_X_test': X_test_sorted, 'sorted_y_test': y_test_sorted, 'sorted_predictions': xgb_predictions_sorted,
                          'mse': mean_squared_error(y_test, xgb_predictions), 'mae': mean_absolute_error(y_test, xgb_predictions), 'r2': r2_score(y_test, xgb_predictions)}
    
    dl_results[rate] = {'sorted_X_test': X_test_sorted, 'sorted_y_test': y_test_sorted, 'sorted_predictions': dl_predictions_sorted,
                         'mse': mean_squared_error(y_test, dl_predictions), 'mae': mean_absolute_error(y_test, dl_predictions), 'r2': r2_score(y_test, dl_predictions)}

# Plot and compare models
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

for idx, rate in enumerate(scan_rates):
    X_test_sorted = rf_results[rate]['sorted_X_test']
    y_test_sorted = rf_results[rate]['sorted_y_test']
    rf_predictions_sorted = rf_results[rate]['sorted_predictions']
    xgb_predictions_sorted = xgb_results[rate]['sorted_predictions']
    dl_predictions_sorted = dl_results[rate]['sorted_predictions']
    
    ax = axes[idx]
    ax.scatter(X_test_sorted, y_test_sorted, label='Actual Data', alpha=0.7, s=20)
    ax.scatter(X_test_sorted, rf_predictions_sorted, label='RF Prediction', alpha=0.7, s=20, marker='x')
    ax.scatter(X_test_sorted, xgb_predictions_sorted, label='XGB Prediction', alpha=0.7, s=20, marker='^')
    ax.scatter(X_test_sorted, dl_predictions_sorted, label='DL Prediction', alpha=0.7, s=20, marker='s')
    ax.set_xlabel('Potential (V)', fontsize=18)
    ax.set_ylabel('Current (A)', fontsize=18)
    
    ax.set_title(f'{rate} mV/s Scan Rate')
    ax.legend()
    ax.grid()

fig.delaxes(axes[-1])
fig.tight_layout()
comparison_plot_path = os.path.join(OUTPUT_PATH, 'Comparison_RF_XGB_DL_All.png')
plt.savefig(comparison_plot_path, dpi=600)
plt.show()

print(f"Combined comparison plot saved at: {comparison_plot_path}")

# Plot and save individual figures
for rate in scan_rates:
    plt.figure(figsize=(8, 6))  # Adjust figure size for clarity
    
    X_test_sorted = rf_results[rate]['sorted_X_test']
    y_test_sorted = rf_results[rate]['sorted_y_test']
    rf_predictions_sorted = rf_results[rate]['sorted_predictions']
    xgb_predictions_sorted = xgb_results[rate]['sorted_predictions']
    dl_predictions_sorted = dl_results[rate]['sorted_predictions']
    
    plt.scatter(X_test_sorted, y_test_sorted, label='DeepVolt-X', alpha=0.7, s=20)
    plt.scatter(X_test_sorted, rf_predictions_sorted, label='RF', alpha=0.7, s=20, marker='x')
    plt.scatter(X_test_sorted, xgb_predictions_sorted, label='XGBoost', alpha=0.7, s=20, marker='^')
    plt.scatter(X_test_sorted, dl_predictions_sorted, label='LR', alpha=0.7, s=20, marker='s')

    plt.xlabel('Potential (V)', fontsize=16)
    plt.ylabel('Current (A)', fontsize=16)
    plt.title(f'{rate} mV/s Scan Rate', fontsize=16)
    plt.legend()
    plt.grid()

    figure_path = os.path.join(OUTPUT_PATH, f'Comparison_RF_XGB_DL_{rate}mVs.png')
    plt.savefig(figure_path, dpi=600)
    plt.show()

    print(f"Figure saved at: {figure_path}")
