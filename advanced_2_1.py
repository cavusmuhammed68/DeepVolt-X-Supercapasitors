# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:08:45 2025

@author: nfpm5
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:03:09 2025

@author: cavus
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define constants
OUTPUT_PATH = r'C:\Users\nfpm5\Downloads\Safa & Yusuf_updated\\Results'

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load the dataset
file_path = r'C:\Users\nfpm5\Downloads\Safa & Yusuf_updated\Electrode_data_new.xlsx'
sheet_name = 'CC-Negative electrode'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Define scan rates to process
scan_rates = [2, 5, 10, 20, 50]

# Loop through each scan rate
for rate in scan_rates:
    # Columns for potential and current
    potential_col = f'for {rate} mV/s'
    current_col = f'Unnamed: {2 * (scan_rates.index(rate)) + 1}'
    
    # Convert to numeric and drop NaN
    potential = pd.to_numeric(data[potential_col], errors='coerce')
    current = pd.to_numeric(data[current_col], errors='coerce')
    valid_data = ~potential.isna() & ~current.isna()
    potential = potential[valid_data]
    current = current[valid_data]
    
    # Filter by the specified ranges
    filtered_data = pd.DataFrame({
        'Potential (V)': potential,
        'Current (A)': current
    })
    filtered_data = filtered_data[(filtered_data['Potential (V)'] >= -1.0) & (filtered_data['Potential (V)'] <= 0.0) & 
                                  (filtered_data['Current (A)'] >= -0.008) & (filtered_data['Current (A)'] <= 0.004)]
    
    # Prepare features and target
    X = filtered_data[['Potential (V)']]
    y = filtered_data['Current (A)']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predict using the Random Forest model
    rf_predictions = rf_model.predict(X_test)
    
    # Sort data for smoother line plots
    sorted_indices = X_test.squeeze().argsort()
    X_test_sorted = X_test.iloc[sorted_indices]
    y_test_sorted = y_test.iloc[sorted_indices]
    rf_predictions_sorted = rf_predictions[sorted_indices]
    
    # Plot the comparison of actual and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(X_test_sorted, y_test_sorted, label='Actual Data', linestyle='-', linewidth=2)
    plt.plot(X_test_sorted, rf_predictions_sorted, label='Random Forest Prediction', linestyle='--', linewidth=2)
    plt.xlabel('Potential (V)')
    plt.ylabel('Current (A)')
    plt.title(f'Actual vs Predicted Current Values (Scan Rate: {rate} mV/s)')
    plt.legend()
    plt.grid()
    
    # Save the Random Forest model results and plot
    plot_file = os.path.join(OUTPUT_PATH, f'RF_Actual_vs_Predicted_{rate}mVs.png')
    plt.savefig(plot_file)
    plt.show()
    
    # Save predictions to a CSV file for further analysis
    predictions_df = pd.DataFrame({
        'Potential (V)': X_test_sorted.squeeze(),
        'Actual Current (A)': y_test_sorted,
        'Predicted Current (A)': rf_predictions_sorted
    })
    predictions_file = os.path.join(OUTPUT_PATH, f'RF_Predictions_{rate}mVs.csv')
    predictions_df.to_csv(predictions_file, index=False)
    
    print(f"Scan Rate: {rate} mV/s")
    print(f"MSE: {mean_squared_error(y_test_sorted, rf_predictions_sorted):.4f}, "
          f"R2: {r2_score(y_test_sorted, rf_predictions_sorted):.4f}")
    print(f"Results saved to:\n{predictions_file}\n{plot_file}\n")
