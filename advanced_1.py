# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:55:55 2025

@author: nfpm5
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 22:29:03 2025

@author: cavus
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor

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

# Initialize storage for all data
all_data = {}

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

    # Add polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Store processed data for this rate
    all_data[rate] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

print("Data preparation complete.")

# Part 2: Train and evaluate Random Forest and XGBoost
rf_results = {}
xgb_results = {}

for rate, data in all_data.items():
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    # Train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Train the XGBoost model
    xgb_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    # Sort data for smooth plotting
    sorted_indices = X_test[:, 0].argsort()
    X_test_sorted = X_test[sorted_indices, 0]
    y_test_sorted = y_test.iloc[sorted_indices]
    rf_predictions_sorted = rf_predictions[sorted_indices]
    xgb_predictions_sorted = xgb_predictions[sorted_indices]

    # Calculate metrics for Random Forest
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)

    # Calculate metrics for XGBoost
    xgb_mse = mean_squared_error(y_test, xgb_predictions)
    xgb_r2 = r2_score(y_test, xgb_predictions)

    # Store the results
    rf_results[rate] = {
        'model': rf_model,
        'predictions': rf_predictions,
        'sorted_predictions': rf_predictions_sorted,
        'sorted_X_test': X_test_sorted,
        'sorted_y_test': y_test_sorted,
        'mse': rf_mse,
        'r2': rf_r2
    }

    xgb_results[rate] = {
        'model': xgb_model,
        'predictions': xgb_predictions,
        'sorted_predictions': xgb_predictions_sorted,
        'sorted_X_test': X_test_sorted,
        'sorted_y_test': y_test_sorted,
        'mse': xgb_mse,
        'r2': xgb_r2
    }

    print(f"Scan Rate: {rate} mV/s - Random Forest MSE: {rf_mse:.6f}, R2: {rf_r2:.6f}")
    print(f"Scan Rate: {rate} mV/s - XGBoost MSE: {xgb_mse:.6f}, R2: {xgb_r2:.6f}")

print("Random Forest and XGBoost training and evaluation complete.")

# Part 3: Train and evaluate Deep Learning

dl_results = {}

for rate, data in all_data.items():
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    # Build a Deep Learning model
    dl_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer
    ])

    # Compile the model
    dl_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    dl_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

    # Predict on the test set
    dl_predictions = dl_model.predict(X_test).flatten()

    # Sort data for smooth plotting
    sorted_indices = X_test[:, 0].argsort()
    X_test_sorted = X_test[sorted_indices, 0]
    y_test_sorted = y_test.iloc[sorted_indices]
    dl_predictions_sorted = dl_predictions[sorted_indices]

    # Calculate metrics
    mse = mean_squared_error(y_test, dl_predictions)
    r2 = r2_score(y_test, dl_predictions)

    # Store the results
    dl_results[rate] = {
        'model': dl_model,
        'predictions': dl_predictions,
        'sorted_predictions': dl_predictions_sorted,
        'sorted_X_test': X_test_sorted,
        'sorted_y_test': y_test_sorted,
        'mse': mse,
        'r2': r2
    }

    print(f"Scan Rate: {rate} mV/s - Deep Learning MSE: {mse:.6f}, R2: {r2:.6f}")

print("Deep Learning training and evaluation complete.")

# Part 4: Generate comparative plots

for rate in scan_rates:
    # Retrieve data and predictions
    X_test_sorted = rf_results[rate]['sorted_X_test']
    y_test_sorted = rf_results[rate]['sorted_y_test']
    rf_predictions_sorted = rf_results[rate]['sorted_predictions']
    xgb_predictions_sorted = xgb_results[rate]['sorted_predictions']
    dl_predictions_sorted = dl_results[rate]['sorted_predictions']

    # Create a 2x2 plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Actual Data
    axs[0, 0].plot(X_test_sorted, y_test_sorted, label='Actual Data', linestyle='-', linewidth=2)
    axs[0, 0].set_title('Actual Data')
    axs[0, 0].set_xlabel('Potential (V)')
    axs[0, 0].set_ylabel('Current (A)')
    axs[0, 0].grid()

    # Random Forest Predictions
    axs[0, 1].plot(X_test_sorted, rf_predictions_sorted, label='Random Forest', linestyle='--', linewidth=2)
    axs[0, 1].set_title('Random Forest Predictions')
    axs[0, 1].set_xlabel('Potential (V)')
    axs[0, 1].set_ylabel('Current (A)')
    axs[0, 1].grid()

    # XGBoost Predictions
    axs[1, 0].plot(X_test_sorted, xgb_predictions_sorted, label='XGBoost', linestyle='-.', linewidth=2)
    axs[1, 0].set_title('XGBoost Predictions')
    axs[1, 0].set_xlabel('Potential (V)')
    axs[1, 0].set_ylabel('Current (A)')
    axs[1, 0].grid()

    # Deep Learning Predictions
    axs[1, 1].plot(X_test_sorted, dl_predictions_sorted, label='Deep Learning', linestyle=':', linewidth=2)
    axs[1, 1].set_title('Deep Learning Predictions')
    axs[1, 1].set_xlabel('Potential (V)')
    axs[1, 1].set_ylabel('Current (A)')
    axs[1, 1].grid()

    # Adjust layout and save
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_PATH, f'Comparative_Plot_{rate}mVs.png')
    plt.savefig(plot_file)
    plt.show()

    print(f"Comparative plot saved for scan rate {rate} mV/s: {plot_file}")

# Part 5: Calculate and save overall metrics

# Initialize dictionaries to store overall metrics
overall_metrics = {
    'Random Forest': {'MSE': [], 'R2': []},
    'XGBoost': {'MSE': [], 'R2': []},
    'Deep Learning': {'MSE': [], 'R2': []}
}

# Aggregate metrics for each scan rate
for rate in scan_rates:
    # Retrieve metrics
    rf_mse = rf_results[rate]['mse']
    rf_r2 = rf_results[rate]['r2']
    xgb_mse = xgb_results[rate]['mse']
    xgb_r2 = xgb_results[rate]['r2']
    dl_mse = dl_results[rate]['mse']
    dl_r2 = dl_results[rate]['r2']

    # Store metrics
    overall_metrics['Random Forest']['MSE'].append(rf_mse)
    overall_metrics['Random Forest']['R2'].append(rf_r2)
    overall_metrics['XGBoost']['MSE'].append(xgb_mse)
    overall_metrics['XGBoost']['R2'].append(xgb_r2)
    overall_metrics['Deep Learning']['MSE'].append(dl_mse)
    overall_metrics['Deep Learning']['R2'].append(dl_r2)

# Calculate averages for MSE and R2
rf_avg_mse = np.mean(overall_metrics['Random Forest']['MSE'])
rf_avg_r2 = np.mean(overall_metrics['Random Forest']['R2'])
xgb_avg_mse = np.mean(overall_metrics['XGBoost']['MSE'])
xgb_avg_r2 = np.mean(overall_metrics['XGBoost']['R2'])
dl_avg_mse = np.mean(overall_metrics['Deep Learning']['MSE'])
dl_avg_r2 = np.mean(overall_metrics['Deep Learning']['R2'])

print("Overall Metrics:")
print(f"Random Forest - Average MSE: {rf_avg_mse:.6f}, Average R2: {rf_avg_r2:.6f}")
print(f"XGBoost - Average MSE: {xgb_avg_mse:.6f}, Average R2: {xgb_avg_r2:.6f}")
print(f"Deep Learning - Average MSE: {dl_avg_mse:.6f}, Average R2: {dl_avg_r2:.6f}")

# Save the metrics to a text file
metrics_file = os.path.join(OUTPUT_PATH, 'Overall_Metrics.txt')
with open(metrics_file, 'w') as f:
    f.write("Overall Metrics:\n")
    f.write(f"Random Forest - Average MSE: {rf_avg_mse:.6f}, Average R2: {rf_avg_r2:.6f}\n")
    f.write(f"XGBoost - Average MSE: {xgb_avg_mse:.6f}, Average R2: {xgb_avg_r2:.6f}\n")
    f.write(f"Deep Learning - Average MSE: {dl_avg_mse:.6f}, Average R2: {dl_avg_r2:.6f}\n")

print(f"Overall metrics saved to {metrics_file}")

# Part 6: Fine-tune the Deep Learning model

# Updated deep learning hyperparameters
fine_tuned_results = {}

for rate, data in all_data.items():
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    # Build a new Deep Learning model with fine-tuned hyperparameters
    ft_dl_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),  # Add dropout for regularization
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)  # Output layer
    ])

    # Compile the fine-tuned model
    ft_dl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # Train the fine-tuned model
    ft_dl_model.fit(X_train, y_train, epochs=150, batch_size=16, verbose=0)

    # Predict on the test set
    ft_dl_predictions = ft_dl_model.predict(X_test).flatten()

    # Sort data for smooth plotting
    sorted_indices = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sorted_indices, 0]
    y_test_sorted = y_test.iloc[sorted_indices]
    ft_dl_predictions_sorted = ft_dl_predictions[sorted_indices]

    # Calculate metrics
    mse = mean_squared_error(y_test, ft_dl_predictions)
    r2 = r2_score(y_test, ft_dl_predictions)

    # Store the fine-tuned results
    fine_tuned_results[rate] = {
        'model': ft_dl_model,
        'predictions': ft_dl_predictions,
        'sorted_predictions': ft_dl_predictions_sorted,
        'sorted_X_test': X_test_sorted,
        'sorted_y_test': y_test_sorted,
        'mse': mse,
        'r2': r2
    }

    print(f"Scan Rate: {rate} mV/s - Fine-Tuned DL MSE: {mse:.6f}, R2: {r2:.6f}")

print("Fine-tuned Deep Learning training and evaluation complete.")


# Part 7: Save and compare fine-tuned Deep Learning results

# Save fine-tuned results for each scan rate
for rate in scan_rates:
    # Retrieve data and predictions
    X_test_sorted = fine_tuned_results[rate]['sorted_X_test']
    y_test_sorted = fine_tuned_results[rate]['sorted_y_test']
    ft_dl_predictions_sorted = fine_tuned_results[rate]['sorted_predictions']

    # Create a 2x2 plot for comparative analysis
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Actual Data
    axs[0, 0].plot(X_test_sorted, y_test_sorted, label='Actual Data', linestyle='-', linewidth=2)
    axs[0, 0].set_title('Actual Data')
    axs[0, 0].set_xlabel('Potential (V)')
    axs[0, 0].set_ylabel('Current (A)')
    axs[0, 0].grid()

    # Random Forest Predictions
    axs[0, 1].plot(X_test_sorted, rf_results[rate]['sorted_predictions'], label='Random Forest', linestyle='--', linewidth=2)
    axs[0, 1].set_title('Random Forest Predictions')
    axs[0, 1].set_xlabel('Potential (V)')
    axs[0, 1].set_ylabel('Current (A)')
    axs[0, 1].grid()

    # XGBoost Predictions
    axs[1, 0].plot(X_test_sorted, xgb_results[rate]['sorted_predictions'], label='XGBoost', linestyle='-.', linewidth=2)
    axs[1, 0].set_title('XGBoost Predictions')
    axs[1, 0].set_xlabel('Potential (V)')
    axs[1, 0].set_ylabel('Current (A)')
    axs[1, 0].grid()

    # Fine-Tuned Deep Learning Predictions
    axs[1, 1].plot(X_test_sorted, ft_dl_predictions_sorted, label='Fine-Tuned DL', linestyle=':', linewidth=2)
    axs[1, 1].set_title('Fine-Tuned DL Predictions')
    axs[1, 1].set_xlabel('Potential (V)')
    axs[1, 1].set_ylabel('Current (A)')
    axs[1, 1].grid()

    # Adjust layout and save
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_PATH, f'Fine_Tuned_Comparison_{rate}mVs.png')
    plt.savefig(plot_file)
    plt.show()

    print(f"Fine-tuned comparison plot saved for scan rate {rate} mV/s: {plot_file}")

# Save all fine-tuned metrics
ft_metrics_file = os.path.join(OUTPUT_PATH, 'Fine_Tuned_DL_Metrics.txt')
with open(ft_metrics_file, 'w') as f:
    f.write("Fine-Tuned DL Metrics:\n")
    for rate in scan_rates:
        mse = fine_tuned_results[rate]['mse']
        r2 = fine_tuned_results[rate]['r2']
        f.write(f"Scan Rate {rate} mV/s - MSE: {mse:.6f}, R2: {r2:.6f}\n")

print(f"Fine-tuned DL metrics saved to {ft_metrics_file}")
