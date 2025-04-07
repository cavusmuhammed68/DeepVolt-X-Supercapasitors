# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:06:41 2025

@author: nfpm5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
file_path = r'C:\Users\nfpm5\Downloads\Safa & Yusuf_updated\Electrode_data_new.xlsx'
sheet_name = 'CC-Negative electrode'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Rename columns for better access
df.columns = [
    "Potential_2", "Current_2", 
    "Potential_5", "Current_5", 
    "Potential_10", "Current_10",
    "Potential_20", "Current_20",
    "Potential_50", "Current_50"
]

# Drop the first row containing header descriptions
df_clean = df[1:].apply(pd.to_numeric, errors='coerce')

# Define the save directory
save_dir = r"C:\Users\nfpm5\Downloads\Safa & Yusuf_updated\Results"
os.makedirs(save_dir, exist_ok=True)

# Initialize a figure
plt.figure(figsize=(10, 6))

# Process each scanning speed and plot on the same figure
results = {}
scanning_speeds = [2, 5, 10, 20, 50]

for speed in scanning_speeds:
    potential_col = f"Potential_{speed}"
    current_col = f"Current_{speed}"

    # Filter data based on the given range
    valid_data = df_clean[(df_clean[potential_col] >= -1.0) & (df_clean[potential_col] <= 0.2)]
    valid_data = valid_data[(valid_data[current_col] >= -0.009) & (valid_data[current_col] <= 0.007)]

    # Integrate the absolute area under the curve
    Q = np.trapz(abs(valid_data[current_col]), valid_data[potential_col])
    results[speed] = Q

    # Plot the curve
    plt.plot(
        valid_data[potential_col],
        valid_data[current_col],
        label=f"{speed} mV/s"
    )

# Finalize the combined figure
plt.xlabel("Potential (V)")
plt.ylabel("Current (A)")
plt.title("CV Curves for Different Scanning Speeds")
plt.legend(title="Scan Rate")
plt.grid()
plt.xlim(-1.0, 0)  # Set x-axis limits
plt.ylim(-0.008, 0.006)  # Set y-axis limits

# Save the combined figure
combined_plot_path = os.path.join(save_dir, "Combined_CV_Curves_CC-Negative electrode.png")
plt.savefig(combined_plot_path)
plt.close()

# Save Q values to a CSV file
results_df = pd.DataFrame.from_dict(results, orient='index', columns=["Q (Area)"])
results_df_path = os.path.join(save_dir, "Results_Q.csv")
results_df.to_csv(results_df_path, index_label="Scan Rate (mV/s)")

print(f"Combined figure saved at: {combined_plot_path}")
print(f"Results saved in: {results_df_path}")
