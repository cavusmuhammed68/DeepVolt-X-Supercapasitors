# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:49:33 2025

@author: nfpm5
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:28:48 2025

@author: nfpm5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
file_path = r'C:\Users\nfpm5\Downloads\Safa & Yusuf_updated\Electrode_data_new.xlsx'
sheet_name = 'MMO-Positive electrode'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Extract column names by merging the first two rows
scan_rates = [2, 5, 10, 20, 50]
columns = []
for rate in scan_rates:
    columns.append(f"Potential_{rate}")
    columns.append(f"Current_{rate}")

# Assign new column names and clean the dataset
df.columns = columns
df_clean = df[2:].apply(pd.to_numeric, errors='coerce')

# Define the save directory
save_dir = r"C:\Users\nfpm5\Downloads\Safa & Yusuf_updated\Results"
os.makedirs(save_dir, exist_ok=True)

# Initialize a figure
plt.figure(figsize=(8, 4))
colors = ['black', 'red', 'blue', 'magenta', 'navy']

# Process each scanning speed and plot on the same figure
results = {}
for speed, color in zip(scan_rates, colors):
    potential_col = f"Potential_{speed}"
    current_col = f"Current_{speed}"
    
    # Filter data within the valid range
    valid_data = df_clean[(df_clean[potential_col] >= 0) & (df_clean[potential_col] <= 0.4)]
    valid_data = valid_data[(valid_data[current_col] >= -0.06) & (valid_data[current_col] <= 0.06)]

    # Integrate the absolute area under the curve
    Q = np.trapz(abs(valid_data[current_col]), valid_data[potential_col])
    results[speed] = Q

    # Plot the curve
    plt.plot(valid_data[potential_col], valid_data[current_col], label=f"{speed} mV/s", color=color, linewidth=2)

# Finalize the combined figure
plt.xlabel("Potential (V vs. Ag/AgCl)", fontsize=12)
plt.ylabel("Current (A)", fontsize=12)
#plt.title("ACF", fontsize=14, fontweight='bold')
plt.legend(title=None, fontsize=10)
plt.grid(True)
plt.xlim(0, 0.4)  # Set x-axis limits
plt.ylim(-0.06, 0.06)  # Set y-axis limits

# Save the combined figure
combined_plot_path = os.path.join(save_dir, "CV_Curves_MMO.png")
plt.savefig(combined_plot_path, dpi=600)
plt.close()

# Save Q values to a CSV file
results_df = pd.DataFrame.from_dict(results, orient='index', columns=["Q (Area)"])
results_df_path = os.path.join(save_dir, "Results_Q.csv")
results_df.to_csv(results_df_path, index_label="Scan Rate (mV/s)")

print(f"Combined figure saved at: {combined_plot_path}")
print(f"Results saved in: {results_df_path}")