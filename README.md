DeepVolt-X: AI-Driven Modelling for Supercapacitor Electrochemical Performance

Overview
This project combines experimental electrochemical data and machine learning (ML) techniques to develop DeepVolt-X, an ensemble learning framework designed to predict current–voltage (CV) responses of asymmetric supercapacitor electrodes across varying scan rates. The system models and compares three different regression approaches: Random Forest (RF), XGBoost, and Deep Learning (DL), evaluating their accuracy and generalisation on experimental datasets derived from MgMn₂O₄ (MMO) and carbon cloth (CC) electrodes.

Features
Preprocessing of Electrochemical Data: Loads CV profiles at multiple scan rates (2–50 mV/s), cleans data, and applies polynomial feature expansion.

Hybrid AI Modelling: Implemented and trained RF, XGBoost, and DL models using TensorFlow/Keras.

Ensemble Prediction Comparison: Evaluates model accuracy using R², MSE, and MAE metrics.

Visual Output: Generates comparative plots showing predicted vs experimental CV profiles for all models.

Folder Structure
bash
Copy
Edit
.
├── advanced_new_NEW.py       # Main Python script for model training and prediction
├── Electrode_data_new_V10.xlsx  # Input Excel file containing CV data
├── /Results                  # Output directory for saved figures
Requirements
Ensure the following Python packages are installed:

bash
Copy
Edit
pip install numpy pandas matplotlib scikit-learn xgboost tensorflow openpyxl
How to Run
Edit file paths: Update file_path and OUTPUT_PATH in the script to match your local environment.

Run the script: Execute advanced_new_NEW.py to train models and generate output plots.

Results: Check the /Results folder for the comparison figures and accuracy metrics.

Output
Model performance comparison plots for each scan rate.

Individual prediction plots showing DL, RF, and XGBoost outputs.

Final combined plot for all scan rates.

Purpose
This tool supports the AI-guided design of supercapacitor electrodes by enabling:

Rapid evaluation of experimental CV data

Predictive modelling with limited datasets

Feedback-loop optimisation within the DeepVolt-X framework

Citation
If you use this work, please cite:

Cavus et al. (2025), AI-Driven Design and Modelling of Asymmetric Supercapacitors Made of Hydrothermally-Synthesised MgMn₂O₄ and Waste-Derived Carbon Cloth Electrodes.
