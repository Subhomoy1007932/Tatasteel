# Tata Steel Motor Anomaly Detection - Copilot Instructions

## Project Overview
This project simulates and analyzes sensor data from a roller table motor to detect anomalies and predict maintenance needs. It consists of a synthetic data generator, an exploratory data analysis script, and a machine learning model pipeline.

## Architecture & Data Flow
The workflow is linear and centered around the CSV dataset:
1.  **Data Generation**: `tata.py` generates synthetic sensor data (simulating load cycles, physics lag, and environmental factors) and saves it to `tata_steel_rot_motor_proxy.csv`.
2.  **Analysis**: `eda.py` reads the CSV for visualization and rule-based anomaly detection (e.g., vibration thresholds).
3.  **Modeling**: `ml_model.py` trains regression models (Decision Tree, Random Forest) to predict motor current and identify anomalies based on prediction error.

## Tech Stack & Dependencies
-   **Language**: Python
-   **Libraries**: `pandas`, `numpy`, `matplotlib`, `scikit-learn`
-   **Data**: CSV format (`tata_steel_rot_motor_proxy.csv`)

## Key Workflows & Commands
-   **Generate New Data**: Run `python tata.py`. adjust `NUM_ROWS` at the top of the file to change dataset size.
-   **Run EDA**: Run `python eda.py`. Note that `plt.show()` calls are currently commented out; uncomment them to view plots.
-   **Train Models**: Run `python ml_model.py`. This script outputs MAE scores and anomaly counts to the console.

## Code Conventions & Patterns
-   **Synthetic Data Logic**: Modifications to data behavior (e.g., failure modes, load patterns) should be made in the simulation loop in `tata.py`.
-   **Anomaly Detection**:
    -   *Rule-Based*: Simple thresholds (e.g., `Vibration_mm_s > 7.0`) in `eda.py`.
    -   *ML-Based*: Residual analysis (Prediction Error > Mean + 3*StdDev) in `ml_model.py`.
-   **Visualization**: Plots are disabled by default in `eda.py`; use standard `matplotlib.pyplot` if enabling them.

## File Structure
-   `tata.py`: Synthetic data generator.
-   `eda.py`: Exploratory Data Analysis & rule-based detection.
-   `ml_model.py`: Machine Learning pipeline (Training, Evaluation, Health Index).
-   `tata_steel_rot_motor_proxy.csv`: The shared dataset (generated artifact).
