import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("tata_steel_rot_motor_proxy.csv")


# =========================================================
# 2. FEATURE SELECTION (BEHAVIOR INPUTS)
# =========================================================
features = [
    "Motor_RPM",
    "Vibration_mm_s",
    "Winding_Temp_C",
    "Bearing_Temp_C",
    "Voltage_V"
]

X = df[features]
y = df["Current_Amp"]


# =========================================================
# 3. TRAIN–TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 4. DECISION TREE (BASELINE MODEL)
# =========================================================
dt_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, y_pred_dt)
print("Decision Tree MAE (Amps):", round(dt_mae, 2))


# =========================================================
# 5. RANDOM FOREST (MAIN MODEL)
# =========================================================
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
print("Random Forest MAE (Amps):", round(rf_mae, 2))


# =========================================================
# 6. ML-BASED ANOMALY DETECTION (USING RF)
# =========================================================
prediction_error = np.abs(y_test - y_pred_rf)

error_threshold = prediction_error.mean() + 3 * prediction_error.std()

ml_anomalies = prediction_error > error_threshold

print(f"Anomaly Threshold (Amps error): {error_threshold:.2f}")
print("Number of ML-detected anomalies:", ml_anomalies.sum())


# =========================================================
# 7. HEALTH INDEX (DAMAGE PROGRESSION)
# =========================================================
# Normalize prediction error
normalized_error = prediction_error / prediction_error.max()

# Health Index (1 = healthy, 0 = critical)
health_index = 1 - normalized_error

health_df = pd.DataFrame({
    "Actual_Current": y_test.values,
    "Predicted_Current": y_pred_rf,
    "Prediction_Error": prediction_error,
    "Health_Index": health_index
})

print("\nHealth Index Summary:")
print(health_df["Health_Index"].describe())


# =========================================================
# 8. DAMAGE RISK CLASSIFICATION
# =========================================================
def health_status(h):
    if h > 0.8:
        return "Healthy"
    elif h > 0.6:
        return "Degrading"
    else:
        return "Critical"

health_df["Health_Status"] = health_df["Health_Index"].apply(health_status)

print("\nHealth Status Counts:")
print(health_df["Health_Status"].value_counts())


# =========================================================
# 9. PRESCRIPTIVE ACTION LOGIC
# =========================================================
def prescribe_action(row):
    if row["Health_Status"] == "Healthy":
        return "No action required"
    elif row["Health_Status"] == "Degrading":
        return "Schedule inspection and lubrication check"
    else:
        return "Immediate maintenance: check bearings, alignment, load"

health_df["Prescriptive_Action"] = health_df.apply(prescribe_action, axis=1)

print("\nSample Prescriptive Recommendations:")
print(health_df[["Health_Status", "Prescriptive_Action"]].head(10))
