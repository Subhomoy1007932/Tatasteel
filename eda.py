import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("tata_steel_rot_motor_proxy.csv")

# print(df.head())
# df.info()
# print("Total rows:", len(df))

# plt.figure(figsize=(12, 4))
# plt.plot(df["Current_Amp"])
# plt.title("Motor Current Over Time")
# plt.xlabel("Time Index")
# plt.ylabel("Current (Amp)")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(df["Current_Amp"], label="Motor Current (A)", alpha=0.7)
# plt.plot(df["Winding_Temp_C"], label="Winding Temperature (°C)", alpha=0.7)
# plt.title("Motor Current vs Winding Temperature")
# plt.xlabel("Time Index")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # ---- Zoomed-in view (first 300 seconds) ----
# zoom_df = df.iloc[0:300]  # first 5 minutes (300 seconds)

# plt.figure(figsize=(12, 4))
# plt.plot(zoom_df["Current_Amp"], label="Motor Current (A)")
# plt.plot(zoom_df["Winding_Temp_C"], label="Winding Temperature (°C)")
# plt.title("Zoomed View: Motor Current vs Winding Temperature")
# plt.xlabel("Time Index (seconds)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# ---- Simple Anomaly Detection (Rule-Based) ----
vibration_threshold = 7.0  # mm/s (abnormally high)

anomalies = df[df["Vibration_mm_s"] > vibration_threshold]

print("Number of vibration anomalies detected:", len(anomalies))
print(anomalies[["Timestamp", "Vibration_mm_s"]].head())

# # ---- Plot Vibration with Anomalies Highlighted ----
# plt.figure(figsize=(12, 4))

# plt.plot(df["Vibration_mm_s"], label="Vibration (mm/s)", alpha=0.7)

# plt.scatter(
#     anomalies.index,
#     anomalies["Vibration_mm_s"],
#     color="red",
#     label="Anomaly",
#     zorder=5
# )

# plt.title("Vibration Monitoring with Detected Anomalies")
# plt.xlabel("Time Index")
# plt.ylabel("Vibration (mm/s)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # ---- Zoomed-in Anomaly Visualization (first 300 seconds) ----
# zoom_df = df.iloc[0:300]
# zoom_anomalies = anomalies[anomalies.index < 300]

# plt.figure(figsize=(12, 4))

# plt.plot(
#     zoom_df["Vibration_mm_s"],
#     label="Vibration (mm/s)",
#     alpha=0.7
# )

# plt.scatter(
#     zoom_anomalies.index,
#     zoom_anomalies["Vibration_mm_s"],
#     color="red",
#     label="Anomaly",
#     zorder=5
# )

# plt.title("Zoomed Vibration Monitoring with Anomalies (First 300 Seconds)")
# plt.xlabel("Time Index (seconds)")
# plt.ylabel("Vibration (mm/s)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# ---- Motor RPM vs Current (Load Effect) ----
# ---- Zoomed RPM vs Current (First 300 Seconds) ----
zoom_df = df.iloc[0:300]

# plt.figure(figsize=(12, 4))
# plt.plot(zoom_df["Motor_RPM"], label="Motor RPM")
# plt.plot(zoom_df["Current_Amp"], label="Motor Current (A)")
# plt.title("Zoomed Motor RPM vs Current (First 300 Seconds)")
# plt.xlabel("Time Index (seconds)")
# plt.legend()
# plt.tight_layout()
# plt.show()


# ---- Bearing Temperature Trend ----
# ---- Zoomed Bearing Temperature (First 300 Seconds) ----
plt.figure(figsize=(12, 4))
plt.plot(zoom_df["Bearing_Temp_C"], label="Bearing Temperature (°C)")
plt.title("Zoomed Bearing Temperature (First 300 Seconds)")
plt.xlabel("Time Index (seconds)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()

