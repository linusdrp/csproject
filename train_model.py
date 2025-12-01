import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pickle

# --------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------
df = pd.read_csv("synthetic_waiting_data.csv")

# --------------------------------------------------------
# USE ONLY 6 FEATURES TO MATCH STREAMLIT APP
# --------------------------------------------------------
feature_cols = [
    "num_connectors",
    "power_kw",
    "charger_type",
    "time_of_day",
    "day_of_week",
    "traffic_factor"
]

X = df[feature_cols]
y = df["waiting_time_minutes"]

# --------------------------------------------------------
# TRAIN/TEST SPLIT
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------------
# MODEL
# --------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------------
# TRAIN
# --------------------------------------------------------
model.fit(X_train, y_train)

# --------------------------------------------------------
# EVALUATE
# --------------------------------------------------------
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("Model Performance:")
print("----------------------")
print(f"MAE  : {mae:.2f} minutes")
print(f"RMSE : {rmse:.2f} minutes")

# --------------------------------------------------------
# SAVE MODEL
# --------------------------------------------------------
with open("waiting_time_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as waiting_time_model.pkl")

def train_and_save_model():
    with open("waiting_time_model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_and_save_model()

