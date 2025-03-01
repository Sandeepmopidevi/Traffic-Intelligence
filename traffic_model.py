import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load Dataset
print("🔄 Loading Dataset...")
df = pd.read_csv("traffic_volume.csv")
print(f"✅ Dataset Loaded! Shape: {df.shape}")

# Fill Missing Values
df = df.ffill()

# Encode Categorical Variables
print("🔄 Encoding Categorical Variables...")
encoder_holiday = LabelEncoder()
encoder_holiday.fit(df['holiday'].astype(str))

encoder_weather = LabelEncoder()
encoder_weather.fit(df['weather'].astype(str))

df['holiday'] = encoder_holiday.transform(df['holiday'].astype(str))
df['weather'] = encoder_weather.transform(df['weather'].astype(str))

# Drop unnecessary columns
df.drop(columns=['date', 'Time'], inplace=True)

# Define Features and Target
X = df.drop(columns=['traffic_volume'])
y = df['traffic_volume']

# Handle Missing Values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
print("🔄 Training Random Forest Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"📊 Model RMSE: {rmse:.2f}")

# Save Model & Scalers
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/best_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(imputer, open("models/imputer.pkl", "wb"))
pickle.dump(encoder_holiday, open("models/encoder_holiday.pkl", "wb"))
pickle.dump(encoder_weather, open("models/encoder_weather.pkl", "wb"))

print("🎉✅ Model Training Complete. All Files Saved Successfully!")
