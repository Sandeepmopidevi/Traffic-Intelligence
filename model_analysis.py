# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os

# Load Dataset
print("🔄 Loading Dataset...")
df = pd.read_csv("traffic_volume.csv")
print(f"✅ Dataset Loaded! Shape: {df.shape}\n")

# Create a folder to save images
output_folder = "Analysis_Images"
os.makedirs(output_folder, exist_ok=True)

# Display first few rows
print("📌 First 5 Rows:")
print(df.head(), "\n")

# Summary Statistics
print("📊 Dataset Summary:")
print(df.describe(), "\n")

# Data Types & Missing Values
print("🔎 Data Types & Missing Values:")
print(df.info(), "\n")

# Check for null values
print("🔍 Missing Values in Dataset:")
print(df.isnull().sum(), "\n")

# 🔹 Visualizing Missing Values
plt.figure(figsize=(14, 7))
msno.bar(df, color="dodgerblue")
plt.title("Missing Values Visualization")
plt.savefig(os.path.join(output_folder, "missing_values.png"), bbox_inches='tight')
plt.show()

# 🔹 Handling Missing Values (Forward Fill)
df.ffill(inplace=True)  

# 🔹 Convert 'weather' column to string (Ensure consistency)
df['weather'] = df['weather'].astype(str)

# 🔹 Keep Original Weather Labels for Visualization
weather_mapping = {
    "Clear": "Clear",
    "Clouds": "Cloudy",
    "Mist": "Mist",
    "Rain": "Rain",
    "Drizzle": "Drizzle",
    "Thunderstorm": "Thunderstorm",
    "Snow": "Snow",
    "Haze": "Hazy",
    "Fog": "Foggy",
    "Squall": "Squall",
    "Smoke": "Smoky"
}

df['weather'] = df['weather'].map(weather_mapping).fillna("Unknown")  # Handle missing categories

# Drop non-numeric columns before correlation
df_numeric = df.select_dtypes(include=[np.number]) 

# 🔹 Correlation Heatmap
plt.figure(figsize=(14, 7))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"), bbox_inches='tight')
plt.show()

# 🔹 Traffic Volume Distribution
plt.figure(figsize=(14, 7))
sns.histplot(df['traffic_volume'], bins=30, kde=True, color="purple")
plt.xlabel("Traffic Volume")
plt.ylabel("Frequency")
plt.title("Traffic Volume Distribution")
plt.savefig(os.path.join(output_folder, "traffic_volume_distribution.png"), bbox_inches='tight')
plt.show()

# 🔹 Weather Conditions Analysis (Keep Labels)
plt.figure(figsize=(14, 7))
sns.countplot(x='weather', data=df, hue='weather', legend=False, palette="viridis", order=df['weather'].value_counts().index)
plt.xticks(rotation=45)
plt.xlabel("Weather Condition")
plt.ylabel("Count")
plt.title("Weather Condition Frequency")
plt.savefig(os.path.join(output_folder, "weather_conditions.png"), bbox_inches='tight')
plt.show()

# 🔹 Traffic Volume by Weather (Keep Labels)
plt.figure(figsize=(14, 7))
sns.boxplot(x="weather", y="traffic_volume", data=df, hue="weather", legend=False, palette="mako", order=df['weather'].value_counts().index)
plt.xticks(rotation=45)
plt.xlabel("Weather Condition")
plt.ylabel("Traffic Volume")
plt.title("Traffic Volume vs. Weather Condition")
plt.savefig(os.path.join(output_folder, "traffic_vs_weather.png"), bbox_inches='tight')
plt.show()

# 🔹 Fix Date Parsing Issue
try:
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')  
except Exception as e:
    print(f"⚠️ Date Parsing Error: {e}")

# 🔹 Pair Plot (Visualizing relationships)
sns.pairplot(df_numeric, diag_kind="kde", plot_kws={'alpha': 0.5}) 
plt.savefig(os.path.join(output_folder, "pair_plot.png"), bbox_inches='tight')
plt.show()

# Save Dataset Summary to a Text File
with open("dataset_summary.txt", "w") as file:
    file.write(f"Dataset Shape: {df.shape}\n\n")
    file.write("First 5 Rows:\n")
    file.write(df.head().to_string())
    file.write("\n\nDataset Summary:\n")
    file.write(df.describe().to_string())
    file.write("\n\nMissing Values:\n")
    file.write(df.isnull().sum().to_string())

print("📄✅ Dataset Summary Saved as 'dataset_summary.txt'")
print(f"📷✅ All analysis images saved to '{output_folder}' folder.")