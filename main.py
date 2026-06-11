import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("Dataset Loaded")

df = pd.read_csv("dataset/sales_data.csv")

print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Values:")
print(df.isnull().sum())

# Data Cleaning

df.drop_duplicates(inplace=True)

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nStatistical Analysis")
print(df.describe())

df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Graph 1
plt.figure(figsize=(10,5))
plt.plot(df.index, df['Sales'])
plt.title("Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid()
plt.show()

# Forecast Model
model = ExponentialSmoothing(
    df['Sales'],
    trend='add'
)

fit = model.fit()

forecast = fit.forecast(6)

print("\nForecasted Sales:")
print(forecast)

# Graph 2
plt.figure(figsize=(10,5))
plt.plot(df.index, df['Sales'], label='Actual')
plt.plot(forecast.index, forecast, label='Forecast')
plt.title("Sales Forecast")
plt.legend()
plt.grid()
plt.show()