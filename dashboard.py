import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("dataset/sales_data.csv")

# Create Dashboard
plt.figure(figsize=(12,6))

# Chart 1: Sales Trend
plt.subplot(1,2,1)
plt.plot(df['Sales'])
plt.title("Sales Trend")

# Chart 2: Sales Distribution
plt.subplot(1,2,2)
plt.hist(df['Sales'])
plt.title("Sales Distribution")

# Save Dashboard
plt.savefig("dashboard.png")

plt.show()