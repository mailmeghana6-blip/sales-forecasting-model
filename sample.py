import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load your Excel file
df = pd.read_excel(r"C:\Users\patch\Downloads\receipt_details.xlsx")


# Ensure date is datetime and sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Set date as index
df.set_index('date', inplace=True)

# Split into train (80%) and test (20%)
split_point = int(len(df) * 0.8)
train, test = df.iloc[:split_point], df.iloc[split_point:]

# Forecasting model: Holt-Winters
model = ExponentialSmoothing(
    train['amount'],
    seasonal=None,   # change to 'add' or 'mul' if seasonality exists
    trend="add"      # additive trend
).fit()

# Forecast for test period
forecast = model.forecast(len(test))

# Accuracy scoring
mae = mean_absolute_error(test['amount'], forecast)
rmse = np.sqrt(mean_squared_error(test['amount'], forecast))
accuracy = 100 - (mae / test['amount'].mean() * 100)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Accuracy Score: {accuracy:.2f}%")

# Plot
plt.figure(figsize=(10,5))
plt.plot(train.index, train['amount'], label="Train")
plt.plot(test.index, test['amount'], label="Test", color="orange")
plt.plot(test.index, forecast, label="Forecast", color="red")
plt.legend()
plt.title("Forecasting with Accuracy Score")
plt.show()
