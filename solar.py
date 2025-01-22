import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000

# Random values for solar irradiance (W/m²), temperature (°C), and time of day (hours)
irradiance = np.random.uniform(100, 1000, n_samples)  # 100 to 1000 W/m²
temperature = np.random.uniform(15, 35, n_samples)    # 15°C to 35°C
time_of_day = np.random.randint(0, 24, n_samples)     # Hour from 0 to 23

# Simple relationship for solar power output (in Watts):
# Solar Power = (Irradiance * Temperature Factor) - Temperature * Random Noise + Time of Day Influence
solar_power_output = (irradiance * 0.5) + (temperature * 2) - (time_of_day * 2) + np.random.normal(0, 100, n_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Solar Irradiance': irradiance,
    'Temperature': temperature,
    'Time of Day': time_of_day,
    'Solar Power Output': solar_power_output
})

# Show the first few rows of the dataset
print(data.head())

# Save the dataset to CSV
data.to_csv('solar_power_data.csv', index=False)

# Plot Solar Power Output vs Solar Irradiance
plt.scatter(data['Solar Irradiance'], data['Solar Power Output'], alpha=0.5)
plt.title('Solar Power Output vs Solar Irradiance')
plt.xlabel('Solar Irradiance (W/m²)')
plt.ylabel('Solar Power Output (W)')
plt.show()



# Load the dataset
data = pd.read_csv('solar_power_data.csv')

# Define features (X) and target (y)
X = data[['Solar Irradiance', 'Temperature', 'Time of Day']]
y = data['Solar Power Output']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the predicted vs actual values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted Solar Power Output')
plt.xlabel('Actual Solar Power Output')
plt.ylabel('Predicted Solar Power Output')
plt.show()

