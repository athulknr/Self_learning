#Root Mean Squared Error
import numpy as np

# Actual values
y_actual = np.array([100, 150, 200, 250, 300])

# Predicted values
y_predicted = np.array([110, 140, 210, 260, 290])

# Calculate RMSE
rmse = np.sqrt(np.mean((y_actual - y_predicted) ** 2))

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
