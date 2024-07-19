import numpy as np

# Example actual values and predictions
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Calculate MAE
mae = np.mean(np.abs(y_true - y_pred))
print("Mean Absolute Error (MAE):", mae)
