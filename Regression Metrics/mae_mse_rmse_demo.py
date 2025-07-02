
Question 1:
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_true = np.array([
    10.5, 8.2, 7.3, 15.4, 12.0,
    9.8, 6.7, 11.2, 13.5, 9.0,
    8.1, 14.2, 10.0, 7.5, 12.8,
    9.3, 11.8, 8.9, 10.7, 13.1
])

# predicted values
y_pred = np.array([
    11.2, 7.8, 7.0, 16.1, 11.5,
    9.5, 7.2, 10.8, 14.0, 8.7,
    8.5, 13.9, 10.4, 7.8, 12.5,
    9.0, 12.3, 9.4, 11.1, 12.8
])
1. Calculate the Mean Absolute Error (MAE).
mae = mean_absolute_error(y_true,y_pred)
print(f"MAE: {mae:.3f}")
MAE: 0.415
2. Calculate the Mean Squared Error (MSE).
mse = mean_squared_error(y_true,y_pred)
print(f"MSE: {mse:.3f}")
MSE: 0.187
3. Calculate the Root Mean Squared Error (RMSE).
rmse = mean_squared_error(y_true,y_pred,squared=False)
print(f"RMSE: {rmse:.3f}")
RMSE: 0.433
C:\Users\123\anaconda3\Lib\site-packages\sklearn\metrics\_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
  warnings.warn(
