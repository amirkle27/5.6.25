import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

calls_hours = np.array([2, 3, 4, 5, 6]).reshape(-1, 1)
profit = np.array([50, 70, 90, 100, 110])

model = LinearRegression()
model.fit(calls_hours, profit)
m = model.coef_[0]
b = model.intercept_

print(f"Slope (m): {m:.2f}")
print(f"Intercept (b): {b:.2f}")
equation = f'y = {m:.2f}x + {b:.2f}'
print(f"Line equation: {equation}")

x_pred1 = 8
y_pred1 = m * x_pred1 + b
print(f"Expected profit after {x_pred1} hours calls: {y_pred1:.2f}k$")

y_pred2 = 150
x_pred2 = (y_pred2 - b) / m
print(f"To earn a profit of {y_pred2}k$ You need to make {x_pred2:.2f} hours of calls")

plt.figure(figsize=(10, 6))

plt.scatter(calls_hours, profit, color='blue', label='Data points')

plt.plot(calls_hours, model.predict(calls_hours), color='red', label='Regression line')

plt.scatter([x_pred1], [y_pred1], color='orange', s=100,
            label=f'8 hours → Profit: {y_pred1:.0f}k$')

plt.scatter([x_pred2], [y_pred2], color='green', s=100,
            label=f'Profit 150k$ → Hours: {x_pred2:.2f}')

plt.text(1.9, 129, equation, fontsize=12, color='green')

plt.title('Linear Regression - Calls Hours vs. Profit')
plt.xlabel('Calls Hours')
plt.ylabel('Profit')
plt.grid(True)
plt.legend(loc='upper left')

plt.show()
