import numpy as np
import matplotlib.pyplot as plt

x = np.array([2, 3, 4, 5, 6])
y = np.array([50, 70, 90, 100, 110])

m, b = np.polyfit(x, y, 1)
equation = f'y = {m:.2f}x + {b:.2f}'
print(f"Slope (b1): {m:.2f}")
print(f"Intercept (b0): {b:.2f}")
print(f"Line equation: {equation}")

x_pred1 = 8
y_pred1 = m * x_pred1 + b
print(f"If you talk for {x_pred1} hours, the expected profit will be approximately {y_pred1:.2f}k NIS")

y_target = 150
x_pred2 = (y_target - b) / m
print(f"To earn a profit of {y_target}, you need to talk for approximately {x_pred2:.2f} hours.")

plt.figure(figsize=(10, 6))

plt.scatter(x, y, color='blue', label='Data points')

x_range = np.linspace(6,20, 100)
y_range = m * x_range + b
plt.plot(x_range, y_range, color='red', label='Regression line')

plt.scatter([x_pred1], [y_pred1], color='orange', s=100,
            label=f'8 hours → Profit: {y_pred1:.0f}k$')

plt.scatter([x_pred2], [y_target], color='green', s=100,
            label=f'Profit 150k$ → Hours: {x_pred2:.2f}')

plt.text(1.9, 260, equation, fontsize=12, color='green')

plt.title('Calls Hours vs. Profit (Linear Regression with np.polyfit)')
plt.xlabel('Calls Hours')
plt.ylabel('Profit')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
