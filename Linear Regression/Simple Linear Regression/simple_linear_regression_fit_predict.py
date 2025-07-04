import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

advertising_investment = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50]).reshape(-1, 1)
sales_growth = np.array([25, 30, 40, 45, 50, 60, 65, 70, 80])

model = LinearRegression()
model.fit(advertising_investment, sales_growth)
m = model.coef_[0]
b = model.intercept_
equation = f'y = {m:.2f}x + {b:.2f}'

print(f"Slope (m): {m:.2f}")
print(f"Intercept (b): {b:.2f}")
print(f"Line equation: {equation}")

plt.figure(figsize=(10, 6))

plt.scatter(advertising_investment, sales_growth, color='blue', label='Data points')

plt.plot(advertising_investment, model.predict(advertising_investment), color='red', label='Regression line')

plt.text(9, 72, equation, fontsize=12, color='green')

plt.title('Linear Regression - Advertising Investment vs. Sales Growth')
plt.xlabel('Advertising Investment (in 1000 ILS)')
plt.ylabel('Sales Growth (in 1000 ILS)')
plt.grid(True)
plt.legend(loc='upper left')

plt.show()
