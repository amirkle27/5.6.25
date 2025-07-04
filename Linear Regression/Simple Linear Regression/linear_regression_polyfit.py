import numpy as np
import matplotlib.pyplot as plt
x = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
y = np.array([25, 30, 40, 45, 50, 60, 65, 70, 80])

m, b = np.polyfit(x, y, 1)

print(f"שיפוע (b1): {m:.2f}")
print(f"נקודת חיתוך עם ציר y (b0): {b:.2f}")



plt.figure(figsize=(10, 6))

# פיזור הנתונים המקוריים (הנקודות הכחולות)
plt.scatter(x, y, color='blue', label='Data points')

# קו הרגרסיה המחושב עם polyfit (קו אדום)
y_pred = m * x + b  # מחשבים את כל ערכי y לפי הקו
plt.plot(x, y_pred, color='red', label='Regression line')

# משוואת הקו
equation = f'y = {m:.2f}x + {b:.2f}'
plt.text(9, 73, equation, fontsize=12, color='green')

# כותרות וצירים
plt.title('Linear Regression with np.polyfit')
plt.xlabel('Advertising Investment (in 1000 ILS)')
plt.ylabel('Sales Growth (in 1000 ILS)')
plt.grid(True)
plt.legend(loc = 'upper left')

plt.show()