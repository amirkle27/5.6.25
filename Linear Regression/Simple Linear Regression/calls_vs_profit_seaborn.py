import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x = np.array([2, 3, 4, 5, 6])
y = np.array([50, 70, 90, 100, 110])

plt.figure(figsize=(10, 6))
sns.regplot(x=x, y=y, ci=None, color='red', marker='o', scatter_kws={"s": 100})

plt.title('Linear Regression with sns.regplot')
plt.xlabel('Calls Hours')
plt.ylabel('Profit')
plt.grid(True)
plt.show()
