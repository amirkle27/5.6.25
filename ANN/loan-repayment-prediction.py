import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# שלב 1: הכנסת הנתונים
X = np.array([[30], [35], [40], [45], [50], [55], [60], [65], [70], [75], [80], [85], [90]], dtype=float)
y = np.array([[0], [0], [0], [0], [0], [1], [0], [1], [1], [1], [1], [1], [1]], dtype=float)

# שלב 2: נורמליזציה
# StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# שלב 3: בניית המודל
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(1,)))
model.add(Dense(1, activation='sigmoid'))

# שלב 4: קומפילציה ואימון
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y, epochs=100, verbose=1)

# הסתברות החזר ל-58 אלף ש"ח
income_58 = scaler.transform(np.array([[58]]))
prediction = model.predict(income_58)
print(f" Probability of loan repayment: {prediction[0][0]:.2f}")
