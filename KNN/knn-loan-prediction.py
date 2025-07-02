import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler


X = np.array([
    [22, 60, 1],
    [25, 75, 2],
    [30, 80, 3],
    [35, 120, 5],
    [42, 150, 8],
    [50, 110, 10],
    [23, 95, 1],
    [28, 90, 2],
    [33, 105, 4],
    [48, 135, 9]
])

y = np.array([
    "didn't returned loan", "didn't returned loan", "returned loan", "returned loan", "returned loan",
    "returned loan", "didn't returned loan", "didn't returned loan", "returned loan", "returned loan"
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

#1#
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#2#
knn_model = KNeighborsClassifier(3)
knn_model.fit(scaled_X_train,y_train)

predict = knn_model.predict(scaled_X_test)

print("=== Initial Evaluation ===")
print("Accuracy:", accuracy_score(y_test,predict) )
print("Classification Report:\n", classification_report(y_test,predict))
print("Confusion Matrix:\n", confusion_matrix(y_test,predict))

new_client = np.array([[27,95,3]])
new_client_scaled = scaler.transform(new_client)
new_client_prediction = knn_model.predict(new_client_scaled)

print("Prediction for new client:", new_client_prediction[0])

#3#
scaled_X = scaler.transform(X)
distances = {}

for i, row in enumerate(scaled_X):
    distance = np.sqrt(np.sum((row-new_client_scaled[0])**2))
    distances[i] = float(distance)
print(distances)

distances_sorted = sorted(distances.items(), key=lambda x: x[1])
three_closest = distances_sorted[:3]

print("Three closest clients are:\n")

for idx, dist in three_closest:
    client_data = X[idx]
    category = y[idx]
    print(f"Client No.:{idx}:\n Client's Original Data:\n - Age: {client_data[0]}\n - Income($k): {client_data[1]}\
    \n - Credit History Length: {client_data[2]}\n Category: {y[idx]}\n Distance from New Client: {dist:.4f} \n")

#4#
print("Based on the analysis above, the model predicts that the new client is unlikely to return the loan.")
print("This is because 2 out of the 3 nearest neighbors are labeled as 'didn't return the loan'.")
print("Since KNN predicts based on the majority vote among the nearest neighbors (K=3), the model classifies the new client accordingly.")
