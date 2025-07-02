import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


apartments = pd.read_csv('C:/Users/123/Downloads/git itay/apartments.csv')
X = apartments.drop('type', axis=1)
y = apartments['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

pipeline = make_pipeline(StandardScaler(), LogisticRegressionCV(multi_class='multinomial', solver='lbfgs', cv=5))
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model's Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
disp.plot(cmap=plt.cm.viridis)


plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

new_apt = pd.DataFrame([[90, 4, 10, 4]], columns=["area", "rooms", "age", "distance"])
prediction = pipeline.predict(new_apt)

print(f"Predicted Apartment Type for the new apartment (90sqm, 4 rooms, age 10, distance 4km):\n {prediction[0]}")
