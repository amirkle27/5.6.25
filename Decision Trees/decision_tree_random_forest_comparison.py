from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# large data set
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

decision_tree_model = DecisionTreeClassifier(criterion='gini')
decision_tree_model.fit(X_train, y_train)
tree_preds = decision_tree_model.predict(X_test)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)


accuracy_decision_tree = decision_tree_model.score(X_test, y_test)
accuracy_random_forest = random_forest_model.score(X_test, y_test)

print (f'Accuracy Score for Decision Tree Model: {accuracy_decision_tree:.2f}')
print (f'Accuracy Score for Random Forest Model: {accuracy_random_forest:.2f}')

models = ['Decision Tree' , 'Random Forest']
accuracies = [accuracy_decision_tree, accuracy_random_forest]

plt.bar(models, accuracies)
plt.ylabel('Accuracy Score')
plt.title('Comparison between Decision Tree and Random Forest Accuracy Score')

plt.show()
