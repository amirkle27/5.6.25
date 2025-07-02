
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

print("DecisionTree model using Gini!".center(50,"~"))

data = pd.DataFrame({
    'Discount': ['Yes', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Purchase': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Returned': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No']
})

data_encoded = pd.get_dummies(data)

X = data_encoded.drop(['Returned_Yes'], axis = 1)
y = data_encoded['Returned_Yes']

model = DecisionTreeClassifier(criterion= 'gini')
model.fit(X,y)

new_customer = pd.DataFrame({'Discount': ['Yes'], 'Purchase': ['No']})
new_customer_encoded = pd.get_dummies(new_customer)

for col in X.columns:
    if col not in new_customer_encoded.columns:
        new_customer_encoded[col] = 0

new_customer_encoded = new_customer_encoded[X.columns]
new_customer_prediction = model.predict(new_customer_encoded)

print(f"According to the model's prediction, New customer will return to the store: {new_customer_prediction[0]}\n")

plt.figure(figsize=(8, 6))
plot_tree(model, feature_names=X.columns, class_names=["Not Returned", "Returned"], filled=True)
plt.title("Decision Tree - CART (Gini)")
plt.show()

############################################################
print("DecisionTree model using Entropy!".center(50,"~"))

data = pd.DataFrame({
    'Discount': ['Yes', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Purchase': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Returned': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No']
})

data_encoded = pd.get_dummies(data)

X = data_encoded.drop(['Returned_Yes'], axis = 1)
y = data_encoded['Returned_Yes']

model = DecisionTreeClassifier(criterion= 'entropy')
model.fit(X,y)

new_customer = pd.DataFrame({'Discount': ['Yes'], 'Purchase': ['No']})
new_customer_encoded = pd.get_dummies(new_customer)

for col in X.columns:
    if col not in new_customer_encoded.columns:
        new_customer_encoded[col] = 0

new_customer_encoded = new_customer_encoded[X.columns]
new_customer_prediction = model.predict(new_customer_encoded)

print(f"According to the model's prediction, New customer will return to the store: {new_customer_prediction[0]}")

plt.figure(figsize=(8, 6))
plot_tree(model, feature_names=X.columns, class_names=["Not Returned", "Returned"], filled=True)
plt.title("Decision Tree - CART (entropy)")
plt.show()

