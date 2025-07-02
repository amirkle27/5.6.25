import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

income = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

returned_loan = [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]

X = [[x] for x in income]
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, returned_loan, test_size=0.20, random_state=27
# )
#
model = LogisticRegression()
model.fit(X, returned_loan)
#
# y_pred = model.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
#
# print("דיוק (Accuracy):", accuracy)
# print("דיוק חיובי (Precision):", precision)
# print("רגישות (Recall):", recall)
# print("מדד F1:", f1)
#
# print("תחזיות המודל:", y_pred.tolist())
# print("התשובות האמיתיות:", y_test)

yearly_income = [[58]]
prob = model.predict_proba(yearly_income)

return_loan_prob = prob[0][1]
print(f"The probability that the client returns the loan stands on {return_loan_prob*100:.2f}% ")

checked_income = [i for i in range (30,151)]

for income in checked_income:
    prob = model.predict_proba([[income]])
    return_loan_prob = prob[0][1]
    if return_loan_prob >= 0.75:
        print(f"Customer needs to earn at least {income}k to return loan, with a {return_loan_prob*100:.2f}% probability of loan return")
        break

incomes = np.arange(30, 151).reshape(-1, 1)

probs = model.predict_proba(incomes)[:, 1]

plt.figure(figsize=(8,5))
plt.plot(incomes, probs, marker='o')
plt.title('Probability of Loan Return vs Yearly Income')
plt.xlabel('Yearly Income (in 1000s)')
plt.ylabel('Probability of Loan Return')
plt.grid(True)
plt.axhline(0.75, color='red', linestyle='--', label='75% threshold')
plt.legend()
plt.show()
