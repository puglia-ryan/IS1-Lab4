from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DecisionTree import DecisionTree

df = pd.read_csv("data/heart.csv")
X = df.drop("HeartDisease", axis = 1).values
y = df["HeartDisease"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print("Accuracy = ", acc)

depths = range(1,21)
accuracies = []
for d in depths:
    clf = DecisionTree(max_depth = d)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.sum(y_pred == y_test)/len(y_test)
    accuracies.append(acc)

#graphic
plt.plot(depths, accuracies, marker='o')
plt.xlabel('Maximum tree depth')
plt.ylabel('Accuracy on test data')
plt.title('Dependence of acccuracy on tree depth')
plt.grid(True)
plt.show()
