from pandas._libs.hashtable import mode
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from logistic_regression import LogisticRegression

# Load the binary dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

model = LogisticRegression(lr=0.1, n_iters=5000)
model.fit(X_train.tolist(), y_train.tolist())
predictions = model.predict(X_test.tolist())

print(f"Accuracy: {accuracy_score(y_test, predictions)}")
