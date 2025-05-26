import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

# Load heart failure dataset into pandas dataframe
df = pd.read_csv("src/sup_and_unsup_learning/heart.csv")
print(df.head())

# separate features & target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# one-hot encoding
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# train test split
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# feature scaling
scaler = StandardScaler()
X_train_arr = scaler.fit_transform(X_train_df)
X_test_arr = scaler.transform(X_test_df)

class LogisticRegressionWithHistory(LogisticRegression):

    def fit(self, X, y):
        self.history = []
        n_samples, n_features = len(X), len(X[0])
        
        #weights and biases are initialised
        self.w = [0.0]*n_features
        self.b = 0.0

        for _ in range(self.n_iters):
            dw = [0.0]*n_features
            db = 0.0
            
            # Computing the gradients:
            for xi, yi in zip(X, y):
                #The linear model
                z = sum(wi*xx for wi, xx in zip(self.w, xi)) + self.b
                y_pred = self._sigmoid(z)

                # gradients
                error = y_pred -yi
                for j in range(n_features):
                    dw[j] += xi[j] * error
                db += error
            
            # Average is computed
            dw = [d / n_samples for d in dw]
            db /= n_samples
            
            # Parameters are updated
            self.w = [wi - self.lr * dwi for wi, dwi in zip(self.w, dw)]
            self.b -= self.lr * db

            # Track the history
            predictions = self.predict(X)
            acc = accuracy_score(y, predictions)
            self.history.append(acc)

# build and train model
model = LogisticRegressionWithHistory(lr=0.01, n_iters=5000)
model.fit(X_train_arr.tolist(), y_train.tolist())

# results
preds = model.predict(X_test_arr.tolist())
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# plotting the accuracy of the model's predictions over time
plt.figure(figsize=(8,5))
plt.plot(range(1, len(model.history)+1), model.history, marker="o")
plt.title("Training Accuracy over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
