import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from logistic_regression import LogisticRegression

#Load heart failure dataset into pandas dataframe
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
X_test_arr  = scaler.transform(X_test_df)

# build and train model
model = LogisticRegression(lr=0.01, n_iters=5000)
model.fit(X_train_arr.tolist(), y_train.tolist())

# results
preds = model.predict(X_test_arr.tolist())
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

