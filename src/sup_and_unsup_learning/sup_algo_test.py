import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from logistic_regression import LogisticRegression

#Load heart failure dataset into pandas dataframe
df = pd.read_csv("src/sup_and_unsup_learning/heart.csv")

# Seperate features and target
X = df.drop("DEATH_EVENT", axis=1).values.tolist()
y = df["DEATH_EVENT"].values.tolist()

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(np.array(X_train)).tolist()
X_test = scaler.fit_transform(np.array(X_test)).tolist()

