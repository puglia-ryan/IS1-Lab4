import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from logistic_regression import LogisticRegression

#Load heart failure dataset into pandas dataframe
df = pd.read_csv("src/sup_and_unsup_learning/heart.csv")
