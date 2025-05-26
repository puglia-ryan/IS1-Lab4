import math
# This file implements the logistic regression algorithm (supervised learning)

class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters

    def _sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
