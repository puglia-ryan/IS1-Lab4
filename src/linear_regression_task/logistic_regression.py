import math
# This file implements the logistic regression algorithm (supervised learning)

class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters

    def _sigmoid(self, z):
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            return exp_z / (1.0 + exp_z)

    def fit(self, X, y):
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

    def predict(self, X):
        y_preds = []
        for xi in X:
            z = sum(wi*xx for wi, xx in zip(self.w, xi)) + self.b
            y_pred = self._sigmoid(z)
            y_preds.append(1 if y_pred >= 0.5 else 0)
        return y_preds
