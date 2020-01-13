import numpy as np

class FeatureComposition:
    def __init__(self, m1, f1=[], f2=[]):
        self.m1 = m1
        self.f1 = f1
        self.f2 = f2
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.array(X)
        X1 = X[:, :self.m1]
        X2 = X[:, self.m1:]
        for f in f1:
            X1 = f(X1)
        for f in f2:
            X2 = f(X2)
        X_prime = np.hstack([X1, X2])
        return X_prime
