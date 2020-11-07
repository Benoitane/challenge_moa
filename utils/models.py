import numpy as np

class NullPredictor:
    def __init__(self, value):
        self.value = value

    def fit(self, dataset, labels):
        try:
            self.dimension = labels.shape[1]
        except:
            self.dimension = 1

    def predict(self, X):
        return self.value*np.ones((len(X),self.dimension))

    def score(self, X, y):
        ypred = self.predict(X)
        return np.sqrt(mean_squared_error(y,ypred))
