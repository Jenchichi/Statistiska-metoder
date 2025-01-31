import numpy as np
from scipy import stats

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def d(self):
        return len(self.b) - 1  # Antal funktioner/parametrar/dimensioner

    @property
    def n(self):
        return self.y.shape[0]  # Antal observationer

    @property
    def b(self):
        return np.linalg.pinv(self.x.T @ self.x) @ self.x.T @ self.y

    def predict(self, x):
        # Predict-metod
        return x @ self.b

    def SSE(self):
        # Sum of Squared Errors
        return np.sum(np.square(self.y - self.predict(self.x)))

    def SSR(self):
        # Sum of Squares Regression
        return np.sum((self.predict(self.x) - np.mean(self.y)) ** 2)
    
    def SST(self):
        return np.sum((self.y - np.mean(self.y)) ** 2)

    def variance(self):
        return self.SSE() / (self.n - self.d - 1)

    def standard_deviation(self):
        return np.sqrt(self.variance())  # Roten ur variansen

    def significance(self):
        V = np.sqrt(self.variance())
        f_statistic = (self.SSR() / self.d) / V
        p_value = stats.f.sf(f_statistic, self.d, self.n - self.d - 1)
        return p_value

    def r_squared(self):
        # Sum of squares due to regression / Total sum of squares
        return self.SSR() / self.SST()  
