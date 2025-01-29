import numpy as np
from scipy import stats

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Beräkna koefficienter med linjär regression
        X = np.array(X)
        y = np.array(y)
        self.coefficients, self.intercept, r_value, p_value, std_err = stats.linregress(X, y)

    def predict(self, X):
        # Gör förutsägelser
        return self.intercept + self.coefficients * np.array(X)

    def score(self, X, y):
        # Beräkna R²-värde
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def load_data(self, filename):
        # Läs in data från CSV-fil
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        return data[:, 0], data[:, 1]  # Returnera oberoende och beroende variabler
