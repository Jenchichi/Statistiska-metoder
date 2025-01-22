import numpy as np
from scipy import stats
from scipy.stats import linregress

class LinearRegression:
    def __init__(self):
        self.data = np.genfromtxt('c:/Programmering/IT_hogskolan/Statistiska-metoder/Data/Small-diameter-flow.csv', delimiter=',', skip_header=1)
        self.d = self.data.shape[1] - 1  # Antal funktioner (kolumner) minus 1 för målet
        self.n = self.data.shape[0]  # Storleken på urvalet, antal rader i datan.
        self.variance = self.calculate_variance()
        self.standard_deviation = self.calculate_standard_deviation()

    # Beräkning av medelvärdet och Beräkning av variansen för varje kolumn i datan.
    def calculate_variance(self):
        mean = np.mean(self.data, axis=0)
        return np.var(self.data, axis=0, ddof=1)  # Varians med Bessel's korrigering

    def calculate_standard_deviation(self):
        return np.sqrt(self.calculate_variance())  # Retunerar Standardavvikelsen genom kvadratroten av variansen.

    def report_significance(self):
        # Implementera signifikansrapportering. Utför en linjär regression på första två kolumnerna av datan.
        slope, intercept, r_value, p_value, std_err = linregress(self.data[:, 0], self.data[:, 1])
        print(f"p-value: {p_value}")

    def report_relevance(self):
        # Implementera R²-beräkning. Visar hur mycket av variationen i den beroende variabeln som kan förklaras av den oberoende variblerna.
        slope, intercept, r_value, p_value, std_err = linregress(self.data[:, 0], self.data[:, 1])
        print(f"R²: {r_value**2}")

    def print_results(self):
        print(f"n: {self.n}")
        print(f"Varians: {self.variance}")
        print(f"Standardavvikelse: {self.standard_deviation}")
        self.report_significance()
        self.report_relevance()

# Exempel på hur man kan använda klassen
lr = LinearRegression()
lr.print_results()