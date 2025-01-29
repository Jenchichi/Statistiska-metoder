import numpy as np
from scipy import stats
from scipy.stats import linregress
from linear_regression_v2 import LinearRegression

# Steg 1: Läs in datan
filename = 'c:/Programmering/IT_hogskolan/Statistiska-metoder/Data/Small-diameter-flow.csv'
data = np.genfromtxt(filename, delimiter=',', skip_header=1)

# Steg 2: Definiera oberoende och beroende variabler
X = data[:, 0]  # Anta att första kolumnen är oberoende variabel
Y = data[:, 1]  # Anta att andra kolumnen är beroende variabel

# Steg 3: Skapa en instans av LinearRegression
model = LinearRegression()

# Steg 4: Passa modellen till data
model.fit(X, Y)

# Steg 5: Gör förutsägelser
predictions = model.predict(X)

# Steg 6: Beräkna R²-värde
r_squared = model.score(X, Y)

# Steg 7: Visa resultat
print(f'Förutsägelser: {predictions}')
print(f'R²-värde: {r_squared}')

# Steg 2: Beräkna d-värdet
d = data.shape[1] - 1  # Antal funktioner (kolumner) minus 1 för målet
print(f"d-värde (antal funktioner): {d}")

# Steg 3: Beräkna n-värdet
n = data.shape[0]  # Storleken på urvalet (antal rader)
print(f"n-värde (storlek på urvalet): {n}")

# Steg 4: Beräkna variansen:
def calculate_variance(data):
    return np.var(data, axis=0, ddof=1)  # Varians med Bessel's korrigering
    
variances = calculate_variance(data)
print(f"Varians för varje kolumn: {variances}")
