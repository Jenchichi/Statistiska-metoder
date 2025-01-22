import numpy as np
from scipy import stats
from scipy.stats import linregress

# Steg 1: Läs in datan
data = np.genfromtxt('c:/Programmering/IT_hogskolan/Statistiska-metoder/Data/Small-diameter-flow.csv', delimiter=',', skip_header=1)

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
