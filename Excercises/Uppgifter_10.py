# 10a) Denna övning involverar bostadsdatasettet från Boston.
# För att börja, ladda in bostonsdatasettet, som är en del av ISLP-biblioteket.
from ISLP import load_data
import seaborn as sns
import matplotlib.pyplot as plt

boston = load_data('Boston')

# 10b) hur många rader och kolumner finns det i bostonsdatasettet?
num_rows = boston.shape[0]
num_cols = boston.shape[1]
print(f"Number of rows: {num_rows}, Number of columns: {num_cols}")

# 10c) Gör några parvisa scatterplots av prediktorerna (kolumnerna) i detta dataset.
prediktorer = ["rm", "age", "dis", "medv"]

# Skapa parvisa scatterplots
sns.pairplot(boston[prediktorer])
plt.suptitle('Parvisa scatterplots av prediktorer i bostonsdatasettet')
plt.subplots_adjust(top=0.9)
plt.tight_layout()
#plt.show()

# 10d) Är några av prediktorerna kopplade till brottslighet per capita? 
# rm = antal rum, age= ålder på byggnader, dis = avstånd till centrum, medv = medianvärde på bostäder.
for prediktor in prediktorer:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=boston[prediktor], y=boston["crim"])
    plt.title(f"Samband mellan {prediktor} och brottslighet per capita")
    plt.xlabel(prediktor)
    plt.ylabel("Brottslighet per capita")
    #plt.show()
    
# 10e) Finns det några av förorterna i Boston som verkar ha särskilt höga brottslighetstal? Skattesatser? Förhållanden mellan elever och lärare?
# Definiera tröskelvärden för att identifiera höga värden
brottslighet_tröskel = boston['crim'].quantile(0.75)  # 75:e percentilen
skattesats_tröskel = boston['tax'].quantile(0.75)  # 75:e percentilen
elev_lärare_tröskel = boston['ptratio'].quantile(0.75)  # 75:e percentilen

# Hitta förorter med höga brottslighetstal
höga_brott = boston[boston['crim'] > brottslighet_tröskel]
höga_skatter = boston[boston['tax'] > skattesats_tröskel]
höga_elev_lärare = boston[boston['ptratio'] < elev_lärare_tröskel]  # Lägre värden är bättre

# Skriv ut resultaten
print(f"Antal förorter med hög brottslighet: {len(höga_brott)}")
print(f"Antal förorter med hög skattesats: {len(höga_skatter)}")
print(f"Antal förorter med hög elev-lärare förhållande: {len(höga_elev_lärare)}")

# 10f) Hur många av förorterna i detta dataset gränsar till Charles River?
antal_förorter_till_charles_river = boston[boston['chas'] == 1].shape[0]
print(f"Antal förorter till Charles River: {antal_förorter_till_charles_river}")

# 10g) Vad är medianen för förhållandet mellan elever och lärare bland orterna i detta dataset?
median_elev_lärare = boston['ptratio'].median()
print(f"Medianen för förhållandet mellan elever och lärare: {median_elev_lärare}")

# 10h) Vilken förort i Boston har det lägsta medianvärdet för ägarbelagda bostäder? 
# Vad är värdena för de andra prediktorerna för den förorten, och hur jämförs dessa
# värden med de övergripande intervallen för dessa prediktorer?
# 10h) Vilken förort i Boston har det lägsta medianvärdet för ägarbelagda bostäder?
lägsta_median_värde = boston['medv'].min()  # Hitta det lägsta medianvärdet
lägsta_förort = boston[boston['medv'] == lägsta_median_värde]  # Filtrera förorten med det lägsta värdet

# Skriv ut förorten och dess värden för andra prediktorer
print(f"Förort med lägsta medianvärdet för ägarbelagda bostäder:\n{lägsta_förort}")

# Hämta värden för andra prediktorer
andra_prediktorer = lägsta_förort.drop(columns=['medv'])  # Ta bort 'medv' kolumnen
print(f"Värden för andra prediktorer:\n{andra_prediktorer}")

# Jämför med övergripande intervall för dessa prediktorer
for kolumn in andra_prediktorer.columns:
    min_värde = boston[kolumn].min()
    max_värde = boston[kolumn].max()
    print(f"{kolumn}: Min: {min_värde}, Max: {max_värde}")


# 10i) Hur många av förorterna har i genomsnitt mer än sju rum per bostad? Mer än åtta rum per bostad?

# Beräkna antalet förorter med mer än 7 rum
antal_förorter_mer_an_sju_rum = boston[boston['rm'] > 7].shape[0]

# Beräkna antalet förorter med mer än 8 rum
antal_förorter_mer_an_åtta_rum = boston[boston['rm'] > 8].shape[0]

print(f"Antal förorter med i genomsnitt mer än 7 rum per bostad: {antal_förorter_mer_an_sju_rum} st.")
print(f"Antal förorter med i genomsnitt mer än 8 rum per bostad: {antal_förorter_mer_an_åtta_rum} st.")



#testar