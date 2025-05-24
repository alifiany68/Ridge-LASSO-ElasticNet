import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_excel('/storage/emulated/0/karil/Makro/data_standarisasi.xlsx')

variabel = ['TPT', 'UMP', 'Remitansi', 'GDP', 'Dummy']
X = df[variabel]

print("MATRIKS KORELASI:\n")
print(X.corr().round(2))
print("\n")

X_const = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Variabel"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

print("VARIANCE INFLATION FACTOR (VIF):\n")
print(vif_data.to_string(index=False))
print("\n")

print("INTERPRETASI:\n")
for _, row in vif_data.iterrows():
    var = row['Variabel']
    vif = row['VIF']
    if vif < 5:
        kategori = "Multikolinearitas rendah"
    elif 5 <= vif < 10:
        kategori = "Multikolinearitas sedang"
    else:
        kategori = "🚨 Multikolinearitas tinggi (perlu tindakan)"
    print(f"- {var}: VIF = {vif:.2f} → {kategori}")
