import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = "/storage/emulated/0/karil/Makro/data_fix.xlsx"
df = pd.read_excel(file_path)

cols_to_standardize = ['Migrasi', 'TPT', 'UMP', 'Remitansi', 'GDP',]
non_numeric = df[['tahun', 'kuartal', 'Dummy']].copy()

# Standarisasi
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df[cols_to_standardize])

df_standardized = pd.DataFrame(standardized_data, columns=[f"{col}_zscore" for col in cols_to_standardize])

final_df = pd.concat([non_numeric, df_standardized], axis=1)

# Simpan hasil ke file Excel (.xlsx)
output_path = "/storage/emulated/0/karil/Makro/data_standarisasi.xlsx"
final_df.to_excel(output_path, index=False)


output_path
