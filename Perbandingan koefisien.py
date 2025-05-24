import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Load data
data = pd.read_excel('/storage/emulated/0/KULIAH/LAMPIRAN/data_standarisasi.xlsx')
X = data[['TPT_zscore', 'UMP_zscore', 'Remitansi_zscore', 'GDP_zscore', 'Dummy']]
y = data['Migrasi_zscore']
var_names = X.columns

# Masukkan alpha terbaik hasil tuning untuk masing-masing model
alpha_ridge = 0.066
alpha_lasso = 0.0069
alpha_enet  = 0.0069
l1_ratio_enet = 0.5

# Fit model dengan alpha terbaik
ridge = Ridge(alpha=alpha_ridge)
ridge.fit(X, y)

lasso = Lasso(alpha=alpha_lasso, max_iter=10000)
lasso.fit(X, y)

enet = ElasticNet(alpha=alpha_enet, l1_ratio=l1_ratio_enet, max_iter=10000)
enet.fit(X, y)

# Buat DataFrame perbandingan koefisien
coef_df = pd.DataFrame({
    'Variabel': var_names,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_,
    'Elastic Net': enet.coef_
})

print("\nPerbandingan Koefisien Variabel pada Ketiga Model:")
print(coef_df)

# Plot
labels = coef_df['Variabel']
x = np.arange(len(labels))
width = 0.25

# Perbesar frame secara vertikal agar label tidak menabrak judul/nama variabel
fig, ax = plt.subplots(figsize=(25,15))
rects1 = ax.bar(x - width, coef_df['Ridge'], width, label='Ridge', color='red')
rects2 = ax.bar(x, coef_df['Lasso'], width, label='Lasso', color='green')
rects3 = ax.bar(x + width, coef_df['Elastic Net'], width, label='Elastic Net', color='purple')

# Tambahkan nilai di atas batang, digeser agar tidak bertabrakan dan tidak menutupi judul/nama variabel
for i, rect in enumerate(rects1):
    height = rect.get_height()
    offset = 0.08 if height >= 0 else -0.13
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width()/2 - 0.07, height),
                xytext=(0, 14 if height >= 0 else -19),
                textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=25, color='black')

for i, rect in enumerate(rects2):
    height = rect.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 34 if height >= 0 else -41),  # lebih tinggi dari rects1
                textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=25, color='black')

for i, rect in enumerate(rects3):
    height = rect.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width()/2 + 0.07, height),
                xytext=(0, 14 if height >= 0 else -19),
                textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=25, color='black')

ax.set_ylabel('Koefisien')
ax.set_title('Perbandingan Koefisien Variabel pada Ridge, Lasso, dan Elastic Net', fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=18)
ax.legend()
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Perbesar margin atas dan bawah agar judul dan label tidak tertutup
plt.subplots_adjust(top=0.85, bottom=0.15)
plt.tight_layout()
plt.savefig('/storage/emulated/0/KULIAH/LAMPIRAN/perbandingan_koefisien.png', dpi=800)
plt.close()

print("Gambar perbandingan koefisien berhasil disimpan di /storage/emulated/0/KULIAH/LAMPIRAN/perbandingan_koefisien.png")
