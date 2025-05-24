import numpy as np
import matplotlib.pyplot as plt

# hasil koefisien 
coef_ridge = [-0.2102, -0.9303, 1.2169, -0.2728, 0.0231]
coef_lasso = [-0.2212, -0.9091, 1.1725, -0.2433, 0.0]
coef_enet  = [-0.2240, -0.9045, 1.1634, -0.2392, 0.0]
var_names = ['TPT', 'UMP', 'Remitansi', 'GDP', 'Dummy']

coef_matrix = np.array([coef_ridge, coef_lasso, coef_enet])
model_names = ['Ridge', 'Lasso', 'ElasticNet']
x = np.arange(len(var_names))
width = 0.22

fig, ax = plt.subplots(figsize=(10,6))
bars1 = ax.bar(x - width, coef_matrix[0], width, label='Ridge')
bars2 = ax.bar(x, coef_matrix[1], width, label='Lasso')
bars3 = ax.bar(x + width, coef_matrix[2], width, label='ElasticNet')

plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
plt.xticks(x, var_names, fontsize=12)
plt.ylabel('Nilai Koefisien', fontsize=12)
plt.title('Perbandingan Koefisien Tiap Variabel pada Ridge, Lasso, dan Elastic Net', fontsize=14)
plt.legend(fontsize=12)

def add_value_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.03*np.sign(yval), f'{yval:.3f}', 
                ha='center', va='bottom' if yval>=0 else 'top', fontsize=10)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.tight_layout()

# Simpan gambar 
plt.savefig('/storage/emulated/0/KULIAH/koefisien_regresi_ridge_lasso_elasticnet.png', dpi=400)
plt.show()
