import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

data_path = '/storage/emulated/0/KULIAH/LAMPIRAN/data_standarisasi.xlsx'
save_path = '/storage/emulated/0/KULIAH/LAMPIRAN/cv_elasticnet_rmse.png'

data = pd.read_excel(data_path)
X = data[['TPT_zscore', 'UMP_zscore', 'Remitansi_zscore', 'GDP_zscore', 'Dummy']]
y = data['Migrasi_zscore']

alphas = np.logspace(-4, 2, 50)
l1_ratio = 0.5  # Jika ingin grid search, buat for loop untuk l1_ratio
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mean_rmse, std_rmse = [], []

for alpha in alphas:
    rmses = []
    for train_idx, test_idx in kf.split(X):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[test_idx], y_pred)))
    mean_rmse.append(np.mean(rmses))
    std_rmse.append(np.std(rmses))

mean_rmse = np.array(mean_rmse)
std_rmse = np.array(std_rmse)
idx_min = np.argmin(mean_rmse)
alpha_min = alphas[idx_min]
log_alpha_min = np.log10(alpha_min)
rmse_min = mean_rmse[idx_min]

plt.figure(figsize=(10,6))
plt.errorbar(np.log10(alphas), mean_rmse, yerr=std_rmse, fmt='-o', color='purple', ecolor='gray', capsize=3, label=f'Elastic Net RMSE ± std (l1_ratio={l1_ratio})')
plt.axvline(log_alpha_min, color='blue', linestyle='--', label=f'log(λ_min)={log_alpha_min:.2f}')
plt.xlabel('log10(λ)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title(f'Cross Validation parameter λ pada Elastic Net (l1_ratio={l1_ratio})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Plot Elastic Net disimpan di: {save_path}")
print(f"Elastic Net: alpha (λ) terbaik = {alpha_min:.6f}, RMSE = {rmse_min:.4f}, l1_ratio = {l1_ratio}")
