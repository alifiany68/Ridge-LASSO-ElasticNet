import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge #sesuaikan dengan model yang ingin di lihat
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_excel('/storage/emulated/0/KULIAH/LAMPIRAN/data_standarisasi.xlsx')
X = data[['TPT_zscore', 'UMP_zscore', 'Remitansi_zscore', 'GDP_zscore', 'Dummy']]
y = data['Migrasi_zscore']

alphas = np.logspace(-4, 2, 50)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mean_rmse, std_rmse = [], []
mean_r2, std_r2 = [], []

for alpha in alphas:
    rmses = []
    r2s = []
    for train_idx, test_idx in kf.split(X):
        model = Ridge(alpha=alpha)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[test_idx], y_pred)))
        r2s.append(r2_score(y.iloc[test_idx], y_pred))
    mean_rmse.append(np.mean(rmses))
    std_rmse.append(np.std(rmses))
    mean_r2.append(np.mean(r2s))
    std_r2.append(np.std(r2s))

mean_rmse = np.array(mean_rmse)
mean_r2 = np.array(mean_r2)
idx_min = np.argmin(mean_rmse)
alpha_min = alphas[idx_min]
rmse_min = mean_rmse[idx_min]
r2_best = mean_r2[idx_min]

print(f"Alpha (λ) terbaik: {alpha_min:.6f}")
print(f"RMSE minimum: {rmse_min:.4f}")
print(f"R^2 pada alpha terbaik: {r2_best:.4f}")
