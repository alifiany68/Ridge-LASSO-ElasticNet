import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

data = pd.read_excel('/storage/emulated/0/karil/Makro/data_standarisasi.xlsx', sheet_name='Sheet1')

X = data[['TPT', 'UMP', 'Remitansi', 'GDP', 'Dummy']].values
y = data['Migrasi'].values
var_names = ['TPT', 'UMP', 'Remitansi', 'GDP', 'Dummy']

# cross-validation dan scorer 
cv = KFold(n_splits=5, shuffle=True, random_state=42)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
neg_rmse_scorer = make_scorer(rmse, greater_is_better=False)

results = []

#  Ridge Regression (optimasi alpha)
ridge_params = {'alpha': np.logspace(-4, 2, 50)}
ridge = Ridge(max_iter=10000, random_state=42)
ridge_grid = GridSearchCV(ridge, ridge_params, cv=cv, scoring=neg_rmse_scorer, refit=True)
ridge_grid.fit(X, y)
ridge_best = ridge_grid.best_estimator_
ridge_r2 = np.mean(cross_val_score(ridge_best, X, y, cv=cv, scoring='r2'))
ridge_rmse = np.mean(-cross_val_score(ridge_best, X, y, cv=cv, scoring=neg_rmse_scorer))
results.append({
    'Model': 'Ridge',
    'Best Params': f"alpha={ridge_best.alpha:.6f}",
    'R2': ridge_r2,
    'RMSE': ridge_rmse,
    'Coef': ridge_best.coef_
})

# Lasso Regression (optimasi alpha)
lasso_params = {'alpha': np.logspace(-4, 2, 50)}
lasso = Lasso(max_iter=10000, random_state=42)
lasso_grid = GridSearchCV(lasso, lasso_params, cv=cv, scoring=neg_rmse_scorer, refit=True)
lasso_grid.fit(X, y)
lasso_best = lasso_grid.best_estimator_
lasso_r2 = np.mean(cross_val_score(lasso_best, X, y, cv=cv, scoring='r2'))
lasso_rmse = np.mean(-cross_val_score(lasso_best, X, y, cv=cv, scoring=neg_rmse_scorer))
results.append({
    'Model': 'Lasso',
    'Best Params': f"alpha={lasso_best.alpha:.6f}",
    'R2': lasso_r2,
    'RMSE': lasso_rmse,
    'Coef': lasso_best.coef_
})

#Elastic Net (optimasi alpha & l1_ratio)
enet_params = {
    'alpha': np.logspace(-4, 2, 20),
    'l1_ratio': np.linspace(0.1, 0.9, 9)
}
enet = ElasticNet(max_iter=10000, random_state=42)
enet_grid = GridSearchCV(enet, enet_params, cv=cv, scoring=neg_rmse_scorer, refit=True)
enet_grid.fit(X, y)
enet_best = enet_grid.best_estimator_
enet_r2 = np.mean(cross_val_score(enet_best, X, y, cv=cv, scoring='r2'))
enet_rmse = np.mean(-cross_val_score(enet_best, X, y, cv=cv, scoring=neg_rmse_scorer))
results.append({
    'Model': 'ElasticNet',
    'Best Params': f"alpha={enet_best.alpha:.6f}, l1_ratio={enet_best.l1_ratio:.2f}",
    'R2': enet_r2,
    'RMSE': enet_rmse,
    'Coef': enet_best.coef_
})

# perbandingan
print("Perbandingan Ridge, Lasso, dan Elastic Net (5-fold CV, optimasi hyperparameter, random_state=42)")
print("-----------------------------------------------------------------------------------------------")
print(f"{'Model':<12}{'R2':>8}{'RMSE':>10}{'Best Params':>35}{'Koefisien':>35}")
for res in results:
    print(f"{res['Model']:<12}{res['R2']:8.4f}{res['RMSE']:10.4f}{res['Best Params']:>35}   {np.round(res['Coef'], 4)}")
print("-----------------------------------------------------------------------------------------------")
print("Urutan variabel:", var_names)
