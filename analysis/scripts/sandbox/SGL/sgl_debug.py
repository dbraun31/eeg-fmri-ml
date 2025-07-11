import numpy as np
from groupyr import SGL
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Create synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

# Define dummy groups (e.g., 5 groups of 4 features)
groups = [np.arange(i*4, (i+1)*4) for i in range(5)]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

alphas = [1e-5, 1.0, 1e5]

for alpha in alphas:
    model = SGL(alpha=alpha, l1_ratio=0.5, groups=groups, max_iter=10000, tol=1e-6)
    model.fit(X_scaled, y)
    coef_norm = np.linalg.norm(model.coef_)
    nonzero_count = np.sum(model.coef_ != 0)
    preds = model.predict(X_scaled)
    rmse = np.sqrt(np.mean((preds - y)**2))

    print(f"Alpha: {alpha:.1e}")
    print(f"Coefficient norm: {coef_norm:.6f}")
    print(f"Non-zero coefficients: {nonzero_count}")
    print(f"Training RMSE: {rmse:.6f}\n")

