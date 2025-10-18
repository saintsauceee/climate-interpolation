import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from data import load_data

if __name__ == "__main__":
    df = load_data()
    X = np.array(df["Year"].values).reshape(-1, 1)
    y = np.array(df["Anomaly"].values)

    kernel = 1.0 * RBF(length_scale=30.0) + WhiteKernel(noise_level=0.1)

    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X, y)

    X_pred = np.linspace(df["Year"].min(), df["Year"].max(), 500).reshape(-1, 1)
    y_pred, y_std = gp.predict(X_pred, return_std=True)  # type: ignore

    plt.figure(figsize=(10,5))
    plt.scatter(X, y, s=15, alpha=0.6, label="Observed")
    plt.plot(X_pred, y_pred, "r", lw=2, label="GP mean prediction")
    plt.fill_between(X_pred.ravel(), y_pred - 2*y_std, y_pred + 2*y_std, alpha=0.2, color="red", label="95% confidence interval")

    plt.axhline(0, color="black", ls="--", lw=1)
    plt.title("Gaussian Process Regression: Global Temp Anomalies")
    plt.xlabel("Year")
    plt.ylabel("Anomaly (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()