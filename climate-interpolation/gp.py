from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared

from data import load_data


def fit_gp_basic(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    kernel = 1.0 * RBF(length_scale=30.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0)
    gp.fit(X, y)
    return gp


def fit_gp_periodic(X: np.ndarray, y: np.ndarray,
                    init_len_scale: float = 30.0,
                    init_period: float = 11.0) -> GaussianProcessRegressor:
    kernel = 1.0 * (RBF(length_scale=init_len_scale) +
                    ExpSineSquared(length_scale=50.0, periodicity=init_period)) \
             + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0)
    gp.fit(X, y)
    return gp


def predict_on_grid(gp: GaussianProcessRegressor, year_min: int, year_max: int, n_points: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_pred = np.linspace(year_min, year_max, n_points).reshape(-1, 1)
    y_mean, y_std = gp.predict(X_pred, return_std=True)
    return X_pred, y_mean, y_std


def plot_gp(years: np.ndarray,
            anomalies: np.ndarray,
            X_pred: np.ndarray,
            y_mean: np.ndarray,
            y_std: np.ndarray,
            title: str = "Gaussian Process Regression: Global Temperature Anomalies",
            baseline_span: tuple[int, int] = (1901, 2000),
            save_path: Path | None = None) -> None:
    plt.figure(figsize=(10, 5))

    plt.scatter(years, anomalies, s=15, alpha=0.7, label="Observed anomalies")
    plt.plot(X_pred.ravel(), y_mean, linewidth=2, label="GP mean prediction")
    
    upper = y_mean + 1.96 * y_std
    lower = y_mean - 1.96 * y_std
    plt.fill_between(X_pred.ravel(), lower, upper, alpha=0.2, label="95% confidence interval")

    plt.axhline(0, linestyle="--", linewidth=1, color="black")
    plt.axvspan(baseline_span[0], baseline_span[1], alpha=0.08, label="Baseline (1901–2000)")

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Anomaly (°C, vs 1901–2000 mean)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def main(use_periodic: bool = False, out_file: str | None = None) -> None:
    df = load_data()
    X = np.array(df["Year"].values).reshape(-1, 1)
    y = np.array(df["Anomaly"].values)

    if use_periodic:
        gp = fit_gp_periodic(X, y)
        title = "GP (RBF + Periodic + Noise): Global Temp Anomalies"
    else:
        gp = fit_gp_basic(X, y)
        title = "GP (RBF + Noise): Global Temp Anomalies"

    print("\nLearned kernel:\n", gp.kernel_, "\n")

    X_pred, y_mean, y_std = predict_on_grid(gp, int(X.min()), int(X.max()), n_points=600)
    save_path = Path(out_file) if out_file else None
    plot_gp(
        years=X.ravel(),
        anomalies=y,
        X_pred=X_pred,
        y_mean=y_mean,
        y_std=y_std,
        title=title,
        save_path=save_path
    )


if __name__ == "__main__":
    main(use_periodic=False, out_file=None)