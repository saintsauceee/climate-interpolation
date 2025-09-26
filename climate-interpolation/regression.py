import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from data import load_data

if __name__ == "__main__":
    df = load_data()
    X = df["Year"].to_numpy().reshape(-1, 1)
    y = df["Anomaly"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Linear trend: {slope:.4f} °C per year, intercept: {intercept:.4f} °C")

    y_pred = model.predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(df["Year"], y, alpha=0.8, label="Observed Anomalies")
    plt.plot(df["Year"], y_pred, color="red", linewidth=2, label=f"Linear Trend: {slope:.4f} °C/year")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Global Surface Temperature Anomalies with Linear Trend")
    plt.xlabel("Year")
    plt.ylabel("Anomaly Temperature (°C)")
    plt.legend()
    plt.grid()
    plt.show()