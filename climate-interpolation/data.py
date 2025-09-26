import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_url = "https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v6/access/timeseries/aravg.ann.land_ocean.90S.00N.v6.0.0.202508.asc"

def load_data() -> pd.DataFrame:
    """ Load the NOAA global surface temperature anomaly data. """
    
    df = pd.read_csv(data_url, sep=r"\s+", header=None)

    # Only keep year, anomaly
    df = df[[0, 1]]
    df.columns = ["Year", "Anomaly"]

    # Drop missing values
    df = df[df["Anomaly"] != -999.000000]
    
    return df

def plot_data(df: pd.DataFrame):
    """ Plot the temperature anomaly data. """

    baseline_mean = df[(df["Year"] >= 1901) & (df["Year"] <= 2000)]["Anomaly"].mean()

    plt.figure(figsize=(8, 6))
    plt.plot(df["Year"], df["Anomaly"], marker="o", linestyle="-", label="Anomaly Temperature (°C)")
    plt.axhline(baseline_mean, color="red", linewidth=0.8, linestyle="--")
    plt.title("Global Surface Temperature Anomalies")
    plt.xlabel("Year")
    plt.ylabel("Anomaly Temperature (°C)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    plot_data(df)