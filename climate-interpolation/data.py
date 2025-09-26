import pandas as pd

data_url = "https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v6/access/timeseries/aravg.ann.land_ocean.90S.00N.v6.0.0.202508.asc"

df = pd.read_csv(data_url, sep=r"\s+", header=None)

# Only keep year, anomaly
df = df[[0, 1]]
df.columns = ["Year", "Anomaly"]

# Drop missing values
df = df[df["Anomaly"] != -999.000000]

print(df.head())