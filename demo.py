import pandas as pd

df = pd.read_csv("data/weather.csv")
df["date"] = pd.to_datetime(df["date"])
print(df.info())
