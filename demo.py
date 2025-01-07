import pandas as pd

df = pd.read_csv("data/heart_attack_china_youth_vs_adult.csv", index_col=0)
print(df.info())
