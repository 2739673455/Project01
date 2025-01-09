import numpy as np
import pandas as pd

df = pd.read_csv("data/employees.csv")  # 读取员工数据


def f(x):
    if x["job_id"] == "IT_PROG":
        return 1.1 * x["salary"]
    else:
        return x["salary"]


print(df["salary"] * df["commission_pct"] + df["salary"])
