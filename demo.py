import numpy as np
import pandas as pd


def f(x, y):
    if y == 0:
        return np.nan
    return x / y


df = pd.DataFrame({"a": [10, 20, 30], "b": [40, 0, 60]})
f_vec = np.vectorize(f)
print(f_vec(df["a"], df["b"]))  # [0.25  nan 0.5 ]
