import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]

df = sns.load_dataset("tips")  # 加载内置的 tips 数据集
fig, ax = plt.subplots()
sns.histplot(data=df, x="total_bill", ax=ax)
ax.set_title("总账单金额")
plt.show()
sns.kdeplot(data=df, x="total_bill", ax=ax)
ax.set_title("总账单金额核密度估计图")
plt.show()
