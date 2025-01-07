import pandas as pd

df = pd.read_csv("data/heart_attack.csv", index_col=0)

df.groupby(["Heart_Attack", "Age_Group"])[["Smoking", "Alcohol_Consumption"]].value_counts()
# 拆分
df_group1 = df.groupby(["Heart_Attack", "Age_Group"])  # 按是否有心脏病和年龄段分组，返回一个分组对象(DataFrameGroupBy)
df_group2 = df_group1[["Smoking", "Alcohol_Consumption"]]  # 从分组对象中选择特定的列，即Smoking和Alcohol_Consumption
df_group2.value_counts()  # 对每个组内的Smoking和Alcohol_Consumption组合计数
