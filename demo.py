import time
import numpy as np
import pandas as pd
import pymysql
from pyecharts.charts import Bar3D

sheet_names = ["2015", "2016", "2017", "2018", "会员等级"]
sheet_datas = pd.read_excel("data/sales.xlsx", sheet_name=sheet_names)
