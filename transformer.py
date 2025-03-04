import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # 定义随机失活

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # 初始化位置编码张量
        position = torch.arange(0, max_len).unsqueeze(1)  # 位置序号张量
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # 频率
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置的位置编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置的位置编码
        pe = pe.unsqueeze(0)  # 在第0维增加一个维度
        self.register_buffer(
            "pe", pe
        )  # 将位置编码张量pe注册为buffer，使其称为模型的一部分，模型保存时缓冲区中内容也会一起被保存起来

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].detach()  # 将位置编码张量与输入张量相加
        return self.dropout(x)

    def display(self):
        return self.pe


d_model = 256
positional_encoding = PositionalEncoding(d_model, 0)
pe = positional_encoding.display()
pe_np = pe.squeeze(0)[:50].detach().numpy()
# 设置画布
plt.figure(figsize=(12, 6))

# 使用Seaborn绘制热力图
ax = sns.heatmap(
    pe_np,
    cmap="viridis",
    cbar_kws={"label": "位置编码值"},
    xticklabels=50,  # 每50个维度显示一个刻度
    yticklabels=10,  # 每10个位置显示一个刻度
)

# 设置坐标轴标签和标题
ax.set_xlabel("维度")
ax.set_ylabel("在序列中的位置")
ax.set_title("位置编码热力图")

# 调整刻度方向
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

plt.show()
