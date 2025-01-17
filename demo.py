import numpy as np

np.set_printoptions(linewidth=150)
x = np.random.randint(0, 10, (10, 5))  # 5个样本,10个特征
x = np.vstack((np.ones(shape=(1, x.shape[1])), x))  # 添加一行1用作与偏置运算
y = np.random.randint(0, 10, (1, 5))  # 5个标签
k = np.random.rand(1, 11)  # 1个偏置，10个权重
learn_rate = 0.01  # 学习率
delta_k = np.eye(11) * 0.001
epsilon = y - (k @ x)  # 残差
epsilon2 = y - ((k - delta_k) @ x)
print(epsilon2 - epsilon)  # 残差的变化
print(k - delta_k)  # 权重的变化
