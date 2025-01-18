import numpy as np
import time

np.set_printoptions(linewidth=150)
x = np.random.randint(0, 10, (10, 5))  # 5个样本,10个特征
x = np.vstack((np.ones(shape=(1, x.shape[1])), x))  # 添加一行1用来和偏置运算
y = np.random.randint(0, 10, (1, 5))  # 5个标签
k = np.random.rand(1, 11)  # 1个偏置，10个权重
learn_rate = 1e-4  # 学习率
delta_k = np.eye(11) * 1e-5

epoch = 0
k1 = k.copy()
while True:
    epsilon = y - (k1 @ x)
    epsilon = np.sum(epsilon**2)
    if epsilon < 1e-5:
        break
    epsilon2 = y - ((k1 - delta_k) @ x)
    epsilon2 = np.sum(epsilon2**2, axis=1)
    k1 += learn_rate * (epsilon2 - epsilon) / 1e-5
    epoch += 1
print(epsilon, epoch)

epoch = 0
k2 = k.copy()
while True:
    loss = k2 @ x - y
    if np.sum(loss**2) < 1e-5:
        break
    gradient = loss @ x.T / 5
    k2 -= learn_rate * gradient
    epoch += 1
print(np.sum(loss**2), epoch)
