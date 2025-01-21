import numpy as np
import time

np.set_printoptions(linewidth=500)


def func1():
    global k1
    loss = x @ k1 - y
    loss = np.sum(loss**2)
    loss2 = x @ (k1 + np.eye(11) * delta_k) - y
    loss2 = np.sum(loss2**2, axis=0).reshape(-1, 1)
    gradient = (loss2 - loss) / (delta_k * 2)
    k1 -= learn_rate * gradient
    return loss


def func2():
    global k2
    loss = x @ k2 - y
    gradient = x.T @ loss
    k2 -= learn_rate * gradient
    return np.sum(loss**2)


sample_count = 50
feature_count = 10
x = np.random.randint(0, 10, (sample_count, feature_count))  # 5个样本,10个特征
x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))  # 添加一列1用来和偏置运算
y = np.random.randint(0, 10, (sample_count, 1))  # 5个标签
k = np.random.rand(x.shape[1], 1)  # 1个偏置，10个权重
learn_rate = 1e-5  # 学习率
delta_k = 1e-8
epoch = 0
k1 = k.copy()
k2 = k.copy()
# while True:
#     loss1 = func1()
#     loss2 = func2()
#     if loss1 < 1e-5 and loss2 < 1e-5:
#         print(loss1, "\t", loss2, "\t", epoch)
#         break
#     epoch += 1
#     if epoch % 10000 == 0:
#         print(loss1, "\t", loss2, "\t", epoch)
