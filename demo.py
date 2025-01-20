import numpy as np

np.set_printoptions(linewidth=500)


def func1(k1):
    loss = y - (k1 @ x)
    loss = np.sum(loss**2)
    if loss < 1e-1:
        return k1, loss
    loss2 = y - ((k1 + np.eye(11) * delta_k) @ x)
    loss2 = np.sum(loss2**2, axis=1)
    gradient = (loss2 - loss) / (delta_k * 2)
    k1 -= learn_rate * gradient
    return k1, loss


def func2(k2):
    loss = k2 @ x - y
    if np.sum(loss**2) < 1e-1:
        return k2, loss
    gradient = loss @ x.T
    k2 -= learn_rate * gradient
    return k2, loss


sample_count = 5
x = np.random.randint(0, 10, (10, sample_count))  # 5个样本,10个特征
x = np.vstack((np.ones(shape=(1, x.shape[1])), x))  # 添加一行1用来和偏置运算
y = np.random.randint(0, 10, (1, sample_count))  # 5个标签
k = np.random.rand(1, 11)  # 1个偏置，10个权重
learn_rate = 1e-5  # 学习率
delta_k = 1e-8
epoch = 0
k1 = k.copy()
k2 = k.copy()
while True:
    k1, loss1 = func1(k1)
    k2, loss2 = func2(k2)
    if loss1 < 1e-3 and loss2 < 1e-3:
        break
    epoch += 1
    if epoch % 1000 == 0:
        print(loss1, "\t", np.sum(loss2**2), "\t", epoch)
