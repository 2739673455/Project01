import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

plt.rcParams["font.sans-serif"] = ["kaiti"]

# 生成线性可分的数据
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)

# 训练线性可分SVM（硬间隔，C设为很大值）
clf = svm.SVC(kernel="linear", C=1e5)
clf.fit(X, y)

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)


ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界和间隔
plt.contour(xx, yy, Z, colors="k", levels=[-1, 0, 1], alpha=0.7, linestyles=["--", "-", "--"])
plt.xticks([])
plt.yticks([])
plt.show()
