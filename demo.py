import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# 设置中文字体（如果系统支持）
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 生成更具挑战性的数据集
X, y = make_blobs(n_samples=500, centers=4, n_features=2, cluster_std=2.5, random_state=42)

# 创建不同基学习器数量的 AdaBoost 模型
n_estimators_list = [5, 10, 20, 50]  # 对应不同数量的基学习器
models = []

for n in n_estimators_list:
    # 使用决策树作为基学习器
    base_estimator = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
    ada = AdaBoostClassifier(
        estimator=base_estimator, n_estimators=n, random_state=42  # 指定基学习器  # 基学习器数量  # 固定随机种子
    )
    ada.fit(X, y)
    models.append(ada)

# 创建更高分辨率的网格
x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# 可视化设置，2x2布局
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()  # 将2D数组展平为1D以便迭代
markers = ["o", "^", "s", "D"]  # 匹配4个类别
colors = plt.cm.Set2(np.linspace(0, 1, 4))  # 使用柔和颜色方案

for ax, model, n in zip(axes, models, n_estimators_list):
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set2)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5, alpha=0.8)  # 添加边界线

    # 绘制数据点
    for i in range(4):
        class_data = X[y == i]
        ax.scatter(
            class_data[:, 0],
            class_data[:, 1],
            marker=markers[i],
            color=colors[i],
            edgecolor="k",
            s=80,
            alpha=0.9,
        )

    # 添加标题
    ax.set_title(f"AdaBoost {n}个基学习器", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

# 添加全局标题并调整布局
plt.suptitle("不同基学习器数量的AdaBoost分类效果", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
