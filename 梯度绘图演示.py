import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotate(v, theta, p):
    r = np.array(
        [
            [
                np.cos(theta) + v[0] ** 2 * (1 - np.cos(theta)),
                v[0] * v[1] * (1 - np.cos(theta)) - v[2] * np.sin(theta),
                v[0] * v[2] * (1 - np.cos(theta)) + v[1] * np.sin(theta),
            ],
            [
                v[0] * v[1] * (1 - np.cos(theta)) + v[2] * np.sin(theta),
                np.cos(theta) + v[1] ** 2 * (1 - np.cos(theta)),
                v[1] * v[2] * (1 - np.cos(theta)) - v[0] * np.sin(theta),
            ],
            [
                v[0] * v[2] * (1 - np.cos(theta)) - v[1] * np.sin(theta),
                v[1] * v[2] * (1 - np.cos(theta)) + v[0] * np.sin(theta),
                np.cos(theta) + v[2] ** 2 * (1 - np.cos(theta)),
            ],
        ]
    )
    return r @ p


# 定义 x 和 y 的范围
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)
Z = X**2 + X * Y + Y**2

# 创建三维图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# 绘制表面图
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5)

# 绘制梯度直线
x_point = np.array([1, 1.5])
y_point = np.array([1, 1.5])
z_point = x_point**2 + x_point * y_point + y_point**2
ax.plot(x_point, y_point, z_point, "r")
ax.scatter(x_point[0], y_point[0], z_point[0], s=8, color="r")

# 绘制梯度箭头
p = np.array([x_point[0] - x_point[1], y_point[0] - y_point[1], z_point[0] - z_point[1]])
y_x = p[0] / p[1]
z_x = np.sqrt(p[0] ** 2 + (y_x * p[0]) ** 2) / p[2]
v = np.array([-1 / (1 + y_x**2 + z_x**2), -y_x / (1 + y_x**2 + z_x**2), -z_x / (1 + y_x**2 + z_x**2)])
p1 = 0.2 * rotate(v, np.pi / 7, p) + np.array([x_point[1], y_point[1], z_point[1]])
p2 = 0.2 * rotate(v, -np.pi / 7, p) + np.array([x_point[1], y_point[1], z_point[1]])
ax.plot([x_point[1], p1[0]], [y_point[1], p1[1]], [z_point[1], p1[2]], "r")
ax.plot([x_point[1], p2[0]], [y_point[1], p2[1]], [z_point[1], p2[2]], "r")

# 设置坐标轴标签
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(x, y)")
ax.grid(False)
plt.show()
