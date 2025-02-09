import numpy as np
import matplotlib.pyplot as plt

arr = np.array(
    [
        [0, 0],
        [0, 0.3333],
        [0, 0.6667],
        [0, 1],
        [0.3333, 1],
        [0.6667, 1],
        [1, 1],
    ]
)
plt.rcParams["font.sans-serif"] = ["Cascadia Code"]
plt.figure(figsize=(5, 5))
plt.gca().spines["right"].set_visible(False)  # 隐藏右侧边框
plt.gca().spines["top"].set_visible(False)  # 隐藏上侧边框
plt.plot(arr[:, 0], arr[:, 1])
plt.scatter(arr[:, 0], arr[:, 1], s=100)
plt.axis("equal")
plt.xlim(-0.1, 1.1)
plt.ylim(0, 1.1)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
plt.show()
