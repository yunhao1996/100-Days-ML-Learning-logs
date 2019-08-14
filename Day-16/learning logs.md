# 100_ML_Day16
## 高斯核，非线性数据分类
```python
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 利用随机噪声得到一个异或数据集
np.random.seed(0)  # 随机数种子，保证每次运行数据一样
X_xor = np.random.randn(200, 2)  # 200行2列的服从N(0,1)的随机样本值
A = (X_xor[:, 0] > 0)    # 满足条件，A为True,不满足为False
B = (X_xor[:, 1] > 0)
y_xor = np.logical_xor(A, B)   #相异为真，相同为假
y_xor = np.where(y_xor>0, 1, 0)  # 满足y_xor>0条件，y_nor = 1;不满足y_nor = 0
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], #显示数据
            c='b', marker='x', label='l')
plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
            c='r', marker='s', label='0')
plt.legend()  # 显示标签
plt.show()
# print(A)
# print(B)
# 对训练集应用SVM
classifier = SVC(kernel = 'rbf', random_state = 0, gamma = 0.1, C = 10.0)
classifier = classifier.fit(X_xor, y_xor)

# 可视化
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='b', marker='x', label='l')
plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
            c='r', marker='s', label='0')
# 横轴范围
x1_min, xl_max = X_xor[:, 0].min(), X_xor[:, 0].max()
# 纵轴范围
x2_min, x2_max = X_xor[:, 1].min(), X_xor[:, 1].max()
# 思路：表示出平面的所有点（合适的步长），用上面的模型预测这些点的分类，以此找到决策边界
# linspace（start，stop，num):将区间[start,stop]均匀分成num份
xx1, xx2 = np.meshgrid(
    np.linspace(
        x1_min, xl_max), np.linspace(
            x2_min, x2_max))
grid = np.array([xx1.ravel(), xx2.ravel()]).T
probs = classifier.predict(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], colors='black')
plt. title('SVM_rbf')  # 显示标题
plt. show()  # 显示图片
```
数据显示：
<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day16/blob/master/picture/Figure_1.png">
</p> 
分类显示：
<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day16/blob/master/picture/2.png">
</p> 

<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day16/blob/master/picture/3.png">
</p> 
`C`会限制每个点的重要性，很小时，决策边界接近线性模型；很大时会使决策边界弯曲将正确点分类，模型更复杂
<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day16/blob/master/picture/4.png">
</p> 

<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day16/blob/master/picture/5.png">
</p> 
`gamma`控制高斯核的宽度他决定了点与点之间“靠近”是指多大的距离，数值很小，半径较大；数值大，半径小，模型更复杂  
`C``gamma`要同时调节
