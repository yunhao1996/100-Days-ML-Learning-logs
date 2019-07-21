# 深入研究-NUMPY-花哨的索引，数组排序，结构化数据
1.1花式索引：
```python
输入：
import numpy as np

rand = np.random.RandomState(42)  # 随机数种子，rand每次生成的数据相同
x = rand.randint(100, size=10)  # 返回[0,100)区间的随机整数,一维
print('原始数组', x)

# 使用花式索引时，结果的形状反映索引数组的形状，而不是被索引数组的形状
ind = [3, 7, 4]
print('一维：', x[ind])

ind1 = np.array([[3, 7],
                 [4, 5]])
print('二维：', x[ind1])
输出：
原始数组 [51 92 14 71 60 20 82 86 74 74]
一维： [71 86 60]
二维： [[71 86]
 [60 20]]
```
1.2 花式索引
```python
输入：
import numpy as np

X = np.arange(12).reshape((3, 4))
print('原始数组：\n', X)

row = np.array([0, 1, 2])
col = np.array([2, 1, 3])

print('索引数组1：\n', X[row, col])  # 第一个索引指向行,第二个索引指向列
print('索引数组2;\n', X[row[:, np.newaxis], col])  # 列向量和行向量组合，输出二维数组
输出：
原始数组：
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
索引数组1：
 [ 2  5 11]
索引数组2;
 [[ 2  1  3]
 [ 6  5  7]
 [10  9 11]]
```
1.3 例子：选择随机点
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn

rand = np.random.RandomState(42)
mean = [0, 0]
cov = [[1, 2],
       [2, 5]]

# np.random.multivariate_normal:生成一个多元正态分布矩阵
# 参数：mean是多维分布的均值维度；cov是协方差矩阵，size指定矩阵的维度
X = rand.multivariate_normal(mean, cov, 100)  # (100, 2)

seaborn.set()  # 设置绘图风格
plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# 从100个选择20个数，replace为False表示不放回取，也就是不会重复
indices = np.random.choice(X.shape[0], 20, replace=False)  # 返回的相当于索引值
selection = X[indices]

plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1],
            facecolor='red', s=200)
plt.show()

```
"显示"：  
原始数据图：  
<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>
挑选数据图;
<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>

```python
```
```python
```
```python
```
```python
```
```python
```
```python
```

```python
```
