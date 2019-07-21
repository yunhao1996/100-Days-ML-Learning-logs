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
显示：  
原始数据图：  
<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>
挑选数据图;
<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>

1.4 例子：装箱数据
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # 定义随机数种子
x = np.random.randn(100)  # 生成[0，1）之间的一维数据

# 直方图
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)  # 生成维度和bins一样的全零数组
print(counts.shape)

# 装箱，统计每个区间的个数
i = np.searchsorted(bins, x)  # 判断x在bins中哪两个bins[n-1],bins[n]之间，并返回n-1
print(i.shape)

# 每个箱子加1
np.add.at(counts, i, 1)  # counts[i]+=1

# 绘制结果
plt.plot(bins, counts, linestyle='steps')
# plt.hist(x, bins, histtype='step')  # 简单的做法
plt.show()
```
显示：
<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>
2.1快速排序数组
```python
输入：
import numpy as np

x = np.array([2, 1, 4, 3, 5])
b = np.sort(x)

print(b)
# print(x.sort())

i = np.argsort(x)  # 返回已排序数组的索引值
print(i)
输出：
[1 2 3 4 5]
[1 0 3 2 4]
``` 
2.2 按行或者列进行排序
```python
import numpy as np

rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(np.sort(X, axis=0))  # 按列进行排序
print(np.sort(X, axis=1))  # 按行进行排序
```
2.3 部分分类
```python
输入:
import numpy as np

# 以排序后的第3个数，即3进行分区，小于3的元素2，1位于3的前面，大于等于3的元素在后面
x = np.array([7, 2, 3, 1, 6, 5, 4])
a = np.partition(x, 3)  
print(a)
输出：
[2 1 3 4 6 5 7]
```
3.1 结构化数据
```python
输入：
import numpy as np

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

# 这里“U10”翻译为“最大长度为10的Unicode字符串”，
# “i4”翻译为“4字节(即和“f8”翻译成“8字节(即。， 64位)浮动。
data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'),
                          'formats': ('U10', 'i4', 'f8')})

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)
输出：
[('Alice', 25, 55. ) ('Bob', 45, 85.5) ('Cathy', 37, 68. )
 ('Doug', 19, 61.5)]
```
3.2创建结构化数组的几种形式
```python
1. np.dtype({'names': ('name', 'age', 'weight'),
          'formats': ((np.str_, 10), int, np.float32)})  # python数据类型
2. np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])   
3. np.dtype('S10,i4,f8')
```
3.3 更高级的复合类型，每个元素包含一个数组或者值矩阵
```python
输入：
import numpy as np

tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)  # 填充字典
print(X)
print(X['mat'][0])  # 输出字典中mat的信息
输出：
[(0, [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]

[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
```
