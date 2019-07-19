# 深入研究-Numpy
1.两种不同的求和方式
```
输入：
import numpy as np
import time

big_array = np.random.rand(1000000)
a = time.clock()
np.sum(big_array)
b = time.clock()
sum(big_array)
c = time.clock()
t1 = b-a
t2 = c-b
print('t1：', t1)
print('t1：', t2)
输出：
t1： 0.0011928000000000002
t1： 0.0649628
结论：np.sum()的运行速度要比sum(()的运行速度要快许多，又如np.max()也比max()运行速度快
```
2.例子：总统的平均身高是多少？
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('D:\\ML_100\\46\\height.csv')  # 导入数据
heights = np.array(data['height(cm)'])  # 获取身高信息

# print(heights)  # 输出身高
# print("Mean height:       ", heights.mean())  # 身高平均值
# print("Standard deviation:", heights.std())  # 身高标准差
# print("Minimum height:    ", heights.min())  # 身高最小值
# print("Maximum height:    ", heights.max())  # 身高最大值
# print("25th percentile:   ", np.percentile(heights, 25))  # 第25%分位的数值
# print("Median:            ", np.median(heights))  # 中位数
# print("75th percentile:   ", np.percentile(heights, 75))

# 可视化身高信息
plt.hist(heights)  # 直方图
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number');
plt.show()
```
结果:
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-46/1.png">
</p>
3.数组相加
```python
输入：
import numpy as np

a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
M = np.ones((3, 3))

print('a+b：', a + b)
print('a+5：', a+5)
print('M+a：', M+a)
输出：
a+b: [5 6 7]
a+5： [5 6 7]
M+a: [[1. 2. 3.]
 [1. 2. 3.]
 [1. 2. 3.]]
```
4.画二维函数
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 50)  # 0-5之间线性划分50个数
y = np.linspace(0, 5, 50)[:, np.newaxis]  # 列扩维
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)  # 二维函数
# print(z)

# 图示z，extent：限制坐标范围
plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],
           cmap='binary')
plt.colorbar()  # 图配渐变色
plt.show()
```
显示：
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-46/2.png">
</p>
5.比较、掩码和布尔逻辑
5.1 计算雨天
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

#使用panda提取降雨量英寸作为一个数字数组
rainfall = pd.read_csv('D:\\ML_100\\46\\data46.csv')['PRCP'].values
inches = rainfall / 254.0  # 换算单位
print(inches.shape)
seaborn.set()  # 设置网格风格
plt.hist(inches, 40)  # 画直方图，参数40设置柱状图的粗细
plt.show()
```
显示：
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-46/3.png">
</p>
5.2 条件判断，输出布尔型
```python
输入：
import numpy as np

x = np.array([1, 2, 3, 4, 5])
print(x < 3)
print((2 * x) == (x ** 2))
输出：
[ True  True False False False]
[False  True False False False]
```
5.3 处理布尔型数据
```python
import numpy as np

x = np.random.randint(10, size=(3, 4))
print(x)
a = np.count_nonzero(x < 6)  # 判断有多少数小于6
print(a)
输出：
[[8 4 2 1]
 [9 4 1 8]
 [9 9 6 6]]
5
```
5.4 布尔型运算
```python
import numpy as np

print(bool(0))  # 将参数转为布尔型，没有参数返回False
print(bool(74))
print(bin(42 & 59))  # 按位与
输出：
False
True
0b101010
```



