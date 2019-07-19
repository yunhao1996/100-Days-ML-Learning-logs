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
