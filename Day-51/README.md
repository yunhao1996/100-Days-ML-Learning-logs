# 深入研究---matplotlib

## 1.matplotlib数据可视化

```python
1.1 设置绘图风格以及显示图像
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)  # 0-10之间均匀取100个数
fig = plt.figure()
plt.plot(x, np.sin(x))  # 线的形式表示正弦函数
plt.plot(x, np.cos(x))
plt.style.use('classic')  # 经典风格
fig.savefig('my_figure.png')  # 保存图片到当前目录

plt.show()  # 显示
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-51/pictures/1.png">
</p>

```python
1.2 matlab风格界面
import matplotlib.pyplot as plt
import numpy as np

# matlab风格的工具
plt.figure()  # create a plot figure
x = np.linspace(0, 10, 100)
# create the first of two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))

# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-51/pictures/2.png">
</p>

## 2. 简易线性图
```python
2.1 线性图
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')
fig = plt.figure()  # 创建图
ax = plt.axes()  # 创建轴

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x))
# plt.plot(x, np.sin(x))

plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-51/pictures/3.png">
</p>

## 3.简易散点图
```python
 3.1 plot画散点图
 
 import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.style.use('seaborn-whitegrid')  # 设置画图风格
plt.plot(x, y, 'o', color='black')
plt.show()
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-51/pictures/4.png">
</p>
 ```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(0)  # 固定随机数种子
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:  # 每种类型各画5个
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8)
plt.show()
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-51/pictures/5.png">
</p>
 ```python
3.2 scatter的用法
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.scatter(x, y, marker='o')  # 散点函数
plt.show()
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-51/pictures/6.png">
</p>
```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(0)
x = rng.randn(100)  # 标准正态分布的随机数
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

# 设定透明度和尺寸等属性
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='viridis')

plt.colorbar()  # 显示颜色变化模型柱状图
plt.show()
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-51/pictures/7.png">
</p>

## 4 可视化异常处理
```python
4.1 plt.errorbar()函数用于表现有一定置信区间的带误差数据。
plt.errorbar(x,   
	y,   
	yerr=None,  
	xerr=None,  # xerr,yerr: 数据的误差范围  
	fmt='',   # 数据点的标记样式以及相互之间连接线样式
	ecolor=None, 
	elinewidth=None,   # 误差棒的线条粗细
	capsize=None,   # 误差棒边界横杠的大小
	capthick=None  # 误差棒边界横杠的厚度
)
程序：
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.style.use('seaborn-whitegrid')
plt.errorbar(x, y, yerr=dy, fmt='.k')
plt.show()
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-51/pictures/8.png">
</p>
```python
当数据拥挤时，淡化误差，突出数据点，修改如下：
plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
```    
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-51/pictures/9.png">
</p>


## 5.密度线和等高线
```python
5.1 可视化一个三维函数
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)  # 生成网格点坐标矩阵
Z = f(X, Y)  # 指定坐标点出的高度
# 使用颜色映射来区分不同的高度,在数据范围内绘制3个等间距的间隔
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)  # 标记高度

# 渲染上面的图像，规定区域大小，extend.自适应
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy', alpha=0.5)
plt.colorbar()
plt.show()
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-51/pictures/10.png">
</p>


