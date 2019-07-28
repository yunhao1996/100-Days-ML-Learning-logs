# 深入研究--MATPLOTLIB

## 画三维图
```python
画三维图时，需要给轴传毒关键字，projection='3d'  
from mpl_toolkits import mplot3d  # 导入三维绘图工具包,包含需要传递的关键
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')  # 给项目传递关键字，3维绘图
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>

## 三维点和线的可视化
```python
from mpl_toolkits import mplot3d  # 导入三维绘图工具包,包含需要传递的关键
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')  # 给项目传递关键字，3维绘图

# 线数据
zline = np.linspace(0, 15, 1000)  # 0-15均匀分成1000份
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# 点数据
zdata = 15 * np.random.random(100)  # 一维数据，从0-1中随机冲去抽取
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)  # 正态分布的随机样本数
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')  # 画三维点图
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>

## 三维等高线图
```python
from mpl_toolkits import mplot3d  # 导入三维绘图工具包,包含需要传递的关键
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):

    return np.sin(np.sqrt(x ** 2 + y ** 2))  # 平方根计算

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)  # 用两个坐标轴上的点在平面上画网格
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')  # 创建3维图
ax.contour3D(X, Y, Z, 50, cmap='binary')  # 画等高线3维图
ax.set_xlabel('x')  # 设置label
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>

## 线框图和曲面图

```python
线框图：
from mpl_toolkits import mplot3d  # 导入三维绘图工具包,包含需要传递的关键
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):

    return np.sin(np.sqrt(x ** 2 + y ** 2))  # 平方根计算

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)  # 用两个坐标轴上的点在平面上画网格
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')  # 画线框图,函数plt.wireframe
ax.set_title('wireframe')

plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>

```python
曲面图：
from mpl_toolkits import mplot3d  # 导入三维绘图工具包,包含需要传递的关键
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):

    return np.sin(np.sqrt(x ** 2 + y ** 2))  # 平方根计算

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)  # 用两个坐标轴上的点在平面上画网格
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
# 曲面图，rstride: 行的跨度，cstride：列的跨度
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,  # 曲面图
                cmap='viridis', edgecolor='none')
ax.set_title('surface')
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>

```python
三角网络模型
from mpl_toolkits import mplot3d  # 导入三维绘图工具包,包含需要传递的关键
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)

# 绘制散点图
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
plt.show()
# 补充表面，plot_trisurf
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
ax.plot_trisurf(x, y, z,
                cmap='viridis', edgecolor='none')  #

plt.show()

```

<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>

<p align="center">
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>
