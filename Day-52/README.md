# 深入研究--MATPLOTLIB

## 1.直方图  
```python
1.1 一维直方图
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-white')

np.random.RandomState(1)
data = np.random.randn(1000)

plt.hist(data, bins=30,  # 指定每个箱子的个数，也就是条状图的个数
         normed=True,  # 每个条状图的占比例比,默认为1
         alpha=0.5,  # 透明度
         histtype='stepfilled',  # 线条的类型
         color='steelblue',  # 铁青色
         edgecolor='red')  # 边缘的颜色
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-52/pictures/1.png">
</p>
  
```python
1.2 二维直方图和装箱操作
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-white')

mean = [0, 0]  # 多维分布的均值，维度为1
cov = [[1, 1], [1, 2]]  # 协方差矩阵
x, y = np.random.multivariate_normal(mean, cov, 10000).T  # 多元正态分布矩阵

plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()  # 显示颜色变化柱
cb.set_label('counts in bin')
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-52/pictures/2.png">
</p>

## 2.配置图例
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)

plt.style.use('classic')
plt.plot(x, np.sin(x), '-b', label='Sine')
plt.plot(x, np.cos(x), '--r', label='Cosine')
plt.axis('equal')
# 显示图例,upper left: 表示位置在左上方，frameon:表示是否加框
# 常见的参数 fancybox:控制是否应在构成图例背景的FancyBboxPatch周围启用圆边
# framealpha: 控制框架透明度
# shadow: 控制是否在图例后面画一个阴影
# borderpad: 图例边框的内边距
# scatteryoffsets:为散点图图例条目创建的标记的垂直偏移量
# labelspacing: 图例条目之间的垂直间距
leg = plt.legend(loc='upper left', frameon=False)  # 显示图例
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-52/pictures/3.png">
</p>

## 3.配置颜色条--colorbar
```python
import matplotlib.pyplot as plt
plt.style.use('classic')  # 设置绘画风格
import numpy as np

x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])

plt.imshow(I, cmap='gray')  # 映射为灰度
plt.colorbar()  # 配置颜色条函数
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-52/pictures/4.png">
</p>

```python
本小节的后面给出了一些常用的颜色映射，例如：‘jet’,'viridis','cubehelix'等。用函数的形式解释了原理，如下：
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt


def grayscale_cmap(cmap):  # 返回给定颜色的灰度版本
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # 将RGBA转换为可感知的灰度亮度
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)


def view_colormap(cmap):  # 绘制一个彩色地图与它的灰度等值
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])

view_colormap('jet')
plt.show()
```
显示‘jet’映射：

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-52/pictures/5.png">
</p>

## 4.多子图--图中图

```python
简单的设置axes
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

ax1 = plt.axes()  # 标准图
# 设置x和y轴位置0.65（宽度和高度的比例），设置轴的0.2比例
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-52/pictures/6.png">
</p>


```python
区域垂直放置，函数： add_axes()
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

fig = plt.figure()
# 新增子区域，[left, bottom, width, height],前两个量是相对的，后两个是绝对的xticklabel: x轴的刻度标记
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                   ylim=(-1.2, 1.2))

x = np.linspace(0, 10, 50)  #
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-52/pictures/7.png">
</p>

```python
对齐的列或者子图--plt.subplot
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)),  # 设置x,y 的坐标和显示字符内容
             fontsize=18, ha='center')  # 设置字体大小和文本放置位置
    
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-52/pictures/8.png">
</p>

```python
更复杂的排列方式plt.GridSpec
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

# 非对称子图，设置为2行3列，子图之间宽度间隔为0.4，高度间隔为0.3
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])  # 第一列和第二列合在一起
plt.subplot(grid[1, 2])

plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-52/pictures/9.png">
</p>

```python
更复杂的例子
import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

# 设置子图
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)  # 4行4列，高度和宽度间隔为0.2比例
main_ax = fig.add_subplot(grid[:-1, 1:])  # 不包含最后一行
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)  # x轴刻度标签没有
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)  # 内容

# histogram on the attached axes
x_hist.hist(x, 40, histtype='stepfilled',
            orientation='vertical', color='gray')
x_hist.invert_yaxis()

y_hist.hist(y, 40, histtype='stepfilled',
            orientation='horizontal', color='gray')
y_hist.invert_xaxis()
plt.show()
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-52/pictures/10.png">
</p>

## 5.文字和注释

```python
plt.annotate(s="New Year's Day", xy=('2012-1-1', 4100),  xycoords='data',
            xytext=(50, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2"))
参数：
s 为注释文本内容
xy 为被注释的坐标点
xytext 为注释文字的坐标位置
xycoords 参数如下:

figure points：图左下角的点
figure pixels：图左下角的像素
figure fraction：图的左下部分
axes points：坐标轴左下角的点
axes pixels：坐标轴左下角的像素
axes fraction：左下轴的分数
data：使用被注释对象的坐标系统(默认)
polar(theta,r)：if not native ‘data’ coordinates t

weight 设置字体线型
{‘ultralight’, ‘light’, ‘normal’, ‘regular’, ‘book’, ‘medium’, ‘roman’, ‘semibold’, ‘demibold’, ‘demi’, ‘bold’, ‘heavy’, ‘extra bold’, ‘black’}

color 设置字体颜色：
{‘b’, ‘g’, ‘r’, ‘c’, ‘m’, ‘y’, ‘k’, ‘w’}
‘black’,'red’等

[0,1]之间的浮点型数据：
RGB或者RGBA, 如: (0.1, 0.2, 0.5)、(0.1, 0.2, 0.5, 0.3)等

arrowprops  #箭头参数,参数类型为字典dict：
width：箭头的宽度(以点为单位)
headwidth：箭头底部以点为单位的宽度
headlength：箭头的长度(以点为单位)
shrink：总长度的一部分，从两端“收缩”
facecolor：箭头颜色

bbox给标题增加外框 ，常用参数如下：
boxstyle：方框外形
facecolor：(简写fc)背景颜色
edgecolor：(简写ec)边框线条颜色
edgewidth：边框线条大小
```





























