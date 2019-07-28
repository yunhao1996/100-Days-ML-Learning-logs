
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
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
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
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
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
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
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
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
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
  <img src="https://github.com/yunhao1996/100_ML_Day3/blob/master/微信图片_20190417210717.png">
</p>

## 4.多子图


