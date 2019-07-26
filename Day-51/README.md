# 深入研究---matplotlib

## 1.matplotlib数据可视化

```python
1.1 设置绘图风格以及显示图像
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)  # 0-10之间均匀取100个数

plt.plot(x, np.sin(x))  # 线的形式表示正弦函数
plt.plot(x, np.cos(x))
plt.style.use('classic')  # 经典风格

plt.show()  # 显示


```
