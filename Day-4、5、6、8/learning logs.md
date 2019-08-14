
# 学习记录_ML_Day4/5/6/8
## 逻辑回归  4.19--4.22  4.26
初看逻辑回归，感觉很难理解这到底是什么，于是决定按照程序一步一步理解。  
* 关于特征缩放*StandardScaler*:  

1.几个概念：  
`分类器`：分类的概念是在已有数据的基础上学会一个分类函数或构造出一个分类模型（即我们通常所说的分类器(*Classifier*)）  
`欧几里得距离`:简单来说，就是两点间的距离  
为什么要特征缩放？-------特征征缩放可以使机器学习算法工作的更好。比如在K近邻算法中，分类器主要是计算两点之间的欧几里得距离，如果一个特征比其它的特征有更大的范围值，那么距离将会被这个特征值所主导，也就是说对于数据产生了不同的权重。因此每个特征应该被归一化，比如将取值范围处理为0到1之间。比如：有两个点，之间的距离为无穷大，其他的点，距离接近为零。因为涉及到距离的数据，其他的距离在无穷大面前，忽略不计，那么得出的模型，无法反映出大多数数据的特征。所以，需要特征缩放。  
什么是特征缩放？------本例采用的是特征标准化，使每个特征的值有零均值和单位方差，即均值为0，方差为1  

2.测试：  
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

A = [[1, 1], [2, 2], [3, 3]]
X = np.array(A)
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)
```
输出结果：
```python
[[-1.22474487 -1.22474487]
 [ 0.          0.        ]
 [ 1.22474487  1.22474487]]
```
可以看到每一列的均值为0，经过计算机验证，方差为1.这就是*Standardscaler*的功能  

3.画图测试：
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\ML_100\Day_456\Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1 = sc.fit_transform(X)
plt.subplot(1,2,1)
plt.scatter(X[:,0] ,X[:,1] , color = 'blue')
plt.subplot(1,2,2)
plt.scatter(X1[:,0], X1[:,1], color = 'red')
plt.show()
```
输出图片：

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/Figure_1.png">
</p> 

左边图片为原始数据，右边图片为特征缩放数据，对比，可以看出，特征缩放只是把横纵坐标搞的更小了，但是，散点的分布形式完全相同。

4.这里有会涉及常见的一个错误。对自变量X的训练集和测试集正确的缩放语句：
```python
X_train= sc.fit_transform(X_train)
X_test=sc.transform(X_test)
```
错误的写法：
```python
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
```
分析：当对`X_train` `fit`时，已经得出规则，对`X_test`只需要`transform`就行，确保二者在同一个标准下。如果对`X_test`也采用`fit_tranform`,又会产生一种新的规则。错误的语句表现在图中，即散点图分布形状和原始数据不同，可按照上面的程序改写验证。

* 关于*LogisticRegression*:  

这个地方看了一些资料也没有吃透工作的原理，[参考文章](https://www.cnblogs.com/lianyingteng/p/7701801.html)，具体的公式等到第8天整理上传。  
概念理解------通俗说，就是找到一种规律，将事物进行分类，名为回归，实际就是分类。主要用于二分类问题（其结果只有两种可能），比如：`是与否`，通过逻辑函数将`是与否`转化为二进制数，即`是`为`1`,`否`为`0`。  

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/biji3.png">
</p>
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/biji4.png">
</p>

 <p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/_20190426202602.png">
</p>

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/1556281452(1).png">
</p>

* 关于混淆矩阵（confusion matrix）

对于二分类问题的评估结果，一种最全面的表示方法是使用混淆矩阵，介绍如下：  
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/123_20190421203235.png">
</p>
错误的分类反映在效果图上为红色区域的绿点，绿色区域的红点。  

* 关于可视化：  

出现了一些没有见过的函数：  
语句1：
```python
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() -1, stop=X_set[:, 0].max() +1, step=0.01), 
                     np.arange(start=X_set[:, 1].min() -1, stop=X_set[:, 1].max()+ 1, step=0.01))
```
测试分析：  
```python
import numpy as np

A=[[1, 1], [0, 0]]
X_set = np.array(A)
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() -1, stop=X_set[:, 0].max() +1, step=0.5),
                     np.arange(start=X_set[:, 1].min() -1, stop=X_set[:, 1].max()+ 1, step=0.5))
print('X1:', X1)
print('X2:', X2)
```
输出结果：
```python
X1: [[-1.  -0.5  0.   0.5  1.   1.5]
     [-1.  -0.5  0.   0.5  1.   1.5]
     [-1.  -0.5  0.   0.5  1.   1.5]
     [-1.  -0.5  0.   0.5  1.   1.5]
     [-1.  -0.5  0.   0.5  1.   1.5]
     [-1.  -0.5  0.   0.5  1.   1.5]]
     
X2: [[-1.  -1.  -1.  -1.  -1.  -1. ]
     [-0.5 -0.5 -0.5 -0.5 -0.5 -0.5]
     [ 0.   0.   0.   0.   0.   0. ]
     [ 0.5  0.5  0.5  0.5  0.5  0.5]
     [ 1.   1.   1.   1.   1.   1. ]
     [ 1.5  1.5  1.5  1.5  1.5  1.5]]
```
分析：  
第84和85行语句，总的结构为`X1, X2 = np.meshgrid(X,Y)`,其中  
`X = np.arange(start=X_set[:, 0].min() -1, stop=X_set[:, 0].max() +1, step=0.5)`  
`Y = np.arange(start=X_set[:, 1].min() -1, stop=X_set[:, 1].max()+ 1, step=0.01)`结构清楚了，然后我们结合测试程序分析具体的意思。  
`np.arange(start=X_set[:, 0].min() -1, stop=X_set[:, 0].max() +1, step=0.5)`:`X_ set[:, 0].min()-1`为`X_set`第0列中的最小数减1，结果为-1，
步长为0.5,所以具体就是`np.arange(-1,2,0.5)`.输出为：`[-1.  -0.5  0.   0.5  1.   1.5]`  
`X1, X2 = np.meshgrid(X,Y)`：[参考文章](https://blog.csdn.net/baoqian1993/article/details/52116164)  
语句2：  
```python
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
```
测试代码(接上面的测试代码）：
```python
X3= np.array([X1.ravel(),X2.ravel()]).T
X4 =np.array([X1.ravel(),X2.ravel()])
```
分析：
上面测试代码输出就可以看到区别。`ravel`实现降维的作用，`shape`确定矩阵是几行几列，`reshape`实现将矩阵转化为几行几列。`.T` 实现转置。  
`classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape)`将X1,X2展开后进行转置，使得矩阵每一行元素表示一个点，然后预测每个点的类别，将结果大小调整到和坐标数据相同的矩阵大小。.csv文件中就是预测`X_train`对应的是否购买的情况。输出只有两种，0和1。  
`contourf(x,y,f(x,y),其他）`:实现画等高线，本例中，横坐标为x`Age`,纵坐标y为`Estimated Salary`.f(x,y)的作用在本例中就是预测出`purchased`,也就是0或者1.相同的数值可以理解为等高处，因为步长很小，用不同的颜色对不同的等高线上色，看到的就是面的效果。因为后面还需要评价的分类好坏。所以需要设置透明度`alph`,取值为0~1.  
语句3：
```python
for i,j in enumerate(np. unique(Y_set)):
    plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)
```
`enumerate()`[参考文章](http://www.runoob.com/python/python-func-enumerate.html)
`unique()`[参考文章](https://blog.csdn.net/u012193416/article/details/79672729)  
一开始，i=j=0,语句变为` plt.scatter(X_set[Y_set==0,0],X_set[Y_set==0,1],c = ListedColormap(('red', 'green'))(0), label=0)`画散点图，其中，横坐标为`X_set[Y_set==0,0]`(表示对应于`Y_set = 0`时`X_set`的第0列*age*。纵坐标为`X_set[Y_set==0,1]`(表示对应于`Y_set = 0`时`X_set`的第1列*Estimated Salary*。选取第0个颜色`red`,生成类别标签（label），在图中显示为右上角，红点后边跟个 0。当i=j=1时，分析略。

