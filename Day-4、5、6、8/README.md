# 100_ML_Day4-5-6-8
```python
import matplotlib.pyplot as plt  # 画图
from matplotlib.colors import ListedColormap  # 上色
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.preprocessing import StandardScaler  # 特征缩放
from sklearn.model_selection import train_test_split  # 划分数据
import numpy as np
import pandas as pd

# Step1.数据预处理
dataset = pd.read_csv(r'D:\ML_100\Day_456\Social_Network_Ads.csv')
# X = dataset.iloc[:,2:3].values
X = dataset.iloc[:, [2, 3]].values  # 分析年龄，薪资与是否购买的关系
Y = dataset.iloc[:, 4].values
# print(X)
# print(Y)

# 将数据集划分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
# print(X_train)

# 特征缩放
# 分析见学习记录
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step2.逻辑回归模型
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
# print(classifier.fit(X_train, Y_train))

# Step3.预测
# 预测测试集结果
Y_pred = classifier.predict(X_test)

# Step4.评估预测
# 生成混淆矩阵,confusion:混淆；metrix:矩阵
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# Step5.可视化
# 训练集效果图
X_set, Y_set = X_train, Y_train
X1, X2 = np. meshgrid(np. arange(start=X_set[:, 0].min() -
                                 1, stop=X_set[:, 0].max() +
                                 1, step=0.01), np. arange(start=X_set[:, 1].min() -
                                                           1, stop=X_set[:, 1].max() +
                                                           1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())  # 横坐标的范围
plt.ylim(X2.min(), X2.max())
# plt.show() #显示红绿色背景
# 获取Y_set中的所有类别“标记“ j 及其 索引 i
for i, j in enumerate(np. unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt. title(' LOGISTIC(Training set)')  # 显示标题
plt. xlabel(' Age')  # 横坐标属性
plt. ylabel(' Estimated Salary')
plt. legend()  # 用于显示类别标签
plt. show()  # 显示图片

# 画验证集效果图
X_set, Y_set = X_test, Y_test
X1, X2 = np. meshgrid(np. arange(start=X_set[:, 0].min() -
                                 1, stop=X_set[:, 0].max() +
                                 1, step=0.01), np. arange(start=X_set[:, 1].min() -
                                                           1, stop=X_set[:, 1].max() +
                                                           1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np. unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
# print(X_set[:,0])
# print((X_set[Y_set==1,0]))
# print((X_set[Y_set==0,0]))
plt. title(' LOGISTIC(Test set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt. legend()
plt. show()

# 另一种方式画逻辑回归二分类图

# 先显示训练集的散点图
plt.figure(1)
# 为0时的散点图
plt.scatter(X_train[Y_train == 0][:, 0],
            X_train[Y_train == 0][:, 1], marker='o', color='b')
# 为1时的散点图
plt.scatter(X_train[Y_train == 1][:, 0],
            X_train[Y_train == 1][:, 1], marker='x', color='r')
# 按照顺序加label
plt.legend(['0', '1'])
# 横轴范围
x1_min, xl_max = X_train[:, 0].min(), X_train[:, 0].max()
# 纵轴范围
x2_min, x2_max = X_train[:, 1].min(), X_train[:, 1].max()
# 思路：表示出平面的所有点（合适的步长），用上面的模型预测这些点的分类，以此找到决策边界
# linspace（start，stop，num):将区间[start,stop]均匀分成num份
xx1, xx2 = np.meshgrid(
    np.linspace(
        x1_min, xl_max), np.linspace(
            x2_min, x2_max))
grid = np.array([xx1.ravel(), xx2.ravel()]).T
probs = classifier.predict(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], colors='black')
plt. title(' LOGISTIC(Training set)')  # 显示标题
plt. xlabel(' Age')  # 横坐标属性
plt. ylabel(' Estimated Salary')
plt. show()  # 显示图片
plt.show()

# 测试集的散点图
plt.figure(2)
plt.scatter(X_test[Y_test == 0][:, 0], X_test[Y_test == 0]
            [:, 1], marker='o', color='b')
plt.scatter(X_test[Y_test == 1][:, 0], X_test[Y_test == 1]
            [:, 1], marker='x', color='r')
plt.legend(['0', '1'])
x3_min, x3_max = X_test[:, 0].min(), X_test[:, 0].max()
x4_min, x4_max = X_test[:, 1].min(), X_test[:, 1].max()
xx3, xx4 = np.meshgrid(
    np.linspace(
        x3_min, x3_max), np.linspace(
            x4_min, x4_max))
grid1 = np.array([xx3.ravel(), xx4.ravel()]).T
probs1 = classifier.predict(grid).reshape(xx3.shape)
plt.contour(xx3, xx4, probs1, [0.5], colors='black')
plt. title(' LOGISTIC(Test set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt.show()

```
显示红绿色背景
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/Figure_4.png">
</p> 
训练集效果图
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/Figure_2.png">
</p> 
测试集效果图
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/Figure_3.png">
</p> 

第二种显示：
<p align="center">
    <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/1Figure_1.png">
</p>    
<p align="center">
    <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-4、5、6、8/pictures/2Figure_2.png">
</p>  
 

