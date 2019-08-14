# 100_ML_Day9/10/12/13/14
```python
# 导入库
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 导入数据
dataset = pd.read_csv(r'D:\ML_100\Day7_11\Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
Y = dataset.iloc[:, 4].values

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# 特征量化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 对训练集应用SVM
classifier = SVC(kernel='linear', random_state=0)
print(classifier)
classifier = classifier.fit(X_train, Y_train)

# 预测测试集结果
Y_pred = classifier.predict(X_test)

# 评价（创建混淆矩阵）
cm = confusion_matrix(Y_test, Y_pred)
# print(cm)

# 可视化
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
plt. title('SVM(Training set)')  # 显示标题
plt. xlabel(' Age')  # 横坐标属性
plt. ylabel('Estimated Salary')
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
plt. title('SVM(Test set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt.show()
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-9、10、12、13、14/pictures/Figure_1.png">
</p> 
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-9、10、12、13、14/pictures/Figure_2.png">
</p> 

