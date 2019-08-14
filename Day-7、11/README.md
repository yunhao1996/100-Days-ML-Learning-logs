

           
# 100_ML_Day7-11
```python
import numpy as np
import pandas as pd

# 读入，确定自变量和因变量
# 自变量：Age, Estimated
# 因变量：Purchased
dataset = pd.read_csv(r'D:\ML_100\Day7_11\Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
Y = dataset.iloc[:, 4].values

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
print(Y_test)

# 特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 使用KNN对训练集进行训练
# n_neighbors:选取最近的点的个数（k）
# p=2:默认欧几里德指标
# metric='minkowski：默认
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)

# 用上面的模型预测测试集
Y_pred = classifier.predict(X_test)

# 混淆矩阵，评估模型(对角线为正确)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
```

