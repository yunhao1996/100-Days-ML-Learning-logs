# 学习记录
## 朴素贝叶斯分类器naive bayes[参考文章](https://blog.csdn.net/fisherming/article/details/79509025) 
朴素贝叶斯分类器是与线性模型（LR,SVC）非常类似的一种分类器，但它的训练速度往往更快，这种高效率的代价是泛化能力较差.  
主要有三种：  
`GaussianNB`应用于任意连续数据  
`BernoulliNB`假定输入数据为二分类数据，主要用于文本数据分类  
`MultinomialNB`假定输入数据为计数数据（即每个特征代表的某个对象的整数计数，比如一个单词在句子中出现的次数），主要用于文本数据分类  
在参考文章的实例解析中，把例子创建成了.csv文件，由于都是二值分布，直接输入的0和1代替标签，没有经过`LabelEncoder`进行编号。  
```python
import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB

trainset = pd.read_csv(r'D:\ML_100\day15\ceshi.csv')
X_train = trainset.iloc[:, 0:4].values
Y_train = trainset.iloc[:, 4].values
# print(trainset)
# print(X_train)

NB = BernoulliNB()  # 伯努利型
NB.fit(X_train, Y_train)

X_test = [[0, 0, 0, 0]]
Y_test = [[0]]
predict = NB.predict(X_test)
print(predict)
# 二分类数据模型正确率的一种方法
# np.mean():取均值
print(np.mean(predict == Y_test))
```
输出：  
```python
[0]  # 不嫁
1.0
```
## Black Box Machine Learning  
1.Feature Extraction特征提取：判断条件输出二值，或者进行编码  
2.Evaluating a Prediction Function评价预测函数：第一种就是`0/1损失`,预测错了损失就是1，预测对了损失为0；第二种是`回归平方损失`其值为预测值与真实值差的平方。预测就引入了测试集，需要做好训练集和测试集的平衡。  
3.交叉验证：[参考文章](https://cloud.tencent.com/developer/news/238899)   
4.sample bias样本偏差:常见的样本偏差有两种，一种是所抽取的样本不是随机的,不具普遍性；另一种是抽取的样本数量不够多  
5.非平稳性：非平稳性通常有两种形式：第一种是样本值改变，针对不同的主体；第二种是样本点漂移，我的理解是样本点代表的是过去，不具有现在的属性，例如，上周我学习，这周我不学习，针对主体相同但是时间变了。  
6.模型复杂度与过拟合：几乎每种学习算法至少包含一个`Hyperparameters`超参数或者`Tuning parameter`调整参数。超参数是在开始学习过程之前设置值的参数，而不是通过训练得到的参数数据。通常情况下，需要对超参数进行优化，给学习机选择一组最优超参数，以提高学习的性能和效果。超参数控制了模型复杂度和模型的效果等。

