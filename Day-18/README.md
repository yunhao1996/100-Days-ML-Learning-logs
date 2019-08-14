# 100_ML_Day18
## 简单LR-NN
```python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import h5py  # 提供读取HDF5二进制数据格式文件的接口，本次的训练及测试图片集是以HDF5储存的。

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r") # 读取h5格式的文件
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 提取训练集，（209*64*64*3）
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 提取训练集的标签,（209*1）

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 提取测试集,（50*64*64*3）
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 提取测试集标签,（50*1）

    classes = np.array(test_dataset["list_classes"][:])  # 类别，1为猫，0为非猫

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 对训练集和测试集标签进行reshape设为（1*209）
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# 定义激活函数，sigmoid主要用在二分类的输出层
def sigmoid(z):

    s = 1 / (1 + np.exp(-z))

    return s

# 初始化参数，向量化思想
def initialize_with_zeros(dim):
    '''
    :param dim: w 的矢量大小
    :param b: 初始化对应偏差
    '''
    w = np.zeros((dim, 1))
    b = 0

    # 利用断言来确保使用数据的维度
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def propagate(w, b, X, Y):
    '''
    forword and backword function

    :param w:权重，维度（num_px*numpy*3,1）
    :param b:偏差,维度(1,m_train)
    :param X:训练集，维度（num_px*num_py*3,m_train）
    :param Y:真实标签，维度（1，m_train）
    :return:
    cost - 逻辑回归的成本函数，为lost function 的均值
    dw - 相对于w的损失梯度，维度（num_px*numpy*3,1）
    db - 相对于b的损失梯度，维度（1，m_train）
    '''
    
    m = X.shape[1]

    # 前向传播
    # Z=np.dot(w.T,X)+b
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # 反向传播
    # dZ=A-Y
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)  # 降维，维度为1的去掉，保存为一维形式，列表

    # 创建字典存储dw和db,用于更新
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    '''
    :param num_iterations:  迭代次数，超参数
    :param learning_rate:   学习率，超参数
    :param print_cost:  每一百步打印一次损失值
    :return:
    params -  包含权重w和偏差b的字典，用于存储更新
    grads - 包含权重和偏差相对于成本函数的梯度的字典，用于存储更新的梯度值
    costs - 梯度优化计算的成本值列表，用于绘制图像
    '''

    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw  #更新
        b = b - learning_rate * db


        if i % 100 == 0:
            costs.append(cost)  # 在末尾添加新对象
        
        # 每一百次输出一次
        if print_cost and i % 100 == 0:   # print_cost=False,这一步不运行
            print ("Cost after iteration %i: %f" %(i, cost))

    # 创建字典存储
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    '''
       由上面得到的w,b，预测训练集X的标签
       :return:
       Y_prediction - 训练集的预测标签
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))  #初始化
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)  # 预测值，值为0-1

    for i in range(A.shape[1]):   # 设置阈值，判决
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert(Y_prediction.shape == (1, m))  # 得到的新的矩阵确保维度
    
    return Y_prediction

# 通过调用之前实现的函数来构建逻辑回归模型,整合
def model(X_train, Y_train, X_test, Y_test, num_iterations , learning_rate , print_cost ):
    """
    参数：
        X_train  - numpy的数组,维度为（num_px * num_pxy* 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_py * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本

    返回：
        d  - 包含有关模型信息的字典。
    """
    # 用0初始化参数
    w, b = initialize_with_zeros(X_train.shape[0])

    # 优化得到合适的w和b
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # 从字典中检索参数w和b
    w = params["w"]
    b = params["b"]
    
    # 预测训练集和测试集
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 输出训练集/测试集中正确率
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]  # 209
m_test = test_set_x_orig.shape[0]    # 50
num_px = train_set_x_orig.shape[1]   # 64
num_py = train_set_x_orig.shape[2]   # 64

# 降维，每一列代表一张图片
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 数据标准化，每个像素点由8位二值组成，映射到0-1.除以255
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

d = model(train_set_x, train_set_y, test_set_x, test_set_y, 2000, 0.005, False)

# 可视化
costs=d['costs']
plt.plot(costs)
plt.title("Learning rate = 0.005")
plt.xlabel("iterations (per hundreds)")
plt.ylabel("Cost")
plt.show()
```
结果：
```python
train accuracy: 74.16267942583733 %
test accuracy: 56.0 %
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-18/Figure_1.png">
</p> 

