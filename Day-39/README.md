# 手写数字识别

## 基于tensorflow

```python
import tensorflow.keras as keras
import tensorflow as tf  # 深度学习库，Tensor 就是多维数组  张量
import matplotlib.pyplot as plt
import numpy as np

f = np.load('D:/ML_100/39 30 41 42/mnist.npz')
# 自变量中每一行是一张图片的全部像素，每一列代表图片的一排像素
# 因变量中输出的是对应的数字
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']

plt.imshow(x_train[0], cmap='binary')  # 显示黑白图像
# plt.show()

# 对数据进行归一化处理，把数据值缩放到0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

plt.imshow(x_train[0],cmap='binary')
plt.show()

# 构建与训练模型
# 构建神经网络
model = tf.keras.models.Sequential()  # 基础的前馈神经网络模型
model.add(tf.keras.layers.Flatten())  # 把图片展成 1×784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 简单的全连接图元，128个单元，激活函数为relu
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  #
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # 输出层，10个单元，使用Softmax获得概率分布

# print(model)

model.compile(optimizer='adam',  # 默认的较好的优化器
              loss='sparse_categorical_crossentropy',  # 评估“错误的损失函数”，模型应该尽量降低损失
              metrics=['accuracy'])  # 评价指标
model.fit(x_train, y_train, epochs=3)  # 训练模型
val_loss, val_acc = model.evaluate(x_test, y_test)  # 评估模型对样本数据的输出结果
# print(val_loss)  # 模型的损失值
# print(val_acc)  # 模型的准确度
predictions = model.predict(x_test)  # 预测验证集

print(np.argmax(predictions[0]))

plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()
```
### 归一化输出测试集一张图片:

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-39/1.png">
</p>

### 预测验证集第一张图上的数字是否是7

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-39/2.png">
</p>

### 测试集迭代结果：

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-39/3.png">
</p>

## 基于pytorch和CNN实现

```python
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)    # 设定随机数种子

# Hyper Parameters
EPOCH = 2           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 False

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    # transforms 会把像素值从（0， 255）压缩到（0-1）.灰度图片是一个通道
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成 tensor
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)

# print(train_data.train_data.size())  # 图片数据信息
# print(train_data.train_labels.size())  # 数据标签信息
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')  # 灰白显示
# print('%i'% train_data.train_labels[0])
# plt.show()

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)，shuffle:是否随机打乱顺序,一共1200批
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
# 在第1维增加维度1，维度从0开始数，这里要手动压缩
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)  # 10000
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 =nn.Sequential(  # input shape (1, 28, 28)
        nn.Conv2d(  # 二维卷积
            in_channels=1,  # 输入通道数
            out_channels=16,  # 输出通道数
            kernel_size=5,  # 每个卷积核为5*5
            stride=1,  # 卷积步长
            padding=2,  # 填充值为2，输出图像长宽不变
           ),  # 输出大小（16,28,28)
        nn.ReLU(),  # 卷积完使用激活函数

         # 最大池化层，在2*2的空间里向下采样，输出（16，14，14）
        nn.MaxPool2d(kernel_size=2),  # 这里stride默认值为kernel_size
        )
        self.conv2 = nn.Sequential(  # 在来一层卷积层，输入（16，14，14）
            nn.Conv2d(16, 32, 5, 1, 2),  # 数值按照上面的属性排列
            nn.ReLU(),  # 32*14*14
            nn.MaxPool2d(2),  # 筛选出重要的特征
        )  # 32*7*7
        self.out = nn.Linear(32*7*7, 10)  # 全连接层，输入32*7*7，输出10个数

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 展平多维的卷积图成（batch_size,32*7*7）
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output  # 返回的是每个数值也就是索引值的可能性


cnn = CNN()  # 定义实体
# print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    # enumerate：返回索引序列，step是索引值
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            correct = 0
            test_output = cnn(test_x)  # （[2000，10]）
            # 按照行进行取最大值,[1],返回最大值的每个索引值，也就是说，可能性比较大
            pred_y = torch.max(test_output, 1)[1].numpy()
            correct += sum(pred_y == test_y.numpy())
            accuracy = correct/ 2000
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

# test_output = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')

```

