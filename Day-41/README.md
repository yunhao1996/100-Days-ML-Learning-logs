# 应用Keras构建神经网络，识别猫和狗

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

pickle_in = open("D://ML_100//40//PetImages//X.pickle", "rb")
X = pickle.load(pickle_in)  # 读取数据集

pickle_in = open("D://ML_100//40//PetImages//y.pickle", "rb")
y = pickle.load(pickle_in)  # 读取标签

X = X/255.0  # 将数据压缩到0-1，类似于归一化处理
print(X.shape[1:-1])  # （100，100，1）
model = Sequential()  # 多个网络层的线性堆栈，可以直接在列表中写网络层

# input_shape:指定第一层的输入维度，2维出错，显示层属性时为4维，后面的层会自动推导
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))  # 卷积层,输出256通道，卷积核大小3×3
model.add(Activation('relu'))  # 激活函数rule
model.add(MaxPooling2D(pool_size=(2, 2)))  # 最大池化，这里的stride=kernel_size,也就是2

model.add(Conv2D(256, (3, 3)))  # 卷积层
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 将三维特征映射为1维

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())  # 查看各层的属性

# 定义损失函数,优化器
model.compile(loss='binary_crossentropy',
              optimizer='adam',  # 优化方式
              metrics=['accuracy'])  # 衡量模型的指标

# validation_split指定比例划分验证集，不参与训练，每次epock后，测试模型的指标
model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)

```
## 网络层的属性

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-41/1.png">
</p>

## 迭代运行结果，批大小为32，epock为3

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-41/2.png">
</p>
