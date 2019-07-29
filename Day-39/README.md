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

<p align="center">
  <img src="">
</p>


