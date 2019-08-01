# 给的程序版本可能存在版本的问题，复制下来，最后一步callback功能无法运行，通过对最近的网页搜索，我发现调用tensorboard时，直接从Keras中调用即可，
# 所以我直接把框架模型换成了Keras,

## 关于tensorboard 可用于程序运行过程中的损失，准确率等一些属性的可视化表达 
```python

import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import pickle
import time

# 创建TensorFlow backend
NAME = "Cats-vs-dogs-CNN"

pickle_in = open("D://ML_100//40//PetImages//X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("D://ML_100//40//PetImages//y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))  # 将数据映射到0-1

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit(X, y,
          batch_size=32,
          epochs=3,
          validation_split=0.3,
          callbacks=[tensorboard])

```

## 目前还不能解决的是如何打开初始化结果
