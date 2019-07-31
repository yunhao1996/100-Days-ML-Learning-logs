# 数据集操作---将图片转化为数据，创建数据集，灰度变换，尺寸变换

```python

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm  # 作用在终端出现一个进度条，使得代码进度可视化，用于长循环
import random
import pickle

DATADIR = "D://ML_100//40//PetImages"  # 数据集的路径

CATEGORIES = ["Dog", "Cat"]  # 列表获取分类,也就是下一级的文件夹名称

# for category in CATEGORIES:  #
#     path = os.path.join(DATADIR, category)  # 拼接目录路径
#     # print(path)
#     for img in os.listdir(path):  # 迭代遍历每个图片
#         # cv2.imread读的是图片的完整地址，读进来直接是RGB格式在0-255，cv2.IMREAD_GRAYSCALE以灰度模式读取图片
#         # 以灰度形式读取图像，输出结果会自动变为1通道,对于三通道的，cmap可能会失效
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # 转化成array
#         # plt.imshow(img_array, cmap='gray')  # 转换成图像展示
#         # plt.show()  # 显示
#
#     #     break  # 我们作为演示只展示一张，所以直接break了
#         #     # break  # 同上
#
#
#         IMG_SIZE = 100
#         new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 调整像素值大小
#         plt.imshow(new_array, cmap='gray')
#         plt.show()

IMG_SIZE = 100
training_data = []


def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)  # 拼接目录路径
        class_num = CATEGORIES.index(category)  # 得到分类，其中 0=dog 1=cat
        print(category, class_num)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 大小转换
                training_data.append([new_array, class_num])  # 加入训练数据中
                # Exception是万能的异常捕捉方法，可以捕捉到任何错误
            except Exception as e:  # 为了保证输出是整洁的
                pass
            # except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            # except Exception as e:
            #    print("general exception", e, os.path.join(path,img))


create_training_data()  # 调用函数
print(len(training_data))

# 创建数据集
random.shuffle(training_data)  # 打乱训练集
X = []
y = []

for features, label in training_data:
    X.append(features)  # X
    y.append(label)  # 标签y

# print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
# print(np.array(X).shape)  # （24946，100，100），也就说，reshape也可以实现扩维
# 不太清楚，这里为什么要进行扩维
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # （24946，100，100，1）

# “wb”: wrint binary： 将图片转化为数据后，写入文件
pickle_out = open("D://ML_100//40//PetImages//X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("D://ML_100//40//PetImages//y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# 读取测试集数据
pickle_in = open("D://ML_100//40//PetImages//X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("D://ML_100//40//PetImages//y.pickle", "rb")
y = pickle.load(pickle_in)
```
