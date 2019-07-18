# 深入研究 -- numpy

1.查看该库的版本号
```pyhon
IN：
import numpy as np
print(np.__version__)

OUT：
1.16.4
```
2.列表及其类型
```python
IN：
L = list(range(10))
print(L)
print(type(L[0]))
print('\n')

L2 = [str(c) for c in L]  # 将int转化为字符
print(L2)
print(type(L2[0]))
print('\n')

L3 = [True, "2", 3.0, 4]
L4 = [type(item) for item in L3]
print(L4)

OUT：
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
<class 'int'>

['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
<class 'str'>

[<class 'bool'>, <class 'str'>, <class 'float'>, <class 'int'>]
```
3. 数组
```python
IN：
import array

L = list(range(10))
A = array.array('i', L)
print(A)

OUT：
array('i', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  #Here 'i' is a type code indicating the contents are integers.
```
4.numpy 中的数组
```python
IN：
import numpy as np

a = np.array([1, 4, 2, 5, 3])
print(a)
print(type(a[0]), '\n')  

b = np.array([1, 2, 3, 4], dtype='float32')  # 将int32转化为float32
print(b)
print(type(b[0]))

c = np.array([range(i, i+3) for i in [2, 4, 6]])
print(c)

OUT：
[1 4 2 5 3]
<class 'numpy.int32'> 

[1. 2. 3. 4.]
<class 'numpy.float32'>

[[2 3 4]
 [4 5 6]
 [6 7 8]]

其他形式的数组：
import numpy as np

a = np.zeros(10, dtype=int)  # 输出长度为10，全零的整形数组
b = np.ones((3, 5), dtype=float)  # 输出3×5的全1的float数组
c = np.full((3, 5), 3.14)  # 输出全3.14的3×5的数组
d = np.arange(0, 20, 2)  # 前闭后开，返回array,步长为2的等差数列
e = np.linspace(0, 1, 5)  # 前闭后闭，线性产生5个数
f = np.random.random((3, 3))  # 生成3行，3列的浮点数，浮点数都是从0-1中随机
g = np.random.normal(0, 1, (3, 3))  # 生成高斯分布的概率密度随机数,(均值， 标准差， shape)
h = np.random.randint(0, 10, (3, 3))  # 随机生成[0, 10）之间的整数
i = np.eye(3)   # 返回一个二维数组，3×3，对角线的地方为1，其他为0.默认0为主对角线
j = np.empty(3)  # 创建以为空数组
```





