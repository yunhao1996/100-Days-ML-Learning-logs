# 深入研究-PANDAS
1.向量化字符串操作
```python
1.1 pandas字符串操作
import pandas as pd

data = ['peter', 'Paul', 'MARY', 'gUIDO']
data = [s.capitalize() for s in data]  # 不能处理缺失值
print(data)

names = pd.Series(data)
print(names)
输出：
['Peter', 'Paul', 'Mary', 'Guido']

0    Peter
1     Paul
2     Mary
3    Guido
dtype: object

1.2 对表格应用字符串方法
import pandas as pd

monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
print(monte.str.lower(), '\n')  # 返回一系列字符串
print(monte.str.len(), '\n')  # 返回数字
print(monte.str.startswith('T'), '\n')  # 返回布尔值
print(monte.str.split(), '\n')  # 返回列表或每个元素的其他复合值
输出：
0    graham chapman
1       john cleese
2     terry gilliam
3         eric idle
4       terry jones
5     michael palin
dtype: object 

0    14
1    11
2    13
3     9
4    11
5    13
dtype: int64 

0    False
1    False
2     True
3    False
4     True
5    False
dtype: bool 

0    [Graham, Chapman]
1       [John, Cleese]
2     [Terry, Gilliam]
3         [Eric, Idle]
4       [Terry, Jones]
5     [Michael, Palin]
dtype: object 

1.3 使用正则表达式的方法
import pandas as pd

monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])

print(monte.str.extract('([A-Za-z]+)', expand=False), '\n')  # 用字符组来提取每个元素的名字
# 查找所有以辅音开头和结尾的名称，使用字符串开始(^)和字符串结束($)正则表达式字符
print(monte.str.findall(r'^[^AEIOU].*[^aeiou]$'), '\n')  
输出：
0     Graham
1       John
2      Terry
3       Eric
4      Terry
5    Michael
dtype: object 

0    [Graham Chapman]
1                  []
2     [Terry Gilliam]
3                  []
4       [Terry Jones]
5     [Michael Palin]
dtype: object 
```
2.处理时间序列
```python
2.1 datetime和dateutil用法
from datetime import datetime  # 处理日期和时间的基本对象
from dateutil import parser  # 解析来自各种字符串格式的日期:

a = datetime(year=2015, month=7, day=4)  # 手动构建一个日期
print(a)

date = parser.parse("4th of July, 2015")
print(date)
print(date.strftime('%A'))  # 输出该日期是那一天
输出:
2015-07-04 00:00:00
2015-07-04 00:00:00
Saturday

2.2 处理时间类型--datetime64
import numpy as np

date = np.array('2015-07-04', dtype=np.datetime64)  # 格式化日期
print(date)
print(date + np.arange(12))  # 矢量运算
输出：2015-07-04
['2015-07-04' '2015-07-05' '2015-07-06' '2015-07-07' '2015-07-08'
 '2015-07-09' '2015-07-10' '2015-07-11' '2015-07-12' '2015-07-13'
 '2015-07-14' '2015-07-15']
 
 2.3 按时间索引
 import pandas as pd

index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)  # 构建具有时间索引数据的Series对象
print(data)
print('\n')
print(data['2014-07-04':'2015-07-04'])  # 开始索引
print('\n')
print(data['2015'])
输出：
2014-07-04    0
2014-08-04    1
2015-07-04    2
2015-08-04    3
dtype: int64

2014-07-04    0
2014-08-04    1
2015-07-04    2
dtype: int64

2015-07-04    2
2015-08-04    3
dtype: int64

2.4 常规序列:pd.date_range ()
import pandas as pd

a = pd.date_range('2015-07-03', '2015-07-10')  # 默认频率为1，输出日期
b = pd.date_range('2015-07-03', periods=8)  # 使用起始点和若干周期:
c = pd.date_range('2015-07-03', periods=8, freq='H')  #间距可以通过改变freq参数来修改，freq参数默认为d，这里使用小时
d = pd.timedelta_range(0, periods=10, freq='H')  # 一系列的时间增加一小时

print(a)
print(b)
print(c)
print(d)
输出：
DatetimeIndex(['2015-07-03', '2015-07-04', '2015-07-05', '2015-07-06',
               '2015-07-07', '2015-07-08', '2015-07-09', '2015-07-10'],
              dtype='datetime64[ns]', freq='D')
DatetimeIndex(['2015-07-03', '2015-07-04', '2015-07-05', '2015-07-06',
               '2015-07-07', '2015-07-08', '2015-07-09', '2015-07-10'],
              dtype='datetime64[ns]', freq='D')
DatetimeIndex(['2015-07-03 00:00:00', '2015-07-03 01:00:00',
               '2015-07-03 02:00:00', '2015-07-03 03:00:00',
               '2015-07-03 04:00:00', '2015-07-03 05:00:00',
               '2015-07-03 06:00:00', '2015-07-03 07:00:00'],
              dtype='datetime64[ns]', freq='H')
TimedeltaIndex(['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00',
                '05:00:00', '06:00:00', '07:00:00', '08:00:00', '09:00:00'],
               dtype='timedelta64[ns]', freq='H')
           
2.5 重新采样、移动和开窗
from pandas_datareader import data

goog = data.DataReader('GOOG', start='2004', end='2016',
                       data_source='google')

print(goog.head())
由于版本和包的问题，无法运行
```
3.高性能Pandas：eval()与query()
```python
3.1 复合表达式
import numpy as np
rng = np.random.RandomState(42)
x = rng.rand(1000000)
y = rng.rand(1000000)

# 这比循环运行要快的多
%timeit x + y

# 但是，当计算复合表达式时，这种抽象可能会变得更低效
%timeit np.fromiter((xi + yi for xi, yi in zip(x, y)), dtype=x.dtype, count=len(x))

3.2 eval()用于有效的操作
import pandas as pd
import numpy as np

nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols))
                      for i in range(4))
# 这个表达式的eval()版本大约快了50%(并且使用了更少的内存)，同时得到了相同的结果
print(pd.eval('df1 + df2 + df3 + df4'))   # 求和运算

3.3 eval()用于列操作
import pandas as pd
import numpy as np

rng = np.random.RandomState(42)
# 对任何列进行赋值
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
print(df.head())

df.eval('D = (A + B) / C', inplace=True)  # 列之间进行运算
print(df.head())

df.eval('D = (A - B) / C', inplace=True)  # 列之间进行运算
print(df.head())
输出：
          A         B         C
0  0.374540  0.950714  0.731994
1  0.598658  0.156019  0.155995
2  0.058084  0.866176  0.601115
3  0.708073  0.020584  0.969910
4  0.832443  0.212339  0.181825
          A         B         C         D
0  0.374540  0.950714  0.731994  1.810472
1  0.598658  0.156019  0.155995  4.837844
2  0.058084  0.866176  0.601115  1.537576
3  0.708073  0.020584  0.969910  0.751263
4  0.832443  0.212339  0.181825  5.746085
          A         B         C         D
0  0.374540  0.950714  0.731994 -0.787130
1  0.598658  0.156019  0.155995  2.837535
2  0.058084  0.866176  0.601115 -1.344323
3  0.708073  0.020584  0.969910  0.708816
4  0.832443  0.212339  0.181825  3.410442

3.4 DataFrame.query()方法
import pandas as pd
import numpy as np

rng = np.random.RandomState(42)
# 对任何列进行赋值
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])

result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
a = np.allclose(result1, result2)  # 在默认误差下，每一个元素是否接近。返回布尔值
print(a)

# 这种类型的过滤操作，您可以使用query()方法
result2 = df.query('A < 0.5 and B < 0.5')
b = np.allclose(result1, result2)
print(b)
输出：
True
True

3.5 性能:何时使用这些功能  
在考虑是否使用这些函数时，需要考虑两个因素:计算时间和内存使用。内存使用是最可预测的方面。如前所述，每个涉及NumPy数组或panda数据流的复合表达式都会隐式地创建临时数组:如果临时数据流的大小与可用的系统内存(通常是几gb)相比非常大，那么最好使用eval()或query()表达式。
```
