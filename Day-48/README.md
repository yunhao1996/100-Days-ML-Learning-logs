# 深入研究|PANDAS|数据处理
1. Pandas数据处理 
1.1 Pandas对象
```python
输入：
import pandas as pd

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)
print('values:', data.values)
print('index:', data.index)
print('data[1]:', data[1])
print('data[1:3]:',data[1:3])
输出：
data[1]: 0.5
data[1:3]: 1    0.50
2    0.75
dtype: float64
```
1.2 Series可以和一维数组Numpy互换
```python
输入：
import pandas as pd

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
print('data:', data)
print('data[b]:', data['b'])
输出：
data: a    0.25
b    0.50
c    0.75
d    1.00
dtype: float64
data[b]: 0.5
```
1.3 Series作为字典使用
```python
import pandas as pd

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
print('population:\n', population)
print("population['California']:\n", population['California'])
print("population['California':'Illinois']:\n", population['California':'Illinois'])
输出：
population:
 California    38332521
Texas         26448193
New York      19651127
Florida       19552860
Illinois      12882135
dtype: int64
population['California']:
 38332521
population['California':'Illinois']:
 California    38332521
Texas         26448193
New York      19651127
Florida       19552860
Illinois      12882135
dtype: int64
```
1.4 DataFrame作为Numpy数组使用
```python
输入：
import pandas as pd

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)

states = pd.DataFrame({'population': population,
                       'area': area})
print('states:\n', states)

# 索引功能
print('states.index:\n', states.index)  # 行信息
print('states.columns:\n', states.columns)  # 列信息
输出：
states:
             population    area
California    38332521  423967
Texas         26448193  695662
New York      19651127  141297
Florida       19552860  170312
Illinois      12882135  149995
states.index:
 Index(['California', 'Texas', 'New York', 'Florida', 'Illinois'], dtype='object')
states.columns:
 Index(['population', 'area'], dtype='object')
```
1.5 DataFrame作为字典使用
```python
接上输入：
print("states['area']:\n", states['area'])
输出：states['area']:
 California    423967
Texas         695662
New York      141297
Florida       170312
Illinois      149995
Name: area, dtype: int64
```
1.6 几种构建DataFrame的方式
```python
第一种由单个序列组成，接上输入：
a = pd.DataFrame(population, columns=['population'])
print(a)
输出：
            population
California    38332521
Texas         26448193
New York      19651127
Florida       19552860
Illinois      12882135
第二种由列表创建一些数据，输入：
import pandas as pd

data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
a = pd.DataFrame(data)
print(a)
输出：
   a  b
0  0  0
1  1  2
2  2  4
这里的话，即使使用的列表缺少一些值，输出也会用“NaN”补充  

第三种由一系列对象的字典创建，即label 1.4  

第四种由二维数组加索引等创建，输入：
import pandas as pd
import numpy as np

a = pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])
print(a)  
输出:
        foo       bar
a  0.814909  0.160992
b  0.174402  0.305773
c  0.395690  0.016511
```
2. 数据取值与选择
2.1  隐式索引
```python
输入：
import pandas as pd

data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])

print(data.loc[1], '\n')
print(data.loc[1:3], '\n')
print(data.iloc[1], '\n')
print(data.iloc[1:3], '\n')
输出：
a 

1    a
3    b
dtype: object 

b 

3    b
5    c
dtype: object 
```
3.Pandas数值运算方法  
对于创建的DataFrame可以直接进行运算操作，索引行标不参与  
3.1 多个DataFrame进行操作
```python
输入:
import pandas as pd
import numpy as np

rng = np.random.RandomState(42)
A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns=list('AB'))
B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))

print(A+B)  # 对齐的维度内进行运算，其他部分用NaN代替

# 用A中的均值代替缺失值
fill = A.stack().mean()
print(A.add(B, fill_value=fill))
输出：
      A     B   C
0  10.0  26.0 NaN
1  16.0  19.0 NaN
2   NaN   NaN NaN
       A      B      C
0  10.00  26.00  18.25
1  16.00  19.00  18.25
2  16.25  19.25  15.25
```
4.1处理缺失值None 
```python
如果在一个没有值的数组中执行sum()或min()之类的聚合，通常会得到一个错误  
此外，整数和None之间的加法是未定义的，也会产生错误
```
4.2 另一种缺失值NaN
```python
输入:
import numpy as np

vals2 = np.array([1, np.nan, 3, 4])
print(1 + np.nan)
print(0 * np.nan)
print(vals2.sum())
print(vals2.min())
print(vals2.max())
输出：全部是NaN
应该注意到NaN有点像数据病毒——它会感染它所接触的任何其他对象。无论操作如何，  
与NaN运算的结果将是另一个NaN:

但是一些特殊的聚合，可以忽略NaN,输入：  
import numpy as np

vals2 = np.array([1, np.nan, 3, 4])
print(np.nansum(vals2))
print(np.nanmin(vals2))
print(np.nanmax(vals2))
输出：
8.0
1.0
4.0
```
4.3 .isnull()判断数据是否是无效值，是返回True,否返回False;.notnull()用法相反  
4.4 删除缺失值  
```python
输入：
import numpy as np
import pandas as pd

data = pd.Series([1, np.nan, 'hello', None])
print(data.dropna())  # 删除缺失值

df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
print(df.dropna())  # 删除含有缺失值的整行（默认）或整列
print(df.dropna(axis='columns'))  # 删除整列

df[3] = np.nan  # 给第3列填充所有NaN值
print(df.dropna(axis='columns', how='all'))  # how='all'只删除全部为NaN的值
print(df.dropna(axis='rows', thresh=3))  # thresh=3指定行中最少有3个非空值
```
4.5 填充缺失值
```python
import numpy as np
import pandas as pd

data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
print(data.fillna(0))  # 用0填充缺失值
print(data.fillna(method='ffill'))  # 用缺失值的上一个数填充
print(data.fillna(method='bfill'))  # 用缺失值的后一个数进行填充
```
5.1 层次索引
```python
输入：
import numpy as np
import pandas as pd

index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)

index = pd.MultiIndex.from_tuples(index)  # 采用多索引，返回索引和编码值
print('index:\n', index)

pop = pop.reindex(index)  # 重新索引，数据出现层次变化
print('pop:\n', pop)
输出：
index:
 MultiIndex(levels=[['California', 'New York', 'Texas'], [2000, 2010]],
           codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
pop:
 California  2000    33871648
            2010    37253956
New York    2000    18976457
            2010    19378102
Texas       2000    20851820
            2010    25145561
dtype: int64
```
5.2 创建多索引的方法
```python
输入:
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
print('df:\n', df)

data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}

print('pd.Series(data):\n', pd.Series(data))
输出：
df:
         data1     data2
a 1  0.751396  0.229805
  2  0.224526  0.116114
b 1  0.925029  0.230258
  2  0.832608  0.191625
pd.Series(data):
 California  2000    33871648
            2010    37253956
Texas       2000    20851820
            2010    25145561
New York    2000    18976457
            2010    19378102
dtype: int64
```
5.3 层次索引和列之间数据的重新排列
```python
输入：
import numpy as np
import pandas as pd

index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
print('data:\n', data)

try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)

data = data.sort_index()

print('data:\n', data)
print(data['a':'b'])
输出：
data:
 char  int
a     1      0.959623
      2      0.890802
c     1      0.690089
      2      0.061819
b     1      0.717915
      2      0.050789
dtype: float64
<class 'pandas.errors.UnsortedIndexError'>
'Key length (1) was greater than MultiIndex lexsort depth (0)'
data:
 char  int
a     1      0.959623
      2      0.890802
b     1      0.717915
      2      0.050789
c     1      0.690089
      2      0.061819
dtype: float64
char  int
a     1      0.959623
      2      0.890802
b     1      0.717915
      2      0.050789
dtype: float64
```
6.1 定义函数拼接数据表
```python
输入：
import pandas as pd

def make_df(cols, ind):  # 定义拼接函数
    data = {c: [str(c) + str(i) for i in ind]  # 快速生成数据表
            for c in cols}
    return pd.DataFrame(data, ind)

# example DataFrame
a = make_df('ABC', range(3))
print(a)
输出   ： 
A   B   C
0  A0  B0  C0
1  A1  B1  C1
2  A2  B2  C2
```
6.2 Numpy数组的连接
```python
输入：
import numpy as np

x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
# 调用np中的连接函数，若参数axis=1,按列拼接；为0，按行拼接
x_y_z = np.concatenate([x, y, z])  

print('x_y_z:\n', x_y_z)
输出：
x_y_z:
 [1 2 3 4 5 6 7 8 9]
```
6.3 应用pd中的函数进行拼接，pd.concat
```python
输入：
import pandas as pd

ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
ser1_ser2 = pd.concat([ser1, ser2])

print('ser1_ser2:\n', ser1_ser2)
输出：
ser1_ser2:
 1    A
2    B
3    C
4    D
5    E
6    F
dtype: object
```
6.4 append()的用法
```python
输入：
import pandas as pd

def make_df(cols, ind):  # 调用函数，实现数据集
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])

print('df1:\n', df1)
print('df2:\n', df2)
# 默认按行连接
print('df1.append(df2):\n', df1.append(df2))
输出：
df1:
     A   B
1  A1  B1
2  A2  B2
df2:
     A   B
3  A3  B3
4  A4  B4
df1.append(df2):
     A   B
1  A1  B1
2  A2  B2
3  A3  B3
4  A4  B4
```




