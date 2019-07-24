# 深入研究---pandas



2.累计与分组
```python
2.1 应用Series
import seaborn as sns
import numpy as np
import pandas as pd

planets = sns.load_dataset('planets')  # 包自带数据集
# print(planets.shape)
# print(planets.head())

# 简单聚合
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))  # 默认创建整形索引
print('ser:\n', ser)
print('ser.sum:\n', ser.sum())
print('ser.mean:\n', ser.mean())
输出：
ser:
 0    0.374540
1    0.950714
2    0.731994
3    0.598658
4    0.156019
dtype: float64
ser.sum:
 2.811925491708157
ser.mean:
 0.5623850983416314
2.2 对于DataFrame,聚合返回每个列中的结果
import numpy as np
import pandas as pd

rng = np.random.RandomState(42)
df = pd.DataFrame({'A': rng.rand(5),
                   'B': rng.rand(5)})

print(df)
print(df.mean())
print(df.mean(axis='columns'))
输出:
          A         B
0  0.374540  0.155995
1  0.950714  0.058084
2  0.731994  0.866176
3  0.598658  0.601115
4  0.156019  0.708073

A    0.562385
B    0.477888
dtype: float64

0    0.265267
1    0.504399
2    0.799085
3    0.599887
4    0.432046
dtype: float64
2.3 采用describe聚合
import seaborn as sns

# describe()，它为每个列计算几个公共聚合并返回结果。
planets = sns.load_dataset('planets')
print(planets.dropna().describe())
输出：
    number  orbital_period        mass    distance         year
count  498.00000      498.000000  498.000000  498.000000   498.000000
mean     1.73494      835.778671    2.509320   52.068213  2007.377510
std      1.17572     1469.128259    3.636274   46.596041     4.167284
min      1.00000        1.328300    0.003600    1.350000  1989.000000
25%      1.00000       38.272250    0.212500   24.497500  2005.000000
50%      1.00000      357.000000    1.245000   39.940000  2009.000000
75%      2.00000      999.600000    2.867500   59.332500  2011.000000
max      6.00000    17337.500000   25.000000  354.000000  2014.000000
2.4 Split, apply, combine
import pandas as pd

# 生成数据集
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)}, columns=['key', 'data'])
df_sum = df.groupby('key').sum()  # 合并同类项
print(df_sum)
输出：
     data
key      
A       3
B       5
C       7
2.5 聚合、筛选、转换、应用-aggregation

import pandas as pd
import numpy as np

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns =['key', 'data1', 'data2'])

print('df:\n', df)

# 使用aggregation聚合有更大的灵活性，求列数据中的最小、最大、均值值
df_aggregation = df.groupby('key').aggregate(['min', np.median, max])
print(df_aggregation)
输出：
df:
   key  data1  data2
0   A      0      5
1   B      1      0
2   C      2      3
3   A      3      3
4   B      4      7
5   C      5      9
    data1            data2           
      min median max   min median max
key                                  
A       0    1.5   3     3    4.0   5
B       1    2.5   4     0    3.5   7
C       2    3.5   5     3    6.0   9
2.6 过滤操作--基于属性删除数据
接上输入:
def filter_func(x):
    return x['data2'].std() > 4
print(df', "df.groupby('key').std()", "df.groupby('key').filter(filter_func)")
2.7 转化功能-transform
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns = ['key', 'data1', 'data2'])
# 减去组均值来居中数据
a = df.groupby('key').transform(lambda x: x - x.mean())
print(a)
输出：
   data1  data2
0   -1.5    1.0
1   -1.5   -3.5
2   -1.5   -3.0
3    1.5   -1.0
4    1.5    3.5
5    1.5    3.0
2.8 提供分组键的列表
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns = ['key', 'data1', 'data2'])
L = [0, 1, 0, 1, 2, 0]
# 按照L的索引顺序，求和聚合
a = print(df, '\n', df.groupby(L).sum())
print(a)
输出: key  data1  data2
0   A      0      5
1   B      1      0
2   C      2      3
3   A      3      3
4   B      4      7
5   C      5      9 
    data1  data2
0      7     17
1      4      3
2      4      7
None
3.数据透视表
3.1 激励数据透视表

```
















