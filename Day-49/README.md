# 深入研究---pandas
```python
1.合并与连接
1.1 合并数据集--merge
import pandas as pd

df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})

print('df1:\n', df1)
print('df2:\n', df2)
# 合并数据集，相同的标签合并在一起，字母按照顺序排
df3 = pd.merge(df1, df2)
print('df3:\n', df3)
输出：
df1:
   employee        group
0      Bob   Accounting
1     Jake  Engineering
2     Lisa  Engineering
3      Sue           HR
df2:
   employee  hire_date
0     Lisa       2004
1      Bob       2008
2     Jake       2012
3      Sue       2014
df3:
   employee        group  hire_date
0      Bob   Accounting       2008
1     Jake  Engineering       2012
2     Lisa  Engineering       2004
3      Sue           HR       2014
1.2 多对1合并
import pandas as pd

df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
df3 = pd.merge(df1, df2)
df1_df5 = pd.merge(df3, df4)  # 保留重复数目

print(df1_df5)
输出:
  employee        group  hire_date supervisor
0      Bob   Accounting       2008      Carly
1     Jake  Engineering       2012      Guido
2     Lisa  Engineering       2004      Guido
3      Sue           HR       2014      Steve

1.3 多对多合并
import pandas as pd

df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df1_df5 = pd.merge(df1, df5)  # 合并无关列会自动扩展
print(df1_df5)
输出：
  employee        group        skills
0      Bob   Accounting          math
1      Bob   Accounting  spreadsheets
2     Jake  Engineering        coding
3     Jake  Engineering         linux
4     Lisa  Engineering        coding
5     Lisa  Engineering         linux
6      Sue           HR  spreadsheets
7      Sue           HR  organization

1.4 此外，merge还可以指定关键字进行一些合并的操作

2.累计与分组
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
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
a = titanic.groupby('sex')[['survived']].mean()  # 按性别划分的存活率
print('a:\n', a)
b = titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()  # 加入阶级的影响
print('b:\n', b)
age = pd.cut(titanic['age'], [0, 18, 80])  # 继续加入年龄的因素
c = titanic.pivot_table('survived', ['sex', age], 'class')
print('c:\n', c)
fare = pd.qcut(titanic['fare'], 2)  # 自动计算分数位
d = titanic.pivot_table('survived', ['sex', age], [fare, 'class'])  # 加入票价信息
print('d:\n', d)
输出：
a:
         survived
sex             
female  0.742038
male    0.188908
b:
 class      First    Second     Third
sex                                 
female  0.968085  0.921053  0.500000
male    0.368852  0.157407  0.135447
c:
 class               First    Second     Third
sex    age                                   
female (0, 18]   0.909091  1.000000  0.511628
       (18, 80]  0.972973  0.900000  0.423729
male   (0, 18]   0.800000  0.600000  0.215686
       (18, 80]  0.375000  0.071429  0.133663
d:
 fare            (-0.001, 14.454]            ... (14.454, 512.329]          
class                      First    Second  ...            Second     Third
sex    age                                  ...                            
female (0, 18]               NaN  1.000000  ...          1.000000  0.318182
       (18, 80]              NaN  0.880000  ...          0.914286  0.391304
male   (0, 18]               NaN  0.000000  ...          0.818182  0.178571
       (18, 80]              0.0  0.098039  ...          0.030303  0.192308

[4 rows x 6 columns]
```
















