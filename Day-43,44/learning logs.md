# K-均值聚类
```
1.什么是无监督学习？  
  
无监督学习允许我们在对结果无法预知时接近问题。无监督算法只基于输入数据找到模式。当我们无法确定寻找内容时，这个技术很有用。
```
```
2.聚类算法？  

聚类算法用于把族群或数据点分割成一系列的组，使得相同簇中的数据点比其他的组更相似。基本上，目的是分  隔具有相似形状的组，并  
且分配到簇中。  
```
```
3.K-均值聚类介绍?

在这个算法中。我们把所有项分成k个簇，使得相同簇中的所有项彼此尽量相似，而不同簇的项尽量不同。  
距离测量（类似欧式距离）用于计算数据点的相似度和相异度。每个簇有一个形心。形心可以理解为最能代表簇的点。
```
```
4.K-均值聚类如何工作？
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-43%2C44/pictures/1.jpg">
</p> 

```
我们执行K均值聚类算法是这样的。首先随机选择两个点，这两个点叫做聚类中心（cluster centroids），也就是图中红色和蓝色的交叉。  
K均值聚类 一个迭代的方法，它要做两件事，一件是簇分配，另一件是移动聚类中心。
在K均值聚类算法的每次循环里面，第一步要进行的是簇分配。首先要历遍所有的样本，也就是上图中每一个绿色的点，然后根据每一个点  
是更接近红色的这个中心还是蓝色的这个中心，将每一个数据点分配到两个不同的聚类中心。
```

```
例如第一次我们随机定的两个中心点和其簇分配如下图所示:
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-43%2C44/pictures/2.jpg">
</p> 
```
第二步要做的自然是要移动聚类中心。我们需要将两个中心点移动到刚才我们分成两类的数据各自的均值处。那么所要做的就是找出所有红色的点计算出他们的均值，然后把红色叉叉移动到该均值处，蓝色叉叉亦然。
```

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-43%2C44/pictures/3.jpg">
</p> 

```
然后通过不断重复上述两个步骤，通过不断迭代直到其聚类中心不变，那么也就是说K均值聚类已经收敛了，我们就可以从该数据中找到两个最有关联的簇了。其过程大概如下图所示：
```
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-43%2C44/pictures/4.jpg">
</p> 

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-43%2C44/pictures/5.jpg">
</p> 

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-43%2C44/pictures/6.jpgg">
</p> 

