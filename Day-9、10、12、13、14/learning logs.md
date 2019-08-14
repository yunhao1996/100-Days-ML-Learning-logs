# 学习记录-支持向量机（SVM）  
1.`SVM`:*Support Vetor Machine*  ,监督学习，可用于分类和回归，主要用于分类。关键是找到一个超平面用于分割，简单来说，找到一个分割界面，放到两个类别之   间，把他们分开。基本思想：如果在低维不可分时，就先通过核函数（KERNEL）把低维数据映射到高维，高维上变成线性可分，然后在高维上再用支持向量机进行分.

2.`SVC`参数说明:[参考文章](https://blog.csdn.net/guanyuqiu/article/details/85109441)    
  几个重要的参数：  
  核函数（kernel）：寻找超平面，就是通过线性代数转化问题。通过核函数来完成。本例中为linear。
  [参考文章](https://www.cnblogs.com/jerrylead/archive/2011/03/18/1988406.html)  
  正则化（regularization）：较大时，选较小间距的超平面  
  系数（gamma）：小的系数值，距离远的点也会用于计算,gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合  
  间距（margin）：到最近点的分界线  
  
3.`s.t.`:subject to(受限制于...)  

4.对偶问题:针对于线性可分时的分析，原问题是求*W*在约束条件下的最小值，通过拉格朗日乘子，相当于转化为求*α*的在约束条件下的最大值，这就是对偶问题。[参考文章](https://zhidao.baidu.com/question/42948452.html)  
5.拉格朗日乘子法（Lagrange Multiplier) 和KKT条件:[参考文章](https://blog.csdn.net/xianlingmao/article/details/7919597)  

6.`LR`与`SVM`的比较：[参考文章](https://blog.csdn.net/zwqjoy/article/details/82312783)   

7.`Kernels`：比如二维线性不可分时，将数据映射到三维，让`z=x**2`,增加第三个特征，或许此时可是用一个超平面进行分开,然后再变换回二维空间，超平面变成了线，可能是圆或者椭圆等。对于SVM,将数据映射到更高维的空间中有两种常用的方法：一种时多项式核，在一定阶数内计算原始特征所有可能的多项式（比如前面的形式）；另一种是径向基函数（RBF:radial basis function）核，也就是nb的高斯核（理解不了），因为它对应无限维的特征空间。关于其他的了解[参考文章](https://www.cnblogs.com/jerrylead/archive/2011/03/18/1988406.html)  

8.按照图片的内容，调整参数对SVM非常重要，这同样也是他的一个缺点,虽然我也体会不到这种感觉。`gamma`和`C`控制的都是模型复杂度，强烈相关，要同时调节，`RBF`核只有一个参数`gamma`，是高斯核宽度的倒数。  

9.原理理解  
  线性可分时：  
<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-9、10、12、13、14/pictures/1.png">
</p> 

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-9、10、12、13、14/pictures/4.png">
</p> 

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-9、10、12、13、14/pictures/3.png">
</p> 

<p align="center">
  <img src="https://github.com/yunhao1996/100-Days-ML-Learning-logs/blob/master/Day-9、10、12、13、14/pictures/2.png">
</p> 

