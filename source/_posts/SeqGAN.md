---
layout:     post
title:      "SeqGAN"
subtitle:   "SeqGAN介绍"
date:       2018-05-15
author:     "Shi Lou"
catalog: true
tags:
    - SeqGAN
    - GAN
    - RL
---
> SeqGAN的介绍、作用及训练过程。  
<!-- more -->
  
## GAN介绍
GAN$^{[1]}$的本质是最小化生成样本分布和真实数据分布之间的差距。实现思路是交替地训练两个网络$Generator$、$Discriminator$，其中$Gen$要生成尽可能真实的样本，目标是迷惑$Dis$，使得$Dis$难以判别该样本是来自real data还是synthesized data；$Dis$的目标则显然是最大化判别样本真实与否的能力。 
## GAN训练过程

令$x$表示real data(可以看成一张真实的图片)，$z$表示随机噪声，$D(\cdot)$表示$Discriminator$，$G(\cdot)$表示$Generator$，则$G(z)$即表示生成的假样本synthesized data(可以看成是一张生成的假图)。 

GAN的目标对$D(\cdot)$而言，是最大化$D(x)$，最小化$D(G(z))$；对$G(\cdot)$而言，要最大化$D(G(z))$。以上的过程可以用一个数学公式统一表达： 

$$arg\underset{G}{Min}\underset{D}{Max}V(G,D)=\mathbb E_{x～P_{real}(x)} \cdot logD(x) + \mathbb E_{z～P_{z}(z)} \cdot log[1-D(G(z))] \tag{1}$$

其中$\underset{D}{max}V(G,D)$表示$V(G,D)$最大时$D$的取值。 
训练时首先固定$G$，优化$D$。$V(G,D)$对$D$求导，可得最优的$D$为 
$$D_G^\ast=\frac{P_{real}(x)}{P_{real}(x)+P_{g}(x)}\tag{2}$$
证明:
对于连续空间，期望$\mathbb E$的计算方法是求积分，因此有
$$\begin{eqnarray}
V(G,D)&=&\int_x{P_{real}(x)logD(x){\rm d}x} +\int_z{P_{z}(z)log[1-D(G(z))]{\rm d}z} \tag{3}\\\\
      &=&\int_x{P_{real}(x)logD(x)+P_{g}(x)log[1-D(x)]{\rm d}x} \tag{3})
\end{eqnarray}$$

将其积分符号内的部分对$D$求导，并使其为$0$。
$$\frac{dV(G,D)}{dD}=\frac{P_{data}(x)}{D(x)}-\frac{P_g(x)}{1-D(x)}=0\tag{4}$$ 
即得式$(2)$，将其带入式$(3)$得 
$$\begin{eqnarray}
V(G,D^*)&=&-log(4)+KL(P_{data}||\frac{P_{data}+P_{g}}{2})+KL(P_{g}||\frac{P_{data}+P_{g}}{2}) \tag{5}\\\\
&=&-log(4)+2JSD(P_{data}||P_{g}) \tag{5}
\end{eqnarray}$$
其中$KL(\cdot)$，$JSD(\cdot)$分别表示[$KL$散度](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)和[$JS$散度](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)，关于熵跟$KL$散度的关系，也可以点[这里$^{[2]}$](https://www.zhihu.com/question/41252833)。 

GAN的训练过程可以用下图表示
$$\color{red}{\rm{IMAGE}}$$
其中黑线表示真实数据分布，绿线表示生成数据分布，蓝线为$Dis$，对于蓝线而言，纵坐标可以表示$Dis$判别为真的概率。图a中是初始化的$Dis$；图b开始训练$Dis$，逐渐能开始识别出real data和synthesized data(分别给出概率为1和0)；图c开始训练$Gen$，生成的样本分布(绿线)开始向真实分布靠拢；图d是最终训练完成的理想状态，生成样本分布和真实样本分布完全重合，$Dis$的判断结果总是0.5，即无法分辨出样本的真假。 

## SeqGAN
### Why SeqGAN？
为什么要引入SeqGAN呢？因为传统的GAN只能解决数据空间连续分布的问题，例如图像的生成问题。图像在计算机中是以二维矩阵的模式存储的，对一张1024\*1024的黑白图片而言，就可以用1024\*1024个Pixel来表示。如果图片是彩色对，只需再引入一个维度来表示RGB通道。但无论怎样，每个Pixel都是实数空间上可微的，即对某个Pixel进行一个细小的变化${\rm d}Pixel$，在原始图像上就可以表现出该像素点颜色加深/变浅之类的变化。GAN也跟传统的networks一样，使用了back propagation技术，通过对权重矩阵进行一个细微的变化来影响产生图片的质量。由于Pixel连续分布的性质，使得weight matrix的细微变化是有意义的，且能产生作用。 

而对于离散分布的数据而言，以上的过程就没有意义了。如果我们假设Pixel的分布不再是连续的，而是离散的整数值。比如Pixel=0表示红色，Pixel=1表示黄色，Pixel=2表示蓝色。现在对于图片中的某个像素点而言，它可能原始值是Pixel=0，即表示一个红色的像素点。经过一轮forward propagation和back propagation后，图片的weight matrix发生了一些改变，最终结果是这个Pixel值变成了1.1。问题是，这个1.1表示什么呢？在连续空间上，它可能表示比红色深一点点，或者浅一点点，无论如何，它总是有意义的。但是在离散空间中，它就失去了意义。 

再来看文本的例子，文字就是典型的离散数据，尽管有word embedding技术，给人感觉上word变成了连续的值，但事实上word在空间上的分布依然是离散的。比如“中国”的embedding是(1.1, 1.2, 1.3, ...)，假设它的第一个维度变成了1.2，在word embedding空间上很可能就找不到对应的点了，即使有，它表示的含义跟“中国”也是天差地别，而非之前像素渐变的过程。 

那如果硬套GAN到文本生成的任务会发生什么呢？首先来看一下整个过程： 

Gen输出softmax之后的结果，即对下一个词在整个词表上的概率预测，sampling一个概率最大的词，重复上述过程直到生成一个句子。Dis对该句子进行判断并优化Gen，优化的方式是通过最小化Dis的loss，然后反传回更新后的参数。问题是梯度在反传时遇到sampling这个环节就中断了（微小的变化并不能影响Sampling的结果）。那有人会说，如果不进行sampling，直接把word distribution传给$Dis$不就好了吗？这样也是不行的，这样对于$Dis$来说，他看到的real data本质上是one hot的形式，而synthesized data则是实数向量的形式，那就$Dis$很容易就判断one hot形式的是真，实数向量形式的是假，而不能真正意义上区分出real data和synthesized data。（如果这里使用了word embedding的技术，可以解决问题。）李宏毅教授在他的课程中也对这个问题作出了一些解释，他的课讲得非常好，可以看[这里$^4$](https://www.youtube.com/watch?v=pbQ4qe8EwLo)。
### SeqGAN是怎么做的？
跟普通的GAN相比，SeqGAN引入了强化学习的Policy Network。
待续


## REFERENCE 
1. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
2. [熵和交叉熵](https://www.zhihu.com/question/41252833)
3. [SeqGAN介绍](https://zhuanlan.zhihu.com/p/29168803)
4. [李宏毅老师的homepage](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html)