---
layout:     post
title:      "RNN, LSTM记录"
subtitle:   "RNN, LSTM介绍与实现"
date:       2018-07-07
author:     "Shi Lou"
catalog: true
categories: Deep Learning
tags:
    - RNN
    - LSTM
    - Deep Learning
---
> RNN, LSTM详细介绍，以及利用tf api和手动实现两种方式复现。  
<!-- more -->

# RNN,LSTM介绍和实现  
## RNN  
RNN有两种，一种是基于时间序列的循环神经网络Recurrent Neural Network，另一种是基于结构的递归神经网络Recursive Neural Network。我们平时讲的RNN一般情况下是指第一种。  

所谓基于时间序列，是指输入的数据可以认为在时间片上是有前后关系的，即当前时间点产生的状态受到之前时间点的影响，同时会影响后续时间点的输出状态。例如文本、语音都可以看成是时间序列数据。图像在理论上不是时间序列的，但是RNN的核心是捕捉前后两个input之间的联系，而图像的像素彼此之间也是存在一定关联的，因此通过对像素构建序列，使得RNN在图像任务上也取得了一些不错的表现，比如手写识别$^{[1]}$任务中。  

RNN的结构如下： 

<!-- <div align=center> -->

![unrolled_rnn](https://i.loli.net/2018/08/04/5b654b0b9d8ba.png) <!--/div --> 
<center>图 1</center>

以文本任务举例，输入$(x_0,...,x_n)$就是一句话，每个$x_t$对应句子中的一个字，输入的长度就是句子的长度。如果输入的是一个batch的数据，则需要对句子的长度进行预处理，对短句子补<PAD>，对长句子进行截断，确保batch中各个句子的长度是相等的。而对于每个字，可以用one-hot编码，也可以用word embedding技术进行编码。如果用word embedding，在初始化vocabulary dictionary矩阵的时候可以是使用预训练的embedding，也可以随机初始化，因为模型会在训练的时候同时把word embedding也训练出来。因此输入部分的维度就是`(batch_size, seq_len, embed_dim)`，对单个$x_n$来说即`(batch_size, embed_dim)`  

再来看$A$，$A$就是一个线性变换加激活函数  
$$A=f(x_t, h_{t-1})=active\_func(W_{hx}x_t+W_{hh}h_{t-1}+b)$$
其中激活函数$active\_func$可以取$tanh(\cdot)$、$relu(\cdot)$等。假设RNN中隐藏单元数为`rnn_units`。则$W_{hx}$的维度就是`(embed_dim, rnn_units)`，$W_{hh}$的维度为`(rnn_units, rnn_units)`，$b$的维度为`(rnn_units)`，计算后得到的$h_t$维度即`(batch_size, rnn_units)`。这个$h_t$就是RNN的输出，这个输出分两个方向，一是输出到RNN单元外，二是和下一个$x_{t+1}$一起作为下一个时间序列的输入。  

**NOTE:** 图1中**所有**的$A$是**同一个**完整的RNN单元，里面包含了`rnn_units`个隐藏单元，即所有的$A$是共享参数的，因为同一个嘛。  

以上就是一个完整的RNN单元，这种结构充分考虑了前后序列的信息关系，但它本质上是一种递归嵌套结构，如果不作任何处理，当时间序列长度为$n$很大时，在BPTT过程（梯度根据时间的反向传播）受到$n$次幂的影响，其值会积累到很大或衰减为0，这样就失去了之前序列的信息。因此它在处理长时间序列问题时效果不好$^{[2]}$。为了解决长依赖的问题，就有了后来的LSTM和GRU等方法。对于BPTT过程为何会产生gradient explode/gradient vanish问题，这里只提供一个直觉上的理解：设想你在读一句很长很长的话，可能长达几百上千字，当你读到最后几个字的时候，早先的记忆是不是就已经模糊了？如果想理解更多细节，具体的公式推理可以看[这里$^{[3]}$](https://zhuanlan.zhihu.com/p/27485750)。

## LSTM 
Long-Short Term Memory简单来说，就是在之前RNN单元$A$中加了一些门控单元gate。LSTM结构如下：
<!-- <div align=center> -->

![lstm_cell](https://i.loli.net/2018/08/04/5b655b8e0a24e.png)
<!-- </div> -->

经典的LSTM比起RNN，在输出隐层状态$h_t$的同时还输出了当前单元的记忆$c_t$（图中上面那个水平箭头），并且在RNN的基础上加入了三个gate，分别是：  
- 遗忘门$f$: 控制遗忘多少前一时刻的记忆
- 输入门$i$: 控制当前时刻多少信息能被有效输入
- 输出门$o$: 控制当前时刻记忆输出多少  

LSTM的核心公式为以下五个：
<center>
$f_t=sigmoid(W_{fx}x_t+W_{fh}h_{t-1}+b_f)$
$i_t=sigmoid(W_{ix}x_t+W_{ih}h_{t-1}+b_i)$
$o_t=sigmoid(W_{ox}x_t+W_{oh}h_{t-1}+b_o)$
$c_t=f_t \cdot c_{t-1}+i_t \cdot tanh(W_{cx}x_t+W_{ch}h_{t-1}+b_c)$
$h_t=o_t \cdot tanh(c_t)$
</center>

三个gate选择$sigmoid(\cdot)$的原因是gate只是负责控制信息**流通率**，本身是不产生额外信息的，$sigmoid(\cdot)$能很好地表现这个性质。额外信息只由自时刻$t$中的输入$(x_t,h_{t-1})$产生。LSTM更详细的工作流程可以看[这篇$^{[4]}$](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)。  


同样的，为什么LSTM能够解决BPTT中的gradient exploed/gradient vanish问题，这里也只给出一个直觉上的解释：还是之前那个阅读长句子的例子，之所以读到最后记不住早前的记忆，是因为RNN尝试把所有的内容都记下来。LSTM中的gate控制着信息的流通率，可以看成只尝试去记忆之前关键的信息，这样就减少了记忆的负担，因此能很好地解决长依赖问题。如果更深入地理解BPTT过程，可以看之前提到的RNN BPTT那篇文章。  

## 实现部分  
这部分用一个简单的word rebuild任务来验证LSTM结构，即训练数据的source是word，target是这个word的一个随机排序，目标是通过source预测target。模型还会用到seq2seq的相关知识，想要详细了解seq2seq的，可以看seq2seq的[原始论文$^{[5]}$](https://arxiv.org/pdf/1409.3215.pdf)。实现会以Tensorflow封装好的seq2seq模块和自我实现两种方式。

### 数据集介绍
原始数据集是小说*On Sunset Highways*，进行简单的预处理，去除掉连在单词后面的标点符号，提取出所有字符长度大于1的不重复单词，对各单词进行随机排序，确保source跟target单词是不同的。  

### 利用Tensorflow中seq2seq模块实现  
Tensorflow为了方便用户使用，已经把基本的seq2seq组建都封装好了，使用的时候直接调用就可以，非常方便。下面简单介绍：  

```python
import tensorflow.contrib as contrib
from tensorflow.contrib.seq2seq import *

def lstm(rnn_units):
    return contrib.DropoutWrapper(contrib.rnn.BasicLSTMCell(rnn_units))

def encoder(encoder_input, encoder_length):
    cell = lstm(128)
    _, encoder_states = tf.nn.dynamic_rnn(cell,
                                          input=encoder_input,
                                          sequence_length=encoder_length,
                                          dtype=tf.float32)
    return encoder_states

def decoder(encoder_states, embedding, decoder_input, decoder_length):
    cell = lstm(128)
    output_layer = tf.layers.Dense(vocab_size)
    rnn_output = decoder_train(cell, decoder_input, decoder_length, encoder_states, output_layers)
    sample_id = decoder_infer(cell, encoder_states, output_layers, embedding)
    return rnn_output, sample_id

def decoder_train(cell, decoder_input, decoder_length, encoder_states, output_layer):
    with tf.variable_scope("decoder"):
        train_helper = TrainingHelper(decoder_input, decoder_length)
        decoder = BasicDecoder(cell, 
                               train_helper, 
                               encoder_states, 
                               output_layer)
        decoder_output, _, _= dynamic_decode(decoder,
                                             impute_finished=True, 
                                             maximum_iterations=30)
    return decoder_output.rnn_output

def decoder_infer(cell, encoder_states, output_layer, embedding):
    with tf.variable_scope("decoder", reuse=True):
        start_tokens = tf.tile(tf.constant([word2idx["<SOS>"]], dtype=tf.int32), [batch_size])
        end_token = word2idx["<EOS>"]
        infer_helper = GreedyEmbeddingHelper(embedding,
                                             start_tokens=start_tokens,
                                             end_token=end_token)
        decoder = BasicDecoder(cell, infer_helper, encoder_states, output_layer)
        decoder_output, _, _ = dynamic_decode(decoder,
                                              impute_finished=True, 
                                              maximum_iterations=30)
    return decoder_output.sample_id
```

主要分为创建lstm单元、创建encoder、创建decoder三个部分。encoder的输出encoder_states即lstm的最终输出，是最后一个时刻$t$输出的隐状态$h_t$和记忆$c_t$的组合，因为这里只用了一层LSTM，因此就是一个二元组$(c_t, h_t)$，两者的维度都是`(batch_size, rnn_units)`。  
decoder部分训练和预测的时候有些不一样，因为LSTM是根据当前状态预测下一个状态的，会有一个误差累积的过程，当序列很长时误差会累积到很大，结果是序列末的预测变得不可信。因此在训练时用了一个trick，即decoder在预测下一时刻的输出时总是用本时刻的真实输入，而不是预测产生的值，这就是Teacher Forcing方法，减轻了误差累积的影响，具体可以看[这里$^{[6]}$](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)。seq2seq模块中的TrainingHelper()已经为我们封装好了，直接用就行。预测的时候由于没有真实的target作为辅助，因此只能用生成的token作为下一时刻的输入。decoder的输出包含了rnn_output和sample_id，前者softmax后的概率分，用来计算loss，后者是预测的token，作为模型的预测输出。完整的代码可以看[Github]()。  

### 自主实现  
实现seq2seq比较简单，只要实现LSTM、全连接层output，以及encoder和decoder部分即可。先来看LSTM，根据之前的公式，只需要定义各gate的权重参数$W$和$b$，进行线性变换再激活一下就行了，这部分代码如下:  
```python
# input_x: (batch_size, embed_dim)
h_pre, c_pre = last_states
x = tf.concat([input_x, h_pre], axis=1)   # (input_x; last_states)
i, f, o, c_ = tf.split(tf.nn.xw_plus_b(x, self.W, self.b), 4, axis=1)
c = tf.sigmoid(f) * c_pre + tf.sigmoid(i) * tf.tanh(c_)
h = tf.sigmoid(o) * tf.tanh(c)
output, states = h, (h, c)
if mask is not None:
    output = tf.where(mask, tf.zeros_like(h), h)
    states = (tf.where(mask, h_pre, h), tf.where(mask, c_pre, c))
```

其中`mask`是掩码矩阵，对于一个batch中长度较短的句子，因为之前进行了PAD处理，因此计算FP和BP时不应计算PAD部分，以免对梯度造成影响。  

output部分比较简单，就不展开了。再看一下encoder和decoder部分，这里自主实现利用了Tensorflow中的TensorArray。TensorArray可以看作是装Tensor的数组，比较重要的几个方法有read，write，stack，unstack，详细用法可以看Tensorflow[官方文档](https://www.tensorflow.org/api_docs/python/tf/TensorArray)。实现的代码如下：  
```python
time = tf.constant(0, dtype=tf.int32)

h0 = (tf.zeros([batch_size, rnn_units], dtype=tf.float32),
        tf.zeros([batch_size, rnn_units], dtype=tf.float32))
mask = tf.zeros([batch_size], dtype=tf.bool)
inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_length)
inputs_ta = inputs_ta.unstack(tf.transpose(encoder_input, [1, 0, 2]))
outputs_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

def loop_func(t, x_t, s_pre, outputs_ta, mask):
    o_t, s_t = cell(x_t, s_pre, mask)
    outputs_ta = outputs_ta.write(t, o_t)
    mask = tf.greater_equal(t+1, encoder_input_length)
    x_next = tf.cond(tf.reduce_all(mask),
                        lambda: tf.zeros([batch_size, embed_dim], dtype=tf.float32),
                        lambda: inputs_ta.read(t+1))
    return t+1, x_next, s_t, outputs_ta, mask

_, _, state, output_ta, _ = tf.while_loop(
    cond=lambda t, _1, _2, _3, _4 : t < max_length,
    body=loop_func,
    loop_vars=(time, inputs_ta.read(0), h0, outputs_ta, mask)
)
```

实现的逻辑也很简单，对时刻$t$的输入$x_t$和前一时刻的隐层状态$h_{t-1}$执行一次LSTM计算过程，把结果写进TensorArray中，继续读入下一个时刻的输入$x_{t+1}$，判断一下是不是PAD，是的话就置0。然后重复这个过程就行了。  

### 评测
用Tensorflow提供的API训练完模型后的预测结果： 
   
```
source: Encinitas, predict: niscntaiE<EOS>
source: destroying, predict: diusrntiah<EOS>
source: tape, predict: atpe<EOS>
source: pier, predict: rpie<EOS>
source: unexpected, predict: teceupedxn<EOS>
source: selecting, predict: tleinsecg<EOS>
source: stocked, predict: cdoesctk<EOS>
```
  
可以看到，模型已经学到了预测规则，预测的结果基本都是输入的一个排列。  

用自己实现的组建训练完模型后的预测结果：  
```
Model loaded.
source: Encinitas, predict: initnanh<EOS>
source: destroying, predict: otirenidh<EOS>
source: tape, predict: ptea<EOS>
source: pier, predict: epir<EOS>
source: unexpected, predict: eecdnetcm<EOS>
source: selecting, predict: einletics<EOS>
source: stocked, predict: eocseao<EOS>
```

可以看到，我们自己实现的组建也能够学到预测规则，但是比起Tensorflow提供的API，其预测能力要差一点，这里原因没有深入分析，如果有深入了解的，还请多多指教。  

整个demo的代码以及数据我都放到了[Github](https://github.com/ybshen007/tensorflow-practice/tree/master/rnn)上，有需要的同学可以自取。


## REFERENCE
[[1] Fast and robust training of recurrent neuralnetworks for offline handwriting recognition](http://www.icfhr2014.org/wp-content/uploads/2015/02/ICFHR2014-Doetsch.pdf)  
[[2] Hochreiter (1991) [German]](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf)  
[[3] 当我们在谈论 Deep Learning：RNN 其常见架构](https://zhuanlan.zhihu.com/p/27485750)  
[[4] Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
[[5] Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)  
[[6] What is Teacher Forcing for Recurrent Neural Networks?](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)
