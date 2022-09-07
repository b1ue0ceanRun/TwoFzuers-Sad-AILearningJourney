# Task 1: Introduction and Word Vectors

1st Day  2022/9/6一周入门NLP

https://github.com/datawhalechina/team-learning-nlp/tree/master/IntroductionToNLP

https://www.bilibili.com/video/BV1s4411N7fC

### 理论部分

1. 介绍NLP研究的对象
2. 如何表示单词的含义
3. Word2Vec方法的基本原理

### Human language and word meaning

Language is pretty special thing. The meaning of a word can be represented by word itself and the context.

传统词典[Wordnet](https://zh.wikipedia.org/wiki/WordNet)拉了，不能解决语境问题。经典方法one-hot vector表示words拉了，不能表示**词间相似度**且有**维度爆炸**问题。

### Distributed representation

分布式表示，将词（word）Violet稠密向量表示，词的维度通常几百到几千，但是都比one-hot vector法小。

$$
\text { \textcolor{Cyan}{\textit{expect}} }=\left(\begin{array}{r}
0.286 \\
0.792 \\
-0.177 \\
-0.107 \\
0.109 \\
-0.542 \\
0.349 \\
0.271 \\
0.487
\end{array}\right)
$$

### [Word2vec](https://arxiv.org/pdf/1301.3781.pdf)

[word2vec介绍_vincent_duan的博客-CSDN博客_word2vec](https://blog.csdn.net/vincent_duan/article/details/117967110)

***Word2vec*** 是一种学习词向量的框架

* 包含大量的文本语料
* 固定词表中的每一个单词由一个词向量表示
* 文本中的每个单词位置$t$，有一个中心词$c$，和它的上下文$o$(除了$c$的外部单词）。
* 通过$c$和$o$的词向量相似性来计算 $P(o|c)$
* 不断的调整词向量，以达到最大化概率

probability module
对每个词的位置$t$，固定窗口大小$m$，扫一遍文本$T$，扫的中心词为$w_t$。

$$
Likelihood=L(\theta)=\prod_{t=1}^{T}\prod_{-m \le j \le m\ \ \ \ \ j\ne0}^{}P(w_{t+j}|w_t;\theta),
$$

objective function

$$
J(\theta)=-\frac 1TlogL(\theta)=-\frac1T\sum_{t=1}^T\sum_{-m\le j\le m \ \ \  j\ne 0}logP(w_{t+j}|w_t;\theta)
$$

其中$P(\cdot)$为概率函数

#### 概率函数的计算

对于每个单词$w$使用两个向量$v_w$和$u_w$

$u_w$：当$w$是中心词

$v_w$：当$w$是上下文词

中心词$c$，上下文词$o$，有

$$
P(o|c)=\frac {exp(u_o^Tv_c)}{\sum_{w\epsilon V}exp(u_w^Tv_c)}
$$

向量点积后做softmax

#### 训练优化参数$\theta$

$$
\theta=\left[\begin{array}{l}
v_{\text {aardvark }} \\
v_{a} \\
\vdots \\
v_{z e b r a} \\
u_{\text {aardvark }} \\
u_{a} \\
\vdots \\
u_{z e b r a}
\end{array}\right] \in \mathbb{R}^{2V\times }
$$

用$d$维向量表示一个单词，词典大小为$V$
**梯度下降**优化$J(\theta)$



