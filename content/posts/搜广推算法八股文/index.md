+++
title = '搜广推算法八股文'
date = 2019-03-22T19:47:48+08:00
draft = false
math = false
tags = ['面试','算法']
+++

### 机器学习基础

- AUC/F1，能手写代码实现
- L1/L2正则化，区别
- lightgbm/catboost/xgboost区别
- 激活函数
- word2vec, Skip-Gram和BOW
- DNN能直接把所有权重初始化为0吗
- 各种Normalization

#### Transformer

Transformer结构问题：精读llama的paper和Kapthy的GPT代码对Transformer的结构就能很清晰，

- Transformer为什么不用BN，LN和BN的区别
- Transformer multi-attention为什么会更好，怎么计算
- Transformer的fc在个位置
- Transformer激活函数
- GPT和Bert对比
- Transformer和GPT的优化器
- 位置编码，RoPE

Transformer推理问题：
- KV Cache
- 

### 搜广推算法

- 怎么判断学到的Embedding好不好？
- user_id可有用作特征吗？
- 哪个特征提点最多
- 召回路能作为特征加到精排中作为特征吗

#### 召回和粗排

- 召回常见通路，多路召回怎样融合
- 召回评价指标。场景内HitRate，进一步可以优化全域的HitRate
- 新增召回链路遇到什么问题
- 双塔召回能使用交叉特征吗？
- TDM流程，怎么处理新的广告

#### 介绍一下wide&deep？

- 哪些特征适合放在deep侧，哪些特征适合放在wide侧？
- wide&deep用的什么优化器？介绍一下FTRL？

#### 多目标和多任务

- 介绍下ESSM，ESSM是为了解决什么问题，彻底解决了吗？
    - 什么情况下适合用ESSM，或者说ESSM比每个目标独立建模效果要好？
    - 在ESSM上，有哪些改进的地方？
- ESMM（注意是ESMM不是ESSM）相比ESSM的改进是什么，为什么要对部分样本stop-gradient？
- 介绍下MMoE模型
    - MMoE为什么有效，解决什么问题？
    - 实践中MMoE遇到什么问题？Gate坍缩问题（gate学到的权重极度不均衡，少数experts接近1，其他experts接近0）如何缓解？
    - MMoE在实践中还有什么tricks吗，expert和gate的选取，expert权重的计算？
    - MMoE缺点？PLE相比MMoE改进了什么？
- 在实践中会遇到增加targets的情况，如何热启动？哪种方法效果好？
- 为什么要共享Embedding
- 不同场景的数据联合训练更好还是拆分训练成多个模型更好？
- 怎么解决延迟转化问题？从模型角度和延迟转化数据的利用角度

#### 序列模型

- 序列特征怎么做预处理
- DIN及相关模型介绍？DIEN相比DIN的改变？
- 使用了什么特征？
- DIN的target attention是怎么计算的？
- DIN和transformer对比



