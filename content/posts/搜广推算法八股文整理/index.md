+++
title = '搜广推算法八股文整理'
date = 2019-03-22T19:47:48+08:00
draft = true
math = false
tags = ['面试','算法']
+++

### 机器学习基础

- AUC/F1
- Transformer为什么不用BN，LN和BN的区别
- Transformer multi-attention为什么会更好，怎么计算
- Transformer的fc在个位置
- Transformer激活函数
- GPT和Bert对比
- Transformer和GPT的优化器
- L1/L2正则化
- lightgbm/catboost/xgboost区别
- 激活函数
- word2vec, Skip-Gram和BOW

### 介绍一下wide&deep？

- 哪些特征适合放在deep侧，哪些特征适合放在wide侧？
- wide&deep用的什么优化器？介绍一下FTRL？

### 多目标和多任务

- 介绍下ESSM，ESSM是为了解决什么问题，彻底解决了吗？
    - 什么情况下适合用ESSM，或者说ESSM比每个目标独立建模效果要好？
    - 在ESSM上，有哪些改进的地方？
- 介绍下MMoE模型
    - MMoE为什么有效，解决什么问题？
    - 实践中MMoE遇到什么问题？Gate坍缩问题（gate学到的权重极度不均衡，少数experts接近1，其他experts接近0）如何缓解？
    - MMoE在实践中还有什么tricks吗，expert和gate的选取，expert权重的计算？
    - MMoE缺点？PLE相比MMoE改进了什么？
- 在实践中会遇到增加targets的情况，如何热启动？哪种方法效果好？


### 序列模型

- DIN及相关模型介绍？DIEN相比DIN的改变？
- 使用了什么特征？
- DIN的target attention是怎么计算的？
- DIN和transformer对比



