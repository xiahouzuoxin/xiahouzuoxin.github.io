+++
title = 'Twitter推荐系统概览'
date = 2024-10-11T10:59:01+08:00
draft = false
math = false
tags = ['Recommendation']
+++

Twitter去年开源推荐系统代码到现在一年多了，一直没时间去看一看，今天想起来瞄了会，本文简单记录下。

开源：
- 系统代码：https://github.com/twitter/the-algorithm
- 算法代码：https://github.com/twitter/the-algorithm-ml
- 博客：https://blog.twitter.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm

我先读的博客，基本了解其推荐系统的概况，总体来说，算法相比国内阿里电商首猜/小红书feed流的推荐是更初级很多。

![](./image/image.png)

整个系统基本就是召回 + 精排，可以理解没有粗排，在In-Network召回时也就是RealGraph那里会有一个LR模型类似粗排的功能。

# 数据：

因为Twitter是社交app，所以：
1. 主要是用户交互数据，包括用户-用户Follow的社交图，Tweet engagement通常指用户和内容的的交互，比如点赞、分享、关注等等；
2. 用户的profiles数据：语言、地理位置等

# 召回

召回是本次开源里面的重头部分。Twitter召回包括In-Network的召回和Out-of-Network的召回。In-Network指从用户的Follow用户的内容中召回，Out-of-Network指从未Follow的其他用户中召回，两者占比大概各50%。召回最终返回1500个候选集给到精排（Twitter叫Heavy Rank）打分排序。

## In-Network召回，占比50%

从Follow的用户中召回所有的内容，然后通过一个LR模型对这些内容进行排序。这部分叫[Real Graph](https://www.ueo-workshop.com/wp-content/uploads/2014/04/sig-alternate.pdf)，看了下这个是2014的工作了，博客中也提到要对LR模型重构。

![](image/realgraph.png)

RealGraph核心逻辑就是构建User间的有向图，然后通过LR学习Edges的权重，权重高的用户的内容多推荐一些。

## Out-of-Network召回，占比50%

包括直接从U2U2I召回、Sparse Embedding召回、Dense Embedding召回。

**U2U2I召回**：即从关注的用户喜欢的内容中召回，占比15%。有个问题，这里如果U2U2I内容数量超过15%怎么截断呢？

**Sparse Embedding召回**：也是U2U2I，只不过是通过Sparse Embedding找相似用户。首先通过[SimClusters](https://github.com/twitter/the-algorithm/blob/main/src/scala/com/twitter/simclusters_v2/README.md)社区发现算法，找到用户的社区representation embedding，然后通过ANN找到相似用户。

![](image/simcluster.png)

**Dense Embedding召回**：相比用社区向量表征用户，Dense Embedding采用了[TwHIN](https://github.com/twitter/the-algorithm-ml/blob/main/projects/twhin/README.md)模型学习用户的向量表征，思路和其他的U2U召回算法类似，和用户有engagement的用户作为正样本，然后batch内采样负样本，通过`binary_cross_entropy_with_logits` loss函数train一个Model。模型离线Train完之后，离线存储user vector，然后线上通过ANN索引查找最相似的用户。

召回里面的算法其实都比较中规中矩，没有很fancy的模型。很多工作在推荐工程上，比如设计GraphJet去维护user-tweet-entity-graph (UTEG)即用户的交互图，比如follow-recommendation-service (FRS)推荐服务的工程实现等等。

# 精排

感觉没啥可说的，召回的1500个候选送到大概48M参数的MaskNet中打分。

# 重排

一些基于规则的策略为主：

- 过滤：过滤看过的内容，用户设置block的内容、色情等；个人感觉这个应该放在召回前？
- 打散：
    - 同一作者的内容打散排布
    - 不同召回渠道的内容分布相对均匀
- ...

# 混排

混排广告、其他UI内容（推荐的Follow用户等）。

好了，感觉就是这些，有时间再细看下代码。


