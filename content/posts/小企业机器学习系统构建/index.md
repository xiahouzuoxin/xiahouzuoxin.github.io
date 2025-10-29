+++
title = '小企业机器学习模型训练实践——从单机Pandas到Ray的迭代'
date = 2025-10-20T10:55:30+08:00
draft = false
math = false
tags = ['ML']
+++

在大公司里通常有专门的团队负责机器学习平台与基础设施（例如在阿里，数据预处理常在 ODPS 完成，训练跑在 PAI/XDL 上）。但在小公司里，特别是那些没有自建完善数据流水线的团队，做模型就没有那么顺利：需要自己摸索如何加速特征生成、在单机上处理与训练上亿样本而不爆内存，以及如何在需要时平滑迁移到可维护的分布式方案。本文基于在 Jerry.ai 的实践，记录了从单机到分布式、从 Pandas/ClickHouse 到 Polars/Ray 的演进思路与经验总结。

## 背景

我们的原始数据以应用产生的表格为主，且常包含大量 json/xml 等复杂字段，提取与清洗这些字段需要较复杂的 SQL/ETL 逻辑。典型训练场景包括推荐、LTV 预估、价格预估及若干分类任务等。

在架构上，公司采用部署在Kubernetes上的单机大内存 ClickHouse 做数据分析报表，数据由应用先落到 RDS，再同步到 ClickHouse，因此绝大多数离线特征构建是从 ClickHouse 出发。早期的机器学习流程通常是：在 ClickHouse 做查询、用 Pandas 做预处理、在 Jupyter/单机上训练模型。

与大公司相比，中小企业（如样本规模 < 1 亿）面临的特点和挑战主要有：
1. 初期分布式不是刚需——单机能解决大部分问题，工程成本和维护成本更敏感；
2. 随着数据增长，预处理首先成为瓶颈——一方面是耗时，另一方面是内存不足导致容易 OOM；
3. 当数据继续增长到千万/亿级，单机难以支撑快速迭代与模型实验，这时需要引入更高效、易维护的分布式数据处理与训练框架。

## 从单机到Ray的迭代

![image1](assets/image1.png)
![image2](assets/image2.png)

上图展示了随模型与数据增长的方案迭代路径，下图是我们最终的训练架构示意。

### 第一阶段

当样本规模增长到百万级，pandas 在预处理上开始吃力，主要表现：
1. apply/map等操作是按行序列执行，没法高效利用单机多核，处理耗时长
2. 没法流式读取数据，rawdata json/xml复杂的情况下内存压力大

因此，基于内部主要Data ETL都是Clickhouse，我们把尽可能多的特征预处理也下沉到 ClickHouse，利用Clickhouse的并行计算能力，从而显著提升样本生成效率并降低单机内存压力。

### 第二阶段

随着数据规模继续增长，单节点 ClickHouse 出现的局限开始暴露：
- 单机版Clickhouse容易OOM，且由于内部Clickhouse底层使用的JuseFS导致系统不稳定经常出现“File System Error”。所以经常中途跑失败，跑不出来的情况；
- Clickhouse主要还是以离线SQL为主，复杂的特征变换，在线实时打分过程，需要用Redis/RDS再实现一边。这一定程度上导致在离线的不一致性；

为此我们引入了 Polars，**我强烈推荐使用[Polars](https://pola.rs/)**，单机数据处理的神器：
- Polars通过LazyDataFrame流式处理，可以处理比内存大几倍的数据集。我们试着从 eager DataFrame 迭代到 LazyDataFrame，完全避免了Clickhouse的OOM问题，且python代码可复用；
- Polars支持多核并行计算，能显著提升单机数据处理效率，我所在的场景下对比 pandas 提升显著——几十倍，有些公开测试性能甚至优于 ClickHouse，因而很适合中小团队在没有集群基建的情况下快速迭代；

在此阶段，我还特地写了个[Polars做Feature Transformer的通用代码](https://github.com/xiahouzuoxin/torchctr/blob/main/torchctr/transformer.py)，通过对多核的充分利用，相比pandas提升接近100倍。

> 思考：与Dask的对比，Polars的优劣势？
> 我的经验：Dask的生态更成熟，且能无缝对接pandas代码，但Dask的调度开销较大，且在中小数据集上性能不如Polars。另一个更重要的原因是Dask是按行式分块处理的，如果我要计算某个特征的统计值就会比较麻烦，而Polars是列计算效率非常高。

在模型训练方面，进入千万级样本后我们开始使用DNN，下面的选项各有优劣，需要根据场景和成本选择：
1. 单卡多GPU：训练速度快，但对稀疏高维 embedding 的利用率偏低。价格上来说，一般同样的训练，比CPU会更贵
2. 多 CPU：更能利用稀疏/高维特征的特性，整体利用率高但训练速度相对较慢

> 思考：能不能把DNN的训练Batch Size加大，直到填满GPU的显存，从而提升GPU利用率同时降低训练时间？
> 我实验的结论：大规模稀疏推荐场景比如CTR模型中，不行。分析原因可能是：（1）这会导致模型的iterations不够，严重影响模型的性能；（2）大Batch Size下，梯度方向太过稳定，模型快速的收敛到一个陡峭的局部最优点，影响模型的泛化能力。

在这一阶段，使用 PyTorch+多CPU/GPU 能满足大多数训练需求，多卡训练也能通过 `torch.distributed` 或者 `accelerate` 等工具实现。

### 第三阶段

当数据规模上升到亿级或更高时，单机的迭代效率成为不可接受的瓶颈：

- 训练：我们把训练从单机 PyTorch 迁移到 RayTrain。选择 Ray 的原因是
    - 单机也能跑，即使不作为多机使用，也能通过Ray很好的利用单机多核和多GPU资源；
    - 其良好的扩展性，不用做任何改动单机代码可以平滑地迁移到集群，同时Ray对xgboost/lightgmb/pytorch等主流框架均有良好支持；
- 特征变换：逐步迁移到 RayData + Polars的模式，以支持分布式、高性能的数据处理，同时保持训练与部署的一致性；
- 原始数据ETL：主要是对json/xml等原始数据的ETL，随着数据规模的增加，也逐步从Clickhouse迁移到 Snowflake（处于长远考虑，避免对Spark维护，直接用Snowflake而非自建Spark），从而在这个阶段完全脱离对Clickhouse的依赖（Clickhouse本身也不适合大规模数据的预处理，更适合做报表类业务）。

> Ray怎么部署的？
> 我们是采用的RayOnK8S在K8S上部署的，和公司整体的K8S架构结合在一起，方便管理与监控。

总结：我们先用更高效的单机工具（ClickHouse / Polars）实现单机上的效率提升，当样本与计算需求确实超出单机能力时，再用 Ray（Ray Data + Ray Train）实现平滑放大与可维护的分布式训练。

## 为什么没有选择其他方案

市面上有不少分布式数据处理与训练框架，例如 Spark、Hadoop 等。我们最终选择 Ray 主要基于以下考虑：

- 公司内部的整体数据架构一开始都是基于clickhouse，与HDFS/Hadoop的生态有一定差异；
- Ray 提供了较为统一的分布式计算框架，既支持数据处理（Ray Data），也支持模型训练（Ray Train），且对多种主流机器学习框架均有良好支持；
- Ray 的学习曲线很低，部署比较容易，且单机到集群的迁移成本较低，适合中小团队快速上手与迭代。
