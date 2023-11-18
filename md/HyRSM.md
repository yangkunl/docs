**Hybrid Relation Guided Set Matching for Few-shot Action Recognition**(CVPR2022)

## 1.当前问题

* 在不考虑整个任务的情况下学习单个特征，可能会丢失当前事件中最相关的信息；
*  这些对齐策略可能会在错误对齐的实例中失效。

提出方法：混合关系模块和集合匹配度量中视频内部和视频间的关系。

混合关系模块：充分利用在一个episode中的视频内和视频间的

> by fully exploiting associated relations within and cross videos in an episode  

集合匹配度量：将query和support videos之间的距离度量表述为集合匹配问题,设计了 Mean Hausdorff Metric用来提高对错位实例的适应能力。

> Bidirectional Mean Hausdorff Metric（BMHD）是一种用于改善对齐不准确实例鲁棒性的度量方法。
>
> 首先，让我们了解一下一般的Hausdorff距离。Hausdorff距离是用于衡量两个点集之间的相似性的一种度量方式，它考虑到两个点集中每个点到另一个集合的最远距离。具体而言，Hausdorff距离定义为两个点集之间的最大距离：H(A,B)=max⁡(sup⁡a∈Ainf⁡b∈Bd(a,b),sup⁡b∈Binf⁡a∈Ad(a,b))H(A,B)=max(supa∈Ainfb∈Bd(a,b),supb∈Binfa∈Ad(a,b))
>
> 其中，AA 和 BB 是两个点集，d(a,b)d(a,b) 表示点 aa 到点 bb 之间的距离。Hausdorff距离的计算对于不准确对齐的实例可能会很敏感，因为它只考虑最大距离，这可能导致对异常值敏感。
>
> Bidirectional Mean Hausdorff Metric 提出了一种改进的方法。它考虑到了两个点集之间的平均距离，并且是双向的，即从第一个集合到第二个集合和从第二个集合到第一个集合的平均距离。这种度量方法的目标是提高对不准确对齐的实例的鲁棒性，因为它通过考虑两个方向的平均距离，降低了对异常值的敏感性。
>
> 具体形式可能因文献或方法而异，但一般而言，Bidirectional Mean Hausdorff Metric的计算方式是通过分别计算两个方向上的平均Hausdorff距离，然后将它们结合起来以得到一个综合的度量。

![image-20231117221200956](attachments\image-20231117221200956.png)

## 2.方法

### pipeline

便于理解，接下来都使用Nway1shot首先使用一个embedding model 来提取特征，获得每一个视频的support feature， $F_s = {f_{s1},f_{s2},...,f_{sN}}$, 以及每一个$f_{si} = \{f_i^1,f_i^2,...,f_i^T\}$,以及$f_{q} = \{f_q^1,f_q^2,...,f_q^T\}$。然后将$F_q$和$f_q$输入到 hybrid relation module 去学习 task-specific features，得到$\hat{F_s} , \hat{f_q}$,再将$\hat{F_s} , \hat{f_q}$进行匹配获得匹配的分数。

### Hybrid relation module

整个混合关系模块如下

$\tilde{f}_{i}=\mathcal{H}\left(f_{i}, \mathcal{G}\right) ; f_{i} \in\left[F_{s}, f_{q}\right], \mathcal{G}=\left[F_{s}, f_{q}\right]$

通过聚合跨视频特征，来改进特征$f_i$，使得改进后的特征更具区分度（more discriminative），为了提升效率将$\mathcal{H}$分为两部分，包括intra-relation部分（$\mathcal{H}_a$）和inter-relation部分（$\mathcal{H}_e$）。

- intra-relation function 的目的是增强一个视频内的结构化pattern，通过捕捉长时间的依赖性（long-range tempoal dependencies）。可以将其表达为$f^a_i = \mathcal{H_a}(f_i),f^a_i \in R^{T\times C}$,这个intra-relation有多种实现方式，包括 muti-head self-attention（MSA）， Transformer， Bi-LSTM， Bi-GRU。
- inter-relation function是基于intra-relation function得到的特征，从语义上增强不同视频的交叉特征：$f_{i}^{e}=\mathcal{H}_{i}^{e}\left(f_{i}^{a}, \mathcal{G}^{a}\right)=\sum_{j}^{\left|\mathcal{G}^{a}\right|}\left(\kappa\left(\psi\left(f_{i}^{a}\right), \psi\left(f_{j}^{a}\right)\right) * \psi\left(f_{j}^{a}\right)\right)$，其中$\mathcal{G^a}=[F^a_s, f^a_q]$,$\psi(\cdot)$是全局平均池化，$\kappa(f^a_i,f^a_j)$是一个可学习的函数，来计算两个视频间的语义相关性。该作用的意义是两个视频的相关性程度高，说明可以借用更多的信息。相同的如果相关性低，那么无关性将会被抑制。

得到相关性分数后通过Expend-Concatenate-Convolution operation来聚合信息

$y_i = \mathcal{P}((\tilde{f}_{s_i},\tilde{f}_q)|\mathcal{H}(f_{s_i},\mathcal{G}),\mathcal{H}(f_q,\mathcal{G}));\mathcal{G}=[F_s,f_q]$。其中$\tilde{f_{s_i}}$一个是上述两个特征concat后的特征。

### Set matching metric

鉴于关系增强特征$ ̃\tilde{Fs} 和 ̃\tilde{fq}$，提出了一种新的度量方法，以实现高效灵活的匹配。在该指标中，我们将每个视频视为一组 T 帧，并将视频间的距离测量重新表述为一个**集合匹配**问题，该问题对复杂的实例具有鲁棒性，无论它们是否对齐。具体来说，我们通过修改典型的集合匹配方法（Hausdorff distance）来实现这一目标。标准的豪斯多夫距离 D 可以表述为

![image-20231118112252979](attachments\image-20231118112252979.png)

设计的Bi-MHM如下：

![image-20231118112345409](attachments\image-20231118112345409.png)

## 3.实验设置

使用使用imageNet预训练的ResNet-50作为backbone，每一个视频均匀采样8帧，使用的data augmentation方法是随机裁剪和color jitter，使用10000个task做test

![image-20231118113704861](attachments\image-20231118113704861.png)
