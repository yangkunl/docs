**MoLo: Motion-augmented Long-short Contrastive Learning for Few-shot Action Recognition**(cvpr2023)

## Introduction

现有的FSAR方法大多局限在frame-level,导致由于缺失强制的全局上下文意识,不匹配的视频帧会造成干扰。

![image-20231214152456807](attachments\image-20231214152456807.png)

因此作者认为如果局部视频帧能够预测全局上下文，将有利于实现精确匹配，其次没有明确的探索帧间丰富的运动线索来进行匹配，从而导致性能不尽如人意。传统的动作识别方法使用光流或者将frame difference转化为额外的深度网络，但这会导致巨大的计算开销。因此develop motion-augmented long-short contrastive learning来获取全局的上下文信息和动态运动线索，使用一个long-short contrastive objective 来增强frame feature去预测一个视频的global 上下文信息，为了进行运动补偿，作者设计了一个motion autodecoder，通过重建像素运动来明确提取帧表示之间的运动特征。

## Method

![image-20231214160102848](attachments\image-20231214160102848.png)

$S = {s_1,s_2,\dots,s_N}$,其中$s_i \in \mathbb{R}^{T\times 3\times H\times W}$,其中$T$是视频的帧数,$q \in \mathbb{R}^{T\times 3\times H\times W}$,其目标是分类$q$,与其他的方法一样,$F_S = \{f_{s1},f_{s1},\dots,f_{sN}\}$这是支持集提取的特征,其中$f_{i} = \{f^1_i,f^2_i,\dots,f^T_i\},f^j_i\in\mathbb{R}^{C\times H_f \times W_f}$,其中$C$是通道数,与此同时作者设计了一个motion autodecoder 去重建the frame differences,如下图所示:
![image-20231214161515479](attachments\image-20231214161515479.png)

将经过Feature difference generator的数据与base head一样作为motion head输入,并且将结果合并起来。

- **Long-short contrastive objective**

  对于输入的$F_S$和$f_q$特征,首先使用一个空间上的全局平均池化,去掉空间维度,同时添加一个可学习的token,然后将其输入到tempoal Transformer上 得到$\tilde{f_i} = \{\tilde{f}^{token}_i,\tilde{f}^1_i,\dots,\tilde{f}^T_i\}$,计算过程如下:
  $$
  \tilde{f_i} = \mathrm{Tformer}([f^{token},\mathrm{GAP}(f^1_i,\dots,f^T_i)]+f_{pos})
  $$
  $f^{token}\in\mathbb{R}^C$,$f_{pos}\in\mathbb{R}^{(T+1)\times C},$这个可学习的token用于聚合视频的全局特征,loss如下:
  $$
  \begin{array}{c}
  \mathcal{L}_{L G}^{\text {base }}=-\log \frac{\sum_{i} \operatorname{sim}\left(\tilde{f}_{q}^{\text {token }}, \tilde{f}_{p}^{i}\right)}{\sum_{i} \operatorname{sim}\left(\tilde{f}_{q}^{\text {token }}, \tilde{f}_{p}^{i}\right)+\sum_{j \neq p} \sum_{i} \operatorname{sim}\left(\tilde{f}_{q}^{\text {token }}, \tilde{f}_{j}^{i}\right)} \\
  -\log \frac{\sum_{i} \operatorname{sim}\left(\tilde{f}_{q}^{i}, \tilde{f}_{p}^{\text {token }}\right)}{\sum_{i} \operatorname{sim}\left(\tilde{f}_{q}^{i}, \tilde{f}_{p}^{\text {token }}\right)+\sum_{j \neq q} \sum_{i} \operatorname{sim}\left(\tilde{f}_{j}^{i}, \tilde{f}_{p}^{\text {token }}\right)}
  \end{array}
  $$
  其中$\tilde{f}_q$是查询特征,$\tilde{f}_p$是与查询相同类别的样本(训练过程),$\tilde{f}_j$是其他样本,

- **Motion autodecoder**

  如上图,计算公式如下:
  $$
  f_{i}^{\prime 1}, \ldots, f_{i}^{\prime T-1}=\mathcal{F}\left(f_{i}^{1}, \ldots, f_{i}^{T}\right)
  $$
  其中$\mathcal{F}$是feature difference generator,$f_{i}^{\prime j}$是输出的motion feature,通过提取器我们得到了帧间的motion特征,使用类似MAE的自监督方法进行重建像素。

- **loss**
  $$
  \mathcal{L}=\mathcal{L}_{C E}+\lambda_{1}\left(\mathcal{L}_{L G}^{\text {base }}+\mathcal{L}_{L G}^{\text {motion }}\right)+\lambda_{2} \mathcal{L}_{\text {Recons }}
  $$
  其中$\mathcal{L}_{C E}$的得到源自与其他方法相同的帧间匹配计算距离。