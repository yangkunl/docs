**Boosting Few-shot Action Recognition with Graph-guided Hybrid Matching**(ICCV 2023)

## 问题:

忽视了类原型构建和匹配的价值,导致在每个任务中识别相似性能不尽如人意,其通过图神经网络的引导来学习面向任务的特征,明确优化类内和类间的特征相关性,设计了一种混合匹配策略,将帧级匹配和元组级匹配结合起来,对具有多元风格的视频进行分类。类似的动作场景在很大程度上取决于时间顺序。

提出了一个learnable dense tempoal modeling module 来 consolidate representation foundation。其包括tempoal patch 和 tempoal channel relation module。

其创新点如下：

- 在类别原型的构建过程中使用GNN来指导task-oriented features 学习,明确了视频特征类内和类间的关系
- 提出了一个hybird class prototype matching 对于frame 和 tuple -level matching。从而有效应对多种风格的视频任务。
- 设计了一个可学习的tempoal modeling module consisting tempoal patchs and tempoal channel relation

## 方法

### pipeline

跟TSN一样的方法,每个视频选8帧,为了简单描述使用5-way 1-shot 进行展示,query set $Q = \{q_1,q_2, \cdots ,q_T\}$,以及support set$S^n = \{s^n_1,s^n_2,\cdots,s^n_T\}(S^n \in \mathcal{S} = \{S^1,S^2,\cdots,S^5\})$,$n$为第几个类别,在每一次任务中(each eposode)通过特征提取模块(feature extractor)获得query set和support set的特征,分别为$\mathbf{F}_{\mathcal{Q}},\mathbf{F}_{S^n}(\mathbf{F}_{S^n}\in\mathbf{F}_{\mathcal{S}})$,然后将$\mathbf{F}_{\mathcal{Q}}$和$\mathbf{F}_{\mathcal{S}}$输入到作者所提出的learnable dense tempoal modeling module。得到增强的时序特征$\widetilde{\mathbf{F}_{\mathcal{Q}}}$ 和$\widetilde{\mathbf{F}_{\mathcal{S}}}$。

然后对于$\widetilde{\mathbf{F}_{\mathcal{Q}}}$ 和$\widetilde{\mathbf{F}_{\mathcal{S}}}$在tempoal 维度使用平均池化得到池化后的特征$\widetilde{\mathbf{F}^{avg}_{\mathcal{Q}}}$ 和$\widetilde{\mathbf{F}^{avg}_{\mathcal{S}}}$,作为关系节点特征为之后的graph network做准备,然后将这个关系节点特征和初始边特征(init edge feature)一起纳入图网络进行关系传播(relation propagation)。更新后的边特征和增强的时序特征通过graph metric 可以生成任务导向特征(task-oriented features)$\mathbf{F}^{task}_{\mathcal{Q}}$和$\mathbf{F}^{task}_{\mathcal{S}}$,同时还能获得loss $\mathcal{L}_{graph}$,然后得到的任务导向特征$\mathbf{F}^{task}_{\mathcal{Q}}$和$\mathbf{F}^{task}_{\mathcal{S}}$会被输入到hybird class prototype matching metric 去获得对于query的类别预测$\hat{y}_Q$和loss $\mathcal{L}_{match}$。

### Learnable Dense Temporal Modeling Module（LDTM）

LDTM包含两个模块,分别是tempoal patch relation modeling block 和 tempoal channel relation modeling block。结合这两个模块,可在spatial 和 channel 进行dense tempoal modeling,如下图所示:

![image-20231130125642312](attachments\image-20231130125642312.png)

- **Patch Tempoal Relation Modeling(PTRM)**

  从feature extractor 得到的特征$\mathbf{F}\in \mathrm{R}^{N\times T \times C\times H \times W}$,首先对其进行reshape得到$\mathbf{F_{seq1}}\in \mathrm{R}^{N\times HW \times C\times T}$,然后将$\mathbf{F_{seq1}}$输入到tempoal MLP 中得到 hidden tempoal feature $\mathbf{H}_T$:
  $$
  \mathbf{H}_T= relu(\mathbf{W}_{t1}\mathbf{F}_{seq1})\mathbf{W}_{t2} + \mathbf{F}_{seq1}
  $$
  其中$\mathbf{W}_{t1}$和$\mathbf{W}_{t2}\in \mathrm{R}^{T\times T}$,是不同视频帧时间信息交互的可学习的帧,然后将得到的$\mathbf{H}_T$插入到原来的特征$\mathbf{F_{seq1}}$中,使得单个视频帧能够得到全部视频帧的语义信息,得到的tempoal patch relation modeling feature $\mathbf{F_{tp}}$ 如下:
  $$
  \mathbf{F}_{tp}[:, n,:,:]=\left\{\begin{array}{c}
  \mathbf{F}{_{seq 1}[:, n,:,:] } \quad if {{ n \% }}  gap =0 \\
  \mathbf{H}_{T}[:, n,:,:]  \quad if {{ n \% }}  gap  \neq 0
  \end{array}\right.
  $$
  其中$n$是patch index,$gap$是一个正数,其能控制the frequency of the patch shift,在可学习的patch shift operation (这个gap是一个可以学习的数吗)后,将$\mathbf{F_{tp}}$reshpe为$\mathbf{F^{\star}_{tp}}\in \mathrm{R}^{NT\times HW \times C}$,然后做空间自注意力:

  <img src="attachments\image-20231130132557789.png" alt="image-20231130132557789" style="zoom: 67%;" />

  其中的original Features是$ \mathbf{F}_{seq1}$,这种方式可以稀疏的收集时间信息,但是会牺牲每个帧内的空间信息为了缓解这个问题,作者将$\mathbf{F}\in \mathrm{R}^{N\times T \times C\times H \times W}$原始输入的特征直接reshape为$\mathbf{F^{\star}}\in \mathrm{R}^{NT\times HW \times C}$然后输入空间自注意模块,最后将这两块结果加起来,公式如下$\gamma$是一个超参数:
  $$
  \mathbf{F_{tp}} = \gamma  SA_{spa}(\mathbf{F^{\star}_{tp}}) + (1-\gamma)SA_{spa}(\mathbf{F^{\star}})
  $$

- **Channel Temporal Relation Modeling(CTRM)**

  从feature extractor 得到的特征$\mathbf{F}\in \mathrm{R}^{N\times T \times C\times H \times W}$,首先对其进行reshape得到$\mathbf{F_{seq2}}\in \mathrm{R}^{NHW \times C\times T}$,然后将其输入可学习的channel shift operation,得到特征$\mathbf{F}_{tc}$,这个channel shift operation是一个1D channel-wise temporal convolution,公式如下:
  $$
  \mathbf{F}_{t c}^{t, c}=\sum_{i} \mathbf{K}_{c, i} \mathbf{F}_{s e q 2}^{c, t+i}
  $$
  然后在通过时空自注意力层得到最终的$\mathbf{F}_{tc}$。

然后将得到的$\mathbf{F}_{tp}$和$\mathbf{F}_{tc}$加权求和得到最终的$\widetilde{\mathbf{F}}$。
$$
\widetilde{\mathbf{F}} = \beta \mathbf{F}_{tp} + (1-\beta)\mathbf{F}_{tc}
$$
其中$\beta,\gamma \in [0,1]$。总之，PTRM 聚合了部分斑块的时间信息，而 CTRM 则学习通道的时间偏移。因此， LDTM 可以在空间和信道维度上以密集和可学习的方式实现充分的时间关系建模。

### Graph-guided Prototype Construction(GgPC)

之前的许多工作利用图网络优化簇内相似性和簇间不相似性，并将图像分类问题**转化为节点或边缘分类问题**。与此不同的是，直接将视频特征（通常是经过时序池操作后的特征）输入图网络可能会导致时序信息的丢失，从而导致不理想的结果。因此，作者仅使用图网络作为优化特征类内和类间相关性的指导。具体结构如下图:

![image-20231130135930081](attachments\image-20231130135930081.png)

以下讨论的均为 $N_S$-way 1-shot problem,考虑query set$\mathcal{Q}$,包含$N_{\mathcal{Q}}$ 个videos,这个过程可以分为两个部分: Graph neural network propagation 和 task-oriented obtaining。

对于 GNN propagation,对于上一个环节得到的$\widetilde{\mathbf{F}}$,首先在时间维度上做一个平均池化,得到$\widetilde{\mathbf{F}^{avg}}$,然后将这个平均池化特征$\widetilde{\mathbf{F}^{avg}}$,作为图神经网络的初始节点特征,边特征$\mathbf{A}$表征了两个节点之间的关系(即类内和类间关系的强度，其初始化取决于标签),propagation过程包括节点聚合和边缘聚合。在完成propagation之后使用`Seclect`操作从最后一层更新的边缘特征中提取相似度分数。`Seclect`是指从输出的整个边缘特征中选择与每个查询视频特征相关的边缘特征，并进一步形成总共$N_Q$ 个新的边缘特征。

task-oriented obtaining ,详见算法1:

![image-20231130143313162](attachments\image-20231130143313162.png)

其中的$f_{FNN}$为前馈神经网络,$f_{emb}$和$f_{fuse}$是MLP,以及$\bigotimes$是矩阵乘法,特别对于K-shot任务在构建节点特征时，在特征维度上对同类别支持视频的特征进行均值池化处理，而其他方面则与单次拍摄任务保持一致。其中的`Select`操作如下:

![image-20231130143923553](attachments\image-20231130143923553.png)

### Hybrid Prototype Matching Strategy (HPM)

设计了一种混合原型匹配策略,其可以combine frame-level 和 tuple-level,基于bidirectional Hausdorff Distance,从上一个环节得到task-oriented features $\mathbf{F}^{task}_{\mathcal{Q}}$和$\mathbf{F}^{task}_{\mathcal{S}}$,第$m$-th个support video对于第$k$ 类,将其表示为$s^k_m\in \mathrm{R}^{T\times C}$,$q_p \in \mathrm{R}^{T \times C}$,然后使用Mean Hausdorff metric:
$$
\mathcal{D}_{ frame }=  \frac{1}{T}[\sum_{\mathbf{s}_{m, i}^{k} \in \mathbf{s}_{m}^{k}}(\min _{\mathbf{q}_{p, j} \in \mathbf{q}_{p}}\|\mathbf{s}_{m, i}^{k}-\mathbf{q}_{p, j}|) 
 +\sum_{\mathbf{q}_{p, j} \in \mathbf{q}_{p}}(\min _{\mathbf{s}_{m, i}^{k} \in \mathbf{s}_{m}^{k}}\|\mathbf{q}_{p, j}-\mathbf{s}_{m, i}^{k}\|)]
$$
对于tuple-level 的特征:
$$
\begin{aligned}
& \mathbf{t s}_{m, i}^k=\left[\mathbf{s}_{m, i_1}^k+\mathbf{P E}\left(i_1\right), \mathbf{s}_{m, i_2}^k+\mathbf{P E}\left(i_2\right)\right] 1 \leqslant i_1 \leqslant i_2 \leqslant T \\
& \mathbf{t} \mathbf{q}_{p, j}=\left[\mathbf{q}_{p, j_1}+\mathbf{P E}\left(j_1\right), \mathbf{q}_{p, j_2}+\mathbf{P E}\left(j_2\right)\right] 1 \leqslant j_1 \leqslant j_2 \leqslant T
\end{aligned}
$$
其中$L = \frac{1}{2}(T-1)T$,跟TRX一样,
$$
\begin{aligned}
\mathcal{D}_{\text {tuple }}= & \frac{1}{L} \sum_{\mathbf{t s}_{m, i}^k \in \mathbf{t} \mathbf{s}_m^k}\left(\min _{\mathbf{t} \mathbf{q}_{p, j} \in \mathbf{t} \mathbf{q}_p}\left\|\mathbf{t} \mathbf{s}_{m, i}^k-\mathbf{t} \mathbf{q}_{p, j}\right\|\right) \\
& \left.+\sum_{\mathbf{t} \mathbf{q}_{p, j} \in \mathbf{t} \mathbf{q}_p}\left(\min _{\mathbf{t} \mathbf{s}_{m, i}^k \in \mathbf{t s}_m^k}\left\|\mathbf{t} \mathbf{q}_{p, j}-\mathbf{t} \mathbf{s}_{m, i}^k\right\|\right)\right]
\end{aligned}
$$
得到最终的:
$$
\mathcal{D}_{\text {hybrid }} = \alpha \mathcal{D}_{\text {tuple }} +(1-\alpha)\mathcal{D}_{\text {frame }}
$$

## 实验

![image-20231130151238078](attachments\image-20231130151238078.png)



