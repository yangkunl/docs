**Spatio-temporal Relation Modeling for Few-shot Action Recognition**(CVPR2022)

## 1.baseline(TRX)的问题

使用的是TRX作为baseline,TRX利用cross-transformer 将query video和support video中以**不同速度和瞬间发生的动作**进行了匹配。首先，对于查询视频中的每个子序列，都会通过聚合某个动作类别的支持视频中所有可能的子序列，计算出查询的(对于某一个子序列)特定类别原型。相当于TRX使用了hand-crafted representations

limitaition:

- TRX是对query 和 support action sub-sequence,但是对于spatial context variation( relevant object appearance),tempoal context(视频背景发生变化),具有局限性。


- TRX使用了多个cross-Transformer。因此，这就导致模型的灵活性较差，除了需要对不同的 Ω 组合进行人工模型搜索以找到最佳 Ω∗ 之外，还需要为不同的心值建立专门的分支。接下来，我们将介绍旨在综合处理上述问题的拟议方法。


为了解决以上问题同时增强class-specific的discriminability

本文在patch-level 和 frame-level上进行了特征的聚合。

## pipeline

 ![image-20231202135112590](attachments\image-20231202135112590.png)

L个视频通过feature extractor 得到尺度为$P\times P \times D$的特征,将这个特征flatten 得到$x_i \in \mathrm{R}^{P^2\times D},i\in [1,L]$,作为spatio-tempoal enrichment module 的输入,patch-level enrichment(PLE),通过观察每帧中的空间背景来局部增强patch features,得到每一帧增强后的特征 $f_i \in \mathrm{R}^{P^2 \times D}$,然后再在空间维度上做了一个平均池化得到D维的每帧特征,将L帧concat起来得到$\mathbf{H} \in \mathrm{R}^{L \times D}$,然后将得到的$\mathbf{H}$输入到frame-level enrichment(FLE)模块,得到farme-level增强后的特征$\mathbf{E} \in \mathrm{R}^{L \times D}$,将得到的$\mathbf{E}$输入到 temporal realtionship modeling(TRM) module,这个模块**通过sub-sequences来匹配**query video 和 support video(类似于TRX)。此外，通过引入查询类相似性分类器对中间表征 $\mathbf{H}$ 进行分类，可以在不同阶段加强对相应类级信息的学习，有助于进一步提高整体特征的可辨别性。通过这两个匹配(分类)模块可以得到两个loss,$\mathcal{L}_{TM},\mathcal{L}_{QC}$。

## 方法

- **Spatio-temporal Enrichment**

  引入一个时空增强模块，该模块致力于增强 (i) 单个帧中局部斑块的空间特征，以及 (ii) 在一个video内的跨帧的时序特征

  - **Enriching Local Patch Features**

    ![image-20231202143652022](attachments\image-20231202143652022.png)

    如上图所示,使用了自注意力机制。$x_i \in \mathrm{R}^{P^2\times D},i\in [1,L]$,作为输入模型的特征,使用权重$\mathbf{W_1},\mathbf{W_2},\mathbf{W_3} \in \mathrm{R}^{D\times D}$,得到query-key-value三元组如下:
    $$
    x^q_i = x_i\mathbf{W}_1, x^k_i = x_i\mathbf{W}_2, x^v_i = x_i\mathbf{W}_3,
    $$
    输入自注意力成的序列$p\in [1,P^2]$时,通过以下公式得到query 和 key 之间的patch 序列的注意力分数,因此得到'token-mixed' $\alpha _i$,计算公式如下:
    $$
    \alpha_i = \eta(\frac{x^q_i {x^k_i}^T}{\sqrt D})x^v_i + x_i
    $$
    其中$\eta$是softmax函数使得,注意力分数的权重值为1,然后再通过一个sub-network $\psi(\cdot)$,refine 得到的$\alpha_i$特征,最终得到patch enriched 后的特征$\mathbf{f}_i \in \mathrm{R}^{P^2 \times D}$,公式如下:
    $$
    \mathbf{f}_i = \psi(\alpha_i) + \alpha_i
    $$

  - **Enriching Global Frame Features**

    ![image-20231202155359541](attachments\image-20231202155359541.png)

    使用一个由 MLP mixer层组成的帧级增强（FLE）子模块，对视频中各帧的时间背景进行全局增强。

    自我注意是基于取样依赖性（特定输入）混合，以标记之间的成对相似性为指导，而 MLPmixers 中的标记混合则是通过与输入无关的持久关系记忆来同化整个全局感受野。

    首先将得到的增强特征$\mathbf{f}_i \in \mathrm{R}^{P^2 \times D}$,做一个平均池化得到$\mathbf{f}_i \in \mathrm{R}^{D}$,然后在将这些特征按帧数维度concat起来,得到$$\mathbf{H}=[h_1;\cdots;h_L] \in \mathrm{R}^{L \times D}$$,然后在将其输入到两层的MLP层,其公式如下:
    $$
    \mathbf{H}_{*}=\sigma\left(\mathbf{H}^{\top} \mathbf{W}_{t_{1}}\right) \mathbf{W}_{t_{2}}+\mathbf{H}^{\top}, \\
    \mathbf{E}=\sigma\left(\mathbf{H}_{*}^{\top} \mathbf{W}_{r_{1}}\right) \mathbf{W}_{r_{2}}+\mathbf{H}_{*}^{\top},
    $$
    其中的$\mathbf{W_{t1}},\mathbf{W_{t2}}\in \mathrm{R}^{L\times L},\mathbf{W_{r1}},\mathbf{W_{r2}} \in \mathrm{R}^{D\times D}$,其都为可学习的参数,$\sigma$是激活函数relu,$\mathbf{E}$的一个帧级特征为$\mathbf{e_i}$for i in [1,L]。

  - **temporal relationship modeling (TRM) module**

    将上述分别得到的query和support特征输入进入TRM模块,该TRM就是TRX在$\Omega = \{2\}$的时候的模块,需要通过cross-Transformer。

- **Query-class Similarity**

  对于得到的$\mathbf{h}_i$特征,获得了其tuple reoresentation $\mathbf{l}_t= [\mathbf{h}_{t_1};\cdots ; \mathbf{h}_{t_w}] \in  \mathrm{R}^{w D} $(跟TRX一样的帧组合),对于$t= (t_1,\cdots,t_w) \in \Pi_w$(这是帧组合)在一个视频中,然后使用$\mathbf{W}_{cls} \in  \mathrm{R}^{w D \times D}$,得到$\mathbf{z_t} = \sigma({\mathbf{W}_{cls}}^T\mathbf{l}_{t})$,然后$\mathbf{z_t^Q}$代表query video的特征,
  $$
  M(Q, c)=\sum_{\omega \in \Omega} \frac{1}{\left|\Pi_{\omega}\right|} \sum_{t \in \Pi_{\omega}} \max _{j} \phi\left(\mathbf{z}_{t}^{Q}, \mathbf{z}_{j}^{c}\right)
  $$
  
