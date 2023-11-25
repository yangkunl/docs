**M$^3$Net: Multi-view Encoding, Matching, and Fusion for Few-shot Fine-grained Action Recognition**(ACM MM2023)

## 存在的问题

- 无法捕捉细微的动作细节
- 无法从表现出高类内差异和类间相似性的有限数据中学习

需要的信息Discriminative Embedding:

- Discriminative Embedding:基于large intra-class variance and subtle inter-class difference ,捕捉subtle spatial semantics 和 complicated tempoal dynamics
- 需要对丰富的上下问细节进行编码

Robust Matching:

- 由于fine-grained涉及到不同的速度,顺序,和偏移的子动作因此需要更robust matching.理想情况下,应该从多个维度采用更加稳健的匹配函数,来解决多尺度时空变化引起的不对齐问题

Good Generaliztion

- 获得良好的泛化性能

$M^3Net$由多视图编码,多视图匹配和多视图融合组成。在多视图编码中我们认为

## 问题设置

将数据集分为两个$D_{base} = \{(x_i,y_i)|y_i \in C_{base}\}$和$D_{novel} = \{(x_i,y_i)|y_i \in C_{novel}\}$

而且$D_{base}$和$D_{novel}$的类别是不想交的,即$C_{base}\cap C_{novel}\in \oslash$,然后从$D_{base}$训练一个模型或者优化器,随后从$D_{novel}$中采样support set 和 

query set

## Pipeline

提出了一个通过多任务协作学习模式,可实现识别 fine-grained 数据的 基于 匹配的few-shot learning框架。其通过多视图编码,多视图匹配和多视图融合(multi-view encoding , multi-view matching 和 multi-view fusion)实现互补的embedding learning,相似度匹配和度量决策。

**multi-view encoding :**

提出可 intra-frame context encoding (IFCE)module。使用了**patch-level**级别的语义信息去补齐instance-level级别的语义信息。另外还使用了intra-video context encoding(IVCE)module 作为捕捉运动的特定时空背景。此外还包括了 intra-episode context encoding(IECE) module,用于捕捉一个episode中不同视频间的特定任务线索,减轻inter-class variance

**Multi-view matching:**

分为三块,特定实例匹配,特定类别匹配,和特定任务匹配。为了处理细粒度动作的多尺度时空变化，我们首先提出了针对帧级丰富特征的特定实例匹配函数。对于视频级丰富特征和任务级丰富特征，我们提出的类别特定匹配函数和任务特定匹配函数充分利用了有限的可用样本，从而**促进了灵活而稳健的视频匹配**。

**Multi-view fusion:**

为了鼓励泛化嵌入并提高多任务学习框架中的决策能力，我们整合了之前多视图匹配过程中呈现的多个匹配分支的不同损失和预测。所提出的多视图融合程序扩大了匹配的多样性，提高了嵌入的泛化程度，最终有利于 FS-FG 动作识别。

![image-20231124172616546](attachments\image-20231124172616546.png)

## 方法

- **Multi-view Encoding**

  **Intra-frame Context Encoding:IFCE** (帧内)

  旨在通过在视频的每一帧中利用基于空间的patch交互来强调特定实例的语义。从本质上讲，IFCE 模块能在特征转换过程中有效管理与类别相关的视觉线索的影响，从而实现后续的特定实例匹配。例如$f_i \in R^{h \times w \times d}$表示了一个视频第$i$帧,其中共有$h\times w$个patches ,每一个patch有一个$d$维的embedding 向量。使用**自适应池化**操作将$f_i$转化为$n\times n (n \ll h/w)$,从而有效限制了点积操作在整个空间维度上所消耗的计算量。

  ![image-20231124161753289](attachments\image-20231124161753289.png)

  得到每一帧的patches序列$Z_i= ([f^1_i,f_i^2,\cdots,f^{n\times n}_i]+P) \in \mathrm{R}^{n^2\times d}$,$P$是一个可学习的位置编码 $P\in \mathrm{R}^{n^2 \times d}$。然后输入的$Z_i$通过线性层$W_Q,W_K,W_V$被映射为$Z_i^q,Z_i^k,Z_i^v$。然后进行点积过一个Softmax再通过残差连接得到$\hat{Z_i}$,具体公式如下:
  $$
  \hat{Z_i} = \alpha \cdot Softmax(\frac{Z_i^q \cdot {Z^k_i}^T}{\sqrt{d}})\cdot Z^v_i + Z_i
  $$
  这里的$Softmax(\cdot)$是按行做的,$\alpha$是自适应的。得到的$\hat{Z_i}$再通过一个MLP层得到最终的空间感知特征$f_i^\prime = \mathrm{MLP}(\hat{Z_i})+ \hat{Z_i}$。MLP包括两个线性层和三个Relu层。

  **intra-video Context Encoding:IVCE**（视频内，帧间）

  IFCE可以发现单个视频内的subtle spatial semantics 信息,但是对于fine-grained的不同顺序和方差不能够很好的解决,IFCF仅从空间角度而非是时间角度,所以引入了IVCE,其能够自适应的捕捉视频中的远距离时间动态,并根据时间背景增强所有帧的特性。

  ![image-20231124165028469](attachments\image-20231124165028469.png)

  输入的video sequence $v_j \in \mathrm{R}^{t\times n^2 \times d}$,该序列已经过IFCE模块,使用平均池化去压缩空间维度,生成帧级的全局表示$V_j \in \mathrm{R}^{t\times d}$,然后使用可学习的位置编码 $P \in \mathrm{R^{t\times d}}$,应用于$V_j$,由于是"Token Perceptron Layer",所以首先对$V_j$进行转置,然后是"Channel Perceptron Layer",所以又要将其转置回来。整个过程如下：
  $$
  \hat{V_j} = \mathrm{ReLU}(V^T_j W_{t_1})W_{t_2} + V^T_j \\
  \hat{V_j^\prime} = \mathrm{ReLU}(\hat{V}^T_j W_{t_1})W_{t_2} + \hat{V}^T_j
  $$
  整个过程是4层的线性层。

  **Intra-episode Context Encoding**（IECE）（视频间）

  然而，由于细粒度动作的类间方差较小，上述任务区分嵌入容易过度拟合所见类别的线索，这可能会妨碍 FS-FG 动作的良好泛化。为了缓解这一问题，我们引入了 IECE 模块作为任务适应路径，以捕捉视频中的交互式辨别线索。这样就可以在一集视频之间进行共同适应，从而获得特定任务的嵌入。

  ![image-20231124171511652](attachments\image-20231124171511652.png)

  

  结构与IVCE相同,首先给定一个输入集e,其中经过IFCE模块,同时**挤压其空间特征维度**得到,$E\in \mathrm{l \times t \times d}$,然后继续挤压其时间维度dedao$\hat{E} \in R^{l\times d}$,每一个视频的全局Embedding是$\hat{E_i} \in \mathrm{R}^d$,之后的操作就与上述模块相同。

- Multi-view Matching

  上一节中提出的多视图编码可以捕捉帧内、视频内和集内的上下文，从而增强用于匹配的子序列表示。为了确保相互补充的能力，在多视图匹配中采用了定制的匹配函数来匹配丰富的视频表示，其中涉及两种类型的匹配函数，即时间和非时间匹配函数，这取决于对时间信息的利用。具体来说，特定实例匹配（I-M）作为一种**时态函数**被用于 IFCE 模块丰富的嵌入，而特定类别匹配（C-M）和特定任务匹配（T-M）作为两种**非时态函数**被分别用于 IVCE 和 IECE 模块丰富的嵌入。

  **Instance-specific Matching**

  在得到通过IFCE模块的特征之后,使用Instance-specific matching函数去match query video 。$V_j = \{f^j_y\}_{y=1}^t \in \mathrm{R^{t \times d}}$,以及$V_i = \{f^i_x\}_{x=1}^t \in \mathrm{R^{t \times d}}$,由于 IFCE 模块不包含时间信息，我们将两段视频的帧对齐，并推断出有序的时间对齐得分，作为视频与视频之间的相似度。.根据 OTAM 的启发，计算成对匹配矩阵 $M_{ij} \in \mathrm{R}^{t \times t}$ ,$M_{ij}(x,y)$代表视频i第x帧和视频j的第y帧。使用的是余弦相似性。
  $$
  M_{ij}(x,y) = 1-\frac{f^i_x \cdot f^j_y}{\parallel f^i_x \parallel \parallel f^j_y \parallel}
  $$
  同时也使用了DTW算法,得到视频i和j的分数。然后们对查询视频和所有支持视频之间的视频到视频相似度进行平均，得出类别 c 的最终得分$\mathcal{D}^c_1$。

  **Category-specific Matching**

  在获得由 IVCE 模块丰富的支持视频和查询视频的嵌入信息后，通过从查询样本到支持原型的cross-attention过程来构建**以查询为中心的**原型重构。具体来说，**将每个support video的所有帧embedding向量进行stack**，以获得每个动作类别 c 的类别原型 $V_c$。给一个query video clip $V_j \in \mathrm{R}^{t\times d}$,重建类别原型如下:
  $$
  \hat{V}^c = \mathrm{Softmax}(\frac{V_jW_Q(V^cW_K)^T}{\sqrt{d_k}})V^cW_V
  $$
  然后获得以这里的$W_Q/W_K/W_V \in\mathrm{R}^{d \times d_k}$,然后可以获得query video 到类别c原型的距离,$D^{q \rightarrow s}_s = \parallel V_jW_V - \hat{V}^c \parallel$,然而仅仅是以原型为中心,不够因此进行了一个对称操作，及以原型为中心。然后将查询视频的重构帧Embedding与c类原型之间的距离汇总,得到$D^{s \rightarrow q}_c$,最后将这两个距离相加得到最终的距离,$\mathcal{D}^c_2 = D^{s \rightarrow }_c + D^{q \rightarrow s}_c$ 。

  **Task-specific Matching**

  给定support video $V_i$ 和query video $V_j$,他们是经过IECE的模块,其中$V_j = \{f^j_y\}_{y=1}^t ,V_i = \{f^i_x\}_{x=1}^t \in \mathrm{R^{t \times d}}$,考虑到支持视频和查询视频都以特定任务的方式进行了语境化处理，从而实现了隐式对齐效果，查询集中的每个帧都与支持集中最接近的帧进行了匹配，并对所有查询帧的得分进行了平均，从而得到最终的视频与视频相似度。
  $$
  {\mathrm{D}_{j}^{i}=\frac{1}{t} \sum_{f_{x}^{i} \in V_{i}}\left(\min _{f_{y}^{j} \in V_{j}}\left\|f_{x}^{i}-f_{y}^{j}\right\|\right)}
  $$
  以及其对称过程:
  $$
  {\mathrm{D}_{i}^{j}=\frac{1}{t} \sum_{f_{y}^{j} \in V_{j}}\left(\min _{f_{x}^{i} \in V_{i}}\left\|f_{y}^{j}-f_{x}^{i}\right\|\right)}
  $$
  因此得到$\mathcal{D}^c_3 = D^{i}_j + D^{j}_i$

- Multi-task Learning with Multi-view Fusion

  为了加强多视角编码的协作能力，并融合不同多视角匹配函数的贡献，我们将拟议框架的优化重新表述为多任务协作学习范式，并从多视角融合的角度提高性能。将上述环节得到的$\mathcal{D}_1,\mathcal{D}_2,\mathcal{D}_3$分别通过$Softmax(\cdot)$得到类概率$Y_1,Y_2,Y_3$,然后分别计算这三个概率的交叉熵,$\mathcal{L}_1 = CE(Y_1,Y^q),\mathcal{L}_2 = CE(Y_2,Y^q),\mathcal{L}_3 = CE(Y_3,Y^q),$因此得到总的损失$\mathcal{L}= \mathcal{L_1}+\mathcal{L_2}+\mathcal{L_3}$,在推理阶段预测为$Y= Y_1+Y_2+Y_3$

## 实验

数据集:FineGym, Diving48

- FineGym:提供了体操比赛中分层fine-grained动作的注释，其中包括两个层次的细粒度类别，即 Gym99 和 Gym288，分别有超过 34k 个（99 个类别）和 38k 个（288 个类别）样本。在 Gym99 中，对训练/验证/测试类别采用了 61/12/26 的分割。由于**multi-shot**的支持样本不足，Gym288 中的一些动作被排除在外，这导致训练/验证/测试类别的分割被设置为 128/25/61。
- Diving48 是一个精细的竞技潜水视频数据集，由 48 个定义明确的潜水序列的 18,404 个剪辑视频片段组成。这对于精细动作识别来说是一项具有挑战性的任务，因为不同阶段的潜水动作可能不同，因此需要对长期时间动态进行建模。在此，作者随机选择了 28 个训练类别、5 个验证类别和 15 个测试类别。

实验细节:

- 使用ResNet-50作为embedding model,其使用在ImageNet 预训练的模型。每个视频均匀采样8帧,设置IFCE的module的$n=4$,$f_i$的embedding 维度为2048（RN50第4个stage输出特征）,输入的input video frame被随机裁剪成$224\times 224$,使用了基本的数据增强,包括随机裁剪和随机水平翻转。用于优化 M3Net 的优化器是 SGD 优化器，其初始化学习率为 1e - 4。训练过程在所有数据集上持续 60 000 次，每 2 000 次后学习率下降 0.5。在进行中心裁剪之前，所有输入的大小调整为 256 × 256。在所有数据集上对 FS-FG 动作识别进行了 5 路 1/3/5 镜头评估，并报告了从测试集中随机选取的 **6 , 000** 个事件所获得的平均准确率。

SOTA:

- ![image-20231124194014332](attachments\image-20231124194014332.png)

消融实验：

- ![image-20231124194117417](attachments\image-20231124194117417.png)