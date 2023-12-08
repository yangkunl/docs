**On the Importance of Spatial Relations for Few-shot Action Recognition**(ACM MM2023)

## 现有方法的问题:

- 大多都聚集在建模support video 和 query video的时间关系上,却忽视了二者之间的空间关系
- 而在实际任务中,空间错位比时间错位来得更加严重
- 提出了一种新颖的空间对齐cross-transformer模型(SA-CT)
- 提出了一个简单但是有效的Tempoal Mixer module(时间混合模块)

---



## 创新点

- 对于FSAR问题,提出了建模空间关系的module
- 将时间信息利用起来作为空间信息的补充
- 将上述两个模块结合起来,得到(SA-CT)模块
- 使用了更强力的预训练模型来提取特征

---



## 方法

**Pipeline**

![image-20231207211613764](attachments\image-20231207211613764.png)

首先feature extractor被使用来提取support video和query video的特征,然后将得到的support feature和 query feature输入到TMixer module去获取高阶的时间特征,**同时减少视频的帧数**,然后将其输入到SCA module 再来建立query-specific的原型去分类,然后将query与prototypes的距离通过对数再传递给训练和推理过程。

---



- **Spatial Cross-Attention** 

  cross-attention 对于视频和图像对齐方面有着不错的能力,因此作者使用这个机制来对齐在query set 和 support set 上的spatial objects。结构如下图:

  ![image-20231207214242869](attachments\image-20231207214242869.png)

  如图首先定义**空间位置$p$的**query video 的$i^{th}$帧输入特征如下:
  $$
  qf_{ip} = \left[\mathrm{\Phi}(q_i)_p + \mathrm{CPE}\left(\Phi(q_i)_p\right)\right]
  $$
  其中$\Phi:\mathbb{R}^{H\times W\times 3}\rightarrow \mathbb{R}^{P^2 \times D}$,代表一个特征提取器(feature extractor),例如:Resne-50(与STRM一样没有使用预训练的Resnet50的最后一个平均池化层),然后$\mathrm{CPE}$是一个位置编码。

  ---

  >CPE是出自ICLR2023的一篇论文,其通过对图像Embedding后的特征先通过一层Encoder,再加上位置编码来避免之前位置编码会破坏图像的平移不变性的问题。

  ---

  同样的**空间位置为m**的,class 为c,的video为第k个的,第$i^{th}$输入特征如下:
  $$
  sf_{ikm}^c = \left[\mathrm{\Phi}(s_{ik}^c)_m + \mathrm{CPE}\left(\Phi(s_{ik}^c)_m\right)\right]
  $$
  在得到输入特征之后的,计算query和support的相似性如下:
  $$
  a^c_{ikmp} = LN\left(\mathrm{W_q}\cdot sf^c_{ikm}\right)\cdot LN(\mathrm{W_k}\cdot qf_{ip})
  $$
  $LN(\cdot)$是layer normalization,$W_q$和$W_k$是可学习的提取注意力的机制,然后使用一个$\mathrm{Softmax}$去获取对于query和support video的attention map:
  $$
  \tilde{a}^c_{ikmp} = \frac{\mathrm{exp}(a^c_{ikmp}/\tau)}{\sum_{l,n}\mathrm{exp}(a^c_{ilnp}/\tau)} ,\tau= \sqrt{d_k}
  $$
  通过上述得到的attention map,可以计算通过query-specific后的原型如下:
  $$
  t_{ip}^c = \sum_{km}\tilde{a}^c_{ikmp} \cdot (\mathbf{W_v}\cdot \Phi(\mathrm{s}^c_{ik})_m)
  $$
  最后，该模块使用欧氏平方距离计算特定查询原型与查询之间的距离，并将距离解析为对数，以表示类别的分布情况：
  $$
  d\left(Q, S^{c}\right)=\frac{1}{P^{2}} \sum_{p}\left\|\frac{1}{L^{2}} \sum_{i}\left(t_{i p}^{c}-\left(\mathbf{W}_{\mathbf{v}} \cdot \Phi\left(q_{i}\right)_{p}\right)\right)\right\|_{2}^{2}
  $$

  ---

- **Temporal Mixer Module(TMixer)**

  这个TMixer 修改(modifies)了两个MLP-mixer,一个标准的MLP-Mixer包括两种MLP-layers:channel-mixing MLP和 token-mixing MLPs,

  ![image-20231207231018216](attachments\image-20231207231018216.png)

  video的特征可以被表示为$\mathbf{F}=[\mathbf{f}_1;\cdots;\mathbf{f}_L\in\mathbb{R}^{L\times P^2\times D}]$,
  $$
  \begin{array}{c}
  \mathbf{U}_{i, *, *}=\mathbf{F}_{i, *, *}+\mathbf{W}_{2} \cdot \sigma\left(\mathbf{W}_{1} \cdot \mathbf{F}_{i, *, *}\right), i=1 \ldots L, \\
  \mathbf{V}_{*, *, j}=\mathbf{U}_{*, *, j}+\mathbf{W}_{4} \cdot \sigma\left(\mathbf{W}_{3} \cdot \mathbf{U}_{*, *, j}\right), j=1 \ldots D,
  \end{array}
  $$

  其中$\sigma$是Relu,其中$\mathbf{W}_1,\mathbf{W}_2\in \mathbb{R}^{L\times L}$,$\mathbf{W}_3,\mathbf{W}_4\in \mathbb{R}^{D\times D}$。首先用上述的公式在frame level 增强了特征,接下来就是减少帧数,公式如下:
  $$
  \begin{array}{c}
  \mathbf{Y}_{*, *, *}=\mathbf{W}_{6} \cdot \sigma\left(\mathbf{W}_{5} \cdot \mathbf{V}_{i, *, *}\right), i=1 \ldots L, \\
  \mathbf{Z}_{*, *, j}=\mathbf{Y}_{*, *, j}+\mathbf{W}_{8} \cdot \sigma\left(\mathbf{W}_{7} \cdot \mathbf{U}_{*, *, j}\right), j=1 \ldots D,
  \end{array}
  $$
  其中$\sigma$是Relu,其中$\mathbf{W}_5\in \mathbb{R}^{L\times L/2},\mathbf{W}_6\in \mathbb{R}^{L/2\times L/2}$,$\mathbf{W}_7,\mathbf{W}_8\in \mathbb{R}^{D\times D}$。然后可以将整个视频的帧数减少到$L/2$帧。

  ---

- 与TRX和STRM的不同

  TRX和STRM是使用子序列来进行匹配,而SA-CT使用cross-attention来进行匹配。

  | TRX                                                |
  | -------------------------------------------------- |
  | 使用cross-attention对frame-level的特征进行增强     |
  | 使用子序列进行匹配                                 |
  | **STRM**                                           |
  | 使用cross-attention对patch-level的特征进行增强     |
  | 使用cross-attention 对frame-level的特征进行增强    |
  | patch 数使用自适应池化减少到$4\times 4=16$个       |
  | 使用frame-level的子序列进行匹配                    |
  | **SA-CT**                                          |
  | 使用cross-attention对patch-level的特征进行增强     |
  | 对于frame-level使用4个MLP层,同时还将帧数减少到一半 |
  | patch数为$7\times 7=49$个,                         |
  | 使用patch-level的距离进行匹配                      |

  

  

