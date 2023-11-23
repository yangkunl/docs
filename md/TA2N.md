**TA2N: Two-Stage Action Alignment Network for Few-Shot Action Recognition**(AAAI2022)

## 摘要

现有的小样本视频行为识别都是基于metric learning,通过相似度比较的方法,但是不同的动作例子可能有着不同的时序分布。导致在对其query video 和 support video的时候出现了严重的错位问题。本文作者从时间错位(duration misalignment)和动作演化错位(action evolution misalignment)两个方面去解决了这个问题。其提出了(two-stage action alignment network, $\mathbf{TA^2N}$)。第一个stage是童年各国学习时间仿射变换(tempoal affine transform)去定位动作。通过时间重排和空间偏移预测,来协调query feature匹配spatial-tempoal。

## 创新点

- 从两个不同的角度表明存在明显的错位问题:

  ![image-20231122225409989](attachments\image-20231122225409989.png)

  - 首先由于开始的时间和持续的时间不同,不同视频中动作的相对时间位置通常不一致(定义这个问题为,action duration misalignment, ADM)
  - 动作通常是以非线性的方式演变的,因此对于不同action的时空判别部分不同(即使是同一个类别)(定义这个问题为action evolution misalignment,AEm)

- 为了解决上述两个问题,设计了一个两阶段的网络:

  - 第一个阶段,利用时空变换模块(Tempoal Transformation Module,TTM),预测输入视频的时空扭曲参数,然后对特征序列进行仿射变换,使其与动作持续时间保持一致
  - 第二个阶段,采用行动坐标模块(Action Coordinate Module,ACM),从时间和空间两方面对行动演变进行调整。预测配对帧的空间偏移,然后在空间特征上执行相应的运动使他们对齐。

## 方法

### 量化时间错位

- 对三个数据集(UCF101,HMDB51,SSv2)的时间错位进行了量化

- 结果如下:

  - 动作开始时间在整个数据集的分布

    ![image-20231122230333785](attachments\image-20231122230333785.png)

    对于 UCF101 和 HMDB，由于大多数视频都经过粗略剪辑，因此开始时间主要分布在第一帧或第二帧。相反，SSv2 数据集上的开始时间平均分布在前四帧。这表明 SSv2 数据集中的动作**更有可能在不同的时间段执行**，这可能导致动作开始时间和持续时间不一致

  - 对于AEM问题,估计了AEM分数去计算视频中的动作演变

    AEM计算公式:
    $$
    AEM = \frac{1}{M}\sum_{i,j}[1-cos(P_i,P_j)],\forall{i,j}
    $$
    其中,$P, P\in R^{C \times T}$是frame-level上的类别可能性矩阵(使用模型TSM提取的),$C$是类别数,$M = 2 \cdot v_{num} \cdot(v_{num}-1)$,$v_{num}$是数据集的视频中数
    
    ![image-20231122230857910](attachments\image-20231122230857910.png)
    
    这说明 AEM 问题更为严重。总之，通过以上分析，可以得出结论：在这三个数据集中普遍存在着不同程度的动作错位问题。SSv2 数据集的问题最为严重，而 HMDB 受此问题的影响比 UCF101 更大。因此，作者认为解决动作不对齐问题对于少镜头动作识别至关重要，尤其是在 SSv2 数据集上。基于这些观察结果，本文试图通过提出一个可行的框架来解决错位问题。

### 方法

- Feature embedding 使用与TSN相同的方法,将一个视频分成T段,然后从每一段里随机均匀采样.因此每个视频被表示为序列。

  $X = \{x_1,x_2,..,x_T\}$,使用一个预训练的网络去提取特征f($\cdot$),我们可以得到特征序列 
  $$
  f_X = f(X) = \{f(x_1),f(x_2), ..,f(x_T)\} \in \mathrm{R^{C\times T\times H \times W}}
  $$
  使用$f_s,f_q$来表征query video和support video 在video-level的feature

- Temporal Transform Module(TTM 模块)

  TTM模块可以定位动作开始的时间,duration feature 会被强调,与动作无关的特征将被忽略。其包含了两部分定位网络(localization network **L**)和时间仿射变换(tempoal affine transformation T)。具体来说,输入帧级的特征序列$f_X$,定位网络**L**生成扭曲参数(warping parameters) $\phi = (a,b) = \mathbf{L}(f_x)$（**定位网络输出的尺寸为$n\times 2,n$是输入的视频数**）。定位网络结构如下:

  ```python
       self.locnet=torch.nn.Sequential(
              nn.Conv3d(dim[0],64,3,padding=1),
              nn.BatchNorm3d(64),
              nn.MaxPool3d(2),
              nn.ReLU(),#5,4,4
              nn.Conv3d(64,128,3,padding=1),
              nn.BatchNorm3d(128),
              nn.MaxPool3d(2),
              nn.ReLU(),#3,2,2
              nn.AdaptiveMaxPool3d((1,1,1)),
              nn.Flatten(),#128
              nn.Linear(128,32),
              nn.ReLU(),
              nn.Linear(32,2),
              nn.Tanh(),
          )
  ```

  然后输入的特征序列被仿射变换矫正,总过程如下:
  $$
  \hat{f_X} = \mathbf{T}_{\phi}(f_X),\phi = \mathbf{L}(f_X)
  $$
  $\hat{f_X}$表示了动作持续期一致的特征序列,$\mathbf{L}$由几层可训练的网络结构组成,由于帧序列之间的动作持续,时间错位具有线性特征。使用线性时间插值来进行仿射变换。线性时间插值代码如下(核心为$ax + b$,$a,b$为通过定位网络获得的扭曲参数,$x$为等间距的参数):

  ```python
   grid_t=torch.linspace(start=-1.0,end=1.0,steps=self.T).cuda().unsqueeze(0).expand(n,-1)
          grid_t=grid_t.reshape(n,1,T,1) #source uniform coord
          grid_t=torch.einsum('bc,bhtc->bht',theta_support,torch.cat([grid_t,torch.ones_like(grid_t)],-1)).unsqueeze(-1)
          grid=torch.cat([grid_t,torch.zeros_like(grid_t)-1.0],-1) # N*1*T*2 -> (t,-1.0)
          # grid=torch.min(torch.max(grid,-1*torch.ones_like(grid)),torch.ones_like(grid))
          grid_support=grid
  ```

  **(注意经过仿射变换之后的特征尺寸是不发生变化的)**

  这也有利于整个管道的可微分性，因此可以端到端方式与 TTM 联合训练分类器。具体结构如下:

  ![image-20231123104941420](attachments\image-20231123104941420.png)

- Action Coordinate Module

  第二种错位，即动作演化错位，是由于视频中**动作的非线性演化**造成的，基于线性的 TTM 无法充分解决这一问题。为此，从**时间和空间两方面**协调视频中的动作演变。

  - Tempoal coordination

    为了在时间上调整视频间的动作演变，应将视频间相似的运动模式汇总到相同的时间位置。具体流程如下:

    ![image-20231123105554523](attachments\image-20231123105554523.png)

    将其视为一项全局协调任务，在这项任务中，可以对查询视频的运动演化进行时间上的重新安排，使其与支持视频相匹配。我们对support 和 query 之间的视频 motion evolution correlation 进行建模。$M \in \mathrm{R}^{T \times T}$,建模公式如下:
    $$
    M=Softmax(\frac{(W_k \cdot G(\hat{f_s}))(W_q \cdot G(\hat{f_q}))^T}{\sqrt{dim}})
    $$
    这个就是注意力公式。$W_k和W_q$都是线性层,$G$是在空间维度上全局平均池化($C \times T \times H \times W \rightarrow C \times T \times 1 \times 1$),**其中$C$是通道维度,$H$和$W$是长和宽,根据代码为(2048,7,7)代表Resnet50第四个stage输出的特征尺寸**。之后再通过了一个Softmax层使该值位于$[0,1]$之间。然后使用得到的M矩阵对query video 的时序特征进行重排。**注意上述的注意力分数都是在frame-level级计算的attention score**,所以M的尺寸为$N\times M\times T_1 \times T_2$,其中$N,M$分别为s和q的视频数,$T_1,T_2$为s和q的帧数。
    $$
    \tilde{f_q} =\hat{f_q}+ M \cdot (W_v \cdot G(\hat{f_q}))
    $$
    与上述两个$W$一样,$W_v$也是线性层,$G$也是在空间维度上的全局平均池化,同时为了保证support video 和 query video的一致性因此:
    $$
    \tilde{f_s} = \hat{f_s}+W_v \cdot G(\hat{f}_s)
    $$

  - Spatial coordination

    TC确保action在持续时间内以相同的进程发展,但是空间变化,比如actors的位置。因此提出了SC模块,如下:

    ![image-20231123131557494](attachments\image-20231123131557494.png)

    其旨在预测每个配对帧的空间偏移,其包含两个阶段一个轻量化的偏移预测和偏移掩码的生成。给定上述得到的$\tilde{f_s}$和$\tilde{f_q}$,将他们通过一个偏移预测器(offse predictor )S,来预测偏移 $O,O \in \mathrm{R}^{T \times 2}$,两维代表了 x 和 y。$O = S(Cat(\tilde{f_s},\tilde{f_q}))$,然后可以通过相应的空间偏移量来定位交叉区域。然后使用掩码生成器生成尺寸为($T \times H \times W$的mask矩阵)。生成器的公式如下:
    $$
    m_{x}(o)=\left\{\begin{array}{ll}
    \max \left(0,1-\gamma\left(o_{x}-1-x\right)\right) & , x \leq o_{x}-1 \\
    1 & ,\left|x-o_{x}\right|<1 \\
    \max \left(0,1-\gamma\left(x-o_{x}-1\right)\right) & , x \geq o_{x}+1
    \end{array}\right.
    $$
    生成的mask 矩阵为$I_o = m_x \times m_y$

    

    此外，掩码值在交叉区域内为 1，在边缘处逐渐减小为 0。
    $$
    \bar{f}_{s,i} = \sum_{HW}(I_{o_i}* \tilde{f_s})/\sum I_{o_i}, i = 0,...,T \\
    \bar{f}_{q,i} = \sum_{HW}(I_{o_i}* \tilde{f_s})/\sum I_{o_i}, i = 0,...,T
    $$
    经过TC和SC处理后,视频中的动作演化错位将在时空方面被消除。对其良好的成对特征$\bar{f_s}$和$\bar{f_q}$将作为原型网络方案用于最终的距离测量和分类。

  - Optimization

    使用原型网络框架进行训练,
    $$
    \mathrm{P}\left(x_{q} \in c_{i}\right)  =\frac{\exp \left(-d\left(\overline{f_{q}}, \bar{p}_{s}^{c_{i}}\right)\right)}{\sum_{c_{j} \in C} \exp \left(-d\left(\overline{f_{q}}, \bar{p}_{s}^{c_{j}}\right)\right)} \\
    d(f, p)  =\sum_{t=1}^{T} 1-\frac{<f_{[t]}, p_{[t]}>}{\left\|f_{[t]}\right\|_{2}\left\|p_{[t]}\right\|_{2}}
    $$
    其中$d(f,p)$是帧向余弦距离度量,分类的loss如下:
    $$
    \mathcal{L}_{c l s}=-\sum_{q \in Q} \mathbb{I}\left(q \in c_{i}\right) \log \mathrm{P}\left(x_{q} \in c_{i}\right)
    $$

  

  

## 实验设置

- 数据集

  UCF101,HMDB51,SSv2,Kinetics-CMN

- 比较的方法

  ProtoNet, TARN, ARN, TRN++,OTAM

- 实验细节

  做了 5-way 1 shot 和 5-way 5-shot 分类方法,对视频按TSN的方法采样8帧。将每一帧resize成$256\times 256$,使用了随机水平翻转。在训练中还使用了随机裁剪。使用ImageNet 上的pre-trained 模型 ResNet-50作为feature extractor。在一个epoch中采样200个task训练。