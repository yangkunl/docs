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

    ![image-20231122230857910](attachments\image-20231122230857910.png)

    这说明 AEM 问题更为严重。总之，通过以上分析，可以得出结论：在这三个数据集中普遍存在着不同程度的动作错位问题。SSv2 数据集的问题最为严重，而 HMDB 受此问题的影响比 UCF101 更大。因此，作者认为解决动作不对齐问题对于少镜头动作识别至关重要，尤其是在 SSv2 数据集上。基于这些观察结果，本文试图通过提出一个可行的框架来解决错位问题。

### 方法

- 