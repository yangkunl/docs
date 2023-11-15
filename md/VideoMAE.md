**VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training(NeurIPS 2022)**

# 模型结构图

![\<img alt="" data-attachment-key="GJZ2YCZ9" width="1026" height="319" src="attachments/GJZ2YCZ9.png" ztype="zimage">](attachments/GJZ2YCZ9.png)

## 方法

### 现有问题

*   由于视频的语义特征在时空变化的时候难以正常的mask（如果按照MAE在空间上随机mask的方式，可能会导致当前mask的会在接下来的帧数暴露出来），因此使用**静态图像的掩膜方式得到效果往往不理想**(类似于MAE的图像随机mask方式)

### 创新点

*   提出了一个视频领域的类似MAE的自监督预训练模型（SSVD，self-supervised video pre-training）（优于从头开始训练或使用对比学习方法预训练的模型）
*   提出了**针对视频的mask机制**，视频自动编码器（VideoMAE）
*   相比于对比学习，VideoMAE 进行 预训练的**数据量较小**（3.5k）

### 方法

* 由于视频具有时间属性，所以使用图像MAE的mask手段，难以mask住某一部分，因为可能在不同时刻进行泄漏,为了避免这种时空冗余，所以使用了极高的遮蔽率。

*   设计了一种$tube  masking$的mask方法

    ![\<img alt="" data-attachment-key="9UTBKG5D" width="987" height="310" src="attachments/9UTBKG5D.png" ztype="zimage">](attachments/9UTBKG5D.png)

*   tempoal downsampling 时间下采样

    使用分步时间采样策略来进行更有效的视频预训练，形式上首先从原视频V中随机抽取一个由t个连续帧组成的视频，然后使用时间采样将片段压缩为 T 个帧，每个帧包含 H × W × 3 个像素（通过时间下采样来避免时间冗余，即通过对视频选取帧，来避免）

*   cube embedding

    将每个大小为 2 × 16 × 16 的立方体视为一个token  emdding。这样，立方体嵌入层就得到了 $\frac{T}{2}×\frac{H}{16} × \frac{W}{16}$ 个三维token，并将每个tokrn映射到通道维度 D 上。这种设计可以**降低输入的空间和时间维度**，有助于减轻视频中的时空冗余。(使用的backbone为ViT)

*   tube masking with extremely high ratios(将90\~95%都mask)

    tube masking 将mask拓展到整个时间轴，即**不同帧共享同一个mask**

*   backbone

    使用了vanilla ViT backbone 以便于能够更好的捕捉时空联合信息，使用了联合时空注意力机制

    ![\<img alt="" data-attachment-key="NL4AACN5" width="698" height="525" src="attachments/NL4AACN5.png" ztype="zimage">](attachments/NL4AACN5.png)
