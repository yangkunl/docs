**CLIP-guided Prototype Modulating for Few-shot Action Recognition（IJCV在投）**

# 模型结构图

![image-20231108140018745](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20231108140018745.png)

## 方法

这篇文章如其标题，其使用了CLIP来指导其做few-shot learning时候的原型生成。CLIP是一个跨模态的pre-trained模型，CLIP的backboone部分，视觉可选Resnet和VIt，文本部分则使用transformer。

###训练阶段

- 加载视觉编码器和文本编码器来分别提取视频和类别的特征

- 冻住文本编码器（提供知识的可迁移性，减少优化负担）

- **将提取的两种特征计算相似性，进而计算对比学习的loss**（相比于直接加载CLIP预训练模型，提点的主要创新）

  ![image-20231108142851629](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20231108142851629.png)

- 将support video两种特征**concat**一起输入tempoal transformer 得到原型

- query video 不通过text Encoder ，**不将两种特征concat**起来

- 通过query与原型的比较，得到few-shot loss ，将两种loss加起来做更新

### 测试阶段

- CLIP本来就能做zero-shot预测，因此其将CLIP的zero-shot预测与few-shot预测合并起来

  ![image-20231108141420524](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20231108141420524.png)

### 实验设置

####SOTA

- 其在Kinetics，UCF101，SSv2，HMDB51上做了实验，5-way K-shot 其中K 从 1到5

![image-20231108141648571](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20231108141648571.png)

- 其加载ViT的模型在各个数据集上均取得了最好，但是今年另一篇文章**超过了它的效果**，其同样也是使用了CLIP（Xing J, Wang M, Hou X, et al. Multimodal Adaptation of CLIP for Few-Shot Action Recognition[J]. arXiv preprint arXiv:2308.01532, 2023.）

![image-20231108142109400](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20231108142109400.png)

#### plug and play

- 作为一个即插即用的模型，该方法还可加在其他的方法上来**提升效果**，没看过其他方法不清楚在干嘛。

  ![image-20231108142502929](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20231108142502929.png)
