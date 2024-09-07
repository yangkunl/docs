# Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding

## overview

![image-20240318213950622](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20240318213950622.png)

对于多模态模型来说，视觉的token相比于文本的token太少了。

采用一组动态视觉标记来统一表示图像和视频。这种表示框架使模型能够有效地利用数量有限的视觉标记，同时捕捉图像所需的空间细节和视频所需的综合时间关系。此外，作者利用多尺度表示法，使模型能够感知高级语义概念和低级视觉细节。值得注意的是，ChatUniVi 是在包含图像和视频的混合数据集上进行训练的，因此可以直接应用于涉及这两种媒介的任务，而无需进行任何修改。广泛的实验结果表明，Chat-UniVi 作为一个统一的模型，其性能甚至一直优于专为图像或视频设计的现有方法。

现有问题：

专注于图像输入的模型，更加看中空间关系。而专注于视频输入的模型，更加看中时间关系。

一些方法对每个图像和视频生成固定数量的token，但是这些方法缺乏时间理解（model tempoar comprehension）

视频最初被划分为若干事件，然后视觉标记扩展到每个事件中的帧，



通过merge方法逐步合并相似语义的视觉token，



先使用vision Transformer初始化，再使用K-NN进行merge

具体merge的时候就是将相似token的特征进行平均

- 统一的图像和视频建模方法允许在图像和视频混合数据集上训练，从而无需任何修改即可直接应用于图像和视频任务‘
- 多尺度的特征有助于理解视频和图片
- 我们使用多尺度动态视觉标记统一表示图像和视频，并提出了一种标记合并方法来获取这些动态视觉标记。

## Related Work

Large Language Models：最近，大型语言模型[27, 45, 48, 58]取得了颠覆性的进展，这主要归功于训练数据的扩大和模型参数的大幅增加。受 GPT-3 [7]的成功启发，许多大型语言模型随后被开发出来，包括 PaLM [13]、OPT [71]、BLOOM [50]、InstructGPT [43] 和 ChatGPT [41]。然而，语言只是交流的一个方面。视觉信息可以增强我们对世界的理解[5, 23-26, 29, 59]。在这项工作中，我们引入了 Chat-UniVi，它不仅能理解文本并生成回复，还能结合视觉输入，从而为回复生成提供更全面、更身临其境的语境。

Large-scale Multimodal Models：可以将现存的多模态模型分为两类，第一类：对于不同的视觉任务使用不同的专家模型。包括VisualChatGPT，HuggingGPT，MMREACT和ViperGPT。

第二类：将不同模式的模型整合为端到端可训练的模型，包括GPT-4，LLaMA-Adapter V2。

尽管取得了值得称道的进展，但现有方法往往只关注图像或视频输入。

Flamingo 也支持image和vedio输入，其是提取固定数量的token对于视频

最近的研究[9, 61]探索了使用单独预训练的图像和视频编码器进行处理的方法，但这些方法会带来模型冗余，而且一起训练也具有挑战性。因此，这与我们实现统一视觉语言模型的重点并不一致。

**Dynamic Visual Token**：

一些方法探索使用动态token对于transformer 框架，但是并未扩展到视频输入。

作者认为其方法的优点如下：

- 支持视频输入，整合了视频和图片的输入
- 无参数方法

## Methodlogy

Chat-UniVi 的目标是在一个统一的框架内，在大型语言模型（LLM）可以理解的语言序列中同时为图像和视频建模。为了实现这一目标，Chat-UniVi 通过一组动态视觉标记来统一表示图像和视频，并将图像和视频连接起来。

### Dynamic Visual Tokens for Image and Video

基于ViT方法，许多方法通过将图像划分为网格，将这些网格作为同等重要的视觉token，不过，显然并非所有区域在视觉语言任务中都具有同等重要性。

### Spatial Visual Token Merging

对于输入的图片，使用CLIP的视觉编码器来提供原始的视觉token $\mathrm{Z} = \{\mathcal{z_i}\}^L_{i=1}$,L是每张图片划分的tokens数量，然后使用DPC-KNN，一种基于近邻的密度峰聚类算法来聚类这些visual tokens。对于每一个$z_i$首先计算出local density $\rho _i$,使用K-NN近邻法，格式化如下：
$$
\rho_i = \mathrm{exp}(-\frac{1}{K}\sum_{z_k \in \mathrm{KNN(z_i,Z)}})\| z_k -z_i\|^2 
$$
其中$\mathrm{KNN}(z_i,Z)$，是K-NN算法对于$Z\backslash{\{z_i\}}$,其意味着从$Z$中移除$z_i$。然后我们计算距离index$\delta_i$
$$
\delta_{i}=\left\{\begin{array}{ll}
\min _{j: \rho_{j}>\rho_{i}}\left\|z_{j}-z_{i}\right\|^{2}, & \text { if } \exists j \text { s.t. } \rho_{j}>\rho_{i} \\
\max _{j}\left\|z_{j}-z_{i}\right\|^{2}, & \text { otherwise. }
\end{array}\right.
$$
将相对$\rho_i \times \delta_i$相对较高的token作为聚类中心，然后根据欧式距离将其他token分配到其最近的聚类中心

### Temporal Visual Token Merging

因此，我们提出了时间视觉标记合并法，首先将视频划分为几个关键事件。然后，我们让视觉标记只在同一事件的帧上扩展。
