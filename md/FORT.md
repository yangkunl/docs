**Focus Your Attention when Few-Shot Classification**(CVPR2023)

## 摘要:

由于pre-trained model在下游任务上的广泛使用。通常的输入图片包含多个实体,模型没有focus on class-related entities 对于当前的few-shot task。在使用support set fine-tuned ,来自class-independent entities的noise information会影响性能,首次提出利用注意力和梯度信息来定位support images 的key entities。被叫做position prompt。

使用 many-hot attention个 attention logits来优化这个query的性能

方法可以应用在各种vision transformer上

---

## Introduction

meta-learning 方法学习这个**task-shared inductive bias**,

人类的few-shot能力源自其自出生起就一直进行的对大量信号进行监督或无监督的学习,预训练模型可以从大量未标注的数据中学到足够泛化的能力,例如现今的NLP的大语言模型,同时vision transformers 也在计算机视觉领域扮演着越来越重要的角色

很大程度上依赖image-encoder与text-encoder之间的语义对齐,所以只适用于vision-language model,使用了一种position prompts使得vision transformers to few-shot 任务,

- 提出了new form of prompts for vision data,
- 适用于不同transformer模型的attention module

---

## method

![image-20231209150802785](attachments\image-20231209150802785.png)

**问题设置**

backbone使用ViT和swin Transformer,使用support set $\mathcal{T}_s$和query set$\mathcal{T}_q$,support set 包含了$\mathcal{T}_s =\{(x^s_i,y^s_i)\}^{C\times K}_{i=1}$,其包含了$C$个不同的类和每一类$K$个样本,$\mathcal{T}_q =\{(x^q_i,y^q_i)\}^{M}_{i=1}$,整个few-shot任务过程可以被记为:
$$
\psi = \mathcal{A}(f_{\theta},\mathcal{T}_s)
$$
其中$\mathcal{A}$是一个解决方案合集,包含linear probing 和被冻住的$f_{\theta}$。或者是参数高效微调方法，只被允许fine-tune 存在的部分参数或者是新增加的参数。

**多头自注意力机制**

> 已经了解其机制，这里就不在进行赘述了

**分析和动机(Analysis and motivation)**

作者首先尝试了很多种设计对于task solver$\mathcal{A}$,例如:

1. simple machine learning classifier: K-NN(Nearest Neighbor classifier),Ridge Regression 和 SVM
2. plug-and-play inductive meta-solver:ProtoNet[^1] ,R2D2[^2] and MetaOptNet[^3]
3. full or parameter-efficient fine-tuning:VPT[^4],LoRA[^5],SSF[^6]

结果如下图:

![image-20231209203211243](attachments\image-20231209203211243.png)

小样本学习最大的挑战是,在预训练阶段没有见过novel task,而且每一个类只有(1~5)个labeled的样本,尤其是面对未见过的类别时，它们可能会同时关注多个实体[21]。当支持样本足够多时，模型可以更多地关注每个类别中经常出现的关键实体，以缓解这一问题，如附录 B 所示。但是，由于标注样本极少，微调方法无法达到这一目的，而且与类别无关的实体所产生的噪声信息会误导微调过程，损害微调性能。为了提供定性证据，我们在图 4 中展示了关注度得分最高且覆盖 [CLS] 标记约 95% 关注度的补丁，更多细节和定量证据见第 4.2 节。可以看出，原始模型和微调模型都无法将大部分注意力集中在关键实体上。为此，我们提出了位置提示，以明确提示模型关键实体的位置，并在微调过程中引导模型将注意力集中在关键实体上。这种能力可以推广到查询样本中。

**Position prompts**

为了在每张支持图像中找到关键位置,作为微调的prompt,即当前few-shot 任务中与类最相关的关键补丁,我们需要在每张支持图像中找到关键实体的位置，作为微调的提示，即当前少量拍摄任务中与类最相关的关键补丁。人工标注可以提供准确的位置，但需要领域专业知识和人力成本。我们的目标是设计一种自动方法，该方法应：1）针对具体类别来定位与类别相关的补丁；2）充分利用注意力信息。深度解释方法（Deep explanation methods）通过**模型预测计算输入的局部相关性**，可用于突出重点补丁进行分类。对于 Transformers，Rollout [1] 发现使用顶层的注意力进行解释会忽略大部分注意力成分，从而误判patch的重要性，因此我们像 Rollout 一样整合了多层的注意力信息。此外，对于一些预先训练好的模型（图 3），仅使用注意力信息无法实现特定类别的解释，因此我们进一步引入梯度信息作为辅助。

![image-20231209213716617](attachments\image-20231209213716617.png)

---

对于L层的columnar architecture，第l-th的input feature可以写为$\mathbf{Z}^l_{in}\in\mathbb{R}^{N\times d}$,第l-th的attention matrices是$\{\mathbf{A}^l_h\}^H_{h=1}$,在上述分类的基础上对于support sample-label pair$(x^s,y^s)$,计算了$y^s$类的预测得分梯度$p_{y^S}$,$\mathbf{Z}{_{i n}^{l} }$  as $ \nabla_{l}=\partial p_{u^{s}} / \partial \mathbf{Z}{_{i n}^{l} }\in \mathbb{R}^{N \times d}, l=1, \ldots, L$,它可以沿着反向传播路径逐层放大。为此，我们只使用顶层输入特征的梯度，即∇L，并保留其第一原理分量用于去噪。得到的梯度项为
$$
\mathbf{G}=\nabla_{L} \cdot \mathbf{V}_{1} \in \mathbb{R}^{N \times 1}, \quad \mathbf{U}, \mathbf{S}, \mathbf{V}=\operatorname{svd}\left(\nabla_{L}\right)
$$
svd是奇异值分解,$\nabla_L = \mathbf{USV}^T$,$\mathbf{V}_1$是$\mathbf{V}$的第一列向量,对应于$\nabla_L$的最大奇异值,梯度向$\mathbf{G}$可以提供特定类别的信息,，残差连接在从输入到模型预测的信息传播过程中发挥着重要作用，并能在前向传播过程中保留位置信息。为此，我们使用identity矩阵 $\mathbf{I}$ 来增强注意力矩阵。总体而言，第 l-th 层的最终注意力图为:
$$
\widehat{\mathbf{A}}^l = \mathrm{norm}(\mathbf{I}+\mathbf{A}^l +\lambda\cdot\mathbf{G}^T)\in \mathbb{R}^{N\times N}
$$
其中$\mathrm{norm}(\cdot)$代表着对输入进行normalization,使其总和为1,$\lambda$作为超参控制Attention和梯度的比例,对于多头注意力机制$\mathbf{A}^l= \frac{1}{H}\sum^H_{h=1}\mathbf{A}^l_h$,将每一层的注意力平均可以得到最后每一个patch的importance如下:
$$
\mathbf{s}=\mathrm{mean}(\widehat{\mathbf{A}}^1\cdot \widehat{\mathbf{A}}^2\cdots \widehat{\mathbf{A}}^L)\in\mathbb{R}^N
$$
将上述的$\mathbf{s}$矩阵作为每一个patch的highest importance scores as prompts。

对于金字塔结构[36, 70]，（即有特征尺度缩放的结构）由于下采样操作会破坏层与层之间的斑块位置对应关系，因此我们直接使用斑块上的平均注意力分数作为重要性分数，并在必要时引入梯度信息以实现分级。

**Attention enhancement**

由于查询样本的位置提示（或关键补丁）在没有标签信息的情况下是未知的，与现有的提示不同，我们的位置提示不能用于前向推理的输入或中间阶段,为此，我们将位置提示作为注意力的预测目标,并在微调过程中优化其多热呈现2 和注意力对数之间的交叉熵损失。如等式 2 和 3 所示，给定输入tokens,
$$
\min _{\theta} \sum_{\left(x^{s}, y^{s}\right) \in \mathcal{T}_{s}}\left[\operatorname{ce}\left(f_{\theta}\left(x^{s}\right), y^{s}\right)-\alpha \cdot \mathcal{R}\right], \quad \mathcal{R}=\frac{1}{N \cdot|\Omega|} \sum_{n=1}^{N} \sum_{t \in \Omega} \ln \frac{\exp \left(\mathbf{q}_{n} \mathbf{k}_{t}^{T} / \tau\right)}{\sum_{m=1}^{N} \exp \left(\mathbf{q}_{n} \mathbf{k}_{m}^{T} / \tau\right)}
$$


## 引用

[^1]:Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In Advances in neural information processing systems, pages 4077–4087, 2017.
[^2]:Luca Bertinetto, Joao F Henriques, Philip HS Torr, and Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. arXiv preprint arXiv:1805.08136, 2018.
[^3]:Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, and Stefano Soatto. Meta-learning with differentiable convex optimization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 10657–10665, 2019.