**FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning**(NeurIPS2023)

## 存在的问题

现有的方式尝试微调模型的分类头,或者引入额外的结构。为了提高模型性能,需要在下游任务上进行微调,然而在存在分布偏移(数据的分布出现偏移)的情况下,对预训练模型进行微调会导致其**泛化能力下降**,而在few-shot learning的时候由于样本数量有限**模型极易出现过拟合**。

本文的作者尝试通过通过保持fine-tuning过程中保持spurious feature的一致性,来增强模型的generalizability(泛化性)。其提出的模型能够简单的提高现有模型的性能。

## Pipeline

CLIP对齐了visual 和 textual 特征在 unified Embedding space,在下游任务中表现了卓越的性能,包括iamge 。经典做法是fully fine-tuning CLIP(使用下游数据对整个模型进行微调),但是在真实任务中没有这么多labeled data 可以获得,因此使用 fully fine-tuning会导致**过拟合和模型表现降低**。

为了缓解(mitigate) 这个few-shot challenge,使用与下游目标数据集相关的代理数据集(proxy dataset)来fine-tuning CLIP模型(为了高效的泛化到整个 target task),直接fully fine-tuning CLIP是不可行的(feasible),(直接使用会导致过拟合以及worse out-of-distribution (OOD,指**模型在训练时未见过的数据分布**) generalization),例如下图所示fully fine-tuned的model与原始的CLIP相比更关注局部的区域(local regions),

![image-20231125133611484](attachments\image-20231125133611484.png)

这种local attention会削弱模型对虚假相关性(spurious correlation)的鲁棒性(会导致模型被虚假的相关性欺骗,对于无用的特征过于关注)

>"spurious  correlation"（虚假相关性）通常指的是模型在训练过程中学到的表面上看起来很强的特征或模式，但这些特征或模式实际上是无关或不可解释的。这种现象可能导致模型在测试数据上表现不佳，因为模型在训练时过于依赖这些虚假相关性，而不是真正与任务相关的特征。

在这篇文章,作者主要的目标和任务就是在微调过程中保护CLIP模型对于spurious correlation的鲁棒性,也就是说能够分辨spurious and causal feature。causal feature可能是与classes相关的特征,而spurious features可能是与类别上下文相关的信息。

>"Proxy dataset"（代理数据集）通常指的是一个在某种程度上代表或模拟目标任务的数据集。这个代理数据集通常不是直接来自目标任务，但具有一些与目标任务相关的特性，使得在代理数据集上训练的模型能够在目标任务上表现良好。代理数据集的关键就是使模型在目标任务上的良好泛化性能。

为了达成这个目标,作者提出了 Feature Discrimination Alignment,先是引入了一个 spurious feature classifier,保证在fine-tuning过程中spurious feature的分类概率保持一致(remains consistent),将与类别无关的描述(即上下文)的文本特征作为**虚假特征**(spurious feature prototypes)原型,对图像特征和虚假特征原型进行相似性测量，以确定当前图像虚假特征的概率分布。通过限制微调前后模型提取图像特征的概率分布，我们确保了模型提取虚假特征的一致性。同时，在学习代理数据集的分类能力时，还能确保模型在微调后对**虚假关联的鲁棒性**。

**创新点如下**:

- 使用文本特征获取图像的spurious features
- 提出了特征判别对齐微调架构
- 可以在不引入额外的训练和推理成本的情况下提高现有方法的性能。

## 方法

![image-20231125155344483](attachments\image-20231125155344483.png)

- **Problem Definition**

  使用一个预先训练好的 CLIP ，其中包含一个视觉编码器 $f_0$ 和一个文本编码器$g_0$。此外，我们还可以访问一个few-shot proxy数据集 $\mathcal{D} \subset X \times Y$，其中每个类别的样本都非常有限，每个样本包括一幅图像 $x$ 及其相应的标签 $y$。我们的目标是利用这个 proxy数据集对预先训练好的 CLIP 进行微调，以提高它在与代理数据集相关的未见目标任务中的zero-shot性能。

- **Fine-Tuning  on Proxy Dataset**

  在fine-tuning 过程中,冻住了CLIP的text encoder $g_0$和 visual encoder $f_0$,同时使这个visual encoder 可学习,首先使用CLIP预训练权重$f_0$初始化了visual encoder $f_t$,视觉编码器用于提取图像特征,i.e 图片x的特征 $f_t(x)$。借助于CLIP的文本视觉对齐功能,我们将每个类的类名的文本特征作为类原型,如下:

  对于任意的类别$y$,将M个prompt 模板$(P_1,\cdots,P_M)$与类名结合起来,获得M的prompts $[P_1,y],\cdots ,[P_M,y]$,使用文本编码器$g_0$来提取这M个prompts,然后计算这M个类别的平均作为有关类的原型(prototype),公式如下:
  $$
  \mu ^{class}_y := \frac{1}{M}\sum^M_{j=1}g_0([P_j,y])
  $$
  使用余弦相似度$s(\cdot)$计算图片特征和文本特征之间的相似性,生成图片的类分布,使用交叉熵计算损失:
  $$
  \mathcal{L_{class}} = -\frac{1}{\mid \mathcal{D} \mid} \sum_{(x_1,y_i)\in\mathcal{D}} log\frac{exp(s(f_t(x_i)),\mu^{class}_{y_i})}{\sum_{y\in \mathcal{Y}}exp(s(f_t(x_i)),\mu^{class}_{y})}
  $$
  其中$\mathcal{Y}$是label set。

- **Spurious Feature Constraint(杂项特征约束)**

  在代理数据上对 CLIP 进行完全微调会影响模型对未见数据的稳健性。为了在微调过程中保持模型对分布外数据的性能，我们在微调过程中保持模型对假相关的鲁棒性。也就是说，**保持微调前后模型提取的虚假特征不变**。首先计算了每个prompt 模板 $P_j$,对于所有类别特征的平均作为prompt 模板$P_j$的原型如下:
  $$
  \mu^{spurious}_{P_j} := \frac{1}{\mid \mathcal{Y} \mid}\sum_{y \in \mathcal{Y}}g_0 ([P_j,y])
  $$
  然后使用已经微调的模型(fine-tuned model)提取的特征与 spurious prototypes 计算余弦相似性,然后计算**虚假特征的分布**(distribution over spurious features),
  $$
  \mathcal{P}{_ {spurious} }\left(x ; f_t\right)=\operatorname{SoftMax}\left[s\left(f_t(x), \mu_{P_1}^{ spurious }\right), \ldots, s\left(f_{t}(x), \mu_{P_M}^{ spurious }\right)\right]\\
  \mathcal{P}{_ {spurious} }\left(x ; f_0\right)=\operatorname{SoftMax}\left[s\left(f_0(x), \mu_{P_1}^{ spurious }\right), \ldots, s\left(f_{0}(x), \mu_{P_M}^{ spurious }\right)\right]
  $$
  通过保持微调前后模型对虚假特征的概率分布一致来确保微调前后模型的虚假特征保持一致,即:
  $$
  \mathcal{L _{spurious }}=\frac{1}{|\mathcal{D}|} \sum_{(x_i,y_i) \in \mathcal{D}} \operatorname{KL}\left(\mathcal{P_{spurious}}(x_i; f_t) \| \mathcal{P_{spurious}}(x_i ; f_0t)\right)
  $$
  然后将得到的两个loss加起来:
  $$
  \mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{class} + \beta \cdot \mathcal{L}_{spurious}
  $$
  设置$\alpha$为1,$\beta$为20

- Spurious Prototype Correction

  提示模板由人工设计或者由GPT等LLM生成,这些模板通常包括冗余和不合逻辑的提示。由这些不精确和冗余的提示模板计算出的虚假特征原型缺乏准确性。因此，有必要对这些虚假特征原型进行过滤和处理。

  采用 Isolation Forest 算法来消除与虚假特征相关的无意义原型，如下:
  $$
  \mu^{spurious} := \mathrm{IsoLAIONFOREST}(\mu^{spurious},n)
  $$
  保留rationality 程度最高的n个prototype,然而有时候prompts 会显示出过度的相似性,因此使用k-means来合并由相似信息,即:
  $$
  \tilde\mu^{spurious} := \mathrm{k-means}(\mu^{spurious},k)
  $$
  k是聚类数

## 实验设置

- 设置

  使用ViT-B/32作为CLIP的backbone,prompt模板使用的是OpenAI ImageNet prompt templates,使用SGD ,共60个epoch,使用两个数据集ImageNetV2 和 ImageNet SKetch

