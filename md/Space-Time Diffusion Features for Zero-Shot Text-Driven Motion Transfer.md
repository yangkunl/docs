# Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer

## overview

![image-20240330104611046](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20240330104611046.png)

提出了一种新的文本驱动运动转移（motion transfer）方法--在保持输入视频的运动和场景布局的同时，合成符合描述目标物体和场景的输入文本提示的视频。先前的方法仅限于在相同或密切相关的物体类别中跨两个主体进行运动转移，并且适用于有限的领域（如人类）。

本文着重于shape和fine-grained motion 特征。使用pre-trained 和 fixed 的**text--to -vedio diffusion model**，该模型提供了生成和动作的先验信息。该方法的核心是一个从模型中得到的时空损失。这种损失会引导生成过程保留输入视频的整体运动，同时在形状和细粒度运动特征方面与目标对象保持一致。保留输入视频关键运动特征，同时形状发生改变是一件非常难的事情。要解决这个问题首先需要先了解物体在变形和不同视角下的外观、姿态和运动。之前的方法都聚焦在相似的类别，以及相似的视频。我们的目标是解决一个更通用的设置，其中涉及在形状和变形随时间发生显着变化的情况下跨不同对象类别传递运动（图 1）

通过避免pose和shape的mid-level建模，我们深入研究了视频模型学习到的中间时空特征表示，并引入了一种新的损失，指导目标视频的生成过程以保留原始视频的整体场景布局和运动。

- 一种有效的零镜头框架，利用预先训练的文本到视频模型的生成运动先验来执行运动转移任务。
- 关于通过预训练的文本到视频扩散模型学习的时空中间特征的新见解。
- 用于评估两个视频之间结构偏差下的运动保真度的新指标。
- 与竞争方法相比，这是最先进的结果，在运动保持和目标提示保真度之间实现了显着更好的平衡。

## Related Works

**T2V Model**

中间视频表示$x\in \mathbb{R}^{F\times H^\prime \times W^\prime \times 4}$，通过decode 可以得到输出的RGB vedio $\mathcal{V} \in \mathbb{R}^{F\times H \times W \times 3}$。在本文中使用ZeroScope T2V。

## Method

给一个输入的视频$\mathcal{V}$和一个目标文本prompt P，目标是生成一个新视频$\mathcal{J}$保留原视频的动作和语义信息，同时还要符合P的要求。本文使用了ZeroScope （预训练的latent T2V model）。本文的核心方法是一个新的目标函数，在生成$\mathcal{J}$的过程中起指导作用。具体来说，展示了空间维度上一维的统计量。特征的空间边缘均值，可以作为强大的每帧全局描述符，它（i）保留空间信息，例如对象的位置、姿势和场景的语义布局，以及（ii）对像素级变化具有鲁棒性外观和形状。

### Space-time analysis of diffusion features

给定一个输入视频$\mathcal{V}$，使用带有空提示的DDIM反演，获得latents的序列$[x_1,\dots,x_T]$，其中$x_t$代表第生成第$t$步的video latent。我们将latent $x_t$输入到网络提取器时空特征$f(x_t)\in \mathbb{R}^{F\times M\times N\times D}$。其中$F,M,N$分别代表帧数，高，宽。以及D维特征激活的宽度。

**Diffusion feature inversion**

为了更好的理解特征$\{f(x_t\}^T_{t=1}$的编码内容，本文采用了“feature inversion”的概念，目标是优化随机初始化的视频 $\mathcal{V}^*$，当输入网络时会产生这些特征。具体来说，这是在 $\mathcal{V}^*$ 采样过程中使用特征重建指导来实现的。具体来说：
$$
\hat{x}_T  \sim \mathcal{N}(0,\mathcal{I}),\\
\hat{x}_{t-1} = \Phi(x^*_t,P_s),\mathrm{where}\  x^*_t = \mathrm{argmin}_{\hat{x}} \|f(x_t) -f(\hat{x}_t)\|^2
$$
其中$\Phi$是Diffusion model，$P_s$是描述输入视频的一般文本提示（例如“汽车”）。我们在每个生成步骤中使用梯度下降来最小化特征重建目标。

![image-20240330131103164](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20240330131103164.png)

上述图展示了从输入视频中提取的时空特征的反演结果，我们多次重复反转过程，每次都有不同的随机初始化（即不同的种子）。我们观察到倒置的视频几乎重建了原始帧（图2（b））。最终，我们选择找到一个特征描述符，它保留有关对象姿势和场景语义布局的信息，但对外观和形状的变化具有鲁棒性。为了减少对像素级信息的依赖，我们引入了一种新的特征描述符，称为空间边缘均值（SMM），通过减少空间维度获得。正式地，
$$
\mathrm{SMM}[f(x_t)] = \frac{1}{M\cdot N}\sum^M_{i=1}\sum^N_{j=1}f(x_t)_{i,j}
$$
其中$f(x_t)_{i,j}\in \mathbb{R}^D$是时空特征体积 $f (x_t)$ 中空间位置 $(i, j)$ 处的特征。我们重复反演实验（方程 1），以  $\{\mathrm{SMM}[f(x_t)]\}^T_{t=1}$作为要重建的目标特征。

### Motion-guided video generation

我们的特征反演分析提出了这样的问题：是否可以使用相同的方法进行编辑，只需用等式（1）中的编辑提示 P 替换源提示 Ps 即可。 1.

图 7 显示了几个视频的结果，其中演示了两个问题：(i) 根据初始化，优化可能会收敛到局部最小值，其中对象的准确位置及其方向可能与输入不同，(ii) ）SMM特征仍然保留外观信息，这降低了文本提示的保真度。我们提出以下两个组件来解决这些问题。

**Pairwise SMM differences**

如图7所示，直接针对SMM特征进行优化通常可以防止我们偏离原始外观。为了解决这个问题，我们提出了一个目标函数，旨在保留 SMM 特征的成对差异，而不是它们的精确值。具体来说，使用$\phi^t_i,\tilde{\phi}^t_i \in \mathbb{R}^d$,分别针对原始视频和生成视频的第 i 帧和第 t 步的 SMM 特征。pairwise SMM differences $\Delta ^t,\tilde{\Delta}^t \in \mathbb{R}^{F\times F\times d}$，被定义如下：
$$
\Delta ^t_{(i,j)} = \phi^t_i -\phi^t_j \quad \tilde{\Delta}^t_{(i,j)} = \tilde{\phi}^t_i -\tilde{\phi}^t_j
$$
对于$i,j \in \{1,\cdots,F\}$。时间步t的loss如下:
$$
\mathcal{L}(\mathrm{SMM}(f(x_t)),\mathrm{SMM}(f(\tilde{x}_t))) = \sum_i \sum_j \|\Delta^t_{(i,j)} -\tilde\Delta^t_{(i,j)}\|^2_2
$$
直观上，这种损失让我们保留了特征随时间的相对变化，同时丢弃了源视频的确切外观信息（图 7）。

**Initialization**

![image-20240330142723986](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20240330142723986.png)

众所周知，扩散去噪过程是以从粗到细的方式执行的，因此，初始化在定义生成内容的低频方面起着重要作用[3, 37]。从随机点进行初始化通常可能会收敛到不期望的局部最小值，其中对象位置没有得到很好的保留。请注意，原始视频的低频信息在 DDIM 倒置中很容易获得

然而，我们凭经验发现这种初始化通常可能会限制可编辑性[44]（参见 SM 的示例）。因此，我们仅从 xT 中提取低频。具体来说，令 x ∈ RF ×M×N 为表示 F 帧的张量，空间分辨率为 M ×N 。我们用 LFψ(x) 表示对 x 进行空间下采样和上采样的操作。然后，我们的初始潜在 ̃ xT 由下式给出：
$$
\tilde{x}_T = LF_{\xi}(x_T) +(\epsilon _0 -LF_\xi(\epsilon_0))
$$
其中$\epsilon_0 \sim \mathcal{N}(0,I)$是一个随机噪声，直觉的说，$\tilde{x}_T$保留 DDIM 噪声的低频，其中高频由 ε0 确定。总而言之，从过滤后的潜在 ̃ xT 开始，我们的方法部署了以下引导生成过程：
$$
x^*_t = \mathrm{argmin}_{\tilde{x}_t}\mathcal{L}(\mathrm{SMM}(f(x_t)),\mathrm{SMM}(f(\tilde{x}_t)) \\
\tilde{x}_{t-1} = \Phi(x^*_t,P)
$$

## Results

我们将我们的方法与以下文本驱动的视频编辑方法进行比较：（i）形状感知分层神经图谱（SA-NLA）[31]，它利用预训练的分层视频表示[27]和预训练的T2I模型[50] ]。 (ii) TokenFlow [19]，一种在预训练 T2I 模型的特征空间中工作的零样本方法 (iii) GEN-1 [17] 和 (iv) Control-A-Video [13]，两者都是视频到视频的扩散模型，以生成为条件