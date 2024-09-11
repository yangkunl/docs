**Delta Denoising Score（ICCV2023）**

# 一、摘要

Delta Denoise Score（DDS）是一种分数方法，用于图像编辑，用于解决Score Distillation Sampling (SDS)在图像编辑应用过程中的模糊。与SDS相比，DDS能够获取更加纯净的梯度。

![image-20240907203736002](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20240907203736002.png)

# 二、引言

SDS是一种采样机制，利用概率密度蒸馏来优化原始图像，**使用 2D 扩散模型作为先验**。SDS的有效性来自其使用的Diffusion model采样而来的先验信息，SDS有向特定模式靠拢的趋势。特别是，使用 SDS 来编辑现有图像，从该图像初始化优化程序，可能会导致图像在编辑元素之外出现严重模糊。DDS使用了两个分支，能够缓解SDS这一问题。

# 三、相关工作

使用Diffusion model做图像编辑，有人直接在image上加噪声再去噪声，但是这样做很难保存想要的image原始的不想要进行编辑的信息，例如布局，背景等。因此要保存相关的信息需要使用用户提供的mask等。DiffEdit也使用了DDIM Inversion 来进行图像编辑，但是为了不使得图像失真其也使用了mask。然后一些text-only的方法，一些使用了文本本地化的方法，也能够不使用mask，但是其只在CLIP上做了工作，而不在Diffusion Model上做工作，SDS分数是由StyleGAN提出的，主要是用来指导领域自适应。

# 四、方法

首先是总结了一下SDS，作者证明了在使用SDS进行图像编辑的时候会引入一个噪声方向。因此作者提出了DDS，通过一个引用对或者说是引用分支，将SDS引入的噪声方向进行修正。

**SDS overview**

$t \sim \mathcal{U}(0,1)$, 这是采样的时间步，也就是确定是否加噪的。同时$\epsilon \sim \mathcal{N}(0,1)$表明加的噪声是一个正态分布。因此扩散损失可以表示为如下式：
$$
\mathcal{L}_{\text{Diff}} \left( \phi, \mathbf{z}, y, \epsilon, t \right) = w(t) \| \epsilon_{\phi} \left( \mathbf{z}_t, y, t \right) - \epsilon \|_2^2,
$$
其中$w_t$是一个权重函数，而$\mathbf{z}_t$是$\mathbf{z}$加噪后得到的，具体的加噪公式为：$\mathbf{z}_t = \sqrt{\alpha_t}\mathbf{z} + \sqrt{1-\alpha_t}\epsilon$.其实就是扩散模型的前向加噪过程，然后$\alpha_t$其实就是扩散模型的noise schedule。为了简化过程，作者省略了权重函数$w(t)$。由$\mathcal{L}_{Diff}$关于$g(\theta)$求导可以得到其梯度如下：
$$
\nabla_{\theta} \mathcal{L}_{\text{Diff}} = \left( \epsilon_{\phi}^{\omega}(\mathbf{z}_t, y, t) - \epsilon \right) \frac{\partial \epsilon_{\phi}^{\omega}(\mathbf{z}, y, t)}{\partial \mathbf{z}_t} \frac{\partial \mathbf{z}_t}{\partial \theta}
$$
这里的$g(\theta)$是一个能渲染图像的任意可变参数函数，如果是运用在图像编辑领域的话，一般是使用$g(\theta) = \theta =\mathbf{z}$。然后根据之前的工作，SDS对中间项也就是$\frac{\partial \epsilon_{\phi}^{\omega}(\mathbf{z}, y, t)}{\partial \mathbf{z}_t}$进行了省略，也就是不在需要扩散模型中的预测噪声模型的梯度，这样更加方便快捷。因此得到的SDS梯度如下：
$$
\nabla_{\theta} \mathcal{L}_{\text{SDS}}(\mathbf{z}, y, \epsilon, t) = \epsilon_{\phi}^{\omega}((\mathbf{z}_t, y, t) - \epsilon) \frac{\partial \mathbf{z}_t}{\partial \theta}.
$$
这样该梯度就能进行图像编辑，或者直接将$g(\theta)$设置为随机噪声，然后对噪声进行优化，进行图片生成等。但是使用SDS做图像生成，存在的问题是会导致生成的图片趋近于特定的模式，也就是多样性不足。如下图所示：

![image-20240911091742129](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20240911091742129.png)

其中左图是使用SDS优化随机噪声来生成图像，右图是使用Diffusion model正常生成图像，可以发现使用SDS的图像生成是趋近于特定的模式，生成的火烈鸟都有这近乎相同的朝向。

**Delta Denoising Score (DDS)**

要使用SDS做图像编辑，首先需要将$g(\theta)$设置为原图像的像素值或者是VAE encode之后的隐变量$\mathbf{z}$，通过对其进行SDS 迭代使其逼近给定的目标文本$y$。就像SDS在图片生成的时候会趋近于特定的模式，使用其进行编辑的时候也会，甚至还会造成图片过度模糊。作者认为使用SDS的梯度进行优化，该梯度会引入两个主要成分：
$$
\nabla_{\theta} \mathcal{L}_{\text{SDS}}(\mathbf{z}, y, \epsilon, t) := \delta_{\text{text}} + \delta_{\text{bias}},
$$
其中$\delta_{text}$是我们想要的，使用该成分我们可以将源图像向目标图像靠拢，而$\delta_{bias}$则是噪声其会使得编辑后的图片过度模糊同时趋近于特定的模式。因此作者提出了DDS，来解决这一问题：
$$
\mathcal{L}_{\text{DD}}(\phi, \mathbf{z}, y, \hat{\mathbf{z}}, \hat{y}, \epsilon, t) = \left\| \epsilon_{\phi}^{\omega}(\mathbf{z}_t, y, t) - \epsilon_{\phi}^{\omega}(\hat{\mathbf{z}}_t, \hat{y}, t) \right\|_2^2,
$$
$\mathcal{L}_{DD}$中涉及到两个分支，作者将其称为 aligned 和 unaligned分支。aligned分支代表输入的prompt $\hat{y}$作为描述源图像的prompt，与$\mathbf{\hat{z}}$是对齐的，而$y$描述的是我们目标的prompt，因此是unaligned的。由SDS的梯度公式，作者同样省略中间项，可以得到DDS的梯度公式如下：
$$
\nabla_{\theta} \mathcal{L}_{\text{DDS}} = \left( \epsilon_{\phi}^{\omega}(\mathbf{z}_t, y, t) - \epsilon_{\phi}^{\omega}(\hat{\mathbf{z}}_t, \hat{y}, t) \right) \frac{\partial \mathbf{z}}{\partial \theta}.
$$
接下来，作者证明了之所以这么做的原因，作者认为对齐分支由于prompt直接就是描述原视频的，因此其$\delta_{text}$近似等于0，也就是$\nabla_{\theta} \mathcal{L}_{\text{SDS}}(\hat{\mathbf{z}}, \hat{\mathbf{y}}) = \hat{\delta}_{\text{bias}}$。同时相似的prompt中的$\delta_{bias}$也是相似的。

作者还使用一个简单的实验进行了证明：

![image-20240910220404717](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20240910220404717.png)

就算是编辑的prompt和原图像匹配，也就是根本不做编辑。随着迭代次数的增加，图片也会越来越模糊和不清晰。

而aligned-branch的生成梯度是较小的，也就是$\nabla_{\theta} \mathcal{L}_{\text{SDS}}(\hat{\mathbf{z}}, \hat{\mathbf{y}})$其实主要就是噪声梯度。

因此我们可以将DDS拆成两个SDS如下：
$$
\nabla_{\theta} \mathcal{L}_{\text{DDS}} = \nabla_{\theta} \mathcal{L}_{\text{SDS}}(\mathbf{z}, \mathbf{y}) - \nabla_{\theta} \mathcal{L}_{\text{SDS}}(\hat{\mathbf{z}}, \hat{\mathbf{y}}).
$$

这样就能够得到近似即：$\nabla_{\theta} \mathcal{L}_{\text{DDS}} = \delta_{text}$

**Image-to-Image Translation**

作者还训练了一个I2I的模型，使用DDS。这里不详细介绍。

# 实验

主要进行了DDS和SDS的评估