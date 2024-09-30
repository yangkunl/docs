# Evidential Deep Learning

最近文章中有涉及到Evidential Deep Learning，而我一无所知，所以记录一下学习过程（2024/9/29）

## MIT 6.S191关于EDL的部分

![img](D:\blog\docs\others\img\ae43f42dc7ca2f002c937d49092c69b4.png)

***

#### **1.为什么我们需要使用EDL？**

EDL 可以帮助使用者及时发现模型对其输出不自信的情况，判断神经网络是否可信。

 现如今，随着深度学习技术的发展，从实验室走进现实，深刻影响着人们的日常生活，使得不确定性估计越来越重要。通常情况下，我们训练机器学习模型时总是基于以下假设：训练集与测试集都是从**相同的分布**中抽取得到的。而事实往往并不是这样。例如，我们目前所能看到的几乎所有的SOTA算法都是在非常干净的预处理数据集上进行训练的，几乎没有noise或者ambiguity。但是在现实世界中我们面临着很多模型无法完全处理的edge cases。
![img](D:\blog\docs\others\img\04c3309cd23621e1c60f2cfb68531bab.png)

> 比如我们在非常clean的数据集上训练一个分类器来识别“狗”（左上）。但将其应用到现实世界中，当我们向模型展示倒立的狗、甚至从天而降的狗，那么模型的性能就会变得非常差；对于驾驶场景亦是如此。



​	 传统的深度学习模型容易存在偏见（biases），并且常常容易在全新的、超出分布范围（out of distribution, OOD）的数据集上发生错误预测。因此，**我们需要能够快速、可靠地估计模型对所看到的数据以及所预测的输出的不确定性**，以帮助我们对输出的判断。即，模型不仅会给出预测的结果，还会提供它在预测时具有多少证据，我们可以信任它的程度如何。

**总而言之就一句话，实际上我们并不需要模型拥有完美的100%的准确率，但即使准确率略低，但如果我们能够知道什么情况下可以信任模型的输出，那这个模型就是非常强大的。**

***

#### **2.如何估计不确定性？**

在普通的监督学习中，给定数据$x$和标签$y$(分类为离散，回归为连续)，我们的目的其实是学习从数据$x$到标签$y$的一个映射。当给定新的$x$时输出**对应$y$的期望**。但是实际上如果我们只对期望建模，实际上得到的是预测结果的一个**点估计**，而缺乏对这个预测结果的不确定性的估计。因此，我们实际上不只预测一个期望值，**还估计预测值的方差**。如下图所示：

![img](D:\blog\docs\others\img\dd7e636d300a37cc6f4ef7c02e1cf492.png)

#### 3.监督学习的概率建模

**举一个例子：**

我们将一幅图像输 入到神经网络中，并且需要将其分类为“猫”或者“狗”。在这种情况下，每个输出就是图像属于该类别的概率，两个概率的和为1。

![img](D:\blog\docs\others\img\1dc72663924a0b7140ac65fbae04fff5.png)

实际上，输出是一个概率分布。在这个例子中，分布是离散的，因此，网络必须对输出加以限制，使其满足概率分布的定义：第一个约束条件是**概率输出必须大于等于0**，第二个约束条件是**每个类别的概率和为1**，因此在操作中通常使用softmax激活函数来满足这两个条件。我们定义一种特殊的loss function，使我们能够进行优化以学习到理想的分布。我们可以通过最小化预测分布与实际类别分布之间的negative log-likelihood来实现这一点，这个loss也被叫做交叉熵（cross entropy）损失。 那么为什么要选择softmax作为激活函数以及为什么选择negative log-likelihood作为损失函数呢？原因在于，我们假设目标类别label y是从某个似然函数中抽取得到的，在上面这个例子中，**这个似然函数就是由分布参数p所定义的概率分布。其中p的维度是K，即类别数，第i类的概率正好等于第i个参数$p_i$**。

![img](D:\blog\docs\others\img\63ac46b74e231949efe613c24e13c913.png)

对于连续场景情况是类似的。在这种情况下，我们不是在学习某个类别的概率，而是在整个实数轴上学习概率分布，比如自动驾驶项目中方向盘转动的角度。此时，我们不能像在分类问题那样直接输出原始概率，因为这将要求网络产生无限数量的输出。不过，我们可以输出分布的参数，即均值和标准差（或方差），从而定义概率密度函数。

![img](D:\blog\docs\others\img\37573af866f1830e550b0346ea179492.png)

其中，均值是unbounded的，因此不需要约束，而标准差$\sigma$必须严格为正，因此可以使用一个指数激活函数来进行约束。

>标准差是方差的平方根，而方差是数据点与均值之间差异的平方的平均值。因此，方差总是非负的，因为平方的结果不会是负数。标准差作为方差的平方根，自然也必须是非负的。

与分类问题类似，可以使用negative log-likelihood作为损失以优化网络。此时，我们假设标签（或数据点）是从一个正态分布（也称为高斯分布）中抽取出来的，并且这个分布的参数是已知的。 而我们需要训练网络以预测$\mu$和$\sigma$。其实这个过程是很神奇的，因为我们并没有$\mu$和$\sigma$对应的金标准，唯一有的只是真实的label，但我们却可以利用这种建模方式和损失函数的优化，去学习到label的分布，而不是点估计，并得到分布对应的参数$\mu$和$\sigma$（假设是高斯分布）。

![img](D:\blog\docs\others\img\764804f9d1ec33bb98c12750f84b3f35.png)

  总结一下，对于离散的分类问题，我们的targets是固定的K个类别之一，而对于连续的回归问题，targets可以是任意实数。现在，无论对于什么问题，假设它们的label都来自于某种基本的似然函数。对于分类，label来自于分类分布（Categorical distribution）；而对于回归，假设label来自于正态分布（Normal distribution）。而每个这样的似然函数都由一组分布参数所定义（分类问题：分类分布的概率；回归问题：$\mu$和$\sigma$）。之后，为了确保这些都是有效的概率分布，通过激活函数以施加相关的约束。接下来，使用negative log-likelihood作为损失函数来优化整个系统，使得我们能够学习label的分布参数。

![img](D:\blog\docs\others\img\fd824ae3b65ba046b699c16fd166cfb2.png)

但是，需要特别注意的一点是，**通过这种建模方式获得的概率与模型的置信度并不是同一回事**。比如我们有一张既有“猫”又包含“狗”的图像，输入到“猫狗分类器”中。如果这个分类器被很好地训练过，那么它将同时从图像中提取出“猫”和“狗”的特征，在做出最终决定时将会产生混淆。但是这并不意味着我们对答案的不自信（即使不太符合直觉），恰恰相反，我们确实在这个图片中检测到了猫和狗的特征，从而导致输出结果是0.5、0.5。
![img](D:\blog\docs\others\img\44fb4ee7f4664bd6b8ac0e4995f72109.png)

当我们使用同样的网络，但输入一张训练时完全没有接触过的图像（OOD），比如一艘船。此时模型仍然需要输出这张图像是猫的概率与是狗的概率。这是一个由softmax函数训练出来的概率分布，因此**二者之和仍然是1**，这种情况下，输出的概率值就将会变得非常不可靠，也不应该被信任。

#### 4. 不确定性的分类

![img](D:\blog\docs\others\img\ed262ef47e7df4a1a13a88105e817c0c.png)

Uncertainty的分类如上。

 分别是**known knowns（我们知道的、确信的）**，**known unknowns（我们知道自己不知道，有自知之明的）**，**unknown knowns（你不知道，别人知道）**，**unknown unknowns（没人知道）**。举一个机场的例子：

> known knowns：我乘坐的是哪个航班、哪些航班今天会起飞。
>
> known unknowns：航班的具体起飞时间，因为可能会延误。
>
> unknown knowns：别人预定的航班时间。
>
> unknown unknowns：流星撞击飞机跑道。

实际上，理解并实现深度学习模型不确定性的量化是十分困难的，因为模型通常具有数以百万计、十亿记甚至万亿级别的参数量，理解和审视这些模型内部以估计它们何时无法给出正确答案绝对不是一个简单的问题。通常，人们其实并不会训练神经网络来考虑uncertainty，比如上图（右）中的例子，我们可以对黑色的observed data做很好的拟合，但在蓝色区域的预测却失败了。

这里涉及到uncertainty的两种形式，**分别是aleatoric uncertainty（data uncertainty）和epistemic uncertainty（model uncertainty）。**

>还有其他对不确定性的分类，这里不做说明

![img](D:\blog\docs\others\img\5158984ff874d398b6674cae0c161264.png)

**aleatoric uncertainty，偶然不确定性**

>​    指的是数据本身的不确定性，也被称为不可约（irreducible）不确定性，由于数据收集过程中存在的干扰/影响，造成了数据本身固有的噪声，从而无法通过数据的增加以减小此不确定性。减少aleatoric uncertainty的唯一方法就是提高sensor的质量获得更加准确的数据。

**epistemic uncertainty，认知不确定性**

>​    模拟的是预测过程的不确定性。当模型不知道正确答案并且对自己的答案不够自信的时候就会出现这种情况。

![img](D:\blog\docs\others\img\9bd845d04db7d1479c00ac40881dc8ea.png)

#### 5. 估计不确定性的其他方式  

Epistemic uncertainty要比aleatoric uncertainty难以估计得多，因为我们很难意识到自己并不知道某件事（认知受限）。对于传统的确定性神经网络即deterministic neural network，对于一组给定的weights，多次将同一个输入传递给模型将会产生完全相同的输出，从而导致我们无法获得epistemic uncertainty。这也可以体现模型的“过度自信”，毫无自知之明。

**贝叶斯网络**

相反，如果我们网络的weight并不是一个deterministic single number，而是每一个weight都用一个probability distribution来表示。此时，每一个weight都取决于当前时间点对其概率分布的采样结果，那么当我们输入同样的数据时，就会产生（稍微）不太一样的output。拥有这种特性的网络就被称为bayesian neural networks。

​    总结来说，就是对网络权重本身的分布/似然函数进行建模。与model a single number for every weight不同，bayesian NNs try to learn NNs that capture a full distribution over every singe weight，从而也学习了**模型本身的epistemic uncertainty**。

![img](D:\blog\docs\others\img\0f6ce956a87ef2783c7f34be52e56517.png)

Now we can formulate this epistemic uncertainty and formulate this learning of bayesian neural networks as follows.

While deterministic neural networks learn this fixed set of weights ω, bayesian neural networks learn a posterior distribution over the weights. This is a probability of our weights given our input data and our labels (x and y).

Q: 为什么叫做bayesian neural network？

A: Because they formulate this posterior probability of our weights given our data  **using bayes rule**.

​    然而在实际中，这个posterior是intractable的（cannot compute analytically）。这就意味着我们必须求助于采样技术，从而近似/估计这个后验分布。其核心思想是，每次通过采样以得到具有不同参数（权重）的模型进而对模型本身进行评估。



![img](D:\blog\docs\others\img\8aedde111e98f63a685871acaac6f87a.png)

有两种实现方式：

1. Dropout。通常情况下dropout只在训练过程中使用，而在这种情况下，我们在测试时使用dropout，从而对网络中的结点进行采样，以得到不同的权重（不同的模型）。输入相同的数据，这样的过程重复t次，而每一次网络都会有对应的输出（彼此之间不同）。

2. 独立训练t个模型。模型架构一样，但是训练得到的权重不同，使得彼此之间的输出不同（略有差异）。

不管通过哪种方式，道理都是类似的。我们通过t次前向传播（要么使用dropout，要么由t个模型集成），都会产生t个结果，从而得到y的期望与y的方差。**因此，如果这些预测的方差非常大，即输出彼此之间没有良好的一致性，那么模型就具有相对较高的epistemic uncertainty**。



![img](D:\blog\docs\others\img\2f994f99a0ee76cf7aff9303e2e6f5a4.png)

然而，虽然这种基于采样的方法是用来估计epistemic uncertainty常用的方法，但是它们有明显的缺点和局限。首先，因为需要多次进行前向传播，多次训练以得到预测结果（对于集成t个模型的情况更糟，因为需要初始化并且独立训练多个模型），computationally costly，并且模型的保存也占用内存。而uncertainty estimation需要real time on edge devices (e.g. robotics or other mobile devices)，这其实限制了基于采样的方法的应用。

   其实像dropout这种近似方法往往会产生overconfident uncertainty estimates（觉得自己对uncertainty估计得很准确），which may be problematic in safety critical domains where we need calibrated uncertainty estimates。

#### 6. 使用EDL进行不确定性的估计

![img](D:\blog\docs\others\img\89af0aff084c076c08d81381f947a608.png)

回到自动驾驶的例子，我们重新梳理一下uncertainty estimate的整个过程。自动驾驶任务的输出是一个具有$\mu$作为均值、$\sigma$作为标准差的高斯分布，而variance (标准差的平方)其实代表的就是**data uncertainty / aleatoric uncertainty**。

那么所谓的**epistemic uncertainty**是什么呢？我们把每个模型（一共t个）预测的$\mu$和$\sigma^2$画在一个二维平面上（横轴$\mu$，纵轴$\sigma^2$），就得到了这个平面上的t个点，每个点都代表某个高斯分布具有的参数。

此时，我们计算在$\mu$方向上的方差，这个方差代表的就是epistemic uncertainty。直观来讲，如果这些点分散的很开，说明模型不够自信，相反，如果这些点聚在一起，表明模型非常自信。

现在，我们把视角放到这些点上。可以想象，实际上，这些点也满足某个分布，当我们将t无限增大的时候，样本将更加符合如图像中的背景所示的分布。如果我们能直接获得这个分布而不是从中抽取样本（即训练t次），那么我们就可以更好、更快速地理解model / epistemic uncertainty。

![img](D:\blog\docs\others\img\e45b5ce68a16c12b37554f244a93f2f4.png)

我们尝试使用deep learning直接学习这个分布的参数（高阶分布，即分布的分布），这种方法被称为evidential deep learning。

1.上图（左）展示的是具有low uncertainty和high confidence的情况，此时点的分布十分集中；

2.上图（中）是具有high aleatoric / data uncertainty的情况，此时在纵轴方向上（$\sigma^2$）会有非常高的方差；

3.上图（右）是具有high epistemic / model uncertainty的情况，即模型预测的$\mu$值方差很大。

这种高阶分布称为evidential distribution，代表的是分布的分布，那么如何获得evidential distribution呢？

![img](D:\blog\docs\others\img\1e82f78b9b3c9fd64e39106bd027da6c.png)

首先考虑连续场景下的回归问题。label $y$是从某个正态分布（参数$\mu$，$\sigma$）中获得的，而之前我们假设$\mu$和$\sigma$是客观上固定的、已知的、是网络可以预测的，而现在我们假设$\mu$和$\sigma$同样也服从一个分布，同样用概率的方式估计它们。实际上，我们可以通过对这些分布参数施加先验来使其形式化。

​    如上图（左下）所示，我们假设$\mu$服从高斯分布，$\sigma^2$服从inverse gamma分布，那么$\mu$和$\sigma$的联合分布服从Normal Inverse Gamma（具有四个参数）。这就是evidential distribution，如上图（中）所示。当我们从这个分布中采样时，实际上获得的是某个$\mu$和$\sigma$的组合，从而定义了y所服从的高斯分布（同时也是likelihood function），如上图（右）所示。

![img](D:\blog\docs\others\img\68f17fb2fb88c52211badb554f0ca00f.png)

对于离散的分类问题也是类似的，类别y服从的分布（categorical）的参数为p，而我们可以进一步对p施加先验，假设p服从Dirichlet分布。Dirichlet分布由一组concentration parameters参数化（$\alpha$，K-dimensional）。我们可以从这个Dirichlet distribution中采样得到分类损失函数中的categorical loss function。    

以上图（右）为例，我们有三个可能的类别（K=3），那么the mass of Dirichlet distribution将会完全存分布于这个三角形单纯形（simplex）内，在三角形内部的任意点进行采样将会产生对应的“brand new” categorical probability distribution。

假设我们从这个simplex的中心采样，就得到了三个类别是等概率的情况，而simplex的每个角则代表着某个类别的预测概率为1（其它类别预测概率为0），从中间采样就代表三个类别等概率。simplex中颜色的深浅代表着质量的分布情况。

需要注意的是我们的网络将尝试预测任何给定输入的分布，因此这个分布可能会变化，而我们对分类似然函数的采样方式也会因此发生变化。即，对于每个不同的输入，这个分布也是不同的，会变化，然后我们从这个分布中采样，得到y服从的分布，也就是结果。



![img](D:\blog\docs\others\img\e330b3860dc6c361003a7d468f058759.png)

**总结一下**：

在回归问题中，目标值是连续的。假设目标值y是从由$\mu$和$\sigma$数化的正态分布中抽取的，然后我们对这些似然参数有了更高阶的证据分布，这个分布就是normal inverse gamma distribution，分布的参数有四个；

在分类问题中，目标值是离散的K个类别。假设观察到某个特定类别标签y的likelihood来自于一个具有类别概率p的categorical distribution，而这个 p又是由**高阶的evidential Dirichlet distribution参数化**而来的，这个分布的参数是α。

那么，我们有那么多不同的分布，为什么要用NIG与Dirichlet distribution作为先验呢？这涉及到所谓的conjugate prior。

Conjugate prior：给定似然，使得先验与后验具有相同形式（属于同一分布族）的先验。

如果选择conjugate prior作为evidential distribution，在计算损失的时候就会更加容易处理。即，后验有解析解，不需要进行复杂的积分运算（这个积分通常是intractable的，需要使用数值计算方法/近似推断来计算），使得整个过程更加可行和高效。





![img](D:\blog\docs\others\img\8505149599c941b499a384c24263f5c3.png)

**训练**

训练网络以output the parameters of these higher order evidential distributions。

 对于回归，网络输出$\gamma$、$\epsilon$、$\alpha$和$\beta$；而在分类问题中，网络预测一个$\alpha$向量（K维，K是类别数）。**一旦我们有了这些参数，就能直接计算aleatoric and epistemic uncertainty的估计值（determined by the resulting  distributions over these likelihood parameters）**。

训练时，通过目标函数的优化以获得最佳的参数，目标函数中有两项：

1. **最大化模型拟合的准确度**，使得对具有参数m的evidential distribution采样得到的likelihood最大化；

2. 一个正则项，即最小化incorrect evidence，将所有的evidential distribution拟合到数据中，确保它们尽可能吻合（**一个数据对应一个m**）。

此时，当我们看到新数据时，就可以输出higher order evidential distribution的参数m（m随着输入的变化而不同，是网络的输出），并从这个evidential distribution中采样得到likelihood function，**进一步得到预测结果，同时还可以使用m计算uncertainty**。

#### 7.参考文献

[MIT 6.S191: Evidential Deep Learning学习笔记-CSDN博客](https://blog.csdn.net/Rad1ant_up/article/details/135007454)