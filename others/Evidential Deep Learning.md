# Evidential Deep Learning

之前主流量化不确定性的方法是Deep ensembling 和 Bayesian 神经网络

Deep ensembling 做量化不确定性的方法可能是将多个模型的结果ensembling，都相似则不确定性低，否则不确定性高

EDL的理论基础是主观逻辑理论，

主管逻辑理论：主观逻辑中意见的四个主要组成部分：信念（Belief）、不信（Disbelief）、不确定性（Uncertainty）和基础率（Base Rate）。

通过建模一个狄利克雷分布，当不确定的mass 逐渐增加的时候会更倾向于一个预先设定的分布，原因就是不确定性增加，信息量减少？



EDL可以使用一个神经网络作为一个证据收集器， 



证据收集流程：

![image-20240923075857211](C:\Users\19475\AppData\Roaming\Typora\typora-user-images\image-20240923075857211.png)

不确定性：

这种情况通常发生在模型根据特定样本的不同部分或特征给出不一致的预测时。即使模型可能对个别特征有足够的信息，但由于这些信息之间的冲突，整体的不确定性也会增加。不一致性并不直接对应于不确定性或认识不确定性，但它可以被视为一种复杂的不确定性形式，包含这两种不确定性的要素。文献[29]提供了 EDL 框架下的不确定性表述，其细节将在第 4.3 节中介绍。