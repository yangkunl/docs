![\<img alt="" data-attachment-key="27ZMG2IN" width="1206" height="453" src="attachments/27ZMG2IN.png" ztype="zimage">](attachments/27ZMG2IN.png)

本文介绍了一种新的骨骼动作识别训练范式，称为SkeletonGCL。SkeletonGCL通过对不同序列之间的学习图进行对比，引导图表示与特定类别相关联，从而提高图卷积网络（GCN）对不同动作的识别能力。作者将SkeletonGCL应用于现有的GCN方法，并在三个基准测试上取得了最先进的性能。\
文章的创新点在于，SkeletonGCL引入了跨序列上下文，利用对比学习来提取丰富的语义信息。通过对比学习来训练GCN，使其能够更好地捕捉到不同骨骼动作之间的差异，从而提高了动作识别性能。此外，文章还提到了改进的空间图卷积网络和注意力增强图卷积长短时记忆网络等模型的应用，并取得了显著的性能提升。\
值得注意的是，本文的一个局限性在于在进行对比学习时，并未考虑不同类别之间的内在关系，未来的工作可以更细致地涉及到跨类关系，从而实现更全面的对比学习方式。\
本文参考的相关研究包括：

1.  Fanfan Ye等人的"Dynamic gcn: Context-enriched topology learning for skeleton-based action recognition"\[4]
2.  Pengfei Zhang等人的"Semantics-guided neural networks for efficient skeleton-based human action recognition"\[4]
3.  Lei Shi等人的"Two-stream adaptive graph convolutional networks for skeleton-based action recognition"\[5]
4.  Chenyang Si等人的"An attention enhanced graph convolutional lstm network for skeleton-based action recognition"\[5]\
    \[3]
