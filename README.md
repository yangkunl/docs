<!-- 在你的文档中添加滑动开关的 HTML 结构 -->
<label class="switch">
  <input type="checkbox" id="modeSwitch" onclick="toggleMode()">
  <span class="slider round"></span>
  <span class="label-text">模式:crescent_moon:/:sun_with_face:</span>
</label>

# <big>论文阅读总览</big>

# 本周新读文献(11月30日)

| 文章名称                                                     | PDF                                                          | 代码                                           | 笔记                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------- | -------------------------- |
| TA2N: Two-Stage Action Alignment Network for Few-Shot Action Recognition | [AAAI2022](https://arxiv.org/pdf/2107.04782.pdf)             | [GitHub](https://github.com/R00Kie-Liu/TA2N)   | [TA2N](md/TA2N.md)         |
| M3Net: Multi-view Encoding, Matching, and Fusion for Few-shot Fine-grained Action Recognition | [MM2023](http://arxiv.org/abs/2308.03063)                    | 无                                             | [M3Net](md/M3Net.md)       |
| FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning | [NeurIPS2023](http://arxiv.org/abs/2310.15105)               | [GitHub](https://github.com/skingorz/FD-Align) | [FD-Align](md/FD-Align.md) |
| Boosting Few-shot Action Recognition with Graph-guided Hybrid Matching | [ICCV2023](http://openaccess.thecvf.com/content/ICCV2023/html/Xing_Boosting_Few-shot_Action_Recognition_with_Graph-guided_Hybrid_Matching_ICCV_2023_paper.html) | 无                                             | [GgHM](md/GgHM.md)         |
| Spatio-temporal Relation Modeling for Few-shot Action Recognition | [CVPR2022](http://arxiv.org/abs/2112.05132)                  | [GitHub](https://github.com/Anirudh257/strm)   | [STRM](md/STRM.md)         |
| Sequential Modeling Enables Scalable Learning for Large Vision Models | [arXiv](http://arxiv.org/abs/2312.00785)                     | 无                                             | [LVM](md/LVM.md)           |
| On the Importance of Spatial Relations for Few-shot Action Recognition | [MM2023](http://arxiv.org/abs/2308.07119)                    | 无                                             | [SA-CT](md/SA-CT.md)       |

# 文献仓库

- **视频理解(动作识别)**

| **文章名称**                                                 | PDF                                                          | 代码                                                         | 笔记                             |    阅读时间    | 复现笔记 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------- | :------------: | -------- |
| CLIP-guided Prototype Modulating for Few-shot Action Recognition | [IJCV2023](http://arxiv.org/abs/2303.02982)                  | [GitHub](https://github.com/alibaba-mmai-research/CLIP-FSAR) | [CLIP-FSAR](md/CLIP-FSAR.md)     | 2023.11.1~11.6 |          |
| Revisiting the Spatial and Temporal Modeling for Few-Shot Action Recognition | [AAAI2023](https://ojs.aaai.org/index.php/AAAI/article/view/25403) | 无                                                           | [SloshNet](md/SloshNet.md)       | 2023.11.1~11.6 |          |
| Learning Discriminative Representations for Skeleton Based Action Recognition | [CVPR2023]([openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Learning_Discriminative_Representations_for_Skeleton_Based_Action_Recognition_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Learning_Discriminative_Representations_for_Skeleton_Based_Action_Recognition_CVPR_2023_paper.pdf)) | [GitHub](https://github.com/zhysora/FR-Head)                 | [FR_Head](md/FR_Head.md)         | 2023.11.1~11.6 |          |
| Graph Contrastive Learning for Skeleton-based Action Recognition | [ICLR2023](http://arxiv.org/abs/2301.10900)                  | [GitHub](https://github.com/OliverHxh/SkeletonGCL)           | [SkeletonGCL](md/SkeletonGCL.md) | 2023.11.1~11.6 |          |
| AIM: Adapting Image Models for Efficient Video Action Recognition | [ICLR2023](http://arxiv.org/abs/2302.03024)                  | [GitHub](https://adapt-image-models.github.io/)              | [AIM](md/AIM.md)                 |   2023.11.10   |          |
| Temporal-Relational CrossTransformers for Few-Shot Action Recognition | [CVPR2021](http://arxiv.org/abs/2101.06184)                  | [GitHub](https://github.com/tobyperrett/TRX)                 | [TRX](md/TRX.md)                 |   2023.11.11   |          |
| Few-shot Action Recognition via Intra- and Inter-Video Information Maximization | [arxiv](https://arxiv.org/abs/2305.06114)                    | 无                                                           | [VIM](md/VIM.md)                 |   2023.11.15   |          |
| Hybrid Relation Guided Set Matching for Few-shot Action Recognition | [CVPR2022](http://arxiv.org/abs/2204.13423)                  | [GitHub](https://hyrsm-cvpr2022.github.io/)                  | [HyRSM](md/HyRSM.md)             |   2023.11.17   |          |
| TA2N: Two-Stage Action Alignment Network for Few-Shot Action Recognition | [AAAI2022](https://arxiv.org/pdf/2107.04782.pdf)             | [GitHub](https://github.com/R00Kie-Liu/TA2N)                 | [TA2N](md/TA2N.md)               |   2023.11.22   |          |
| M3Net: Multi-view Encoding, Matching, and Fusion for Few-shot Fine-grained Action Recognition | [MM2023](http://arxiv.org/abs/2308.03063)                    | 无                                                           | [M3Net](md/M3Net.md)             |   2023.11.24   |          |
| Boosting Few-shot Action Recognition with Graph-guided Hybrid Matching | [ICCV2023](http://openaccess.thecvf.com/content/ICCV2023/html/Xing_Boosting_Few-shot_Action_Recognition_with_Graph-guided_Hybrid_Matching_ICCV_2023_paper.html) | 无                                                           | [GgHM](md/GgHM.md)               |   2023.11.30   |          |
| Spatio-temporal Relation Modeling for Few-shot Action Recognition | [CVPR2022](http://arxiv.org/abs/2112.05132)                  | [GitHub](https://github.com/Anirudh257/strm)                 | [STRM](md/STRM.md)               |   2023.12.2    |          |

- **小样本学习**

| 文章名称                                                     | PDF                                                          | 代码                                                         | 笔记                         | 阅读时间       | 复现笔记 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------- | -------------- | -------- |
| CLIP-guided Prototype Modulating for Few-shot Action Recognition | [IJCV2023](http://arxiv.org/abs/2303.02982)                  | [GitHub](https://github.com/alibaba-mmai-research/CLIP-FSAR) | [CLIP-FSAR](md/CLIP-FSAR.md) | 2023.11.1~11.6 |          |
| Revisiting the Spatial and Temporal Modeling for Few-Shot Action Recognition | [AAAI2023](https://ojs.aaai.org/index.php/AAAI/article/view/25403) | 无                                                           | [SloshNet](md/SloshNet.md)   | 2023.11.1~11.6 |          |
| Few-shot Action Recognition via Intra- and Inter-Video Information Maximization | [arxiv](https://arxiv.org/abs/2305.06114)                    |                                                              | [VIM](md/VIM.md)             | 2023.11.15     |          |
| Hybrid Relation Guided Set Matching for Few-shot Action Recognition | [CVPR2022](http://arxiv.org/abs/2204.13423)                  | [GitHub](https://hyrsm-cvpr2022.github.io/)                  | [HyRSM](md/HyRSM.md)         | 2023.11.17     |          |
| TA2N: Two-Stage Action Alignment Network for Few-Shot Action Recognition | [AAAI2022](https://arxiv.org/pdf/2107.04782.pdf)             | [GitHub](https://github.com/R00Kie-Liu/TA2N)                 | [TA2N](md/TA2N.md)           | 2023.11.22     |          |
| FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning | [NeurIPS2023](http://arxiv.org/abs/2310.15105)               | [GitHub](https://github.com/skingorz/FD-Align)               | [FD-Align](md/FD-ALign.md)   | 2023.11.25     |          |
| Boosting Few-shot Action Recognition with Graph-guided Hybrid Matching | [ICCV2023](http://openaccess.thecvf.com/content/ICCV2023/html/Xing_Boosting_Few-shot_Action_Recognition_with_Graph-guided_Hybrid_Matching_ICCV_2023_paper.html) | 无                                                           | [GgHM](md/GgHM.md)           | 2023.11.30     |          |
| Spatio-temporal Relation Modeling for Few-shot Action Recognition | [CVPR2022](http://arxiv.org/abs/2112.05132)                  | [GitHub](https://github.com/Anirudh257/strm)                 | [STRM](md/STRM.md)           | 2023.12.2      |          |

- **多模态**

| 文章名称                                                     | PDF                                      | 代码                                                         | 笔记                         | 阅读时间       | 复现笔记 |
| ------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------ | ---------------------------- | -------------- | -------- |
| CLIP-guided Prototype Modulating for Few-shot Action Recognition | [arxiv](http://arxiv.org/abs/2303.02982) | [GitHub](https://github.com/alibaba-mmai-research/CLIP-FSAR) | [CLIP-FSAR](md/CLIP-FSAR.md) | 2023.11.1~11.6 |          |
|                                                              |                                          |                                                              |                              |                |          |
|                                                              |                                          |                                                              |                              |                |          |

- **对比学习**

| 文章名称                                                     | PDF                                         | 代码                                               | 笔记                             | 阅读时间       | 复现笔记 |
| ------------------------------------------------------------ | ------------------------------------------- | -------------------------------------------------- | -------------------------------- | -------------- | -------- |
| Graph Contrastive Learning for Skeleton-based Action Recognition | [ICLR2023](http://arxiv.org/abs/2301.10900) | [GitHub](https://github.com/OliverHxh/SkeletonGCL) | [SkeletonGCL](md/SkeletonGCL.md) | 2023.11.1~11.6 |          |

- **自监督**

| 文章名称                                                     | PDF                                            | 代码                                          | 笔记                       | 阅读时间  | 复现笔记 |
| ------------------------------------------------------------ | ---------------------------------------------- | --------------------------------------------- | -------------------------- | --------- | -------- |
| VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training | [NeurIPS2022](http://arxiv.org/abs/2203.12602) | [Github](https://github.com/MCG-NJU/VideoMAE) | [VideoMAE](md/VideoMAE.md) | 2023.11.9 |          |
| Sequential Modeling Enables Scalable Learning for Large Vision Models | [arXiv](http://arxiv.org/abs/2312.00785)       | 无                                            | [LVM](md/LVM.md)           | 2023.12.5 |          |

