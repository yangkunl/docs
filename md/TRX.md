<link rel="stylesheet" href="custom.css">

**Temporal-Relational CrossTransformers for Few-Shot Action Recognition**

# 模型结构图

![\<img alt="" data-attachment-key="PGBUM4ZJ" width="1581" height="495" src="attachments/PGBUM4ZJ.png" ztype="zimage">](attachments/PGBUM4ZJ.png)

使用了crossTransformer注意力机制来构建类原型,而不是使用类平均值或者单一最佳匹配.

## 创新点

*   将查询集(query set)与支持集(support set),通过注意力机制进行匹配,进而构建了类原型,提出了**TRX模块**
*   通过选帧(2帧或3帧),将所有帧数组合进行了遍历计算**TRX**,同时将遍历结果进行了平均
*   将不同选帧进行了组合

## 方法

* 首先是跟TSN一样,每个视频均匀采样8帧,其将每个视频的8帧进行抽样,例如8帧中选两帧,那么就有$C^{2}_8$种选法,那么所有可能的采样集合$\left.\Pi=\left\{\left(n_1, n_2\right) \in \mathbb{N}^2: 1 \leq n_1<n_2 \leq F\right)\right\}$,其还抽样了三帧,将这两种方式的结果进行了组合(平均)

  ```python
  all_logits = [t(context_features, context_labels, target_features)['logits'] for t in self.transformers]
            all_logits = torch.stack(all_logits, dim=-1)
            sample_logits = all_logits 
            sample_logits = torch.mean(sample_logits, dim=[-1])
  ```

*    其使用以下公式得到query的表示,support 同理.$p_1,p_2$ 代表抽样的两帧图片

    $Q_p=\left[\Phi\left(q_{p_1}\right)+\operatorname{PE}\left(p_1\right), \Phi\left(q_{p_2}\right)+\operatorname{PE}\left(p_2\right)\right] \in \mathbb{R}^{2 \times D}$,$D$ 为卷积网络提取特征的维度,这里resnet50是1024(应该是加载的预训练在Imagenet上的),由于将两帧拼接起来然后做了一个reshape,所以实际维度为2048,采样三帧的同理为3064.

* $S^c$在这里做了一个堆叠,现在尺寸变为了 视频数X帧数采样总数X(2048 or 3064), $\mathbf{S}^{c}=\left\{S_{k m}^{c}:(1 \leq k \leq K) \wedge(m \in \Pi)\right\}$

  ```python
   s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
   q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]#tuple为采样两帧的组合,格式为tensor([0,1])
   support_set = torch.stack(s, dim=-2)
   queries = torch.stack(q, dim=-2)
  ```

* 然后是通过两个线性层计算K,V,之后再计算其不同类别的分数(**拿query的K与support的K进行计算分数**),同时对获得的分数做了一个$Softmax$ 操作

  

  ```python
  class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)
  ```

*   将计算的分数与support的value相乘,得到类别原型,在计算query的V与类别原型计算距离得到最终的query的类别

    $T\left(Q_{p}, \mathbf{S}^{c}\right)=\left\|\mathbf{t}_{p}^{c}-\mathbf{u}_{p}\right\|$

*   注意这只是所有两帧(三帧)组合的一种,因此最后对**所有组合进行了平均**

    $T\left(\mathbf{Q}, \mathbf{S}^{c}\right)=\frac{1}{|\Pi|} \sum_{n \in \Pi} T\left(Q_{p}, \mathbf{S}^{c}\right)$

## 实验设置

*   数据集使用Kinetic,SSv2,HMDB和UCF,其中Kinetic和SSv2使用的是CMN这篇文章的分割方式,HMDB和UCF使用的是ARN的分割方式
*   每次只迭代一个任务,每16次**更新一次梯度**.
