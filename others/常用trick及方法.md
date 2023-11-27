# Tranformer的位置编码

##一 .为什么要进行位置编码

由于输入self-attention模型的是一整排的tokens,对于人来说我们很容易知道tokens的位置信息,比如:

​	(1) 绝对位置信息.a1是第一个token, a2是第二个token

​	(2)相对位置信息. a2在a1的后面一位,a4在a2的后面两位

​	(3)不同位置间的距离. a1和a3差两个位置,a1和a4差三个位置

但是对于self-attention来说,是无法分辨的信息,因为self-attention的运算是无向的.因此要想办法把tokens的位置信息,喂给模型

## 二. 构造位置编码的方法/演变历程

## 1. 用整型值标记位置

很自然的想法就是,给第一个token标记1,给第二个token标记2,以此类推,但是其有以下几个问题:

​	(1)模型可能遇见比训练时所用的序列更长的序列.不利于模型的泛化.

​	(2)模型的位置表示是无界的,随着序列长度的增加,位置值会越来越大

## 2.用[0,1]范围标记位置

为了解决整型值带来的问题，可以考虑将位置值的范围限制在[0, 1]之内，其中，0表示第一个token，1表示最后一个token。比如有3个token，那么位置信息就表示成[0, 0.5, 1]；若有四个token，位置信息就表示成[0, 0.33, 0.69, 1]。
但这样产生的问题是，当序列长度不同时，**token间的相对距离**是不一样的。例如在序列长度为3时，token间的相对距离为0.5；在序列长度为4时，token间的相对距离就变为0.33。

因此，我们需要这样一种位置表示方式，满足于：
（1）它能用来表示一个token在序列中的绝对位置
（2）在序列长度不同的情况下，不同序列中token的相对位置/距离也要保持一致
（3）可以用来表示模型在训练过程中从来没有看到过的句子长度。

## 3.用二进制向量标记位置

考虑到位置信息作用在input embedding上，因此比起用单一的值，更好的方案是用一个和input embedding维度一样的向量来表示位置。这时我们就很容易想到二进制编码。如下图，假设d_model = 3，那么我们的位置向量可以表示成：

![img](img\v2-60d7a554b442eebe967d8e07eb941039_r.jpg)

这下所有的值都是有界的（位于0，1之间），且transformer中的d_model本来就足够大，基本可以把我们要的每一个位置都编码出来了。

但是这种编码方式也存在问题：这样编码出来的位置向量，处在一个离散的空间中，**不同位置间的变化是不连续的**。假设d_model = 2，我们有4个位置需要编码，这四个位置向量可以表示成[0,0],[0,1],[1,0],[1,1]。我们把它的位置向量空间做出来：

![img](img\v2-fd65171f6f594aa62cf14d64ce8d043e_1440w.webp)

如果我们能把离散空间（黑色的线）转换到连续空间（蓝色的线），那么我们就能解决位置距离不连续的问题。同时，我们不仅能用位置向量表示整型，我们还可以用位置向量来表示浮点型。

## 4.用周期函数(sin)来表示位置

回想一下，现在我们需要一个有界又连续的函数，最简单的，正弦函数sin就可以满足这一点。我们可以考虑把位置向量当中的每一个元素都用一个sin函数来表示，则第t个token的位置向量可以表示为：

$P E_t=\left[\sin \left(\frac{1}{2^0} t\right), \sin \left(\frac{1}{2^1} t\right) \ldots, \sin \left(\frac{1}{2^{i-1}} t\right), \ldots, \sin \left(\frac{1}{2^{d_{\text {model }}-1}} t\right)\right]$

结合下图，来理解一下这样设计的含义。图中每一行表示一个 $P E_t$，每一列表示 $P E_t$ 中的第$i$个元素。旋钮用于调整精度，越往右边的旋钮，需要调整的精度越大，因此指针移动的步伐越小。每一排的旋钮都在上一排的基础上进行调整（函数中t的作用）。通过频率 $\frac{1}{2^{i−1}}$ 来控制sin函数的波长，频率不断减小，则波长不断变大，此时sin函数对t的变动越不敏感，以此来达到越向右的旋钮，指针移动步伐越小的目的。 这也类似于二进制编码，每一位上都是0和1的交互，越往低位走（越往左边走），交互的频率越慢。

为了避免这种情况，我们尽量将函数的波长拉长。一种简单的解决办法是同一把所有的频率都设成一个非常小的值。因此在transformer的论文中，采用了 $\frac{1}{10000^{i/(d_{model}−1)}}$这个频率（这里i其实不是表示第i个位置，但是大致意思差不多，下面会细说）

![image-20231113091227158](img\image-20231113091227158.png)

其中$w_i= \frac{1}{10000^{i/(d_{model}−1)}}$

## 5.用sin和cos交替来表示位置

目前为止，我们的位置向量实现了如下功能：
（1）每个token的向量唯一（每个sin函数的频率足够小）
（2）位置向量的值是有界的，且位于连续空间中。模型在处理位置向量时更容易泛化，即更好处理长度和训练数据分布不一致的序列（sin函数本身的性质）

那现在我们对位置向量再提出一个要求，**不同的位置向量是可以通过线性转换得到的**。这样，我们不仅能表示一个token的绝对位置，还可以表示一个token的相对位置，即我们想要：

![image-20231113091550410](img\image-20231113091550410.png)

这里，T表示一个线性变换矩阵。观察这个目标式子，联想到在向量空间中一种常用的线形变换——旋转。在这里，我们将t想象为一个角度，那么 $△t$就是其旋转的角度，则上面的式子可以进一步写成：

$\left(\begin{array}{c}\sin (t+\triangle t) \\ \cos ((t+\triangle t)\end{array}\right)=\left(\begin{array}{cc}\cos \triangle t & \sin \triangle t \\ -\sin \triangle t & \cos \triangle t\end{array}\right)\left(\begin{array}{c}\sin t \\ \cos t\end{array}\right)$

在这样的表示下，我们可以很容易用一个线性变换，把 $PE_t$ 转变为 $PE_{t+△t}$

## 三.常见位置编码的代码

### 1.可学习的绝对位置编码

```python
class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.is_absolute = True
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings))

    def forward(self, x):
        """
        return (b l d) / (b h l d)
        """
        position_ids = self.position_ids[:x.size(-2)]

        if x.dim() == 3:
            return x + self.embeddings(position_ids)[None, :, :]

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> b l (h d)')
            x = x + self.embeddings(position_ids)[None, :, :]
            x = rearrange(x, 'b l (h d) -> b h l d', h=h)
            return x
```

### 2.三角式绝对位置编码

```python
class FixedAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, position_embedding_type):
        super().__init__()

        self.position_embedding_type = position_embedding_type
        self.is_absolute = True

        inv_freq = 1. / (10000 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size))
        position = torch.arange(max_position_embeddings, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('embeddings', embeddings)

    def forward_fixed(self, x):
        """
        return (b l d)
        """
        return x + self.embeddings[None, :x.size(1), :]

    def forward_rope(self, x):
        """
        return (b l d)
        """
        embeddings = self.embeddings[None, :x.size(1), :] # b l d
        embeddings = rearrange(embeddings, 'b l (j d) -> b l j d', j=2)
        sin, cos = embeddings.unbind(dim=-2) # b l d//2
        sin, cos = map(lambda t: repeat(t, '... d -> ... (d 2)'), (sin, cos)) # b l d
        return x * cos + self.rotate_every_two(x) * sin

    @staticmethod
    def rotate_every_two(x):
        x = rearrange(x, '... (d j) -> ... d j', j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d j -> ... (d j)')

    def _forward(self, x):
        if self.position_embedding_type == 'fixed':
            return self.forward_fixed(x)

        elif self.position_embedding_type == 'rope':
            return self.forward_rope(x)

    def forward(self, x):
        if x.dim() == 3:
            return self._forward(x)

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> (b h) l d')
            x = self._forward(x)
            x = rearrange(x, '(b h) l d -> b h l d', h=h)
            return x 
```

### **相对位置编码**

```python
class RelativePositionEmbedding(nn.Module):
    def __init__(self, 
                 relative_attention_num_buckets, num_attention_heads, 
                 hidden_size, position_embedding_type):

        super().__init__()

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.position_embedding_type = position_embedding_type
        self.num_attention_heads = num_attention_heads
        self.is_absolute = False

        if position_embedding_type == 'bias':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, num_attention_heads)

        elif position_embedding_type == 'contextual(1)':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, hidden_size)
            self.to_r = nn.Linear(hidden_size, hidden_size, bias=False)

        elif position_embedding_type == 'contextual(2)':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, hidden_size)

    def compute_bias(self, q, k, to_q=None, to_k=None):
        """
        q, k: [b h l d]
        return [b h l l]
        """
        h = self.num_attention_heads
        query_position = torch.arange(q.size(2), dtype=torch.long, device=self.embeddings.weight.device)[:, None]
        key_position   = torch.arange(k.size(2), dtype=torch.long, device=self.embeddings.weight.device)[None, :]

        relative_position = query_position - key_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets
        )

        if self.position_embedding_type == 'bias':
            bias = self.embeddings(relative_position_bucket)
            bias = rearrange(bias, 'm n h -> 1 h m n')

        elif self.position_embedding_type == 'contextual(1)':
            r = self.embeddings(relative_position_bucket)
            r = self.to_r(r)
            r = rearrange(r, 'm n (h d) -> h m n d', h=h)

            bias = torch.einsum('b h m d, h m n d -> b h m n', q, r)

        elif self.position_embedding_type == 'contextual(2)':
            r = self.embeddings(relative_position_bucket)

            kr = to_k(r)
            qr = to_q(r)

            kr = rearrange(kr, 'm n (h d) -> h m n d', h=h)
            qr = rearrange(qr, 'm n (h d) -> h m n d', h=h)

            bias1 = torch.einsum('b h m d, h m n d -> b h m n', q, kr)
            bias2 = torch.einsum('b h n d, h m n d -> b h m n', k, qr)

            bias = bias1 + bias2

        return bias

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance=128):
        """
        relative_position: [m n]
        """

        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
```

#### Embedding

```python
class Embedddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.embedding_size, config.hidden_size)
        
        if config.position_embedding_type == 'learnable':
            self.position_embeddings = LearnableAbsolutePositionEmbedding(
                max_position_embeddings=config.max_position_embeddings, 
                hidden_size=config.hidden_size
            )
        
        elif config.position_embedding_type in ('fixed', 'rope'):
            self.position_embeddings = FixedAbsolutePositionEmbedding(
                max_position_embeddings=config.max_position_embeddings,
                hidden_size=config.hidden_size,
                position_embedding_type=config.position_embedding_type
            )

    def forward(self, input_ids):
        embeds = self.word_embeddings(input_ids)
        embeds = self.dropout(embeds)
        embeds = self.dense(embeds)

        if hasattr(self, 'position_embeddings'):
            embeds = self.position_embeddings(embeds)

        return embeds
```