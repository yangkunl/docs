## 1.einops包(对高维数据的处理方法)

- einops.layers.torch 中的Rearrange

  **用于搭建网络结构时对张量进行"隐式"的处理**

  例如:

  ```python
  class PatchEmbedding(nn.Module):
      def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
          self.patch_size = patch_size
          super().__init__()
          self.projection = nn.Sequential(
              # using a conv layer instead of a linear one -> performance gains
              nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
              Rearrange('b e (h) (w) -> b (h w) e'),
          )
  ```

  这里的**Rearrange('b e (h) (w) -> b (h w) e')**，表示将4维张量转换为3维，且原来的最后两维合并为一维：（16，512，4，16）->（16，64，512）,这样只要知道初始的张量维度就可以操作注释来对其维度进行重排

- einops 中的rearrange

  用于对张量的显示处理,是一个函数,例如

  ```python
  rearrange(images, 'b h w c -> b (h w) c')
  ```

  将4维张量转换为3维,同样的只要知道初始维度,就可以操作注释对其进行重排

  ```python
  image = torch.randn(1,2,3,2)  # torch.Size([1,2,3,2]) 
   
  out = rearrange(image, 'b c h w -> b (c h w)', c=2,h=3,w=2) # torch.Size([1,12])
  # h,w的值更改
  err1 = rearrange(image, 'b c h w -> b (c h w)', c=2,h=2,w=3) # 报错
  ```

- repeat::即将tensor中的某一维度进行重复，以扩充该维度数量

  ```python
  B = 16
  cls_token = torch.randn(1, 1, emb_size)
  cls_tokens = repeat(cls_token, '() n e -> b n e', b=B)#维度为1的时候可用（）代替
  ```

  将(1,1,emb_size)的张量处理为（B,1,emb_size）

  ```python
  R = 16
  a = torch.randn(2,3,4)
  b = repeat(a, 'b n e -> (r b) n e', r = R)
  #(2R, 3, 4)
  c = repeat(a, 'b n e -> b (r n) e', r = R)
  #(2, 3R, 4)
   
  #错误用法:
  d = repeat(a, 'b n e -> c n e', c = 2R)
  ```

- Reduce 和 reduce同理

  ```python
  x = torch.randn(100, 32, 64)
  # perform max-reduction on the first axis:
  y0 = reduce(x, 't b c -> b c', 'max') #(32, 64)
   
  #指定h2,w2，相当于指定池化核的大小
  x = torch.randn(10, 512, 30, 40)
  # 2d max-pooling with kernel size = 2 * 2 
  y1 = reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2)
  #(10, 512, 15, 20)
   
  # go back to the original height and width
  y2 = rearrange(y1, 'b (c h2 w2) h1 w1 -> b c (h1 h2) (w1 w2)', h2=2, w2=2)
  #(10, 128, 30, 40)
  #指定h1,w1，相当于指定池化后张量的大小
  # 2d max-pooling to 12 * 16 grid:
  y3 = reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h1=12, w1=16)
  #(10, 512, 12, 16)
   
  # 2d average-pooling to 12 * 16 grid:
  y4 = (reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'mean', h1=12, w1=16)
  #(10, 512, 12, 16)
   
  # Global average pooling
  y5 = reduce(x, 'b c h w -> b c', 'mean')
  #(10, 512)
  ```

  

