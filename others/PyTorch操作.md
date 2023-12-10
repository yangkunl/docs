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

  

## 2.PyTorch 中交换tensor维度的几种方法

除了上述的einops包,其在高维张量交换维度时特别有优势,原因是其可以显示的表示矩阵维度的操作。但需要先`import`这个包,同时要使用其api,可能会增加不知道这个包的人的代码阅读难度。所以这里介绍其他基于torch api的交换维度的方法。

### 1.permute方法

`permute`方法可以用来重新排列张量的维度。我们需要提供一个维度的排列顺序,它返回一个新的张量(注意其是返回一个新的张量,因此需要对原张量的赋值操作),该张量的维度安装我们所提供的顺序进行排列。

```python
import torch
x = torch.randn(3,4,5) #create a tensor:shape(3,4,5)
y = x.permute(2,0,1) #exchange dim , y:shape(5,3,4)
```

### 2.transpose 方法

`transpose`方法用于交换两个维度的顺序。可以通过传递两个维度的索引来指定交换的维度。

```python
import torch
x = torch.randn(3, 4, 5) #create a tensor
y = x.transpose(0, 2) #y:shape(5, 4, 3)
```

### 3.transpose函数

`torch.transpose`函数也可以用来交换张量的维度,类似于`transpose`方法

```python
import torch
x = torch.randn(3, 4, 5)
y = torch.transpose(x, 0, 2)
```

### 4.view方法

`view`方法可以用于重新排列张量的维度,但前提是新的形状能够正确容纳原始张量的元素个数

```python
import torch
x = torch.randn(3, 4, 5)
y = x.view(5, 3, 4)
```

### 5.reshape方法

`reshape` 方法也可以用于重新排列张量的维度，类似于 `view` 方法。

```python
import torch

# 创建一个示例张量
x = torch.randn(3, 4, 5)

# 使用reshape重新排列维度
y = x.reshape(5, 3, 4)
```

这些方法中，`permute` 和 `transpose` 通常用于更灵活地交换多个维度，而 `view` 和 `reshape` 主要用于改变张量的形状。选择合适的方法取决于你的具体需求和张量的形状。

## 3.Pytorch-Lightning

以下教程来自知乎答主[Takanashi](https://www.zhihu.com/people/miracleyoo)

PyTorch-Lightning ,不断地在工程代码上花更多的时间,同时代码也越来越长(如果想要添加例如TensorBoard支持, Early Stop, LR Scheduler, 分布式训练,快速测试等),代码无可避免的越来越长,但是PyTorch-Lighting难学,其[官网教程](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html).

### 1.核心

- Pytorch-lighting的特点是把模型和系统分开来看。模型像是Resnet18, RNN之类的纯模型,而系统定义了一组模型如何相互交互,如GAN(生成器与判别器网络)、Seq2Seq（Encoder与Decoder网络）和Bert。同时，有时候问题只涉及一个模型，那么这个系统则可以是一个通用的系统，用于描述模型如何使用，并可以被复用到很多其他项目。

- Pytorch-Lighting 的核心设计思想是“自给自足”。每个网络也同时包含了如何训练、如何测试、优化器定义等内容。

  <img src="img\v2-9a288ab9e69d0365139102e2d722ff8d_r.jpg" alt="img" style="zoom:50%;" />

### 2.推荐使用方法

Pytorch-Lightning 是一个很好的库，或者说是pytorch的抽象和包装。它的好处是可复用性强，易维护，逻辑清晰等。缺点也很明显，这个包**需要学习和理解的内容还是挺多**的，或者换句话说，很重。如果直接按照官方的模板写代码，小型project还好，如果是大型项目，有复数个需要调试验证的模型和数据集，那就不太好办，甚至更加麻烦了。经过几天的摸索和调试，我总结出了下面这样一套好用的模板，也可以说是对Pytorch-Lightning的进一步抽象。

欢迎大家尝试这一套代码风格，如果用习惯的话还是相当方便复用的，也不容易半道退坑。

```markdown
root-
    |-data
        |-__init__.py
        |-data_interface.py
        |-xxxdataset1.py
        |-xxxdataset2.py
        |-...
    |-model
        |-__init__.py
        |-model_interface.py
        |-xxxmodel1.py
        |-xxxmodel2.py
        |-...
    |-main.py
```

如果对每个模型直接上plmodule，对于已有项目、别人的代码等的转换将相当耗时。另外，这样的话，你需要给每个模型都加上一些相似的代码，如`training_step`，`validation_step`。显然，这并不是我们想要的，如果真的这样做，不但不易于维护，反而可能会更加杂乱。同理，如果把每个数据集类都直接转换成pl的DataModule，也会面临相似的问题。基于这样的考量，我建议使用上述架构：

- 主目录下只放一个`main.py`文件。

- `data`和`model`两个文件夹中放入`__init__.py`文件，做成包。这样方便导入。两个`init`文件分别是：

- - `from .data_interface import DInterface`
  - `from .model_interface import MInterface`

- 在`data_interface`中建立一个`class DInterface(pl.LightningDataModule):`用作所有**数据集文件的接口**。`__init__()`函数中import相应Dataset类，`setup()`进行实例化，并老老实实加入所需要的的`train_dataloader`, `val_dataloader`, `test_dataloader`函数。这些函数往往都是相似的，可以用几个输入args控制不同的部分。

- 同理，在`model_interface`中建立`class MInterface(pl.LightningModule):`类，作为模型的中间接口。`__init__()`函数中import相应模型类，然后老老实实加入`configure_optimizers`, `training_step`, `validation_step`等函数，用一个接口类控制所有模型。不同部分使用输入参数控制。

- `main.py`函数只负责：

- - 定义parser，添加parse项。
  - 选好需要的`callback`函数们。
  - 实例化`MInterface`, `DInterface`, `Trainer`。

### 3. Lighting Module

- 三个核心组件:

  - 模型
  - 优化器
  - Train/Val/Test步骤

- 数据流伪代码

  ```python
  outs = []
  for batch in data:
      out = training_step(batch)
      outs.append(out)
  training_epoch_end(outs)
  ```

  等价Lighting代码:

  ```python
  def training_step(self, batch, batch_idx):
      prediction = ...
      return prediction
  
  def training_epoch_end(self, training_step_outputs):
      for prediction in predictions:
          # do something with these
  ```

  我们需要做的，就是像填空一样，填这些函数。

- 组件与函数

  [API页面](https://link.zhihu.com/?target=https%3A//pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html%23lightningmodule-api)

  - 一个Pytorch-Lighting 模型必须含有的部件是：

  - `init`: 初始化，包括模型和系统的定义。

  - `training_step(self, batch, batch_idx)`: 即每个batch的处理函数。
    参数：

  - - **batch** (`Tensor` | (`Tensor`, …) | [`Tensor`, …]) – The output of your `DataLoader`. A tensor, tuple or list.
    - **batch_idx** (*[int](https://link.zhihu.com/?target=https%3A//docs.python.org/3/library/functions.html%23int)*) – Integer displaying index of this batch
    - **optimizer_idx** (*[int](https://link.zhihu.com/?target=https%3A//docs.python.org/3/library/functions.html%23int)*) – When using multiple optimizers, this argument will also be present.
    - **hiddens** (`Tensor`) – Passed in if `truncated_bptt_steps` > 0.

  - 返回值：Any of.

  - - - `Tensor` - The loss tensor
      - `dict` - A dictionary. Can include any keys, but must include the key `'loss'`
      - `None` - Training will skip to the next batch

    - 返回值无论如何也需要有一个loss量。如果是字典，要有这个key。没loss这个batch就被跳过了。例：

    - ```python
      def training_step(self, batch, batch_idx):
          x, y, z = batch
          out = self.encoder(x)
          loss = self.loss(out, x)
          return loss
      
      # Multiple optimizers (e.g.: GANs)
      def training_step(self, batch, batch_idx, optimizer_idx):
          if optimizer_idx == 0:
              # do training_step with encoder
          if optimizer_idx == 1:
              # do training_step with decoder
              
      # Truncated back-propagation through time
      def training_step(self, batch, batch_idx, hiddens):
          # hiddens are the hidden states from the previous truncated backprop step
          ...
          out, hiddens = self.lstm(data, hiddens)
          ...
          return {'loss': loss, 'hiddens': hiddens}
      ```

  - `configure_optimizers`: 优化器定义，返回一个优化器，或数个优化器，或两个List（优化器，Scheduler）。如：

    ```python
    # most cases
    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=1e-3)
        return opt
    
    # multiple optimizer case (e.g.: GAN)
    def configure_optimizers(self):
        generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
        disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
        return generator_opt, disriminator_opt
    
    # example with learning rate schedulers
    def configure_optimizers(self):
        generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
        disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
        discriminator_sched = CosineAnnealing(discriminator_opt, T_max=10)
        return [generator_opt, disriminator_opt], [discriminator_sched]
    
    # example with step-based learning rate schedulers
    def configure_optimizers(self):
        gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
        dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
        gen_sched = {'scheduler': ExponentialLR(gen_opt, 0.99),
                     'interval': 'step'}  # called after each training step
        dis_sched = CosineAnnealing(discriminator_opt, T_max=10) # called every epoch
        return [gen_opt, dis_opt], [gen_sched, dis_sched]
    
    # example with optimizer frequencies
    # see training procedure in `Improved Training of Wasserstein GANs`, Algorithm 1
    # https://arxiv.org/abs/1704.00028
    def configure_optimizers(self):
        gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
        dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
        n_critic = 5
        return (
            {'optimizer': dis_opt, 'frequency': n_critic},
            {'optimizer': gen_opt, 'frequency': 1}
        )
    ```

    - 可以指定的部件有：

    - - `forward`: 和正常的`nn.Module`一样，用于inference。内部调用时：`y=self(batch)`

      - `training_step_end`: 只在使用多个node进行训练且结果涉及如softmax之类需要全部输出联合运算的步骤时使用该函数。同理，`validation_step_end`/`test_step_end`。

      - `training_epoch_end`:

      - - 在一个训练epoch结尾处被调用。
        - 输入参数：一个List，List的内容是前面`training_step()`所返回的每次的内容。
        - 返回：None

      - `validation_step(self, batch, batch_idx)`/`test_step(self, batch, batch_idx)`:

      - - 没有返回值限制，不一定非要输出一个`val_loss`。

      - `validation_epoch_end`/`test_epoch_end`

    - 工具函数有：

    - - `freeze`：冻结所有权重以供预测时候使用。仅当已经训练完成且后面只测试时使用。
      - `print`：尽管自带的`print`函数也可以使用，但如果程序运行在分布式系统时，会打印多次。而使用`self.print()`则只会打印一次。
      - `log`：像是TensorBoard等log记录器，对于每个log的标量，都会有一个相对应的横坐标，它可能是batch number或epoch number。而`on_step`就表示把这个log出去的量的横坐标表示为当前batch，而`on_epoch`则表示将log的量在整个epoch上进行累积后log，横坐标为当前epoch。
        | LightningMoule Hook | on_step | on_epoch | prog_bar | logger | | --------------------- | ------- | -------- | -------- | ------ | | training_step | T | F | F | T | | training_step_end | T | F | F | T | | training_epoch_end | F | T | F | T | | validation_step *| F | T | F | T | | validation_step_end* | F | T | F | T | | validation_epoch_end* | F | T | F | T |
        `*` also applies to the test loop

    > 参数
    > **name** (`str`) – key name
    > **value** (`Any`) – value name
    > **prog_bar** (`bool`) – if True logs to the progress bar
    > **logger** (`bool`) – if True logs to the logger
    > **on_step** (`Optional`[`bool`]) – if True logs at this step. None auto-logs at the training_step but not validation/test_step
    > **on_epoch** (`Optional`[`bool`]) – if True logs epoch accumulated metrics. None auto-logs at the val/test step but not training_step
    > **reduce_fx** (`Callable`) – reduction function over step values for end of epoch. Torch.mean by default
    > **tbptt_reduce_fx** (`Callable`) – function to reduce on truncated back prop
    > **tbptt_pad_token** (`int`) – token to use for padding
    > **enable_graph** (`bool`) – if True, will not auto detach the graph
    > **sync_dist** (`bool`) – if True, reduces the metric across GPUs/TPUs
    > **sync_dist_op** (`Union`[`Any`, `str`]) – the op to sync across GPUs/TPUs
    > **sync_dist_group** (`Optional`[`Any`]) – the ddp group

    - - `log_dict`：和`log`函数唯一的区别就是，`name`和`value`变量由一个字典替换。表示同时log多个值。如：
        `python values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n} self.log_dict(values)`
      - `save_hyperparameters`：储存`init`中输入的所有超参。后续访问可以由`self.hparams.argX`方式进行。同时，超参表也会被存到文件中。

    - 函数内建变量：

    - - `device`：可以使用`self.device`来构建设备无关型tensor。如：`z = torch.rand(2, 3, device=self.device)`。
      - `hparams`：含有所有前面存下来的输入超参。
      - `precision`：精确度。常见32和16。

    - ### 要点

    - - 如果准备使用DataParallel，在写`training_step`的时候需要调用forward函数，`z=self(x)`

## 4.PyTorch Hook 操作

### Hook函数概念

Hook函数是在不改变主体的情况下,实现额外功能。由于PyTorch是基于动态图实现的,因此在一次迭代运算结束后,一些中间变量如非叶子节点的梯度和特征图,会被释放掉。在这种情况下想要提取和记录这些中间变量,就需要使用Hook函数。

PyTorch提供了4种Hook函数。

### torch.Tensor.register_hook(hook)

功能:注册一个反向传播hook函数,仅输入一个参数,为张量的梯度,hook函数:`hook(grad)`

代码如下:

```python
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

# 保存梯度的 list
a_grad = list()

# 定义 hook 函数，把梯度添加到 list 中
def grad_hook(grad):
    a_grad.append(grad)

# 一个张量注册 hook 函数
handle = a.register_hook(grad_hook)

y.backward()

# 查看梯度
print("gradient:", w.grad, x.grad, a.grad, b.grad, y.grad)
# 查看在 hook 函数里 list 记录的梯度
print("a_grad[0]: ", a_grad[0])
handle.remove()
```

结果如下:

```python
gradient: tensor([5.]) tensor([2.]) None None None
a_grad[0]:  tensor([2.])
```

在反向传播结束后，非叶子节点张量的梯度被清空了。而通过`hook`函数记录的梯度仍然可以查看。

`hook`函数里面可以修改梯度的值，无需返回也可以作为新的梯度赋值给原来的梯度。代码如下：

```python
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

a_grad = list()

def grad_hook(grad):
    grad *= 2
    return grad*3

handle = w.register_hook(grad_hook)

y.backward()

# 查看梯度
print("w.grad: ", w.grad)
handle.remove()
```

结果是`w.grad:  tensor([30.])`.

### torch.nn.Module.register_forward_hook(hook)

功能：注册 module 的前向传播`hook`函数，可用于获取中间的 feature map。

`hook`函数：`hook(module, input, output)`

代码如下:

```python
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 2, 3)
            self.pool1 = nn.MaxPool2d(2, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            return x

    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    # 初始化网络
    net = Net()
    net.conv1.weight[0].detach().fill_(1)
    net.conv1.weight[1].detach().fill_(2)
    net.conv1.bias.data.detach().zero_()

    # 注册hook
    fmap_block = list()
    input_block = list()
    net.conv1.register_forward_hook(forward_hook)

    # inference
    fake_img = torch.ones((1, 1, 4, 4))   # batch size * channel * H * W
    output = net(fake_img)


    # 观察
    print("output shape: {}\noutput value: {}\n".format(output.shape, output))
    print("feature maps shape: {}\noutput value: {}\n".format(fmap_block[0].shape, fmap_block[0]))
    print("input shape: {}\ninput value: {}".format(input_block[0][0].shape, input_block[0]))
```

输出如下:

```python
output shape: torch.Size([1, 2, 1, 1])
output value: tensor([[[[ 9.]],
         [[18.]]]], grad_fn=<MaxPool2DWithIndicesBackward>)
feature maps shape: torch.Size([1, 2, 2, 2])
output value: tensor([[[[ 9.,  9.],
          [ 9.,  9.]],
         [[18., 18.],
          [18., 18.]]]], grad_fn=<ThnnConv2DBackward>)
input shape: torch.Size([1, 1, 4, 4])
input value: (tensor([[[[1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.]]]]),)
```

### torch.Tensor.register_forward_pre_hook()

功能：注册 module 的前向传播前的`hook`函数，可用于获取输入数据。

`hook`函数：

hook(module, input)

参数：

- module：当前网络层
- input：当前网络层输入数据

### torch.Tensor.register_backward_hook()

功能：注册 module 的反向传播的`hook`函数，可用于获取梯度。

`hook`函数：

hook(module, grad_input, grad_output)

参数：

- module：当前网络层
- input：当前网络层输入的梯度数据
- output：当前网络层输出的梯度数据

代码如下：

    ```python
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 2, 3)
                self.pool1 = nn.MaxPool2d(2, 2)
    
            def forward(self, x):
                x = self.conv1(x)
                x = self.pool1(x)
                return x
    
        def forward_hook(module, data_input, data_output):
            fmap_block.append(data_output)
            input_block.append(data_input)
    
        def forward_pre_hook(module, data_input):
            print("forward_pre_hook input:{}".format(data_input))
    
        def backward_hook(module, grad_input, grad_output):
            print("backward hook input:{}".format(grad_input))
            print("backward hook output:{}".format(grad_output))
    
        # 初始化网络
        net = Net()
        net.conv1.weight[0].detach().fill_(1)
        net.conv1.weight[1].detach().fill_(2)
        net.conv1.bias.data.detach().zero_()
    
        # 注册hook
        fmap_block = list()
        input_block = list()
        net.conv1.register_forward_hook(forward_hook)
        net.conv1.register_forward_pre_hook(forward_pre_hook)
        net.conv1.register_backward_hook(backward_hook)
    
        # inference
        fake_img = torch.ones((1, 1, 4, 4))   # batch size * channel * H * W
        output = net(fake_img)
    
        loss_fnc = nn.L1Loss()
        target = torch.randn_like(output)
        loss = loss_fnc(target, output)
        loss.backward()
    ```

输出如下：

```python
forward_pre_hook input:(tensor([[[[1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.]]]]),)
backward hook input:(None, tensor([[[[0.5000, 0.5000, 0.5000],
          [0.5000, 0.5000, 0.5000],
          [0.5000, 0.5000, 0.5000]]],
        [[[0.5000, 0.5000, 0.5000],
          [0.5000, 0.5000, 0.5000],
          [0.5000, 0.5000, 0.5000]]]]), tensor([0.5000, 0.5000]))
backward hook output:(tensor([[[[0.5000, 0.0000],
          [0.0000, 0.0000]],
         [[0.5000, 0.0000],
          [0.0000, 0.0000]]]]),)
```

### Hook函数实现机制

`hook`函数实现的原理是在`module`的`__call()__`函数进行拦截，`__call()__`函数可以分为 4 个部分：

- 第 1 部分是实现 _forward_pre_hooks
- 第 2 部分是实现 forward 前向传播
- 第 3 部分是实现 _forward_hooks
- 第 4 部分是实现 _backward_hooks

由于卷积层也是一个`module`,因此可以记录`_forward_hooks`

```python
    def __call__(self, *input, **kwargs):
        # 第 1 部分是实现 _forward_pre_hooks
        for hook in self._forward_pre_hooks.values():
            result = hook(self, input)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                input = result

        # 第 2 部分是实现 forward 前向传播       
        if torch._C._get_tracing_state():
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)

        # 第 3 部分是实现 _forward_hooks   
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                result = hook_result

        # 第 4 部分是实现 _backward_hooks
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result
```

## 5.PyTorch常见报错

[文档](https://shimo.im/docs/bdV4DBxQwUMLrfX5/read)

## 6.Sacred-深度学习实验管理工具

### **介绍**

本文参考学习自[https://www.jarvis73.com/2020/11/15/Sacred/](https://link.zhihu.com/?target=https%3A//www.jarvis73.com/2020/11/15/Sacred/)，内容有所增删。

`Sacred`是一个 Python 库, 可以帮助研究人员配置、组织、记录和复制实验。它旨在完成研究人员需要围绕实际实验进行的所有繁琐的日常工作, 以便:

- **跟踪实验的所有参数**
- 轻松进行**不同设置**的实验
- 将单个运行的配置保存在数据库中
- 重现结果

`Sacred` 通过以下的机制实现上述目标:

- `Config Scopes` (配置范围): 通过局部变量的方式来定义实验参数的模块。
- `Config Injection` (配置注入): 可以从任意函数中获取实验参数。
- `Command-Line Interface` (命令行接口): 可以获得一个非常强大的命令行接口用于修改参数和运行实验的变体。
- `Observers` (观察器): 提供了多种观察器来记录实验中所有相关的信息: 依赖、配置、机器和结果, 可以保存到 MongoDB、文件等。
- `Automatic Seeding` (自动种子): 自动设置并保存随机种子以确保实验的可重复性。

`Sacred`最简单的运行例子：

```python
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver.creat("my_runs"))

@ex.config
def my_config():
    message = "hello world"

@ex.automain
def my_main(message):
    print(message)
```

### **教程**

### **1、快速开始**

```python
from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment           # Sacred 相关
ex = Experiment('iris_rbf_svm')         # Sacred 相关

@ex.config          # Sacred 相关
def cfg():
    C = 1.0
    gamma = 0.7

@ex.automain        # Sacred 相关
def run(C, gamma):
    iris = datasets.load_iris()
    per = permutation(iris.target.size)
    iris.data = iris.data[per]
    iris.target = iris.target[per]

    clf = svm.SVC(C, 'rbf', gamma=gamma)
    clf.fit(iris.data[:90], iris.target[:90])
    return clf.score(iris.data[90:], iris.target[90:])  # Sacred 相关
```

运行上面的脚本, 输出如下:

```text
WARNING - iris_rbf_svm - No observers have been added to this run
INFO - iris_rbf_svm - Running command 'run'
INFO - iris_rbf_svm - Started
INFO - iris_rbf_svm - Result: 0.9833333333333333
INFO - iris_rbf_svm - Completed after 0:00:00
```

可以看到代码中没有使用任何的 `print` 函数时, 该脚本的输出会包含如下几个结果:

1. 一个警告: 表示在这次代码运行中没有添加**任何观察器(observer)**
2. 当前运行的命令 `run`
3. 开始运行
4. 运行的结果
5. 运行的时间

### **2、运行组件**

在上一节中可以看到只要代码具有了以下三部分往往就可以进行使用。下面分别对其所起作用进行介绍。

- `Experiment` 类是 Sacred 框架的核心类. `Sacred`是一个辅助实验的框架, 因此在实验代码的最开始, 我们首先要定义一个 `Experiment` 的实例。
- 带有装饰器 `@ex.automain` 的函数 `train()` 是这个脚本的主入口, 当运行该脚本时, 程序会从 `train()` 函数进入开始执行。
- 带有 `@ex.config` 装饰器的函数中的**局部变量会被 Sacred 搜集起来作为参数**, 之后可以在任意函数中使用它们. 配置函数中的变量可以是整型, 浮点型, 字符串, 元组, 列表, 字典等可以`json` 序列化的类型. 并且可以在配置函数中使用之类的逻辑语句如 `if...else...` 来使得参数之间存在依赖关系. 变量的行内注释会被搜集起来写入变量的帮助文档.

除此之外，`Sacred`框架还拥有 `@ex.capture`这一用于在任意函数中写入参数并进行调用的使用方法如下。

```python
from sacred import Experiment
ex = Experiment('exp_name')

@ex.config
def my_config():    # 任意名称
    """ The core configuration. """
    batch_size = 16             # int, batch size of the training
    lr = 0.001                  # float, learning rate
    optimizer = 'adam'          # str, the optimizer of training

@ex.capture
def train(batch_size, optimizer, lr=0.1):
    if optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr)
    else:
        optim = torch.optim.SGD(model.parameters(), lr)
    # ...

@ex.automain
def main():
    train()                     # 16, 'adam', 0.001
    train(32)                   # 32, 'adam', 0.001
    train(optimizer='sgd')      # 16, 'sgd', 0.001
```

### **3、参数配置方式**

`Sacred` 提供了三种定义配置的方法:

- `Config Scope`
- 字典
- 配置文件

`Config Scope` 是通过装饰器 `@ex.config` 来实现的, 它在实验运行之前 (即定义阶段) 执行. 装饰器装饰的函数中**所有受到赋值的局部变量**会被搜集并整合为实验配置. 在函数内部可以使用 python 的任意常用的语句:

```python
@ex.config
def my_config():
    """This is my demo configuration"""

    a = 10  # some integer

    # a dictionary
    foo = {
        'a_squared': a**2,
        'bar': 'my_string%d' % a
    }
    if a > 8:
        # cool: a dynamic entry
        e = a/2
        
        
my_config()
# {'foo': {'bar': 'my_string10', 'a_squared': 100}, 'a': 10, 'e': 5}
```

**作为`Config Scope` 的函数不能包含任何的 `return` 或者 `yield` 语句。**

字典也可以直接用来添加配置

```python
ex.add_config({
    'foo': 42,
    'bar': 'baz'
})

# 或者
ex.add_config(foo=42, bar='baz')
```

**配置文件**

此外, `Sacred` 还允许用户通过文件来添加配置, 支持 `json`, `pickle` 和 `yaml` 三种格式

```python
ex.add_config('conf.json')
ex.add_config('conf.pickle')    # 要求配置保存为字典
ex.add_config('conf.yaml')      # 依赖 PyYAML 库
```

### **4、捕获参数**

`Sacred`会自动为**捕获函数 (captured functions)** 注入需要的参数, 捕获函数指的是那些被

- `@ex.main`
- `@ex.automain`
- `@ex.capture`
- `@ex.command`

装饰了的函数, 其中 `@ex.capture` 是普通的捕获函数装饰器, 具体的例子和参数优先级顺序上文已经给出, 此处不再赘述.

**注意, 捕获函数的参数在实验配置中不存在且也没有传参时会报错，同时注意实验参数的命名，避免冲突**

捕获函数可以获取一些 `Sacred`内置的变量:

- `_config` : 所有的参数作为一个字典(只读的)
- `_seed` : 一个随机种子
- `_rnd` : 一个随机状态
- `_log` : 一个 logger
- `_run` : 当前实验运行时的 run 对象

除此之外，也可以通过前缀的使用来调用之前所定义的实验参数，使用方式如下所示：

```python
@ex.config
def my_config1():
    dataset = {
        'filename': 'foo.txt',
        'path': '/tmp/'
    }

@ex.capture(prefix='dataset')
def print_me(filename, path):  # direct access to entries of the dataset dict
    print("filename =", filename)
    print("path =", path)
```

同时，`_config` 默认返回的是一个字典, 调用参数时需要大量的 `[""]` 符号, 因此本文参考链接的作者实现了一个映射的功能, 把字典的键值对映射为了成员变量, 可以直接通过 `.` 来调用. 这个映射支持字典的嵌套映射。（不过我还是习惯于 `[""]` 符号2333333

```python
class Map(ReadOnlyDict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj, **kwargs):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = Map(v)
                else:
                    new_dict[k] = v
        else:
            raise TypeError(f"`obj` must be a dict, got {type(obj)}")
        super(Map, self).__init__(new_dict, **kwargs)
```

使用例子如下:

```python
@ex.automain
def train(_config):
    cfg = Map(_config)
    
    lr = cfg.lr							# lr = cfg["lr"]
    batch_size = cfg.batch_size			# batch_size = cfg["batch_size"]
    channels = cfg.net.channels			# channels = cfg["net"]["channels"]
```

### **5、钩子机制**

配置的钩子 (hooks) 在初始化阶段执行, 可以在运行任何命令之前更新实验参数:

```python
@ex.config_hook
def hook(config, command_name, logger):
    config["a"] = 10
    return config
```

但看代码好像用的不多，且在Ingredient(本文后面有介绍) 和 hook 一起用的时候，更新参数有时不成功。建议谨慎使用。

### **6、命令行使用**

### **（1）命令**

`Sacred`内置了一系列命令, 同时用户可以自定义命令. 命令的使用方式如下:

```bash
# 内置命令
python demo.py print_config
python demo.py print_config with a=1
# 自定义命令
python demo.py train
python demo.py train with batch_size=16
```

| 命令 | 说明 |
| ---- | ---- |
|      |      |

再次考虑前文所给出的例子

```python
@ex.config
def my_config():
    """This is my demo configuration"""

    a = 10  # some integer

    # a dictionary
    foo = {
        'a_squared': a**2,
        'bar': 'my_string%d' % a
    }
    if a > 8:
        # cool: a dynamic entry
        e = a/2
```

可以使用命令行接口来获取所有的实验配置:

```bash
python config_demo.py print_config
```

这个语句中的第一个参数 `print_config` 是 `Sacred`内置的一个命令, 用于打印所有实验参数. 输出结果如下:

```bash
INFO - config_demo - Running command 'print_config'
INFO - config_demo - Started
Configuration (modified, added, typechanged, doc):
  """This is my demo configuration"""
  a = 10                             # some integer
  e = 5.0                            # cool: a dynamic entry
  seed = 954471586                   # the random seed for this experiment
  foo:                               # a dictionary
    a_squared = 100
    bar = 'my_string10'
INFO - config_demo - Completed after 0:00:00
```

自定义命令:

```python
@ex.command
def train(_run, _config):
    """
    Training a deep neural network.
    """
    pass
```

运行命令:

```bash
python demo.py train
```

查看命令帮助：

```bash
python demo.py help train

#输出：
train(_run, _config)
    Training a deep neural network.
```

### **（2）参数**

命令行接口最大的作用就是更新实验参数。

**参数的类型**: 要注意的是, Linux 中参数都是作为字符串来对待的, 因此在解析参数时遵循如下的准则

- 先给参数加一层引号(不论单双), 已经有引号的不再加
- 然后蜕掉一层引号, 此时如果是数字, 那么就解析为数字, 否则解析为字符串

```bash
# 假设参数 a 默认为数字类型
python demo.py with a=1         # a=1
python demo.py with a='1'       # a=1
python demo.py with a='"1"'     # a='1'
    
# 假设参数 b 默认为字符串类型
python demo.py with b=1         # b=1
python demo.py with b='1'       # b=1
python demo.py with b='"1"'     # b='1'
python demo.py with b=baz       # b='baz'
python demo.py with b='baz'     # b='baz'
python demo.py with b='"baz"'   # b='baz'
```

**参数中的空格**: 由于 `Sacred`是以 `<key>=<value>` 的形式来传参的, 而这样的形式在 Python 程序的传参中是被当做一个字符串参数的, 因此 `<key>` , `<value>` 内部和等号两边都不能显式的出现空格, 可以用字符逃逸或者引号。

```bash
# 下面三种写法是等价的
python demo.py with a=hello\ world      # a='hello world'
python demo.py with 'a=hello world'     # a='hello world'
python demo.py with a='hello world'     # a='hello world'
```

**参数中的括号**: 括号是 Linux 中的关键字, 因此使用圆括号时要加引号, 而方括号不是关键字, 所以可以不加。

```bash
# 圆括号
python demo.py with a=(1,2)                 # 报错
python demo.py with a='(1,2)'               # a=[1, 2]
python demo.py with a='(hello,world)'       # a='(hello,world)'
python demo.py with a='("hello","world")'   # a=["hello", "world"]
    
# 方括号
python demo.py with a=[1,2]                 # a=[1, 2]
python demo.py with a='[1,2]'               # a=[1, 2]
python demo.py with a='[hello,world]'       # a='[hello,world]'
python demo.py with a='["hello","world"]'   # a=["hello", "world"]
    
# 花括号
python demo.py with a='{"c":1,"d":2}'       # a={"c": 1, "d": 2}
```

### **7、实验结果记录**

`Sacred`提供了记录结果的方法。

```python
@ex.capture
def some_function(_run):
    loss = 0.001
    _run.log_scalar('loss', float(loss), step=1)
```

`Sacred`默认是可以捕获标准输出的, 但是如果用到的 `tqdm`之类的进度条时, 需要过滤一些参数。

```python
ex.captured_out_filter = apply_backspaces_and_linefeeds
```

关于Mongdb观察器、Mongdb及可视化之类的相关功能，由于本人并不使用，也就不在此多写，如有需要请自行查找使用。

FileStorageObserver观察器

```python3
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver.creat("my_runs"))

@ex.config
def my_config():
    message = "hello world"

@ex.automain
def my_main(message):
    print(message)
```

### **8、配料 (Ingredient)**

`Sacred`提供了一种代码模块化的机制——Ingredient, 其作用是把实验的配置模块化, 从而实现不同模块(和配置)的方便组合. 先看一个模块化数据加载的例子:

```python
import numpy as np
from sacred import Ingredient

data_ingredient = Ingredient('data')

@data_ingredient.config
def cfg():
    filename = 'my_dataset.npy'
    normalize = True

@data_ingredient.capture
def load_data(filename, normalize):
    data = np.load(filename)
    if normalize:
        data -= np.mean(data)
    return data
```

这样就把和数据加载有关的配置全部装进了 `data_ingredient` 中, 它的名称为 `data` .

接下来把配料 `data_ingredient` 引入主脚本, 并加入 `Experiment` 的 `ingredients` 参数的列表中.

```python
from sacred import Experiment
from dataset_ingredient import data_ingredient, load_data
from utils import Map

ex = Experiment('my_experiment', ingredients=[data_ingredient])

@ex.automain
def run(_config):
    cfg = Map(_config)  # Map是前文所定义的
    data = load_data()  # 直接调用函数而无需参数 (参数由 Sacred 注入)
    # 获取参数
    print(cfg.data.filename)
    print(cfg.data.normalize)
```

配料也可以拥有自己的命令:

```python
@data_ingredient.command
def stats():
    print("Status: 123")
```

运行时用 `.` 来调用:

```bash
python demo.py data.stats

# Status: 123
```

配料可以进一步嵌套:

```python
data_ingredient = Ingredient('data', ingredients=[my_subingredient])
```

在捕获函数中获取配料的参数:

```python
@ex.capture
def some_function(data):   # 配料的名称
    if dataset['normalize']:
        print("Dataset was normalized")
```
