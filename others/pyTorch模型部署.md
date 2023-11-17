# 第一章、PyTorch模型部署

## 1.前言

* 中间表示ONNX的定义标准
* PyTorch 模型转换到ONNX模型的方法
* 推理引擎  ONNX Runtime, TensorRT 的使用方法
* 部署流水线 PyTorch -ONNX -ONNX Runtime/TensorRT的示例及常见部署的解决方法
* MMDeploy C/C++ 推理 SDK

​	将PyTorch 模型部署到ONNX Runtime/TensorRT 上.

## 2.初识模型部署

在软件工程中,部署指把开发完毕的软件投入使用的过程,包括环境配置,软件安装等步骤。类似的， 对于深度学习模型来说， 模型部署指让训练好的模型在特定环境中运行的过程。相比于软件部署，模型部署会面临更多的难题：

1. 运行模型所需的环境难以配置。深度学习模型通常通常是由一些框架编写，比如PyTorch、TensorFlow。由于框架规模、依赖环境的限制，这种框架不适合在手机、开发板等生产环境中安装。
2. 深度学习模型的结构通常比较庞大，需要**大量的算力**才能满足实时运行的需求。模型的运行效率需要优化。 因为这些难题的存在，模型部署不能靠简单的环境配置与安装完成。经过工业界和学术界数年的探索，模型部署有了一条流行的流水线：

![img](img\156556619-3da7a572-876b-4909-b26f-04e81190c546.png)

为了让模型最终能够部署到某一环境上，开发者们可以使用任意一种**深度学习框架**来定义网络结构，并通过训练确定网络中的参数。之后，模型的结构和参数会被转换成一种**只描述网络结构的中间表示**，一些针对网络结构的优化会在中间表示上进行。最后，用面向硬件的**高性能编程框架(如 CUDA，OpenCL）**编写，能高效执行深度学习网络中算子的推理引擎会把中间表示转换成特定的文件格式，并在对应硬件平台上高效运行模型。

这一条流水线解决了模型部署中的两大问题：使用对接深度学习框架和推理引擎的中间表示，开发者不必担心如何在新环境中运行各个复杂的框架；**通过中间表示的网络结构优化和推理引擎**对运算的底层优化，模型的运算效率大幅提升。

## 3.部署第一个模型

### 创建PyTorch模型

* 配置环境（略过）

* 创建一个PyTorch模型

  ```python
  import os
  
  import cv2
  import numpy as np
  import requests
  import torch
  import torch.onnx
  from torch import nn
  
  class SuperResolutionNet(nn.Module):
      def __init__(self, upscale_factor):
          super().__init__()
          self.upscale_factor = upscale_factor
          self.img_upsampler = nn.Upsample(
              scale_factor=self.upscale_factor,
              mode='bicubic',
              align_corners=False)#bicubic是双线性插值
  
          self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4)
          self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0)
          self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2)
  
          self.relu = nn.ReLU()
  
      def forward(self, x):
          x = self.img_upsampler(x)
          out = self.relu(self.conv1(x))
          out = self.relu(self.conv2(out))
          out = self.conv3(out)
          return out
  
  # Download checkpoint and test image
  urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
      'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png']
  names = ['srcnn.pth', 'face.png']
  for url, name in zip(urls, names):
      if not os.path.exists(name):
          open(name, 'wb').write(requests.get(url).content)
  
  def init_torch_model():
      torch_model = SuperResolutionNet(upscale_factor=3)
  
      state_dict = torch.load('srcnn.pth')['state_dict']
  
      # Adapt the checkpoint
      for old_key in list(state_dict.keys()):
          new_key = '.'.join(old_key.split('.')[1:])
          state_dict[new_key] = state_dict.pop(old_key)
  
      torch_model.load_state_dict(state_dict)
      torch_model.eval()
      return torch_model
  
  model = init_torch_model()
  input_img = cv2.imread('face.png').astype(np.float32)
  
  # HWC to NCHW
  input_img = np.transpose(input_img, [2, 0, 1])
  input_img = np.expand_dims(input_img, 0)
  
  # Inference
  torch_output = model(torch.from_numpy(input_img)).detach().numpy()
  
  # NCHW to HWC
  torch_output = np.squeeze(torch_output, 0)
  torch_output = np.clip(torch_output, 0, 255)
  torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)
  
  # Show image
  cv2.imwrite("face_torch.png", torch_output)
  ```

  

在这份代码中，我们创建了一个经典的超分辨率网络 [SRCNN](https://arxiv.org/abs/1501.00092)。SRCNN 先把图像上采样到对应分辨率，再用 3 个卷积层处理图像。为了方便起见，我们跳过训练网络的步骤，直接下载模型权重（**由于 MMEditing 中 SRCNN 的权重结构和我们定义的模型不太一样，我们修改了权重字典的 key 来适配我们定义的模型**），同时下载好输入图片。为了让模型输出成正确的图片格式，我们把模型的输出转换成 HWC 格式，并保证每一通道的颜色值都在 0~255 之间。如果脚本正常运行的话，一幅超分辨率的人脸照片会保存在“face_torch.png”中。

在 PyTorch 模型测试正确后，我们来正式开始部署这个模型。我们下一步的任务是把 PyTorch 模型转换成用中间表示 ONNX 描述的模型。

### 中间表示——ONNX

在介绍 ONNX 之前，我们先从本质上来认识一下神经网络的结构。神经网络实际上只是描述了数据计算的过程，其结构可以用计算图表示。比如 a+b 可以用下面的计算图来表示：

![img](img\156558717-96bbe544-4dc7-4460-8850-3cb1790e39ec.png)

为了加速计算，一些框架会使用对神经网络“先编译， 后执行”的静态图来描述网络。静态图的缺点是**难以描述控制流**（比如 if-else 分支语句和 for 循环语句），直接对其引入控制语句会导致产生不同的计算图。比如循环执行 n 次 a=a+b，对于不同的 n，会生成不同的计算图：

![img](img\156558606-6ff18e19-f3b1-463f-8f83-60bf6f7ef64b.png)

ONNX （Open Neural Network Exchange）是 Facebook 和微软在2017年共同发布的，用于标准描述计算图的一种格式。目前，在数家机构的共同维护下，ONNX 已经对接了多种深度学习框架和多种推理引擎。因此，ONNX 被当成了深度学习框架到推理引擎的桥梁，就像编译器的中间语言一样。由于各框架兼容性不一，我们通常**只用 ONNX 表示更容易部署的静态图**。

使用onnx将pytorch 模型转为onnx：

```python
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "srcnn.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])
```

其中， torch.onnx.export是PyTorch自带的把模型转换成ONNX格式的函数。让我们先看一下前三个必选参数：前三个参数分别是要转换的模型、模型的任意一组输入、导出的 ONNX 文件的文件名。转换模型时，需要原模型和输出文件名是很容易理解的，但**为什么需要为模型提供一组输入呢**？

这就涉及到 ONNX 转换的原理了。从 PyTorch 的模型到 ONNX 的模型，本质上是一种语言上的翻译。直觉上的想法是**像编译器一样彻底解析原模型的代码，记录所有控制流**。但前面也讲到，我们通常**只用 ONNX 记录不考虑控制流的静态图**。因此，PyTorch 提供了一种叫做**追踪（trace）的模型**转换方法：给定一组输入，再实际执行一遍模型，即把这组输入对应的计算图记录下来，保存为 ONNX 格式。export 函数用的就是追踪导出方法，需要给任意一组输入，让模型跑起来。我们的测试图片是三通道，256x256大小的，这里也构造一个**同样形状的随机张量**。

剩下的参数中，**opset_version** 表示 ONNX 算子集的版本。深度学习的发展会不断诞生新算子，为了支持这些新增的算子，**ONNX会经常发布新的算子集**，目前已经更新15个版本。 我们令 opset_version = 11，即使用第11个 ONNX 算子集，是因为 SRCNN 中的 bicubic （双三次插值）在 opset11 中才得到支持。剩下的两个参数 **input_names, output_names** 是输入、输出 tensor 的名称，我们稍后会用到这些名称。

如果上述代码运行成功，目录下会新增一个**"srcnn.onnx"的 ONNX 模型文件**。我们可以用下面的脚本来验证一下模型文件是否正确。

```python
import onnx

onnx_model = onnx.load("srcnn.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")
```

* 使用开源模型可视化工具

其中，onnx.load 函数用于读取一个 ONNX 模型。onnx.checker.check_model 用于检查模型格式是否正确，如果有错误的话该函数会直接报错。我们的模型是正确的，控制台中应该会打印出"Model correct"。 接下来，让我们来看一看 ONNX 模型具体的结构是怎么样的。我们可以使用 **Netron （开源的模型可视化工具）**来可视化 ONNX 模型。把 srcnn.onnx 文件从本地的文件系统拖入网站，即可看到如下的可视化结果：

<img src="img\image-20231116191754250.png" alt="image-20231116191754250" style="zoom: 80%;" />

每个算子记录了**算子属性、图结构、权重**三类信息。

- 算子属性信息即图中 attributes 里的信息，对于卷积来说，算子属性包括了卷积核大小(kernel_shape)、卷积步长(strides)等内容。这些算子属性最终会用来生成一个具体的算子。
- 图结构信息指算子节点在计算图中的名称、邻边的信息。对于图中的卷积来说，该算子节点叫做 Conv_2，输入数据叫做 11，输出数据叫做 12。根据每个算子节点的图结构信息，就能完整地复原出网络的计算图。
- 权重信息指的是网络经过训练后，算子存储的权重信息。对于卷积来说，权重信息包括卷积核的权重值和卷积后的偏差值。点击图中 conv1.weight, conv1.bias 后面的加号即可看到权重信息的具体内容。

现在，我们有了 SRCNN 的 ONNX 模型。让我们看看最后该如何把这个模型运行起来。

## 4.推理引擎——ONNX Runtime

**ONNX Runtime** 是由微软维护的一个跨平台机器学习推理加速器，也就是我们前面提到的”推理引擎“。ONNX Runtime 是直接对接 ONNX 的，即 ONNX Runtime 可以直接读取并运行 .onnx 文件, 而**不需要再把 .onnx 格式的文件转换成其他格式的文件**。也就是说，对于 PyTorch - ONNX - ONNX Runtime 这条部署流水线，只要在目标设备中得到 .onnx 文件，并在 ONNX Runtime 上运行模型，模型部署就算大功告成了。

通过刚刚的操作，我们把 PyTorch 编写的模型转换成了 ONNX 模型，并通过可视化检查了模型的正确性。最后，让我们用 ONNX Runtime 运行一下模型，完成模型部署的最后一步。

ONNX Runtime 提供了 Python 接口。接着刚才的脚本，我们可以添加如下代码运行模型：

```python
import onnxruntime

ort_session = onnxruntime.InferenceSession("srcnn.onnx")
ort_inputs = {'input': input_img}
ort_output = ort_session.run(['output'], ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)#设置像素为0到255
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_ort.png", ort_output)
```

这段代码中，除去后处理操作外，和 ONNX Runtime 相关的代码只有三行。让我们简单解析一下这三行代码。**onnxruntime.InferenceSession** 用于获取一个 ONNX Runtime 推理器，其参数是用于推理的 ONNX 模型文件。推理器的 run 方法用于模型推理，其第一个参数为**输出张量名的列表**，第二个参数为输入值的字典。其中输入值字典的 **key 为张量名**，value 为 numpy 类型的张量值。输入输出张量的名称需要和 **torch.onnx.export** 中设置的输入输出名对应。

如果代码正常运行的话，另一幅超分辨率照片会保存在"face_ort.png"中。这幅图片和刚刚得到的"face_torch.png"是一模一样的。这说明 ONNX Runtime 成功运行了 SRCNN 模型，模型部署完成了！以后有用户想实现超分辨率的操作，我们只需要提供一个 "srcnn.onnx" 文件，并帮助用户配置好 ONNX Runtime 的 Python 环境，用几行代码就可以运行模型了。或者还有更简便的方法，我们可以利用 ONNX Runtime 编译出一个可以直接执行模型的应用程序。我们只需要给用户提供 ONNX 模型文件，并让用户在应用程序选择要执行的 ONNX 模型文件名就可以运行模型了。

## 5.总结

使用了成熟的模型部署工具，部署了一个初始版本的超分辨率模型SRCNN，但在实际应用场景中，随着模型结构的复杂程度不断加深，碰到的困难也会越来越多看完这篇教程，是不是感觉知识太多一下消化不过来？没关系，模型部署本身有非常多的东西要学。为了举例的方便，这篇教程包含了许多未来才会讲到的知识点。事实上，读完这篇教程后，记下以下知识点就够了：

- 模型部署，指把训练好的模型在特定环境中运行的过程。模型部署要解决模型框架兼容性差和模型运行速度慢这两大问题。
- 模型部署的常见流水线是“深度学习框架-中间表示-推理引擎”。其中比较常用的一个中间表示是 ONNX。
- 深度学习模型实际上就是一个计算图。模型部署时通常把模型转换成静态的计算图，即没有控制流（分支语句、循环语句）的计算图。
- PyTorch 框架自带对 ONNX 的支持，只需要构造一组随机的输入，并对模型调用 torch.onnx.export 即可完成 PyTorch 到 ONNX 的转换。
- 推理引擎 ONNX Runtime 对 ONNX 模型有原生的支持。给定一个 .onnx 文件，只需要简单使用 ONNX Runtime 的 Python API 就可以完成模型推理。

# 第二章、解决模型部署的中的难题

我们部署了一个简单的超分辨率模型，一切都十分顺利。但是，上一个模型还有一些缺陷——图片的放大倍数固定是 4，我们无法让图片放大任意的倍数。现在，我们来尝试部署一个支持动态放大倍数的模型，体验一下在模型部署中可能会碰到的困难。

## 1. 模型部署中常见的难题

在之前的学习中，我们在模型部署上顺风顺水，没有碰到任何问题。这是因为 SRCNN 模型只包含几个简单的算子，而这些卷积、插值算子已经在各个中间表示和推理引擎上得到了完美支持。如果模型的操作稍微复杂一点，我们可能就要为兼容模型而付出大量的功夫了。实际上，模型部署时一般会碰到以下几类困难：

- 模型的动态化。出于性能的考虑，各推理框架都默认模型的输入形状、输出形状、结构是静态的。而为了让模型的泛用性更强，部署时需要在尽可能不影响原有逻辑的前提下，让模型的输入输出或是结构动态化。
- 新算子的实现。深度学习技术日新月异，提出新算子的速度往往快于 ONNX 维护者支持的速度。为了部署最新的模型，部署工程师往往需要自己在 ONNX 和推理引擎中支持新算子。
- 中间表示与推理引擎的兼容问题。由于各推理引擎的实现不同，对 ONNX 难以形成统一的支持。为了确保模型在不同的推理引擎中有同样的运行效果，部署工程师往往得为某个推理引擎定制模型代码，这为模型部署引入了许多工作量。

现在，让我们对原来的 SRCNN 模型做一些小的修改，体验一下模型动态化对模型部署造成的困难，并学习解决该问题的一种方法。

**问题：实现动态放大的超分辨率模型**

在原来的 SRCNN 中，图片的放大比例是写死在模型里的：

```python
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

...

def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)
```

我们使用 upscale_factor 来控制模型的放大比例。初始化模型的时候，我们默认令 upscale_factor 为 3，生成了一个放大 3 倍的 PyTorch 模型。这个 PyTorch 模型最终被转换成了 ONNX 格式的模型。如果我们需要一个放大 4 倍的模型，需要重新生成一遍模型，再做一次到 ONNX 的转换。

现在，假设我们要做一个超分辨率的应用。我们的用户希望图片的放大倍数能够自由设置。而我们交给用户的，只有一个 .onnx 文件和运行超分辨率模型的应用程序。我们在不修改 .onnx 文件的前提下改变放大倍数。

因此，我们必须修改原来的模型，令模型的放大倍数变成推理时的输入。在第一章中的 Python 脚本的基础上，我们做一些修改，得到这样的脚本：

```python
import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np


class SuperResolutionNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        x = interpolate(x,
                        scale_factor=upscale_factor,
                        mode='bicubic',
                        align_corners=False)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model():
    torch_model = SuperResolutionNet()

    # Please read the code about downloading 'srcnn.pth' and 'face.png' in
    # https://mmdeploy.readthedocs.io/zh_CN/latest/tutorials/chapter_01_introduction_to_model_deployment.html#pytorch
    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


model = init_torch_model()

input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img), 3).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("face_torch_2.png", torch_output)
```

SuperResolutionNet 未修改之前，**nn.Upsample** 在初始化阶段固化了放大倍数，而 PyTorch 的 interpolate 插值算子可以在运行阶段选择放大倍数。因此，我们在新脚本中使用 interpolate 代替 nn.Upsample，从而让模型支持动态放大倍数的超分。 在第 55 行使用模型推理时，我们把放大倍数设置为 3。最后，图片保存在文件 "face_torch_2.png" 中。一切正常的话，"face_torch_2.png" 和 "face_torch.png" 的内容一模一样。

通过简单的修改，PyTorch模型已经支持了动态分辨率。但是在导出的时候出现了问题，碰到了模型部署的兼容性问题，

## 2.解决方法：自定义算子

直接使用PyTorch 模型的话，我们**修改几行代码**就能实现模型输入的动态化。但在模型部署中，我们要花数倍的时间来设法解决这一问题。现在，让我们顺着解决问题的思路，体验一下模型部署的困难，并学习使用自定义算子的方式，解决超分辨率模型的动态化问题。

刚刚的报错是因为 PyTorch 模型在导出到 ONNX 模型时，模型的输入参数的类型必须全部是 torch.Tensor。而实际上我们传入的第二个参数**" 3 "是一个整形变量。这不符合 PyTorch 转 ONNX 的规定。我们必须要修改一下原来的模型的输入**。为了保证输入的所有参数都是 torch.Tensor 类型的，我们做如下修改：

```python
...
class SuperResolutionNet(nn.Module):

    def forward(self, x, upscale_factor):
        x = interpolate(x,
                        scale_factor=upscale_factor.item(),
                        mode='bicubic',
                        align_corners=False)
...
# Inference
# Note that the second input is torch.tensor(3)
torch_output = model(torch.from_numpy(input_img), torch.tensor(3)).detach().numpy()
...
with torch.no_grad():
    torch.onnx.export(model, (x, torch.tensor(3)),
                      "srcnn2.onnx",
                      opset_version=11,
                      input_names=['input', 'factor'],
                      output_names=['output'])
```

由于 PyTorch 中 interpolate 的 scale_factor 参数必须是一个数值，我们使用 torch.Tensor.item() 来把只有一个元素的 torch.Tensor 转换成数值。之后，在模型推理时，我们使用 torch.tensor(3) 代替 3，以使得我们的所有输入都满足要求。现在运行脚本的话，无论是直接运行模型，还是导出 ONNX 模型，都不会报错了。

但是，导出 ONNX 时却报了一条 TraceWarning 的警告。这条警告说有一些量可能会追踪失败。这是怎么回事呢？让我们把生成的 srcnn2.onnx用 Netron 可视化一下：

![img](img\157626717-ff60a656-7246-4742-a227-d9f41921ecad.png)

可以发现，虽然我们把模型推理的输入设置为了两个，但 ONNX 模型还是长得和原来一模一样，只有一个叫 " input " 的输入。这是由于我们使用了 torch.Tensor.item() 把数据从 Tensor 里取出来，而导出 ONNX 模型时这个操作是无法被记录的，只好报了一条 TraceWarning。这导致 interpolate 插值函数的放大倍数还是被设置成了" 3 "这个固定值，我们导出的" srcnn2.onnx "和最开始的" srcnn.onnx "完全相同。

直接修改原来的模型似乎行不通，我们得从 PyTorch 转 ONNX 的原理入手，强行令 ONNX 模型明白我们的想法了

仔细观察 Netron 上可视化出的 ONNX 模型，可以发现在 PyTorch 中无论是使用最早的 nn.Upsample，还是后来的 interpolate，PyTorch 里的插值操作最后都会转换成 ONNX 定义的 Resize 操作。也就是说，所谓 PyTorch 转 ONNX，实际上就是把每个 PyTorch 的操作映射成了 ONNX 定义的算子。

点击该算子，可以看到它的详细参数如下：

其中，展开 scales，可以看到 scales 是一个长度为 4 的一维张量，其内容为 [1, 1, 3, 3], **表示 Resize 操作每一个维度的缩放系数**；其类型为 Initializer，表示这个值是根据常量直接初始化出来的。如果我们能够自己生成一个 **ONNX 的 Resize 算子**，让 scales 成为一个可变量而不是常量，就像它上面的 X 一样，那这个超分辨率模型就能动态缩放了。

现有实现插值的 PyTorch 算子有一套规定好的映射到 ONNX Resize 算子的方法，这些映射出的 Resize 算子的 scales 只能是常量，无法满足我们的需求。我们得自己定义一个实现插值的 PyTorch 算子，然后让它映射到一个我们期望的 ONNX Resize 算子上。

```python
import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np


class NewInterpolate(torch.autograd.Function):

    @staticmethod
    def symbolic(g, input, scales):
        return g.op("Resize",
                    input,
                    g.op("Constant",
                         value_t=torch.tensor([], dtype=torch.float32)),
                    scales,
                    coordinate_transformation_mode_s="pytorch_half_pixel",
                    cubic_coeff_a_f=-0.75,
                    mode_s='cubic',
                    nearest_mode_s="floor")

    @staticmethod
    def forward(ctx, input, scales):
        scales = scales.tolist()[-2:]
        return interpolate(input,
                           scale_factor=scales,
                           mode='bicubic',
                           align_corners=False)


```

在具体介绍这个算子的实现前，让我们先理清一下思路。我们希望新的插值算子有**两个输入**，一个是被用于操作的图像，一个是图像的放缩比例。前面讲到，为了对接 ONNX 中 Resize 算子的 scales 参数，这个**放缩比例是一个 [1, 1, x, x] 的张量**，其中 x 为放大倍数。在之前放大3倍的模型中，这个参数被固定成了[1, 1, 3, 3]。因此，在插值算子中，我们希望模型的第二个输入是一个 [1, 1, w, h] 的张量，其中 **w 和 h 分别是图片宽和高的放大倍数**。

搞清楚了插值算子的输入，再看一看算子的具体实现。算子的推理行为由算子的 forward 方法决定。该方法的第一个参数必须为 ctx，后面的参数为算子的自定义输入，我们设置两个输入，分别为被操作的图像和放缩比例。为保证推理正确，需要把 [1, 1, w, h] 格式的输入对接到原来的 interpolate 函数上。我们的做法是截取输入张量的后两个元素，把这两个元素以 list 的格式传入 interpolate 的 scale_factor 参数。

搞清楚了插值算子的输入，再看一看算子的具体实现。算子的推理行为由算子的 **forward 方法**决定。该方法的第一个参数必须为 ctx，后面的参数为算子的自定义输入，我们设置两个输入，分别为被操作的图像和放缩比例。为保证推理正确，需要把 [1, 1, w, h] 格式的输入对接到原来的 interpolate 函数上。我们的做法是截取输入张量的后两个元素，把这两个元素以 list 的格式传入 interpolate 的 scale_factor 参数。

接下来，我们要决定新算子映射到 ONNX 算子的方法。映射到 ONNX 的方法由一个算子的 symbolic 方法决定。symbolic 方法第一个参数必须是g，之后的参数是算子的自定义输入，和 forward 函数一样。ONNX 算子的具体定义由 g.op 实现。g.op 的每个参数都可以映射到 ONNX 中的算子属性：

把导出的 " srcnn3.onnx " 进行可视化：

![img](img\157627615-b923fcb5-45ab-42c0-87b5-167cd657e04e.png)

可以看到，正如我们所期望的，导出的 ONNX 模型有了两个输入！第二个输入表示图像的放缩比例。

之前在验证 PyTorch 模型和导出 ONNX 模型时，我们宽高的缩放比例设置成了 3x3。现在，在用 ONNX Runtime 推理时，我们尝试使用 4x4 的缩放比例：

```
import onnxruntime

input_factor = np.array([1, 1, 4, 4], dtype=np.float32)
ort_session = onnxruntime.InferenceSession("srcnn3.onnx")
ort_inputs = {'input': input_img, 'factor': input_factor}
ort_output = ort_session.run(None, ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_ort_3.png", ort_output)
```

## 3.总结

通过学习前两篇教程，我们走完了整个部署流水线，成功部署了支持动态放大倍数的超分辨率模型。在这个过程中，我们既学会了如何简单地调用各框架的API实现模型部署，又学到了如何分析并尝试解决模型部署时碰到的难题。

同样，让我们总结一下本篇教程的知识点：

- 模型部署中常见的几类困难有：模型的动态化；新算子的实现；框架间的兼容。
- PyTorch 转 ONNX，实际上就是把每一个操作转化成 ONNX 定义的某一个算子。比如对于 PyTorch 中的 Upsample 和 interpolate，在转 ONNX 后最终都会成为 ONNX 的 Resize 算子。
- 通过修改继承自 **torch.autograd.Function** 的算子的 symbolic 方法，可以改变该算子映射到 ONNX 算子的行为。

至此，"部署第一个模型“的教程算是告一段落了。是不是觉得学到的知识还不够多？没关系，在接下来的几篇教程中，我们将结合 MMDeploy ，重点介绍 ONNX 中间表示和 ONNX Runtime/TensorRT 推理引擎的知识，让大家学会如何部署更复杂的模型。

# 第三章、PyTorch转ONNX详解

ONNX 是目前模型部署中最重要的中间表示之一。学懂了 ONNX 的技术细节，就能规避大量的模型部署问题。从这篇文章开始，在接下来的三篇文章里，我们将由浅入深地介绍 ONNX 相关的知识。在第一篇文章里，我们会介绍更多 PyTorch 转 ONNX 的细节，让大家完全掌握把简单的 PyTorch 模型转成 ONNX 模型的方法；在第二篇文章里，我们将介绍如何在 PyTorch 中支持更多的 ONNX 算子，让大家能彻底走通 PyTorch 到 ONNX 这条部署路线；第三篇文章里，我们讲介绍 ONNX 本身的知识，以及修改、调试 ONNX 模型的常用方法，使大家能自行解决大部分和 ONNX 有关的部署问题。

在把 PyTorch 模型转换成 ONNX 模型时，我们往往只需要轻松地调用一句`torch.onnx.export`就行了。这个函数的接口看上去简单，但它在使用上还有着诸多的“潜规则”。在这篇教程中，我们会详细介绍 **PyTorch 模型转 ONNX 模型**的原理及注意事项。除此之外，我们还会介绍 PyTorch 与 ONNX 的算子对应关系，以教会大家如何处理 PyTorch 模型转换时可能会遇到的算子支持问题

## 1.`torch.onnx.export` 细解

在这一节里，我们将详细介绍 PyTorch 到 ONNX 的转换函数—— torch.onnx.export。我们希望大家能够更加灵活地使用这个模型转换接口，并通过了解它的实现原理来更好地应对该函数的报错（由于模型部署的兼容性问题，部署复杂模型时该函数时常会报错）。

### 计算图导出方法

 TorchScript 也被常当成一种中间表示使用。其他文章](https://zhuanlan.zhihu.com/p/486914187)中对 TorchScript 有详细的介绍，这里介绍 TorchScript 仅用于说明 PyTorch 模型转 ONNX的原理。 `torch.onnx.export`中需要的模型**实际上是一个`torch.jit.ScriptModule`**。而要把普通 PyTorch 模型转一个这样的 TorchScript 模型，有**跟踪（trace）和脚本化（script）**两种导出计算图的方法。如果给`torch.onnx.export`传入了一个普通 PyTorch 模型（`torch.nn.Module`)，那么这个模型会默认使用**跟踪的方法导出**。这一过程如下图所示

![img](img\163531613-9eb3c851-933e-4b0d-913a-bf92ac36e80b.png)

回忆一下我们[第一篇教程](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/01_introduction_to_model_deployment.md) 知识：跟踪法只能通过实际运行一遍模型的方法导出模型的静态图，即无法识别出模型中的控制流（如循环）；脚本化则能通过解析模型来正确记录所有的控制流。我们以下面这段代码为例来看一看这两种转换方法的区别：

而用脚本化的话，最终的 ONNX 模型用 Loop 节点来表示循环。这样哪怕对于不同的 n，ONNX 模型也有同样的结构。 由于推理引擎对静态图的支持更好，通常我们在模型部署时不需要显式地把 PyTorch 模型转成 TorchScript 模型，直接把 PyTorch 模型用 torch.onnx.export 跟踪导出即可。了解这部分的知识主要是为了在模型转换报错时能够更好地定位问题是否发生在 PyTorch 转 TorchScript 阶段。

**常见参数**：

**export_params**

模型中是否存储模型权重。一般中间表示包含两大类信息：模型结构和模型权重，这两类信息可以在同一个文件里存储，也可以分文件存储。ONNX 是用同一个文件表示记录模型的结构和权重的。 我们部署时一般都默认这个参数为 True。如果 onnx 文件是用来在不同框架间传递模型（比如 PyTorch 到 Tensorflow）而不是用于部署，则可以令这个参数为 False。

**input_names,output_names**

设置输入和输出张量的名称。如果不设置的话，会自动分配一些简单的名字（如数字）。ONNX模型的每个输入和输出张量都有一个名字。很多推理引擎在运行 **ONNX** 文件时，都需要以“名称-张量值”的数据对来输入数据，并根据输出张量的名称来获取输出数据。在进行跟张量有关的设置（比如添加动态维度）时，也需要知道张量的名字。 在实际的部署流水线中，我们都需要设**置输入和输出张量的名称，并保证 ONNX 和推理引擎中使用同一套名称**。

**opset_version**

转换时参考哪个ONNX算子集的版本，默认为9。

**dynamic_axes**

指定输入输出张量的哪些维度是动态的。 为了追求效率，ONNX 默认所有参与运算的张量都是静态的（张量的形状不发生改变）。但在实际应用中，我们又**希望模型的输入张量是动态的，尤其是本来就没有形状限制的全卷积模型**。因此，我们需要显式地指明输入输出张量的哪几个维度的大小是可变的。 我们来看一个`dynamic_axes`的设置例子：

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        return x


model = Model()
dummy_input = torch.rand(1, 3, 10, 10)
model_names = ['model_static.onnx',
'model_dynamic_0.onnx',
'model_dynamic_23.onnx']

dynamic_axes_0 = {
    'in' : [0],
    'out' : [0]
}
dynamic_axes_23 = {
    'in' : [2, 3],
    'out' : [2, 3]
}

torch.onnx.export(model, dummy_input, model_names[0],
input_names=['in'], output_names=['out'])
torch.onnx.export(model, dummy_input, model_names[1],
input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_0)
torch.onnx.export(model, dummy_input, model_names[2],
input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_23)
```

首先，我们导出3个 ONNX 模型，分别为没有动态维度、第0维动态、第2第3维动态的模型。 在这份代码里，我们是用列表的方式表示动态维度，例如：

````python
dynamic_axes_0 = {
    'in' : [0],
    'out' : [0]
}
```

由于 ONNX 要求每个动态维度都有一个名字，这样写的话会引出一条 UserWarning，警告我们通过列表的方式设置动态维度的话系统会自动为它们分配名字。一种显式添加动态维度名字的方法如下：
```python
dynamic_axes_0 = {
    'in' : {0: 'batch'},
    'out' : {0: 'batch'}
}
````

由于在这份代码里我们没有更多的对动态维度的操作，因此简单地用列表指定动态维度即可。 之后，我们用下面的代码来看一看动态维度的作用：

**使用技巧**

通过学习之前的知识，我们基本掌握了 `torch.onnx.export` 函数的部分实现原理和参数设置方法，足以完成简单模型的转换了。但在实际应用中，使用该函数还会踩很多坑。这里我们模型部署团队把在实战中积累的一些经验分享给大家。

**使模型在 ONNX 转换时有不同的行为**

有些时候，我们希望模型在直接用 PyTorch 推理时有一套逻辑，而在导出的ONNX模型中有另一套逻辑。比如，我们可以把一些后处理的逻辑放在模型里，以简化除运行模型之外的其他代码。torch.onnx.is_in_onnx_export()可以实现这一任务，该函数仅在执行 torch.onnx.export()时为真。以下是一个例子：

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        if torch.onnx.is_in_onnx_export():
            x = torch.clip(x, 0, 1)
        return x
```

有些时候，我们希望模型在直接用 PyTorch 推理时有一套逻辑，而在导出的ONNX模型中有另一套逻辑。比如，我们可以把一些后处理的逻辑放在模型里，以简化除运行模型之外的其他代码。`torch.onnx.is_in_onnx_export()`可以实现这一任务，该函数仅在执行 `torch.onnx.export()`时为真。以下是一个例子：

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        if torch.onnx.is_in_onnx_export():
            x = torch.clip(x, 0, 1)
        return x
```

这里，我们仅在模型导出时把输出张量的数值限制在[0, 1]之间。使用 `is_in_onnx_export` 确实能让我们方便地在代码中添加和模型部署相关的逻辑。但是，这些代码对只关心模型训练的开发者和用户来说很不友好，突兀的部署逻辑会降低代码整体的可读性。同时，`is_in_onnx_export` 只能在每个需要添加部署逻辑的地方都“打补丁”，难以进行统一的管理。我们之后会介绍如何使用 MMDeploy 的重写机制来规避这些问题。

**利用中断张量跟踪的操作**

PyTorch 转 ONNX 的跟踪导出法是不是万能的。如果我们在模型中做了一些很“出格”的操作，跟踪法会把某些取决于输入的中间结果变成常量，从而使导出的ONNX模型和原来的模型有出入。以下是一个会造成这种“跟踪中断”的例子：

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * x[0].item()
        return x, torch.Tensor([i for i in x])

model = Model()
dummy_input = torch.rand(10)
torch.onnx.export(model, dummy_input, 'a.onnx')
```

如果你尝试去导出这个模型，会得到一大堆 warning，告诉你转换出来的模型可能不正确。这也难怪，我们在这个模型里使用了`.item()`把 torch 中的张量转换成了普通的 Python 变量，还尝试遍历 torch 张量，并用一个列表新建一个 torch 张量。这些涉及张量与普通变量转换的逻辑都会导致最终的 ONNX 模型不太正确。 另一方面，我们也可以利用这个性质，在保证正确性的前提下令模型的中间结果变成常量。这个技巧常常用于模型的静态化上，即令模型中所有的张量形状都变成常量。在未来的教程中，我们会在部署实例中详细介绍这些“高级”操作。

#### 使用张量为输入（PyTorch版本 < 1.9.0）

正如我们第一篇教程所展示的，在较旧(< 1.9.0)的 PyTorch 中把 Python 数值作为 `torch.onnx.export()`的模型输入时会报错。出于兼容性的考虑，我们还是推荐以张量为模型转换时的模型输入。

PyTorch 对 ONNX 的算子支持

在确保`torch.onnx.export()`的调用方法无误后，PyTorch 转 ONNX 时最容易出现的问题就是算子不兼容了。这里我们会介绍如何判断某个 PyTorch 算子在 ONNX 中是否兼容，以助大家在碰到报错时能更好地把错误归类。而具体添加算子的方法我们会在之后的文章里介绍。 在转换普通的`torch.nn.Module`模型时，PyTorch 一方面会用跟踪法执行前向推理，把遇到的算子整合成计算图；另一方面，PyTorch 还会把遇到的每个算子翻译成 ONNX 中定义的算子。在这个翻译过程中，可能会碰到以下情况：

- 该算子可以一对一地翻译成一个 ONNX 算子。
- 该算子在 ONNX 中没有直接对应的算子，会翻译成一至多个 ONNX 算子。
- 该算子没有定义翻译成 ONNX 的规则，报错。

那么，该如何查看 PyTorch 算子与 ONNX 算子的对应情况呢？由于 PyTorch 算子是向 ONNX 对齐的，这里我们先看一下 ONNX 算子的定义情况，再看一下 PyTorch 定义的算子映射关系。

### ONNX 算子文档

ONNX 算子的定义情况，都可以在官方的[算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md)中查看。这份文档十分重要，我们碰到任何和 ONNX 算子有关的问题都得来”请教“这份文档。

这份文档中最重要的开头的这个算子变更表格。表格的第一列是算子名，第二列是该算子发生变动的算子集版本号，也就是我们之前在`torch.onnx.export`中提到的`opset_version`表示的算子集版本号。通过查看算子第一次发生变动的版本号，我们可以知道某个算子是从哪个版本开始支持的；通过查看某算子小于等于`opset_version`的第一个改动记录，我们可以知道当前算子集版本中该算子的定义规则。

**ONNX的底层实现**

**ONNX的存储格式**

ONNX底层是用Protobuf定义的全称 **Protocol Buffer**，是 Google 提出的一套表示和序列化数据的机制。使用 Protobuf 时，用户需要先写一份数据定义文件，再**根据这份定义文件把数据存储进一份二进制文件**。可以说，数据定义文件就是数据类，二进制文件就是数据类的实例。 这里给出一个 Protobuf 数据定义文件的例子：

```python
message Person {
  required string name = 1;
  required int32 id = 2;
  optional string email = 3;
}
```

这段定义表示在 `Person` 这种数据类型中，必须包含 `name`、`id` 这两个字段，选择性包含 `email` 字段。根据这份定义文件，用户就可以选择一种编程语言，定义一个含有成员变量 `name`、`id`、`email` 的 `Person` 类，把这个类的某个实例用 Protobuf 存储成二进制文件；反之，用户也可以**用二进制文件和对应的数据定义文件**，读取出一个 `Person` 类的实例。

而对于 ONNX ，它的 Protobuf 数据定义文件在其[开源库](https://github.com/onnx/onnx/tree/main/onnx)中，这些文件定义了神经网络中模型、节点、张量的数据类型规范；而数据定义**文件对应的二进制文件就是我们熟悉的“.onnx"文件**，每一个 ".onnx" 文件按照数据定义规范，存储了一个神经网络的所有相关数据。直接用 Protobuf 生成 ONNX 模型还是比较麻烦的。幸运的是，ONNX 提供了很多实用 API，我们可以在完全不了解 Protobuf 的前提下，构造和读取 ONNX 模型。

**ONNX 的结构定义**

在应用API对ONNX进行操作之前，我们还需要先了解一下ONNX的结构定义规则，学习一下ONNX在Protobuf定义文件里是怎么描述一个神经网络的

回想一下，神经网络本质上是一个计算图。**计算图的节点是算子**，**边是参与运算的张量**。而通过可视化 ONNX 模型，我们知道 ONNX 记录了所有算子节点的属性信息，并把**参与运算的张量信息存储在算子节点的输入输出信息**中。事实上，ONNX 模型的结构可以用类图大致表示如下：

![img](img\170020689-9a069a63-a4b7-44c0-8833-59e07c52fd5e.jpg)

如图所示，一个 ONNX 模型可以用 `ModelProto` 类表示。`ModelProto` 包含了版本、创建者等日志信息，还包含了存储计算图结构的 `graph`。**`GraphProto` 类则由输入张量信息、输出张量信息、节点信息组成。张量信息** `ValueInfoProto` 类包括张量名、基本数据类型、形状。节点信息 `NodeProto` 类包含了算子名、算子输入张量名、算子输出张量名。 让我们来看一个具体的例子。假如我们有一个描述 `output=a*x+b` 的 ONNX 模型 `model`，用 `print(model)` 可以输出以下内容：

```
ir_version: 8
graph {
  node {
    input: "a"
    input: "x"
    output: "c"
    op_type: "Mul"
  }
  node {
    input: "c"
    input: "b"
    output: "output"
    op_type: "Add"
  }
  name: "linear_func"
  input {
    name: "a"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {dim_value: 10}
          dim {dim_value: 10}
        }
      }
    }
  }
  input {
    name: "x"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {dim_value: 10}
          dim {dim_value: 10}
        }
      }
    }
  }
  input {
    name: "b"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {dim_value: 10}
          dim {dim_value: 10}
        }
      }
    }
  }
  output {
    name: "output"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim { dim_value: 10}
          dim { dim_value: 10}
        }
      }
    }
  }
}
opset_import {version: 15}
```

模型的信息由ir_version,opset_import等全局信息和graph图信息构成。而graph包含一个乘法节点、一个加法节点、三个输入张量a,x,b以及一个输出张量output。在下一节里，我们会用API构造出这个模型，并输出这段结果

## 2.读写ONNX模型

**构造ONNX模型**

在上一小节中，我们知道了ONNX模型是按以下的结构组织起来的：

* ModelProto
  * GraphProto
    * NodeProto
    * ValueInfoProto

现在，让我们抛开PyTorch，尝试完全用ONNX的Python API构建一个描述线性函数output = a*x +b的ONNX模型，我们将根据上面的模型自底向上地构造这个模型。

首先，我们可以用 `helper.make_tensor_value_info` 构造出一个描述张量信息的 `ValueInfoProto` 对象。如前面的类图所示，我们要传入张量名、张量的基本数据类型、张量形状这三个信息。在 ONNX 中，不管是输入张量还是输出张量，它们的表示方式都是一样的。因此，这里我们用类似的方式为三个输入 `a, x, b` 和一个输出 `output` 构造 `ValueInfoProto` 对象。如下面的代码所示

```
import onnx
from onnx import helper
from onnx import TensorProto

a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 10])
```

**子模型提取**

ONNX官方为开发者提供了子模型提取的功能。子模型提取，就是从一个给定的ONNX模型中，拿出一个子模型。从边22到边28的子图提取出来，并组成一个模型

添加冗余输出，无论给这个传入什么值，都不会影响子模型的输出

# 第四章、TensorRT模型的构建与推理

模型部署入门教程继续更新啦！相信经过前几期的学习，大家已经对 ONNX 这一中间表示有了一个比较全面的认识，但是在具体的生产环境中，ONNX 模型常常需要被转换成能被具体推理后端使用的模型格式。本篇教程我们就和大家一起来认识大名鼎鼎的推理后端 TensorRT。

## 1.TensorRT简介

TensorRT是由NVIDIA发布的深度学习框架，用于其在硬件上运行深度学习推理。TensorRT提供量化感知训练和离线量化功能，用户可以选择INT8和FP16两种优化模式，将深度学习模型应用到不同任务的生产部署，如视频流、语音识别、推荐、欺诈检测。TensorRT经过高度优化，可在NVIDIA GPU上运行，并且可能是目前在NVIDIA GPU上运行最快的推理引擎。

## 2.TensorRT



