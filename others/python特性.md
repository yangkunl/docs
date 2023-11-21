# python

## python代码执行

通过将python代码compile成bytecode,bytecode在c上进行执行.

## GIL(global interpret lock)

**线程**:是操作系统进行计算和调度的最小单位(可以理解为程序都是运行在线程里的)

**进程**:是比线程更大一点的单位,每一个进程有自己的内存.一个进程可以有好几个线程,同一个进程的线程共享进程的内存.(这些线程都可以**读写同样的变量**)

**问题出现**:当一个进程有不只一个线程的时候,就会出现**racing**(竞争冒险),原因是一个进程的若干个线程既有可能同时运行,也有可能交替运行.(不管是同时运行还是交替运行都没有办法控制其运行相对顺序),由于线程之间的相对运行顺序不同导致的结果不同的情况就叫做racing condition

**举例**:以python解释器的内存管理(memory manager)为例

​		  python 做到自动内存分配和释放,使用了reference count(引用计数),每一个python object都有一个引用计数,每有一个地方用到自己,就把自己的引用计数加一,这个地方不用了,就把引用计数减一.当引用计数为0时,就将其内存释放掉.

​		 竞争冒险问题,做引用计数减一时,有其他线程进来也做 这个操作,会导致引用计数数错,导致某个object的内存没有办法被释放掉,出现了**内存泄漏**的问题.

**解决方式**:

​		常用的方法是**加锁**,加锁的意思是保证这段程序只有一个线程在运行,其他的线程不允许再运行.

```python
#伪代码不能运行
a = 1
lock.aquire()
if a > 0;
	a -= 1
lock.release()
```

所有跟python object 有关的代码都有可能有这个问题,所以设计python 的设计者,给python设计一个全局的锁GIL

保证没有bytecode会被其他线程打断,线程会拿到GIL锁.

**优点**:

1. GIL全局锁,非常简单,对于单进程单线程性能非常优秀
2. 只有一个线程锁,避免了死锁的问题
3. 对于单线程的程序,全局锁的性能非常优秀
4. 便于 c extension开发

**缺点**:

1. 没法多线程并发(一个进程的多个线程同时运行在多个cpu核心上)

解决问题:

1. 使用多进程方式

```python
from multiprocessing import Pool
```

2. 自己写c extension

3. 没有GIL的python解释器:Jython 

## python 描述器(descriptor)

只要**class**定义了任何一个` def __get__()` ` def __set__()` `def__delete__()`,都会把这个class变为descriptor.

```python
class Name:
	def __get__(self, obj, objtype):
		return "Peter"
class A:
	name = Name()
o = A()
print(o.name)
print(A.name)
```

结果都为`Peter`,说明在我们使用`o.name`和`A.name`的时候实际上是调用了`__get__`函数.

描述器是属性 方法 静态方法 类方法 和 super()背后的实现机制

## python 装饰器(decorator)

python中的所有东西都是object

```python
def dec(f):
	pass
@dec
def double(x):
	return x * 2
```

等价于

```python
def dec(f):
	pass
def double(x):
	return x * 2
double = dec(double)
```

在class中的使用

```python
import time

class Timer:
	def __init__(self, func):
		self.func = func
    # __call__让实例化的对象成为一个可以调用的类
	def __call__(self, *args, **kwargs):
		start = time.time()
		ret = self.func(*args, **kwargs)
		print(f"Time: {time.time() - start}")
		return ret
@Timer
def add(a, b):
	return a + b
print(add(2, 3))
```

等价于`add = Timer(add)`,把一个函数变成了一个类的对象

 ## 迭代器

可迭代对象(iterable):一个对象可以一个一个返回他的成员(for loop 后面的 in的对象必须是一个iterable),要么有`__iter__`,要么是一个序列,有`__getitem__`这个方法(pytorch 中的dataset).

迭代器:一个表示数据流的对象,可以使用next获取新的数据,必须要有`__next__`这个方法.保证了在next方法下能取出数据

 ## 生成器

生成器函数与生成器对象

有yield关键词

调用生成器函数会返回一个生成器对象

对生成器函数使用code

返回num

## def \__call__(self)

在 Python 中，`__call__` 方法是一种特殊的方法，用于使对象实例可以像函数一样被调用。当你在一个类中定义了 `__call__` 方法时，该类的实例可以被当作函数来调用，就像调用普通的函数一样。

对于一个类，定义了 `__call__` 方法后，当你创建该类的实例并调用它时，实际上是在调用 `__call__` 方法。这允许对象实例表现得像一个函数，提供了一种自定义的可调用行为。

例如，如果你有一个类定义如下：

```python
class MyCallableClass:
    def __call__(self, arg):
        print(f"Calling with argument: {arg}")
```

然后你可以创建该类的实例，并像调用函数一样调用它：

```python
my_instance = MyCallableClass()
my_instance("Hello")
```

上面的代码会输出：

```
Calling with argument: Hello
```

在深度学习中，`__call__` 方法常常被用于定义一个可调用的网络层或模型，使得该层或模型的实例可以像函数一样处理输入。这样，模型的使用就变得更加自然和灵活。例如，一个可以处理视频剪辑（clip）的类可能会定义 `__call__` 方法来处理这个视频剪辑。

```python
class VideoProcessor:
    def __call__(self, clip):
        # 处理视频剪辑的逻辑
        processed_clip = clip + " (processed)"
        return processed_clip
```

然后，你可以创建 `VideoProcessor` 的实例并调用它来处理视频剪辑：

```python
processor = VideoProcessor()
result = processor("VideoClip1")
print(result)
```

这样的设计可以使得代码更加清晰，使得模型或处理逻辑可以以一种更自然的方式被调用。

## isinstance()

检查是否为同一类别



## python抽象基类

## with

`with` 语句是 Python 的一种语法糖，它用于简化资源管理，例如文件处理、网络连接、数据库连接等。使用 `with` 语句可以确保在代码块执行完毕后资源会被正确释放，即便发生异常也能正确处理。

具体而言，`with` 语句用于创建一个上下文管理器（Context Manager），该管理器定义了在进入和退出代码块时应该执行的操作。通常，这个上下文管理器是一个实现了 `__enter__` 和 `__exit__` 方法的对象。当进入 `with` 代码块时，`__enter__` 方法被调用，而在退出 `with` 代码块时，`__exit__` 方法被调用。

以下是 `with` 语句的基本结构：

```python
with context_manager as variable:
    # 代码块
```

- `context_manager` 是一个实现了 `__enter__` 和 `__exit__` 方法的对象。
- `variable` 是一个用于存储 `__enter__` 方法返回的对象的变量。这是可选的。

`with` 语句的主要优点包括：

1. **资源管理**：自动管理资源的分配和释放，无论代码块是否发生异常。

2. **代码简洁性**：避免了手动管理资源的繁琐步骤，使代码更加简洁、清晰，并降低出错的可能性。

3. **上下文管理器的多用途性**：`with` 不仅仅用于文件处理，还可以用于其他需要管理资源的情境，例如数据库连接、网络请求等。

下面是一个使用 `with` 语句处理文件的例子：

```python
# 不使用 with 语句
file = open('example.txt', 'r')
content = file.read()
file.close()

# 使用 with 语句
with open('example.txt', 'r') as file:
    content = file.read()
# 在退出 with 代码块时，文件会自动关闭，即便发生异常也会被正确处理
```

在这个例子中，`with` 语句确保在代码块执行完毕后文件被正确关闭，无需显式调用 `file.close()`。

## Jit

jit是numba库里面最核心最厉害的功能：我们知道，python是解释性语言，数据类型可以是动态地，带来了很多方便，但是速度也大大降低。而编译性语言，例如c/c++很快，所以jit就是用来编译python的，编译好处就是，可以对代码进行优化，从而加速。jit是一个修饰符decorator，作用对象是函数。即对函数进行编译优化，产生一个高效代码。

```python
from numba import jit
```

* 编译模式

  * Lazy compilation

    ```python
    @jit
    def f(x, y):
    	return x + y
    ```

    用法：

    ```python
    f(1, 2)#整形
    #3
    f(1j, 2)#复数类型
    #(2+1j)
    ```

    解释：jit会先让你的函数运行一次，摸清楚了传入的变量类型之后，针对这种变量类型进行优化。

  * Eager compilation

    ```python
    from numba import jit, int32
    @jit(int32(int32, int32))
    def f(x, y):
    	return x + y
    ```

    解释：第一种模式是让jit自己推断数据变量的类型，而这里是你自己指定，所以优化速度明显。

    注意的是，像上面，指定了数据类型都是int32，如果你拿一些其他的数据类型来，将被强制转换数据类型，因此引来精度损失或者报错。

- 编译选项

  下面介绍如何继续提供更加精细化的控制。下面又有两个编译模式模式，nopython和object，前面章节是编译模式，是对一个东西（编译）的不同角度的分类。比如人可以分成高人矮人，胖子瘦子，角度不一样。

  - nopython

    这个模式，是被推荐的模式

    ```python
    @jit(nopython=True)
    def f(x, y):
        return x + y
    ```

    这段代码脱离了python解释器， 变成机器码来实现，所以速度超快

    这个这么好，那么下面这个object模式还有什么意义呢？你错了，这个这么好是有前提的，需要你的函数代码是循环比较多，然后进行一些数学运算，这种代码在这个模式可以超级快。如果你的函数代码不是这种数学计算类的循环，比如下面这样：

    ```python
    def foo():
    	A#非数学计算类。
    	for i in range(1000):
    		B#数学计算类。
    	C#非数学计算类。
    ```

    这个模式将自动识别那个循环，然后优化，脱离python解释器，运行。而对于A,C这两个东西无法优化，需要切换回到python解释器，极其浪费时间，效果差。切换很费时间，这种情况，最好不要用nopython的模式，而使用下面地这种普通模式

  - object

​				普通模式，就是在python解释器里运行的模式。没有写nopython=True那么就默认是这个。

​				从上面的描述来看，似乎已经透露了：循环，数学类的运算将大大优化，有一些代码不能优化。下面给个例子，什么东西				Numba喜欢，什么类型的代码numba不喜欢。

```
from numba import jit
import numpy as np

x = np.arange(100).reshape(10, 10)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

print(go_fast(x))

```

不喜欢的就是那些numba看不懂的对象，例如你自己定义的对象object，或者一些库里面的对象，例如dataframe对象。也就是说numba喜欢搞数学有关的，以及那些基本数据类型（整形等）的运行，而对象这种东西太高级了。

例如下面这个，jit加速没有用。

```
from numba import jit
import pandas as pd

x = {'a': [1, 2, 3], 'b': [20, 30, 40]}

@jit
def use_pandas(a): # Function will not benefit from Numba jit
    df = pd.DataFrame.from_dict(a) # Numba doesn't know about pd.DataFrame
    df += 1                        # Numba doesn't understand what this is
    return df.cov()                # or this!

print(use_pandas(x))

```

一个坑
本人在使用时发现一个特别有意思的事，可能别人都不知道，算是一个告诉你的小秘密吧。那就是使用下面模式的时候

```
@jit(nopython=True)
def foo():
```

我们知道，这个编译优化，然后运行的时候会脱离python解释器。这就导致了！！！！！！！！！！！！

这一部分无法调试，你在函数内部打断点，没有用！！！！！！！！！

所以，在你调试的时候，可以先把加速给关了，不要使用jit。

## class 机制

```python
class A:
	name = "AAA"
	def f(self):
		print(1)
```

当定义一个class的时候,先运行一遍class,将class中的局部变量和函数以键值对的形式保存在一个dict中,接着建立一个type.使用type建立一个动态的类.

直接通过class定义的class的type都是`type`,用Ns保存A里面的局部变量,以key-value pair 保存在dictionary,`class.__dict__`,做A.f的时候会在class

## class中的self

self保存的是其本身的object,self这个魔法只会在对象上出现,不会在原class中出现,

```python
class A:
	def f(self, data):
		print(self.name)
		print(data)
o = A()
print(A.f)
print(o.f)
```

我们打印这两个`A.f`和`o.f`发现如下输出:

```python
<function A.f at 0x000002778115B040>
<bound method A.f of <__main__.A object at 0x00000277810A6730>>
```

可以发现`A.f`是函数,而`o.f`则变成了`bound method`(意思只这个函数绑定了一个对象,这个对象就是`o`).

## python中的魔术方法

在 Python 中，魔术方法（也称为特殊方法或双下划线方法）的命名约定是以双下划线开头和结尾，例如 `__init__`、`__str__`、`__len__` 等。这些方法在类中有特殊的含义，用于实现特定的行为。魔术方法是 Python 面向对象编程的一部分，它们允许程序员在对象生命周期的不同阶段插入自定义的逻辑。

以下是一些常见的魔术方法：

1. **`__init__`:** 对象初始化方法，在创建对象时调用。

    ```python
    class MyClass:
        def __init__(self, value):
            self.value = value
    ```

2. **`__str__`:** 将对象转换为字符串的方法，在使用 `str()` 函数或 `print()` 函数时调用。

    ```python
    class MyClass:
        def __str__(self):
            return f"MyClass instance with value: {self.value}"
    ```

3. **`__len__`:** 获取对象长度的方法，在使用 `len()` 函数时调用。

    ```python
    class MyList:
        def __init__(self, elements):
            self.elements = elements

        def __len__(self):
            return len(self.elements)
    ```

4. **`__getitem__` 和 `__setitem__`:** 获取和设置对象的元素的方法，允许对象像序列一样进行索引和切片。

    ```python
    class MyList:
        def __init__(self, elements):
            self.elements = elements
    
        def __getitem__(self, index):
            return self.elements[index]
    
        def __setitem__(self, index, value):
            self.elements[index] = value
    ```

这些魔术方法允许开发者定制对象在不同操作中的行为，使得 Python 的类能够更灵活地适应各种用途。魔术方法的名称和用途都是由 Python 解释器定义的，并在特定情况下由解释器自动调用。

## python中的import

module是python object,常常对应于.py文件,module是python运行时的概念,其本身是python object.文件是操作系统的概念,我们需要操作系统从一个文件中生成一个module.package是一种特殊的module,package和module几乎有着一模一样的功能,只是多了一个`__path__`,但是在操作系统级package往往对应一个文件夹.

package可以有其他的subpackage,module.

一个文件夹只有有这个`__init__`他才在python中算一个package,但是在python3这是错误的,无论有没有都可以作为package

`import`是将python中的文件夹或者文件变成python中的module或者package

- `import test`

- 首先把你import的这个字符串作为名字来寻找这个module,首先检查缓存去看看有没有叫test的module已经被读取过来了,如果有的话就不用load的过程直接赋值给test

- 如果没有就要寻找这个叫test的module,首先看这个名字是不是一个buildin module(python自带的module, ex: sys)

- 如果不是buildin module,那么首先要在几个文件夹类寻找可以被load成test的文件(最常见的就是test.py的文件)

  > 通过打印`sys.path`可以发现他会在那几个文件夹中寻找module,一般来说如果我们是以`python example.py`的形式运行一个脚本的话,打印的第一个值一般是文件所在的文件夹,同时在python运行的时候会往里方一些python自带的package, 例如asyncio,multiprocessing,还有这个site-packages就是`pip install`时候的位置,在python运行时可以手动更改这个`sys.path`

- 注意命名冲突的问题(例如当前文件夹你有一个test.py和安装了一个test的package,你`import`会import test.py而不是那个package),因为`import`会优先从当前文件夹`import module`,原因是 `sys.path`**当前文件夹排在第一位**.

- 当找到文件后会**在单独的命名空间中运行这个文件**(import 若干次也只会执行一次,就是因为有缓存机制)建立module,例如文件中有class A, 那么就会在命名空间中生成class A,也就是建立一个module,同时更新一下缓存,这样其他的代码再import的时候就不会再load一遍

- 最后会将这个module object赋值给这个test变量,打印一下test你会发现`<module 'dsada' from 'D:\\pythonProject1\\dsada.py'>`

- 如果想要保存在另外一个变量名里可以`import test as t`,意思就是根据这个test去找module,然后将这个module保存到t中

- 只需要module里面的某一个object,`from test import A`

- 可以通过import package,将文件夹import进来,如果没有`__init__`的话不会运行任何文件,如果有的话会运行`__init__.py`.(是在单独的命名空间中运行`__init__.py`这个文件)

- 前面都是absolute import与此对应的还有relative import

- > from .util import f,注意relative import只能在module的import中使用,如果不是会报错
  >
  > 同理还有 from ..util import f 代表的是上一个文件夹.
  >
  > 

## python中的mutable和immutable

python是不知道什么东西是mutable和immutable的,immutable的internal state是不能被改变的,python对immutable和mutable写的函数

# python buildin 函数

## abs（x）

- 返回一个数值的绝对值
- x可以是整数，浮点数，复数
- 如果参数是复数，返回复数的模

## all(iterable) 

- 可迭代对象为空或元素全为True时返回True
- 元素除了是0、空、None、False外都算True
- 若是元素中有false，则会返回false

## any(iterable)

- 可迭代对象有一个元素为True时返回True
- 类似于or的逻辑

## ascii(object)

- `ascii()` 是一个内建函数，用于返回一个对象的 ASCII 表示。它会创建一个包含对象可打印表示的字符串。

  以下是 `ascii()` 函数的基本语法：

  ```python
  ascii(object)
  ```

  其中，`object` 是你想要获取 ASCII 表示的对象，可以是字符串、数字、列表等。

  示例：

  ```python
  # 使用 ascii() 获取字符串的 ASCII 表示
  string_example = "Hello, world!"
  ascii_representation = ascii(string_example)
  
  print(ascii_representation)
  ```

  在这个例子中，`ascii()` 函数将字符串 "Hello, world!" 转换为其 ASCII 表示，并将结果打印出来。注意，`ascii()` 会在非 ASCII 字符前加上转义符号。

  你可以根据需要传递不同类型的对象给 `ascii()` 函数，它会尝试生成适当的 ASCII 表示。

  结果为：'Hello, world!'

- 对于字符串中的非ASCII字符则返回通过repr()函数使用\x,\u或\U编码的字符

## bin(x)

- x是int 或者 long int数字，浮点数不是通过二进制保存
- 返回二进制表示的字符串，以'0b'开头

## bool(x)

- 将给定参数转换为布尔类型
- 如果没有参数，返回False

## bytearray([source[,encoding[,errors]]])

○ 返回一个新字节数组，其中元素是可变的，并且每个元素的值范围：0 <= x < 256
○ 对于source参数:

- 如果 source 为整数，则返回一个长度为 source 的初始化数组；
-  如果 source 为字符串，则按照指定的 encoding 将字符串转换为字节序列；
-  如果 source 为可迭代类型，则元素必须为[0 ,255] 中的整数；
-  如果没有输入任何参数，默认就是初始化数组为0个元素。

用法：

```python
# 创建一个空的字节数组
empty_bytearray = bytearray()
print(empty_bytearray)  # 输出 bytearray(b'')

# 从字符串创建字节数组
bytearray_from_string = bytearray("Hello, world!", "utf-8")
print(bytearray_from_string)  # 输出 bytearray(b'Hello, world!')

# 从可迭代对象创建字节数组
bytearray_from_iterable = bytearray([65, 66, 67, 68, 69])
print(bytearray_from_iterable)  # 输出 bytearray(b'ABCDE')

# 从整数创建字节数组
bytearray_from_integer = bytearray(5)
print(bytearray_from_integer)  # 输出 bytearray(b'\x00\x00\x00\x00\x00')
```

## bytes([source[,encoding[,errors]]])

- 返回一个新的 bytes 对象，该对象是一个 0 <= x < 256 区间内的整数不可变序列
- 是 bytearray 的不可变版本

## callable(object)

- 检查一个对象是否是可调用的，即是否实现了call方法
- 函数、方法、类、lambda函式、类的返回结果都为True
- callable(类名)一定为True，callable(实例名)取决于有没有实现\__call__()

## chr(i)

- i可以是10进制也可以是16进制形式的数字(0~1,114,111)
- 返回值是当前整数对应的ASCII字符

## compile(source,filename,mode[,flags[,dont_inherit]])

- 将一个字符串编译为字节代码

- 参数

  - `source`：表示要编译的源代码，可以是字符串、AST 对象或代码对象
  - `filename`：表示源代码的文件名，如果源代码不是从文件中读取的，通常可以传递一个有意义的字符串。
  - `mode`：表示编译代码的模式。可以是 `"exec"`（用于模块级别代码）、`"eval"`（用于单个表达式）或 `"single"`（用于交互式环境中的语句）
  - `flags`：可选参数，用于指定编译器的标志。可以是 `ast.PyCF_ONLY_AST`（返回 AST 对象而不是代码对象）等标志的组合。
  - `dont_inherit`：可选参数，默认为 `False`。如果设置为 `True`，则不继承来自外部源（如环境变量 `PYTHONOPTIMIZE`）的编译器标志

- 示例：

  ```python
  # 使用 compile() 编译一段简单的代码
  code_to_compile = "print('Hello, world!')"
  compiled_code = compile(code_to_compile, "example", "single")
  
  # 执行编译后的代码对象
  exec(compiled_code)
  print(compiled_code)
  ```

## complex([real[,imag]])

- 用于创建一个值为 real + imag * j 的复数或者转化一个字符串或数为复数

- 如果第一个参数为字符串，则不需要指定第二个参数

  ```python
  complex(str("2+5j"))
  ```

- 如果是字符串，字符串里不能有空格

## delattr(object,name)

- 用于删除属性，name必须是object的属性名(`python`中一切都是object，所以这个还是挺有用的)
- delattr(x,'foobar')相等于 del.x.foobar

##dict(\*\*kwarg) or dict(mapping, \*\*kwarg) or dict(iterable,\**kwarg)

- 用于创建字典
- 参数说明：
  - \*\*kwarg关键字
  - mapping 对象与对象之间的映射关系
  - iterable 可迭代对象
- dict(a=‘a’, b=‘b’, t=‘t’) # 传入键值对
  dict(zip([‘one’, ‘two’, ‘three’], [1, 2, 3])) # 映射函数方式来构造字典
  dict([(‘one’, 1), (‘two’, 2), (‘three’, 3)]) # 可迭代对象方式来构造字典
  dict({‘x’: 4, ‘y’: 5}) # 映射方式来构造字典

## dir([object])

- 不带参数的时候,返回当前范围内的变量、方法和定义的类型列表(收集参数的信息)
- 带参数时，返回参数的属性、方法列表
-  如果参数包含方法__dir__()，该方法将被调用。如果参数不包含__dir__()，该方法将最大限度地收集参数信息

## divmod(a,b)

- 返回一个包括商和余数的元组
- 如果参数a 与参数 b都是整数,函数返回的结果相当于 (a // b, a % b)
- 如果其中一个参数为浮点数时，函数返回的结果相当于 (q, a % b)，q通常是math.floor(a / b)

## enumerate(sequence, [start=0])

- 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
-  常用于for循环中
- 参数
  -  sequence – 一个序列、迭代器或其他支持迭代对象。
  - start – 下标起始位置。

## eval(expression[, globals[, locals]])

- 用来执行一个字符串表达式，并返回表达式的值(执行字符串表达式例如"5 + 4")

## exec(object[, globals[, locals]])

- 执行储存在字符串或文件中的 Python 语句
-  相比于 eval，exec可以执行更复杂的 Python 代码
-  返回值永远为None

##filter(function,iterable)

- 用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象
- 如果要转换为列表，可以使用 list() 来转换
- 接收两个参数，第一个为函数，第二个为序列

## float(x)

- 将整数或字符串转换成浮点数

## str.format()

-  通过 {} 和 : 来代替以前的 %
- 用大括号{}来转义大括号

## frozenset([iterable])

- 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素
- 如果不提供任何参数，默认会生成空集合。

## getattr(object, name[, default])

- 返回一个对象的属性值。
- 参数
  - object 对象
  - name 字符串(属性必须是一个字符串)
  - default 默认返回值

## globals()

- 以字典类型返回当前位置的全部全局变量

## hasattr(object, name)

- 用于判断对象是否包含对应的属性

## hash(object)

-  获取取一个对象（字符串或者数值等）的哈希值

## help([object])

-  用于查看函数或模块用途的详细说明,获取python帮助文档,在命令行直接输入`help`.

## hex(x)

-  将一个指定数字转换为 16 进制数(必须是整形)
- 以字符串形式返回，开头为0x

## id([object])

- 返回对象的唯一标识符，标识符是一个整数

## input([prompt])

- 接受一个标准输入数据，返回为 string 类型
- 将所有输入默认为字符串处理，并返回字符串类型

## int(x, base=10)

- 将一个字符串或数字转换为整型

## isinstance(object, classinfo)

- 判断一个对象是否是一个已知的类型

- 可以判断是否是元组中的一个

## issubclass(class, classinfo)

- 判断参数 class 是否是类型参数 classinfo 的继承类（子类）

## iter(object[, sentinel])

- 生成迭代器(使用了`iter`,我们可以使用next获取下一个元素,使用next必须使用`iter`函数,初始化迭代器后,next就是)
- 参数
  - object - 支持迭代的集合对象
  - sentinel -如果传递了第二个对象,则参数 object 必须是一个**可调用的对象（如，函数）**，此时，iter 创建了一个迭代器对象，每次调用这个迭代器对象的__next__()方法时，都会调用 object。(sentinel代表在next等于sentinel时返回)

## len(object)

- 返回对象(字符,列表,元组)长度或项目的个数

## list(seq)

- 将元组或字符串转换为列表

##  local()

-  以字典类型返回当前位置的全部局部变量

## map(function, iterable, ...)

-  会根据提供的函数对指定序列做映射
-  iterable中的每一个元素调用function，返回新的元素值组成的iterable

## max( x, y, z, … )

-  返回给定参数的最大值，参数可以为序列

## memoryview(obj)

- 以元组形式返回给定参数的内存查看对象,只能是byte对象

## min( x, y, z, … )

-  返回给定参数的最小值，参数可以为序列

## next(iterable[, default])

- 返回迭代器的下一个项目
- default – 可选，用于设置在没有下一个元素时返回该默认值，如果不设置，又没有下一个元素则会触发 StopIteration 异常。
- 要和生成迭代器的 iter() 函数一起使用

## oct(x)

- 将一个整数转换为8进制字符串,以'0o'作为前缀(与hex一样不能接受浮点数)

## ord

- 是 chr() 函数（对于 8 位的 ASCII 字符串）的配对函数
- 以一个字符串（Unicode 字符）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值

## pow(x, y[, z])

- 计算x的y次方，如果z在存在，则再对结果进行取模
- 与math.pow相比参数必须是整形,math.pow的参数是浮点型(整形也行,但是输出是浮点数)

## print(*objects, sep=’ ‘, end=’\n’, file=sys.stdout, flush=False)

- objects – 复数，表示可以一次输出多个对象。输出多个对象时，需要用 , 分隔。
- sep – 用来间隔多个对象，默认值是一个空格。
- end – 用来设定以什么结尾。默认值是换行符 \n，我们可以换成其他字符串
-  file – 要写入的文件对象。
- flush – 输出是否被缓存通常决定于 file，但如果 flush 关键字参数为 True，流会被强制刷新。

## property([fget[, fset[, fdel[, doc]]]])

-  在新式类中返回属性值(改实例的属性就不用重新初始化)
- 参数
  -  fget – 获取属性值的函数
  -  fset – 设置属性值的函数
  -  fdel – 删除属性值函数
  - doc – 属性描述信息

```python
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """Get the person's name."""
        print("Getting name")
        return self._name

    @name.setter
    def name(self, value):
        """Set the person's name."""
        print("Setting name")
        self._name = value

    @name.deleter
    def name(self):
        """Delete the person's name."""
        print("Deleting name")
        del self._name

# 使用 property 创建一个名为 'name' 的特性
person = Person("John")
print(person.name)  # 调用 getter 方法，输出 "Getting name" 和实际名字 "John"
person.name = "Doe"  # 调用 setter 方法，输出 "Setting name"
del person.name  # 调用 deleter 方法，输出 "Deleting name"

```

## range(start=0,stop[,step])

- 返回可迭代对象，而不是列表！
- 计数到stop停止，但不包括step

```python
# 生成范围为 [0, 1, 2, 3, 4]
r1 = range(5)
print(list(r1))  # 输出 [0, 1, 2, 3, 4]

# 生成范围为 [2, 5, 8, 11]
r2 = range(2, 12, 3)
print(list(r2))  # 输出 [2, 5, 8, 11]
```

## repr(object)

- 返回一个对象的 string 格式

## reversed(seq)

- 返回一个反转的迭代器
-  seq可以是 tuple, string, list 或 range

## round(x[,n])

- 返回浮点数 x 的四舍五入值
- n为保留几位小数，默认为0
- 并不是严格的四舍五入，受浮点数精度影响

## set([iterable])

- 创建一个无序不重复元素集，自动删除重复元素

## setattr(object, name, value)

- 设置属性值，该属性不一定是存在的

## slice(start, stop[, step])

- 实现切片对象，主要用在切片操作函数里的参数传递

```python
# 使用 slice() 函数创建切片对象
my_slice = slice(2, 7, 2)

# 使用切片对象切割列表
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
result = my_list[my_slice]

print(result)  # 输出 [2, 4, 6]
```

## sorted(iterable, key=None, reverse=False)

- 参数
  - iterable – 可迭代对象。
  - key – 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
  - reverse – 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
-  返回一个list，不改变原始对象
-  list.sort()**会改变原始对象**，返回值为None

## staticmethod(function)

- 返回函数的静态方法(静态方法是与类关联的,无需创建实例就能调用静态方法)

```python
class MathOperations:
    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def subtract(x, y):
        return x - y

# 调用静态方法，无需创建类的实例
result_add = MathOperations.add(5, 3)
result_subtract = MathOperations.subtract(8, 2)

print(result_add)       # 输出 8
print(result_subtract)  # 输出 6
```

## str(object=‘’)

- 返回一个对象的string格式

## sum(iterable[, start])

-  对序列(可迭代对象)进行求和计算
- 参数
  - iterable – 可迭代对象，如：列表、元组、集合。
  -  start – 指定相加的参数，如果没有设置这个值，默认为0。

## super(type[, object-or-type])

-  用来解决多重继承问题，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题
- super().xxx 相当于 super(Class, self).xxx
-  例如：super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类 FooChild 的对象转换为类 FooParent 的对象'

在子类中super,使得子类拥有父类的属性

```python
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, age):
        # 调用父类的 __init__ 方法
        super().__init__(name)
        self.age = age

child = Child("Alice", 5)
print(child.name)  # 输出 "Alice"
print(child.age)   # 输出 5
```

## tuple(iterable)

- 将可迭代系列（如列表）转换为元组

## type(object) 或者 type(name, bases, dict)

- 返回对象的类型 或者 新的类型对象
-  isinstance() 与 type() 区别：
  - type() 不会认为子类是一种父类类型，不考虑继承关系。
  -  isinstance() 会认为子类是一种父类类型，考虑继承关系。
  - name – 类的名称。
  -  bases – 基类的元组
  - dict – 字典，类内定义的命名空间变量。

## vars([object])

- 返回对象object的属性和属性值的字典对象
- 如果没有参数，就打印当前调用位置的属性和属性值，类似 locals()

## zip([iterable, …])

-  将一个或多个迭代器打包成一个个元组，然后返回由这些元组组成的对象
- 返回列表长度与最短的迭代器相同
- 利用 * 号操作符，可以将元组解压为列表

## **import**(name[, globals[, locals[, fromlist[, level]]]])

- 用于动态加载类和函数
-  如果一个模块经常变化就可以使用 **import**() 来动态载入

## classmethod修饰符

- classmethod修饰符对应的函数不需要实例化，不需要 self 参数(类方法可以访问和修改类的属性。这使得在不创建类的实例的情况下，对类级别的变量进行操作成为可能。)
- 第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等
- 在方法的前一行加上@classmethod，不需要实例化类就可以被类本身调用，cls表示没用被实例化的类本身

# python魔术方法

魔术方法(special method),python提供的让用户客制化类的方法,是定义在类里面的一些特殊的方法.

- 特点:method的名字,前后都有两个下划线

## def \_\_new_\_(cls): 和 def \_\_init\_\_(self):

这两个方法可以改变从一个类建立一个对象的时候的行为

 \_\_new_\_(cls)是从一个class建立一个object的过程, \_\_init\_\_(self),是有了这个object,给object初始化时候的过程

new使用较少,如果不需要客制化建立object的过程,只需要用到init函数

啥时候会用到new,比如说想做一个Singleton class(只允许实例化一个对象的类),在建立object之前,首先判断一下，有没有其他的object被建立了。如果有则不在建立新的object。

- new函数是有返回值的，init函数是没有返回值的（如果不给new函数返回值，那么得到的是一个空的objec，也无法初始化）

## def \_\_del\_\_(self):

与python的关键字`del`是没有关系的,可以将其理解为析构函数(当然其不是析构函数)

> 析构函数（Destructor）是在对象生命周期结束时被调用的特殊成员函数，用于清理对象所分配的资源或执行其他必要的清理操作

当对象被释放的时候,想跑点什么可以写在del里,当对象被释放的时候就会执行其中的代码例如释放对象`del(a)`

**python中对象被释放比较复杂**

##def \_\_repr\_\_(self):和def \_\_str\_\_(self):

这两个函数的功能相似,都是返回这个object的字符串表示.这两个方法之间主要是语义上的不同.`str`返回的具有更高的可读性,而`repr`返回的内容具有更详细的信息.在这两个方法都定义了的情况下,`print`是会调用这个`str`的函数,会打印这个函数的返回值

要想调用`repr`函数,可以使用python buildin 方法repr,同理也可以使用buildin 函数 str来调用`str`函数.

可以不同时定义这两个special method,这样可以直接打印那个定义了的method

## def \_\_format\_\_(self):

尝试使用某种格式打印这个object的时候,就有可能会调用这个`__format__`函数

##`def __bytes__(self):`

使用案例如下:

`print(bytes(A())`

## 使用魔术方法来进行rich comparison `def __eq__(self)`

可以通过定义这个`__eq__`函数,来比较两个对象.

例如

```python
def __eq__(self, other):
        return self.x == other.x
```

而且不只能返回boolen,可以设置返回成任何值

一般来说定义`__eq__`就足够了,但是也可以定义不等于的魔术方法如下

## `def __ne__(self)`

运行不等于时调用`__ne__`

## `def __gt__`

大于和小于号

## `def __lt__`

## 如果定义了自己的`__eq__`函数,那么python 类中的hash函数就会被删除,必须定义自己的hash函数`def __hash__(self)`

hash函数的要求

- 第一必须返回一个整数
- 对于两个相等的对象,必须要有同样的hash值

python官方建议是使用python的buildin function **hash**

## `def __bool__`

对于所有的自定义对象,当你将其放入if statement 的时候都会默认为真

所以需要自己定义`__bool__`函数

## `def __getattr__`

当你调用一个对象的属性,而这个属性不存在的时候,就会调用这个方法.只有在不存在这个属性的时候才会被调用!!!

## `def __getattribute__`

`__getattr__`是只有当属性不存在的时候才会被调用,而这个`__getattribute__`是只要你尝试读取属性都会被调用.

注意里面可能会产生的不显眼的递归.记住默认的behavior是`super().__getattribute__()`

## `def __setattr__(self, name, val)`

在`__init__`的时候也会调用`__setattr__`

## `def __delattr__`

在尝试删除属性时会调用`__delattr__`

## `def __dir__(self)`

## `def __get__()`

## `__slots__`specialname

白名单机制,规定了哪些是能够自定义的attribute

## `__init_subclass_(cls)`





## `def __add__:`

定义当遇到号后该怎么办 Vector()

`def __sub__`

`def__mod__`

`def __pow__`

`def __lshift__`

`def __mul__`

做出C++中多态的效果

def __rmul

二元操作都有他们的r版本和i版本就是+=

一元操作 abs, invert, 

`def __int__(self)`

`def __index__`

# python buildin package

## asyncio

```python
import asyncio
async def main():
	print('hello')
	await asyncio.sleep(1)
	print('world')
asyncio.run(main())
```

同时执行的任务只有一个,event loop,面对很多可以执行的任务,进行系统级的上下文切换.

- coroutine function ,调用的时候是返回的coroutine object,直接调用不会运行

  >协程（Coroutines）是一种在异步编程中用于实现并发操作的概念。在Python中，协程通常使用`async`和`await`关键字来定义和管理。
  >
  >协程与普通函数类似，但有一些关键区别：
  >
  >1. **关键字：** 协程函数使用 `async def` 来定义，而不是普通函数的 `def`。例如：
  >
  >    ```python
  >    async def my_coroutine():
  >        # 协程的主体代码
  >    ```
  >
  >2. **`await`关键字：** 在协程中，可以使用 `await` 关键字来暂时挂起协程的执行，等待某个异步操作的完成。`await`通常用于调用其他协程、异步函数或者异步操作，以便让事件循环在等待的过程中执行其他协程。
  >
  >3. **事件循环：** 协程通常在事件循环中运行。事件循环负责调度和执行协程，使得程序能够高效地处理异步操作。
  >
  >    ```python
  >    import asyncio
  >          
  >    async def main():
  >        await foo()
  >        await bar()
  >          
  >    loop = asyncio.get_event_loop()
  >    loop.run_until_complete(main())
  >    ```
  >
  >上述代码中，`main` 函数是一个协程，通过事件循环的 `run_until_complete` 方法运行。在 `main` 函数中，`await foo()` 和 `await bar()` 表示在执行这两个协程时，事件循环会在遇到异步操作时挂起当前协程，转而执行其他协程，以达到并发执行的效果。
  >

- 怎么调用,首先是进入async模式,也就是进入python的event loop模式然后是将coroutine

- event loop的核心是有很多个task,然后决定那个task来运行

- 将coroutine变成task,然后可以排队执行

  - 第一个将coroutine变成task的方法就是使用关键字`await`

    >当使用await,coroutine被包装成task,并且告诉envent loop这里有一个新的task,使用awit

  - 

# python  关键字

## 1.False

布尔类型的值,表示假,与True相反

## 2.None

None比较特殊,表示什么也没有,其有自己的数据类型-NoneType

## 3.True

布尔类型的值,表示真,与False相反

## 4.and

用于表达式运算,逻辑与操作

## 5.as

用于类型转换,取别名

## 6.assert

断言,用于判断变量或者条件表达式的值是否为真

## 7.async

声明一个函数为异步函数

## 8.await

声明程序挂起

## 9.break

中断循环语句的执行

## 10.class

用于定义类

## 11.continue

跳出本次循环,继续执行下一次循环

## 12.def

用于定义函数或方法

## 13.del

删除变量或序列的值

## 14.elif

条件语句,与if,else结合使用

## 15.else

条件语句,与if,else结合使用,也可用于异常和循环语句

## 16.except

except包含捕获异常后的操作代码块,与try,finally结合使用

## 17.finally

用于异常语句,出现异常后,始终要执行finally包含的代码块,与try,except结合使用

## 18.for

for循环语句

## 19.from

用于导入模块,与import结合使用

## 20.global

定义全局变量

## 21.if

条件语句,与else, elif结合使用

## 22.import

用于导入模块,与from结合使用

## 23.in

判断变量是否在序列中

## 24.is

判断两个实例是否是同一个

## 25.lambda

定义匿名函数

## 26.nonlocal

用于标识外部作用域的变量

## 27.not

用于表达式运算,逻辑或非操作

## 28.or

用于表达式计算,逻辑或操作

## 29.pass

空的类,方法或函数的占位符

## 30.raise

异常抛出操作

## 31.return

用于从函数返回计算

## 32.while

while循环语句

## 33.with

`with`语句在 Python 中用于简化资源管理，主要用于对资源进行获取和释放。
