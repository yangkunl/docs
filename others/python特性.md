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
