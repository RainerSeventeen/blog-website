# Python in LeetCode 

> 在这里不写特别基础的语法，主要是写 LeetCode 遇到的一些可用的 api
>
> 自己是 cpp 学起的，难免有点 cpp 的代码思维，这里放一点地道的 python 写法指南

## 1 语法

### 1.1 `for`

在 py 中有一个坑，`for` 语法内部改变的 `i` 是不会影响外层的迭代的

```python
for i in range(9):
	...
	i = 9 # 实际上没有作用，仍然是循环 9 次
```

### 1.2 `while`

`while` 有一个 `else` 的用法，表示没有经过 `break` 的正常语法结束

```python
i = 0
while i < 3:
    print(i)
    i += 1
else:
    print("loop finished normally") 	# 会被答打印，因为不是由于 break 退出的
```

### 1.3 `lambda` 表达式

实现临时函数的功能

```python
lambda 参数列表: 表达式
# 等价于
def f(参数列表):
    return 表达式
```

`lambda` 的返回值永远是那个表达式，没法更细化的自定义，例如用三元表达式实现条件判断逻辑

```python
lambda x: x if x > 0 else -x
```

### 1.4 数值

python 中的 int 是无限精度的，所以没有最小值，一般这么指定最小值

```python
ret = float("-inf")
```

### 1.5 位运算

python 中的位运算和 c++ 的差异不是特别大

| **运算** | **符号** | **含义** | **示例** |
| -------- | -------- | ----------------- | ----------- |
| 按位与 | & | 两位都为 1 才为 1 | 5 & 3 = 1 |
| 按位或 | \| | 有一位为 1 即为 1 | 5 \| 3 = 7 |
| 按位异或 | ^ | 两位不同为 1 | 5 ^ 3 = 6 |
| 按位取反 | ~ | 逐位取反 | ~5 = -6 |
| 左移 | << | 左移并补 0 | 5 << 1 = 10 |
| 右移 | >> | 算术右移 | 5 >> 1 = 2 |

位运算有一些神奇的用法

例如 XOR（异或）可以在 LeetCode 136 中使用

另外也有统计 1 个数的方法

```python
x.bit_count() # 统计 1 的个数
```

### 进制转化

```python
int("1011", 2)  # str 进制转化 int
int("0b1011", 0) # 前缀自动识别转化
# int 转 str, 使用 format
format(10, 'b')   # '1010'
format(10, 'o')   # '12'
format(255, 'x')  # 'ff'
format(255, 'X')  # 'FF'
format(5, '08b')  # '00000101' 补充0
# 或者可以用 fstring 格式化
f"{10:b}"
f"{10:08b}"
```



### 1.6 深拷贝与浅拷贝

python 中的 `copy` 库可以对对象进行拷贝

`copy` 只复制容器本身，子对象仍共享； `deepcopy` 递归复制整个对象结构，完全独立

`list` 的 `copy` 方法就是 浅拷贝

```python
import copy
b = [2, 3]
a = [1, b]

a1 = copy.copy(a) # a1 = a.copy() 也是同样的效果
a2 = copy.deepcopy(a)

b.append(4)
print(a)   # [1, [2, 3, 4]]
print(a1)  # [1, [2, 3, 4]]  ← 被一起改了
print(a2)  # [1, [2, 3]]     ← 不受影响
```


-----


## 2 常规数据结构

### 2.1 `list`

#### 2.1.1 增删改查

增加：支持末尾和中间

`l.insert(k, x)` 表示的是把 `x` 插入到当前的 `k` 下标的左侧，原来的从 `k` 开始的元素全部向后移动一位

如果插入的位置 `k > len(l)` 则会直接 `append` 到队伍尾部

```python
lst = [1, 2]
lst.append(3)          # [1,2,3]	# 在末尾增加一个元素
lst.extend([4, 5])     # [1,2,3,4,5]	# 在末尾增添一组元素
lst.insert(1, 99)      # [1,99,2,3,4,5] 在索引 1 之前加一个元素
```

删除：

```python
lst = [1, 2, 3, 2]
lst.remove(2)   # [1,3,2]	# 删除第一个等于 2 的元素，找不到会报错
x = lst.pop()   # x=2, lst=[1,3]	# 删除最后一个元素
del lst[0]      # lst=[3]	# 按照索引删除
```

#### 2.1.2 数组分配内存

这里有个坑就是对于二维以上的分配，是不同的写法

```python
n = 10
lst = [0] * n # 这是分配一个 size 的 10 的 list

dp = [[0] * n for _ in range(m)] # 这是二维的分配方法
```

这个的含义是创建 1 个长度为 `n` 的 `list`，然后让 `m` 个元素都指向它，是**引用复制**

```python
dp = [[0] * n] * m # 错误用法
```

那为什么一维不会出错呢？ **因为一维是 `int` 元素而不是 `list`的引用，不是指向同一个容器**

#### 2.1.3 访问数组下标

对于 cpp 中用索引访问的方式，使用 `enumerate`：

当然用 `range`也行（我更习惯这个）

```python
nums = [10, 20, 30]
for i, num in enumerate(nums):
    print(i ,num)
```

#### 2.1.4 `list` 排序

（默认升序）排序用 `nums.sort()`，也有更加通用的自定义算法，结合 `lambda` 表达式使用参数 `key=`

```python
# 初等用法
arr = [(1, 3), (2, 1), (4, 2)]
arr.sort(key=lambda x: x[1])
# [(2, 1), (4, 2), (1, 3)]
```

也有多关键字排序算法，如果第一个比较相同，就去执行第二个的比较

```python
arr.sort(key=lambda x: (key1, key2, key3)) # 对 tuple 进行字典序排序

# 首先按起点排序，如果起点相同按终点排序
intervals = [[1, 3], [2, 6], [1, 2]]
intervals.sort(key=lambda x: (x[0], x[1]))
# [[1, 2], [1, 3], [2, 6]]

arr.sort(key=lambda x: (x[0], -x[1])) 	# 起点升序，终点降序
# [(1, 3), (1, 1), (2, 2)]
```

#### 2.1.5 二分查找

```python
import bisect

a = [1, 2, 2, 2, 4, 5]

bisect.bisect_left(a, 2)   # 1
bisect.bisect_right(a, 2)  # 4

# 判断元素是否存在
i = bisect.bisect_left(a, x)
exists = i < len(a) and a[i] == x

# 查找给定数字的一个区间
l = bisect.bisect_left(a, x)
r = bisect.bisect_right(a, x) - 1

# 自定义比较法
a = [(x.val, x) for x in arr] # 重新构建为字典
bisect.bisect_left(a, (target, -inf)) # 根据 (target, -inf) 依次比较， -inf 为了不影响左边界
```

#### 2.1.6 数组求和

```python
sum(lst) # 直接求
sum(x for x in lst if x > 0) # 条件遍历求和
```

#### 2.1.7 模拟栈

栈使用的是 `list` 并且有以下几种 api 来执行相似的操作

同时需要注意一下几点：

1. 用 `-1`来访问 `top`
2. 巨大的语言差异：`python`中的 `pop` 是会返回被弹出的数据的

```python
stk = []
stk.append(x)     # push
top = stk[-1]     # top
top = stk.pop()   # pop, 注意这个值是会被返回的
```

### 2.2 `dict`

#### 2.2.1 增删改查

添加元素一般有两种方式
```python
a = {}
a['a'] = 1 # 直接赋值
a.setdefault('a', 1)	# 如果不存在，就赋值为1

# 一般统计 + 1 这种就可以使用
a['a'] = a.get('a', 0) + 1
```

删除元素使用 `del` 但是需要保证 key 存在

等价于在 cpp 中使用 `mp.erase(key);` 但是 cpp 不存在的时候不会报错

```python
d = {"a": 1, "b": 2}
del d["a"]
d.pop("a")          # 返回被删的 value
d.pop("x", None)    # key 不存在时不报错
d.clear() # 清空
```

查找可以用 `in`

```python
if k in d: # 表示从键中找目标
```

#### 2.2.2 排序

字典排序实际上不是对字典本身排序，而是通过排序创建一个新的 `list` 或者一个新的 `dict`

```python
mp = {'b': 2, 'a': 3, 'c': 1}
# 默认按照键排序（升序） [('a', 3), ('b', 2), ('c', 1)]
sorted(mp.items()) 	

# 按照值排序 [('c', 1), ('b', 2), ('a', 3)]
sorted(mp.items(), key=lambda kv: kv[1])


# 频率高在前面，次数相同用字典序
sorted(mp.items(), key=lambda kv: (-kv[1], kv[0]))

# 只返回键，不返回键值对， 因为遍历字典默认遍历 key, sorted 只会返回他遍历的值
sorted(dict_name, key=lambda k: (-dict_name[k], k))
```

#### 2.2.3 `defaultdict`

这是一个 dict 的子类型，增加了默认构建值的功能

这个功能只会在读取的时候生效，写入的时候和普通字典完全一致

```python
from collections import defaultdict
d = defaultdict(int)

x = d['a']

# 如果不存在 'a' 这个键,类似于这个步骤
if 'a' not in d:
    d['a'] = int()   # 这里才调用
x = d['a']
```

这里的初始化参数的意义是，如果访问一个不存在的值，就调用 `int()`，所以参数必须是一个可以调用的函数类似的东西

### 2.3 `str`

`str` 是不可修改的，这是最大最主要的和 cpp 的区别

下面都使用 `s = "hello world"` 的字符串作为演示

#### 2.3.1 ASCII 码转化

python 中没有 char 类型，直接进行 str 转化就可以拿到 unicode 的编码（包含 ASCII 码在其中）

```python
ord('A')   # 65
ord('a')   # 97
ord('0')   # 48
ord('\n')  # 10

chr(65)   # 'A'
chr(97)   # 'a'
chr(48)   # '0'
```

#### 2.3.2 求子串

用切片（切片是左闭右开的）

```python
s[start:end:step]
s[0:5]     # 'hello'
s[:5]      # 'hello'
s[6:]      # 'world'
s[::-1]    # 'dlrow olleh'（反转）
```

#### 2.3.3 大小写转化

```python
s.upper()       # 'HELLO WORLD'
s.lower()       # 'hello world'
s.capitalize()  # 'Hello world'
s.title()       # 'Hello World'
s.swapcase()    # 'HELLO WORLD' -> 'hello world'
```

#### 2.3.4 查找

```python
s.find("world")     # 6，找不到返回 -1

# 是否存在于
"world" in s    # True
```

#### 2.3.5 替换，修改

```python
s.replace("world", "python")  # 'hello python'
```

#### 2.3.6 去除字符

```python
"xxhelloxx".strip("x")  # 'hello'
```

#### 2.3.7 分割与拼接

```python
s = "a,b,c"
s.split(",")    # ['a', 'b', 'c']

lst = ["a", "b", "c"]
",".join(lst)   # 'a,b,c'
```

#### 2.3.8 多次修改

因为 python 中没有办法原地修改某一个字符，所以这里特地写一下怎么等价实现这种方式

1. 使用切片，一次修改 `O(n)`

```python
s = "abcd"
i = 3
s = s[:i] + 'X' + s[i+1:]
```

2. 转化 `list`，在频繁修改的时候比较好用，单次修改`O(1)`, 合并`O(n)`

```python
s = "abcd"
lst = list(s)

lst[3] = 'X'
s = "".join(lst)
```

### f string 格式化

f string 可以经常用来输出数字格式化

```python
# {变量:填充符 对齐 宽度 类型}
f"{5:0>3d}" 
print(f"{5:0>5d}") # 数字 5， 使用 0 填充，右侧对齐，填充到 5 宽度，使用十进制
# 00005
x = -5
print(f"{x:03d}") # 0-5
```



### 2.4 `set`

就是哈希，只有 key，没有 value

下面是常用的创建，增删，查找操作

```python
s = set()                 # 空集合
s = {1, 2, 3}             # 字面量
s = set([1, 2, 2, 3])     # 自动去重
s = set("abc")            # {'a', 'b', 'c'}

s.add(x)                  # 添加单个元素
s.update([a, b, c])       # 批量添加（可迭代对象）

s.discard(x)              # x 不存在不会报错 remove 会报错
s.pop()                   # 随机弹出一个元素
s.clear()                 # 清空

x in s        # O(1)，最常用
x not in s
len(s)
```

### 2.5 `tuple`

`tuple` 在 python 中类似于一个不可以修改的`list`，但是如果要让 hash 的 key 是类似于 list 的方式就只能使用 tuple

一般来说我们会先写一个 list 随后 tuple 化

```python
a = [1, 2, 3]
a[1] = 3
t = tuple(a)
```

-----


## 3 API 数据结构

> `collections` 中一般来说， 数据类型会被写成首字母大写，但是 `deque`, `defaultdict`这种除外，这种属于历史遗留问题

### 3.1 `deque`

这是一个双端队列，默认的话是左进右出的，访问数据方式也等同于 `list`

```python
from collections import deque

q = deque()

q.append(x)        # 右入
q.appendleft(x)    # 左入
q.pop()            # 右出
q.popleft()        # 左出

# 访问
left  = q[0]    # 左端
right = q[-1]   # 右端
```

### 3.2 `OrderedDict`

这个主要用于 LRU 的内部数据结构，是 hash + 双向链表

```python
from collections import OrderedDict

od = OrderedDict()

od[key] = val                 # 插入（默认在末尾）
od.move_to_end(key)           # 刷新为最近使用
od.pop(key)                   # 删除任意 key

od.popitem(last=True)         # 默认，删末尾
od.popitem(last=False)        # 删头部
od.popitem(last=False)        # 弹出最久未使用（LRU）
```

### 3.3 `heapq`

这个 api 的操作方式不是创建一个类，而是对已有的 list 进行建堆以及执行其他操作

Top - K 常用数据结构，这个是小顶堆，一般建议要会写这个底层实现的

```python
heap = []
heapq.heappush(heap, x)      # 入堆
x = heapq.heappop(heap)      # 弹出最小值
x = heap[0]                  # 查看最小值（不弹）

# 替换堆顶
heapq.heapreplace(heap, x)
```

对于大顶堆我们一般用这种构造负 key 的方式，注意 `heapq` 是按照 `tuple` 字典序（即相同就比较下一个）

```python
heapq.heappush(heap, (-key, value))
key, value = heapq.heappop(heap)
key = -key
```

### 3.4 `Counter`

用来统计一个 `list` 中出现的次数

```python
from collections import Counter

nums = [1, 1, 2, 2, 2, 3]
cnt = Counter(nums)

cnt[2]      # 3
cnt[99]     # 0 不存在时返回 0，而不是 KeyError

for num, freq in cnt.items():
    pass # 访问方式如同字典
```

## 4 自定义数据结构

### 4.1 链表

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
dummy = ListNode(0)
dummy.next = head
...
```
