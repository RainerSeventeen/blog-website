# C++ STL

## `std::vector`

### `for`循环

cpp 支持了一些新特性可以访问元素：

```cpp
for (auto p : nums) {
    ...
}
```

但是这个访问速度其实是比传统的下标访问要慢的，因为每一次访问都会复制一份，而不是原地访问

另外的一点同样需要注意，就是按照数值访问而不是复制

```cpp
int (vector<int> a)
int (vector<int>& a) // 会快很多
```

### 代替栈

在写代码的时候经常使用 `vector` 来代替 `stack` ，因为这两个的操作实际上是差不多的（甚至 `vector` 性能会更好一点）

```c++
vector<int> stk;
stk.reserve(10) // 预留10个空间，但是不影响 back 数值
if (!stk.empty()) {
	int top = stk.back(); // stack.top()
    stk.pop_back();	// stack.pop()
}
int n = 0;
stk.push_back(n);	// stack.push(n)
```

### `reserve()`

`reserve(n)` 表示为这个容器预留空间，但是没有创建对应的元素

| **写法**                     | **size()** | **capacity()** | **是否有元素** |
| ---------------------------- | ---------- | -------------- | -------------- |
| `vector<T> v; v.reserve(n);` | 0          | ≥ n            | ❌              |
| `vector<T> v(n);`            | n          | ≥ n            | ✅              |

通过 `v.back()`访问不可以是空元素，即便已经 `reserve` 过

使用 `vector` 模拟栈的时候务必要使用 `reserve()`，不然所有的 `push_back()`等涉及队尾的操作都要发生错误







## `std::string`

### `string.find()`

在字符串中寻找字串，和普通的迭代器返回值是不同的

[cpp reference](https://en.cppreference.com/w/cpp/string/basic_string/find.html)

在找到时候返回字串的下标，否则返回 `std::string::npos`

包含一些优化的算法，例如 KMP

### 添加字符

使用 `append` 成员函数可以末尾添加

注意：`append` 不能添加 `char`

```c++
#include <string>
#include <iostream>
using namespace std;

int main() {
    string s = "hello";
    s += "_world";             // 方式一：运算符
    // s.append("_world");     // 方式二：append() 函数
    cout << s << endl;         // 输出：hello_world
}
```

使用 `insert` 成员可以在特定位置插入

```c++
s.insert(pos, "text");  // 在索引 pos 前插入字符串
```

### 删除字符

删除末尾给定的长度，可以用通用的 `erase()`

```c++
s.erase(s.size() - 6);  // 删除最后 6 个字符
```

## `std::pair`

对于存放坐标这种数据的时候，可以和 `queue` 联动，例如

```c++
queue<pait<int, int>> que;
que.push({x, y}); // 原地构造直接放入

auto [cx, cy] = que.front(); // 取出数据
que.pop();
```







##  `std::deque`

双端队列，是 std::stack 的底层逻辑实现，允许进行前后双端口的出入操作

不同于 std::vector, `std::deque` 使用分段内存，迭代器在经过操作后会有可能失效       

让所有迭代器失效的操作：

- `deque.push_back()`

- `deque.push_bront()`

- `deque.emplace_back()`

- `deque.emplace_font()`

只会影响被删除及末尾迭代器：

- `deque.pop_fornt()`

- `deque.pop_back()`

## `std::priority_queue`

[cpp reference](https://en.cppreference.com/w/cpp/container/priority_queue.html)

优先级队列，可以按照优先级顺序出入队列

参考初始化方式

```C++
priority_queue<int, vector<int>, less<int>> q
// 最后一个类似于 sort 中的 比较规则函数

// 你可以自定义个一个类并重载 () 用于比较，需要注意成员操作符函数必须对外可见
// 运算符重载语法 定义
bool operator() (const int& a, const int &b)
```

小顶堆：定义重载 return a > b ，则每次弹出的时最小的数值



## `std::stoi` 与 ``std::to_string``

### `std::stoi`

把一个字符串转成 `signed int` 类型变量

```c++
#include <iostream>
#include <string>

int main() {
    std::string s = "123";
    int num = std::stoi(s);
    std::cout << num << std::endl;
    return 0;
}
```

若字符串中含非数字字符，`std::stoi` 会抛出 `std::invalid_argument` 或 `std::out_of_range` 异常。

可以使用 `try` 包裹

```c++
try {
    int num = std::stoi(s);
} catch (const std::exception& e) {
    std::cerr << "转换失败: " << e.what() << std::endl;
}
```

对于 `long`  和 `long long` 类型的转化，可以分别使用 `std::stol`, `std::stoll`

### `std::to_string`

把一个整数转成 字符串类型

```c++
#include <iostream>
#include <string>

int main() {
    int num = 123;
    std::string s = std::to_string(num);
    std::cout << s << std::endl;
    return 0;
}
```

如果需要控制格式可以使用  `std::ostringstream`

