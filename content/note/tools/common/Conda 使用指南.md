# Conda 使用指南

conda 是一个包管理工具，使用起来有点类似于 python 的 venv

## 环境管理

**创建一个名为 "myenv" 的新环境:**

```
conda create --name myenv
```

**创建指定版本的环境**：

```
conda create --name myenv python=3.8
```

以上代码创建一个名为 "myenv" 的新环境，并指定 Python 版本为 3.8。

**激活环境：**

```
conda activate myenv
```

以上代码激活名为 "myenv" 的环境。

**要退出当前环境使用以下命令：**

```
deactivate
```

**查看所有环境：**

```
conda env list
```

以上代码查看所有已创建的环境。

**复制环境：**

```
conda create --name myclone --clone myenv
```

以上代码通过克隆已有环境创建新环境。

**删除环境：**

```
conda env remove --name myenv
```

以上代码删除名为 "myenv" 的环境。

## 包管理

**安装包：**

```
conda install package_name
```

以上代码安装名为 "package_name" 的软件包。

**安装指定版本的包：**

```
conda install package_name=1.2.3
```

以上代码安装 "package_name" 的指定版本。

**更新包：**

```
conda update package_name
```

以上代码更新已安装的软件包。

**卸载包：**

```
conda remove package_name
```

以上代码卸载已安装的软件包。

**查看已安装的包：**

```
conda list
```

查看当前环境下已安装的所有软件包及其版本。

## 其他常用命令

**查看帮助：**

```
conda --help
```

以上代码获取 conda 命令的帮助信息。

**查看 conda 版本：**

```
conda --version
```

以上代码查看安装的 conda 版本。

**搜索包：**

```
conda search package_name
```

以上代码在 conda 仓库中搜索指定的软件包。

**清理不再需要的包：**

```
conda clean --all
```

以上代码清理 conda 缓存，删除不再需要的软件包。