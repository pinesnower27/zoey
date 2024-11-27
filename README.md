# zoey

# 使用：
1. 在终端创建虚拟环境
以macOS为例：
```bash
cd zoey
python -m venv .venv
```
2. 在PyCharm打开项目（会自动加载虚拟环境）
3. 在终端（虚拟环境内）安装依赖
```bash
pip install -r requirements.txt
```

# 文件或目录说明：
## `./task.todo.md`: 
任务文件，文件内有`[ ] 任务1`这种东西，如果完成了任务请将其修改为`[x] 任务1`，同时将下面的`[ ] 任务1`修改为`[x] 20241027`(20241027应为完成日期)

详细请见`./docs/task.todo.demo.md`

## `./MDs/`
Means that McDonald's(bushi), The real meaning is "Models"

该目录下将模型进行了基础分类，实现的功能分类。

如其下的`./MDs/linear/`实现的是线性回归等功能。

几乎每个二级目录下都会有一个`utils`目录，这是一个可以被用到的小工具箱。

