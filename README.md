## 项目概述

这个项目涉及两个主要部分：`data.py` 和 `transformer.py`。其中`data.py` 包含数据处理模块。 `transformer.py`包含transformer模型，训练和预测模块。

### 文件结构

项目的文件结构如下：

- `data.py`: 包含数据集下载和数据集读取的模块。
- `transformer.py`: 包含Transformer模型的定义，以及训练和预测的模块。
- `data/`: 存放数据集的文件夹。
- `README.md`: 项目说明文档，你正在阅读的内容。

## 数据集

项目已经包含了所需的数据集，无需手动下载。数据集文件存放在`data/`文件夹中。

`cn.txt.vocab.tsv`与`en.txt.vocab.tsv`中每个词语后面的数字表示在训练文本中的出现次数。其中前几个特殊符号的含义如下：

- `<PAD>`: 填充符号。
- `<UNK>`: 未知符号。
- `<S>`: 句子开始。
- `</S>`: 句子结束。

`cn.txt`与`en.txt`中每行为一句话，一句话中任何两两相邻的词语和标点符号之间都使用空格分隔，以便于分词。

### 训练

要训练模型，请执行以下命令：

```bash
python transformer.py --cfg train
```

### 预测

要进行预测，请执行以下命令：

```bash
python transformer.py --cfg predict
```

注意训练与预测的具体参数设置都位于`transformer.py`中。
