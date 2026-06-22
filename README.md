# AI谣言检测器 (AI Rumor Detector)

基于 TextCNN 和 RAG（检索增强生成）技术的谣言检测与解释系统。

---

##  项目简介

本项目是一个融合深度学习与检索增强生成（RAG）技术的谣言检测系统。系统首先使用 **TextCNN** 模型对输入文本进行二分类，判断其是否为谣言；随后利用 **LLM（大语言模型）** 结合 **RAG 技术**，从自建的知识库、案例库和语言学特征库中检索相关证据，生成对分类结果的可解释性分析。

### 核心特性

-  **TextCNN 分类**：基于卷积神经网络的谣言二分类
-  **RAG 增强解释**：结合检索增强生成技术，提供可解释的判断依据
-  **多源知识库**：整合案例库、知识库和语言学特征库，支持多角度证据检索
-  **多种输入方式**：支持控制台交互、文本文件和 JSON 文件输入

---

##  目录结构

```
AI_Rumor_Detector
├─ README.md                      # 项目说明文档
├─ best_textcnn.pt                # 预训练的 TextCNN 模型权重
├─ data                           # 数据文件夹
│  ├─ pheme_faknow.json           # PHEME 数据集
│  ├─ test.json                   # 测试集
│  ├─ train.json                  # 训练集
│  └─ val.json                    # 验证集
├─ db                             # 自建向量数据库（需另外下载）
│  ├─ chroma_fever_knowledge      # FEVER 知识库
│  ├─ chroma_linguistic_features  # 语言学特征库
│  ├─ chroma_pheme_cases          # Pheme,liar,isot 案例库
│  └─ glove.6B.100d.txt           # 预训练向量
├─ requirements.txt               # Python 依赖包
├─ src                            # 核心源代码
│  ├─ EvdienceRAG.py              # RAG 证据检索与生成模块
│  ├─ extract_data.py             # 数据提取工具
│  ├─ main.py                     # 主程序入口
│  ├─ split_data.py               # 数据集划分工具
│  ├─ sth.py                      # TextCNN 构建模块
│  └─ train_cnn.py                # TextCNN 数据
└─ utils                          # 数据库构建工具
   ├─ fever.py                    # FEVER 数据集处理
   ├─ isot.py                     # ISOT 数据集处理
   ├─ liar.py                     # LIAR 数据集处理
   ├─ linguistic.py               # 语言学特征构建
   └─ pheme.py                    # PHEME 数据集处理
```

> **注意**：`db/` 文件夹为自建向量数据库，包含 Chroma 向量库文件，预训练向量**需要另外下载**。
>
>  **下载地址**：[https://pan.sjtu.edu.cn/web/share/fd5cc6c83ed5adffabb5393060d545d5](https://pan.sjtu.edu.cn/web/share/fd5cc6c83ed5adffabb5393060d545d5)  

---

## 环境要求

| 项目         | 要求                              |
| ---------- | ------------------------------- |
| **Python** | 3.10+                           |
| **CUDA**   | 需要（用于 PyTorch 和 LangChain 相关依赖） |

### 主要依赖

- `torch` — PyTorch 深度学习框架
- `langchain*` — LangChain RAG 框架及相关组件
- `chromadb` — 向量数据库
- 其他依赖详见 `requirements.txt`

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/zhw-345/AI_Rumor_Detector.git
cd AI_Rumor_Detector
```

### 2. 下载数据库文件

访问 [上海交通大学云盘](https://pan.sjtu.edu.cn/web/share/fd5cc6c83ed5adffabb5393060d545d5)，下载 `db/` 文件夹内容，解压后放置到项目根目录下。目录结构应为：

```
AI_Rumor_Detector-rag/
└─ db/
   ├─ glove.6B.100d.txt/
   ├─ chroma_fever_knowledge/
   ├─ chroma_linguistic_features/
   └─ chroma_pheme_cases/
```

### 3. 运行程序

```bash
# 方式一：控制台交互模式
python ./src/main.py -i console

# 方式二：JSON 文件输入（带 LLM 解释）
python ./src/main.py -i json -f ./example.json

# 方式三：JSON 文件输入（静默模式，仅输出分类结果）
python ./src/main.py -i json -f ./example.json -s

# 方式四：文本文件输入
python ./src/main.py -i txt -f ./example.txt
```

---

## 帮助信息

```
TextCNN 推理程序

options:
  -h, --help            显示帮助信息并退出
  -i {console,txt,json}, --input_type {console,txt,json}
                        输入类型：console（控制台交互）、txt（文本文件）、json（JSON文件）
  -f FILE, --file FILE  当 input_type 为 txt 或 json 时，指定文件路径
  -s, --silent          静默输出，不使用 LLM 解释
```

---

##  数据格式

### JSON 输入格式

`example.json` 应为 JSON 数组，每个元素包含 `text`（文本内容）和 `label`（标签，0 为真实，1 为谣言）字段：

```json
[
  {
    "text": "Charlie Hebdo will publish this week...",
    "label": 0
  }
]
```

### 数据集说明

| 文件 | 说明 |
|------|------|
| `train.json` | 训练集，用于 TextCNN 模型训练 |
| `val.json` | 验证集，用于模型调参和早停 |
| `test.json` | 测试集，用于最终模型评估 |
| `pheme_faknow.json` | PHEME 数据集 FakeNews 格式版本 |

> 这些数据集从老师下发数据集切分而来

---

## 向量数据库说明

`db/` 文件夹包含三个 Chroma 向量数据库，用于 RAG 检索：

| 数据库                          | 内容        | 用途          |
| ---------------------------- | --------- | ----------- |
| `chroma_fever_knowledge`     | FEVER 知识库 | 提供事实性知识检索   |
| `chroma_linguistic_features` | 语言学特征库    | 提供谣言语言学模式检索 |
| `chroma_pheme_cases`         | PHEME等案例库 | 提供历史相似案例检索  |

---

##  模型训练

如需重新训练 TextCNN 模型：

```bash
python ./src/sth.py
```

> **注意**：运行main.py时不会训练TextCNN 模型，若使用扩展的PHEME数据集或其他数据集，必须重新训练模型。规定格式详见`./src/extract_data.py`和`split_data.py`。

---

## 单独使用LLM分析

```shell
python ./src/EvdienceRAG.py
```

> **注意**：申请学校API，使用deepseek模型，6月底过期，此后如需LLM解释器请修改`EvdienceRAG.py`配置。

---

##  模型性能

| 指标              | 数值     |
| --------------- | ------ |
| 准确率 (Accuracy)  | 0.8257 |
| 召回率 (Recall)    | 0.8501 |
| F1 分数           | 0.8377 |

