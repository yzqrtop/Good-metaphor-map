# GoodM - Metaphor Analysis System

GoodM (Good Metaphor) 是一个模块化的 Python 项目，用于隐喻生成、评估和分析，支持论文中"方法+实验"部分的实现。

## 项目结构

```
goodm/
├── core/                    # 核心数据结构
│   ├── __init__.py
│   └── schema.py            # GoodM七元组 (E, V, EE, VE, C, Sc, Sy)
│
├── generation/              # 数据生成流水线
│   ├── __init__.py
│   └── pipeline.py          # 四阶段生成 + 多轮共识 + 一致性检查
│
├── models/                  # 模型封装
│   ├── __init__.py
│   ├── llm_wrappers/        # GPT-4, Qwen 等 LLM 包装器
│   │   ├── __init__.py
│   │   ├── base.py          # 基类定义
│   │   ├── qwen.py          # 通义千问
│   │   └── gpt4.py          # GPT-4
│   ├── multimodal/          # 多模态模型 (BART + ViT)
│   └── semantic/            # 语义模型 (DeepMet-S, Jina Embedding)
│
├── evaluation/              # MiniEvalFeature 评估
│   ├── __init__.py
│   ├── mef.py               # 7指标加权聚合
│   └── metrics/             # f1~f7 各指标实现
│
├── tasks/                   # 下游任务
│   ├── __init__.py
│   └── intent.py            # 意图预测
│
├── data/                    # 数据集
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后数据
│   └── prompts/             # 生成提示模板
│       ├── entity_expansion.txt
│       ├── commonality_extraction.txt
│       ├── scene_generalization.txt
│       └── synthesis.txt
│
└── experiments/             # 实验脚本
    ├── __init__.py
    └── ablation.py          # 消融实验
```

## 核心概念

### GoodM 七元组

GoodM 使用七元组表示隐喻的完整语义结构：

- **E** (Entity/本体): 歇后语的前半部分，被比喻的事物
- **V** (Vehicle/喻体): 歇后语的后半部分，用来比喻的事物
- **EE** (Entity Explanation/本体阐释): 对本体的详细解释
- **VE** (Vehicle Explanation/喻体阐释): 对喻体的详细解释
- **C** (Commonality/共性特征): EE 和 VE 的共享特征
- **Sc** (Scenario/场景泛化): 从 EE 和 VE 泛化出的场景
- **Sy** (Synthesis/综合意义): 综合 Sc 和 C 的深层含义

## 快速开始

### 安装依赖

```bash
pip install -e .
```

### 基础使用

```python
from goodm import GoodMTuple, MEF

# 创建七元组
tuple_obj = GoodMTuple(
    E="竹篮",
    V="一场空",
    EE="用竹子编织的篮子，质地疏松，无法盛水",
    VE="什么都没有，完全落空的状态",
    C="无法保持或留住",
    Sc="努力付出但没有任何收获的场景",
    Sy="徒劳无功，白费力气"
)

# 验证
print(tuple_obj.validate())  # True

# 评估
mef = MEF()
scores = mef.evaluate(tuple_obj)
print(f"MEF 总分: {scores['MEF']}")
```

### 生成七元组

```python
import asyncio
from goodm import GenerationPipeline
from goodm.models.llm_wrappers import QwenWrapper

async def generate():
    # 初始化 LLM
    llm = QwenWrapper(model_name="qwen-plus")
    
    # 创建流水线
    pipeline = GenerationPipeline(
        llm_client=llm,
        temperature=0.7,
        consensus_rounds=3
    )
    
    # 生成
    result = await pipeline.generate(E="竹篮", V="一场空")
    print(result.to_prefix())

asyncio.run(generate())
```

### 消融实验

```python
from goodm import MEF, AblationStudy

# 准备测试数据
test_tuples = [...]  # List[GoodMTuple]

# 运行消融实验
mef = MEF()
ablation = AblationStudy(mef)

# 留一法
results = ablation.leave_one_out(test_tuples)

# 全部实验
all_results = ablation.run_all_experiments(
    test_tuples,
    output_path="results/ablation.json"
)
```

## 模块说明

### core/schema.py

定义 `GoodMTuple` 类，提供：
- 七元组数据验证
- 前缀格式转换 (`to_prefix()`)
- 嵌入输入格式 (`to_embedding_input()`)
- JSON 序列化/反序列化

### generation/pipeline.py

实现四阶段生成流水线：
1. **Entity/Vehicle Expansion**: 多轮共识生成 EE 和 VE
2. **Commonality Extraction**: 提取共享特征 C
3. **Scene Generalization**: 泛化场景 Sc
4. **Synthesis**: 综合生成 Sy

包含两阶段一致性检查：
- 自动逻辑验证
- 人工修正（验证失败时）

### evaluation/mef.py

MiniEvalFeature 评估器，7个指标：

| 指标 | 名称 | 权重 | 说明 |
|-----|------|-----|------|
| f1 | 认知准确性 | 0.20 | EE与E、VE与V的语义相似度 |
| f2 | 特征匹配度 | 0.15 | EE与VE的相似度 |
| f3 | 复合对齐度 | 0.15 | Sy与{EE, VE}的对齐度 |
| f4 | 概念深度 | 0.15 | C的长度和复杂度 |
| f5 | 场景相关性 | 0.15 | Sc与{EE, VE}的相关性 |
| f6 | 象征共鸣 | 0.10 | Sy与C的相似度 |
| f7 | 整体连贯性 | 0.10 | 字段完整性和内部一致性 |

### tasks/intent.py

意图预测任务：
- 输入: 隐喻文本 + GoodM七元组（可选）
- 输出: educate / persuade / warn / entertain / other
- 基线: BERT 76.2% → GoodM增强 88.5%

### experiments/ablation.py

消融实验：
- `leave_one_out()`: 每次移除1个组件
- `pairwise_combination()`: 测试21种两两组合
- `remove_key_components()`: 同时移除 {C, Sc, Sy}

## 数据流

```
[原始数据] ──→ [GenerationPipeline] ──→ [两阶段一致性检查]
                                              ↓
                                    [通过] → [GoodM数据集]
                                    [失败] → [人工修正]
                                              ↓
[GoodM数据集] ──→ [MEF评估] ──→ [筛选高分样本]
                                              ↓
                              [拼接为前缀] 或 [Jina向量]
                                              ↓
                    [基线模型] ←────→ [GoodM增强模型]
                                              ↓
                              [消融实验 + 敏感性分析]
```

## 示例

查看 `examples/` 目录：
- `basic_usage.py`: 基础使用示例
- `ablation_study.py`: 消融实验示例

## 许可证

MIT License
