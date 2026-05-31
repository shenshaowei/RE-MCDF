# RE-MCDF

基于知识图谱增强大语言模型的电子病历诊断系统

## 项目简介

RE-MCDF (Retrieval-Enhanced Multi-modal Clinical Diagnosis Framework) 是一个基于知识图谱增强的大语言模型电子病历诊断系统。该项目整合了医学知识图谱、检索增强生成（RAG）和多模态信息处理技术，用于辅助临床诊断。

**论文**：[RE-MCDF: Closed-Loop Multi-Expert LLM Reasoning for Knowledge-Grounded Clinical Diagnosis](https://arxiv.org/pdf/2602.01297)

**数据集参考**：[medIKAL](https://github.com/CSU-NLP-Group/medIKAL/)

## 项目结构

```
RE-MCDF/
├── src/                    # 源代码
│   ├── configs/           # 配置文件
│   │   └── config_e2e/   # 主配置文件
│   │       ├── config_main.py      # STROKE 数据集配置
│   │       └── config_main_2.py   # XMEMRs 数据集配置
│   ├── main/             # 主程序
│   │   ├── main.py       # 主入口
│   │   ├── main1.py      # 变体版本1（推荐使用）
│   │   ├── main2.py      # 变体版本2
│   │   ├── doctor.py     # 医生 Agent
│   │   ├── kg_func.py    # 知识图谱工具
│   │   ├── retriever.py  # 检索器
│   │   ├── ner.py        # 命名实体识别
│   │   ├── models.py     # 模型定义
│   │   ├── utils.py      # 工具函数
│   │   ├── config.py     # 配置加载
│   │   └── build_dense_retriever.py
│   └── evaluate/          # 评估脚本
│       ├── evaluate_f1_medikal.py
│       ├── evaluate_emr.py
│       └── ...
├── data/                  # 数据目录
│   ├── STROKE/           # 脑血管病数据集
│   ├── XMEMRs/           # 影像病历数据集
│   ├── CMEMR/            # 完整病历数据集
│   ├── KG_entities2id_merge.txt    # 知识图谱实体ID
│   └── entity_type_map_merge.json   # 实体类型映射
├── opendata/              # 公开可共享的数据
├── kgfiles/               # 知识图谱文件（运行脚本后生成）
├── models/                # 模型文件（需下载）
├── output/                # 输出目录
└── requirements.txt       # 依赖包
```

## 环境配置

### 1. 创建虚拟环境

```bash
conda create -n remcdf python=3.10
conda activate remcdf
pip install -r requirements.txt
```

### 2. 准备知识图谱数据

#### 方式一：使用脚本生成

```bash
# 1. 申请下载 CPubMed-KG
#    https://cpubmed.openi.org.cn/graph/wiki
#    将 CPubMed-KGv2_0.txt 放到项目根目录的 data/ 下

# 2. 运行脚本生成 Neo4j 导入文件
python opendata/create_nodes_relationships.py
```

#### 导入 Neo4j

```bash
# 请确保先停止 Neo4j
neo4j-admin import \
   --database=medkg \
   --nodes="./kgfiles/nodes.csv" \
   --relationships="./kgfiles/relationships.csv" \
   --id-type=INTEGER \
   --delimiter="|"
```

#### 启动 Neo4j

```bash
# 直接安装
neo4j start

# 访问 http://localhost:7474
```

### 3. 下载模型文件

前往 ModelScope 下载以下模型：

- **Qwen2.5-7B-Instruct**：大语言模型
- **bge-small-zh-v1.5**：中文向量模型
- **nlp_raner_named-entity-recognition_chinese-base-cmeee**：医学实体识别模型

### 4. 修改配置文件

编辑 `src/configs/config_e2e/config_main.py`，根据实际情况修改路径：

```python
config_dict["model_name_or_path"] = "./models/Qwen2.5-7B-Instruct"
config_dict["bge_model_path"] = "./models/bge-small-zh-v1.5"
config_dict["icd_database_path"] = "./opendata/国际疾病分类ICD-10北京临床版v601.xls"
config_dict["ner_model_id"] = "./models/nlp_raner_named-entity-recognition_chinese-base-cmeee"
```

## 运行项目

### 运行主程序

```bash
cd src/main
python main1.py
```

### 运行评估

```bash
cd src/evaluate
python evaluate_f1_medikal.py
```

## 数据说明

### 输入数据格式

项目接受 JSON 格式的病历数据，详细格式见 `opendata/example_data.json`

```json
{
  "基本信息": "女，58岁",
  "主诉": "反复胸闷、心悸2年，加重伴气短1周",
  "现病史": "患者2年前...",
  "查体": "T 36.5℃，P 88次/分...",
  "辅助检查": "心电图示：...",
  "label": ["冠状动脉粥样硬化性心脏病", "心力衰竭"]
}
```

### 数据来源

由于版权和隐私限制，原始病历数据需从以下网址获取：
```
https://bingli.iiyi.com/show/{emrid}-1.html
```
参考：[medIKAL Dataset](https://github.com/CSU-NLP-Group/medIKAL/)

## 核心功能

1. **知识图谱增强**：利用医学知识图谱提供诊断推理支持
2. **检索增强生成**：结合 BM25 和向量检索
3. **多路径推理**：支持多跳知识路径搜索
4. **动态权重**：根据上下文动态调整实体权重

## 许可证

本项目仅供学术研究使用。
