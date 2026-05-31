# RE-MCDF 开放数据

> [English](README.md)

本项目所需的数据文件和脚本。

## 📁 文件

| 文件 | 说明 |
|------|------|
| `create_nodes_relationships.py` | 从 CPubMed-KG 生成 Neo4j 导入文件 |
| `create_ref.py` | 生成参考文献数据 |
| `entity_type_map_merge.py` | 提取实体 ID 和类型映射 |
| `entity_weight_emr.json` | 医学实体权重配置 |
| `国际疾病分类ICD-10北京临床版v601.xls` | ICD-10 疾病编码数据库 |
| `example_data.json` | 示例输入数据格式 |
| `evaluate_f1.py` | F1 评估脚本 |
| `CMEMR-Cerebrovascular/` | 脑血管病例 ID 列表 |

## 🚀 快速开始

### 构建知识图谱

```bash
# 1. 下载 CPubMed-KG: https://cpubmed.openi.org.cn/graph/wiki
# 2. 运行脚本
python create_nodes_relationships.py
```

### 导入 Neo4j

```bash
neo4j-admin import \
   --database=medkg \
   --nodes="./kgfiles/nodes.csv" \
   --relationships="./kgfiles/relationships.csv" \
   --id-type=INTEGER \
   --delimiter="|"
```

### 运行评估

```bash
python evaluate_f1.py
```

## 📖 数据来源

- **知识图谱**: [CPubMed-KG](https://cpubmed.openi.org.cn/graph/wiki)（哈尔滨工业大学）
- **病历数据**: [medIKAL Dataset](https://github.com/CSU-NLP-Group/medIKAL/) - 原始病历可在 `https://bingli.iiyi.com/show/{emrid}-1.html` 获取
- **ICD-10**: 北京临床版 v6.01

## 📎 链接

- [论文](https://arxiv.org/pdf/2602.01297) · [代码](https://github.com/shenshaowei/RE-MCDF)
