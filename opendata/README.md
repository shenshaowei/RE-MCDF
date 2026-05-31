# RE-MCDF Open Data

> [中文](README_zh.md)

Data files and scripts for the [RE-MCDF](https://github.com/shenshaowei/RE-MCDF) project.

## 📁 Files

| File | Description |
|------|-------------|
| `create_nodes_relationships.py` | Generate Neo4j import files from CPubMed-KG |
| `create_ref.py` | Generate reference data |
| `entity_type_map_merge.py` | Extract entity ID and type mappings |
| `entity_weight_emr.json` | Medical entity weight configuration |
| `国际疾病分类ICD-10北京临床版v601.xls` | ICD-10 disease coding database |
| `example_data.json` | Example input data format |
| `evaluate_f1.py` | F1 score evaluation script |
| `CMEMR-Cerebrovascular/` | Cerebrovascular case ID lists |

## 🚀 Quick Start

### Build Knowledge Graph

```bash
# 1. Download CPubMed-KG: https://cpubmed.openi.org.cn/graph/wiki
# 2. Run script
python create_nodes_relationships.py
```

### Import to Neo4j

```bash
neo4j-admin import \
   --database=medkg \
   --nodes="./kgfiles/nodes.csv" \
   --relationships="./kgfiles/relationships.csv" \
   --id-type=INTEGER \
   --delimiter="|"
```

### Run Evaluation

```bash
python evaluate_f1.py
```

## 📖 Data Sources

- **Knowledge Graph**: [CPubMed-KG](https://cpubmed.openi.org.cn/graph/wiki) (Harbin Institute of Technology)
- **Medical Records**: [medIKAL Dataset](https://github.com/CSU-NLP-Group/medIKAL/) - Original records available at `https://bingli.iiyi.com/show/{emrid}-1.html`
- **ICD-10**: Beijing Clinical Version v6.01

## 📎 Links

- [Paper](https://arxiv.org/pdf/2602.01297) · [Code](https://github.com/shenshaowei/RE-MCDF)
