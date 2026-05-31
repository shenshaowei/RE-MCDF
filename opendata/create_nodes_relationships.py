import os
import json
import csv
from tqdm import tqdm

# ================= 配置路径 =================
BASE_DIR = "/home/a/SSW/RE-MCDF/kgfiles" 
INPUT_FILE = "/home/a/SSW/RE-MCDF/data/CPubMed-KGv2_0.txt"
# 输出文件路径 - 保持与config一致
OUT_NODES = os.path.join(BASE_DIR, "nodes.csv")
OUT_RELS = os.path.join(BASE_DIR, "relationships.csv")
OUT_ENTITIES2ID = os.path.join(BASE_DIR, "KG_entities2id_merge.txt")
OUT_TYPE_MAP = os.path.join(BASE_DIR, "entity_type_map_merge.json")
OUT_TRIPLES = os.path.join(BASE_DIR, "KG_triples.txt")
# ============================================

def main():
    print("🚀 正在读取原始数据 CPubMed-KGv2_0.txt ...")
    
    entities = set()
    entity_type_map = {}
    triples = []
    
    # 1. 读取并解析原始文件
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing"):
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) != 3: continue
            
            head_raw, rel, tail_raw = parts
            
            # 解析 Head (格式: 名字@@类型)
            if '@@' in head_raw:
                h_name, h_type = head_raw.split('@@', 1)
                h_name, h_type = h_name.strip(), h_type.strip()
                entities.add(h_name)
                entity_type_map[h_name] = h_type
            else: continue
            
            # 解析 Tail
            if '@@' in tail_raw:
                t_name, t_type = tail_raw.split('@@', 1)
                t_name, t_type = t_name.strip(), t_type.strip()
                entities.add(t_name)
                entity_type_map[t_name] = t_type
            else: continue

            triples.append((h_name, rel.strip(), t_name))

    print(f"✅ 提取完成：共 {len(entities)} 个实体，{len(triples)} 条关系")

    # 2. 构建 ID 映射
    sorted_entities = sorted(list(entities))
    entity2id = {name: i+1 for i, name in enumerate(sorted_entities)}

    # ---------------- 生成 Python 依赖文件 ----------------
    print(f"📄 生成 KG_entities2id_merge.txt ...")
    with open(OUT_ENTITIES2ID, 'w', encoding='utf-8') as f:
        for i, name in enumerate(sorted_entities):
            f.write(f"{name}\t{i}\n")

    print(f"📄 生成 entity_type_map_merge.json ...")
    with open(OUT_TYPE_MAP, 'w', encoding='utf-8') as f:
        json.dump(entity_type_map, f, ensure_ascii=False, indent=2)

    print(f"📄 生成 KG_triples.txt ...")
    with open(OUT_TRIPLES, 'w', encoding='utf-8') as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

    # ---------------- 生成 Neo4j 依赖文件 ----------------
    print(f"📄 生成 nodes.csv (修复逻辑：使用动态Label) ...")
    with open(OUT_NODES, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        # Header: ID | 名字 | 类型属性 | Neo4j标签(关键!)
        writer.writerow([':ID', 'name', 'type', ':LABEL'])
        
        for name in sorted_entities:
            eid = entity2id[name]
            etype = entity_type_map.get(name, 'Unknown')
            # 🔴 修正：最后一列必须是 etype (如 "疾病")，不能是 "Entity"
            # 这样 Neo4j 才能支持 MATCH (n:疾病)
            writer.writerow([eid, name, etype, etype])

    print(f"📄 生成 relationships.csv (修复逻辑：使用动态Type) ...")
    with open(OUT_RELS, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        # Header: Start | End | 关系名属性 | Neo4j关系类型(关键!)
        writer.writerow([':START_ID', ':END_ID', 'relation', ':TYPE'])
        
        for h, r, t in tqdm(triples, desc="Writing CSV"):
            if h in entity2id and t in entity2id:
                # 🔴 修正：最后一列必须是 r (如 "影像学检查")，不能是 "RELATED_TO"
                writer.writerow([entity2id[h], entity2id[t], r, r])

    print("🎉 所有文件生成完毕！")
    print("\n" + "="*60)
    print("🔧 Neo4j 导入命令 (请确保先停止Neo4j):")
    print("rm -rf data/databases/medkg data/transactions/medkg")
    print("./neo4j-admin database import full medkg \\")
    print(f"  --nodes=\"{OUT_NODES}\" \\")
    print(f"  --relationships=\"{OUT_RELS}\" \\")
    print("  --delimiter=\"|\" \\")
    print("  --skip-bad-relationships=true \\")
    print("  --overwrite-destination=true")
    print("="*60)

if __name__ == "__main__":
    main()