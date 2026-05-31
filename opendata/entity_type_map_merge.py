entities = set()
entity_type_map = {}
#申请链接：https://cpubmed.openi.org.cn/graph/wiki 哈工大开源知识图谱CPubMed-KG
with open('/data108/user_hzx/SSW/work2/dataset/CPubMed-KGv2_0.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:  # 跳过空行
            continue
        parts = line.split('\t')
        if len(parts) != 3:  # 确保是三元组格式
            print(f"跳过格式错误的行: {line}")
            continue
        
        head, rel, tail = parts
        
        # 分割实体和类型
        if '@@' not in head or '@@' not in tail:
            print(f"跳过缺少 @@ 标记的行: {line}")
            continue
        
        head_entity, head_type = head.split('@@', 1)   # split 最多分成2部分，防止类型中有 @@
        tail_entity, tail_type = tail.split('@@', 1)
        
        entities.add(head_entity)
        entities.add(tail_entity)
        
        # 保存实体到类型的映射（如果实体重复出现，后面的会覆盖前面的）
        entity_type_map[head_entity] = head_type
        entity_type_map[tail_entity] = tail_type

# 写入实体到 ID 的映射文件
with open('/home/a/SSW/RE-MCDF/data/KG_entities2id_merge.txt', 'w', encoding='utf-8') as f:
    for idx, entity in enumerate(sorted(entities)):  # 推荐排序，保证顺序一致
        f.write(f"{entity}\t{idx}\n")

# 保存实体类型映射为 JSON
import json
with open('/home/a/SSW/RE-MCDF/data/entity_type_map_merge.json', 'w', encoding='utf-8') as f:
    json.dump(entity_type_map, f, ensure_ascii=False, indent=2)