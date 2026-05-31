from tqdm import tqdm
import re
import json
import time
import numpy as np
from collections import defaultdict
import math
from collections import defaultdict
from doctor import *
class DiseaseRelation:
    def __init__(self):
        self.similar_pairs = []  # [(disease1, disease2, similarity_score)]
        self.exclusive_pairs = []  # [(disease1, disease2, reason, description)]
        self.disease_relations = defaultdict(lambda: {'confusing_with': [], 'conflicts_with': []})  # disease -> relations dict
    
    def add_similarity(self, disease1, disease2, score):
        """添加相似疾病对"""
        if score > 0:
            self.similar_pairs.append((disease1, disease2, score))
            self.disease_relations[disease1]['confusing_with'].append((disease2, score))
            self.disease_relations[disease2]['confusing_with'].append((disease1, score))
    
    def add_exclusivity(self, disease1, disease2, reason, description):
        """添加互斥疾病对，无置信度"""
        self.exclusive_pairs.append((disease1, disease2, reason, description))
        self.disease_relations[disease1]['conflicts_with'].append({
            'disease': disease2,
            'reason': reason,
            'description': description
        })
        self.disease_relations[disease2]['conflicts_with'].append({
            'disease': disease1,
            'reason': reason,
            'description': description
        })
    
    def get_relations_for_disease(self, disease):
        """获取特定疾病的所有关系"""
        return self.disease_relations.get(disease, {'confusing_with': [], 'conflicts_with': []})
    
    def has_relations(self):
        """检查是否有任何关系"""
        return len(self.similar_pairs) > 0 or len(self.exclusive_pairs) > 0
    
class KGTools:
    """ 一些工具函数 """
    def __init__(self, ner_model, retriever, kg, rerank_topn, dis_topn, path_topn, entity_weight_map_file, args, doctor=None):
        self.ner_model = ner_model
        self.retriever = retriever
        self.kg = kg
        self.dis_topn = dis_topn
        self.rerank_topn = rerank_topn
        self.path_topn = path_topn
        self.args = args  # 传递消融实验参数
        self.doctor = doctor  # 传递 Doctor 实例
        self.last_kg_supplement_details = None 
        # 改为使用 args 中的消融实验参数
        self.use_relation_mining = getattr(args, 'use_relation_mining', True)
        self.use_kg_knowledge = getattr(args, 'use_kg_knowledge', True)
        self.use_lab_expert = getattr(args, 'use_lab_expert', True)
        self.use_dynamic_weighting = getattr(args, 'use_dynamic_weighting', True)
        
        # 初始化其他必要参数
        self.entity_weight_map_file = entity_weight_map_file
        with open(entity_weight_map_file, 'r', encoding="utf-8") as f:
            self.entity_weight_map = json.load(f)
        # ========== 新增：医学语义权重 ==========
        # 默认初始化
        self.entity_type_weights = {
            "sym": 0.6297, "dis": 0.1638, "dru": 0.1391, 
            "bod": 0.0212, "ite": 0.0372, "equ": 0.0029,
            "mic": 0.0009, "dep": 0.0004, "pro": 0.0043
        }
        # ==========================================
    
    def extract_evidence_entities(self, evidence_texts):
        """
        从证据文本中提取实体，保留所有提到的实体
        """
        evidence_entities = []
        
        for text in evidence_texts:
            if not text or text.strip() == "":
                continue
                
            # 按逗号、分号等分隔符提取潜在实体
            potential_entities = re.split(r'[，,；;、\s]+', text.strip())
            for entity in potential_entities:
                kg_entity = self.check_match(entity.strip())
                if kg_entity:  # 即使映射失败也会返回原始实体
                    evidence_entities.append({
                        'text': entity.strip(),
                        'kg_entity': kg_entity,
                        'type': 'unknown'  # 无法确定类型时设为unknown
                    })
        
        # 去重
        unique_entities = {}
        for entity in evidence_entities:
            # 使用kg_entity作为唯一标识
            unique_entities[entity['kg_entity']] = entity
        
        return list(unique_entities.values())
    

    
    def process_output(self, text):
        pred_response = [t.split("：")[1].split("\n")[0] for t in text.replace(":", "：").split("预测疾病")[1:] if "：" in t]
        pred_response = [t for t in pred_response if t != ""]
        if pred_response == []:
            extract_dis = self.ner_model.ner(text=text)
            for extract_res in extract_dis:
                for out in extract_res['output']:
                    if out['type'] == 'dis':
                        pred_response.append(out['span'])
            if pred_response == []:
                return []
        res = [self.check_match(dis) for dis in pred_response]
        return res
    
    def check_match(self, entity):
        assert entity is not None
        if entity in self.kg.kg_entities:
            return entity
        else:
            retrieve_res = self.retriever.retrieve(query=entity, top_k=5)
            if len(retrieve_res) != 0:
                return retrieve_res[0]['text']
            else:
                # return ""
                return entity.strip()
    
    def get_past_dis(self, ner_result):
        past_dis = []
        exam_dis = [] # 辅助检查中出现的疾病
        past_dict = ner_result[1]
        exam_dict = ner_result[2]
        for dis in past_dict['dis']:
            past_dis.append(dis['kg_entity'])
        for dis in exam_dict['dis']:
            exam_dis.append(dis['kg_entity'])
        return past_dis, exam_dis

    def set_dynamic_entity_weights(self, dynamic_weights: dict):
        """
        设置 LLM 动态生成的实体类型权重
        :param dynamic_weights: dict, e.g. {"sym": 0.8, "ite": 0.6, ...}
        """
        # 验证完整性
        required_keys = {"sym", "dis", "dru", "bod", "ite", "equ", "mic", "dep", "pro"}
        if set(dynamic_weights.keys()) != required_keys:
            print("[警告] 动态权重键不完整，使用默认权重")
            return False
        
        self.entity_type_weights = dynamic_weights
        print(f"✅ 动态实体权重已更新: {dynamic_weights}")
        return True
    
    def get_ner_result(self, chief_complaint, fst_rd_summary, scd_rd_summary):
        """ 获取实体识别结果，并保存在字典中 """
        total_ner_dict = {'bod':[], 'dep':[], 'dis':[], 'dru':[], 'equ':[], 'ite':[], 'mic':[], 'pro':[], 'sym':[]}
        ner_dict1 = {'bod':[], 'dep':[], 'dis':[], 'dru':[], 'equ':[], 'ite':[], 'mic':[], 'pro':[], 'sym':[]}
        ner_dict2 = {'bod':[], 'dep':[], 'dis':[], 'dru':[], 'equ':[], 'ite':[], 'mic':[], 'pro':[], 'sym':[]}
        ner_dict3 = {'bod':[], 'dep':[], 'dis':[], 'dru':[], 'equ':[], 'ite':[], 'mic':[], 'pro':[], 'sym':[]}
        EMR2kg_entity_map = {}
        extract_result1 = self.ner_model.ner(text=chief_complaint)
        extract_result2 = self.ner_model.ner(text=fst_rd_summary)
        extract_result3 = self.ner_model.ner(text=scd_rd_summary)
        
        def process_extract_result(extract_result, ner_dict, total_ner_dict):
            for extract_res in extract_result:
                for out in extract_res["output"]:
                    cur_entity_dict = {}
                    mapped_entity = self.check_match(out['span'])
                    if mapped_entity == "":
                        continue
                    EMR2kg_entity_map[out['span']] = mapped_entity
                    cur_entity_dict["kg_entity"] = mapped_entity
                    if mapped_entity in self.kg.entity_type_map:
                        cur_entity_dict["kg_entity_type"] = self.kg.entity_type_map[mapped_entity]
                    else:
                        continue
                    cur_entity_dict["EMR_entity"] = out['span']
                    if out['type'] in ner_dict:
                        ner_dict[out['type']].append(cur_entity_dict)
                        total_ner_dict[out['type']].append(cur_entity_dict)
        
        # 处理三个文本的NER结果
        process_extract_result(extract_result1, ner_dict1, total_ner_dict)
        process_extract_result(extract_result2, ner_dict2, total_ner_dict)
        process_extract_result(extract_result3, ner_dict3, total_ner_dict)
        
        # 对每个字典的键对应的value列表进行去重
        for key in total_ner_dict.keys():
            total_ner_dict[key] = [dict(t) for t in {tuple(d.items()) for d in total_ner_dict[key]}]
        for key in ner_dict1.keys():
            ner_dict1[key] = [dict(t) for t in {tuple(d.items()) for d in ner_dict1[key]}]
        for key in ner_dict2.keys():
            ner_dict2[key] = [dict(t) for t in {tuple(d.items()) for d in ner_dict2[key]}]
        for key in ner_dict3.keys():
            ner_dict3[key] = [dict(t) for t in {tuple(d.items()) for d in ner_dict3[key]}]
        
        self.EMR2kg_entity_map = EMR2kg_entity_map
        ner_result = [ner_dict1, ner_dict2, ner_dict3]
        return ner_result, total_ner_dict, EMR2kg_entity_map
    
    
    def get_path_str(self, node_id_list):
        path_str = ""
        node_list = [self.kg.gds.util.asNode(node_id)["name"] for node_id in node_id_list]
        # 现在已知node_list里的节点(已经转换为名称了)是依次连接成一条路径的，现在需要查出它们之间依次连接时的关系
        for i in range(len(node_list) - 1):
            path_str += node_list[i] + "->"
        path_str += node_list[-1]
        return path_str

    # 在 KGTools 类中添加以下方法
    # 新增方法：批量获取疾病邻居（按类型分类）
    def _batch_get_disease_neighbors_by_type(self, diseases):
        """单次查询获取所有疾病的邻居（按实体类型分类）"""
        # 定义要获取的实体类型（排除注释掉的类型）
        entity_types = ["疾病", "症状", "药物", "部位", "流行病学", "其他治疗", "手术治疗", "其他"]
        
        query = """
        MATCH (d:`疾病`)-[r]-(n)
        WHERE d.name IN $diseases
        AND any(label IN labels(n) WHERE label IN $entity_types)
        RETURN d.name AS source, 
            head([label IN labels(n) WHERE label IN $entity_types]) AS entity_type,
            collect(DISTINCT n.name) AS neighbors
        """
        
        result = self.kg.session.run(query, diseases=diseases, entity_types=entity_types)
        
        # 初始化缓存
        neighbor_cache = {}
        for disease in diseases:
            neighbor_cache[disease] = {etype: set() for etype in entity_types}
        
        # 填充缓存
        for record in result:
            disease = record["source"]
            entity_type = record["entity_type"]
            neighbors = set(record["neighbors"])
            
            if disease in neighbor_cache and entity_type in neighbor_cache[disease]:
                neighbor_cache[disease][entity_type] = neighbors
        
        # 确保所有疾病都有条目
        for d in diseases:
            if d not in neighbor_cache:
                neighbor_cache[d] = {etype: set() for etype in entity_types}
        
        print(f"  ✅ 获取 {len(neighbor_cache)} 个疾病的邻居数据（按类型分类）")
        return neighbor_cache
    
    def _calculate_weighted_jaccard(self, neighbors1, neighbors2, weights):
        type_to_key = {
            "症状": "sym", 
            "疾病": "dis", 
            "药物": "dru", 
            "部位": "bod",
            "其他治疗": "ite", 
            "手术治疗": "ite",  # 合并到ite
            "流行病学": "mic", 
            "其他": "pro"
        }

        total_weighted_intersection = 0.0
        total_weighted_union = 0.0

        all_types = set(neighbors1.keys()) | set(neighbors2.keys())
        
        for entity_type in all_types:
            key = type_to_key.get(entity_type, "pro")
            weight = weights.get(key)
            
            if weight is None:
                weight = 0.01
            
            set1 = neighbors1.get(entity_type, set())
            set2 = neighbors2.get(entity_type, set())
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            if union > 0:  # 实际上 union >= intersection >= 0，union=0 仅当两者都空
                total_weighted_intersection += weight * intersection
                total_weighted_union += weight * union
            else:
                # 两者都为空集合，对相似度无贡献，可跳过
                pass

        return total_weighted_intersection / total_weighted_union if total_weighted_union > 0 else 0.0

    def mine_disease_relations(self, candidate_diseases, sim_threshold=0.65, use_relation_mining=True, emr_context=""):
        """
        挖掘候选疾病间的关系 - 适配重构后的关系数据
        """
        if not use_relation_mining:
            print("  [关系挖掘] 跳过（消融实验）")
            return {
                'relations': DiseaseRelation(),
                'raw_exclusivity_data': {'exclusive_pairs': []},
                'raw_similarity_data': {'similar_pairs': []}
            }
        
        relations = DiseaseRelation()
        print(f"  [关系挖掘] 开始分析 {len(candidate_diseases)} 个候选疾病间的关系...")
        
        if not candidate_diseases or len(candidate_diseases) < 2:
            print("  [关系挖掘] 候选疾病少于2个，无需分析关系")
            return {
                'relations': relations,
                'raw_exclusivity_data': {'exclusive_pairs': []},
                'raw_similarity_data': {'similar_pairs': []}
            }
        
        # ===== 1. 从知识图谱批量获取所有疾病关系=====
        disease_relations_batch = {}
        print("  🔍 从知识图谱批量获取疾病关系...")
        try:
            disease_relations_batch = self.kg.get_disease_all_relations_batch(candidate_diseases)
            print(f"    ✅ 成功获取疾病关系数据")
        except Exception as e:
            print(f"    ⚠️ 获取疾病关系失败: {str(e)}")
        
        # ===== 2. 提取疾病子图上下文 =====
        disease_graph_context = ""
        # try:
        #     disease_graph_context = self._extract_disease_subgraph_context(candidate_diseases)
        #     print("    ✅ 成功提取疾病子图上下文")
        # except Exception as e:
        #     print(f"    ⚠️ 提取疾病子图上下文失败: {str(e)}")
        #     disease_graph_context = "无法获取疾病子图上下文"

        print("  🔍 预计算疾病对的BGE相似度...")
        bge_similarities = {}
        try:
            # 生成所有疾病对
            disease_pairs = []
            for i in range(len(candidate_diseases)):
                for j in range(i+1, len(candidate_diseases)):
                    d1 = candidate_diseases[i]
                    d2 = candidate_diseases[j]
                    disease_pairs.append((d1, d2))
            
            # 批量预计算BGE相似度
            if hasattr(self.kg, 'bge_model') and self.kg.bge_model is not None:
                print(f"    ✅ 预计算 {len(disease_pairs)} 个疾病对的BGE相似度...")
                unique_diseases = list(set([d for pair in disease_pairs for d in pair]))
                self.kg.batch_encode_diseases(unique_diseases)
                
                for d1, d2 in disease_pairs:
                    try:
                        sim = self.kg.get_bge_similarity(d1, d2)
                        bge_similarities[f"{d1}|{d2}"] = sim
                    except Exception as e:
                        bge_similarities[f"{d1}|{d2}"] = 0.0
                        print(f"    ⚠️ BGE计算失败: {d1} ↔ {d2}, 使用默认值 0.0")
            else:
                print("    ⚠️ BGE模型不可用，使用默认相似度 0.0")
                for d1, d2 in disease_pairs:
                    bge_similarities[f"{d1}|{d2}"] = 0.0
        except Exception as e:
            print(f"    ⚠️ BGE预计算出错: {str(e)}")
            bge_similarities = {f"{d1}|{d2}": 0.0 for d1, d2 in disease_pairs}

        # ===== 3. 调用互斥专家分析 =====
        print("\n[互斥专家] 分析疾病间互斥关系...")
        exclusivity_data = {'exclusive_pairs': []}
        try:
            exclusivity_data = self.doctor.analyze_disease_exclusivity(
                emr_context,
                candidate_diseases,
                disease_graph_context,
                disease_relations_batch
            )
            print(f"    ✅ 互斥专家返回 {len(exclusivity_data.get('exclusive_pairs', []))} 个互斥对")
        except Exception as e:
            print(f"    ⚠️ 互斥专家分析失败: {str(e)}")
        
        # ===== 4. 调用混淆度专家分析 =====
        print("\n[混淆度专家] 分析疾病间混淆关系...")
        similarity_data = {'similar_pairs': []}
        try:
            similarity_data = self.doctor.analyze_disease_similarity(
                emr_context,
                candidate_diseases,
                disease_graph_context,
                disease_relations_batch
            )
            print(f"    ✅ 混淆度专家返回 {len(similarity_data.get('similar_pairs', []))} 个混淆对")
        except Exception as e:
            print(f"    ⚠️ 混淆度专家分析失败: {str(e)}")
        
        # ===== 5. 构建疾病关系 =====
        print("  📊 构建疾病关系...")
        try:
            # 构建互斥关系
            for pair in exclusivity_data.get("exclusive_pairs", []):
                d1 = pair.get("disease1")
                d2 = pair.get("disease2")
                reason = pair.get("reason", "专家判定")
                kg_evidence = pair.get("kg_evidence", "无知识图谱证据")
                if d1 and d2:
                    description = f"理由: {reason}, 知识图谱证据: {kg_evidence}"
                    relations.add_exclusivity(d1, d2, reason, description)
                    print(f"      ⚠️ 互斥关系: {d1} ↔ {d2}")
            
            # ===== 考虑BGE相似度阈值 =====
            for pair in similarity_data.get("similar_pairs", []):
                d1 = pair.get("disease1")
                d2 = pair.get("disease2")
                reason = pair.get("reason", "专家判定临床表现高度重叠")
                if d1 and d2:
                    # 从BGE预计算结果中获取相似度
                    key = f"{d1}|{d2}"
                    similarity_score = bge_similarities.get(key, 0.0)  # 默认0.0
                    
                    if similarity_score >= 0.6:
                        relations.add_similarity(d1, d2, similarity_score)
                        print(f"      ⭐ 混淆关系: {d1} ↔ {d2} (相似度: {similarity_score:.2f})")
                        print(f"        原因: {reason}")
                    else:
                        # 调试信息：记录被过滤的低相似度对
                        print(f"      ❌ 过滤: {d1} ↔ {d2} (相似度: {similarity_score:.2f} < 阈值 {0.6})")
                        print(f"        原因: {reason}")
            
            print(f"    ✅ 最终构建 {len(relations.similar_pairs)} 个混淆关系，{len(relations.exclusive_pairs)} 个互斥关系")
        except Exception as e:
            print(f"    ⚠️ 构建关系失败: {str(e)}")
        
        print(f"\n[关系挖掘] 完成! 找到 {len(relations.similar_pairs)} 个混淆疾病对，{len(relations.exclusive_pairs)} 个互斥疾病对")
        
        return {
            'relations': relations,
            'raw_exclusivity_data': exclusivity_data,
            'raw_similarity_data': similarity_data
        }           


        
    def _extract_disease_subgraph_context(self, diseases):
        """
        从知识图谱提取疾病子图的文本描述，用于LLM互斥分析
        """
        subgraph_context = ["【疾病分类学关系】"]
        
        # 1. 获取疾病的分类学路径
        disease_ancestors = {}
        for disease in diseases:
            ancestors = self.kg.get_disease_ancestors(disease)
            if ancestors:
                disease_ancestors[disease] = ancestors
        
        if disease_ancestors:
            for disease, ancestors in disease_ancestors.items():
                # 只显示最近的3个祖先节点
                ancestor_str = " → ".join(ancestors[-3:]) if len(ancestors) > 3 else " → ".join(ancestors)
                subgraph_context.append(f"- {disease} 分类路径: {ancestor_str}")
        else:
            subgraph_context.append("无明确分类学层级信息")
        
        # 2. 获取疾病间的直接关系
        subgraph_context.append("\n【疾病间直接关系】")
        direct_relations_found = False
        
        for i, d1 in enumerate(diseases):
            for d2 in diseases[i+1:]:
                path_str, path_len = self.kg.find_shortest_path(d1, d2)
                if 0 < path_len <= 3:  # 有意义的关系
                    subgraph_context.append(f"- {d1} 与 {d2}: {path_str} (路径长度:{path_len})")
                    direct_relations_found = True
        
        if not direct_relations_found:
            subgraph_context.append("无直接相连的疾病关系")
        
        # 3. 获取疾病的典型特征
        subgraph_context.append("\n【疾病典型特征】")
        for disease in diseases:
            symptoms = self.kg.get_disease_symptoms(disease, top_k=3)
            if symptoms:
                symptom_str = "、".join(symptoms)
                subgraph_context.append(f"- {disease} 典型表现: {symptom_str}")
        
        return "\n".join(subgraph_context)
 

    def get_kg_supplement_diseases(self, total_ner_dict, high_evidence_items, existing_diseases, abnormal_entities, top_k=3, use_kg_knowledge=True, use_lab_expert=True):
        """
        KG补充疾病检索 
        """
        if not use_kg_knowledge:
            print("  [KG补充] 跳过（消融实验）")
            return []
        
        print(f"KG补充疾病检索 (使用{'异常实体' if use_lab_expert else '普通实体'})")
        
        # ========== 构建实体池 ==========
        entity_pool = {}
        predefined_weights = getattr(self, 'entity_type_weights', {
            "sym": 0.6297, "dis": 0.1638, "dru": 0.1391, "bod": 0.0212,
            "ite": 0.0372, "equ": 0.0029, "mic": 0.0009, "dep": 0.0004, "pro": 0.0043
        })
        
        # 构建 kg_entity -> type 的映射
        kg_ent_to_type = {}
        for ent_type, ents in total_ner_dict.items():
            for ent_info in ents:
                kg_ent_to_type[ent_info['kg_entity']] = ent_type
        
        # 异常实体（如果启用检验科专家）
        if use_lab_expert:
            for ent in abnormal_entities:
                kg_ent = self.check_match(ent)
                if kg_ent and kg_ent in self.kg.kg_entities:
                    ent_type = kg_ent_to_type.get(kg_ent, "sym")
                    base_weight = predefined_weights.get(ent_type, 0.5)
                    entity_pool[kg_ent] = base_weight * 1.5  # 异常实体有特殊权重
        
        # 普通NER实体
        for ent_type, ents in total_ner_dict.items():
            base_weight = predefined_weights.get(ent_type, 0.01)
            if base_weight <= 0:
                continue
            for ent_info in ents:
                kg_ent = ent_info['kg_entity']
                # 跳过已包含的实体
                if kg_ent in entity_pool or kg_ent in existing_diseases:
                    continue
                entity_pool[kg_ent] = base_weight
        
        # 没有可用实体时跳过
        if not entity_pool:
            print("  [KG补充] 无可用实体，跳过")
            return []
        
        print(f"  使用{len(entity_pool)}个实体进行筛选")
        
        disease_score = {}
        covered_entities = {}
        
        for entity, weight in entity_pool.items():
            neighbors = self.kg.get_neighbor_disease(entity)
            for disease in neighbors:
                if disease in existing_diseases:  # 跳过已有的疾病
                    continue
                if disease not in disease_score:
                    disease_score[disease] = 0.0
                    covered_entities[disease] = []
                disease_score[disease] += weight
                covered_entities[disease].append(entity)
        
        # 按得分排序，取前top_k个
        sorted_diseases = sorted(disease_score.items(), key=lambda x: x[1], reverse=True)
        
        # 直接构造结果，跳过重排
        top_supplement = []
        for disease, score in sorted_diseases[:top_k]:
            top_supplement.append({
                'disease': disease,
                'score': score,  # 使用原始权重和作为分数
                'covered_entities': covered_entities.get(disease, [])[:3]
            })
        
        print(f"  筛选完成 → 补充疾病 Top-{top_k}: {[d['disease'] for d in top_supplement]}")
        
        return top_supplement