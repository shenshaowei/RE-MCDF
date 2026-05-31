from neo4j import GraphDatabase, basic_auth
from graphdatascience import GraphDataScience
import json
import torch
import numpy as np
from collections import defaultdict
from rapidfuzz import process, fuzz
import xlrd,os
from tqdm import tqdm
# from FlagEmbedding import FlagModel
from sentence_transformers import SentenceTransformer
from config import *
entity_list = [
    "None",
    "其他",
    "其他治疗",
    "手术治疗",
    # "检查",
    "流行病学",
    "疾病",
    "症状",
    "社会学",
    "药物",
    "部位",
    # "预后",
]

class MyKnowledgeGraph:
    def __init__(self, uri, username, password, kg_database, subgraph_name, kg_entity_path, entity_type_map_path, use_semantic_similarity=False):
        """ neo4j driver和session """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()
        self.session = self.driver.session()
        """ gds以及子图加载 """
        self.gds = GraphDataScience.from_neo4j_driver(uri, auth=(username, password), database=kg_database)
        # sub_graph就是原来的G
        self.kg_database = kg_database
        self.sub_graph = self.gds.graph.get(subgraph_name)
        self.sub_graph_entity_list = entity_list
        """ 知识图谱实体集合加载 """
        kg_entities = []
        kg_entity_filepath = kg_entity_path
        with open(kg_entity_filepath, 'r', encoding='utf-8') as kg_entity_file:
            kg_entity_lines = kg_entity_file.readlines()
            for kg_entity_line in kg_entity_lines:
                kg_entities.append(kg_entity_line.strip().split('\t')[0])
        kg_entity_file.close()
        self.kg_entities = kg_entities
        """ 知识图谱实体类型映射加载 """
        fin_entity_type_map = open(entity_type_map_path, "r", encoding="utf-8")
        entity_type_map = json.load(fin_entity_type_map)
        self.entity_type_map = entity_type_map
        # 在 __init__ 中，替换原有 ICD 加载逻辑（如果你已预加载缓存，可整合）
        icd_path = config_dict["icd_database_path"]
        self.icd_name_to_code = {}
        self.icd_disease_names = []  # ← 新增：用于 fuzzy match 的候选池
        self.icd_similarity_cache = {}

        if os.path.exists(icd_path):
            xls = xlrd.open_workbook(icd_path)
            sheet = xls.sheet_by_index(0)
            disease_ids = sheet.col_values(colx=0, start_rowx=1)
            disease_names = sheet.col_values(colx=1, start_rowx=1)
            self.icd_name_to_code = {name.strip(): id_.strip() for id_, name in zip(disease_ids, disease_names) if name and id_}
            self.icd_disease_names = list(self.icd_name_to_code.keys())
            print(f"✅ 加载 {len(self.icd_disease_names)} 个 ICD 疾病（仅映射表）")
        else:
            print("⚠️ ICD 文件未找到")
        
        # 添加BGE模型和缓存
        self.disease_embeddings = {}
        self.bge_model_path = config_dict["bge_model_path"]
        self._init_bge_model()
    def _init_bge_model(self):
        """懒加载BGE模型"""
        try:
            self.bge_model = SentenceTransformer(self.bge_model_path)
            print("✅ BGE模型加载成功")
        except Exception as e:
            print(f"⚠️ BGE模型加载失败: {e}")
    
    # 在 MyKnowledgeGraph 类中添加
    def batch_encode_diseases(self, diseases):
        """一次性编码所有疾病并缓存结果"""
        # 过滤出尚未缓存的疾病
        diseases_to_encode = [d for d in diseases if d not in self.disease_embeddings]
        
        if not diseases_to_encode:
            return
        
        print(f"  🔍 批量编码 {len(diseases_to_encode)} 个疾病...")
        embeddings = self.bge_model.encode(diseases_to_encode, normalize_embeddings=True)
        
        # 存入缓存
        for disease, emb in zip(diseases_to_encode, embeddings):
            self.disease_embeddings[disease] = emb
        
        print(f"  ✅ 完成批量编码，当前缓存疾病数: {len(self.disease_embeddings)}")
        
    def close(self):
        self.session.close()
        self.driver.close()

    def get_disease_triples(self, disease_name, max_triples=5):
        """获取疾病的三元组信息"""
        if not disease_name or not isinstance(disease_name, str) or len(disease_name.strip()) == 0:
            return []
        disease_name = disease_name.strip()
        query = """
        MATCH (d:`疾病`)-[r]-(e)
        WHERE d.name = $disease_name
        RETURN type(r) AS relation_type, e.name AS entity_name, labels(e)[0] AS entity_type
        LIMIT $max_triples
        """
        try:
            result = self.session.run(query, disease_name=disease_name, max_triples=max_triples)
            triples = []
            for record in result:
                triples.append({
                    'head': disease_name,
                    'relation': record["relation_type"],
                    'tail': record["entity_name"],
                    'tail_type': record["entity_type"]
                })
            return triples
        except Exception as e:
            print(f"获取三元组时出错: {e}")
            return []
    # ==========================================
    def get_neighbor_disease(self, entity_name):
        # 按照关系类型查询实体的邻居实体
        query = """
        MATCH (e)-[r]-(n:`疾病`)
        WHERE e.name = $entity_name
        RETURN collect(n.name) AS neighbor_entities
        """
        result = self.session.run(query, entity_name=entity_name)
        neighbor_list = []
        try:
            for record in result:
                neighbors = record["neighbor_entities"]
                neighbor_list.extend(neighbors)
        except:
            neighbor_list = []
        return neighbor_list
    # ==========================================
    def find_shortest_path(self, start_entity_name, end_entity_name):
        if start_entity_name == end_entity_name:
            return "", 0
        query = """
        MATCH (start_entity), (end_entity)
        WHERE start_entity.name = $start_entity_name AND end_entity.name = $end_entity_name
        MATCH p = shortestPath((start_entity)-[*..10]-(end_entity))
        RETURN p
        """
        result = self.session.run(
            query,
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        # 用paths记录路径的字符串表示和对应的路径长度，方便后续排序并输出
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            path_len = len(path.relationships)
            entities = []
            relations = []
            if path is not None:
                for i in range(len(path.nodes)):
                    node = path.nodes[i]
                    entity_name = node["name"]
                    entities.append(entity_name)
                    if i < len(path.relationships):
                        relationship = path.relationships[i]
                        relation_type = relationship.type
                        relations.append(relation_type)
                path_str = ""
                for i in range(len(entities)):
                    entities[i] = entities[i]
                    path_str += entities[i]
                    if i < len(relations):
                        relations[i] = relations[i]
                        path_str += "->" + relations[i] + "->"
                paths.append((path_str, path_len))
        if len(paths) != 0:
            # 按照长度排序
            paths.sort(key=lambda x: x[1])
            # 取最短的那条路径输出
            return paths[0]
        else:
            return ("", 0)
        
    def get_disease_similarity(self, target_disease, candidate_diseases, top_k=3):
        """计算目标疾病与候选疾病在知识图谱中的相似度"""
        similarities = []
        for candidate in candidate_diseases:
            if candidate == target_disease:
                continue
            _, path_len = self.find_shortest_path(target_disease, candidate)
            if path_len > 0 and path_len <= 5:  # 限制最大路径长度
                similarity = 1.0 / path_len
                similarities.append({
                    'disease': candidate,
                    'similarity': similarity
                })
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    # /home/a/SSW/RE-MCDF/src/main/kg_func.py
    def calculate_connectivity_score(self, disease, evidence_entities, max_hops=3):
        """计算疾病与证据集合的平均最短路径倒数（即连通性得分）"""
        if not evidence_entities:
            return 0.0
        total_score = 0.0
        valid_cnt = 0
        for entity in evidence_entities:
            try:
                path_info = self.find_shortest_path(entity, disease)
                if path_info[1] > 0:  # path_len > 0
                    score = 1.0 / path_info[1]
                    total_score += score
                    valid_cnt += 1
            except:
                continue
        return total_score / max(valid_cnt, 1) if valid_cnt > 0 else 0.0
    
    def calculate_egcd_score(self, disease: str, evidence_entities: list, entity_weights: dict, 
                            alpha=0.7, beta=0.3, epsilon=0.1, max_hops=3):
        """
        优化版 Evidence-Graph Connectivity Degree (EGCD) 得分计算
        :param disease: 候选疾病名称
        :param evidence_entities: 证据实体列表
        :param entity_weights: 实体类型权重字典，格式: {"sym": 0.63, "dis": 0.16, ...}
        :param alpha, beta: 连通性项和覆盖率项的混合系数 (alpha + beta = 1)
        :param epsilon: 平滑因子，防止除零
        :param max_hops: 最大允许的路径长度，超过此长度视为无连接
        :return: (egcd_score, connectivity_term, coverage_term)
        """
        if not evidence_entities or not disease:
            return 0.0, 0.0, 0.0
        
        connectivity_sum = 0.0
        total_weight = 0.0  # 所有实体的总权重
        connected_entities = []  # 存储有连接的实体
        
        # 遍历所有证据实体
        for ent in evidence_entities:
            # 1. 获取实体类型和对应权重
            ent_type = self.entity_type_map.get(ent, "sym")  # 默认为症状类型
            w_e = entity_weights.get(ent_type, 0.01)  # 获取权重，设置默认值
            
            # 累加总权重
            total_weight += w_e
            
            # 2. 计算最短路径
            path_str, path_len = self.find_shortest_path(ent, disease)
            
            # 3. 应用最大路径长度限制
            if 0 < path_len <= max_hops:
                # 计算带权重的连通性贡献: weight * (1 / (path_length + epsilon))
                weighted_conn = w_e * (1.0 / (path_len + epsilon))
                connectivity_sum += weighted_conn
                connected_entities.append((ent, path_len))
        
        # 4. 计算连通性项 - 加权平均路径倒数
        if total_weight > 0:
            connectivity_term = connectivity_sum / total_weight
        else:
            connectivity_term = 0.0
        
        # 5. 计算覆盖率项 - 有连接的实体权重占比
        connected_weight = sum(
            entity_weights.get(self.entity_type_map.get(ent, "sym"), 0.01) 
            for ent, _ in connected_entities
        )
        coverage_term = connected_weight / max(total_weight, epsilon)
        
        # 6. 组合两项得到最终EGCD得分
        egcd_score = alpha * connectivity_term + beta * coverage_term
        
        return egcd_score, connectivity_term, coverage_term
    
    def get_icd_max_similarity_score(self, disease_name: str, threshold=50) -> float:
        """
        与 evaluate.py 完全一致：返回疾病与ICD数据库的最大 fuzzy match 分数 (0-100)
        """
        if not disease_name or not self.icd_disease_names:
            return 0.0
        
        # 检查缓存（注意：这里缓存的是 fuzzy match 分数，不是相似度对）
        cache_key = ("fuzzy_score", disease_name)
        if cache_key in self.icd_similarity_cache:
            return self.icd_similarity_cache[cache_key]
        
        # 精确匹配
        if disease_name in self.icd_name_to_code:
            score = 100.0
        else:
            # 模糊匹配（与 evaluate.py 完全一致）
            matches = process.extract(
                disease_name,
                self.icd_disease_names,
                limit=1,
                scorer=fuzz.WRatio
            )
            score = matches[0][1] if matches else 0.0
        
        # 缓存并返回
        self.icd_similarity_cache[cache_key] = score
        return score
# 在 MyKnowledgeGraph 类中添加以下方法

    import math

    def find_lca_in_hierarchy(self, node1, node2):
        """在疾病层次结构中找到最近公共祖先"""
        # 支持多种关系类型：is_a、subclass_of、属于
        query = """
        MATCH p1 = (n1:`疾病`)-[:is_a|subclass_of|属于*0..5]->(ancestor),
            p2 = (n2:`疾病`)-[:is_a|subclass_of|属于*0..5]->(ancestor)
        WHERE n1.name = $node1 AND n2.name = $node2
        RETURN ancestor.name AS lca, 
            length(p1) AS depth1, 
            length(p2) AS depth2,
            length(p1) + length(p2) AS total_path_length
        ORDER BY total_path_length ASC
        LIMIT 1
        """
        try:
            result = self.session.run(query, node1=node1, node2=node2)
            record = result.single()
            if record:
                lca_depth = min(record["depth1"], record["depth2"])
                return record["lca"], lca_depth
            return None, 0
        except Exception as e:
            print(f"查找LCA时出错: {e}")
            return None, 0

    def get_node_depth_in_hierarchy(self, node):
        """获取节点在层次结构中的深度"""
        query = """
        MATCH p = (n:`疾病`)-[:is_a|subclass_of|属于*0..10]->(root)
        WHERE n.name = $node 
        AND NOT (root)-[:is_a|subclass_of|属于]->()
        RETURN length(p) AS depth
        LIMIT 1
        """
        try:
            result = self.session.run(query, node=node)
            record = result.single()
            return record["depth"] if record else 1
        except Exception as e:
            print(f"获取节点深度时出错: {e}")
            return 1
        
    import math
    def calculate_disease_exclusivity(self, disease1, disease2, lambda_param=0.2):
            """
            基于结构分裂和语义竞争计算疾病互斥度
            """
            # 1. 获取BGE语义相似度
            # 假设 bge_sim 在 0~1 之间
            bge_sim = self.get_bge_similarity(disease1, disease2)
            
            # 2. 计算结构分裂度 (Structure Score)
            lca_node, lca_depth = self.find_lca_in_hierarchy(disease1, disease2)
            
            conf_score = 0.0
            reason_struct = "无结构关系"
            
            # 只有当存在LCA且深度有效时才计算结构分
            if lca_node and lca_depth > 0:
                depth1 = self.get_node_depth_in_hierarchy(disease1)
                depth2 = self.get_node_depth_in_hierarchy(disease2)
                
                if depth1 and depth2:
                    min_depth = min(depth1, depth2)
                    _, kg_dist = self.find_shortest_path(disease1, disease2)
                    
                    # 只有距离大于0（非自身）才计算
                    if kg_dist > 0:
                        import math
                        # 深度比：越接近1越好（同层级）
                        depth_ratio = lca_depth / min_depth
                        # 距离衰减：距离越近（如2跳兄弟），互斥可能性越大
                        decay = math.exp(-lambda_param * kg_dist)
                        conf_score = depth_ratio * decay
                        reason_struct = f"LCA:{lca_node}({lca_depth}), Dist:{kg_dist}"

            # BGE 加权
            semantic_booster = 0.5 + bge_sim  
            
            adjusted_conf_score = conf_score * semantic_booster
            
            reason_extra = ""
            if conf_score == 0 and bge_sim > 0.85:
                adjusted_conf_score = 0.15 
                reason_extra = "(结构缺失，仅基于高语义疑似)"

            # 5. 打印调试
            print(f"📊 {disease1}↔{disease2} | 结构:{conf_score:.3f}[{reason_struct}] | 语义:{bge_sim:.3f}(x{semantic_booster:.2f}) | 互斥度:{adjusted_conf_score:.3f}{reason_extra}")


            # 6. 阈值判断
            # 阈值建议设为 0.4 左右，因为如果有结构分(0.5) * BGE高(1.4) = 0.7 (高互斥)
            # 如果只有结构分(0.5) * BGE低(0.6) = 0.3 (低互斥，可能是共病)
            # is_exclusive = adjusted_conf_score > 0.4
            
            reason = f"结构({conf_score:.2f}) × 语义增强({semantic_booster:.2f}) =  {adjusted_conf_score:.2f}"
            if reason_extra: reason += reason_extra
            
            return adjusted_conf_score, reason

    
    def get_bge_similarity(self, disease1, disease2):
        """获取两个疾病的BGE相似度"""
        # 确保疾病已被编码
        if disease1 not in self.disease_embeddings:
            self.batch_encode_diseases([disease1])
        if disease2 not in self.disease_embeddings:
            self.batch_encode_diseases([disease2])
        
        # 从缓存获取嵌入并计算相似度
        emb1 = self.disease_embeddings[disease1]
        emb2 = self.disease_embeddings[disease2]
        
        # 点积计算余弦相似度 (已归一化)
        return float(np.dot(emb1, emb2))
    
    def calculate_disease_similarity(self, disease1, disease2, threshold=0.5):
        """计算两个疾病的相似度，基于邻居节点的Jaccard系数、路径距离和BGE语义相似度"""
        # 1. 计算邻居节点Jaccard相似度
        neighbors1 = self.get_neighbor_disease(disease1)
        neighbors2 = self.get_neighbor_disease(disease2)
        
        if not neighbors1 or not neighbors2:
            jaccard_sim = 0.0
        else:
            set1 = set(neighbors1)
            set2 = set(neighbors2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard_sim = intersection / union if union > 0 else 0.0
        
        # 2. 计算路径距离相似度
        path_sim = 0.0
        try:
            _, path_len = self.find_shortest_path(disease1, disease2)
            if 1 <= path_len <= 5:  # 限制最大路径长度
                path_sim = 1.0 / path_len
        except Exception as e:
            pass
        
        # 3. 获取BGE语义相似度
        bge_sim = self.get_bge_similarity(disease1, disease2)
        
        # 4. 加权融合 (BGE权重最高，因为它包含语义信息)
        # 权重分配: BGE(0.5) + Jaccard(0.3) + Path(0.2)
        final_sim = 0.5 * bge_sim + 0.3 * jaccard_sim + 0.2 * path_sim
        
        # 调试信息
        print(f"{disease1} ↔ {disease2} | Jaccard: {jaccard_sim:.3f}, Path: {path_sim:.3f}, BGE: {bge_sim:.3f} → 最终: {final_sim:.3f}")
        
        return final_sim

    def get_disease_connectivity(self, disease, evidence_entities, max_hops=5):
        """
        计算疾病与证据实体集合的连通性得分
        连通性 = 有效连接证据的权重平均(1/路径长度)
        计算疾病与证据实体集合的连通性得分，并返回连接路径详情
        返回: (连通性得分, 路径详情列表)
        """
        if not evidence_entities or not disease:
            return 0.0, []
        
        total_score = 0.0
        valid_count = 0
        path_details = []  # 存储路径详情
        epsilon = 1e-8
        
        for entity in evidence_entities:
            if not entity or not isinstance(entity, str):
                continue
            try:
                # 获取最短路径
                path_str, path_len = self.find_shortest_path(entity, disease)
                # 仅考虑有效路径 (1-5跳)
                if 1 <= path_len <= max_hops:
                    score = 1.0 / (path_len + epsilon)
                    total_score += score
                    valid_count += 1
                    # 保存路径详情
                    simplified_path = path_str.replace('->', ' → ')
                    path_details.append({
                        'entity': entity,
                        'path': simplified_path,
                        'length': path_len,
                        'score': score
                    })
            except Exception as e:
                continue
        
        connectivity_score = total_score / max(valid_count, 1) if valid_count > 0 else 0.0
        return connectivity_score, path_details
    
    def get_disease_ancestors(self, disease_name):
        """
        获取疾病在分类树中的祖先节点
        """
        try:
            query = """
            MATCH path=(d:Disease {name: $disease})-[:HAS_PARENT*]->(ancestor:Disease)
            RETURN [node in nodes(path) | node.name] as ancestors
            ORDER BY length(path) ASC
            LIMIT 1
            """
            result = self.driver.execute_query(
                query, 
                disease=disease_name,
                database_=self.kg_database
            )
            if result.records:
                # 返回从根到当前节点的路径
                path = result.records[0]["ancestors"]
                return path
            return []
        except Exception as e:
            print(f"获取疾病祖先失败: {e}")
            return []

    def get_disease_symptoms(self, disease_name, top_k=5):
        """
        获取疾病的典型症状
        """
        try:
            query = """
            MATCH (d:Disease {name: $disease})-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN s.name as symptom, COUNT(*) as count
            ORDER BY count DESC
            LIMIT $top_k
            """
            result = self.driver.execute_query(
                query,
                disease=disease_name,
                top_k=top_k,
                database_=self.kg_database
            )
            symptoms = [record["symptom"] for record in result.records]
            return symptoms
        except Exception as e:
            print(f"获取疾病症状失败: {e}")
            return []


    def get_disease_all_relations_batch(self, diseases):
        """
        一次性获取多个疾病的所有关系
        返回结构化的疾病关系数据，包含互斥相关和相似相关的关系
        区分直接关系和疾病属性，为互斥度和混淆度计算提供数据
        """
        if not diseases:
            return {
                'exclusivity_relations': {},  # 互斥相关关系
                'similarity_relations': {}    # 混淆相关关系 (重命名为混淆度)
            }
        
        print(f"🔍 批量获取 {len(diseases)} 个疾病的关系及属性...")
        
        # 构建疾病名称条件
        disease_names_str = "','".join([d.replace("'", "''") for d in diseases])
        
        # 同时获取疾病间直接关系和疾病属性
        query = f"""
        // 第一部分：疾病间直接关系
        MATCH (d1:`疾病`)-[r]-(d2:`疾病`)
        WHERE d1.name IN ['{disease_names_str}'] AND d2.name IN ['{disease_names_str}']
        AND d1.name <> d2.name
        RETURN 
            d1.name AS source,
            d2.name AS target,
            type(r) AS relation_type,
            COALESCE(r.evidence, r.name, '') AS evidence_text,
            'DIRECT' AS relation_category
        
        UNION
        
        // 第二部分：疾病属性（用于互斥度计算）
        MATCH (d:`疾病`)-[r]->(attr)
        WHERE d.name IN ['{disease_names_str}']
        AND type(r) IN ['病因', '发病机制', '病理分型']
        RETURN 
            d.name AS source,
            d.name AS target,  // 目标设为自己，表示这是属性
            type(r) + '|' + attr.name AS relation_type,  // 组合类型和值
            COALESCE(r.evidence, attr.name, '') AS evidence_text,
            'EXCLUSIVITY_ATTR' AS relation_category
        
        UNION
        
        // 第三部分：疾病属性（用于混淆度计算）
        MATCH (d:`疾病`)-[r]->(attr)
        WHERE d.name IN ['{disease_names_str}']
        AND type(r) IN ['临床表现', '相关（症状）', '发病机制', '影像学检查', '实验室检查']
        RETURN 
            d.name AS source,
            d.name AS target,  // 目标设为自己，表示这是属性
            type(r) + '|' + attr.name AS relation_type,  // 组合类型和值
            COALESCE(r.evidence, attr.name, '') AS evidence_text,
            'CONFUSION_ATTR' AS relation_category
        """
        
        try:
            result = self.session.run(query)
            exclusivity_relations = {}
            similarity_relations = {}  # 保持变量名不变，但含义是混淆度
            
            # 互斥相关的核心关系类型（疾病间直接关系）
            exclusivity_direct_types = ["病理分型"]
            # exclusivity_direct_types = ["分子亚型"]
            
            # 混淆相关的核心关系类型（疾病间直接关系）
            confusion_direct_types = ["鉴别诊断"]
            
            # 存储疾病属性
            disease_exclusivity_attrs = {d: [] for d in diseases}
            disease_confusion_attrs = {d: [] for d in diseases}
            
            for record in result:
                source = record["source"]
                target = record["target"]
                rel_type = record["relation_type"]
                evidence = record["evidence_text"]
                category = record["relation_category"]
                
                # 初始化关系字典
                for d in [source, target]:
                    if d not in exclusivity_relations:
                        exclusivity_relations[d] = {}
                    if d not in similarity_relations:  # 保持变量名不变
                        similarity_relations[d] = {}
                
                # 处理直接关系
                if category == 'DIRECT':
                    relation_info = {
                        'relation_type': rel_type.split('|')[0] if '|' in rel_type else rel_type,
                        'evidence': f"{source} --[{rel_type}]--> {target}: {evidence}"
                    }
                    
                    # 互斥直接关系
                    if rel_type in exclusivity_direct_types:
                        if target not in exclusivity_relations[source]:
                            exclusivity_relations[source][target] = []
                        exclusivity_relations[source][target].append(relation_info)
                        
                        if source not in exclusivity_relations[target]:
                            exclusivity_relations[target][source] = []
                        exclusivity_relations[target][source].append(relation_info)
                    
                    # 混淆直接关系（鉴别诊断）
                    elif rel_type in confusion_direct_types:
                        if target not in similarity_relations[source]:
                            similarity_relations[source][target] = []
                        similarity_relations[source][target].append(relation_info)
                        
                        if source not in similarity_relations[target]:
                            similarity_relations[target][source] = []
                        similarity_relations[target][source].append(relation_info)
                
                # 处理互斥属性
                elif category == 'EXCLUSIVITY_ATTR':
                    attr_type, attr_value = rel_type.split('|', 1) if '|' in rel_type else (rel_type, evidence)
                    disease_exclusivity_attrs[source].append({
                        'type': attr_type,
                        'value': attr_value.strip(),
                        'evidence': evidence
                    })
                
                # 处理混淆属性
                elif category == 'CONFUSION_ATTR':
                    attr_type, attr_value = rel_type.split('|', 1) if '|' in rel_type else (rel_type, evidence)
                    disease_confusion_attrs[source].append({
                        'type': attr_type,
                        'value': attr_value.strip(),
                        'evidence': evidence
                    })
            
            # 构建属性比较的虚拟关系（用于后续分析）
            for i, d1 in enumerate(diseases):
                for d2 in diseases[i+1:]:
                    # 互斥度：比较疾病属性
                    if d1 in disease_exclusivity_attrs and d2 in disease_exclusivity_attrs:
                        d1_attrs = disease_exclusivity_attrs[d1]
                        d2_attrs = disease_exclusivity_attrs[d2]
                        
                        # 创建虚拟关系存储属性比较
                        relation_info = {
                            'relation_type': 'ATTRIBUTE_COMPARISON',
                            'evidence': f"属性比较: {d1} vs {d2}",
                            'disease1_attrs': d1_attrs,
                            'disease2_attrs': d2_attrs,
                            'comparison_type': 'exclusivity'
                        }
                        
                        if d2 not in exclusivity_relations[d1]:
                            exclusivity_relations[d1][d2] = []
                        exclusivity_relations[d1][d2].append(relation_info)
                        
                        if d1 not in exclusivity_relations[d2]:
                            exclusivity_relations[d2][d1] = []
                        exclusivity_relations[d2][d1].append(relation_info)
                    
                    # 混淆度：比较疾病属性
                    if d1 in disease_confusion_attrs and d2 in disease_confusion_attrs:
                        d1_attrs = disease_confusion_attrs[d1]
                        d2_attrs = disease_confusion_attrs[d2]
                        
                        # 计算属性重叠
                        overlap_attrs = []
                        unique_to_d1 = []
                        unique_to_d2 = []
                        
                        # 计算属性重叠
                        d1_values = {attr['value'] for attr in d1_attrs}
                        d2_values = {attr['value'] for attr in d2_attrs}
                        overlap_values = d1_values & d2_values
                        
                        for attr in d1_attrs:
                            if attr['value'] in overlap_values:
                                overlap_attrs.append(attr)
                            else:
                                unique_to_d1.append(attr)
                        
                        for attr in d2_attrs:
                            if attr['value'] not in overlap_values:
                                unique_to_d2.append(attr)
                        
                        # 创建虚拟关系存储混淆度分析
                        relation_info = {
                            'relation_type': 'CONFUSION_ANALYSIS',
                            'evidence': f"混淆度分析: {d1} vs {d2}",
                            'overlap_attrs': overlap_attrs,
                            'unique_to_disease1': unique_to_d1,
                            'unique_to_disease2': unique_to_d2,
                            'overlap_count': len(overlap_values),
                            'comparison_type': 'confusion'
                        }
                        
                        if d2 not in similarity_relations[d1]:
                            similarity_relations[d1][d2] = []
                        similarity_relations[d1][d2].append(relation_info)
                        
                        if d1 not in similarity_relations[d2]:
                            similarity_relations[d2][d1] = []
                        similarity_relations[d2][d1].append(relation_info)
            
            print(f"✅ 获取关系完成: 互斥关系疾病数={len(exclusivity_relations)}, 混淆关系疾病数={len(similarity_relations)}")
            print(f"   • 互斥属性覆盖: {sum(1 for d in diseases if disease_exclusivity_attrs.get(d))}/{len(diseases)} 个疾病")
            print(f"   • 混淆属性覆盖: {sum(1 for d in diseases if disease_confusion_attrs.get(d))}/{len(diseases)} 个疾病")
            
            return {
                'exclusivity_relations': exclusivity_relations,
                'similarity_relations': similarity_relations  # 保持变量名不变，但实际是混淆度
            }
            
        except Exception as e:
            print(f"⚠️ 批量获取疾病关系失败: {e}")
            return {
                'exclusivity_relations': {},
                'similarity_relations': {}
            }