import json
from config import *
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = config_dict["cuda_device"]
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ====== >>> 在这里插入日志重定向代码 <<< ======
import sys
import os
from datetime import datetime
LOG_DIR = config_dict["log_dir"]
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "ours_1_8.log")

class DualWriter:
    def __init__(self, stdout, logfile):
        self.stdout = stdout
        self.logfile = logfile
    def write(self, message):
        self.stdout.write(message)
        self.logfile.write(message)
        self.logfile.flush()
    def flush(self):
        self.stdout.flush()
        self.logfile.flush()

log_file = open(LOG_PATH, "a", encoding="utf-8")
log_file.write("\n" + "="*80 + "\n")
log_file.write(f"【开始运行】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write(f"config_main: {config_dict}\n")  # 新增这一行
log_file.write("="*80 + "\n")

sys.stdout = DualWriter(sys.__stdout__, log_file)

import atexit
def cleanup():
    sys.stdout = sys.__stdout__
    log_file.close()
atexit.register(cleanup)


# ====== <<< 日志代码结束 >>> ======

from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import sys
import re
import math

"""导入自建模块"""
from utils import *
from ner import *
from kg_func import *
from models import *
from retriever import *
from doctor import *
from config import *
from utils import DiseaseRelation
# ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
# ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->

def ae_preprocess(auxiliary_examination):
    """对辅助检查进行预处理"""
    auxiliary_examination = auxiliary_examination.replace("  - ", " ")
    auxiliary_examination = auxiliary_examination.replace("- ", " ")
    return auxiliary_examination

def main(doctor, KG_Tools, chief_complaint, current_medical_history, past_disease_history, body_check, auxiliary_exam, args, case_index ):
    print(f"\n------------------ 第{case_index}个病例处理开始 ------------------")
    
    # 步骤1-3保持不变
    print("步骤1: 基本信息总结...")
    fst_rd_summary, _ = doctor.general_info_summary(chief_complaint, current_medical_history, past_disease_history)
    print(f"内容: {fst_rd_summary}")

    print("步骤2: 检查结果总结...")
    scd_rd_summary, _ = doctor.examination_summary(body_check, auxiliary_exam)
    print(f"内容: {scd_rd_summary}")

    print("步骤3: 命名实体识别...")
    ner_result, total_ner_dict, EMR2kg_entity_map = KG_Tools.get_ner_result(chief_complaint, fst_rd_summary, scd_rd_summary)
    
    # 获取药物实体
    ner_dict_drug = ner_result[1]
    drug_entity = [ent['EMR_entity'] for ent in ner_dict_drug.get('dru', [])]
    print(f"识别到的药物实体: {drug_entity}")
    
    # 准备EMR上下文摘要
    emr_context_summary = f"""
    【主诉】: {chief_complaint}
    【现病史摘要】: {fst_rd_summary}
    【检查结果摘要】: {scd_rd_summary}
    """
    
    # ====== 【步骤4】LLM生成诊断假设 ======
    print("\n【步骤4】LLM生成诊断假设...")
    
    # 添加错误处理和重试机制
    llm_diagnoses = {"diagnoses": []}
    max_retries = 2
    for retry in range(max_retries):
        try:
            llm_diagnoses_raw, _ = doctor.generate_diagnosis_with_evidence(fst_rd, scd_rd, args.direct_topn)
            # 增强非空检查
            if isinstance(llm_diagnoses_raw, dict) and 'diagnoses' in llm_diagnoses_raw and len(llm_diagnoses_raw['diagnoses']) > 0:
                llm_diagnoses = llm_diagnoses_raw
                # 提取疾病列表
                llm_diseases = [d['disease'] for d in llm_diagnoses_raw['diagnoses']]
                print(f"  LLM生成的诊断 ({len(llm_diseases)}): {llm_diseases}")  # 在成功时立即打印
                break
            elif retry < max_retries - 1:
                print(f"  警告: 诊断格式错误或为空，正在重试 ({retry + 1}/{max_retries})")
                # 如果有temperature属性则增加0.1
                if hasattr(doctor, 'chat_model') and hasattr(doctor.chat_model, 'temperature'):
                    doctor.chat_model.temperature = min(doctor.chat_model.temperature + 0.1, 1.0)
                time.sleep(1)  # 添加短暂延迟
            else:
                print(f"  警告: 诊断生成失败，使用默认诊断列表")
                llm_diagnoses = {"diagnoses": []}
                llm_diseases = []  # 定义空的疾病列表
                print(f"  LLM生成的诊断 ({len(llm_diseases)}): {llm_diseases}")  # 打印空列表信息
        except Exception as e:
            print(f"  错误: 第{retry + 1}次尝试失败: {str(e)}")
            if retry < max_retries - 1:
                # 增加温度再试
                if hasattr(doctor, 'chat_model') and hasattr(doctor.chat_model, 'temperature'):
                    doctor.chat_model.temperature = min(doctor.chat_model.temperature + 0.1, 1.0)
                time.sleep(1)  # 添加短暂延迟
            else:
                print(f"  错误: 所有重试均失败，使用默认诊断列表")
                llm_diagnoses = {"diagnoses": []}
                llm_diseases = []  # 定义空的疾病列表
                print(f"  LLM生成的诊断 ({len(llm_diseases)}): {llm_diseases}")  # 打印空列表信息

    
    # ====== 【步骤5】生成动态实体权重和异常实体 ======
    print("\n【步骤5】生成动态实体权重和异常实体提取...")
    abnormal_entities = []
    if args.use_lab_expert:
        print("使用检验科专家生成动态权重和异常实体...")
        dynamic_weights, abnormal_entities = doctor.generate_dynamic_entity_weights(
            fst_rd_summary, scd_rd_summary, total_ner_dict
        )
        if args.use_dynamic_weighting:
            KG_Tools.set_dynamic_entity_weights(dynamic_weights)
            print(f"  识别到 {len(abnormal_entities)} 个异常实体: {abnormal_entities[:5]}")
        else:
            default_weights = {
            "sym": 0.6297, "dis": 0.1638, "dru": 0.1391,
            "bod": 0.0212, "ite": 0.0372, "equ": 0.0029,
            "mic": 0.0009, "dep": 0.0004, "pro": 0.0043
        }
            KG_Tools.set_dynamic_entity_weights(default_weights)
    else:
        print("跳过动态权重和异常实体生成（检验科专家/动态权重已禁用）")
        default_weights = {
            "sym": 0.6297, "dis": 0.1638, "dru": 0.1391,
            "bod": 0.0212, "ite": 0.0372, "equ": 0.0029,
            "mic": 0.0009, "dep": 0.0004, "pro": 0.0043
        }
        KG_Tools.set_dynamic_entity_weights(default_weights)
        abnormal_entities = []
    
    # ====== 【步骤6】KG补充候选疾病 ======
    print("\n【步骤6】KG补充候选疾病检索...")
    kg_supplement = []
    existing_diseases = set(llm_diseases)
    
    if args.use_kg_knowledge:
        kg_supplement = KG_Tools.get_kg_supplement_diseases(
            total_ner_dict=total_ner_dict,
            high_evidence_items=[],
            existing_diseases=existing_diseases,
            abnormal_entities=abnormal_entities,
            top_k=args.kg_supplement_topk,
            use_kg_knowledge=args.use_kg_knowledge,
            use_lab_expert=args.use_lab_expert
        )
        print(f"  检索到 {len(kg_supplement)} 个补充疾病: {[d['disease'] for d in kg_supplement]}")
    else:
        print("  跳过KG补充检索（消融实验）")
    
    # 合并所有候选疾病
    all_candidate_diseases = []
    for diag in llm_diagnoses['diagnoses']:
        evidence_entities = [KG_Tools.check_match(ent) for ent in diag.get('key_evidence', [])]
        all_candidate_diseases.append({
            'disease': diag['disease'],
            'source': 'llm',
            'evidence': diag.get('key_evidence', []),
            'evidence_entities': evidence_entities
        })
    
    for item in kg_supplement:
        if item['disease'] not in [d['disease'] for d in all_candidate_diseases]:
            all_candidate_diseases.append({
                'disease': item['disease'],
                'source': 'kg',
                'evidence': item.get('covered_entities', []),
                'evidence_entities': item.get('covered_entities', []),
                'kg_score': item['score']
            })
    
    print(f"\n候选疾病总数: {len(all_candidate_diseases)}")
    
    # ====== 【步骤7】疾病关系挖掘 ======
    print("\n【步骤7】疾病关系挖掘...")
    try:
        disease_relations_result = KG_Tools.mine_disease_relations(
            [item['disease'] for item in all_candidate_diseases],
            sim_threshold=0.6,
            use_relation_mining=args.use_relation_mining,
            emr_context=emr_context_summary
        )
        
        # 获取实际的关系对象
        disease_relations = disease_relations_result.get('relations', DiseaseRelation())
        
        # 构建 agent_b_before 字段
        agent_b_before = {
            'exclusive_pairs': [
                {
                    'disease1': pair.get('disease1', ''),
                    'disease2': pair.get('disease2', ''),
                    'reason': pair.get('reason', ''),
                    'kg_evidence': pair.get('kg_evidence', '')
                } for pair in disease_relations_result.get('raw_exclusivity_data', {}).get('exclusive_pairs', [])
            ],
            'similar_pairs': [
                {
                    'disease1': pair.get('disease1', ''),
                    'disease2': pair.get('disease2', ''),
                    'reason': pair.get('reason', '')
                } for pair in disease_relations_result.get('raw_similarity_data', {}).get('similar_pairs', [])
            ]
        }
    except Exception as e:
        print(f"❌ 疾病关系挖掘过程中出错: {str(e)}")
        print("⚠️ 将使用空关系继续执行")
        disease_relations = DiseaseRelation()
        agent_b_before = {
            'exclusive_pairs': [],
            'similar_pairs': []
        }
    # ====== 【步骤8】多专家协同评估 ======
    print("\n【步骤8】多专家协同评估...")
    
    # 准备疾病评估信息
    diseases_to_evaluate = []
    for item in all_candidate_diseases:
        disease = item['disease']
        evidence_entities = item['evidence_entities']
        # 替换这里：获取连通性得分和路径详情
        kg_connectivity, connection_paths = kg_system.get_disease_connectivity(disease, evidence_entities)
        icd_similarity = kg_system.get_icd_max_similarity_score(disease)
        
        # 也获取疾病的三元组信息
        disease_triples = kg_system.get_disease_triples(disease, max_triples=3)
        
        diseases_to_evaluate.append({
            'disease': disease,
            'evidence_entities': evidence_entities,
            'kg_connectivity': kg_connectivity,
            'icd_similarity': icd_similarity,
            'connection_paths': connection_paths,  # 限制路径
            'disease_triples': disease_triples
        })
    
    # 1. 智能体A：单疾病证据评估
    print("  [智能体A] 单疾病证据评估...")
    evidence_scores = doctor.agent_a_single_disease_evaluator(
        diseases_to_evaluate, emr_context_summary
    )
    
    # 2. 智能体B：多疾病关系评估
    print("\n  [智能体B] 多疾病关系评估...")
    candidate_disease_names = [item['disease'] for item in diseases_to_evaluate]
    # adjustment_factors = doctor.agent_b_multi_disease_evaluator(
    #     candidate_disease_names, disease_relations, evidence_scores, emr_context_summary
    # )
    adjustment_factors = doctor.agent_b_multi_disease_evaluator(
        candidate_disease_names, disease_relations, evidence_scores, emr_context_summary,high_similarity_pairs=disease_relations_result.get('high_similarity_pairs', [])  # 传入高相似度对
    )
    
    # 3. 应用调整并计算逻辑分
    logic_scores = {}
    for item in diseases_to_evaluate:
        disease = item['disease']
        base_score = evidence_scores.get(disease, 0.5)
        adj_factor = adjustment_factors.get(disease, 1.0)
        logic_score = base_score * adj_factor
        logic_score = max(0.0, min(1.0, logic_score))
        logic_scores[disease] = logic_score
    
    # 4. 智能体C：最终融合
    print("\n  [智能体C] 最终融合评分...")
    disease_scores_for_c = {}
    for item in diseases_to_evaluate:
        disease = item['disease']
        disease_scores_for_c[disease] = {
            'evidence_score': evidence_scores.get(disease, 0.5),
            'logic_score': logic_scores.get(disease, 0.5),
            'connectivity': item['kg_connectivity'],
            'icd_similarity': item['icd_similarity']
        }
    
    final_scores = doctor.agent_c_final_integrator(disease_scores_for_c)
    
    # ====== 【步骤9】结果排序与输出 ======
    print("\n【步骤9】结果排序与输出...")
    
    # 创建一个映射，便于根据疾病名称快速查找预先计算的信息
    disease_info_map = {item['disease']: item for item in diseases_to_evaluate}
    # 按最终得分排序
    sorted_diseases = sorted(candidate_disease_names, key=lambda d: final_scores[d], reverse=True)
    
    # 准备输出结果
    final_items = []
    for disease in sorted_diseases:
        score = final_scores[disease]
        # if score < 0.1:  # 过滤低分疾病
        #     continue
        
        # 收集详细信息
        disease_info = disease_info_map.get(disease)
        item = next((i for i in all_candidate_diseases if i['disease'] == disease), None)
        if not item:
            continue
        
        evidence_entities = item['evidence_entities']
        # kg_connectivity = kg_system.get_disease_connectivity(disease, evidence_entities)
        # icd_similarity = kg_system.get_icd_max_similarity_score(disease)
        kg_connectivity = disease_info['kg_connectivity']
        icd_similarity = disease_info['icd_similarity']
        result_str = f"【诊断】{disease}\n"
        result_str += f"【来源】{item['source']}\n"
        result_str += f"【证据实体】{', '.join(evidence_entities[:5])}\n"
        result_str += f"【证据分】{evidence_scores.get(disease, 0.5):.3f}\n"
        result_str += f"【逻辑分】{logic_scores.get(disease, 0.5):.3f}\n"
        result_str += f"【连通性】{kg_connectivity:.3f}\n"
        result_str += f"【ICD相似度】{icd_similarity:.1f}%\n"
        result_str += f"【最终得分】{score:.3f}\n"
        
        # 添加关系影响
        relations = disease_relations.get_relations_for_disease(disease)
        if relations['conflicts_with'] or relations['confusing_with']:
            result_str += "【关系影响】\n"
            for conflict in relations['conflicts_with']:
                result_str += f"  ⚠️ 与 {conflict['disease']} 互斥: {conflict['description']}\n"
            for similar_disease, sim_score in relations['confusing_with']:
                result_str += f"  ⭐ 与 {similar_disease} 高度相似 (相似度: {sim_score:.3f})\n"
        
        final_items.append((disease, result_str))
    
    # 附加原始信息
    direct_diagnos_dis = [d['disease'] for d in llm_diagnoses['diagnoses']]
    past_dis, exam_dis = KG_Tools.get_past_dis(ner_result)
    final_items.append([direct_diagnos_dis, past_dis, exam_dis, drug_entity])
    
    diagnoses_with_scores = []
    for i in range(len(final_items) - 1):
        disease, _ = final_items[i]
        score = final_scores.get(disease, 0.0)
        diagnoses_with_scores.append(f"{i+1}.{disease}({score:.3f})")
    print(f"\n完成: 最终诊断数量: {len(final_items)-1}")
    # 或者合并显示（只显示疾病名称）
    # 收集所有需要的信息
    all_original_diagnoses = direct_diagnos_dis + [d['disease'] for d in kg_supplement]
    reranked_diagnoses = []
    for i, disease in enumerate(sorted_diseases, 1):
        score = final_scores.get(disease, 0.0)
        reranked_diagnoses.append({
            "rank": i,
            "disease": disease,
            "score": round(score, 3),
            "source": next((item['source'] for item in all_candidate_diseases if item['disease'] == disease), "unknown")
        })
    
    print(f"\n完成: 最终诊断数量: {len(final_items)-1}")
    print(f"  所有原始诊断（LLM+KG）: {all_original_diagnoses}")
    print(f"  最终诊断: {' '.join(diagnoses_with_scores)}")
    
    old_reranked = reranked_diagnoses  # 原始列表字典格式
    new_reranked = {
        "ranks": [item["rank"] for item in old_reranked],
        "scores": [round(item["score"], 3) for item in old_reranked],
        "sources": [item["source"] for item in old_reranked],
        "diseases": [item["disease"] for item in old_reranked]
    }
    
    visualization_data = {
        "relations": {
            "similar_pairs": [(d1, d2, score) for d1, d2, score in disease_relations.similar_pairs],
            "exclusive_pairs": [(d1, d2, reason, desc) for d1, d2, reason, desc in disease_relations.exclusive_pairs]
        },
        "expert_scores": {
            "agent_a": evidence_scores,
            "agent_b_factors": adjustment_factors,
            "final_scores": final_scores
        },
        # 新增字段：智能体B之前的专家输出
        "agent_b_before": agent_b_before
    }
        # ====== 添加：构建独立的 diagnosis_info ======
    diagnosis_info = {
        "llm_diagnoses": direct_diagnos_dis,
        "kg_supplement": [d['disease'] for d in kg_supplement],
        "all_original_diagnoses": all_original_diagnoses,
        "reranked_diagnoses": new_reranked  # 使用上面已经构建的 new_reranked
    }
    print(f"\n【真实标签】{label}")
    print("✅ RE-CLD 诊断流程完成")
    print("----------------------------------------------------")
    
    return final_items, visualization_data, diagnosis_info

if __name__ == "__main__":
    # ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
    """初始化各个模块"""
    # 知识图谱初始化 uri, username, password, kg_database, subgraph_name, kg_entity_path, entity_type_map_path
    kg_system = MyKnowledgeGraph(
        uri=args.uri,
        username=args.username,
        password=args.password,
        kg_database=args.kg_database_name,
        subgraph_name=args.subgraph_name,
        kg_entity_path=args.kg_entity_path,
        entity_type_map_path=args.entity_type_map_path,
        # 注意：这里不再传递 use_semantic_similarity 参数，因为已删除相关代码
    )
    
    # NER模型初始化
    ner_model = NER_Model(args.ner_model_id, device='gpu')
    
    # 检索器初始化
    retriever = Retriever(args.retriever_type, args.retriever_version)
    
    # LLM初始化
    chat_model = ChatModel(args.model_type, args.model_name_or_path, args.model_version)
    
    # prompt初始化
    doctor = Doctor(chat_model)
    
    # 其它工具模块初始化
    KG_Tools = KGTools(
        ner_model=ner_model,
        retriever=retriever,
        kg=kg_system,
        rerank_topn=args.rerank_topn,
        dis_topn=args.dis_topn,
        path_topn=args.path_topn,
        entity_weight_map_file=args.entity_weight_map_file,
        args=args,  # 传递消融实验参数
        doctor=doctor
    )
    
    # ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
    # 用一个总的json文件记录各个数据集的输出结果
    if not os.path.exists(args.result_log_pred_dir):
        os.makedirs(args.result_log_pred_dir)
    total_record_file = open(args.result_log_pred_dir + "total_pred_record.json", "a", encoding="utf-8")
    
    # ========== 打印配置信息 ==========
    print("\n" + "="*60)
    print(f"【系统配置】")
    print(f"- 模型版本: {args.model_version}")
    print(f"- 检索器: {args.retriever_version}")
    print(f"- KG补充疾病数量: {args.kg_supplement_topk}")
    print(f"- 输出目录: {args.result_log_pred_dir}")
    print("="*60 + "\n")
    # ==========================================
    
    # finished_list = ["肿瘤科", "口腔科"]
    # 对指定目录下的每个json文件进行读取
    for file_name in os.listdir(args.fin_directory):
        if file_name.endswith('.json'):
            # check：如果当前文件属于finished_list，说明之前已经运行完了；如果当前文件不属于task_list，说明不在任务列表里，不需要运行
            if file_name.split(".")[0] in args.finished_list or file_name.split(".")[0] not in args.task_list:
                continue
            # 用一个json文件记录当前数据集的输出结果
            record_file = open(args.result_log_pred_dir + file_name[:-5] + "_pred_record.json", "a", encoding="utf-8")
            with open(args.fin_directory + file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for i, EMR_dict in tqdm(enumerate(data), total=len(data), desc=f'开始运行{file_name.split(".")[0]}诊断程序', file=sys.__stdout__):
                    print(f"\n{'='*80}")
                    print(f"【开始处理】科室: {file_name.split('.')[0]}, 病例索引: {i+1}")
                    print(f"{'='*80}")
                    # check: cur_dep和cur_idx分别表示程序上一次运行停在了哪个科室的具体哪个病历上
                    if file_name.split(".")[0] == args.cur_dep and i < args.cur_idx: 
                        continue
                    if i == args.stop_idx: 
                        break
                    # 用于记录输出信息
                    pred_dict = {}
                    # 输入病历的各项内容
                    personal_info = EMR_dict["基本信息"] if "基本信息" in EMR_dict else "无"
                    chief_complaint = EMR_dict["主诉"] if "主诉" in EMR_dict else "无"
                    current_medical_history = EMR_dict["现病史"] if "现病史" in EMR_dict else "无"
                    past_disease_history = EMR_dict["既往史"] if "既往史" in EMR_dict else "无"
                    body_check = EMR_dict["查体"] if "查体" in EMR_dict else "无"
                    # auxiliary_exam = EMR_dict["辅助检查"] if "辅助检查" in EMR_dict else "无"
                    auxiliary_exam = (EMR_dict.get("辅助检查", "") if EMR_dict.get("辅助检查", "") != "无" else "") + "\n" + EMR_dict.get("主要检查结果", "")
                    label = EMR_dict["label"]
                    # 对病历内容进行预处理
                    if auxiliary_exam != "无":
                        auxiliary_exam = ae_preprocess(auxiliary_exam)
                    # 主函数
                    final_response, vis_data, diag_info = main(doctor, KG_Tools, chief_complaint, current_medical_history, past_disease_history, body_check, auxiliary_exam, args, i)
                    # 记录输出结果
                    pred_dict = {
                        "index": i + 1,
                        "pred_response": final_response,
                        "visualization": vis_data,
                        "label": label,
                        # 移除重复字段，只保留diagnosis_info中的核心数据
                        "diagnosis_info": diag_info,  # 包含新格式的reranked_diagnoses
                    }
                    # 将结果写入当前文件
                    record_file.write(json.dumps(pred_dict, ensure_ascii=False) + "\n")
                    record_file.flush()
                    # 将结果按行写入总文件
                    total_record_file.write(json.dumps(pred_dict, ensure_ascii=False) + "\n")
                    total_record_file.flush()
            record_file.close()
    
    # ->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->
    # Calculate result
    print("diagnosis program over!!!")
    total_record_file.close()
    # 关闭运行的neo4j session
    kg_system.close()