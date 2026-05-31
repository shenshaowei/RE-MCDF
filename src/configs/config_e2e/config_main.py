config_dict = {}
config_dict["cuda_device"] = "1"  # 比如根据命令行或实验需求
config_dict["dataset"] = "XMEMRs"
# 模型的版本，通过是否带有"api"来区别本地模型和api
# config_dict["model_version"] = "qwen2.5_api"
config_dict["model_version"] = "Qwen2.5-7B-Instruct-hf"
print(f"使用模型{config_dict['model_version']}")
# 模型的路径(如果用的是modelscope则此参数不需要填，默认都存在.cache里)
config_dict["model_name_or_path"] = "./models/Qwen2.5-7B-Instruct"
config_dict["bge_model_path"] = "./models/bge-small-zh-v1.5"
config_dict["icd_database_path"] = "./opendata/国际疾病分类ICD-10北京临床版v601.xls"
# config_dict["model_type"] = None
config_dict["model_type"] = "qwen"
config_dict["api_key"] = "YOUR_API_KEY_HERE"
# config_dict["api_key"] = "YOUR_API_KEY_HERE"

# 在 config_dict 中添加新参数
config_dict["max_path_hops"] = 3  # 最大路径跳数
config_dict["path_evidence_threshold"] = 0.3  # 路径证据阈值

# ========== 新增：消融实验配置 ==========
# config_dict中添加消融实验配置参数
config_dict["use_relation_mining"] = True  # 是否使用关系挖掘
config_dict["use_kg_knowledge"] = True     # 是否使用KG知识
config_dict["use_lab_expert"] = False       # 是否使用检验科专家
config_dict["use_dynamic_weighting"] = True  # 启用动态权重
# ==========================================
if "glm" in config_dict["model_version"]:
    config_dict["model_type"] = "glm"
elif "baichuan" in config_dict["model_version"]:
    config_dict["model_type"] = "baichuan"
elif "qwen" or "Qwen" in config_dict["model_version"]:
    if "qwen_api" in config_dict["model_version"]:
        config_dict["model_type"] = "qwen_api"
        config_dict["model_version_api"] = "Qwen2.5-7B-Instruct"
    elif "qwen2.5_api" in config_dict["model_version"]:
        config_dict["model_type"] = "qwen2.5_api"
        config_dict["model_version_api"] = "Qwen2.5-7B-Instruct"
    else:
        config_dict["model_type"] = "qwen"
# 数据相关：输入数据的文件夹路径
config_dict["fin_directory"] = f'./data/STROKE/'
# NER模型，默认为RaNER模型
config_dict["ner_model_id"] = "./models/nlp_raner_named-entity-recognition_chinese-base-cmeee"
# retriever
config_dict["retriever_type"] = "sparse" # 检索器的类型：["sparse", "dense", "hybrid"]
config_dict["retriever_version"] = "bm25_merge_kg" # 检索器的基模型：sr: ["sparse"]; dr: ["dr_corom_emb",  "dr_bge_emb"]; hr: ["hr_bm25_corom", "hr_bm25_bge"]
# config_dict["retriever_type"] = "dense" # 检索器的类型：["sparse", "dense", "hybrid"]
# config_dict["retriever_version"] = "dr_corom_emb" # 检索器的基模型：sr: ["bm25_merge_kg"]; dr: ["dr_corom_emb",  "dr_bge_emb"]; hr: ["hr_bm25_corom", "hr_bm25_bge"]
# neo4j，知识图谱相关参数
config_dict["uri"] = "bolt://localhost:7687" # neo4j连接ip
config_dict["username"] = "neo4j" # neo4j连接用户名
config_dict["password"] = "YOUR_NEO4J_PASSWORD" # neo4j连接密码
config_dict["kg_database_name"] = "medkg" # 图数据库的名字
config_dict["kg_type"] = "merge" # 类型，这个参数主要是为了加载下面几个文件
config_dict["kg_entity_path"] = './kgfiles/KG_entities2id_{}.txt'.format(config_dict["kg_type"]) # 知识图谱的全部实体
config_dict["entity_type_map_path"] = "./kgfiles/entity_type_map_{}.json".format(config_dict["kg_type"]) # 知识图谱全部的实体->实体类型的映射关系
config_dict["subgraph_name"] = "subgraph" # gds子图的名字
# neo4j， 实体权重相关参数
config_dict["entity_weight_map_file"] = "./opendata/entity_weight_emr.json"
# 候选疾病相关参数
config_dict["direct_topn"] = 3 # LLM直接进行预测的疾病个数
# config_dict["dis_topn"] = 2 * config_dict["direct_topn"] # rerank之前保留的候选疾病个数
# config_dict["dis_topn"] = 2 + config_dict["direct_topn"] # rerank之前保留的候选疾病个数
config_dict["dis_topn"] = 1
config_dict["rerank_topn"] = config_dict["direct_topn"] # rerank之后保留的候选疾病的个数
config_dict["final_topn"] = config_dict["direct_topn"] # 最终保留的候选疾病的个数
config_dict["kg_supplement_topk"] = 1  # KG补充疾病的数量
config_dict["path_topn"] = 1

# 总的科室列表：["儿科", "耳鼻咽喉科", "妇产科", "护理科", "急诊科", "精神科", "康复科", "口腔科", "麻醉疼痛科", "内科", "皮肤性病科", "外科", "眼科", "肿瘤科"]
# 已经完成的数据集/科室列表
config_dict["finished_list"] = [""]
# 本次运行需要完成的任务列表
config_dict["task_list"] = ["急诊科", "内科"]
# config_dict["task_list"] = ["内科"]
# config_dict["task_list"] = ["儿科", "耳鼻咽喉科", "妇产科", "护理科", "急诊科", "精神科", "康复科", "口腔科", "麻醉疼痛科", "内科", "皮肤性病科", "外科", "眼科", "肿瘤科"]
# 上次运行终止的任务/科室/数据集
config_dict["cur_dep"] = ""
# 上次运行终止的断点位置索引
config_dict["cur_idx"] = -1 #输入i的话从i+1开始继续 例如636会从637开始
# 本次运行终止的断点位置索引
config_dict["stop_idx"] = 3500
# 输出相关：通常来说，输出需要记录-1.用了哪个模型；2.检索器；3.候选疾病的数量；
import time
config_dict["result_log_pred_dir"] = f"./output/v1/{config_dict['dataset']}/{config_dict['dataset']}_{config_dict['model_version']}_{config_dict['retriever_version']}_{config_dict['direct_topn']}_{config_dict['path_evidence_threshold']}_{config_dict['use_relation_mining']}_{config_dict['use_kg_knowledge']}_{config_dict['use_lab_expert']}_{config_dict['use_dynamic_weighting']}_kg_{config_dict['kg_supplement_topk']}/"

config_dict["log_dir"] = config_dict["result_log_pred_dir"] + f"logs_{config_dict['model_version']}_{config_dict['retriever_version']}_{config_dict['direct_topn']}_{config_dict['path_evidence_threshold']}_{config_dict['use_relation_mining']}_{config_dict['use_kg_knowledge']}_{config_dict['use_lab_expert']}_{config_dict['use_dynamic_weighting']}/"