# generate_ref.py
import json
import xlrd
from rapidfuzz import process, fuzz
from tqdm import tqdm
import os

# 加载 ICD 数据库（和 evaluate_emr.py 一致）
ICD_PATH = '/home/a/SSW/RE-MCDF/src/evaluate/国际疾病分类ICD-10北京临床版v601.xls'
xls = xlrd.open_workbook(ICD_PATH)
sheet = xls.sheet_by_index(0)
disease_ids = sheet.col_values(colx=0, start_rowx=1)
disease_names = sheet.col_values(colx=1, start_rowx=1)
ICD_disease = {name: did for name, did in zip(disease_names, disease_ids)}

TOP_N = 10
THRESHOLD = 50  # 实际存储时可保留所有，评估时再过滤

# 假设你的原始数据在这里
RAW_DATA_DIR = "/home/a/SSW/medIKAL/data/XMEMRs/data8/"  # 包含 儿科.json, 内科.json 等
OUTPUT_REF_DIR = "/home/a/SSW/medIKAL/data/XMEMRs/ref8/"

os.makedirs(OUTPUT_REF_DIR, exist_ok=True)

# 获取所有需要处理的科室文件
dept_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json') and f != 'ref']

# 外层进度条：按科室处理
for dept_file in tqdm(dept_files, desc="Processing departments"):
    dept_name = dept_file.split('.')[0]
    with open(os.path.join(RAW_DATA_DIR, dept_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ref_lines = []
    # 内层进度条：处理该科室下的每条病历
    for i, emr in enumerate(tqdm(data, desc=f"  {dept_name}", leave=False)):
        index = i + 1
        labels = emr.get("label", [])
        if isinstance(labels, str):
            labels = [labels]
        
        ref_match = []
        for label in labels:
            if not label or label in ["无", "暂无"]:
                continue
            matches = process.extract(
                str(label),
                list(ICD_disease.keys()),
                scorer=fuzz.WRatio,
                limit=TOP_N
            )
            # 只保留分数 >= THRESHOLD 的结果，并转为 [name, icd_id, score]
            formatted = [
                [m[0], ICD_disease[m[0]], int(m[1])] 
                for m in matches 
                if m[1] >= THRESHOLD
            ]
            if formatted:
                ref_match.append(formatted)
        
        ref_lines.append({
            "index": index,
            "ref_match": ref_match
        })
    
    # 写入 _ref.json
    output_path = os.path.join(OUTPUT_REF_DIR, f"{dept_name}_ref.json")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for line in ref_lines:
            out_f.write(json.dumps(line, ensure_ascii=False) + "\n")