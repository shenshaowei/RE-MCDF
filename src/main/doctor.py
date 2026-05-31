import re, json
class Doctor:
    def __init__(self, chat_model):
        self.chat_model = chat_model
    
    def general_info_summary(self, chief_complaint, current_medical_history, disease_history):
        prompt = "你是一名优秀的AI医学专家，你可以基于患者的病历内容总结用于诊断的关键信息。以下是一个真实的某患者的电子病历的部分内容，请你仔细阅读以下内容，了解患者的基本情况。\n" \
            + '"""\n' \
            + f"【主诉】：{chief_complaint}\n"\
            + f"【现病史】：{current_medical_history}\n" \
            + f"【既往史】：{disease_history}\n" \
            + '"""\n' \
            + "##任务：\n{请你根据上述内容，总结出对于诊断和治疗有用的关键信息，并生成一份总结报告，报告有具体的格式要求}。\n"\
            + "##报告格式要求：\n{请按照以下格式，填写\"[]\"处的内容，完成报告。语言请尽可能地简洁。}\n"\
            + "1.出现症状：[]\n"\
            + "2.近期就诊经历：[](没有则填无)\n"\
            + "3.既往疾病史：[](没有则填无)\n" \
            + "4.既往手术史：[](没有则填无)\n" \
            + "5.药物使用情况：[](没有则填无)\n" \
            + "请严格按照上述格式进行总结。\n"\
            + "现在请总结：\n"
        fst_rd_summary, fst_rd_history = self.chat_model.chat_([prompt])
        return fst_rd_summary, fst_rd_history
    
    def examination_summary(self, body_check, auxiliary_exam):
        prompt =  "你是一名优秀的AI医学专家，你可以基于患者的检查结果总结用于诊断的关键信息。" \
            + "患者的检查结果如下所示：\n"\
            + '"""\n' \
            + f"【查体结果】：{body_check}\n" \
            + f"【辅助检查结果】：{auxiliary_exam}\n" \
            + '"""\n' \
            + "目前患者的检查结果有很多冗余的内容，请你对上述检查结果进行总结。\n"\
            + "##任务：\n{请你对上述患者检查结果进行总结和概括，保留对诊断有用的信息，删除那些对诊断意义不大的内容。}\n"\
            + "##要求：\n{请按照以下格式，填写\"[]\"处的内容，完成报告。语言清尽可能地简洁。}\n"\
            + "1.查体结果：[]\n"\
            + "2.辅助检查结果：[]\n"\
            + "你的总结：\n"
        scd_rd_summary, scd_rd_history = self.chat_model.chat_([prompt])
        physical_exam = ""
        aux_exam = ""
        # 提取"查体结果："到"辅助检查结果："之间的内容
        if "查体结果：" in scd_rd_summary:
            after_pe = scd_rd_summary.split("查体结果：", 1)[1]  # 只 split 一次，取后面部分
            if "辅助检查结果：" in after_pe:
                physical_exam = after_pe.split("辅助检查结果：", 1)[0].strip()
            else:
                physical_exam = after_pe.strip()
        else:
            physical_exam = ""
        # 提取"辅助检查结果："之后的内容（直到换行或结尾）
        if "辅助检查结果：" in scd_rd_summary:
            after_ae = scd_rd_summary.split("辅助检查结果：", 1)[1]
            # 取第一行或全部
            aux_exam = after_ae.split("\n")[0].strip()
        else:
            aux_exam = ""
        # 返回拼接后的关键信息
        final_summary = f"查体结果：{physical_exam}\n辅助检查结果：{aux_exam}"
        return final_summary, scd_rd_history
    
    def examination_summary2(self, body_check, auxiliary_exam):
        prompt = (
            "你是一名资深内科医生，请结合患者的查体和辅助检查结果，完成有价值的检查总结。请仅保留对诊断有实际价值的异常体征或检查发现，剔除完全正常、常规筛查无异常或临床意义明确缺失的项目。总结需高度精炼，按临床逻辑整合，避免罗列数据。\n" \
         + '"""\n' \
         + f"【查体结果】：{body_check}\n" \
         + f"【辅助检查结果】：{auxiliary_exam}\n" \
         + '"""\n' \
         + "请严格按以下格式输出，不得添加额外内容或解释：\n" \
         + "检查结果：[]"
         + "现在请输出：\n"
        )

        scd_rd_summary, scd_rd_history = self.chat_model.chat_([prompt])
        # 初始化为空
        physical_exam = ""
        aux_exam = ""
        if "检查结果：" in scd_rd_summary or "检查结果:" in scd_rd_summary:
            content = scd_rd_summary.split("检查结果：", 1)[1].strip() if "检查结果：" in scd_rd_summary else scd_rd_summary.split("检查结果:", 1)[1].strip()
            if content.startswith("[") and content.endswith("]"):
                content = content[1:-1].strip()
            content = content.strip("\"' ")
            final_summary = content
        else:
            final_summary = scd_rd_summary.strip()

        return final_summary, scd_rd_history
    
    # 1. 生成最多{topn}个最可能的诊断
    def generate_diagnosis_with_evidence(self, fst_rd_summary, scd_rd_summary, topn=5):
        """
        生成诊断假设与关键证据实体
        """
        prompt = f"""你是一名经验丰富的临床医生，需要基于患者信息进行疾病诊断，并生成每个疾病对应的关键证据实体。

患者基本信息：
{fst_rd_summary}

检查结果：
{scd_rd_summary}

### 任务要求：
1. 给出top-{topn}个最可能的诊断
2. 为每个诊断列出关键的证据实体
3. 证据实体必须是具体的医学概念，并且与诊断高度相关
4. 严格按照以下格式输出，不要添加任何额外内容

### 输出格式：
{{
  "diagnoses": [
    {{
      "disease": "疾病名称1",
      "key_evidence": ["证据实体1", "证据实体2", ...]
    }}
  ]
}}

你的输出：
"""
        print("【初级诊断专家】prompt:", prompt)  # 调试用
        result, history = self.chat_model.chat_([prompt])
        # print("【初级诊断专家】输出:", result)  # 调试用
        result = re.sub(r'```json\s*|```', '', result)
        print("【初级诊断专家】清理后输出:", result)
        # 尝试解析JSON
        try:
            import json
            diagnoses = json.loads(self._extract_json(result))
            return diagnoses, history
        except Exception as e:
            print(f"JSON解析错误: {e}, 原文: {result}")
            return self._parse_diagnoses_from_text(result), history
# ========== 消融实验：只生成诊断 ==========
    def generate_diagnosis_with_evidence2(self, fst_rd_summary, scd_rd_summary, topn=5):
        """
        消融实验用：只生成诊断，无证据
        """
        prompt = f"""你是一名临床医生。请基于患者信息生成诊断。

基本信息：
{fst_rd_summary}

检查结果：
{scd_rd_summary}

### 要求：
1. 生成最多{topn}个最可能的诊断
2. 严格按以下格式输出：

{{
  "diagnoses": [
    {{"disease": "疾病1"}},
    {{"disease": "疾病2"}}
  ]
}}

你的输出：
"""
        print("【初级诊断专家】prompt:", prompt)  # 调试用
        result, history = self.chat_model.chat_([prompt])
        print("【初级诊断专家】输出:", result)  # 调试用
        try:
            import json
            json_str = self._extract_json(result)
            data = json.loads(json_str)
            return data, history
        except Exception as e:
            print(f"JSON解析失败: {e}")
            return {"diagnoses": []}, history
    
    def _parse_diagnoses_from_text(self, text):
        """备用解析方法，当JSON解析失败时使用"""
        import re, json
        diagnoses = {"diagnoses": []}
        
        # 提取疾病和证据
        disease_blocks = re.split(r'预测疾病\d+[:：]', text)[1:]
        
        for block in disease_blocks[:5]:
            disease_match = re.search(r'^([^\\\n]+)', block)
            evidence_match = re.findall(r'[\-•]\s*([^\n]+)', block)
            
            if disease_match:
                disease = disease_match.group(1).strip()
                # 证据提取
                evidence = [e.strip() for e in evidence_match[:3] if len(e.strip()) > 1]
                
                if not evidence:
                    evidence = ["相关症状", "相关体征"]
                
                diagnoses["diagnoses"].append({
                    "disease": disease,
                    "key_evidence": evidence
                })
        
        return diagnoses
    # ==========================================
    
 
    def _extract_json(self, text):
        """提取文本中的JSON部分"""
        import re, json
        # 查找JSON对象开始和结束
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx]
            try:
                # 验证是否为有效JSON
                json.loads(json_str)
                return json_str
            except:
                pass
        
        # 尝试用正则提取
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()
        
        return text
    def parse_direct_diagnosis(self, raw_text: str, topn: int = 5):
        """
        解析 direct_diagnos 返回的纯文本，提取疾病名称列表
        输入示例：
            "预测疾病1：脑梗死\n预测疾病2：高血压\n预测疾病3：糖尿病"
        输出示例：
            ["脑梗死", "高血压", "糖尿病"]
        """
        diseases = []
        lines = raw_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 支持中文冒号和英文冒号
            if '：' in line:
                parts = line.split('：', 1)
            elif ':' in line:
                parts = line.split(':', 1)
            else:
                continue
            
            if len(parts) < 2:
                continue
                
            disease = parts[1].strip()
            # 过滤无效内容
            if disease and disease not in {"", "无", "*", "暂无", "无诊断"}:
                diseases.append(disease)
        
        # 去重并限制数量
        seen = set()
        unique_diseases = []
        for d in diseases:
            if d not in seen:
                seen.add(d)
                unique_diseases.append(d)
        
        return unique_diseases[:topn]

    def generate_dynamic_entity_weights(self, fst_rd_summary: str, scd_rd_summary: str, total_ner_dict: dict):
        """
        基于病历摘要生成动态实体权重和提取异常实体
        :return: (动态权重字典, 异常实体列表)
        """
        # 提取各类型实体示例
        entity_examples = {}
        for ent_type, entities in total_ner_dict.items():
            if entities:
                entity_examples[ent_type] = [ent['EMR_entity'] for ent in entities[:2]]
        examples_str = "\n".join([f"- {ent_type}: {', '.join(exs)}" for ent_type, exs in entity_examples.items()])
        
        prompt = f"""你是一名资深临床专家，需要完成两个任务：
    任务1：评估不同医学实体类型在当前病例中的诊断价值
    任务2：从患者信息中识别能指示特定疾病的异常医学实体

    【患者基本信息摘要】
    {fst_rd_summary}
    【检查结果摘要】
    {scd_rd_summary}
    【当前识别到的实体示例】
    {examples_str if examples_str else "无特定实体识别到"}

    ### 任务1要求
    1. 评估以下9类实体在当前病例中的诊断价值（越重要权重越高）：
    - sym (symptoms): 症状
    - dis (diseases): 疾病
    - dru (drugs): 药品
    - bod (body parts): 肌体
    - ite (examination items): 检查项目
    - equ (equipment): 医疗器械
    - mic (microorganisms): 微生物
    - dep (departments): 科室
    - pro (procedures): 医疗程序
    2. 输出按照格式输出，键为上述9个实体类型，值为0.0001-1.0的权重，总和必须为1

    ### 任务2要求
    1. 识别出具备诊断价值的异常医学实体证据
    2. 这些异常实体必须:
    - 有明确的临床意义，能作为指向特定疾病的证据
    - 与患者情况高度相关
    3. 优先考虑异常指标、影像学发现、关键症状等证据
    4. 仅输出相关的异常实体名称，与原始实体保持一致，不要添加任何描述、原因或括号

    ### 请严格按照以下输出格式输出：
    {{
    "entity_weights": {{
    "sym": <float>,
    "dis": <float>,
    "dru": <float>,
    "bod": <float>,
    "ite": <float>,
    "equ": <float>,
    "mic": <float>,
    "dep": <float>,
    "pro": <float>
    }},
    "abnormal_entities": ["实体1", "实体2", "实体3", ...]
    }}
    你的输出：
    """
        try:
            result, _ = self.chat_model.chat_([prompt])
            import json, re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
            weights_and_abnormal = json.loads(json_match.group() if json_match else result)
            
            # 验证和归一化权重
            required_keys = {"sym", "dis", "dru", "bod", "ite", "equ", "mic", "dep", "pro"}
            if set(weights_and_abnormal["entity_weights"].keys()) != required_keys:
                raise ValueError("缺少必要键")
            
            weights = weights_and_abnormal["entity_weights"]
            values = [max(0.0001, weights.get(k, 0.0001)) for k in required_keys]
            total = sum(values)
            if total == 0:
                raise ValueError("所有权重为零")
            normalized_weights = {k: v/total for k, v in zip(required_keys, values)}
            
            # 提取异常实体（直接使用LLM返回的原始实体名称）
            abnormal_entities = weights_and_abnormal.get("abnormal_entities", [])
            
            # 清洗：移除可能残留的描述性内容
            cleaned_entities = []
            for entity in abnormal_entities:
                # 移除括号及其内容，只保留实体名称
                entity_clean = re.sub(r'\(.*\)', '', entity).strip()
                # 移除中文括号
                entity_clean = re.sub(r'（.*）', '', entity_clean).strip()
                if entity_clean:
                    cleaned_entities.append(entity_clean)
            
            print(f"[动态权重&异常实体] 生成成功 - {len(cleaned_entities)}个异常实体识别")
            # print(f"  异常实体: {cleaned_entities}")
            
            return normalized_weights, cleaned_entities
            
        except Exception as e:
            print(f"[动态权重&异常实体] 生成失败: {e}. 使用默认权重和异常实体提取.")
            default_weights = {
                "sym": 0.6297, "dis": 0.1638, "dru": 0.1391,
                "bod": 0.0212, "ite": 0.0372, "equ": 0.0029,
                "mic": 0.0009, "dep": 0.0004, "pro": 0.0043
            }
            
            abnormal_entities = []
            return default_weights, abnormal_entities
            
        except Exception as e:
            print(f"[动态权重&异常实体] 生成失败: {e}. 使用默认权重和异常实体提取.")
            default_weights = {
                "sym": 0.6297, "dis": 0.1638, "dru": 0.1391,
                "bod": 0.0212, "ite": 0.0372, "equ": 0.0029,
                "mic": 0.0009, "dep": 0.0004, "pro": 0.0043
            }
            
            abnormal_entities = []
            return default_weights, abnormal_entities
    
    def agent_a_single_disease_evaluator(self, diseases_with_evidence, emr_context):
        """
        智能体A：单疾病证据评估专家（批量版）
        一次性评估多个疾病的证据强度，考虑具体连接路径
        """
        # 构建疾病列表
        diseases_list = []
        for i, item in enumerate(diseases_with_evidence):
            disease = item['disease']
            evidence_entities = item['evidence_entities']  # 限制显示5个证据实体
            kg_connectivity = item['kg_connectivity']
            icd_similarity = item['icd_similarity']
            
            # 构建连接路径文本
            connection_details = []
            
            # 1. 添加疾病三元组信息
            triples = item.get('disease_triples', [])
            if triples:
                triple_texts = []
                for triple in triples:
                    triple_texts.append(f"{triple['head']} --[{triple['relation']}]--> {triple['tail']} ({triple['tail_type']})")
                triples_text = "\n".join(triple_texts)
                connection_details.append(f"疾病三元组:\n{triples_text}")
            
            # 2. 添加连接路径详情
            path_details = item.get('connection_paths', [])
            if path_details:
                path_texts = []
                for path in path_details[:3]:  # 限制显示3条路径
                    path_texts.append(f"- {path['entity']}:\n  {path['path']} (长度: {path['length']})")
                connection_details.append("证据连接路径:\n" + "\n".join(path_texts))
            
            connection_text = "\n".join(connection_details) if connection_details else "无详细连接信息"
            
            # 构建疾病描述
            disease_text = (
                f"{i+1}. 疾病: {disease}\n"
                f"   证据实体: {', '.join(evidence_entities)}\n"
                f"   知识图谱连通性: {kg_connectivity:.3f}\n"
                f"   ICD-10标准化匹配度: {icd_similarity:.1f}%\n"
                f"   连接详情:\n{connection_text}"
            )
            diseases_list.append(disease_text)
        
        diseases_list = "\n\n".join(diseases_list)
        
        prompt = f"""你是一名医学诊断专家，结合患者信息、初诊医生给出的初诊以及诊断的证据，你需要结合候选疾病、患者信息、证据在知识图谱的连通性等信息，对疾病进行评分。
    【患者信息】
    {emr_context}
    【候选疾病列表】
    {diseases_list}
    ### 任务
    1. 综合评估每个疾病与患者证据的匹配程度
    2. 考虑疾病的临床表现是否与患者症状一致
    3. 考虑疾病的诊断标准是否被满足
    4. 分析证据在知识图谱中的具体连接路径
    - 评估路径的医学意义
    - 评估连接的直接性和特异性
    - 考虑三元组信息中包含的医学关系
    5. 结合多个维度综合评估疾病可能性，包括但不限于：
    - 疾病与患者临床表现的整体匹配度
    - 证据的可靠性和强度
    - 知识图谱中的关联性
    - 疾病的流行病学特征
    - 与其他疾病的可能性比较
    6. 为每个疾病生成0.0-1.0之间的综合评分
    7. 不止考虑证据的强弱，若疾病所支撑的证据足够强但不符合患者情况，请降低评分
    8. 严格参考以下输出格式
    ### 输出格式
    {{
    "scores": {{
    "疾病1": 评分1,
    "疾病2": 评分2,
    ...
    }},
    "reasoning": "简要解释评分依据，说明综合考虑了哪些维度"
    }}
    你的输出（请严格参考输出格式）:
    """
        # 保持原有的解析和错误处理代码不变
        try:
            print("  [智能体A] 原始输入:", prompt)
            result, _ = self.chat_model.chat_([prompt])
            print("  [智能体A] 原始输出:", result)
            import json, re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
            print("  [智能体A] 提取的JSON:", json_match.group() if json_match else "无")
            if json_match:
                scores_data = json.loads(json_match.group())
                scores = scores_data.get("scores", {})
                reasoning = scores_data.get("reasoning", "未提供推理")
                print(f"  [智能体A] 推理: {reasoning}")
                return scores
            else:
                print(f"  [智能体A] JSON解析失败，使用默认评分")
                return {item['disease']: 0.5 for item in diseases_with_evidence}
        except Exception as e:
            print(f"  [智能体A] 评估失败: {e}，使用默认评分")
            return {item['disease']: 0.5 for item in diseases_with_evidence}
            
    def agent_b_multi_disease_evaluator(self, diseases, disease_relations, disease_scores, emr_context, high_similarity_pairs=None):
        """
        智能体B：多疾病关系评估专家
        考虑疾病间的互斥和混淆关系，调整疾病评分
        """
        # 构建关系描述
        relation_descriptions = []
        print(f"  [互斥] {', '.join([f'{d1}↔{d2}' for d1, d2, _, _ in disease_relations.exclusive_pairs]) if disease_relations.exclusive_pairs else '无互斥'}")
        
        # 添加互斥关系描述
        for d1, d2, reason, description in disease_relations.exclusive_pairs:
            score1 = disease_scores.get(d1, 0.0)
            score2 = disease_scores.get(d2, 0.0)
            relation_descriptions.append(
                f"- 互斥关系: {d1} ↔ {d2}\n"
                f"  原因: {reason} - {description}\n"
                f"  当前评分: {d1}={score1:.3f}, {d2}={score2:.3f}"
            )
        
        # 添加相似关系描述
        for d1, d2, sim_score in disease_relations.similar_pairs:
            score1 = disease_scores.get(d1, 0.0)
            score2 = disease_scores.get(d2, 0.0)
            relation_descriptions.append(
                f"- 相似关系: {d1} ↔ {d2}\n"
                f"  相似度: {sim_score:.3f}\n"
                f"  当前评分: {d1}={score1:.3f}, {d2}={score2:.3f}"
            )
        
        relations_text = "\n".join(relation_descriptions) if relation_descriptions else "无显著疾病间关系"
        
        # 构建疾病列表
        disease_list_text = "\n".join([f"{i+1}. {disease}: 证据分={disease_scores.get(disease, 0.0):.3f}"
                                    for i, disease in enumerate(diseases)])
        
        prompt = f"""你是一名医学逻辑推理专家，需要评估多个疾病间的逻辑关系，并调整评分。
    【患者信息】
    {emr_context}
    【候选疾病列表】
    {disease_list_text}
    【疾病间关系】
    {relations_text}
    ### 任务
    1. 识别互斥疾病对：如果两个疾病在病理上不可能共存，且一个疾病的证据链路清晰于另一个，则适当降低另一疾病的评分
    2. 识别混淆疾病对：如果两个疾病症状高度相似，且一个疾病的证据链路清晰于另一个，则适当调整评分以反映不确定性
    3. 为互斥疾病和混淆疾病生成0.0-1.0的调整系数
    4. 若无需调整的疾病，所有疾病的调整系数均为1.0，并且理由中说明未发现需调整的疾病
    5. 以请严格参考输出格式输出调整系数
    ### 输出格式
    {{
    "adjustment_factors": {{
    "疾病1": 调整系数1,
    "疾病2": 调整系数2,
    "疾病3": 调整系数3,
    ...
    }},
    "reasoning": "简要解释调整原因，特别说明哪些疾病被调整及原因"
    }}
    你的输出（请严格参考输出格式）:
    """
        try:
            result, _ = self.chat_model.chat_([prompt])
            print("  [智能体B] 原始输入:", prompt)
            print("  [智能体B] 原始输出:", result)
            import json, re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
            if json_match:
                adjustment_data = json.loads(json_match.group())
                adjustment_factors = adjustment_data.get("adjustment_factors", {})
                reasoning = adjustment_data.get("reasoning", "未提供推理")
                print(f"  [智能体B] 推理: {reasoning}")
                
                final_adjustment_factors = {disease: 1.0 for disease in diseases}
                for disease, factor in adjustment_factors.items():
                    if disease in final_adjustment_factors:
                        clamped_factor = max(0.0, min(1.0, float(factor)))
                        final_adjustment_factors[disease] = clamped_factor
                        if clamped_factor != float(factor):
                            print(f"  [智能体B] 警告: 系数 {factor} 超出范围，已调整为 {clamped_factor}")
                return final_adjustment_factors
            else:
                print(f"  [智能体B] JSON解析失败，使用默认调整系数")
                return {disease: 1.0 for disease in diseases}
        except Exception as e:
            print(f"  [智能体B] 评估失败: {e}，使用默认调整系数")
            return {disease: 1.0 for disease in diseases}

    def agent_c_final_integrator(self, disease_scores, icd_threshold=70.0, icd_weight=0.2, gamma=1.5):
            """
            智能体C：最终融合专家
            严格对齐论文公式 (10): F_fin(d) = phi(S_ICD) * (H(d)*(1 - w_ICD) + B_ICD * S_ICD)
            """
            final_scores = {}
            for disease, details in disease_scores.items():
                evidence_score = details['evidence_score']
                logic_score = details['logic_score']
                connectivity = details['connectivity']
                icd_sim = details['icd_similarity']  # 0-100
                
                # 1. 计算核心复合分 H(d) (对齐论文 \beta_1, \beta_2, \beta_3)
                h_score = 0.5 * evidence_score + 0.3 * logic_score + 0.2 * connectivity
                
                # 2. ICD 惩罚函数 phi(S_ICD) (对齐论文公式 11)
                if icd_sim >= icd_threshold:
                    phi = 1.0
                else:
                    phi = (icd_sim / icd_threshold) ** gamma
                    
                # 3. 最终融合 (统一使用传入的 icd_weight 作为 w_ICD 和 B_ICD)
                icd_norm = icd_sim / 100.0
                final_score = phi * (h_score * (1.0 - icd_weight) + icd_weight * icd_norm)
                
                final_scores[disease] = max(0.0, min(1.0, final_score))
            return final_scores    
          
    # def agent_c_final_integrator(self, disease_scores):
    #     """
    #     智能体C：最终融合专家（基于规则）
    #     融合证据分、逻辑分、连通性和ICD相似度
    #     """
    #     final_scores = {}
        
    #     for disease, details in disease_scores.items():
    #         evidence_score = details['evidence_score']
    #         logic_score = details['logic_score']
    #         connectivity = details['connectivity']
    #         icd_similarity = details['icd_similarity'] / 100.0  # 转换为0-1
            
    #         # 基于规则的加权融合
    #         # 证据强度和逻辑一致性占主要权重
    #         final_score = (
    #             0.5 * evidence_score + 
    #             0.3 * logic_score + 
    #             0.15 * connectivity + 
    #             0.05 * icd_similarity
    #         )
    #         print(f"【{disease}】证据:{evidence_score:.3f}×0.5 + 逻辑:{logic_score:.3f}×0.3 + 连通:{connectivity:.3f}×0.15 + ICD:{icd_similarity:.3f}×0.05 = {final_score:.3f}")
    #         # 确保在[0,1]范围内
    #         final_score = max(0.0, min(1.0, final_score))
    #         final_scores[disease] = final_score
        
    #     return final_scores
    
    def analyze_disease_exclusivity(self, disease_context, diseases, disease_graph_context, disease_relations_batch, bge_text="无"):
        """
        专门分析疾病间的互斥关系，使用重构后的批量关系数据
        """
        exclusivity_relations = disease_relations_batch.get('exclusivity_relations', {})
        
        # ====== 严格过滤：只保留"病理分型"关系 ======
        filtered_exclusivity_relations = {}
        for d1, targets in exclusivity_relations.items():
            filtered_targets = {}
            for d2, relations in targets.items():
                # 仅保留关系类型为"病理分型"的关系
                valid_relations = [
                    rel for rel in relations 
                    if rel.get('relation_type') == '病理分型'
                ]
                if valid_relations:
                    filtered_targets[d2] = valid_relations
            if filtered_targets:
                filtered_exclusivity_relations[d1] = filtered_targets
        # =================================================
        
        # 构建关系描述（仅病理分型）
        relation_descriptions = []
        processed_pairs = set()
        
        for i, d1 in enumerate(diseases):
            for d2 in diseases[i+1:]:
                pair_key = tuple(sorted([d1, d2]))
                if pair_key in processed_pairs:
                    continue
                    
                # 检查是否存在病理分型关系
                relations_found = []
                if d1 in filtered_exclusivity_relations and d2 in filtered_exclusivity_relations[d1]:
                    for rel in filtered_exclusivity_relations[d1][d2]:
                        if rel.get('relation_type') == '病理分型':
                            relations_found.append(rel)
                
                if relations_found:
                    relation_text = "\n".join([f"  • {rel['evidence']}" for rel in relations_found])
                    relation_descriptions.append(
                        f"- 病理分型关系: {d1} ↔ {d2}\n"
                        f"  证据:\n{relation_text}"
                    )
                    processed_pairs.add(pair_key)
        
        relations_text = "\n".join(relation_descriptions) if relation_descriptions else "无病理分型关系"
        prompt = f"""你是一名资深医学专家，专门分析疾病间是否存在互斥关系。互斥关系指两种疾病在病理生理机制上难以在同一患者身上同时存在。请严格遵循以下原则：

        【核心判定规则】
        1. 仅当满足以下任一条件时，才可判定为互斥：
        - 同一疾病的互斥亚型（例如：1型糖尿病 与 2型糖尿病）
        - 分子特征绝对冲突（例如：HER2阳性乳腺癌 与 HER2阴性乳腺癌）
        - 临床分期互斥（例如：急性心肌梗死 与 陈旧性心肌梗死）
        2. 绝对禁止将以下情况判为互斥：
        - 两种疾病存在因果关联（例如：高血压导致脑梗死）
        - 两种疾病可合并存在（例如：糖尿病与甲状腺功能减退）
        - 仅发病机制、病因或临床表现不同
        - 任何并发症、伴随病或共病关系
        3. 若不存在任何互斥疾病对，必须输出：{{"exclusive_pairs": []}}
        - 禁止输出"无"、"none"、"没有互斥对"等非 JSON 内容
        - 禁止在输出中包含非互斥对的解释或说明

        【患者病历上下文】
        {disease_context}

        【候选疾病列表】
        {', '.join(diseases)}

        【知识图谱证据】
        {relations_text if relations_text.strip() else "无相关知识图谱证据"}

        【任务要求】
        1. 严格基于医学证据判断：
        - 若无明确病理互斥证据，默认认为可共存
        - 优先依据知识图谱中的结构化关系，其次参考权威临床指南
        2. 输出必须严格遵守格式：
        - 仅当确定存在真正的病理互斥时，才输出对应疾病对
        - 每个互斥对必须提供简明、准确的病理机制冲突说明
        - 无互斥对时，只允许输出空数组
        3. 证据字段要求：
        - kg_evidence 必须引用知识图谱中的具体关系路径
        - 若无图谱证据，应写明"无直接证据"

        【严格输出格式示例】

        有互斥对时：
        {{
        "exclusive_pairs": [
            {{
            "disease1": "疾病A",
            "disease2": "疾病B",
            "reason": "病理机制冲突点（30字内）",
            "kg_evidence": "知识图谱关系路径"
            }}
        ]
        }}

        无互斥对时：
        {{
        "exclusive_pairs": []
        }}

        你的输出（必须严格符合上述 JSON 格式，不得包含额外内容）：
        """

        try:
            print("【互斥专家】输入提示词:", prompt)
            result, _ = self.chat_model.chat_([prompt])
            # print("【互斥专家】原始输出:", result)
            cleaned_result = re.sub(r'```json\s*|```', '', result)
            print("【互斥专家】清理后输出:", cleaned_result)
            
            # 尝试多种方法提取JSON
            exclusivity_data = self._robust_json_parse(cleaned_result)
            # 尝试多种方法提取JSON
            
            # 验证和标准化结构
            if "exclusive_pairs" not in exclusivity_data or not isinstance(exclusivity_data["exclusive_pairs"], list):
                exclusivity_data["exclusive_pairs"] = []
            
            # 验证和标准化每个互斥对
            validated_pairs = []
            for pair in exclusivity_data["exclusive_pairs"]:
                validated_pair = {
                    'disease1': str(pair.get('disease1', '')).strip(),
                    'disease2': str(pair.get('disease2', '')).strip(),
                    'reason': str(pair.get('reason', '专家判定存在病理互斥')).strip(),
                    'kg_evidence': str(pair.get('kg_evidence', '无知识图谱证据')).strip()
                }
                
                # 确保疾病名称有效且不相同
                if validated_pair['disease1'] and validated_pair['disease2'] and validated_pair['disease1'].lower() != validated_pair['disease2'].lower():
                    validated_pairs.append(validated_pair)
            
            exclusivity_data["exclusive_pairs"] = validated_pairs
            print(f"  [互斥专家] 识别到 {len(validated_pairs)} 个互斥对")
            return exclusivity_data
        except Exception as e:
            print(f"  [互斥专家] 分析失败: {e}")
            return {"exclusive_pairs": []}

    def analyze_disease_similarity(self, disease_context, diseases, disease_graph_context, disease_relations_batch, bge_text="无"):
        """
        专门分析疾病间的混淆关系（原相似关系），使用重构后的批量关系数据
        """
        # 获取混淆相关的关系 - 新结构
        similarity_relations = disease_relations_batch.get('similarity_relations', {})
        
        # 构建关系描述 - 重点处理属性重叠
        relation_descriptions = []
        processed_pairs = set()
        
        for i, d1 in enumerate(diseases):
            for d2 in diseases[i+1:]:
                pair_key = tuple(sorted([d1, d2]))
                if pair_key in processed_pairs:
                    continue
                    
                # 检查是否存在混淆关系
                relations_found = []
                confusion_analysis = None
                
                # 检查d1->d2的关系
                if d1 in similarity_relations and d2 in similarity_relations[d1]:
                    for rel in similarity_relations[d1][d2]:
                        # 区分直接关系和混淆分析
                        if rel.get('relation_type') == 'CONFUSION_ANALYSIS':
                            confusion_analysis = rel
                        else:
                            relations_found.append(rel)
                
                # 构建关系文本
                relation_text = ""
                if relations_found:
                    relation_text += "\n".join([f"  • {rel['evidence']}" for rel in relations_found])
                
                # 添加混淆分析信息
                if confusion_analysis:
                    overlap_attrs = confusion_analysis.get('overlap_attrs', [])
                    unique1 = confusion_analysis.get('unique_to_disease1', [])
                    unique2 = confusion_analysis.get('unique_to_disease2', [])
                    
                    if overlap_attrs:
                        overlapping_symptoms = [attr['value'] for attr in overlap_attrs if attr['type'] in ['临床表现', '相关（症状）']]
                        if overlapping_symptoms:
                            if relation_text:
                                relation_text += "\n"
                            relation_text += f"  • 共同临床表现: {', '.join(overlapping_symptoms)}"
                    
                    if unique1 or unique2:
                        key_differences = []
                        if unique1:
                            key_diffs1 = [attr['value'] for attr in unique1 if attr['type'] in ['临床表现', '影像学检查']]
                            if key_diffs1:
                                key_differences.append(f"{d1}特有: {', '.join(key_diffs1[:2])}")
                        if unique2:
                            key_diffs2 = [attr['value'] for attr in unique2 if attr['type'] in ['临床表现', '影像学检查']]
                            if key_diffs2:
                                key_differences.append(f"{d2}特有: {', '.join(key_diffs2[:2])}")
                        
                        if key_differences:
                            if relation_text:
                                relation_text += "\n"
                            relation_text += "  • 关键鉴别点: " + "; ".join(key_differences)
                
                if relations_found or confusion_analysis:
                    relation_descriptions.append(
                        f"- 混淆关系: {d1} ↔ {d2}\n"
                        f"  相关信息:\n{relation_text or '无明确混淆证据'}"
                    )
                    processed_pairs.add(pair_key)
        
        relations_text = "\n".join(relation_descriptions) if relation_descriptions else "未发现明确的混淆关系"
        
        prompt = f"""你是一名资深临床诊断专家。候选疾病是一位医生的初步诊断，你的任务是找出候选疾病中极易混淆的疾病对。只有当两种疾病在核心临床表现、关键检查结果上高度重叠，且在临床实践中经常被误诊时，才认为是混淆关系。否则，即使有部分症状重叠，也不应归为混淆关系。

【患者病历上下文】
{disease_context}

【候选疾病列表】
{', '.join(diseases)}

【知识图谱中发现的疾病间的相关关系】
{relations_text}

### 任务要求（必须严格遵守）
1. 仅当同时满足以下条件时，才判定为混淆关系：
   - 两种疾病的核心症状有多个高度重叠
   - 关键检查结果有明确重叠
   - 有权威文献或指南指出这两种疾病容易混淆
2. 严格排除以下情况（满足任一即不判定）：
   - 两种疾病存在因果或并发症关系（如：高血压→脑梗死）
   - 两种疾病是常见的共病组合（如：冠心病+糖尿病）
   - 仅有一个非特异性症状重叠（如：乏力、头晕）
   - 两种疾病可通过单一关键检查明确区分
   - 一种疾病是另一种疾病的并发症（如：糖尿病→糖尿病肾病）
3. 为每个混淆对提供：
   - 具体重叠的核心症状
   - 重叠的关键检查结果
   - 关键鉴别点（如果有）
4. 无混淆关系时，必须输出：{{"similar_pairs": []}}
   - 禁止输出"无"、"none"等非JSON内容

### 严格输出格式
有混淆对时：
{{
"similar_pairs": [
    {{
    "disease1": "疾病A",
    "disease2": "疾病B",
    "reason": "核心症状重叠:[症状1, 症状2]; 关键检查重叠:[检查1]; 鉴别点:..."
    }}
]

无混淆对时：
{{
"similar_pairs": []
}}
}}

你的输出（必须严格符合上述JSON格式）：
"""

        try:
            print("【相似度专家】输入提示词:", prompt)
            result, _ = self.chat_model.chat_([prompt])
            # print("【相似度专家】原始输出:", result)
            cleaned_result = re.sub(r'```json\s*|```', '', result)
            print("【相似度专家】清理后输出:", cleaned_result)
            # 尝试多种方法提取JSON
            similarity_data = self._robust_json_parse(cleaned_result)
            
            # 验证和标准化结构
            if "similar_pairs" not in similarity_data or not isinstance(similarity_data["similar_pairs"], list):
                similarity_data["similar_pairs"] = []
            
            # 验证和标准化每个相似对
            validated_pairs = []
            for pair in similarity_data["similar_pairs"]:
                validated_pair = {
                    'disease1': str(pair.get('disease1', '')).strip(),
                    'disease2': str(pair.get('disease2', '')).strip(),
                    'reason': str(pair.get('reason', '专家判定存在临床相似性')).strip()
                }
                
                # 确保疾病名称有效且不相同
                if validated_pair['disease1'] and validated_pair['disease2'] and validated_pair['disease1'].lower() != validated_pair['disease2'].lower():
                    validated_pairs.append(validated_pair)
            
            similarity_data["similar_pairs"] = validated_pairs
            print(f"  [相似度专家] 识别到 {len(validated_pairs)} 个相似疾病对")
            return similarity_data
        except Exception as e:
            print(f"  [相似度专家] 分析失败: {e}")
            return {"similar_pairs": []}
        
    def _robust_json_parse(self, text):
        # 尝试1: 使用更好的JSON提取，找到最外层的{}
        try:
            stack = []
            start_idx = -1
            for i, char in enumerate(text):
                if char == '{' and (i == 0 or text[i-1] != '\\'):
                    if start_idx == -1:
                        start_idx = i
                    stack.append(i)
                elif char == '}' and stack:
                    stack.pop()
                    if not stack:  # 匹配完成
                        json_str = text[start_idx:i+1]
                        return json.loads(json_str)
        except Exception as e:
            pass

        # 尝试2: 移除代码块标记后直接解析
        try:
            clean_text = re.sub(r'```(?:json)?\s*', '', text)
            clean_text = re.sub(r'```', '', clean_text)

            start = clean_text.find('{')
            end = clean_text.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = clean_text[start:end]
                return json.loads(json_str)
        except Exception as e:
            pass

        # 尝试3: 直接解析（如果文本本身就是JSON）
        try:
            return json.loads(text.strip())
        except Exception as e:
            pass

        # === 新增：关键词检测强制返回空互斥对 ===
        # 若检测到明确表示“非互斥”或“可共存”的语义，直接返回空列表
        normalized_text = text.lower()
        non_exclusive_keywords = ["可共存", "不互斥", "可以同时", "无互斥", "能共存", "能够共存", "并非互斥", "不是互斥","不属于互斥关系", "可以同时存在"]
        if any(kw in normalized_text for kw in non_exclusive_keywords):
            return {"exclusive_pairs": []}

        # 最后手段: 返回对应空结构
        print(f"⚠️ JSON解析失败，返回空结构。原始文本: {text[:200]}...")
        if "exclusive_pairs" in text.lower():
            return {"exclusive_pairs": []}
        elif "similar_pairs" in text.lower():
            return {"similar_pairs": []}
        else:
            return {}