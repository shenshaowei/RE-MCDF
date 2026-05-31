from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import modelscope
import torch
import asyncio
from http import HTTPStatus
import platform
import time
import requests 
from dashscope import Generation
from dashscope.aigc.generation import AioGeneration
from config import *
class ChatModel:
    def __init__(self, model_type, model_name_or_path, model_version, api_token=None):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.model_version = model_version
        self.model_version_api = config_dict.get("model_version_api", None)
        self.load_model(model_type, model_name_or_path, model_version)
        self.api_token = config_dict["api_key"]  # 存储 API token

    def load_model(self, model_type, model_name_or_path, model_version):
        """ 加载模型 """
        if model_type == "glm":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.chat_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.chat_model = self.chat_model.cuda()
            self.chat_model.eval()
        elif model_type == "baichuan":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
            self.chat_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map = "auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
            self.chat_model.generation_config.do_sample = False
            self.chat_model = self.chat_model.cuda().eval()
        elif model_type == "qwen":
            if model_version == "qwen-7b-chat":
                # tokenizer = modelscope.AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", trust_remote_code=True)
                # chat_model = modelscope.AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto",    torch_dtype=torch.bfloat16,    trust_remote_code=True).eval()
                tokenizer = modelscope.AutoTokenizer.from_pretrained(
                    model_name_or_path,  # ← 使用参数
                    trust_remote_code=True
                )
                chat_model = modelscope.AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,  # ← 使用参数
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).eval()
                chat_model.generation_config.do_sample = False
                self.chat_model = chat_model
                self.tokenizer = tokenizer

            elif model_version == "Qwen2.5-7B-Instruct":
                # 使用 vLLM 加载 Qwen2.5-7B-Instruct
                try:
                    from vllm import LLM, SamplingParams
                    self.vllm_llm = LLM(
                        model=model_name_or_path,
                        trust_remote_code=True,
                        tensor_parallel_size=torch.cuda.device_count(),  # 根据GPU数量调整
                        gpu_memory_utilization=0.92,
                        max_model_len=2200,  # 根据需求调整
                        # max_model_len=3000,
                        dtype="bfloat16"
                    )
                    # self.vllm_sampling_params = SamplingParams(
                    #     temperature=0.7,
                    #     top_p=0.8,
                    #     top_k=20,
                    #     max_tokens=1024,
                    #     stop_token_ids=[151643]  # Qwen 的 <|im_end|>
                    # )
                    self.vllm_sampling_params = SamplingParams(
                        temperature=0.0,
                        top_p=1,
                        top_k=-1,
                        max_tokens=1024,
                        stop_token_ids=[151643]  # Qwen 的 <|im_end|>
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True
                    )
                    self.use_vllm = True
                    print(f"成功使用 vLLM 加载 {model_version}")
                    
                except ImportError:
                    print("vLLM 未安装，回退到标准 Hugging Face 加载")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=True
                    )
                    chat_model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                    ).eval()
                    chat_model.generation_config.do_sample = False
                    self.chat_model = chat_model
                    self.use_vllm = False

            elif model_version == "Qwen2.5-7B-Instruct-hf" or "Qwen2.5-3B-Instruct-hf":
                # tokenizer = modelscope.AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", trust_remote_code=True)
                # chat_model = modelscope.AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto",    torch_dtype=torch.bfloat16,    trust_remote_code=True).eval()
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True
                )
                chat_model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,  # 更通用；若支持 bfloat16 可换回
                    trust_remote_code=True
                ).eval()
                chat_model.generation_config.do_sample = False
                chat_model.generation_config.max_length = 2500  # 总长度限制

                self.chat_model = chat_model
                self.tokenizer = tokenizer
                self.use_vllm = False

            elif model_version == "qwen-14b-chat":
                tokenizer = modelscope.AutoTokenizer.from_pretrained("qwen/Qwen-14B-Chat-Int4", trust_remote_code=True)
                chat_model = modelscope.AutoModelForCausalLM.from_pretrained("qwen/Qwen-14B-Chat-Int4", device_map="auto",  trust_remote_code=True).eval()
                chat_model.generation_config.do_sample = False
                self.chat_model = chat_model
                self.tokenizer = tokenizer
        elif model_type == "qwen_api":
            self.seed = 1234

        elif model_type == "qwen2.5_api":
            # 硅基流动 API 不需要本地加载模型，只需要保存配置
            print(f"初始化硅基流动 API 模型: {model_version}")  
    def chat_(self, messages):
        """ 自定义chat接口 """
        if self.model_type == "glm":
            return self.chat_glm(messages)
        elif self.model_type == "baichuan":
            return self.chat_baichuan(messages)
        elif self.model_type == "qwen":
            if self.use_vllm:
                return self.chat_qwen_vllm(messages)
            else:
                return self.chat_qwen_hf(messages)
        elif self.model_type == "qwen_api":
            api_result = self.chat_qwenapi(messages)
            if api_result[0] == "error, no correct response":
                # 再次尝试调用api
                time.sleep(1)
                retry_result = self.chat_qwenapi(messages)
                # 如果再次调用api还是失败，则抛出异常
                if retry_result[0] == "error, no correct response":
                    raise ValueError("api调用失败")
                else:
                    return retry_result
            else:
                return api_result
        elif self.model_type == "qwen2.5_api":
            api_result = self.chat_qwen2_5_api(messages)
            if api_result[0] == "error, no correct response":
                # 再次尝试调用api
                time.sleep(1)
                retry_result = self.chat_qwen2_5_api(messages)
                # 如果再次调用api还是失败，则抛出异常
                if retry_result[0] == "error, no correct response":
                    raise ValueError("硅基流动 API 调用失败")
                else:
                    return retry_result
            else:
                return api_result
        else:
            raise ValueError("model_type must be in ['glm', 'baichuan', 'qwen']")
        
    def chat_glm(self, messages):
        """ glm调用chat，qwen的调用方式和glm相同 """
        # if len(prompt) > 2048:
        #     prompt = prompt[:2048]
        if len(messages) == 1:
            prompt = messages[0]
            old_history = None
        else:
            prompt = messages[0]
            old_history = messages[1]

        if old_history is not None:
            response, history = self.chat_model.chat(self.tokenizer,
                                           prompt,
                                           do_sample=False,
                                           temperature=1.0,
                                           top_p = 1.0,
                                           repetition_penalty = 1.1,
                                           history = old_history
                                           )
        else:
            response, history = self.chat_model.chat(self.tokenizer,
                                           prompt,
                                           do_sample=False,
                                           temperature=1.0,
                                           top_p = 1.0,
                                           repetition_penalty = 1.1,
                                           history = old_history
                                           )
        return response, history
    
    def chat_baichuan(self, texts):
        """ baichuan调用chat """
        if len(texts) == 1:
            prompt = texts[0]
            messages = []
            messages.append({"role": "user", "content": prompt})
        else:
            prompt = texts[0]
            # messages = []
            messages = texts[1]
            messages.append({"role": "user", "content": prompt})
        response = self.chat_model.chat(self.tokenizer, messages)
        messages.append({"role": "assistant", "content": response})

        return response, messages
    
    def chat_qwen(self, messages):
        """ Qwen 推理，支持 vLLM 和标准 Hugging Face """
        # 检查是否使用 vLLM
        if hasattr(self, 'use_vllm') and self.use_vllm and self.model_version == "Qwen2.5-7B-Instruct":
            return self.chat_qwen_vllm(messages)
        else:
            return self.chat_qwen_hf(messages)
        
    def chat_qwen_vllm(self, messages):
        """ 使用 vLLM 进行 Qwen2.5 推理 """
        # 处理多轮对话历史
        if len(messages) > 1:
            prompt = messages[0]
            history = messages[1]  # 历史对话 [ {role, content}, ... ]
        else:
            prompt = messages[0]
            history = []
        
        # 构建完整对话
        conversation = []
        if history:
            conversation.extend(history)
        conversation.append({"role": "user", "content": prompt})
        
        # 应用 Qwen2 的 chat template
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 使用 vLLM 生成
        outputs = self.vllm_llm.generate([text], self.vllm_sampling_params)
        
        # 提取响应
        response = outputs[0].outputs[0].text.strip()
        
        # 构建新历史
        new_history = conversation + [{"role": "assistant", "content": response}]
        return response, new_history
    
    def chat_qwen_hf(self, messages):
        """ Qwen2.5 专用推理 (标准 Hugging Face 方式) """
        # 处理多轮对话历史
        if len(messages) > 1:
            prompt = messages[0]
            history = messages[1]  # 历史对话 [ {role, content}, ... ]
        else:
            prompt = messages[0]
            history = []
        
        # 构建完整对话
        conversation = []
        conversation.append({"role": "system", "content": "You are an experienced medical expert."})
        if history:
            conversation.extend(history)
        conversation.append({"role": "user", "content": prompt})
        
        # 应用 Qwen2 的 chat template
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt").to(self.chat_model.device)
        
        # 生成响应
        outputs = self.chat_model.generate(
            **inputs,
            # max_new_tokens=1024,
            max_new_tokens=1024,
            do_sample=False
            # do_sample=True,
            # temperature=0.7,
            # top_p=0.8,
            # top_k=20,
        )
        
        # 解码输出（去掉输入部分）
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # 构建新历史
        new_history = conversation + [{"role": "assistant", "content": response}]
        return response, new_history
    
    def chat_qwenapi(self, messages):
        """ qwenapi调用chat """

        if len(messages) == 1:
            prompt = messages[0]
            old_history = None
        else:
            prompt = messages[0]
            old_history = messages[1]

        messages = [{'role': 'system', 'content': 'You are an experienced medical expert.'},
                {'role': 'user', 'content': prompt}]
        # messages = [{'role': 'user', 'content': prompt}]
        
        # qwen1.5-72b-chat-api, 把后面的-api去掉，才是模型的名字
        response = Generation.call(model=self.model_version[:-4],
                                   messages=messages,
                                   seed=self.seed,
                                   result_format='message')
        
        if response.status_code == HTTPStatus.OK:
            return response['output']['choices'][0]['message']['content'], response['output']['choices'][0]['message']
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            return "error, no correct response", "error, no history"

    def chat_qwenapi(self, messages):
        """ qwenapi调用chat """

        if len(messages) == 1:
            prompt = messages[0]
            old_history = None
        else:
            prompt = messages[0]
            old_history = messages[1]

        messages = [{'role': 'system', 'content': 'You are an experienced medical expert.'},
                {'role': 'user', 'content': prompt}]
        # messages = [{'role': 'user', 'content': prompt}]
        
        # qwen1.5-72b-chat-api, 把后面的-api去掉，才是模型的名字
        response = Generation.call(model=self.model_version[:-4],
                                   messages=messages,
                                   seed=self.seed,
                                   result_format='message')
        
        if response.status_code == HTTPStatus.OK:
            return response['output']['choices'][0]['message']['content'], response['output']['choices'][0]['message']
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            return "error, no correct response", "error, no history"

        # return response
    def chat_qwen2_5_api(self, messages,max_retries=5):
        """ 硅基流动 API 调用chat """
        
        if len(messages) == 1:
            prompt = messages[0]
            old_history = None
        else:
            prompt = messages[0]
            old_history = messages[1]

        # 构建请求消息
        if old_history is not None:
            # 如果有历史记录，使用完整的历史对话
            api_messages = old_history
            api_messages.append({"role": "user", "content": prompt})
        else:
            # 如果没有历史记录，创建新的对话
            api_messages = [
                {"role": "system", "content": "You are an experienced medical expert."},
                {"role": "user", "content": prompt}
            ]

        url = "https://api.siliconflow.cn/v1/chat/completions"

        payload = {
            "model": f"Qwen/{self.model_version_api}",  # 使用传入的模型版本
            "messages": api_messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048 if config_dict["dataset"] == "STROKE" else 2576,
            "stop": []
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                response_data = response.json()
                
                if response.status_code == 200:
                    content = response_data['choices'][0]['message']['content']
                    # 更新历史记录
                    if old_history is not None:
                        new_history = old_history.copy()
                        new_history.append({"role": "user", "content": prompt})
                        new_history.append({"role": "assistant", "content": content})
                    else:
                        new_history = [
                            {"role": "system", "content": "You are an experienced medical expert."},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": content}
                        ]
                    return content, new_history
                else:
                    print(f"硅基流动 API 错误: 状态码 {response.status_code}, 响应: {response_data}")
                    print(f"API 错误 {response.status_code}，第 {attempt+1} 次重试...")
                    if attempt < max_retries - 1:  # 不是最后一次尝试
                        time.sleep(2 ** attempt)  # 指数退避
                    else:
                        return "error, no correct response", "error, no history"
            except requests.exceptions.RequestException as e:
                print(f"请求异常: {str(e)}，第 {attempt+1} 次重试...")
                if attempt < max_retries - 1:  # 不是最后一次尝试
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return "error, no correct response", "error, no history"
            except Exception as e:
                print(f"处理异常: {str(e)}，第 {attempt+1} 次重试...")
                if attempt < max_retries - 1:  # 不是最后一次尝试
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return "error, no correct response", "error, no history"
        # 如果超过最大重试次数，返回错误
        return "error, no correct response", "error, no history"