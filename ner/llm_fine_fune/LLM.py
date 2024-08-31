from langchain.llms.base import LLM
from typing import Any, List, Optional,Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizerFast
import torch
from peft import LoraConfig, TaskType, get_peft_model


class ChatGLM4_LLM(LLM):
    # 基于本地 ChatGLM4 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    gen_kwargs: dict = None
        
    def __init__(self, mode_name_or_path: str, gen_kwargs: dict = None):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_name_or_path, use_fast=False, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        self.model = PeftModel.from_pretrained(self.model, model_id="llm_微调/output/GLM4_lora")
        print("完成本地模型的加载")
        
        if gen_kwargs is None:
            gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        self.gen_kwargs = gen_kwargs
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        model_inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).to('cuda')
        generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于识别LLM的字典,这对于缓存和跟踪目的至关重要。"""
        return {
            "model_name": "glm-4-9b-chat",
            "max_length": self.gen_kwargs.get("max_length"),
            "do_sample": self.gen_kwargs.get("do_sample"),
            "top_k": self.gen_kwargs.get("top_k"),
        }

    @property
    def _llm_type(self) -> str:
        return "glm-4-9b-chat"



class Qwen2_LLM(LLM):
    # 基于本地 Qwen2 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
        
    def __init__(self, mode_name_or_path :str):

        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()

        # 加载lora权重
        self.model = PeftModel.from_pretrained(self.model, model_id="output/Qwen2_7B_lora")
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
        print("完成本地模型的加载")
        
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):

        messages = [{"role": "user", "content": prompt }]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(model_inputs.input_ids,max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    @property
    def _llm_type(self) -> str:
        return "Qwen2_LLM"


class LLaMA3_LLM(LLM):
    # 基于本地 llama3 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
        
    def __init__(self, mode_name_or_path :str):

        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
        # 加载lora权重
        self.model = PeftModel.from_pretrained(self.model, model_id="output/LLama3_8b_lora")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("完成本地模型的加载")
    
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):

        input_str = prompt
        input_ids = self.tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(
            input_ids=input_ids, max_new_tokens=512, do_sample=True,
            top_p=0.9, temperature=0.5, repetition_penalty=1.1, eos_token_id=self.tokenizer.encode('<|eot_id|>')[0]
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = self.tokenizer.decode(outputs).strip().replace('<|eot_id|>', "").replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '').strip()
        return response
        
    @property
    def _llm_type(self) -> str:
        return "LLaMA3_LLM"