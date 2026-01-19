import torch
from langchain_core.language_models import BaseLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from langchain_core.prompt_values import (
    PromptValue,
)
from langchain_core.callbacks import Callbacks
from typing import (
    Any,
)

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import Input

_our_llm = None

def get_client():
    global _our_llm
    if _our_llm is None:
        _our_llm = OurModel()
    return _our_llm

class OurModel:

    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapters: dict = {
        "family":   "vika11222/qwen2.5-0.5b-family_new-agent",
        "investor": "vika11222/qwen2.5-0.5b-investor_new-agent",
        "young":    "vika11222/qwen2.5-0.5b-young-pro_new-agent"
    }

    def __init__(self, model_name = base_model_name, agent_type = 'family'):
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = PeftModel.from_pretrained(base_model, self.adapters[agent_type], adapter_name=agent_type, max_memory={'cpu': 25e10})
        self.current_type = agent_type

    def invoke(self, messages, agent_type = None):
        if agent_type is None:
            agent_type = self.current_type

        assert agent_type in self.adapters.keys()

        if agent_type != self.current_type:
            self.model.set_adapter(agent_type)
            self.current_type = agent_type

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1200,
                temperature=0.7,
                do_sample=True
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        if "assistant" in decoded:
            return decoded.split("assistant")[-1].strip()

        return decoded.strip()
    
    
# class OurModelWarper(BaseLanguageModel):

#     def __init__(self, model):
#         self.model = model

#     def invoke(
#         self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
#     ):
#         print(input)
#         # self.model.infer()
        

#     async def agenerate_prompt(
#         self,
#         prompts: list[PromptValue],
#         stop: list[str] | None = None,
#         callbacks: Callbacks = None,
#         **kwargs: Any,
#     ):
#         pass

#     def generate_prompt(
#         self,
#         prompts: list[PromptValue],
#         stop: list[str] | None = None,
#         callbacks: Callbacks = None,
#         **kwargs: Any,
#     ):
#         pass