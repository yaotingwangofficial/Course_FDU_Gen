# 通过transformers库加载qwen3-8b模型.
# 要求: bf16混合精度. 

import os
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

qwen_8b_path = os.path.abspath('/root/Qwen/Qwen3-8B')

class CanvasAgentQwen3Text:
    def __init__(self, model_path, device=None):
        if model_path is None:
            model_path = qwen_8b_path
            model_path = os.path.expanduser(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        print(f"Qwen3 model loaded successfully on device: {self.device}")

    def generate(self, prompt):
        print(f"Calling qwen3.generate()...")
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # print(f"Tokenizing text: {text}")
        # input("Press Enter to continue...")
        with torch.inference_mode():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(response)
        response = response.split('</think>')[-1].strip()
        return response
    
    def parse_object_names(self, response):
        # response is a string, e.g., "['chair back', 'furnature table', 'chair leg']"
        # we need to parse it into a list of object names
        object_names = response.replace("'", "").strip("[").strip("]").split(", ")

        return object_names

if __name__ == "__main__":
    qwen3_text_thinker = CanvasAgentQwen3Text(model_path=None)
    response = qwen3_text_thinker.generate("What is the capital of France?")
    print(response)
    input("Press Enter to continue...")