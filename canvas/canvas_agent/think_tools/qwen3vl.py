# /data1/ytwang/Qwen/qwen2.5-omni-7b 

# 1. 加载qwen omni, bf16. 写成 class: CanvasAgentQwen25Omni.
# 2. 根据 https://huggingface.co/Qwen/Qwen2.5-Omni-7B, 在class中实现方法: response_for_image, 允许输入用户prompt, 和image list. 输出文本.

import os
from typing import List, Union
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


from qwen_omni_utils import process_mm_info


qwen3vl_path = os.path.abspath('/dev/shm/Qwen3-VL-8B-Instruct')


class CanvasAgentQwen3VL:
    def __init__(self, model_path=None, device=None):
        """
        初始化 Qwen3-VL-8B-Instruct 模型
        
        Args:
            model_path: 模型路径，如果为 None 则使用默认路径
        """
        if model_path is None:
            model_path = qwen3vl_path
        model_path = os.path.expanduser(model_path)

        # 加载模型，使用 bf16 精度
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            device_map=device
        )


        # 加载 processor 用于处理图像和文本
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        # self.device = device
        # self.model.to(self.device)
        self.model.eval()
    
    def response_for_text(self, prompt: str, max_new_tokens: int = 512):
        """
        根据用户 prompt 生成文本响应
        """
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]   
        with torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            assistant_response = output_text[0].split("assistant")[-1]
        return assistant_response

    def response_for_image(self, prompt: str, image_list: List[Union[str, Image.Image]], max_new_tokens: int = 512):
        """
        根据用户 prompt 和图像列表生成文本响应
        
        Args:
            prompt: 用户输入的文本提示
            image_list: 图像列表，可以是图像路径字符串或 PIL Image 对象
            
        Returns:
            str: 模型生成的文本响应
        """
        # 将图像路径转换为 PIL Image 对象
        images = []
        for img in image_list:
            if isinstance(img, str):
                print(f"> img: use str type.")
                images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                print(f"> img: use Image.Image type.")
                images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        # 构建消息格式
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image", "image": img})
        
        messages = [
            # {
            #     "role": "system",
            #     "content": [{"type": "text", "text": "你是一个专业的图像绘制和编辑大师, 擅长根据用户需求对图像的细节进行优化. "}]
            # },
            {
                "role": "user",
                "content": content
            }
        ]

        with torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(f"> output_text: {output_text}")

            assistant_response = output_text[0].split("assistant")[-1]
        
        return assistant_response


if __name__ == "__main__":
    # 测试代码
    qwen_omni = CanvasAgentQwen25Omni(model_path=None)
    
    # 示例：使用图像路径
    test_prompt = "请描述这张图片中的内容"
    test_images = ["/root/canvas/canavs_agent/draw_tools/output_image.png", "/root/canvas/canavs_agent/draw_tools/output_image.png"]  # 替换为实际图像路径
    
    response = qwen_omni.response_for_image(test_prompt, test_images)
    print(response)