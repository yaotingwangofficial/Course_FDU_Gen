import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"

def init_qwen_image_edit_plus(device):
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "/dev/shm/Qwen-Image-Edit-2511", 
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipeline.vae.enable_slicing()
    pipeline.vae._feat_map = None
    pipeline.vae._conv_idx = 0
    # pipeline.enable_vae_tiling()
    pipeline.enable_model_cpu_offload()

    pipeline.set_progress_bar_config(disable=None)

    return pipeline


class CanvasAgentQwenImageEditPlus:
    def __init__(self, device):
        self.pipe = init_qwen_image_edit_plus(device)
        print("pipeline loaded")
        # self.device = device

    def inference(self, images, prompt, save_path):
        inputs = {
            "image": images,
            "prompt": prompt,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": "",
            "num_inference_steps": 40,
        }
        with torch.inference_mode():
            output = self.pipe(**inputs)
            output_image = output.images[0]
            output_image.save(save_path)
            print(f"image saved at {os.path.abspath(save_path)}")
            return output_image
    
    def edit_with_single_image(self, prompt, state_image_path, save_path):
        image = Image.open(state_image_path).convert("RGB")
        print(f"> image loaded from {state_image_path}")
        images = [image]
        return self.inference(images, prompt, save_path)
    
    def edit_with_multiple_images(self, prompt, state_image_paths, save_path):
        images = [Image.open(path).convert("RGB") for path in state_image_paths]
        return self.inference(images, prompt, save_path)
            

if __name__ == "__main__":
    pipeline = CanvasAgentQwenImageEditPlus(device="npu:0")
    print("> pipeline loaded")

    state_image_path = "/root/canvas/canavs_agent/draw_tools/image.png"
    prompt = """
    把图像变为中国古风风格. 
    """

    output_image = pipeline.edit_with_single_image(prompt, state_image_path, "output_image_edit.png")
