import os


from PIL import Image
import torch

from diffusers import DiffusionPipeline

def init_qwen_image(device=None):
    pipeline = DiffusionPipeline.from_pretrained(
        "/root/Qwen/qwen_image", 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="balanced",
    )
    # pipeline = pipeline.to(device)
    # pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=None)
    
    return pipeline


class CanvasAgentQwenImage:
    def __init__(self, device):
        self.pipe = init_qwen_image(device)
        self.device = device

    def draw(
            self, image_w, image_h, 
            prompt, negative_prompt="",
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.manual_seed(0),
    ):
        width, height = image_w, image_h

        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
            ).images[0]

        return image



if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    pipeline = init_qwen_image()


    # image = Image.open("/home/ytwang/canvas/canvas_agent/demo1/stage_canvas.png").convert("RGB")
    prompt = """

    make a beautiful landscape photo.

    """

    inputs = {
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": "",
        "num_inference_steps": 50,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save("output_image.png")
        print("image saved at", os.path.abspath("output_image.png"))

