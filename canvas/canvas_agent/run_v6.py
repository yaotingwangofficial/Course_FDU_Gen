"""
版本说明:
v.0.6. 重构图像生成逻辑，分为 image_generation 和 image_edit 两大功能.
特点: 
- image_generation: 根据bbox序列生成初始图像，包含judge_and_rebuild机制
- image_edit: 图像编辑功能（待实现）
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import re
import tempfile
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image

from think_tools.qwen3vl import CanvasAgentQwen3VL
from think_tools.qwen3 import CanvasAgentQwen3Text
from draw_tools.zimage import CanvasAgentZImage
from draw_tools.qwen_image import CanvasAgentQwenImage
from draw_tools.qwen_image_edit import CanvasAgentQwenImageEditPlus
from prompt import (
    PROMPT_GENERATE_INITIAL_INSTRUCTION,
    PROMPT_JUDGE_IMAGE_QUALITY,
    PROMPT_GENERATE_EDIT_INSTRUCTION,
    PROMPT_DETECT_OBJECT_BBOX,
    PROMPT_GENERATE_CORRECTION_SUGGESTIONS
)

from helper import build_object_bbox_sequence, format_object_bbox_sequence


class ImageGenerationAgentV6:
    """v6版本：重构后的图像生成代理，分为image_generation和image_edit两大功能"""
    
    def __init__(self, max_retries: int = 3):
        """
        初始化代理
        
        Args:
            max_retries: 每个步骤的最大重试次数
        """
        useImage_draw = 'qwen_image'
        self.qwen_image_edit = CanvasAgentQwenImageEditPlus(device='npu:0')
        self.qwen3vl = CanvasAgentQwen3VL(model_path=None, device='npu:6')
        # self.qwen3 = CanvasAgentQwen3Text(model_path=None, device='npu:7')
        if useImage_draw == 'zimage':
            self.image_draw = CanvasAgentZImage(device=None)
        elif useImage_draw == 'qwen_image':
            self.image_draw = CanvasAgentQwenImage(device='npu:1')
        else:
            raise ValueError(f"Invalid useImage_draw: {useImage_draw}")
        
        self.max_retries = max_retries
        self.max_generation_attempts = 3  # image_generation的最大尝试次数
        self.max_edit_iterations = 3  # image_edit的最大迭代次数
        self.iou_threshold = 0.6  # IOU阈值
    
    def _generate_instruction_from_bbox_sequence(
        self, 
        object_bbox_sequence: List[Dict[str, List[float]]]
    ) -> str:
        """
        根据物体的语义信息和空间关系生成初始instruction
        
        Args:
            object_bbox_sequence: 对象-边界框序列
            
        Returns:
            初始图像生成指令
        """
        formatted_sequence = format_object_bbox_sequence(object_bbox_sequence)
        
        prompt = PROMPT_GENERATE_INITIAL_INSTRUCTION.format(
            formatted_sequence=formatted_sequence
        )
        print(f"[Generate Instruction] Calling qwen3 to generate initial instruction...")
        # response = self.qwen3.generate(prompt)
        response = self.qwen3vl.response_for_text(prompt)  # vl模型可能对空间感知更敏感. 
        # 提取指令文本（去除可能的格式标记）
        instruction = response.strip()
        
        return instruction
    
    def _generate_image_with_instruction(
        self, 
        instruction: str, 
        image_w: int = 512, 
        image_h: int = 512
    ) -> Image.Image:
        """
        根据instruction生成图像
        
        Args:
            instruction: 图像生成指令
            image_w: 图像宽度
            image_h: 图像高度
            
        Returns:
            生成的图像
        """
        print(f"[Generate Image] Generating image with instruction...")
        return self.image_draw.draw(image_w, image_h, prompt=instruction)
    
    def judge_and_rebuild(
        self,
        instruction: str,
        generated_image: Image.Image,
        bbox_sequence: List[Dict[str, List[float]]]
    ) -> Tuple[bool, Optional[str]]:
        """
        判断生成的图像是否满足要求，如果不满足则生成新的instruction
        
        Args:
            instruction: 当前使用的instruction
            generated_image: 生成的图像
            bbox_sequence: 目标对象-边界框序列
            
        Returns:
            (is_satisfied, new_instruction)
            - is_satisfied: True表示图像满足要求，False表示需要重新生成
            - new_instruction: 如果is_satisfied为False，返回新的instruction；否则为None
        """
        formatted_sequence = format_object_bbox_sequence(bbox_sequence)
        
        # 使用qwen3vl（实际是qwen_omni）评估图像
        prompt = PROMPT_JUDGE_IMAGE_QUALITY.format(
            formatted_sequence=formatted_sequence,
            previous_instruction=instruction
        )
        
        print(f"[Judge]: Evaluating generated image with qwen_omni...")
        judge_response = self.qwen3vl.response_for_image(prompt, [generated_image], max_new_tokens=1024)
        response_text = judge_response.strip()

        print(f"[Judge]: {response_text}")
        # input("Press Enter to continue...")
        
        # 解析响应
        if "[CONVERGED]" in response_text.upper() or "meets all requirements" in response_text.lower():
            print(f"[Judge]: Image meets all requirements!")
            return True, None
        elif "[NEEDS_CORRECTION]" in response_text.upper() or "correction needed" in response_text.lower():
            # 需要重新生成，提取出新的instruction
            print(f"[Judge]: Image needs correction, extracting new instruction...")
            new_instruction = response_text.split('[NEW_INSTRUCTION]', 1)[1].strip()

            print(f"[Judge]: New instruction: {new_instruction}")
            input("Press Enter to continue...")
            return False, new_instruction
        else:
            # 如果无法明确判断，默认认为需要重新生成
            print(f"[Judge] Warning: Unable to parse response clearly, assuming needs regeneration. Response: {response_text[:200]}")
            new_instruction = self._generate_instruction_from_bbox_sequence(bbox_sequence)
            return False, new_instruction
    
    def image_generation(
        self,
        bbox_sequence: List[Dict[str, List[float]]],
        image_w: int = 1024,
        image_h: int = 1024
    ) -> Tuple[bool, Optional[Image.Image], Optional[str]]:
        """
        根据bbox序列生成图像
        
        包含以下步骤：
        1. 根据物体的语义信息和空间关系生成初始instruction
        2. 根据instruction生成初始图像
        3. 使用judge_and_rebuild判断图像是否满足要求
        4. 如果不满足，生成新的instruction并重新生成图像（循环）
        
        Args:
            bbox_sequence: 对象-边界框序列
            image_w: 图像宽度
            image_h: 图像高度
            
        Returns:
            (success, final_image, final_instruction)
            - success: True表示成功生成满足要求的图像
            - final_image: 最终生成的图像
            - final_instruction: 最终使用的instruction
        """
        print(f"\n{'='*80}")
        print(f"[Image Generation] Starting image generation with bbox_sequence")
        print(f"{'='*80}")
        
        attempt = 0
        current_instruction = None
        current_image = None
        
        while attempt < self.max_generation_attempts:
            attempt += 1
            print(f"\n[Image Generation] Attempt {attempt}/{self.max_generation_attempts}")
            
            # 生成或更新instruction
            if current_instruction is None:
                # 第一次尝试，生成初始instruction
                current_instruction = self._generate_instruction_from_bbox_sequence(bbox_sequence)
            
            # 生成图像
            current_image = self._generate_image_with_instruction(
                current_instruction,
                image_w,
                image_h
            )

            # 后续尝试，使用judge_and_rebuild返回的新instruction
            is_satisfied, new_instruction = self.judge_and_rebuild(
                current_instruction,
                current_image,
                bbox_sequence
            )
            if is_satisfied:
                break   
            else:
                current_instruction = new_instruction  # 重新开始生成图像
        
        print(f"\n[Image Generation] ✓ Successfully generated image at attempt {attempt}")
        return True, current_image, current_instruction
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        计算两个bbox的IOU (Intersection over Union)
        
        Args:
            bbox1: [x, y, w, h] 格式，normalized to [0, 1]
            bbox2: [x, y, w, h] 格式，normalized to [0, 1]
            
        Returns:
            IOU值，范围[0, 1]
        """
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 转换为 [x_min, y_min, x_max, y_max] 格式
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _detect_object_bboxes(
        self,
        image: Image.Image,
        object_names: List[str]
    ) -> Dict[str, Optional[List[float]]]:
        """
        使用qwen3vl检测图像中物体的bbox
        
        Args:
            image: 输入图像
            object_names: 要检测的对象名称列表
            
        Returns:
            Dict[object_name, bbox] 或 None，bbox格式为[x, y, w, h]，normalized to [0, 1]
        """
        object_list = ", ".join(object_names)
        prompt = PROMPT_DETECT_OBJECT_BBOX.format(object_list=object_list)
        
        print(f"[Detect BBox] Detecting bounding boxes for objects: {object_names}")
        response = self.qwen3vl.response_for_image(prompt, [image], max_new_tokens=512)
        print(f"[Detect BBox] Response: {response}")
        
        # 解析响应，提取bbox信息
        detected_bboxes = {}
        
        # 尝试解析JSON格式的响应
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                for obj_name in object_names:
                    if obj_name in parsed:
                        bbox = parsed[obj_name]
                        if bbox is not None and isinstance(bbox, list) and len(bbox) == 4:
                            detected_bboxes[obj_name] = bbox
                        else:
                            detected_bboxes[obj_name] = None
                    else:
                        detected_bboxes[obj_name] = None
            else:
                # 如果无法解析JSON，尝试手动解析
                for obj_name in object_names:
                    pattern = rf'"{re.escape(obj_name)}"\s*:\s*\[([^\]]+)\]'
                    match = re.search(pattern, response)
                    if match:
                        coords = [float(x.strip()) for x in match.group(1).split(',')]
                        if len(coords) == 4:
                            detected_bboxes[obj_name] = coords
                        else:
                            detected_bboxes[obj_name] = None
                    else:
                        detected_bboxes[obj_name] = None
        except Exception as e:
            print(f"[Detect BBox] Error parsing response: {e}")
            # 如果解析失败，返回None
            for obj_name in object_names:
                detected_bboxes[obj_name] = None
        
        return detected_bboxes
    
    def _generate_correction_suggestions(
        self,
        detected_bboxes: Dict[str, Optional[List[float]]],
        target_bboxes: Dict[str, List[float]]
    ) -> Dict[str, str]:
        """
        根据检测到的bbox和目标bbox生成修正建议
        
        Args:
            detected_bboxes: 检测到的bbox字典
            target_bboxes: 目标bbox字典
            
        Returns:
            Dict[object_name, correction_suggestion]
        """
        # 计算IOU并找出需要修正的对象
        needs_correction = {}
        for obj_name in target_bboxes.keys():
            if obj_name not in detected_bboxes or detected_bboxes[obj_name] is None:
                needs_correction[obj_name] = "object not found in image"
            else:
                iou = self._calculate_iou(detected_bboxes[obj_name], target_bboxes[obj_name])
                print(f"[Correction] IOU for {obj_name}: {iou:.3f}")
                if iou < self.iou_threshold:
                    needs_correction[obj_name] = f"IOU too low ({iou:.3f} < {self.iou_threshold})"
        
        if not needs_correction:
            return {}
        
        # 使用qwen3vl生成修正建议
        detected_str = json.dumps(detected_bboxes, indent=2)
        target_str = json.dumps(target_bboxes, indent=2)
        
        prompt = PROMPT_GENERATE_CORRECTION_SUGGESTIONS.format(
            detected_bboxes=detected_str,
            target_bboxes=target_str
        )
        
        print(f"[Correction] Generating correction suggestions...")
        response = self.qwen3vl.response_for_text(prompt, max_new_tokens=512)
        print(f"[Correction] Response: {response}")
        
        # 解析响应
        suggestions = {}
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                suggestions = parsed
        except Exception as e:
            print(f"[Correction] Error parsing suggestions: {e}")
            # 如果解析失败，使用简单的建议
            for obj_name in needs_correction.keys():
                suggestions[obj_name] = "needs position adjustment"
        
        return suggestions
    
    def _generate_edit_instruction_from_suggestions(
        self,
        suggestions: Dict[str, str]
    ) -> str:
        """
        根据修正建议生成edit_instruction
        
        Args:
            suggestions: 修正建议字典
            
        Returns:
            edit_instruction字符串
        """
        if not suggestions:
            return ""
        
        # 构建edit_instruction
        edit_parts = []
        for obj_name, suggestion in suggestions.items():
            edit_parts.append(f"{obj_name}: {suggestion}")
        
        edit_instruction = "Please adjust the positions of the following objects: " + "; ".join(edit_parts)
        return edit_instruction
    
    def image_edit(
        self,
        current_image: Image.Image,
        edit_instruction: str,
        bbox_sequence: List[Dict[str, List[float]]]
    ) -> Tuple[bool, Optional[Image.Image]]:
        """
        图像编辑功能
        
        流程：
        1. 使用qwen3vl检测图像中物体的bbox（normalized to [0, 1]）
        2. 对比检测到的bbox和目标bbox，计算IOU
        3. 如果IOU < 0.6，生成修正建议
        4. 根据修正建议生成edit_instruction
        5. 使用qwen_image_edit编辑图像
        6. 迭代直到所有物体的IOU >= 0.6或达到最大迭代次数
        
        Args:
            current_image: 当前图像
            edit_instruction: 编辑指令（初始指令，可能不会被使用）
            bbox_sequence: 目标对象-边界框序列
            
        Returns:
            (success, edited_image)
            - success: True表示编辑成功（所有物体IOU >= 0.6）
            - edited_image: 编辑后的图像
        """
        print(f"\n{'='*80}")
        print(f"[Image Edit] Starting image editing")
        print(f"{'='*80}")
        
        # 提取对象名称和目标bbox
        object_names = []
        target_bboxes = {}
        for item in bbox_sequence:
            for obj_name, bbox in item.items():
                object_names.append(obj_name)
                target_bboxes[obj_name] = bbox
        
        # 将bbox转换为normalized格式（假设输入是[x, y, w, h]格式，需要normalize）
        # 这里假设输入的bbox已经是normalized格式，如果不是，需要根据图像尺寸进行normalize
        image_w, image_h = current_image.size
        normalized_target_bboxes = {}
        for obj_name, bbox in target_bboxes.items():
            # 假设输入bbox是绝对坐标，需要normalize
            # 如果已经是normalized，则直接使用
            x, y, w, h = bbox
            # 检查是否需要normalize（如果值都小于等于1，可能已经是normalized）
            if x <= 1.0 and y <= 1.0 and w <= 1.0 and h <= 1.0:
                normalized_target_bboxes[obj_name] = bbox
            else:
                # 需要normalize
                normalized_target_bboxes[obj_name] = [
                    x / image_w, y / image_h, w / image_w, h / image_h
                ]
        
        edited_image = current_image
        iteration = 0
        
        while iteration < self.max_edit_iterations:
            iteration += 1
            print(f"\n[Image Edit] Iteration {iteration}/{self.max_edit_iterations}")
            
            # 1. 检测图像中物体的bbox
            detected_bboxes = self._detect_object_bboxes(edited_image, object_names)
            print(f"[Image Edit] Detected bboxes: {detected_bboxes}")
            
            # 2. 计算IOU并检查是否所有物体都满足要求
            all_satisfied = True
            iou_results = {}
            for obj_name in object_names:
                if obj_name not in detected_bboxes or detected_bboxes[obj_name] is None:
                    all_satisfied = False
                    iou_results[obj_name] = 0.0
                    print(f"[Image Edit] {obj_name}: object not found (IOU = 0.0)")
                else:
                    iou = self._calculate_iou(detected_bboxes[obj_name], normalized_target_bboxes[obj_name])
                    iou_results[obj_name] = iou
                    if iou < self.iou_threshold:
                        all_satisfied = False
                        print(f"[Image Edit] {obj_name}: IOU = {iou:.3f} < {self.iou_threshold}")
                    else:
                        print(f"[Image Edit] {obj_name}: IOU = {iou:.3f} >= {self.iou_threshold} ✓")
            
            if all_satisfied:
                print(f"\n[Image Edit] ✓ All objects satisfy IOU >= {self.iou_threshold}")
                return True, edited_image
            
            # 3. 生成修正建议
            suggestions = self._generate_correction_suggestions(detected_bboxes, normalized_target_bboxes)
            if not suggestions:
                print(f"[Image Edit] No correction suggestions generated, stopping")
                break
            
            print(f"[Image Edit] Correction suggestions: {suggestions}")
            
            # 4. 生成edit_instruction
            edit_instruction = self._generate_edit_instruction_from_suggestions(suggestions)
            print(f"[Image Edit] Generated edit instruction: {edit_instruction}")
            
            # 5. 使用qwen_image_edit编辑图像
            # 需要保存临时图像文件
            tmp_path = None
            output_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    edited_image.save(tmp_path)
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_file:
                    output_path = output_file.name
                
                print(f"[Image Edit] Editing image with qwen_image_edit...")
                edited_image = self.qwen_image_edit.edit_with_single_image(
                    edit_instruction,
                    tmp_path,
                    output_path
                )
                print(f"[Image Edit] Image edited successfully")
            except Exception as e:
                print(f"[Image Edit] Error during image editing: {e}")
                return False, edited_image
            finally:
                # 清理临时文件
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                if output_path and os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except:
                        pass
        
        print(f"\n[Image Edit] ✗ Failed to satisfy all IOU requirements after {self.max_edit_iterations} iterations")
        return True, edited_image
        # return False, edited_image
    
    def process_image_generation(
        self,
        image_id: str,
        image_path: str,
        objects: List[Dict[str, Any]],
        save_dir: str = "demo_v6"
    ) -> Tuple[bool, Optional[Image.Image], Optional[str]]:
        """
        处理图像生成的完整流程
        
        Args:
            image_id: 图像ID
            image_path: 图像路径
            objects: 对象列表
            save_dir: 保存目录
            
        Returns:
            (success, final_image, final_instruction)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取图像尺寸
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image_w, image_h = image.size
            image_w = image_w // 16 * 16
            image_h = image_h // 16 * 16
        else:
            raise ValueError(f"Image not found: {image_path}")
        
        # 构建对象-边界框序列
        object_names = [obj['name'] for obj in objects]
        object_boxes = [obj['bbox'] for obj in objects]
        object_bbox_sequence = build_object_bbox_sequence(object_names, object_boxes)
        
        # 调用image_generation
        success, final_image, final_instruction = self.image_generation(
            object_bbox_sequence,
            image_w,
            image_h
        )

        # if success:
        #     # 调用image_edit
        #     success, final_image = self.image_edit(
        #         final_image,
        #         final_instruction,
        #         object_bbox_sequence
        #     )
        
        if success and final_image is not None:
            # 保存最终结果
            final_path = os.path.join(save_dir, "final_image.png")
            final_image.save(final_path)
            print(f"[Process] Saved final image to {final_path}")
        
        return success, final_image, final_instruction


def main():
    """主函数：遍历数据并处理"""
    # 数据路径
    json_path = "/dev/shm/layout/vg_rebuild/vg_test.json"
    images_dir = "/dev/shm/layout/vg_rebuild/images"
    
    # 创建代理
    agent = ImageGenerationAgentV6(max_retries=3)
    
    # 读取JSON文件
    print(f"Loading data from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # 遍历每个数据项
    for idx, item in enumerate(data):
        image_id = item['id']
        image_path = item['image_path']
        objects = item['objects']
        full_image_path = os.path.join(images_dir, image_path)
        
        if objects is None or len(objects) == 0:
            print(f"Warning: objects is None or empty, image_id={image_id}")
            continue
        
        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found: {full_image_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(data)}] Processing image_id={image_id}")
        print(f"{'='*80}")
        
        # 为每个样本创建独立的保存目录
        save_dir = os.path.join("demo_v6", f"sample_{image_id}")
        
        # 执行处理
        success, final_image, final_instruction = agent.process_image_generation(
            image_id,
            full_image_path,
            objects,
            save_dir
        )
        
        if success:
            print(f"✓ Sample {image_id} completed successfully")
        else:
            print(f"✗ Sample {image_id} failed")


if __name__ == "__main__":
    main()

