from typing import List, Dict

def build_object_bbox_sequence(objects: List[str], bboxes: List[List[float]]) -> List[Dict[str, List[float]]]:
        """
        构建对象-边界框序列
        
        Args:
            objects: 对象名称列表
            bboxes: 边界框列表，格式为 [[x, y, w, h], ...]
            
        Returns:
            对象-边界框序列，格式为 [{obj₁: bbox₁}, {obj₂: bbox₂}, ..., {objₙ: bboxₙ}]
        """
        if len(objects) != len(bboxes):
            raise ValueError(f"对象数量 ({len(objects)}) 与边界框数量 ({len(bboxes)}) 不匹配")
        
        sequence = []
        for obj, bbox in zip(objects, bboxes):
            sequence.append({obj: bbox})
        
        return sequence
    
def format_object_bbox_sequence(sequence: List[Dict[str, List[float]]]) -> str:
    """
    将对象-边界框序列格式化为字符串，用于输入给模型
    
    Args:
        sequence: 对象-边界框序列
        
    Returns:
        格式化后的字符串
    """
    formatted_items = []
    for item in sequence:
        for obj, bbox in item.items():
            x, y, w, h = bbox
            formatted_items.append(f"{{'{obj}': [{x}, {y}, {w}, {h}]}}")
    
    return "[" + ", ".join(formatted_items) + "]"
