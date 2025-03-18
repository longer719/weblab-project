# src/utils/image_utils.py

import cv2
import numpy as np
from typing import Tuple, List, Union
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

def detect_plants(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    检测图片中的植物区域
    返回边界框列表：[(x1,y1,x2,y2),...]
    """
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 绿色植物的HSV范围
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取边界框
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 过滤小区域
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x+w, y+h))
            
    return boxes

def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    加载图像文件并返回PIL Image对象
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        PIL Image对象
    """
    # 确保路径是字符串类型
    path_str = str(image_path) if isinstance(image_path, Path) else image_path
    
    # 打开并返回图像
    image = Image.open(path_str).convert('RGB')
    return image

def preprocess_image(image: Union[np.ndarray, Image.Image, str, Path]) -> torch.Tensor:
    """
    预处理图像用于模型推理
    
    Args:
        image: 图像对象或路径
        
    Returns:
        预处理后的图像张量 [1, C, H, W]
    """
    # 根据输入类型处理图像
    if isinstance(image, str) or isinstance(image, Path):
        image = load_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise TypeError(f"不支持的图像类型: {type(image)}")
    
    # 定义预处理变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小到模型输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet均值
            std=[0.229, 0.224, 0.225]    # ImageNet标准差
        )
    ])
    
    # 应用变换
    tensor = transform(image)
    
    # 添加批次维度
    tensor = tensor.unsqueeze(0)
    
    return tensor