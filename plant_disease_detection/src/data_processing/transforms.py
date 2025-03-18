#src/data_processing/transforms.py
"""
数据转换工具，提供图像预处理转换函数
"""

from typing import Union, Optional, Tuple, Callable
import torch
from torchvision import transforms
from src.utils.augmentation import AugmentationPipeline

def get_transform(train: bool = False, img_size: Optional[Union[int, Tuple[int, int]]] = None) -> Callable:
    """
    获取图像转换函数
    
    Args:
        train: 是否用于训练模式（包括数据增强）
        img_size: 目标图像大小
        
    Returns:
        转换函数（可直接应用于图像）
    """
    # 创建增强管道
    aug_pipeline = AugmentationPipeline()
    
    # 返回相应的转换
    if train:
        return aug_pipeline.get_transform_by_name('train', img_size)
    else:
        return aug_pipeline.get_transform_by_name('test', img_size)