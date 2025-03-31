# src/data_processing/dataset.py

import os
import torch
import pandas as pd
import numpy as np
import logging
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional, Union, Any
import json
from pathlib import Path

from src.utils.config_manager import ConfigManager
from src.utils.augmentation import AugmentationPipeline


class BaseDataset(Dataset):
    """基础数据集类，提供共享的实现"""
    
    def __init__(self, root_dir: str, mode: str = 'train', transform_name: str = None):
        self.root_dir = Path(root_dir)
        self.mode = mode
        
        # 配置管理器
        self.config_manager = ConfigManager()
        
        # 增强管道
        self.aug_pipeline = AugmentationPipeline()
        
        # 设置转换
        if transform_name is None:
            transform_name = mode
        self.transform = self.aug_pipeline.get_transform_by_name(transform_name)
        
        # 加载数据标签
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """加载数据标签(子类必须实现)"""
        raise NotImplementedError("子类必须实现 _load_data 方法")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个数据项(子类必须实现)"""
        raise NotImplementedError("子类必须实现 __getitem__ 方法")


class ClassificationDataset(BaseDataset):
    """植物分类数据集"""
    
    def __init__(self, root_dir: str, mode: str = 'train', transform_name: str = None):
        super().__init__(root_dir, mode, transform_name)
    
    def _load_data(self) -> List[Dict]:
        """加载分类数据标签"""
        labels_file = self.root_dir / f"{self.mode}_labels.csv"
        
        if not labels_file.exists():
            raise FileNotFoundError(f"找不到标签文件: {labels_file}")
            
        df = pd.read_csv(labels_file)
        data = df.to_dict('records')
        
        # 加载类别名称
        class_mapping_file = self.root_dir / "class_mapping.json"
        if class_mapping_file.exists():
            with open(class_mapping_file, 'r') as f:
                self.class_mapping = json.load(f)
            # 反转映射得到id到名称的映射
            self.class_names = [""] * (max(self.class_mapping.values()) + 1)
            for name, idx in self.class_mapping.items():
                self.class_names[idx] = name
        else:
            self.class_names = None
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        item = self.data[idx]
        # 使用Path对象处理路径，确保跨平台兼容
        img_path_str = item['image_path'].replace('\\', '/')
        img_path = self.root_dir / img_path_str
        label = item['label']
        
        # 如果图像不存在，尝试不同的大小写
        if not img_path.exists():
            # 记录警告日志
            logging.warning(f"图像不存在: {img_path}，尝试查找替代文件")
            
            # 尝试查找相同名称但大小写不同的文件
            parent_dir = img_path.parent
            if parent_dir.exists():
                for file in parent_dir.glob(f"{img_path.stem}.*"):
                    if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        img_path = file
                        logging.info(f"找到替代文件: {img_path}")
                        break
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            logging.error(f"无法找到图像文件: {img_path}")
            # 提供一个空白图像作为替代
            image = Image.new('RGB', (224, 224), color='gray')
        except Exception as e:
            logging.error(f"加载图像时出错: {img_path}, 错误: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
        
        # 应用转换
        if self.transform:
            image = self.transform(np.array(image))
        
        return {
            'images': image,
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DetectionDataset(BaseDataset):
    """植物病害检测数据集"""
    
    def __init__(self, 
                 root_dir: Union[str, Path],
                 mode: str = 'train',
                 transform_name: str = None,
                 filter_classes: List[str] = None):  # 添加这个参数
        """
        初始化数据集
        
        Args:
            root_dir: 数据根目录
            mode: 数据集模式 ('train', 'val', 或 'test')
            transform_name: 使用的转换名称
            filter_classes: 要过滤的植物类别列表
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        
        # 配置管理器
        self.config_manager = ConfigManager()
        
        # 增强管道
        self.aug_pipeline = AugmentationPipeline()
        
        # 设置转换
        if transform_name is None:
            transform_name = mode
        self.transform = self.aug_pipeline.get_transform_by_name(transform_name)
        
        # 加载标签文件
        label_file = self.root_dir / f"{mode}_labels.csv"
        if not label_file.exists():
            raise FileNotFoundError(f"找不到标签文件: {label_file}")
            
        self.samples = pd.read_csv(str(label_file))
        
        # 过滤类别
        if filter_classes:
            filtered_samples = []
            for idx, row in self.samples.iterrows():
                class_name = row['class_name']
                # 检查类名是否包含指定的植物类型
                if any(plant in class_name for plant in filter_classes):
                    filtered_samples.append(row)
            
            if filtered_samples:
                self.samples = pd.DataFrame(filtered_samples)
                logging.info(f"过滤后的数据集大小: {len(self.samples)}")
            else:
                logging.warning(f"过滤后没有剩余样本，将使用全部数据集")
        
        # 加载类别映射
        mapping_file = self.root_dir / "class_mapping.json"
        with open(mapping_file, 'r') as f:
            self.class_mapping = json.load(f)
        
        # 获取类别名称
        self.class_names = sorted(self.class_mapping.keys(), key=lambda x: self.class_mapping[x])
    
    def _load_data(self) -> List[Dict]:
        """加载目标检测数据标签"""
        # 假设我们有一个JSON文件包含检测标注数据
        labels_file = self.root_dir / f"{self.mode}_detection.json"
        
        if not labels_file.exists():
            # 如果没有专门的检测标注文件，我们可以使用分类标签并返回空标注
            # 这仅用于开发和测试
            class_labels_file = self.root_dir / f"{self.mode}_labels.csv"
            if not class_labels_file.exists():
                raise FileNotFoundError(f"找不到标签文件: {labels_file} 或 {class_labels_file}")
            
            df = pd.read_csv(class_labels_file)
            data = []
            for _, row in df.iterrows():
                data.append({
                    'image_path': row['image_path'],
                    'label': row['label'],  # 图像级别的标签
                    'boxes': [],  # 空的边界框
                    'labels': []  # 空的边界框类别
                })
        else:
            with open(labels_file, 'r') as f:
                data = json.load(f)
        
        # 加载类别名称
        class_mapping_file = self.root_dir / "class_mapping.json"
        if (class_mapping_file.exists()):
            with open(class_mapping_file, 'r') as f:
                self.class_mapping = json.load(f)
            # 反转映射得到id到名称的映射
            self.class_names = [""] * (max(self.class_mapping.values()) + 1)
            for name, idx in self.class_mapping.items():
                self.class_names[idx] = name
        else:
            self.class_names = ["背景", "健康", "病害1", "病害2", "病害3"]
        
        return data
    
    def __len__(self) -> int:
        """获取数据集样本数量"""
        return len(self.samples)  # 使用samples而不是data
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """获取单个样本"""
        row = self.samples.iloc[idx]
        # 使用Path对象处理路径，确保跨平台兼容
        img_path_str = row['image_path'].replace('\\', '/')
        img_path = self.root_dir / img_path_str
        
        # 如果图像不存在，尝试不同的大小写
        if not img_path.exists():
            # 记录警告日志
            logging.warning(f"图像不存在: {img_path}，尝试查找替代文件")
            
            # 尝试查找相同名称但大小写不同的文件
            parent_dir = img_path.parent
            if parent_dir.exists():
                for file in parent_dir.glob(f"{img_path.stem}.*"):
                    if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        img_path = file
                        logging.info(f"找到替代文件: {img_path}")
                        break
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
            original_width, original_height = image.size
        except FileNotFoundError:
            logging.error(f"无法找到图像文件: {img_path}")
            # 提供一个空白图像作为替代
            image = Image.new('RGB', (224, 224), color='gray')
            original_width, original_height = 224, 224
        except Exception as e:
            logging.error(f"加载图像时出错: {img_path}, 错误: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
            original_width, original_height = 224, 224
        
        # 获取标签，用于检测任务的类别标签
        label = row['label']
        
        # 创建覆盖整个图像的默认边界框
        boxes = torch.tensor([[0, 0, original_width, original_height]], dtype=torch.float32)
        labels = torch.tensor([label], dtype=torch.long)  # 使用原始类别标签
        
        # 应用转换
        if self.transform:
            image = self.transform(np.array(image))
        
        # 构建目标检测格式的输出
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # 修改这里: 只返回图像张量和目标字典
        return image, target