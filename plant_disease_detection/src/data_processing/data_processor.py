# src/data_processing/data_processor.py

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# 新增导入
from src.utils.config_manager import ConfigManager
from src.utils.config_helpers import get_config_value

class DataProcessor:
    """数据处理核心类"""
    
    def __init__(self, config=None):
        """
        初始化数据处理器
        
        Args:
            config: 配置对象或字典（可选，如果未提供则使用ConfigManager）
        """
        self.config_manager = ConfigManager()
        # 保存原始配置，兼容旧代码
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 使用config_helpers获取配置值
        self.image_size = get_config_value(config, 'IMAGE_SIZE', (224, 224))
        
    def load_image(self, image_path: str) -> np.ndarray:
        """加载并预处理单张图片"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法加载图片: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)
            return image
        except Exception as e:
            self.logger.error(f"图片处理错误 {image_path}: {str(e)}")
            return None
            
    def normalize_image(self, image: np.ndarray, method: str = None) -> np.ndarray:
        """
        标准化图像
        
        Args:
            image: 输入图像
            method: 标准化方法，如未指定则从配置获取
        
        Returns:
            标准化后的图像
        """
        # 如果没有指定方法，则从配置获取
        if method is None:
            method = get_config_value(self.config, 'NORMALIZATION_METHOD', 'minmax')
        
        if method == 'minmax':
            # 缩放到[0,1]
            return image.astype(np.float32) / 255.0
        elif method == 'zscore':
            # 从配置获取均值和标准差，而不是硬编码
            mean = self.config_manager.get_dict_compatible(
                self.config, 'NORMALIZATION_MEAN', [0.485, 0.456, 0.406])
            std = self.config_manager.get_dict_compatible(
                self.config, 'NORMALIZATION_STD', [0.229, 0.224, 0.225])
            
            image = image.astype(np.float32) / 255.0
            image = (image - mean) / std
            return image
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    def apply_augmentation(self, image: np.ndarray, augmentation_config: Dict = None) -> np.ndarray:
        """
        应用数据增强
        
        Args:
            image: 输入图像
            augmentation_config: 增强配置，如未指定则从配置获取
            
        Returns:
            增强后的图像
        """
        # 如果没有提供增强配置，从配置获取
        if augmentation_config is None:
            augmentation_config = get_config_value(self.config, 'AUGMENTATION', {})
            
        # 水平翻转
        if augmentation_config.get('horizontal_flip', False) and np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            
        # 旋转
        if 'rotation_range' in augmentation_config and np.random.rand() > 0.5:
            angle = np.random.uniform(-augmentation_config['rotation_range'], 
                                      augmentation_config['rotation_range'])
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
        # 亮度调整
        if 'brightness_range' in augmentation_config and np.random.rand() > 0.5:
            brightness = np.random.uniform(augmentation_config['brightness_range'][0],
                                          augmentation_config['brightness_range'][1])
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
            
        # 缩放
        if 'zoom_range' in augmentation_config and np.random.rand() > 0.5:
            zoom = 1 + np.random.uniform(-augmentation_config['zoom_range'], 
                                         augmentation_config['zoom_range'])
            h, w = image.shape[:2]
            # 计算新尺寸
            new_h, new_w = int(h * zoom), int(w * zoom)
            
            # 缩放图像
            if zoom > 1:  # 放大后裁剪中心区域
                image = cv2.resize(image, (new_w, new_h))
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = image[start_h:start_h+h, start_w:start_w+w]
            else:  # 缩小后填充
                image = cv2.resize(image, (new_w, new_h))
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = cv2.copyMakeBorder(image, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, 
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
                
        return image
    
    def preprocess_image(self, image: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        预处理图像：调整大小、增强、标准化
        
        Args:
            image: 输入图像
            is_training: 是否处于训练模式（应用数据增强）
            
        Returns:
            处理后的图像
        """
        # 调整图像大小
        image = cv2.resize(image, self.image_size)
        
        # 如果是训练模式，应用数据增强
        if is_training:
            # 从配置获取增强参数
            augmentation_config = get_config_value(self.config, 'AUGMENTATION', {})
            image = self.apply_augmentation(image, augmentation_config)
            
        # 从配置获取标准化方法
        norm_method = get_config_value(self.config, 'NORMALIZATION_METHOD', 'zscore')
        return self.normalize_image(image, method=norm_method)
    
    def analyze_dataset(self, dataset_dir: str, output_dir: Optional[str] = None) -> Dict:
        """
        分析数据集并生成统计信息
        
        Args:
            dataset_dir: 数据集目录
            output_dir: 输出统计结果的目录（可选）
            
        Returns:
            包含统计信息的字典
        """
        # 如果未提供输出目录，则从配置获取
        if output_dir is None:
            output_dir = self.config_manager.get('STATS_DIR', 'data/stats')
        
        dataset_path = Path(dataset_dir)
        class_mapping_path = dataset_path / "class_mapping.json"
        
        if not class_mapping_path.exists():
            raise FileNotFoundError(f"找不到类别映射文件: {class_mapping_path}")
        
        # 加载类别映射
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
            
        # 反转类别映射，用于查找类名
        idx_to_class = {v: k for k, v in class_mapping.items()}
        
        # 加载训练集标签
        train_labels_path = dataset_path / "train_labels.csv"
        val_labels_path = dataset_path / "val_labels.csv"
        test_labels_path = dataset_path / "test_labels.csv"
        
        train_df = pd.read_csv(train_labels_path)
        val_df = pd.read_csv(val_labels_path)
        test_df = pd.read_csv(test_labels_path)
        
        # 计算类别分布
        class_counts = {
            'train': train_df['label'].value_counts().to_dict(),
            'val': val_df['label'].value_counts().to_dict(),
            'test': test_df['label'].value_counts().to_dict()
        }
        
        # 转换为类名
        class_distribution = {}
        for split, counts in class_counts.items():
            class_distribution[split] = {idx_to_class[int(idx)]: count for idx, count in counts.items()}
        
        # 计算图像尺寸分布
        image_sizes = []
        # 随机抽样一些图像计算尺寸
        sample_images = train_df['image_path'].sample(min(100, len(train_df))).tolist()
        for img_path in sample_images:
            full_path = dataset_path / img_path
            if full_path.exists():
                img = cv2.imread(str(full_path))
                if img is not None:
                    image_sizes.append(img.shape[:2])
        
        # 计算均值和标准差
        # 随机抽样一些图像计算统计量
        pixel_means = []
        pixel_stds = []
        sample_size = min(100, len(train_df))
        for img_path in tqdm(train_df['image_path'].sample(sample_size).tolist(), desc="计算像素统计量"):
            full_path = dataset_path / img_path
            if full_path.exists():
                img = cv2.imread(str(full_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    pixel_means.append(img.mean(axis=(0, 1)))
                    pixel_stds.append(img.std(axis=(0, 1)))
        
        mean_rgb = np.mean(pixel_means, axis=0) if pixel_means else np.array([0.485, 0.456, 0.406])
        std_rgb = np.mean(pixel_stds, axis=0) if pixel_stds else np.array([0.229, 0.224, 0.225])
        
        # 汇总统计信息
        stats = {
            'dataset_size': {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df),
                'total': len(train_df) + len(val_df) + len(test_df)
            },
            'class_distribution': class_distribution,
            'class_count': len(class_mapping),
            'image_size_stats': {
                'min': tuple(np.min(image_sizes, axis=0)) if image_sizes else (0, 0),
                'max': tuple(np.max(image_sizes, axis=0)) if image_sizes else (0, 0),
                'mean': tuple(np.mean(image_sizes, axis=0).astype(int)) if image_sizes else (0, 0)
            },
            'pixel_stats': {
                'mean_rgb': mean_rgb.tolist(),
                'std_rgb': std_rgb.tolist()
            }
        }
        
        # 如果指定了输出目录，保存统计结果和可视化
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存统计信息
            with open(output_path / "dataset_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            # 生成类别分布可视化
            plt.figure(figsize=(10, 6))
            class_names = list(class_distribution['train'].keys())
            train_counts = [class_distribution['train'].get(cls, 0) for cls in class_names]
            val_counts = [class_distribution['val'].get(cls, 0) for cls in class_names]
            test_counts = [class_distribution['test'].get(cls, 0) for cls in class_names]
            
            x = np.arange(len(class_names))
            width = 0.25
            
            plt.bar(x - width, train_counts, width, label='Train')
            plt.bar(x, val_counts, width, label='Validation')
            plt.bar(x + width, test_counts, width, label='Test')
            
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Class Distribution')
            plt.xticks(x, class_names, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path / "class_distribution.png")
            
            # 生成均值和标准差可视化
            plt.figure(figsize=(10, 4))
            channels = ['R', 'G', 'B']
            
            plt.subplot(1, 2, 1)
            plt.bar(channels, mean_rgb)
            plt.title('RGB Mean Values')
            plt.ylim(0, 1)
            
            plt.subplot(1, 2, 2)
            plt.bar(channels, std_rgb)
            plt.title('RGB Std Values')
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(output_path / "rgb_stats.png")
        
        return stats