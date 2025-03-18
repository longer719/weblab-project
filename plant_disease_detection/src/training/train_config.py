# src/training/train_config.py

import os
import json
from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, StepLR

class OptimizerConfig:
    """优化器配置类"""
    
    def __init__(self, optimizer_type: str, params: Dict[str, Any]):
        """
        初始化优化器配置
        
        Args:
            optimizer_type: 优化器类型，例如'Adam', 'SGD'等
            params: 优化器参数字典
        """
        self.optimizer_type = optimizer_type
        self.params = params
        
        if optimizer_type not in self.available_optimizers():
            raise ValueError(f"不支持的优化器类型: {optimizer_type}，可用选项: {self.available_optimizers()}")
    
    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        """
        创建优化器实例
        
        Args:
            model_params: 模型参数或参数组
            
        Returns:
            配置的优化器实例
        """
        if self.optimizer_type == 'Adam':
            return optim.Adam(model_params, **self.params)
        elif self.optimizer_type == 'AdamW':
            return optim.AdamW(model_params, **self.params)
        elif self.optimizer_type == 'SGD':
            return optim.SGD(model_params, **self.params)
        elif self.optimizer_type == 'RMSprop':
            return optim.RMSprop(model_params, **self.params)
        else:
            raise ValueError(f"未实现的优化器类型: {self.optimizer_type}")
    
    @staticmethod
    def get_optimizer_params(optimizer_type: str) -> Dict[str, Any]:
        """
        获取优化器默认参数
        
        Args:
            optimizer_type: 优化器类型
            
        Returns:
            默认参数字典
        """
        if optimizer_type == 'Adam':
            return {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0}
        elif optimizer_type == 'AdamW':
            return {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01}
        elif optimizer_type == 'SGD':
            return {'lr': 0.01, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False}
        elif optimizer_type == 'RMSprop':
            return {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 0, 'momentum': 0}
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    @staticmethod
    def available_optimizers() -> List[str]:
        """
        获取所有可用的优化器
        
        Returns:
            可用优化器类型列表
        """
        return ['Adam', 'AdamW', 'SGD', 'RMSprop']

class SchedulerConfig:
    """学习率调度器配置类"""
    
    def __init__(self, scheduler_type: str, params: Dict[str, Any]):
        """
        初始化调度器配置
        
        Args:
            scheduler_type: 调度器类型，如'CosineAnnealingLR', 'StepLR'等
            params: 调度器参数字典
        """
        self.scheduler_type = scheduler_type
        self.params = params
        
        if scheduler_type not in self.available_schedulers():
            raise ValueError(f"不支持的调度器类型: {scheduler_type}，可用选项: {self.available_schedulers()}")
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> _LRScheduler:
        """
        创建调度器实例
        
        Args:
            optimizer: 优化器实例
            
        Returns:
            配置的学习率调度器实例
        """
        if self.scheduler_type == 'CosineAnnealingLR':
            return CosineAnnealingLR(optimizer, **self.params)
        elif self.scheduler_type == 'OneCycleLR':
            return OneCycleLR(optimizer, **self.params)
        elif self.scheduler_type == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(optimizer, **self.params)
        elif self.scheduler_type == 'StepLR':
            return StepLR(optimizer, **self.params)
        else:
            raise ValueError(f"未实现的调度器类型: {self.scheduler_type}")
    
    @staticmethod
    def get_scheduler_params(scheduler_type: str) -> Dict[str, Any]:
        """
        获取调度器默认参数
        
        Args:
            scheduler_type: 调度器类型
            
        Returns:
            默认参数字典
        """
        if scheduler_type == 'CosineAnnealingLR':
            return {'T_max': 10, 'eta_min': 0}
        elif scheduler_type == 'OneCycleLR':
            return {'max_lr': 0.01, 'total_steps': None, 'pct_start': 0.3}
        elif scheduler_type == 'ReduceLROnPlateau':
            return {'mode': 'min', 'factor': 0.1, 'patience': 10, 'threshold': 0.0001}
        elif scheduler_type == 'StepLR':
            return {'step_size': 30, 'gamma': 0.1}
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    @staticmethod
    def available_schedulers() -> List[str]:
        """
        获取所有可用的调度器
        
        Returns:
            可用调度器类型列表
        """
        return ['CosineAnnealingLR', 'OneCycleLR', 'ReduceLROnPlateau', 'StepLR']

class CheckpointConfig:
    """模型检查点配置类"""
    
    def __init__(self, save_dir: str, save_freq: int = 1, keep_last: int = 5, 
                save_best: bool = True, monitor: str = 'val_loss', mode: str = 'min'):
        """
        初始化检查点配置
        
        Args:
            save_dir: 检查点保存目录
            save_freq: 每隔多少个epoch保存一次检查点
            keep_last: 保留最新的几个检查点
            save_best: 是否保存最佳模型
            monitor: 监控哪个指标
            mode: 'min'表示越小越好，'max'表示越大越好
        """
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.keep_last = keep_last
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def should_save(self, epoch: int) -> bool:
        """
        检查是否应该保存模型
        
        Args:
            epoch: 当前epoch
            
        Returns:
            是否应该保存检查点
        """
        return (epoch + 1) % self.save_freq == 0
    
    def is_best(self, current_value: float, best_value: float) -> bool:
        """
        检查是否是最佳模型
        
        Args:
            current_value: 当前指标值
            best_value: 最佳指标值
            
        Returns:
            当前模型是否是最佳模型
        """
        if self.mode == 'min':
            return current_value < best_value
        else:
            return current_value > best_value
    
    def clean_old_checkpoints(self, checkpoints: List[str]):
        """
        清理旧的检查点
        
        Args:
            checkpoints: 检查点文件路径列表(按时间排序)
        """
        if len(checkpoints) > self.keep_last:
            checkpoints_to_remove = checkpoints[:-self.keep_last]
            for checkpoint in checkpoints_to_remove:
                try:
                    os.remove(checkpoint)
                except Exception as e:
                    print(f"清理旧检查点失败: {checkpoint}, 错误: {e}")

class TrainingConfig:
    def __init__(self, 
                 batch_size=32, 
                 epochs=50, 
                 optimizer_config=None,
                 scheduler_config=None, 
                 checkpoint_config=None):
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_config = optimizer_config or {'type': 'Adam', 'lr': 0.001}
        self.scheduler_config = scheduler_config or {'type': 'StepLR', 'step_size': 7}
        self.checkpoint_config = checkpoint_config or {'save_dir': 'checkpoints'}

class ClassificationConfig(TrainingConfig):
    """分类任务训练配置类"""
    
    def __init__(self,
                 num_classes: int,
                 class_weights: Optional[List[float]] = None,
                 label_smoothing: float = 0.0,
                 mixup_alpha: float = 0.0,
                 cutmix_alpha: float = 0.0,
                 **kwargs):
        """
        初始化分类任务配置
        
        Args:
            num_classes: 类别数量
            class_weights: 类别权重列表
            label_smoothing: 标签平滑参数
            mixup_alpha: Mixup增强参数
            cutmix_alpha: CutMix增强参数
            **kwargs: 传递给TrainingConfig的参数
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
    
    def get_criterion(self, device: str = 'cuda') -> nn.Module:
        """
        获取损失函数
        
        Args:
            device: 设备字符串
            
        Returns:
            配置的损失函数
        """
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=device)
            return nn.CrossEntropyLoss(weight=weights, label_smoothing=self.label_smoothing)
        else:
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """
        获取数据增强配置
        
        Returns:
            数据增强配置字典
        """
        return {
            'mixup_alpha': self.mixup_alpha,
            'cutmix_alpha': self.cutmix_alpha,
            'label_smoothing': self.label_smoothing,
            'random_resize_crop': True,
            'horizontal_flip': True,
            'vertical_flip': False,
            'color_jitter': {
                'brightness': 0.1,
                'contrast': 0.1,
                'saturation': 0.1,
                'hue': 0.02
            },
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }

class DetectionConfig(TrainingConfig):
    """检测任务训练配置类"""
    
    def __init__(self,
                 num_classes: int,
                 anchor_sizes: List[List[float]],
                 aspect_ratios: List[List[float]],
                 rpn_nms_thresh: float = 0.7,
                 box_nms_thresh: float = 0.5,
                 box_score_thresh: float = 0.05,
                 rpn_batch_size_per_image: int = 256,
                 box_batch_size_per_image: int = 128,
                 **kwargs):
        """
        初始化检测任务配置
        
        Args:
            num_classes: 类别数量
            anchor_sizes: 锚框大小列表
            aspect_ratios: 锚框长宽比列表
            rpn_nms_thresh: RPN非极大值抑制阈值
            box_nms_thresh: 边界框非极大值抑制阈值
            box_score_thresh: 边界框分数阈值
            rpn_batch_size_per_image: 每张图像RPN批次大小
            box_batch_size_per_image: 每张图像边界框批次大小
            **kwargs: 传递给TrainingConfig的参数
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.rpn_nms_thresh = rpn_nms_thresh
        self.box_nms_thresh = box_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.rpn_batch_size_per_image = rpn_batch_size_per_image
        self.box_batch_size_per_image = box_batch_size_per_image
    
    def get_criterion(self) -> Dict[str, Any]:
        """
        获取损失函数配置
        
        Returns:
            损失函数配置字典
        """
        return {
            'classification_loss': 'CrossEntropyLoss',
            'regression_loss': 'SmoothL1Loss',
            'regression_beta': 0.1,
            'balance_factor': 1.0
        }
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """
        获取数据增强配置
        
        Returns:
            数据增强配置字典
        """
        return {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.0,
            'random_resize': {
                'min_size': 800,
                'max_size': 1333,
                'fixed_ratio': False
            },
            'color_jitter': {
                'brightness': 0.1,
                'contrast': 0.1,
                'saturation': 0.1,
                'hue': 0.02
            },
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    
    def get_detection_params(self) -> Dict[str, Any]:
        """
        获取检测器参数
        
        Returns:
            检测器参数字典
        """
        return {
            'anchor_sizes': self.anchor_sizes,
            'aspect_ratios': self.aspect_ratios,
            'rpn_nms_thresh': self.rpn_nms_thresh,
            'box_nms_thresh': self.box_nms_thresh,
            'box_score_thresh': self.box_score_thresh,
            'rpn_batch_size_per_image': self.rpn_batch_size_per_image,
            'box_batch_size_per_image': self.box_batch_size_per_image,
            'rpn_positive_fraction': 0.5,
            'box_positive_fraction': 0.25,
            'rpn_pre_nms_top_n_train': 2000,
            'rpn_pre_nms_top_n_test': 1000,
            'rpn_post_nms_top_n_train': 2000,
            'rpn_post_nms_top_n_test': 1000,
            'rpn_fg_iou_thresh': 0.7,
            'rpn_bg_iou_thresh': 0.3,
            'box_fg_iou_thresh': 0.5,
            'box_bg_iou_thresh': 0.5,
        }

# 预设配置函数
def get_default_classification_config(num_classes: int) -> ClassificationConfig:
    """
    获取默认分类配置
    
    Args:
        num_classes: 类别数量
    
    Returns:
        默认的分类任务配置
    """
    return ClassificationConfig(
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        optimizer_config={
            'type': 'Adam',
            'params': {'lr': 0.001, 'weight_decay': 1e-5}
        },
        scheduler_config={
            'type': 'CosineAnnealingLR',
            'params': {'T_max': 50, 'eta_min': 1e-6}
        },
        checkpoint_config={
            'save_dir': 'checkpoints/classification',
            'save_freq': 5,
            'keep_last': 3
        },
        early_stopping_config={
            'patience': 10,
            'delta': 0.001,
            'monitor': 'val_loss',
            'mode': 'min'
        },
        mixed_precision=True,
        tensorboard_dir='runs/classification'
    )

def get_default_detection_config(num_classes: int) -> DetectionConfig:
    """
    获取默认检测配置
    
    Args:
        num_classes: 类别数量
    
    Returns:
        默认的检测任务配置
    """
    return DetectionConfig(
        num_classes=num_classes,
        anchor_sizes=[[32], [64], [128], [256], [512]],
        aspect_ratios=[[0.5, 1.0, 2.0]] * 5,
        batch_size=8,
        epochs=50,
        optimizer_config={
            'type': 'SGD',
            'params': {'lr': 0.005, 'momentum': 0.9, 'weight_decay': 5e-4}
        },
        scheduler_config={
            'type': 'StepLR',
            'params': {'step_size': 20, 'gamma': 0.1}
        },
        checkpoint_config={
            'save_dir': 'checkpoints/detection',
            'save_freq': 5,
            'keep_last': 3
        },
        early_stopping_config={
            'patience': 10,
            'delta': 0.001,
            'monitor': 'val_loss',
            'mode': 'min'
        },
        mixed_precision=True,
        tensorboard_dir='runs/detection'
    )

def get_resnet50_classifier_config(num_classes: int) -> ClassificationConfig:
    """
    获取ResNet50分类器配置
    
    Args:
        num_classes: 类别数量
    
    Returns:
        ResNet50分类任务配置
    """
    config = get_default_classification_config(num_classes)
    config.batch_size = 64
    config.optimizer_config = OptimizerConfig('AdamW', {'lr': 0.0002, 'weight_decay': 0.05})
    config.scheduler_config = SchedulerConfig('CosineAnnealingLR', {'T_max': 50, 'eta_min': 1e-6})
    config.label_smoothing = 0.1
    config.mixup_alpha = 0.2
    config.cutmix_alpha = 0.2
    return config

def get_faster_rcnn_detector_config(num_classes: int) -> DetectionConfig:
    """
    获取Faster R-CNN检测器配置
    
    Args:
        num_classes: 类别数量
    
    Returns:
        Faster R-CNN检测任务配置
    """
    config = get_default_detection_config(num_classes)
    config.optimizer_config = OptimizerConfig('SGD', {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 5e-4})
    config.scheduler_config = SchedulerConfig('StepLR', {'step_size': 15, 'gamma': 0.1})
    return config

def get_mobile_classifier_config(num_classes: int) -> ClassificationConfig:
    """
    获取移动端分类器配置
    
    Args:
        num_classes: 类别数量
    
    Returns:
        适合移动端部署的分类任务配置
    """
    config = get_default_classification_config(num_classes)
    config.batch_size = 128
    config.optimizer_config = OptimizerConfig('RMSprop', {'lr': 0.001, 'weight_decay': 1e-5})
    config.scheduler_config = SchedulerConfig('OneCycleLR', {'max_lr': 0.01, 'pct_start': 0.3})
    # 禁用混合精度以保证兼容性
    config.mixed_precision = False
    return config