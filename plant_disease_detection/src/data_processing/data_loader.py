# src/data_processing/data_loader.py

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
from src.data_processing.dataset import PlantDataset, BalancedPlantDataset
from src.utils.augmentation import AugmentationPipeline

# 移除直接导入DataConfig
# 导入ConfigManager
from src.utils.config_manager import ConfigManager
from src.utils.config_helpers import get_config_value

def create_data_loaders(
    data_dir: str,
    batch_size: int = None,
    num_workers: int = 4,
    use_balanced_dataset: bool = True,
    config: Any = None
) -> Dict[str, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批处理大小，如果为None则使用配置中的值
        num_workers: 数据加载工作进程数
        use_balanced_dataset: 是否使用类别平衡的数据集
        config: 配置对象或字典（如果为None则使用ConfigManager）
        
    Returns:
        包含训练、验证和测试数据加载器的字典
    """
    # 创建ConfigManager实例
    config_manager = ConfigManager()
    
    # 获取批次大小，优先使用传入的值，其次获取配置
    # 使用get_config_value来处理不同类型的config参数
    if batch_size is None:
        batch_size = get_config_value(config, 'BATCH_SIZE', 32, "data")
    
    # 获取图像大小
    image_size = get_config_value(config, 'IMAGE_SIZE', (224, 224), "data")
    
    # 获取数据转换/增强
    aug_pipeline = AugmentationPipeline(config)
    train_transform = aug_pipeline.get_train_transforms(image_size)
    valid_transform = aug_pipeline.get_valid_transforms(image_size)
    
    # 创建数据集类
    dataset_cls = BalancedPlantDataset if use_balanced_dataset else PlantDataset
    
    # 获取数据划分比例
    train_split = get_config_value(config, 'TRAIN_RATIO', 0.7, "data")
    val_split = get_config_value(config, 'VAL_RATIO', 0.15, "data")
    test_split = get_config_value(config, 'TEST_RATIO', 0.15, "data")
    
    # 创建数据集
    train_dataset = dataset_cls(
        data_dir=data_dir,
        transform=train_transform,
        mode='train',
        split=train_split
    )
    
    val_dataset = PlantDataset(
        data_dir=data_dir,
        transform=valid_transform,
        mode='val',
        split=val_split
    )
    
    test_dataset = PlantDataset(
        data_dir=data_dir,
        transform=valid_transform,
        mode='test',
        split=test_split
    )
    
    # 检查数据集是否为空
    for name, dataset in [('训练', train_dataset), ('验证', val_dataset), ('测试', test_dataset)]:
        if len(dataset) == 0:
            print(f"警告: {name}数据集为空!")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"创建数据加载器: 训练={len(train_dataset)}样本/{len(train_loader)}批次, "
          f"验证={len(val_dataset)}样本/{len(val_loader)}批次, "
          f"测试={len(test_dataset)}样本/{len(test_loader)}批次")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def create_detection_loaders(
    data_dir: str,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    config: Any = None
) -> Dict[str, DataLoader]:
    """
    创建用于目标检测的数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批处理大小
        num_workers: 数据加载工作进程数
        config: 配置对象或字典（如果为None则使用ConfigManager）
        
    Returns:
        包含训练和验证数据加载器的字典
    """
    # 创建ConfigManager实例
    config_manager = ConfigManager()
    
    # 获取批次大小
    if batch_size is None:
        batch_size = get_config_value(config, 'DETECTION_BATCH_SIZE', 4, "data")
    
    # 从配置获取转换参数
    min_size = get_config_value(config, 'min_size', 800, "model")
    max_size = get_config_value(config, 'max_size', 1333, "model")
    
    # 创建transform配置
    train_transform_config = {
        'random_resize': {
            'min_size': min_size,
            'max_size': max_size,
            'fixed_ratio': False
        },
        'horizontal_flip': 0.5,
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
    
    valid_transform_config = {
        'random_resize': {
            'min_size': min_size,
            'max_size': max_size,
            'fixed_ratio': True
        },
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    
    # 获取数据划分比例
    train_split = get_config_value(config, 'TRAIN_RATIO', 0.7, "data")
    val_split = get_config_value(config, 'VAL_RATIO', 0.15, "data")
    
    # 导入检测数据集
    try:
        from src.data_processing.dataset import DetectionDataset
    except ImportError:
        raise ImportError("无法导入DetectionDataset，请确保已实现此类")
    
    # 创建数据集
    train_dataset = DetectionDataset(
        root_dir=data_dir,
        mode='train',
        transform_config=train_transform_config,
        split=train_split
    )
    
    val_dataset = DetectionDataset(
        root_dir=data_dir,
        mode='val',
        transform_config=valid_transform_config,
        split=val_split
    )
    
    # 定义collate_fn以处理不同大小的图像和标注
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"创建检测数据加载器: 训练={len(train_dataset)}样本/{len(train_loader)}批次, "
          f"验证={len(val_dataset)}样本/{len(val_loader)}批次")
    
    return {
        'train': train_loader,
        'val': val_loader
    }