# scripts/train_detector.py
# 检测器专用训练流程

import os
import sys
import argparse
import yaml
import json
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm
import time
from datetime import datetime

# 设置PyTorch内存分配器配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.models import DiseaseDetector
from src.training.trainer import Trainer
from src.data_processing.dataset import DetectionDataset
from src.utils.logging_utils import setup_logging
from src.utils.config_manager import ConfigManager

# 确保logs目录存在
os.makedirs("logs", exist_ok=True)  
setup_logging(log_file="logs/train_detector.log")
logger = logging.getLogger(__name__)


def compute_optimal_anchors(train_dataset: DetectionDataset, num_clusters: int = 5) -> Dict[str, List]:
    """根据训练数据计算最优锚框配置
    
    Args:
        train_dataset: 训练数据集
        num_clusters: 聚类数量，决定锚框个数
        
    Returns:
        包含锚框尺寸和长宽比的字典
    """
    logging.info(f"计算最优锚框配置，使用 {num_clusters} 个聚类...")
    
    # 收集所有边界框的宽高比
    all_widths = []
    all_heights = []
    all_ratios = []
    
    # 使用采样而不是处理全部数据集
    sample_size = min(1000, len(train_dataset))
    indices = np.random.choice(len(train_dataset), sample_size, replace=False)
    
    for idx in tqdm(indices, desc="计算锚框统计数据"):
        sample = train_dataset[idx]
        if 'boxes' in sample['target'] and len(sample['target']['boxes']) > 0:
            boxes = sample['target']['boxes'].numpy()
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            ratios = widths / np.clip(heights, 1e-5, None)  # 防止除零
            
            all_widths.extend(widths)
            all_heights.extend(heights)
            all_ratios.extend(ratios)
    
    # 转换为numpy数组
    all_widths = np.array(all_widths)
    all_heights = np.array(all_heights)
    all_ratios = np.array(all_ratios)
    
    # 对长宽比进行聚类，获取不同形状
    kmeans_ratios = KMeans(n_clusters=3, random_state=42)
    kmeans_ratios.fit(all_ratios.reshape(-1, 1))
    anchor_ratios = kmeans_ratios.cluster_centers_.flatten().tolist()
    
    # 对边界框面积进行聚类，获取不同尺寸
    areas = np.sqrt(all_widths * all_heights)
    kmeans_areas = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_areas.fit(areas.reshape(-1, 1))
    anchor_areas = kmeans_areas.cluster_centers_.flatten().tolist()
    
    # 计算anchor sizes (每个特征层的锚框大小)
    anchor_sizes = []
    for area in sorted(anchor_areas):
        anchor_sizes.append([int(round(area))])
    
    # 修正anchor_ratios格式为每个位置的长宽比列表
    anchor_ratios = [sorted(anchor_ratios)] * len(anchor_sizes)
    
    logging.info(f"计算得到的锚框尺寸: {anchor_sizes}")
    logging.info(f"计算得到的锚框长宽比: {anchor_ratios[0]}")
    
    return {
        'sizes': anchor_sizes,
        'aspect_ratios': anchor_ratios
    }


def create_detector(config, anchor_config=None):
    """创建检测器模型"""
    detector_config = config.get('model', {}).copy()
    
    # 如果有自动计算的锚框配置，使用它
    if anchor_config:
        detector_config['anchor_sizes'] = anchor_config['sizes']
        detector_config['anchor_ratios'] = anchor_config['aspect_ratios']
    else:
        # 确保正确转换YAML中的锚框格式为元组格式
        if 'anchor_sizes' in detector_config:
            # 将嵌套列表转换为元组的元组
            sizes = detector_config['anchor_sizes']
            detector_config['anchor_sizes'] = tuple(tuple(x) for x in sizes)
        
        if 'anchor_ratios' in detector_config:
            # 将嵌套列表转换为元组的元组
            ratios = detector_config['anchor_ratios']
            detector_config['anchor_ratios'] = tuple(tuple(x) for x in ratios)
    
    # 创建检测器
    detector = DiseaseDetector(detector_config)
    return detector


def get_detection_dataset(config: Dict[str, Any], mode: str = 'train') -> DetectionDataset:
    """获取检测数据集"""
    config_manager = ConfigManager()
    data_config = config.get('data', {})
    
    # 获取数据路径
    data_dir = config_manager.get_dict_compatible(data_config, "data_dir", "data/processed", "data")
    
    # 获取数据过滤配置
    data_filter = config.get('data_filter', None)
    filter_classes = None
    if data_filter:
        filter_classes = data_filter.split(',')
        logging.info(f"使用过滤类别: {filter_classes}")
    else:
        logging.info("使用完整PlantVillage数据集，包含所有14种植物")
    
    # 创建数据集
    dataset = DetectionDataset(
        root_dir=data_dir,
        mode=mode,
        transform_name=mode if mode != 'test' else 'val',
        filter_classes=filter_classes
    )
    
    logging.info(f"加载{mode}数据集: {len(dataset)}个样本")
    return dataset


def train_detector(config: Dict[str, Any], experiment_dir: Path, checkpoint_path: Optional[str] = None) -> Tuple[DiseaseDetector, Dict[str, List[float]]]:
    """执行检测器训练流程
    
    Args:
        config: 训练配置
        experiment_dir: 实验目录
        checkpoint_path: 可选的检查点路径，用于恢复训练
        
    Returns:
        训练好的模型和训练历史
    """
    config_manager = ConfigManager()
    
    # 设置随机种子
    seed = config_manager.get_dict_compatible(config, "seed", 42, "train")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 设置设备
    device_name = config_manager.get_dict_compatible(
        config, "device", "cuda" if torch.cuda.is_available() else "cpu", "train")
    device = torch.device(device_name)
    logging.info(f"使用设备: {device}")
    
    # 显示CUDA内存信息
    if torch.cuda.is_available():
        logging.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        logging.info(f"CUDA内存分配: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        logging.info(f"CUDA内存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
    
    # 加载数据集
    logging.info("加载数据集...")
    train_dataset = get_detection_dataset(config, mode='train')
    val_dataset = get_detection_dataset(config, mode='val')
    
    # 检查数据集批次格式
    sample = next(iter(train_dataset))
    logging.info(f"样本格式: {type(sample)}, 键: {sample.keys() if isinstance(sample, dict) else 'not a dict'}")
    
    # 是否计算最优锚框配置
    anchor_config = None
    compute_anchors = config_manager.get_dict_compatible(config, "compute_anchors", False, "model")
    if compute_anchors:
        logging.info("计算最优锚框配置...")
        anchor_config = compute_optimal_anchors(train_dataset)
    
    # 创建检测器模型
    logging.info("创建检测器模型...")
    model = create_detector(config, anchor_config)
    model.to(device)
    
    # 显示模型信息
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型参数总数: {num_params:,}")
    logging.info(f"可训练参数数量: {trainable_params:,}")
    
    # 创建数据加载器
    data_config = config.get('data', {})
    batch_size = config_manager.get_dict_compatible(data_config, "batch_size", 8, "data")
    num_workers = config_manager.get_dict_compatible(data_config, "num_workers", 4, "data")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True  # 使用pin_memory加速数据传输
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True
    )
    
    logging.info(f"创建数据加载器: 训练={len(train_dataset)}样本/{len(train_loader)}批次, 验证={len(val_dataset)}样本/{len(val_loader)}批次")
    
    # 配置优化器
    train_config = config.get('train', {})
    lr = config_manager.get_dict_compatible(train_config, "lr", 0.001, "train")
    weight_decay = config_manager.get_dict_compatible(train_config, "weight_decay", 0.0001, "train")
    optimizer_name = config_manager.get_dict_compatible(train_config, "optimizer", "AdamW", "train")
    
    # 选择优化器
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = config_manager.get_dict_compatible(train_config, "momentum", 0.9, "train")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    # 创建学习率调度器配置
    scheduler_name = config_manager.get_dict_compatible(train_config, "scheduler", "CosineAnnealingLR", "train")
    scheduler_config = {
        'type': scheduler_name,
        'args': {},
        'warmup_epochs': config_manager.get_dict_compatible(train_config, "warmup_epochs", 5, "train")
    }
    
    if scheduler_name == 'CosineAnnealingLR':
        epochs = config_manager.get_dict_compatible(train_config, "epochs", 50, "train")
        scheduler_config['args'] = {'T_max': epochs}
    elif scheduler_name == 'MultiStepLR':
        scheduler_config['args'] = {'milestones': [30, 60, 90], 'gamma': 0.1}
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler_config['args'] = {'mode': 'min', 'factor': 0.5, 'patience': 5}
    
    # 创建Trainer配置
    trainer_config = {
        'epochs': config_manager.get_dict_compatible(train_config, "epochs", 50, "train"),
        'save_freq': config_manager.get_dict_compatible(train_config, "save_interval", 5, "train"),
        'mixed_precision': config_manager.get_dict_compatible(train_config, "mixed_precision", True, "train"),
        'grad_clip': config_manager.get_dict_compatible(train_config, "grad_clip", 1.0, "train"),  # 减小到1.0以提高稳定性
        'grad_accumulation_steps': config_manager.get_dict_compatible(train_config, "grad_accumulation_steps", 4, "train"),
        # 添加梯度累积，处理更复杂的场景
        'scheduler': scheduler_config,
        'early_stopping': {
            'patience': 15,  # 增加耐心值
            'delta': 0.001,
            'mode': 'min'
        },
        'tensorboard': {
            'enabled': True,
            'log_dir': str(experiment_dir / 'tensorboard')
        },
        # 添加检查点配置
        'checkpointing': {
            'enabled': True,
            'interval': 5,  # 每5个epoch保存一次
            'save_best_only': True,
            'save_dir': str(experiment_dir / 'checkpoints')
        }
    }
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=None,  # 检测模型内部有损失函数
        optimizer=optimizer,
        config=trainer_config,
        device=device_name,
        task_type='detection',
        class_names=train_dataset.class_names if hasattr(train_dataset, 'class_names') else None
    )
    
    # 如果提供了检查点路径，从检查点恢复
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)
        logging.info(f"成功从检查点恢复: {checkpoint_path}")
    
    # 执行训练
    logging.info(f"开始训练检测器，epochs: {trainer_config['epochs']}")
    train_results = trainer.train(trainer_config['epochs'], str(experiment_dir / 'checkpoints'))
    
    # 训练完成后，添加训练损失的可视化
    losses_dir = experiment_dir / "losses"
    losses_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制训练损失曲线
    plt.figure(figsize=(12, 8))
    
    # 获取训练历史
    train_history = train_results['train_metrics']
    val_history = train_results['val_metrics']
    
    epochs_range = range(1, len(train_history) + 1)
    
    # 绘制总损失
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, [m.get('loss', 0) for m in train_history], 'b-', label='训练损失')
    plt.plot(epochs_range, [m.get('loss', 0) for m in val_history], 'r-', label='验证损失')
    plt.title('总损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制分类损失
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, [m.get('loss_classifier', 0) for m in train_history], 'b-', label='训练分类损失')
    plt.plot(epochs_range, [m.get('loss_classifier', 0) for m in val_history], 'r-', label='验证分类损失')
    plt.title('分类损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制边界框回归损失
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, [m.get('loss_box_reg', 0) for m in train_history], 'b-', label='训练回归损失')
    plt.plot(epochs_range, [m.get('loss_box_reg', 0) for m in val_history], 'r-', label='验证回归损失')
    plt.title('边界框回归损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制学习率变化
    plt.subplot(2, 2, 4)
    lr_history = []
    for i in range(len(train_history)):
        # 尝试从训练结果中获取学习率，如果没有则使用前一个值或初始值
        if hasattr(trainer, '_get_lr') and i < len(trainer._get_lr()):
            lr_history.append(trainer._get_lr()[0])
        elif i > 0 and lr_history:
            lr_history.append(lr_history[-1])
        else:
            lr_history.append(lr)
            
    plt.plot(epochs_range, lr_history, 'g-')
    plt.title('学习率')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.grid(True)
    
    plt.tight_layout()
    loss_plot_path = losses_dir / "training_losses.png"
    plt.savefig(loss_plot_path)
    plt.close()
    logging.info(f"训练损失图表已保存: {loss_plot_path}")
    
    # 获取测试数据集
    test_dataset = get_detection_dataset(config, mode='test')
    
    # 评估模型
    logging.info("训练完成，开始评估模型性能...")
    evaluation_results = evaluate_detector(model, test_dataset, experiment_dir)
    
    # 返回模型和训练结果，添加评估结果
    return model, {
        'train_history': train_results['train_metrics'],
        'val_history': train_results['val_metrics'],
        'evaluation': evaluation_results
    }


def evaluate_detector(model: DiseaseDetector, dataset: DetectionDataset, experiment_dir: Path) -> Dict[str, float]:
    """评估检测器模型性能"""
    from src.evaluation.evaluator import DetectionEvaluator
    
    # 创建评估器
    evaluator = DetectionEvaluator(model)
    
    # 创建数据加载器，确保使用相同的collate_fn
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=detection_collate_fn  # 使用与训练时相同的collate_fn
    )
    
    # 执行评估
    output_dir = experiment_dir / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    logging.info("开始评估检测模型...")
    results = evaluator.evaluate(data_loader, str(output_dir))
    
    # 输出主要指标
    logging.info(f"mAP (IoU=0.5): {results['mAP']*100:.2f}%")
    
    # 输出每个类别的AP
    for i, ap in enumerate(results['AP_per_class']):
        class_name = dataset.class_names[i] if hasattr(dataset, 'class_names') else f"类别 {i}"
        logging.info(f"AP - {class_name}: {ap*100:.2f}%")
    
    return results


def load_detector_config(config_file: str) -> Dict[str, Any]:
    """加载检测器训练配置"""
    with open(config_file, 'r', encoding='utf-8') as f:  # 指定UTF-8编码
        config = yaml.safe_load(f)
    
    logging.info(f"从 {config_file} 加载配置")
    return config


def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """从命令行参数更新配置
    
    Args:
        config: 当前配置字典
        args: 命令行参数
        
    Returns:
        更新后的配置字典
    """
    # 确保存在基本结构
    if 'model' not in config:
        config['model'] = {}
    if 'train' not in config:
        config['train'] = {}
    if 'data' not in config:
        config['data'] = {}
        
    # 更新模型配置
    if args.backbone:
        config['model']['backbone'] = args.backbone
    if args.num_classes is not None:
        config['model']['num_classes'] = args.num_classes
    if args.compute_anchors is not None:
        config['model']['compute_anchors'] = args.compute_anchors
        
    # 更新训练配置
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.lr:
        config['train']['lr'] = args.lr
    # 删除下面这行，或者注释掉
    # if args.optimizer:
    #    config['train']['optimizer'] = args.optimizer
        
    # 更新数据配置
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
        
    return config


def setup_experiment_folder(config: Dict[str, Any]) -> Path:
    """设置实验文件夹和日志"""
    config_manager = ConfigManager()
    
    # 获取实验名称，优先使用config字典中的值
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get("experiment_name", 
        config_manager.get("experiment_name", f"detector_{timestamp}", "train"))
    
    # 获取输出目录
    output_dir = Path(config.get("output_dir", 
        config_manager.get("output_dir", "experiments", "train")))
    
    # 创建实验目录结构
    experiment_dir = output_dir / experiment_name
    checkpoints_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"
    
    # 创建这些目录
    for directory in [experiment_dir, checkpoints_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = logs_dir / "train.log"
    setup_logging(log_file)
    
    # 保存配置副本
    config_file = experiment_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"实验目录创建成功: {experiment_dir}")
    return experiment_dir


def save_model(model: DiseaseDetector, config: Dict, experiment_dir: Path) -> str:
    """保存检测器模型和配置"""
    # 确保目录存在
    model_dir = experiment_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = model_dir / 'disease_detector.pth'
    torch.save(model.state_dict(), model_path)
    
    # 保存配置
    config_path = model_dir / 'disease_detector_config.json'
    with open(config_path, 'w') as f:
        json.dump(model.detector_config if hasattr(model, 'detector_config') else {}, f, indent=2)
    
    # 更新版本信息
    version_path = Path("models/version_info.json")
    version_info = {}
    
    if version_path.exists():
        try:
            with open(version_path, 'r') as f:
                version_info = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"读取版本文件出错: {e}，创建新的版本文件")
    
    # 确保models目录存在
    version_path.parent.mkdir(exist_ok=True)
    
    # 添加详细的描述，包括训练信息
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_type = "detector"
    
    if model_type not in version_info:
        version_info[model_type] = {"version": "v1.0.0"}
    
    # 更新版本号
    version = version_info[model_type].get('version', 'v1.0.0')
    version_parts = version.lstrip('v').split('.')
    version_parts[-1] = str(int(version_parts[-1]) + 1)
    new_version = 'v' + '.'.join(version_parts)
    
    version_info[model_type] = {
        "version": new_version,
        "date": current_time,
        "description": f"从实验部署的detector模型，使用完整PlantVillage数据集训练",
        "experiment_dir": str(experiment_dir),
        "trained_plants": "苹果、蓝莓、樱桃、玉米、葡萄、橙子、桃子、甜椒、土豆、树莓、大豆、西葫芦、草莓、番茄"
    }
    
    # 保存更新后的版本信息
    with open(version_path, 'w') as f:
        json.dump(version_info, f, indent=2, ensure_ascii=False)
    
    return str(model_path)


def detection_collate_fn(batch):
    """
    自定义收集函数，处理不同大小的图像和目标
    """
    images = []
    targets = []
    
    for sample in batch:
        # 从样本中提取图像和目标
        if 'images' in sample:
            images.append(sample['images'])
        elif 'image' in sample:
            images.append(sample['image'])
        else:
            raise KeyError(f"样本中既没有'images'也没有'image'键。可用的键: {list(sample.keys())}")
            
        if 'targets' in sample:
            targets.append(sample['targets'])
        elif 'target' in sample:
            targets.append(sample['target'])
        else:
            raise KeyError(f"样本中没有'targets'或'target'键。可用的键: {list(sample.keys())}")
    
    return images, targets


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练植物病害检测模型")
    
    parser.add_argument('--config', type=str, default='configs/detector_training.yaml',
                      help='配置文件路径')
    parser.add_argument('--data_dir', type=str, help='数据目录路径')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--epochs', type=int, help='训练轮次')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--backbone', type=str, help='骨干网络')
    parser.add_argument('--num_classes', type=int, help='类别数量')
    parser.add_argument('--compute_anchors', type=bool, help='是否自动计算锚框')
    parser.add_argument('--device', type=str, help='训练设备')
    
    # 恢复训练参数 - 删除重复定义
    parser.add_argument('--resume', action='store_true', 
                      help='从检查点恢复训练')
    parser.add_argument('--resume_path', type=str,
                      help='检查点路径，如果不提供则使用最新的检查点')
    
    return parser.parse_args()


def main():
    """主函数，控制训练流程"""
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 解析命令行参数
    args = parse_args()
    
    # 记录系统信息
    logging.info("=" * 50)
    logging.info("系统信息:")
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info(f"CUDA是否可用: 是")
        logging.info(f"CUDA版本: {torch.version.cuda}")
        logging.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    else:
        logging.info(f"CUDA是否可用: 否")
    logging.info("=" * 50)
    
    # 加载配置文件
    if args.config and os.path.exists(args.config):
        config = load_detector_config(args.config)
        logging.info(f"成功加载配置文件: {args.config}")
    else:
        logging.error(f"找不到配置文件: {args.config}")
        return
    
    # 从命令行参数更新配置
    config = update_config_from_args(config, args)
    
    # 设置随机种子
    seed = config.get('train', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 创建实验目录 - 使用统一的函数
    experiment_dir = setup_experiment_folder(config)
    
    try:
        # 从checkpoint恢复
        checkpoint_path = None
        if args.resume:
            # 检查是否提供了检查点路径，如果没有，寻找最近的检查点
            if args.resume_path and os.path.exists(args.resume_path):
                checkpoint_path = args.resume_path
                logging.info(f"从指定检查点恢复训练: {checkpoint_path}")
            else:
                # 查找最新的自动保存检查点
                checkpoints_dir = experiment_dir / "checkpoints"
                if checkpoints_dir.exists():
                    checkpoint_files = list(checkpoints_dir.glob("checkpoint_epoch*.pth"))
                    if checkpoint_files:
                        # 按修改时间排序，获取最新的检查点
                        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        checkpoint_path = str(checkpoint_files[0])
                        logging.info(f"从最新检查点恢复训练: {checkpoint_path}")
                    else:
                        logging.warning(f"检查点目录存在，但没有找到检查点文件")
                else:
                    logging.warning(f"检查点目录不存在: {checkpoints_dir}")
        
        # 记录训练开始时间
        start_time = time.time()
        logging.info("开始训练检测器模型")
        
        # 修改配置以防止NaN
        if args.resume:
            # 如果是恢复训练，降低学习率以增加稳定性
            if 'train' in config:
                if 'lr' in config['train']:
                    original_lr = config['train']['lr']
                    config['train']['lr'] = original_lr * 0.2  # 降低到原来的20%
                    logging.info(f"恢复训练时降低学习率: {original_lr} -> {config['train']['lr']}")
                
                # 降低梯度裁剪阈值
                config['train']['grad_clip'] = 0.5  # 降低梯度裁剪阈值
                
                # 禁用混合精度
                config['train']['mixed_precision'] = False
                
                logging.info("已调整训练参数以增加稳定性: 降低梯度裁剪阈值，关闭混合精度训练")
        
        # 训练检测器，传递检查点路径
        model, train_results = train_detector(config, experiment_dir, checkpoint_path)
        
        # 保存最终模型
        model_path = save_model(model, config, experiment_dir)
        logging.info(f"训练完成，模型已保存到: {model_path}")
        
        # 记录训练结束时间并计算总时间
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logging.info(f"训练总时长: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("已清理CUDA缓存")
        
        return 0
    except Exception as e:
        logging.error(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())