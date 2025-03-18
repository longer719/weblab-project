# scripts/train.py

import os
import sys
import logging
import argparse
import torch
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple

from src.models import PlantClassifier, DiseaseDetector
from src.training.trainer import Trainer
from src.training.train_config import TrainingConfig, ClassificationConfig, DetectionConfig
from src.data_processing.dataset import PlantDataset, DetectionDataset
from src.utils.logging_utils import setup_logging

# 添加统一配置管理导入
from src.utils.config_manager import ConfigManager
from src.utils.config_helpers import get_config_value

# 添加评估流程
from src.evaluation.evaluator import evaluate_model

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    # 保持此函数不变，因为它只处理命令行参数
    parser = argparse.ArgumentParser(description="训练植物疾病检测模型")
    
    # 基本参数
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--task", type=str, choices=["classification", "detection"], 
                        help="任务类型：分类或检测，默认从配置文件读取")
    parser.add_argument("--experiment-name", type=str, help="实验名称，默认使用时间戳")
    
    # 数据相关参数
    parser.add_argument("--data-dir", type=str, help="数据目录，覆盖配置文件中的设置")
    parser.add_argument("--train-split", type=float, help="训练集比例，覆盖配置文件中的设置")
    parser.add_argument("--val-split", type=float, help="验证集比例，覆盖配置文件中的设置")
    
    # 训练控制参数
    parser.add_argument("--epochs", type=int, help="训练轮数，覆盖配置文件中的设置")
    parser.add_argument("--batch-size", type=int, help="批次大小，覆盖配置文件中的设置")
    parser.add_argument("--lr", type=float, help="学习率，覆盖配置文件中的设置")
    parser.add_argument("--seed", type=int, help="随机种子，覆盖配置文件中的设置")
    
    # 系统相关参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="使用设备，默认使用可用的CUDA设备")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--debug", action="store_true", help="开启调试模式（更详细的日志）")
    
    # 模型和检查点参数
    parser.add_argument("--checkpoint", type=str, help="从检查点恢复训练")
    parser.add_argument("--pretrained", type=str, help="预训练模型路径")
    parser.add_argument("--evaluate-only", action="store_true", help="仅评估模型，不进行训练")
    
    # 输出控制参数
    parser.add_argument("--output-dir", type=str, default="experiments", help="输出目录")
    parser.add_argument("--log-interval", type=int, default=10, help="日志记录间隔（每多少批次记录一次）")
    parser.add_argument("--save-interval", type=int, help="检查点保存间隔（每多少个epoch保存一次）")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件并存入ConfigManager"""
    try:
        # 创建ConfigManager实例
        config_manager = ConfigManager()
        
        # 加载文件
        if config_path.endswith(('.yaml', '.yml')):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path}")
        
        # 将加载的配置存入ConfigManager
        for key, value in config.items():
            if isinstance(value, dict):
                # 处理嵌套字典
                for sub_key, sub_value in value.items():
                    config_type = key if key in ["data", "model", "train"] else "data"
                    full_key = f"{key}_{sub_key}" if key not in ["data", "model", "train"] else sub_key
                    config_manager.override(full_key, sub_value, config_type)
            else:
                # 处理普通键值
                config_type = "data"  # 默认类型
                if key.startswith("model_") or key == "task":
                    config_type = "model"
                elif key.startswith("train_"):
                    config_type = "train"
                
                config_manager.override(key, value, config_type)
        
        logging.info(f"成功加载配置文件: {config_path}")
        return config  # 仍然返回原始配置字典以保持兼容性
    except Exception as e:
        logging.error(f"加载配置文件出错: {e}")
        raise

def setup_experiment_folder(config: Dict[str, Any]) -> Path:
    """设置实验文件夹和日志"""
    # 获取ConfigManager实例
    config_manager = ConfigManager()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用ConfigManager获取参数，同时保持字典兼容性
    experiment_name = config_manager.get_dict_compatible(
        config, "experiment_name", f"experiment_{timestamp}", "train")
    
    # 创建实验目录结构
    output_dir = Path(config_manager.get_dict_compatible(
        config, "output_dir", "experiments", "train"))
    experiment_dir = output_dir / experiment_name
    checkpoints_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"
    tensorboard_dir = experiment_dir / "tensorboard"
    
    # 创建这些目录
    for directory in [experiment_dir, checkpoints_dir, logs_dir, tensorboard_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = logs_dir / "train.log"
    setup_logging(log_file, debug=config_manager.get_dict_compatible(config, "debug", False, "train"))
    
    # 保存配置副本
    config_file = experiment_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"实验目录创建成功: {experiment_dir}")
    return experiment_dir

def get_dataset(config: Dict[str, Any], mode: str = 'train') -> torch.utils.data.Dataset:
    """创建数据集实例"""
    config_manager = ConfigManager()
    
    # 获取基本参数
    task = config_manager.get_dict_compatible(config, "task", "classification", "model")
    
    # 获取数据目录，保持向后兼容性
    data_dir = None
    if "data" in config and "data_dir" in config["data"]:
        data_dir = config["data"]["data_dir"]
    else:
        data_dir = config_manager.get("data_dir", "data/processed", "data")
    
    if task == "classification":
        # 获取转换配置，保持向后兼容性
        transform_config = {}
        if "data" in config and "transforms" in config["data"] and mode in config["data"]["transforms"]:
            transform_config = config["data"]["transforms"][mode]
            
        # 获取数据分割，优先使用config字典，然后尝试ConfigManager
        split = None
        if "data" in config and f"{mode}_split" in config["data"]:
            split = config["data"][f"{mode}_split"]
        else:
            split = config_manager.get(f"{mode}_split", 
                      0.8 if mode == 'train' else 0.2, "data")
        
        dataset = PlantDataset(
            root_dir=data_dir,
            mode=mode,
            transform_config=transform_config,
            split=split,
            seed=config_manager.get_dict_compatible(config, "seed", 42, "train")
        )
        logging.info(f"创建{mode}分类数据集，包含 {len(dataset)} 个样本")
    elif task == "detection":
        # 同样支持两种配置方式
        transform_config = {}
        if "data" in config and "transforms" in config["data"] and mode in config["data"]["transforms"]:
            transform_config = config["data"]["transforms"][mode]
            
        split = None
        if "data" in config and f"{mode}_split" in config["data"]:
            split = config["data"][f"{mode}_split"]
        else:
            split = config_manager.get(f"{mode}_split", 
                      0.8 if mode == 'train' else 0.2, "data")
        
        dataset = DetectionDataset(
            root_dir=data_dir,
            mode=mode,
            transform_config=transform_config,
            split=split,
            seed=config_manager.get_dict_compatible(config, "seed", 42, "train")
        )
        logging.info(f"创建{mode}检测数据集，包含 {len(dataset)} 个样本")
    else:
        raise ValueError(f"不支持的任务类型: {task}")
    
    return dataset

def get_model(config: Dict[str, Any]) -> torch.nn.Module:
    """创建模型实例"""
    config_manager = ConfigManager()
    
    task = config_manager.get_dict_compatible(config, "task", "classification", "model")
    
    # 获取模型配置
    model_params = {}
    if "model" in config:
        model_params = config["model"]
    
    if task == "classification":
        model = PlantClassifier(
            # 这里使用新实现的PlantClassifier构造函数，它已支持ConfigManager
            config=model_params
        )
        logging.info(f"创建分类模型，基于 {config_manager.get_dict_compatible(model_params, 'backbone', 'resnet50', 'model')}，类别数量: {config_manager.get_dict_compatible(model_params, 'num_classes', 0, 'model')}")
    elif task == "detection":
        model = DiseaseDetector(
            # 这里使用新实现的DiseaseDetector构造函数，它已支持ConfigManager
            config=model_params
        )
        logging.info(f"创建检测模型，基于 {config_manager.get_dict_compatible(model_params, 'backbone', 'resnet50', 'model')}，类别数量: {config_manager.get_dict_compatible(model_params, 'num_classes', 0, 'model')}")
    else:
        raise ValueError(f"不支持的任务类型: {task}")
    
    return model

def get_loaders(train_dataset, val_dataset, config: Dict[str, Any]) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """创建数据加载器"""
    config_manager = ConfigManager()
    
    batch_size = config_manager.get_dict_compatible(config, "batch_size", 32, "data")
    num_workers = config_manager.get_dict_compatible(config, "num_workers", 4, "train")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=getattr(train_dataset, 'collate_fn', None)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=getattr(val_dataset, 'collate_fn', None)
    )
    
    logging.info(f"数据加载器创建成功，训练批次: {len(train_loader)}，验证批次: {len(val_loader)}")
    return train_loader, val_loader

def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> Dict[str, Any]:
    """加载训练检查点"""
    # 此函数不需要使用ConfigManager，保持不变
    if not os.path.exists(checkpoint_path):
        logging.error(f"检查点文件不存在: {checkpoint_path}")
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"成功加载模型权重从: {checkpoint_path}")
        else:
            # 直接是模型权重
            model.load_state_dict(checkpoint)
            checkpoint = {'epoch': -1, 'best_metric': None}
            logging.info(f"成功加载模型权重(仅权重无状态): {checkpoint_path}")
        
        return checkpoint
    except Exception as e:
        logging.error(f"加载检查点失败: {e}")
        raise

def save_training_state(trainer: Trainer, epoch: int, best_metric: float, save_path: Path) -> None:
    """保存训练状态，用于恢复训练"""
    # 此函数不需要使用ConfigManager，保持不变
    state = {
        'epoch': epoch,
        'best_metric': best_metric,
        'trainer_state': trainer.__dict__,
    }
    
    # 保存到临时文件，然后重命名，避免因为意外中断导致检查点文件损坏
    temp_path = save_path.with_suffix('.tmp')
    torch.save(state, temp_path)
    temp_path.rename(save_path)
    logging.info(f"训练状态已保存: {save_path}")

def train(config: Dict[str, Any], experiment_dir: Path) -> None:
    """执行模型训练流程"""
    config_manager = ConfigManager()
    
    # 设置随机种子以确保可重现性
    seed = config_manager.get_dict_compatible(config, "seed", 42, "train")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 获取数据集
    train_dataset = get_dataset(config, mode='train')
    val_dataset = get_dataset(config, mode='val')
    
    # 创建数据加载器
    train_loader, val_loader = get_loaders(train_dataset, val_dataset, config)
    
    # 创建模型
    model = get_model(config)
    device = config_manager.get_dict_compatible(config, "device", "cuda" if torch.cuda.is_available() else "cpu", "train")
    model.to(device)
    
    # 获取任务类型
    task_type = config_manager.get_dict_compatible(config, "task", "classification", "model")
    
    # 创建优化器
    optimizer_config = {}
    if "optimizer" in config:
        optimizer_config = config["optimizer"]
        
    optimizer_type = config_manager.get_dict_compatible(optimizer_config, "type", "Adam", "train")
    optimizer_params = config_manager.get_dict_compatible(optimizer_config, "params", {}, "train")
    
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    # 创建损失函数
    if task_type == "classification":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        # 检测任务使用模型内部的损失函数
        criterion = None
    
    # 创建学习率调度器
    scheduler_config = {}
    if "scheduler" in config:
        scheduler_config = config["scheduler"]
        
    scheduler_type = config_manager.get_dict_compatible(scheduler_config, "type", None, "train")
    scheduler_params = config_manager.get_dict_compatible(scheduler_config, "params", {}, "train")
    scheduler = None
    
    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    
    # 恢复检查点（如果有）
    start_epoch = 0
    best_metric = float('inf') if config_manager.get_dict_compatible(config, "monitor_mode", "min", "train") == "min" else float('-inf')
    checkpoint_path = config_manager.get_dict_compatible(config, "checkpoint", None, "train")
    
    if checkpoint_path:
        checkpoint = load_checkpoint(model, checkpoint_path)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_metric = checkpoint.get('best_metric', best_metric)
        logging.info(f"从epoch {start_epoch} 恢复训练，最佳指标: {best_metric}")
    
    # 创建TensorBoard目录
    tensorboard_dir = experiment_dir / "tensorboard"
    
    # 创建训练器
    trainer_config = {
        'device': device,
        'task_type': task_type,
        'epochs': config_manager.get_dict_compatible(config, "epochs", 100, "train"),
        'log_interval': config_manager.get_dict_compatible(config, "log_interval", 10, "train"),
        'tensorboard': {
            'enabled': True,
            'log_dir': str(tensorboard_dir)
        },
        'early_stopping': config_manager.get_dict_compatible(config, "early_stopping", {
            'patience': 10,
            'delta': 0.001,
            'mode': config_manager.get_dict_compatible(config, "monitor_mode", "min", "train")
        }, "train"),
        'mixed_precision': config_manager.get_dict_compatible(config, "mixed_precision", False, "train"),
        'scheduler': {
            'type': scheduler_type,
            'warmup_epochs': config_manager.get_dict_compatible(scheduler_config, "warmup_epochs", 0, "train"),
            'warmup_factor': config_manager.get_dict_compatible(scheduler_config, "warmup_factor", 0.1, "train")
        },
        'save_freq': config_manager.get_dict_compatible(config, "save_interval", 5, "train")
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=trainer_config,
        device=device,
        task_type=task_type
    )
    
    # 设置检查点目录
    checkpoints_dir = experiment_dir / "checkpoints"
    best_model_path = checkpoints_dir / "best_model.pth"
    
    # 训练模型
    logging.info("开始训练...")
    train_metrics = trainer.train(
        epochs=config_manager.get_dict_compatible(config, "epochs", 100, "train") - start_epoch,
        save_dir=str(checkpoints_dir)
    )
    
    # 保存最终模型
    final_model_path = checkpoints_dir / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"训练完成，最终模型已保存: {final_model_path}")
    
    # 训练完成后评估模型
    logging.info("训练完成，开始评估模型...")
    test_dataset = get_dataset(config, mode='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # 使用评估模块
    results = evaluate_model(
        model, 
        test_loader, 
        task_type=config['task_type'],
        output_dir=os.path.join(experiment_dir, 'evaluation')
    )
    
    # 记录关键指标
    for k, v in results.items():
        if isinstance(v, (int, float)):
            logging.info(f"评估 {k}: {v:.4f}")
    
    # 返回训练结果
    return {
        'train_metrics': train_metrics,
        'best_model_path': best_model_path,
        'final_model_path': final_model_path
    }

def monitor_training(trainer: Trainer, config: Dict[str, Any], tensorboard_dir: Path) -> None:
    """设置训练监控"""
    # TensorBoard的设置已在Trainer类中完成，这里添加系统资源监控
    config_manager = ConfigManager()
    
    try:
        import psutil
        import GPUtil
        
        # 是否启用资源监控
        enable_monitoring = config_manager.get_dict_compatible(config, "resource_monitoring", True, "train")
        if not enable_monitoring:
            return
        
        # 记录系统资源使用情况
        logging.info("系统资源监控已启用")
        
        # 监控CPU和内存
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        logging.info(f"CPU使用率: {cpu_percent}%, 内存使用率: {memory.percent}%")
        
        # 监控GPU（如果有）
        if torch.cuda.is_available():
            gpu_list = []
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.memory_allocated(i) / 1024**3  # GB
                gpu_list.append(f"GPU {i} ({gpu_name}): {gpu_mem:.2f} GB")
            logging.info("GPU使用情况: " + " | ".join(gpu_list))
    except ImportError:
        logging.warning("无法导入psutil或GPUtil，系统资源监控已禁用")
    except Exception as e:
        logging.warning(f"资源监控出错: {e}")

def log_training_progress(epoch: int, metrics: Dict[str, float], logger: logging.Logger) -> None:
    """记录训练进度到日志"""
    # 此函数不需要使用ConfigManager，保持不变
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Epoch {epoch+1}: {metrics_str}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        config_manager = ConfigManager()
        
        # 更新配置，命令行参数优先级高于配置文件
        if args.task:
            config_manager.override("task", args.task, "model")
            config["task"] = args.task
        if args.experiment_name:
            config_manager.override("experiment_name", args.experiment_name, "train")
            config["experiment_name"] = args.experiment_name
        if args.data_dir:
            config_manager.override("data_dir", args.data_dir, "data")
            if "data" not in config:
                config["data"] = {}
            config["data"]["data_dir"] = args.data_dir
        if args.train_split:
            config_manager.override("train_split", args.train_split, "data")
            if "data" not in config:
                config["data"] = {}
            config["data"]["train_split"] = args.train_split
        if args.val_split:
            config_manager.override("val_split", args.val_split, "data")
            if "data" not in config:
                config["data"] = {}
            config["data"]["val_split"] = args.val_split
        if args.epochs:
            config_manager.override("epochs", args.epochs, "train")
            config["epochs"] = args.epochs
        if args.batch_size:
            config_manager.override("batch_size", args.batch_size, "data")
            config["batch_size"] = args.batch_size
        if args.lr:
            config_manager.override("lr", args.lr, "train")
            if "optimizer" not in config:
                config["optimizer"] = {"type": "Adam", "params": {}}
            config["optimizer"]["params"]["lr"] = args.lr
        if args.seed:
            config_manager.override("seed", args.seed, "train")
            config["seed"] = args.seed
        if args.device:
            config_manager.override("device", args.device, "train")
            config["device"] = args.device
        if args.num_workers:
            config_manager.override("num_workers", args.num_workers, "train")
            config["num_workers"] = args.num_workers
        if args.debug:
            config_manager.override("debug", args.debug, "train")
            config["debug"] = args.debug
        if args.checkpoint:
            config_manager.override("checkpoint", args.checkpoint, "train")
            config["checkpoint"] = args.checkpoint
        if args.pretrained:
            config_manager.override("pretrained_path", args.pretrained, "model")
            if "model" not in config:
                config["model"] = {}
            config["model"]["pretrained_path"] = args.pretrained
        if args.output_dir:
            config_manager.override("output_dir", args.output_dir, "train")
            config["output_dir"] = args.output_dir
        if args.log_interval:
            config_manager.override("log_interval", args.log_interval, "train")
            config["log_interval"] = args.log_interval
        if args.save_interval:
            config_manager.override("save_interval", args.save_interval, "train")
            config["save_interval"] = args.save_interval
        
        # 设置实验目录
        experiment_dir = setup_experiment_folder(config)
        
        # 执行训练或评估
        if args.evaluate_only:
            # TODO: 实现评估功能
            logging.info("仅评估模式")
            # 加载模型用于评估
            model = get_model(config)
            device = config_manager.get_dict_compatible(config, "device", "cuda" if torch.cuda.is_available() else "cpu", "train")
            if args.checkpoint:
                load_checkpoint(model, args.checkpoint)
            model.to(device)
        else:
            # 执行训练
            train_results = train(config, experiment_dir)
            # 加载最佳模型用于评估
            model = get_model(config)
            best_model_path = train_results.get('best_model_path', experiment_dir / "checkpoints" / "best_model.pth")
            load_checkpoint(model, str(best_model_path))
            device = config_manager.get_dict_compatible(config, "device", "cuda" if torch.cuda.is_available() else "cpu", "train")
            model.to(device)
        
        # 训练完成后评估模型
        logging.info("训练完成，开始评估模型...")
        test_dataset = get_dataset(config, mode='test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=config_manager.get_dict_compatible(config, "batch_size", 32, "data"),
            shuffle=False,
            num_workers=config_manager.get_dict_compatible(config, "num_workers", 4, "train")
        )
        
        # 使用评估模块评估模型
        results = evaluate_model(
            model, 
            test_loader, 
            task_type=config_manager.get_dict_compatible(config, "task", "classification", "model"),
            output_dir=os.path.join(experiment_dir, 'evaluation')
        )
        
        # 记录关键指标
        logging.info("模型评估结果:")
        for k, v in results.items():
            if isinstance(v, (int, float)):
                logging.info(f"{k}: {v:.4f}")
        
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()