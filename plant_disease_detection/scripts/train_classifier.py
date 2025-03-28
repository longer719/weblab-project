# scripts/train_classifier.py
#分类器训练脚本

import os
import sys
import logging
import argparse
import torch
import yaml
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
import time
from datetime import datetime

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 导入必要的项目模块
from src.models import PlantClassifier
from src.training.trainer import Trainer
from src.data_processing.dataset import ClassificationDataset
from src.utils.logging_utils import setup_logging

# 新增: 导入配置管理类
from src.utils.config_manager import ConfigManager
from src.utils.config_helpers import get_config_value

# 训练脚本中添加评估功能
from src.evaluation.evaluator import ClassificationEvaluator
from src.evaluation.model_interpreter import ModelInterpreter

# 在 scripts/train_classifier.py 中修改 parse_args 函数
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练植物分类模型")
    
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, help='数据目录路径')
    parser.add_argument('--output_dir', type=str, help='输出目录路径')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--epochs', type=int, help='训练轮次')
    parser.add_argument('--learning_rate', '--lr', type=float, help='学习率')  # 添加别名
    parser.add_argument('--backbone', type=str, help='主干网络 (resnet18, resnet50等)')
    parser.add_argument('--num_classes', type=int, help='类别数量')
    parser.add_argument('--checkpoint', type=str, help='断点恢复文件路径')
    parser.add_argument('--device', type=str, help='训练设备 (cuda, cpu)')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    return parser.parse_args()

# 修改 scripts/train_classifier.py 中的 load_classifier_config 函数
def load_classifier_config(config_file: str) -> Dict[str, Any]:
    """加载训练配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:  # 指定UTF-8编码
        config = yaml.safe_load(f)
    
    logging.info(f"从 {config_file} 加载配置")
    return config

# 更新 setup_experiment_folder 函数，确保与 train_detector.py 中的结构一致

def setup_experiment_folder(config):
    """设置实验文件夹和日志"""
    config_manager = ConfigManager()
    
    # 获取实验名称，优先使用config字典中的值
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get("experiment_name", 
        config_manager.get("experiment_name", f"plant_classifier", "train"))
    
    # 获取输出目录
    output_dir = Path(config.get("output_dir", 
        config_manager.get("output_dir", "experiments", "train")))
    
    # 创建实验目录结构
    experiment_dir = output_dir / experiment_name
    checkpoints_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"
    tensorboard_dir = experiment_dir / "tensorboard" # 添加tensorboard目录
    
    # 创建这些目录
    for directory in [experiment_dir, checkpoints_dir, logs_dir, tensorboard_dir]:
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

def prepare_data(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """准备训练和验证数据加载器"""
    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', 'data/processed')
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    
    # 修正这里 - 使用mode参数而不是split
    train_dataset = ClassificationDataset(
        root_dir=data_dir,
        mode='train',  # 使用'mode'而不是'split'
        transform_name='train'
    )
    
    val_dataset = ClassificationDataset(
        root_dir=data_dir,
        mode='val',    # 使用'mode'而不是'split'
        transform_name='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

def create_model(config):
    """创建分类器模型"""
    config_manager = ConfigManager()
    
    # 获取模型参数
    model_params = {}
    if "model" in config:
        model_params = config["model"]
    else:
        # 从ConfigManager构建模型参数
        model_params = {
            "backbone": config_manager.get("backbone", "resnet50", "model"),
            "num_classes": config_manager.get("num_classes", 10, "model"),
            "pretrained": config_manager.get("pretrained", True, "model"),
            "dropout_rate": config_manager.get("dropout_rate", 0.3, "model")
        }
    
    # 创建模型
    model = PlantClassifier(model_params)
    
    # 加载检查点(如果提供)
    checkpoint_path = config.get("checkpoint", 
        config_manager.get("checkpoint", None, "train"))
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"从检查点加载模型权重: {checkpoint_path}")
            else:
                model.load_state_dict(checkpoint)
                logging.info(f"从检查点加载模型权重: {checkpoint_path}")
        except Exception as e:
            logging.error(f"加载检查点失败: {e}")
    
    backbone_name = model_params.get("backbone", "resnet50")
    num_classes = model_params.get("num_classes", 10)
    logging.info(f"创建分类器模型: 骨干网络={backbone_name}, 类别数={num_classes}")
    
    return model

# 修改 train_model 函数，添加渐进式微调支持
def train_model(model, train_loader, val_loader, config, experiment_dir):
    """训练模型"""
    config_manager = ConfigManager()
    
    # 获取训练参数
    device = config.get("device", 
        config_manager.get("device", "cuda" if torch.cuda.is_available() else "cpu", "train"))
    epochs = config.get("epochs", 
        config_manager.get("epochs", 100, "train"))  
    learning_rate = config.get("learning_rate", 
        config_manager.get("learning_rate", 0.0001, "train"))
    
    # 新增: 读取微调相关参数
    warmup_epochs = config.get("warmup_epochs", 
        config_manager.get("warmup_epochs", 5, "train"))
    backbone_lr = config.get("backbone_lr", 
        config_manager.get("backbone_lr", 0.00001, "train"))
    
    # 新增: 读取label_smoothing参数
    label_smoothing = config.get("label_smoothing", 
        config_manager.get("label_smoothing", 0.1, "train"))
    
    # 显示训练设备信息
    logging.info(f"使用设备: {device}")
    if device == "cuda" and torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        logging.info(f"当前分配: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
    
    # 将模型移动到设备
    model.to(device)
    
    # 设置损失函数，添加label_smoothing参数
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    logging.info(f"使用Label Smoothing损失函数，系数: {label_smoothing}")
    
    # ===== 阶段1: 仅训练头部 =====
    logging.info(f"--- [阶段 1/2] 开始训练分类器头部 ({warmup_epochs} epochs) ---")
    
    # 冻结骨干网络
    if hasattr(model, 'backbone'):
        logging.info("冻结骨干网络参数...")
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        logging.warning("模型未找到 'backbone' 属性，将训练所有可训练层。")
    
    # 获取优化器设置
    optimizer_name = config.get("optimizer", "AdamW")
    weight_decay = config.get("weight_decay", 0.01)
    
    # 收集需要训练的参数
    if hasattr(model, 'classifier_head'):
        trainable_params = list(model.classifier_head.parameters())
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # 创建第一阶段优化器
    if optimizer_name == "AdamW":
        optimizer_stage1 = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer_stage1 = torch.optim.SGD(trainable_params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer_stage1 = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    
    # 创建第一阶段学习率调度器
    scheduler_name = config.get("scheduler", "OneCycleLR")
    scheduler_stage1 = None
    
    if scheduler_name == "OneCycleLR":
        steps_per_epoch = len(train_loader)
        scheduler_stage1 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_stage1, 
            max_lr=learning_rate * 10,
            steps_per_epoch=steps_per_epoch,
            epochs=warmup_epochs,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_stage1, 
            T_max=warmup_epochs,
            eta_min=learning_rate / 100
        )
    
    # 创建第一阶段训练器配置
    trainer_config_stage1 = {
        'device': device,
        'task_type': 'classification',
        'epochs': warmup_epochs,
        'log_interval': config.get("log_interval", 
            config_manager.get("log_interval", 10, "train")),
        'scheduler': {
            'type': scheduler_name,
            'instance': scheduler_stage1
        },
        'mixed_precision': config.get("mixed_precision", True),
        'early_stopping': {
            'patience': config.get("patience", 
                config_manager.get("patience", 15, "train")),
            'min_delta': config.get("min_delta", 
                config_manager.get("min_delta", 0.001, "train")),
            'monitor': config.get("monitor", 
                config_manager.get("monitor", "val_loss", "train"))
        },
        'tensorboard': {
            'enabled': True,
            'log_dir': str(experiment_dir / 'tensorboard' / 'stage1')
        },
        'grad_clip': 1.0,
        'checkpoint_interval': 5
    }
    
    # 创建第一阶段训练器
    trainer_stage1 = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage1,
        config=trainer_config_stage1,
        device=device,
        task_type='classification',
        class_names=None
    )
    
    # 保存检查点的路径
    checkpoints_dir_stage1 = experiment_dir / "checkpoints" / "stage1"
    checkpoints_dir_stage1.mkdir(parents=True, exist_ok=True)
    
    # 开始第一阶段训练
    logging.info(f"开始第一阶段训练, 设备={device}, 轮数={warmup_epochs}")
    stage1_results = trainer_stage1.train(epochs=warmup_epochs, save_dir=str(checkpoints_dir_stage1))
    
    # ===== 阶段2: 微调整个模型 =====
    logging.info(f"\n--- [阶段 2/2] 开始微调整个模型 ({epochs-warmup_epochs} epochs) ---")
    
    # 解冻骨干网络
    if hasattr(model, 'backbone'):
        logging.info("解冻骨干网络参数...")
        for param in model.backbone.parameters():
            param.requires_grad = True
    
    # 构建参数组，使用差分学习率
    backbone_params = []
    head_params = []
    
    # 区分骨干网络和头部参数
    if hasattr(model, 'backbone') and hasattr(model, 'classifier_head'):
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.classifier_head.parameters())
        logging.info(f"骨干网络参数: {len(backbone_params)}, 头部参数: {len(head_params)}")
    else:
        # 如果结构不明确，所有参数使用同一学习率
        logging.warning("无法区分骨干网络和头部参数，将使用相同的学习率")
        head_params = list(model.parameters())
    
    # 构建差分学习率的参数组
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})  # 骨干网络低学习率
    if head_params:
        param_groups.append({'params': head_params, 'lr': learning_rate})     # 头部高学习率
    
    # 创建第二阶段优化器
    if optimizer_name == "AdamW":
        optimizer_stage2 = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer_stage2 = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer_stage2 = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    
    # 创建第二阶段学习率调度器
    remaining_epochs = epochs - warmup_epochs
    scheduler_stage2 = None
    
    if scheduler_name == "OneCycleLR":
        steps_per_epoch = len(train_loader)
        scheduler_stage2 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_stage2, 
            max_lr=[backbone_lr * 10, learning_rate * 10],  # 为每个参数组指定max_lr
            steps_per_epoch=steps_per_epoch,
            epochs=remaining_epochs,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_stage2, 
            T_max=remaining_epochs,
            eta_min=backbone_lr / 100
        )
    
    # 创建第二阶段训练器配置
    trainer_config_stage2 = {
        'device': device,
        'task_type': 'classification',
        'epochs': remaining_epochs,
        'log_interval': config.get("log_interval", 
            config_manager.get("log_interval", 10, "train")),
        'scheduler': {
            'type': scheduler_name,
            'instance': scheduler_stage2
        },
        'mixed_precision': config.get("mixed_precision", True),
        'early_stopping': {
            'patience': config.get("patience", 
                config_manager.get("patience", 15, "train")),
            'min_delta': config.get("min_delta", 
                config_manager.get("min_delta", 0.001, "train")),
            'monitor': config.get("monitor", 
                config_manager.get("monitor", "val_loss", "train"))
        },
        'tensorboard': {
            'enabled': True,
            'log_dir': str(experiment_dir / 'tensorboard' / 'stage2')
        },
        'grad_clip': 1.0,
        'checkpoint_interval': 5
    }
    
    # 创建第二阶段训练器
    trainer_stage2 = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage2,
        config=trainer_config_stage2,
        device=device,
        task_type='classification',
        class_names=None
    )
    
    # 保存检查点的路径
    checkpoints_dir_stage2 = experiment_dir / "checkpoints" / "stage2"
    checkpoints_dir_stage2.mkdir(parents=True, exist_ok=True)
    
    # 开始第二阶段训练
    logging.info(f"开始第二阶段训练, 设备={device}, 轮数={remaining_epochs}")
    stage2_results = trainer_stage2.train(epochs=remaining_epochs, save_dir=str(checkpoints_dir_stage2))
    
    # 保存最终模型
    model_path = save_model(model, config, experiment_dir)
    logging.info(f"训练完成, 最终模型已保存: {model_path}")
    
    # 合并两个阶段的结果
    combined_results = {
        'stage1': stage1_results,
        'stage2': stage2_results
    }
    
    return combined_results

# 添加评估和解释功能
def generate_evaluation_report(model, data_loader, class_names, output_dir):
    # 创建评估器
    evaluator = ClassificationEvaluator(model)
    
    # 执行评估
    results = evaluator.evaluate(data_loader, output_dir)
    
    # 使用模型解释器生成可视化
    interpreter = ModelInterpreter(model)
    
    # 为测试集中的几个样本生成解释
    for i, batch in enumerate(data_loader):
        if i >= 5:  # 只处理前5个批次
            break
        images = batch['images']
        labels = batch['labels']
        
        for j in range(min(3, len(images))):  # 每批次最多3张图
            explanation = interpreter.explain_prediction(
                images[j], 
                class_idx=labels[j].item(), 
                class_names=class_names
            )
            # 保存解释图
            save_path = os.path.join(output_dir, f'explanation_batch{i}_sample{j}.png')
            plt.imsave(save_path, explanation)

def update_config_from_args(config, args):
    """从命令行参数更新配置并同步到ConfigManager"""
    config_manager = ConfigManager()
    
    # 更新配置字典并同步到ConfigManager
    if args.data_dir:
        config["data_dir"] = args.data_dir
        config_manager.override("data_dir", args.data_dir, "data")
        
    if args.output_dir:
        config["output_dir"] = args.output_dir
        config_manager.override("output_dir", args.output_dir, "train")
        
    if args.batch_size:
        config["batch_size"] = args.batch_size
        config_manager.override("batch_size", args.batch_size, "data")
        
    if args.epochs:
        config["epochs"] = args.epochs
        config_manager.override("epochs", args.epochs, "train")
        
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
        config_manager.override("learning_rate", args.learning_rate, "train")
        
    if args.backbone:
        if "model" not in config:
            config["model"] = {}
        config["model"]["backbone"] = args.backbone
        config_manager.override("backbone", args.backbone, "model")
        
    if args.num_classes:
        if "model" not in config:
            config["model"] = {}
        config["model"]["num_classes"] = args.num_classes
        config_manager.override("num_classes", args.num_classes, "model")
        
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
        config_manager.override("checkpoint", args.checkpoint, "train")
        
    if args.device:
        config["device"] = args.device
        config_manager.override("device", args.device, "train")
        
    if args.seed:
        config["seed"] = args.seed
        config_manager.override("seed", args.seed, "train")
    
    return config

# 修改 save_model 函数，确保保持一致的结构

def save_model(model, config, experiment_dir):
    """保存模型和配置"""
    # 确保目录存在
    model_dir = experiment_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = model_dir / 'plant_classifier.pth'
    torch.save(model.state_dict(), model_path)
    
    # 保存配置
    config_path = model_dir / 'plant_classifier_config.json'
    with open(config_path, 'w') as f:
        json.dump(model.classifier_config if hasattr(model, 'classifier_config') else {}, f, indent=2)
    
    # 更新版本信息 (保持与train_detector.py相同的结构)
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
    model_type = "classifier"
    
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
        "description": f"从实验部署的{model_type}模型，使用完整PlantVillage数据集训练",
        "experiment_dir": str(experiment_dir),
        "trained_plants": "苹果、蓝莓、樱桃、玉米、葡萄、橙子、桃子、甜椒、土豆、树莓、大豆、西葫芦、草莓、番茄"
    }
    
    # 保存更新后的版本信息
    with open(version_path, 'w') as f:
        json.dump(version_info, f, indent=2, ensure_ascii=False)
        
    return str(model_path)

def main():
    """训练植物分类器的主函数"""
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 加载配置
        config = load_classifier_config(args.config)
        
        # 从命令行参数更新配置
        config = update_config_from_args(config, args)
        
        # 获取ConfigManager实例
        config_manager = ConfigManager()
        
        # 设置随机种子
        seed = config.get("seed", config_manager.get("seed", 42, "train"))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 使用setup_experiment_folder函数创建实验目录
        # 替换原来直接创建目录的代码
        experiment_dir = setup_experiment_folder(config)
        
        # 准备数据加载器
        train_loader, val_loader = prepare_data(config)
        
        # 创建模型
        model = create_model(config)
        
        # 训练模型
        train_model(model, train_loader, val_loader, config, experiment_dir)
        
        logging.info("训练成功完成!")
        
    except Exception as e:
        logging.error(f"训练过程中出现错误: {e}", exc_info=True)
        sys.exit(1)
        
if __name__ == "__main__":
    main()