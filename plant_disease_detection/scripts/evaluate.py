# scripts/evaluate.py
#独立评估植物分类和病害检测模型工具

import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 然后再导入模块

import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

from src.utils.config_manager import ConfigManager
from src.models import PlantClassifier, DiseaseDetector
from src.data_processing.dataset import ClassificationDataset, DetectionDataset
from src.evaluation.evaluator import ClassificationEvaluator, DetectionEvaluator

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估植物分类/病害检测模型')
    parser.add_argument('--model_path', required=True, help='模型权重文件路径')
    parser.add_argument('--data_dir', required=True, help='测试数据目录')
    parser.add_argument('--task_type', choices=['classification', 'detection'], 
                       required=True, help='任务类型')
    parser.add_argument('--output_dir', default=None, help='评估结果输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化结果')
    parser.add_argument('--explain', action='store_true', help='是否生成模型解释可视化')
    return parser.parse_args()

def detection_collate_fn(batch):
    """
    自定义的检测任务数据批处理函数
    处理可变大小的目标（边界框、标签）
    
    Args:
        batch: 元组列表 (图像, 目标)
        
    Returns:
        批处理后的图像和目标
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
        
    return images, targets

def setup_logging(output_dir):
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 获取配置管理器
    config_manager = ConfigManager()
    
    # 准备输出目录
    if args.output_dir is None:
        args.output_dir = config_manager.get('EVALUATION_OUTPUT_DIR', 'evaluation_results', "data")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(args.output_dir)
    
    # 加载模型
    logging.info(f"正在加载模型: {args.model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.task_type == 'classification':
        # 创建默认配置字典
        config = {
            "backbone": "resnet50",
            "num_classes": 38,  # PlantVillage完整分类数
            "pretrained": False  # 评估阶段不需要预训练权重
        }
        
        # 尝试从模型相邻的config文件加载配置(如果存在)
        config_path = Path(args.model_path).parent / "plant_classifier_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logging.info(f"已加载模型配置: {config_path}")
            except Exception as e:
                logging.warning(f"加载模型配置失败: {e}，使用默认配置")
        
        model = PlantClassifier(config)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluator = ClassificationEvaluator(model, device)
        
        # 加载数据集
        dataset = ClassificationDataset(
            root_dir=args.data_dir,
            mode='test',
            transform_name='test'
        )
    else:  # detection
        # 检测器配置
        config = {
            "num_classes": 38,  # 数据集类别数 
            "backbone": "resnet50"
        }
        
        # 尝试加载模型配置
        config_path = Path(args.model_path).parent / "disease_detector_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config.update(json.load(f))
                logging.info(f"已加载模型配置: {config_path}")
            except Exception as e:
                logging.warning(f"加载模型配置失败: {e}，使用默认配置")
        
        model = DiseaseDetector(config)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluator = DetectionEvaluator(model, device)
        
        # 加载数据集
        dataset = DetectionDataset(
            root_dir=args.data_dir,
            mode='test',
            transform_name='test'
        )
    
    model = model.to(device)
    
    # 如果是检测任务，添加特殊的收集函数
    collate_fn = detection_collate_fn if args.task_type == 'detection' else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # 执行评估
    logging.info("开始评估...")
    results = evaluator.evaluate(data_loader, args.output_dir)
    
    # 打印结果摘要
    logging.info("评估完成，关键指标:")
    for k, v in results.items():
        if isinstance(v, (int, float)):
            logging.info(f"{k}: {v:.4f}")
    
    logging.info(f"详细结果已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()