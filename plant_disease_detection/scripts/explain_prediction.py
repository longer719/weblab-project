#sripts/explain_prediction.py
"""
Grad-CAM可视化脚本
用于生成模型决策的可视化解释
"""

import os
import sys
import argparse
import logging
import torch
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_manager import ConfigManager
from src.evaluation.model_interpreter import ModelInterpreter
from src.models import PlantClassifier, DiseaseDetector

def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成模型预测的Grad-CAM可视化解释')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='模型和配置文件目录')
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'detection'],
                        help='模型任务类型')
    parser.add_argument('--image', type=str, required=True,
                        help='要解释的输入图像路径')
    parser.add_argument('--output-dir', type=str, default='results/gradcam',
                        help='保存Grad-CAM图像的目录')
    parser.add_argument('--target-class', type=int, default=None,
                        help='要解释的目标类别ID (可选，默认使用预测概率最高的类别)')
    parser.add_argument('--target-layer', type=str, default=None,
                        help='要分析的目标卷积层名称 (可选，默认使用backbone.layer4)')
    return parser.parse_args()

def load_class_mapping(model_dir, task):
    """加载类别映射"""
    mapping_file = None
    if task == 'classification':
        mapping_file = os.path.join(model_dir, 'plant_classes.json')
    elif task == 'detection':
        mapping_file = os.path.join(model_dir, 'disease_classes.json')
    
    if mapping_file and os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载类别映射失败: {str(e)}")
    
    logging.warning(f"找不到类别映射文件，将使用类别索引")
    return None

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    setup_logging()
    
    # 检查图像是否存在
    if not os.path.exists(args.image):
        logging.error(f"图像文件不存在: {args.image}")
        return
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 加载配置管理器
    config_manager = ConfigManager()
    
    # 加载类别映射
    class_mapping = load_class_mapping(args.model_dir, args.task)
    
    # 加载模型
    model = None
    if args.task == 'classification':
        # 尝试加载分类器配置和模型
        config_path = os.path.join(args.model_dir, 'plant_classifier_config.json')
        model_path = os.path.join(args.model_dir, 'plant_classifier.pth')
        
        if not os.path.exists(model_path):
            logging.error(f"模型文件不存在: {model_path}")
            return
        
        # 读取模型配置
        model_config = {'num_classes': 38, 'backbone': 'resnet50'}  # 默认配置
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config.update(json.load(f))
            except Exception as e:
                logging.warning(f"加载模型配置失败: {str(e)}，使用默认配置")
        
        # 创建模型并加载权重
        model = PlantClassifier(config=model_config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"已加载分类器模型: {model_path}")
        
    elif args.task == 'detection':
        # 加载检测器
        config_path = os.path.join(args.model_dir, 'disease_detector_config.json')
        model_path = os.path.join(args.model_dir, 'disease_detector.pth')
        
        if not os.path.exists(model_path):
            logging.error(f"模型文件不存在: {model_path}")
            return
        
        # 读取模型配置
        model_config = {'num_classes': 38, 'backbone': 'resnet50'}  # 默认配置
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config.update(json.load(f))
            except Exception as e:
                logging.warning(f"加载模型配置失败: {str(e)}，使用默认配置")
        
        # 创建模型并加载权重
        model = DiseaseDetector(config=model_config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"已加载检测器模型: {model_path}")
    
    if model is None:
        logging.error("模型加载失败")
        return
    
    # 将模型设置为评估模式
    model.eval()
    
    # 创建模型解释器
    interpreter = ModelInterpreter(model)
    
    # 准备输出路径
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    output_filename = f"gradcam_{args.task}_{image_name}"
    if args.target_class is not None:
        output_filename += f"_class{args.target_class}"
    output_filename += ".png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # 生成并保存Grad-CAM可视化
    logging.info(f"正在为图像 '{args.image}' 生成Grad-CAM可视化...")
    try:
        interpreter.explain_prediction(
            image=args.image,
            class_idx=args.target_class,
            target_layer=args.target_layer,
            output_path=output_path,
            class_names=class_mapping
        )
        logging.info(f"已生成Grad-CAM可视化并保存到: {output_path}")
    except Exception as e:
        logging.error(f"生成Grad-CAM可视化失败: {str(e)}")

if __name__ == "__main__":
    main()