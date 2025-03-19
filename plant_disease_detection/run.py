# run.py - 植物病害检测系统入口文件

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import json
from datetime import datetime
import pickle
from typing import Dict, Any

from flask import Flask, render_template, current_app

from src.api import create_app
from src.api.routes import api_bp
from src.utils.config_manager import ConfigManager
from src.utils.logging_utils import setup_logging
from src.models import PlantClassifier, DiseaseDetector
from src.inference.predictor import create_predictor, DummyPredictor
from src.inference.result_parser import create_result_parser

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

# 添加路径标准化函数
def normalize_path(path_str):
    """标准化路径，确保路径分隔符兼容当前操作系统"""
    return str(Path(path_str.replace('\\', '/')))

def create_app(config: Dict[str, Any] = None) -> Flask:
    """创建Flask应用实例"""
    app = Flask(__name__)
    
    # 应用配置
    app.config.update(config or {})
    
    # 注册蓝图
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # 设置路由
    @app.route('/')
    def index():
        return render_template('index.html')
    
    return app

# 修改 load_models 函数

def load_models():
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = None
    detector = None
    
    # 加载分类模型
    try:
        classifier_path = normalize_path(os.path.join('models', 'plant_classifier.pth'))
        config_path = normalize_path(os.path.join('models', 'plant_classifier_config.json'))
        
        logging.info(f"加载分类模型: {classifier_path}")
        
        # 检查文件存在
        if not os.path.exists(classifier_path):
            logging.warning(f"分类模型文件不存在: {classifier_path}")
            
        if not os.path.exists(config_path):
            logging.warning(f"分类模型配置文件不存在: {config_path}，将使用默认配置")
            model_config = {"backbone": "resnet50", "num_classes": 38}
        else:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                logging.info(f"已加载分类模型配置: {model_config}")
            except json.JSONDecodeError:
                logging.warning(f"配置文件格式错误，将使用默认配置")
                model_config = {"backbone": "resnet50", "num_classes": 38}
            
        # 创建模型并加载权重
        classifier = PlantClassifier(config=model_config)
        state_dict = torch.load(classifier_path, map_location=device)
        
        # 处理state_dict可能有所有键都带有module前缀的情况(DataParallel训练的模型)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            # 如果保存的是checkpoint格式，提取模型状态
            state_dict = state_dict['model_state_dict']
            
        if all(k.startswith('module.') for k in state_dict.keys()):
            # 创建新的state_dict，移除'module.'前缀
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        classifier.load_state_dict(state_dict)
        classifier.eval()
        classifier.to(device)
        logging.info("分类模型加载成功")
    except Exception as e:
        logging.error(f"分类模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 加载检测模型 (类似处理)
    try:
        detector_path = normalize_path(os.path.join('models', 'disease_detector.pth'))
        config_path = normalize_path(os.path.join('models', 'disease_detector_config.json'))
        
        logging.info(f"加载检测模型: {detector_path}")
        
        # 检查文件存在
        if not os.path.exists(detector_path):
            logging.warning(f"检测模型文件不存在: {detector_path}")
            
        if not os.path.exists(config_path):
            logging.warning(f"检测模型配置文件不存在: {config_path}，将使用默认配置")
            model_config = {"backbone": "resnet50", "num_classes": 38}
        else:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                logging.info(f"已加载检测模型配置: {model_config}")
            except json.JSONDecodeError:
                logging.warning(f"配置文件格式错误，将使用默认配置")
                model_config = {"backbone": "resnet50", "num_classes": 38}
        
        # 创建模型并加载权重
        detector = DiseaseDetector(config=model_config)
        state_dict = torch.load(detector_path, map_location=device)
        
        # 处理state_dict可能有所有键都带有module前缀的情况
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            # 如果保存的是checkpoint格式，提取模型状态
            state_dict = state_dict['model_state_dict']
            
        if all(k.startswith('module.') for k in state_dict.keys()):
            # 创建新的state_dict，移除'module.'前缀
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        detector.load_state_dict(state_dict)
        detector.eval()
        detector.to(device)
        logging.info("检测模型加载成功")
    except Exception as e:
        logging.error(f"检测模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return classifier, detector


def check_model_versions(config_manager):
    """
    检查模型版本信息
    
    Args:
        config_manager: 配置管理器实例
        
    Returns:
        模型版本信息字典
    """
    model_dir = config_manager.get('MODEL_DIR', 'models', "model")
    version_file = os.path.join(model_dir, 'version_info.json')
    
    if (os.path.exists(version_file)):
        try:
            with open(version_file, 'r', encoding='utf-8') as f:
                versions = json.load(f)
                logger.info("当前模型版本信息：")
                for model_name, info in versions.items():
                    logger.info(f"  - {model_name}: v{info.get('version', '未知')}, "
                              f"更新时间: {info.get('date', '未知')}")
                return versions
        except Exception as e:
            logger.error(f"读取版本信息失败: {e}")
    else:
        logger.info("未找到模型版本信息文件，将使用默认配置")
    
    # 如果没有版本文件，创建一个默认的
    default_versions = {
        'classifier': {
            'version': '1.0.0',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'description': '初始版本'
        },
        'detector': {
            'version': '1.0.0',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'description': '初始版本'
        }
    }
    
    # 保存默认版本信息
    os.makedirs(model_dir, exist_ok=True)
    with open(version_file, 'w', encoding='utf-8') as f:
        json.dump(default_versions, f, indent=2, ensure_ascii=False)
    
    return default_versions


def export_model(model, model_path, format='onnx'):
    """
    将模型导出为其他格式
    
    Args:
        model: 要导出的模型
        model_path: 模型原始路径
        format: 导出格式，默认为ONNX
        
    Returns:
        是否成功导出
    """
    if format != 'onnx':
        logger.warning(f"目前仅支持ONNX格式导出")
        return False
    
    try:
        # 检查是否安装了onnx
        import onnx
        
        # 准备输入张量 (批次大小=1, RGB图像, 224x224)
        dummy_input = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
        
        # 确定输出文件名
        onnx_path = os.path.splitext(model_path)[0] + '.onnx'
        
        # 导出模型
        logger.info(f"导出ONNX模型到: {onnx_path}")
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # 验证导出模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info("ONNX模型验证通过")
        return True
    
    except ImportError:
        logger.error("未安装ONNX库，请使用 'pip install onnx' 安装")
        return False
    except Exception as e:
        logger.error(f"导出ONNX模型失败: {e}")
        return False


# 在适当位置添加/修改预测相关代码

def create_predictor():
    """创建预测器"""
    # 如果没有加载模型，返回一个虚拟预测器
    if not hasattr(current_app, 'classifier') and not hasattr(current_app, 'detector'):
        from src.inference.predictor import DummyPredictor
        return DummyPredictor()
    
    from src.inference.predictor import ClassificationPredictor, DetectionPredictor
    
    # 创建组合预测器，同时支持分类和检测
    class CombinedPredictor:
        def __init__(self, classifier, detector):
            self.classifier = classifier
            self.detector = detector
            
        def classify(self, image_bytes):
            """分类预测"""
            import io
            import numpy as np
            from PIL import Image
            import torch
            from torchvision import transforms
            
            # 从字节流加载图像
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # 使用分类模型进行预测
            if self.classifier:
                # 执行预处理，将PIL图像转换为Tensor
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                # 应用转换并添加批次维度
                image_tensor = transform(image).unsqueeze(0)
                
                # 将tensor移到与模型相同的设备上
                device = next(self.classifier.parameters()).device
                image_tensor = image_tensor.to(device)
                
                # 执行预测
                with torch.no_grad():
                    outputs = self.classifier(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    
                    # 获取最可能的类别
                    score, class_idx = torch.max(probabilities, dim=0)
                    
                    # 获取类名
                    class_names = getattr(current_app, 'class_names', [f"类别{i}" for i in range(10)])
                    class_name = class_names[class_idx.item()] if class_idx.item() < len(class_names) else f"类别{class_idx.item()}"
                    
                    # 构建结果
                    predictions = []
                    for i, prob in enumerate(probabilities):
                        class_n = class_names[i] if i < len(class_names) else f"类别{i}"
                        predictions.append({
                            "class_name": class_n,
                            "confidence": prob.item()
                        })
                    
                    result = {
                        "predictions": sorted(predictions, key=lambda x: x["confidence"], reverse=True),
                        "top_result": class_name,
                        "confidence": score.item()
                    }
                    
                    return result
            else:
                return {"error": "分类模型未加载", "predictions": []}
        
        def detect(self, image_bytes):
            """检测预测"""
            import io
            from PIL import Image
            
            # 从字节流加载图像
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # 使用检测模型进行预测
            if self.detector:
                class_names = getattr(current_app, 'disease_class_names', [f"病害{i}" for i in range(5)])
                predictor = DetectionPredictor(self.detector, class_names)
                return predictor.predict(image)
            else:
                return {"error": "检测模型未加载", "detections": []}
    
    # 返回组合预测器
    return CombinedPredictor(
        getattr(current_app, 'classifier', None),
        getattr(current_app, 'detector', None)
    )


def load_model_config(config_path):
    """使用UTF-8编码加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return None


def main():
    """主函数，处理命令行参数并执行对应功能"""
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='植物病害检测系统')
    parser.add_argument('--mode', choices=['serve', 'predict', 'export'], default='serve',
                      help='运行模式：serve-启动服务，predict-单图预测，export-导出模型')
    parser.add_argument('--config', type=str, help='自定义配置文件路径')
    parser.add_argument('--task', choices=['classification', 'detection'], 
                      help='任务类型：分类或检测（用于predict和export模式）')
    parser.add_argument('--image', type=str, help='要预测的图像路径（predict模式使用）')
    parser.add_argument('--output', type=str, help='输出路径（predict和export模式使用）')
    parser.add_argument('--port', type=int, help='API服务端口（serve模式使用）')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--classifier-config', type=str, help='分类器配置文件路径')
    parser.add_argument('--detector-config', type=str, help='检测器配置文件路径')
    
    args = parser.parse_args()
    
    # 加载自定义配置
    if args.config and os.path.exists(args.config):
        config_manager.load_from_file(args.config)
        logger.info(f"从 {args.config} 加载配置")
    
    # 加载模型前，先加载对应配置
    if args.classifier_config and os.path.exists(args.classifier_config):
        config_manager.load_from_file(args.classifier_config)
        logger.info(f"从 {args.classifier_config} 加载分类器配置")
    elif os.path.exists("configs/classifier_training.yaml"):
        config_manager.load_from_file("configs/classifier_training.yaml")
        logger.info("从默认路径加载分类器配置")

    if args.detector_config and os.path.exists(args.detector_config):
        config_manager.load_from_file(args.detector_config)
        logger.info(f"从 {args.detector_config} 加载检测器配置")
    elif os.path.exists("configs/detector_training.yaml"):
        config_manager.load_from_file("configs/detector_training.yaml")
        logger.info("从默认路径加载检测器配置")
    
    # 检查模型版本
    versions = check_model_versions(config_manager)
    
    # 根据运行模式执行不同操作
    if args.mode == 'serve':
        # 加载模型
        classifier, detector = load_models()
        if not classifier and not detector:
            logger.error("未能加载任何模型，无法启动服务")
            return
        
        # 创建应用（不传递任何参数）
        app = create_app()
        # 设置应用属性
        app.config_manager = config_manager
        app.classifier = classifier
        app.detector = detector
        
        # 添加类别名称列表
        # 尝试从配置或模型中加载实际的类名
        class_names_path = normalize_path(os.path.join('models', 'plant_classes.json'))
        disease_names_path = normalize_path(os.path.join('models', 'disease_classes.json'))
        plant_mappings_path = normalize_path(os.path.join('models', 'plant_mappings.json'))
        disease_mappings_path = normalize_path(os.path.join('models', 'disease_mappings.json'))

        try:
            # 修改这里，确保正确加载每个文件
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r', encoding='utf-8') as f:
                    app.plant_class_names = json.load(f)
                    logger.info(f"从{class_names_path}加载了{len(app.plant_class_names)}个植物类别")
                    
            if os.path.exists(disease_names_path):
                with open(disease_names_path, 'r', encoding='utf-8') as f:
                    app.disease_class_names = json.load(f)
                    logger.info(f"从{disease_names_path}加载了{len(app.disease_class_names)}个病害类别")
                    
            if os.path.exists(plant_mappings_path):
                with open(plant_mappings_path, 'r', encoding='utf-8') as f:
                    app.plant_mappings = json.load(f)
                    logger.info(f"加载了{len(app.plant_mappings)}个植物名称映射")
                    
            if os.path.exists(disease_mappings_path):
                with open(disease_mappings_path, 'r', encoding='utf-8') as f:
                    app.disease_mappings = json.load(f)
                    logger.info(f"加载了{len(app.disease_mappings)}个病害名称映射")
        except Exception as e:
            logger.error(f"加载映射文件时出错: {e}")
            app.plant_class_names = {}
            app.disease_class_names = {}
            app.plant_mappings = {}
            app.disease_mappings = {}
        
        # 获取服务配置
        host = config_manager.get('API_HOST', '0.0.0.0', "api")
        port = args.port or config_manager.get('API_PORT', 5000, "api")
        debug = args.debug or config_manager.get('DEBUG', False, "api")
        
        # 启动服务
        logger.info(f"启动API服务，监听地址: {host}:{port}")
        app.run(host=host, port=port, debug=debug)
    
    # 修改 predict 模式部分 (第239行左右)
    elif args.mode == 'predict':
        # 验证参数...
        
        # 加载模型
        classifier, detector = load_models()  # 修改这里
        
        # 根据任务类型选择模型
        model = classifier if args.task == 'classification' else detector
        class_names = ["类别1", "类别2", "类别3"]  # 这里应该从某处获取类名，临时用列表替代
        
        if (args.task == 'classification' and not classifier) or \
           (args.task == 'detection' and not detector):
            logger.error(f"未加载{args.task}模型，无法执行预测")
            return
        
        try:
            # 创建预测器
            predictor = create_predictor(
                model,  # 使用选择的模型
                args.task, 
                class_names
            )
            
            # 加载图像并预测
            from PIL import Image
            image = Image.open(args.image).convert('RGB')
            result = predictor.predict(image)
            
            # 解析结果
            parser = create_result_parser(args.task)
            parsed_result = parser.parse_result(result)
            
            # 输出结果
            print(json.dumps(parsed_result, indent=2, ensure_ascii=False))
            
            # 保存结果
            if args.output:
                output_path = args.output
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = config_manager.get('PREDICTION_OUTPUT_DIR', 'results', "data")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'{args.task}_{timestamp}.json')
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"预测结果已保存至: {output_path}")
            
        except Exception as e:
            logger.error(f"预测过程发生错误: {e}")
    
    # 修改 export 模式部分 (第290行左右)
    elif args.mode == 'export':
        # 验证参数...
        
        # 加载模型
        classifier, detector = load_models()  # 修改这里
        
        # 根据任务类型选择模型
        model = classifier if args.task == 'classification' else detector
        
        if (args.task == 'classification' and not classifier) or \
           (args.task == 'detection' and not detector):
            logger.error(f"未加载{args.task}模型，无法执行导出")
            return
        
        # 获取模型路径
        model_dir = normalize_path(config_manager.get('MODEL_DIR', 'models', "model"))
        model_filename = 'plant_classifier.pth' if args.task == 'classification' else 'disease_detector.pth'
        model_path = normalize_path(os.path.join(model_dir, model_filename))
        
        # 导出模型
        success = export_model(model, model_path)  # 使用选择的模型
        
        # 其余代码...
        if success:
            logger.info(f"{args.task}模型已成功导出为ONNX格式")
        else:
            logger.error(f"导出{args.task}模型失败")


if __name__ == '__main__':
    main()