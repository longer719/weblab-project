# src/inference/predictor.py

import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Union, Optional, Any, Tuple
from pathlib import Path

from src.utils.config_manager import ConfigManager
from torchvision import transforms

import logging
from src.models import PlantClassifier, DiseaseDetector
from src.utils.image_utils import load_image, preprocess_image
from src.data_processing.transforms import get_transform

logger = logging.getLogger(__name__)

class BasePredictor:
    """基础模型预测器"""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        初始化预测器
        
        Args:
            model: 要用于预测的模型
            device: 运行设备，默认自动选择
        """
        self.model = model
        self.config_manager = ConfigManager()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        预处理输入图像
        
        Args:
            image: 输入图像路径或数组
            
        Returns:
            预处理后的图像张量
        """
        from src.data_processing.transforms import get_transform
        transform = get_transform(train=False)
        
        # 处理不同类型的输入
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError(f"不支持的图像类型: {type(image)}")
            
        # 应用变换
        image_tensor = transform(image)
        
        # 添加批次维度
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        return image_tensor
    
    def batch_preprocess(self, images: List[Union[str, np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        批量预处理图像
        
        Args:
            images: 图像路径或数组的列表
            
        Returns:
            批量预处理后的图像张量
        """
        batch_tensors = []
        for img in images:
            tensor = self.preprocess_image(img)
            batch_tensors.append(tensor)
            
        return torch.cat(batch_tensors, dim=0)
        
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        对单个图像进行预测
        
        Args:
            image: 输入图像路径或数组
            
        Returns:
            预测结果字典
        """
        # 子类必须实现
        raise NotImplementedError("子类必须实现predict方法")
    
    def batch_predict(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        """
        对批量图像进行预测
        
        Args:
            images: 输入图像路径或数组列表
            
        Returns:
            预测结果字典列表
        """
        # 预处理所有图像
        batch_tensors = self.batch_preprocess(images)
        
        # 进行批量预测
        results = []
        batch_size = self.config_manager.get('INFERENCE_BATCH_SIZE', 16, "model")
        
        for i in range(0, len(batch_tensors), batch_size):
            batch = batch_tensors[i:i+batch_size].to(self.device)
            batch_results = self._predict_batch(batch)
            results.extend(batch_results)
            
        return results
    
    def _predict_batch(self, batch: torch.Tensor) -> List[Dict[str, Any]]:
        """
        对张量批次进行预测
        
        Args:
            batch: 批次张量
            
        Returns:
            预测结果字典列表
        """
        # 子类必须实现
        raise NotImplementedError("子类必须实现_predict_batch方法")


class ClassificationPredictor(BasePredictor):
    """分类模型预测器"""
    
    def __init__(self, model: nn.Module, class_names: Union[List[str], Dict[str, str]], 
                 device: Optional[torch.device] = None):
        """
        初始化分类预测器
        
        Args:
            model: 分类模型
            class_names: 类别名称列表或字典（ID->名称映射）
            device: 运行设备
        """
        super().__init__(model, device)
        self.class_names = class_names
        self.is_dict_mapping = isinstance(class_names, dict)
        
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        对单个图像进行分类预测
        
        Args:
            image: 输入图像
            
        Returns:
            分类预测结果
        """
        # 预处理图像
        image_tensor = self.preprocess_image(image).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            # 获取最可能的类别
            score, class_idx = torch.max(probabilities, dim=0)
            
        # 获取类别名称 - 修改这里的逻辑
        if self.is_dict_mapping:
            class_name = self.class_names.get(str(class_idx.item()), f"未知类别{class_idx.item()}")
        else:
            class_name = self.class_names[class_idx.item()] if class_idx.item() < len(self.class_names) else f"类别{class_idx.item()}"
        
        # 构建结果字典
        result = {
            'class_id': class_idx.item(),
            'class_name': class_name,
            'confidence': score.item(),
            'probabilities': probabilities.cpu().numpy().tolist(),
            'top_classes': self._get_top_predictions(probabilities.cpu().numpy(), k=3)
        }
        
        return result
    
    def _predict_batch(self, batch: torch.Tensor) -> List[Dict[str, Any]]:
        """
        对批次进行分类预测
        
        Args:
            batch: 图像批次
            
        Returns:
            分类预测结果列表
        """
        with torch.no_grad():
            outputs = self.model(batch)
            batch_probabilities = torch.softmax(outputs, dim=1)
            
        results = []
        for probabilities in batch_probabilities:
            score, class_idx = torch.max(probabilities, dim=0)
            
            if self.is_dict_mapping:
                class_name = self.class_names.get(str(class_idx.item()), f"未知类别{class_idx.item()}")
            else:
                class_name = self.class_names[class_idx.item()] if class_idx.item() < len(self.class_names) else f"类别{class_idx.item()}"
            
            result = {
                'class_id': class_idx.item(),
                'class_name': class_name,
                'confidence': score.item(),
                'probabilities': probabilities.cpu().numpy().tolist(),
                'top_classes': self._get_top_predictions(probabilities.cpu().numpy(), k=3)
            }
            results.append(result)
            
        return results
    
    def _get_top_predictions(self, probabilities: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """
        获取概率最高的k个预测
        
        Args:
            probabilities: 类别概率数组
            k: 返回的预测数量
            
        Returns:
            最可能k个类别的列表
        """
        # 获取前k个预测
        k = min(k, len(probabilities))
        top_indices = np.argsort(probabilities)[::-1][:k]
        
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                'class_id': int(idx),
                'class_name': self.class_names[idx],
                'probability': float(probabilities[idx])
            })
            
        return top_predictions


class DetectionPredictor(BasePredictor):
    """检测模型预测器，现在强化其分类能力"""
    
    def __init__(self, model: nn.Module, class_names: Union[List[str], Dict[str, str]], 
                score_threshold: float = 0.5, device: Optional[torch.device] = None):
        """
        初始化检测预测器
        
        Args:
            model: 检测模型
            class_names: 类别名称列表或字典（ID->名称映射）
            score_threshold: 检测阈值
            device: 运行设备
        """
        super().__init__(model, device)
        self.class_names = class_names
        self.is_dict_mapping = isinstance(class_names, dict)
        self.score_threshold = self.config_manager.get(
            'DETECTION_SCORE_THRESHOLD', score_threshold, "model")
        logger.info(f"检测阈值设置为: {self.score_threshold}")
        
    def predict(self, image: Union[str, np.ndarray, Image.Image], plant_type: str = None) -> Dict[str, Any]:
        """
        执行"伪检测"（图像级分类），返回最高置信度的预测和原始检测信息。
        """
        logger.info("--- Entering DetectionPredictor.predict ---") # Log entry
        try:
            # --- 修改开始: 根据输入类型分别处理 ---
            # 1. 确保输入是 PIL Image
            if isinstance(image, str) or isinstance(image, Path):
                original_image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                original_image = Image.fromarray(image).convert('RGB') # 假设是RGB numpy数组
            elif isinstance(image, Image.Image):
                original_image = image # 已经是 PIL Image
            else:
                 raise TypeError(f"Predictor不支持的图像输入类型: {type(image)}")

            original_width, original_height = original_image.size
            logger.info(f"Original image size: {original_width}x{original_height}")

            # 2. 直接定义推理转换管道，跳过ToPILImage步骤
            inference_transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # 3. 直接对PIL图像应用转换
            image_tensor = inference_transform(original_image).unsqueeze(0).to(self.device)
            logger.info(f"Image transformed for inference, tensor shape: {image_tensor.shape}")
            # --- 修改结束 ---

            # --- Model Inference ---
            self.model.eval() # Ensure eval mode
            with torch.no_grad():
                logger.info("Calling model forward pass...")
                # FasterRCNN需要图像列表而非批次
                outputs = self.model([image_tensor[0]]) # 提取单张图像并封装为列表
                logger.info("Model forward pass completed.")
            # --- End Model Inference ---

            # 3. 解析结果 - 重点是找到最高置信度的分类结果
            top_prediction = {"class_name": "未知", "confidence": 0.0, "label_id": -1}
            detections = [] # 保留原始检测列表
            
            # 第一个(也是唯一的)图像的预测结果
            if outputs and len(outputs) > 0:
                output = outputs[0]
                boxes = output.get('boxes', torch.tensor([])).cpu().numpy()
                scores = output.get('scores', torch.tensor([])).cpu().numpy()
                labels = output.get('labels', torch.tensor([])).cpu().numpy()
                
                logger.info(f"Raw model outputs - scores: {scores.tolist()}, labels: {labels.tolist()}")
                
                if len(scores) > 0:
                    # 找到最高分数的索引
                    top_idx = np.argmax(scores)
                    top_score = scores[top_idx]
                    top_label = labels[top_idx]
                    
                    # 这里的class_id直接用模型输出的标签
                    class_id = top_label
                    
                    logger.info(f"Highest score prediction - score: {top_score:.4f}, label: {top_label}, used class_id: {class_id}")
                    
                    # 检查映射是否可用
                    logger.info(f"self.is_dict_mapping: {self.is_dict_mapping}, class_names type: {type(self.class_names)}")
                    logger.info(f"Available class IDs: {list(self.class_names.keys()) if self.is_dict_mapping else 'Not a dict'}")
                    
                    # 使用类别映射获取名称
                    if self.is_dict_mapping:
                        class_name = self.class_names.get(str(class_id), f"未知类别{class_id}")
                    elif isinstance(self.class_names, list):
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"类别{class_id}"
                    else:
                        class_name = f"未知类别{class_id}"
                    
                    logger.info(f"Looked up class name: {class_name}")
                    
                    top_prediction = {
                        "class_name": class_name,
                        "confidence": float(top_score),
                        "label_id": int(top_label)
                    }
                else:
                    logger.warning("Model output contained 0 detections/scores.")
            else:
                logger.warning("Model did not return any output.")
            
            # 筛选掉低置信度的检测结果用于详细显示
            keep = scores >= self.score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # 构建原始检测结果
            for box, score, label in zip(boxes, scores, labels):
                class_id = label
                
                if self.is_dict_mapping:
                    class_name = self.class_names.get(str(class_id), f"未知类别{class_id}")
                elif isinstance(self.class_names, list):
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"类别{label}"
                else:
                    class_name = f"未知类别{class_id}"
                
                # 添加检测结果
                detections.append({
                    'box': box.tolist(),
                    'score': float(score),
                    'class_id': int(label),
                    'class_name': class_name,
                    'plant_type': plant_type  # 添加植物类型信息
                })
            
            # 4. 构建最终返回字典
            result = {
                'top_prediction': top_prediction,
                'detections': detections,
                'plant_type': plant_type,
                'image_size': {
                    'width': original_width,
                    'height': original_height
                }
            }
            
            logger.info(f"--- Exiting DetectionPredictor.predict, returning: {result} ---")
            return result
            
        except Exception as e_pred:
            logger.error(f"Error inside DetectionPredictor.predict: {e_pred}", exc_info=True)
            # 发生错误时返回默认结构
            return {
                'top_prediction': {"class_name": "预测内部错误", "confidence": 0.0, "label_id": -1},
                'detections': [],
                'plant_type': plant_type,
                'image_size': {
                    'width': original_width if 'original_width' in locals() else 0,
                    'height': original_height if 'original_height' in locals() else 0
                }
            }
    
    def _predict_batch(self, batch: torch.Tensor, plant_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        对批次进行检测预测
        
        Args:
            batch: 图像批次
            plant_types: 对应每个图像的植物类型
            
        Returns:
            检测结果列表
        """
        # 进行批量预测
        with torch.no_grad():
            outputs = self.model(batch)
            
        results = []
        for i, output in enumerate(outputs):
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            # 获取当前图像的植物类型（如果提供）
            current_plant_type = plant_types[i] if plant_types and i < len(plant_types) else None
            
            # 筛选掉低置信度的检测结果
            keep = scores >= self.score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # 构建单个图像的结果
            detections = []
            for box, score, label in zip(boxes, scores, labels):
                # 获取类别名称
                class_name = self.class_names[label-1] if label-1 < len(self.class_names) else f"类别{label}"
                
                # 添加检测结果
                detections.append({
                    'box': box.tolist(),
                    'score': float(score),
                    'class_id': int(label),
                    'class_name': class_name,
                    'plant_type': current_plant_type
                })
            
            result = {
                'detections': detections,
                'plant_type': current_plant_type,
                'image_size': {
                    'width': batch.shape[3],
                    'height': batch.shape[2]
                }
            }
            results.append(result)
            
        return results


def create_predictor(model_path: str, task_type: str, class_names: List[str]) -> Union[ClassificationPredictor, DetectionPredictor]:
    """
    创建适合任务类型的预测器
    
    Args:
        model_path: 模型路径
        task_type: 任务类型('classification' 或 'detection')
        class_names: 类别名称列表
        
    Returns:
        预测器实例
    """
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    if task_type == 'classification':
        from src.models import PlantClassifier
        model = PlantClassifier()
        model.load_state_dict(torch.load(model_path, map_location=device))
        return ClassificationPredictor(model, class_names, device)
    elif task_type == 'detection':
        from src.models import DiseaseDetector
        model = DiseaseDetector()
        model.load_state_dict(torch.load(model_path, map_location=device))
        return DetectionPredictor(model, class_names, device=device)
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")


class ModelPredictor:
    """模型预测器类，用于加载模型并进行推理"""
    
    def __init__(self, model_dir='models'):
        """
        初始化预测器
        
        Args:
            model_dir: 模型目录
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载分类器模型
        self.classifier = self._load_classifier()
        
        # 加载检测器模型
        self.detector = self._load_detector()
        
        # 加载类别映射
        self._load_class_mappings()
        
    def _load_classifier(self):
        """加载分类器模型"""
        model_path = self.model_dir / 'plant_classifier.pth'
        config_path = self.model_dir / 'plant_classifier_config.json'
        
        if not model_path.exists():
            logger.warning(f"分类器模型文件不存在: {model_path}")
            return None
            
        if not config_path.exists():
            logger.warning(f"分类器配置文件不存在: {config_path}")
            return None
            
        try:
            classifier = PlantClassifier.from_config_file(str(config_path))
            classifier.load_state_dict(torch.load(str(model_path), map_location=self.device))
            classifier.to(self.device)
            classifier.eval()
            logger.info("分类器模型加载成功")
            return classifier
        except Exception as e:
            logger.error(f"加载分类器模型失败: {str(e)}")
            return None
    
    def _load_detector(self):
        """加载检测器模型"""
        model_path = self.model_dir / 'disease_detector.pth'
        config_path = self.model_dir / 'disease_detector_config.json'
        
        if not model_path.exists():
            logger.warning(f"检测器模型文件不存在: {model_path}")
            return None
            
        if not config_path.exists():
            logger.warning(f"检测器配置文件不存在: {config_path}")
            return None
            
        try:
            detector = DiseaseDetector.from_config_file(str(config_path))
            detector.load_state_dict(torch.load(str(model_path), map_location=self.device))
            detector.to(self.device)
            detector.eval()
            logger.info("检测器模型加载成功")
            return detector
        except Exception as e:
            logger.error(f"加载检测器模型失败: {str(e)}")
            return None
    
    def _load_class_mappings(self):
        """加载类别映射"""
        # 植物种类映射
        try:
            mapping_path = self.model_dir / 'plant_classes.json'
            if mapping_path.exists():
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.plant_classes = json.load(f)
                logger.info(f"从{mapping_path}加载了{len(self.plant_classes)}个植物类别")
            else:
                self.plant_classes = {str(i): f"植物类别{i}" for i in range(10)}
                logger.warning(f"未找到植物类别映射文件，使用默认映射")
        except Exception as e:
            logger.error(f"加载植物类别映射失败: {e}")
            self.plant_classes = {str(i): f"植物类别{i}" for i in range(10)}
        
        # 病害种类映射
        try:
            mapping_path = self.model_dir / 'disease_classes.json'
            if mapping_path.exists():
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.disease_classes = json.load(f)
                logger.info(f"从{mapping_path}加载了{len(self.disease_classes)}个病害类别")
            else:
                self.disease_classes = {str(i): f"病害类别{i}" for i in range(5)}
                logger.warning(f"未找到病害类别映射文件，使用默认映射")
        except Exception as e:
            logger.error(f"加载病害类别映射失败: {e}")
            self.disease_classes = {str(i): f"病害类别{i}" for i in range(5)}
    
    def classify_plant(self, image_path):
        """
        识别植物种类
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: 识别结果，包含类别和置信度
        """
        if self.classifier is None:
            return {"error": "分类器模型未加载"}
        
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            transform = get_transform(train=False)
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # 进行预测
            with torch.no_grad():
                outputs = self.classifier(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
                # 获取最可能的类别
                score, class_idx = torch.max(probabilities, dim=0)
                
                # 获取类别名称
                class_name = self.plant_classes.get(str(class_idx.item()), f"未知类别{class_idx.item()}")
                
                # 构建结果
                result = {
                    'class_id': class_idx.item(),
                    'class_name': class_name,
                    'confidence': score.item(),
                    'probabilities': probabilities.cpu().numpy().tolist()
                }
                
                return result
                
        except Exception as e:
            logger.error(f"植物分类失败: {str(e)}")
            return {"error": f"植物分类失败: {str(e)}"}
    
    def detect_disease(self, image_path):
        """
        检测植物病害
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: 检测结果，包含边界框、类别和置信度
        """
        if self.detector is None:
            return {"error": "检测器模型未加载"}
        
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            transform = get_transform(train=False)
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # 记录原始尺寸
            original_width, original_height = image.size
            
            # 进行预测
            with torch.no_grad():
                outputs = self.detector(image_tensor)
                
            # 处理预测结果
            boxes = outputs[0]['boxes'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()
            labels = outputs[0]['labels'].cpu().numpy()
            
            # 筛选掉低置信度的检测结果
            threshold = 0.5  # 可以从配置获取
            keep = scores >= threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # 构建结果
            detections = []
            for box, score, label in zip(boxes, scores, labels):
                # 获取类别名称
                class_name = self.disease_classes.get(str(label-1), f"未知病害{label}")
                
                # 添加检测结果
                detections.append({
                    'box': box.tolist(),
                    'score': float(score),
                    'class_id': int(label),
                    'class_name': class_name
                })
            
            result = {
                'detections': detections,
                'image_size': {
                    'width': original_width,
                    'height': original_height
                }
            }
            
            return result
                
        except Exception as e:
            logger.error(f"病害检测失败: {str(e)}")
            return {"error": f"病害检测失败: {str(e)}"}

class DummyPredictor:
    """
    虚拟预测器，用于测试和开发环境
    """
    
    def __init__(self):
        """初始化虚拟预测器"""
        logger.info("初始化虚拟预测器")
        
    def classify(self, image_bytes):
        """
        模拟图像分类
        
        Args:
            image_bytes: 图像数据
            
        Returns:
            模拟的分类结果
        """
        return {
            "predictions": [
                {"class_name": "示例植物", "confidence": 0.95},
                {"class_name": "其他植物", "confidence": 0.03},
                {"class_name": "未知植物", "confidence": 0.02}
            ],
            "top_result": "示例植物",
            "confidence": 0.95
        }
        
    def detect(self, image_bytes, plant_type=None):
        """
        模拟病害检测
        
        Args:
            image_bytes: 图像数据
            plant_type: 植物类型，用于专门检测该类植物的病害
            
        Returns:
            模拟的检测结果
        """
        return {
            "detections": [
                {
                    "bbox": {"x": 100, "y": 100, "width": 200, "height": 200},
                    "score": 0.92,
                    "label": 1,
                    "class_name": "示例病害",
                    "severity": "moderate",
                    "plant_type": plant_type or "未知植物"  # 使用提供的植物类型或默认值
                }
            ],
            "count": 1,
            "plant_type": plant_type or "未知植物",  # 添加植物类型信息
            "severity_assessment": {
                "level": "moderate",
                "description": f"检测到{plant_type or '植物'}的中等程度病害"  # 使用植物类型自定义描述
            }
        }