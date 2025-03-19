#src/api/routes.py
"""
API路由模块
提供与前端交互的接口
"""
import os
import logging
import time
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from flask import Blueprint, request, jsonify, current_app

from src.utils.image_utils import preprocess_image
from src.utils.disease_treatments import DiseaseTreatmentDatabase
# 添加映射服务导入
from src.utils.mapping_service import MappingService

# 创建蓝图
api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# 初始化病害治疗数据库（全局单例）
treatment_db = DiseaseTreatmentDatabase()
# 在文件顶部初始化映射服务
mapping_service = MappingService()

# 添加本地版本的 create_predictor 函数
def create_predictor():
    """创建预测器"""
    # 如果没有加载模型，返回一个虚拟预测器
    if not hasattr(current_app, 'classifier') and not hasattr(current_app, 'detector'):
        from src.inference.predictor import DummyPredictor
        return DummyPredictor()
    
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
            
            try:
                # 从字节流加载图像
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # 使用分类模型进行预测
                if self.classifier:
                    # 获取转换函数
                    transform = get_transform(train=False)
                    
                    # 应用转换并添加批次维度
                    image_tensor = transform(image).unsqueeze(0)
                    
                    # 将tensor移到与模型相同的设备上
                    device = next(self.classifier.parameters()).device
                    image_tensor = image_tensor.to(device)
                    
                    # 执行预测
                    with torch.no_grad():
                        # 直接使用模型进行前向传播
                        outputs = self.classifier(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1)[0]
                        
                        # 获取类名
                        class_mapping = getattr(current_app, 'class_mapping', {})
                        class_names = getattr(current_app, 'class_names', [f"类别{i}" for i in range(10)])
                        
                        # 获取最可能的类别
                        score, class_idx = torch.max(probabilities, dim=0)
                        
                        # 使用映射获取类名
                        idx_str = str(class_idx.item())
                        class_name = class_mapping.get(idx_str, f"类别{class_idx.item()}")
                        
                        # 构建预测列表
                        predictions = []
                        for i, prob in enumerate(probabilities):
                            idx_str = str(i)
                            if i < len(probabilities):
                                predictions.append({
                                    "class_name": class_mapping.get(idx_str, f"类别{i}"),
                                    "confidence": prob.item()
                                })
                        
                        result = {
                            "predictions": sorted(predictions, key=lambda x: x["confidence"], reverse=True),
                            "class_name": class_name,  # 这里添加了class_name字段
                            "top_result": class_name,
                            "confidence": score.item()
                        }
                        
                        return result
                else:
                    return {"error": "分类模型未加载", "predictions": [], "class_name": "未知"}
                    
            except Exception as e:
                # 详细记录错误
                import traceback
                error_details = traceback.format_exc()
                print(f"图像处理错误: {str(e)}\n{error_details}")
                return {"error": f"图像处理错误: {str(e)}", "predictions": [], "class_name": "未知"}
        
        def detect(self, image_bytes, plant_type=None):
            """检测预测，添加plant_type参数"""
            import io
            import torch
            import traceback
            from PIL import Image
            
            try:
                # 从字节流加载图像
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                original_width, original_height = image.size
                
                # 使用检测模型进行预测
                if self.detector:
                    # 确保模型在评估模式
                    self.detector.eval()
                    
                    # 获取转换函数
                    transform = get_transform(train=False)
                    
                    # 应用转换并添加批次维度
                    image_tensor = transform(image).unsqueeze(0)
                    
                    # 将tensor移到与模型相同的设备上
                    device = next(self.detector.parameters()).device
                    image_tensor = image_tensor.to(device)
                    
                    # 使用torch.no_grad()包装推理过程
                    with torch.no_grad():
                        outputs = self.detector(image_tensor)
                    
                    # 保存原始图像尺寸
                    for output in outputs:
                        output['image_size'] = (original_width, original_height)
                    
                    # 处理检测结果
                    return self._process_detections(outputs[0], plant_type)
                
                return {"error": "检测模型未加载", "detections": []}
            
            except Exception as e:
                traceback.print_exc()
                return {"error": f"检测失败: {str(e)}", "detections": []}
        
        def _process_detections(self, output, plant_type=None):
            """处理检测输出结果为友好格式"""
            import torch
            
            try:
                # 确保有输出且格式正确
                if not output or not isinstance(output, dict):
                    return {"error": "无效的检测结果", "detections": []}
                
                # 获取边界框、置信度和标签
                boxes = output.get('boxes', torch.tensor([]))
                scores = output.get('scores', torch.tensor([]))
                labels = output.get('labels', torch.tensor([]))
                
                # 获取图像尺寸
                image_size = output.get('image_size', (0, 0))
                width, height = image_size
                
                # 获取类别名称映射
                disease_class_names = getattr(current_app, 'disease_class_names', {})
                
                # 构建检测结果
                detections = []
                for i, (box, score, label) in enumerate(zip(boxes.tolist(), scores.tolist(), labels.tolist())):
                    # 获取类别名称
                    label_str = str(int(label))
                    class_name = disease_class_names.get(label_str, f"未知类别-{label}")
                    
                    # 计算边界框信息
                    x1, y1, x2, y2 = box
                    x = max(0, min(x1, width))
                    y = max(0, min(y1, height))
                    w = max(0, min(x2 - x1, width - x))
                    h = max(0, min(y2 - y1, height - y))
                    
                    # 确定疾病严重程度
                    severity = "unknown"
                    if class_name == "健康":
                        severity = "healthy"
                    elif score > 0.8:
                        severity = "severe"
                    elif score > 0.6:
                        severity = "moderate"
                    else:
                        severity = "mild"
                    
                    # 创建检测对象
                    detection = {
                        "id": i,
                        "bbox": {"x": x, "y": y, "width": w, "height": h},
                        "score": score,
                        "label": label,
                        "class_name": class_name,
                        "severity": severity
                    }
                    
                    # 如果提供了植物类型，添加到检测结果中
                    if plant_type:
                        detection["plant_type"] = plant_type
                    
                    detections.append(detection)
                
                # 如果有植物类型和检测结果，添加治疗建议
                treatment_recommendations = []
                if plant_type and detections:
                    # 获取检测到的所有病害
                    diseases = [det["class_name"] for det in detections if det["class_name"] != "健康"]
                    
                    # 如果检测到病害，获取治疗建议
                    if diseases:
                        for disease in diseases:
                            treatment = treatment_db.get_treatment(plant_type, disease)
                            if treatment and "treatments" in treatment:
                                treatment_recommendations.extend(treatment["treatments"])
                    
                    # 如果没有病害或无法获取治疗建议
                    if not treatment_recommendations:
                        if any(det["class_name"] == "健康" for det in detections):
                            treatment_recommendations = ["植株健康，无需特殊处理", "保持良好的栽培实践"]
                
                # 返回最终结果
                return {
                    "detections": detections,
                    "count": len(detections),
                    "image_dimensions": {"width": width, "height": height},
                    "treatment_recommendations": treatment_recommendations
                }
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"error": f"处理检测结果出错: {str(e)}", "detections": []}
    
    # 返回组合预测器
    return CombinedPredictor(
        getattr(current_app, 'classifier', None),
        getattr(current_app, 'detector', None)
    )

# 在 routes.py 中创建专门的 get_transform 函数

def get_transform(train=False):
    """获取图像转换函数"""
    from torchvision import transforms
    
    if train:
        # 训练时的转换（如果需要）
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # 推理时的转换
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

@api_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "ok", "timestamp": time.time()})

@api_bp.route('/classify', methods=['POST'])
def classify_plant():
    """植物分类接口"""
    if 'image' not in request.files:
        return jsonify({"error": "未找到图像文件"}), 400
    
    # 获取上传的图像
    file = request.files['image']
    image_bytes = file.read()
    logger.info(f"接收到分类请求，图像大小: {len(image_bytes)} 字节")
    
    try:
        # 创建预测器
        predictor = create_predictor()
        
        # 执行预测
        results = predictor.classify(image_bytes)
        
        # 如果分类结果包含"-"，表示是"植物-病害"格式，添加纯植物名称
        if 'class_name' in results and '-' in results['class_name']:
            plant_only = results['class_name'].split('-')[0]
            logger.info(f"从分类结果中提取纯植物类型: {plant_only}，原始结果: {results['class_name']}")
            
            # 保存原始结果
            results['full_class_name'] = results['class_name']
            # 更新为仅植物类型
            results['class_name'] = plant_only
        
        # 获取植物详细信息
        plant_info = treatment_db.get_plant_info(results['class_name'])
        results['plant_info'] = plant_info
        
        # 对预测列表中的其他条目也提取纯植物名称
        if 'predictions' in results:
            for pred in results['predictions']:
                if '-' in pred['class_name']:
                    pred['full_class_name'] = pred['class_name']
                    pred['class_name'] = pred['class_name'].split('-')[0]
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"分类过程中出错: {e}")
        return jsonify({"error": f"分类失败: {str(e)}", "class_name": "未知", "confidence": 0}), 500

# 修改检测API以更好地利用植物类型信息
@api_bp.route('/detect', methods=['POST'])
def detect_diseases():
    """病害检测接口"""
    if 'image' not in request.files:
        return jsonify({"error": "未找到图像文件"}), 400
    
    # 获取上传的图像
    file = request.files['image']
    image_bytes = file.read()
    logger.info(f"接收到检测请求，图像大小: {len(image_bytes)} 字节")
    
    # 获取植物类型参数
    plant_type = request.form.get('plant_type')
    if not plant_type:
        return jsonify({"error": "缺少植物类型参数，请先进行植物识别或指定植物类型"}), 400
    
    # 从分类结果中正确提取植物类型（删除可能附带的病害信息）
    if "-" in plant_type:
        # 分割类似"苹果-黑星病"的格式
        plant_type = plant_type.split('-')[0]
        logger.info(f"从分类结果中提取纯植物类型: {plant_type}")
    
    try:
        # 创建预测器
        predictor = create_predictor()
        
        # 使用映射服务将前端植物名称转换为模型可用格式
        model_plant_type = mapping_service.extract_plant_type_from_class(plant_type)
        logger.info(f"使用植物类型进行检测: {plant_type} (映射为: {model_plant_type})")
        
        # 执行检测后，添加过滤逻辑
        results = predictor.detect(image_bytes, plant_type=model_plant_type)

        # 添加植物类型信息到结果中
        results['plant_type'] = plant_type

        # 过滤不兼容的病害
        filtered_detections = []
        for detection in results["detections"]:
            disease_name = detection["class_name"]
            
            # 检查是否为兼容的病害
            if treatment_db.is_disease_compatible_with_plant(plant_type, disease_name):
                filtered_detections.append(detection)
            else:
                logger.warning(f"过滤不兼容的病害检测结果: {disease_name}，与{plant_type}不兼容")

        # 如果过滤后没有结果，添加"健康"状态
        if not filtered_detections and results["detections"]:
            # 使用原始检测的边界框，但将类别改为"健康"
            detection = results["detections"][0].copy()
            detection["class_name"] = "健康" 
            detection["severity"] = "healthy"
            detection["score"] = 0.95
            filtered_detections.append(detection)
            logger.info(f"将不兼容的病害检测替换为'健康'状态")

        # 更新结果
        results["detections"] = filtered_detections
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"检测过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"检测失败: {str(e)}", "detections": []}), 500

# 修改get_plant_diseases函数

@api_bp.route('/plant_diseases', methods=['GET'])
def get_plant_diseases():
    """获取特定植物可能的病害及其治疗方案"""
    plant_type = request.args.get('plant')
    
    if not plant_type:
        return jsonify({"error": "请提供植物类型参数"}), 400
    
    try:
        # 使用标准化的名称
        plant = treatment_db._normalize_name(plant_type)
        
        if plant in treatment_db.treatments:
            # 获取该植物的所有病害信息
            diseases_info = {}
            
            # 添加概述信息
            if "overview" in treatment_db.treatments[plant]:
                diseases_info["overview"] = treatment_db.treatments[plant]["overview"]
            
            # 添加各种病害信息
            diseases_info["diseases"] = []
            for disease, info in treatment_db.treatments[plant].items():
                if disease != "overview" and isinstance(info, dict):
                    disease_data = {
                        "name": disease,
                        "symptoms": info.get("symptoms", []),
                        "causes": info.get("causes", []),
                        "treatments": info.get("treatments", []),
                        "prevention": info.get("prevention", []),
                        "severity": info.get("severity", "未知"),
                        "organic_solutions": info.get("organic_solutions", [])
                    }
                    diseases_info["diseases"].append(disease_data)
            
            return jsonify(diseases_info)
        else:
            return jsonify({"error": f"未找到植物 '{plant_type}' 的病害信息"}), 404
            
    except Exception as e:
        logger.error(f"获取植物病害信息时出错: {e}")
        return jsonify({"error": f"获取信息时出错: {str(e)}"}), 500

@api_bp.route('/treatment', methods=['GET'])
def get_treatment():
    """获取特定植物病害的治疗建议"""
    plant_type = request.args.get('plant')
    disease_type = request.args.get('disease')
    
    if not plant_type:
        return jsonify({"error": "请提供植物类型参数"}), 400
    
    try:
        treatment_info = treatment_db.get_treatment(plant_type, disease_type)
        return jsonify(treatment_info)
    except Exception as e:
        logger.error(f"获取治疗建议时出错: {e}")
        return jsonify({"error": f"获取治疗建议时出错: {str(e)}"}), 500

# 在文件末尾添加新的路由

@api_bp.route('/plant_info', methods=['GET'])
def get_plant_info():
    """获取植物详细信息的API"""
    plant_type = request.args.get('plant')
    
    if not plant_type:
        return jsonify({"error": "请提供植物类型参数"}), 400
    
    try:
        # 获取植物信息
        plant_info = treatment_db.get_plant_info(plant_type)
        return jsonify(plant_info)
    except Exception as e:
        logger.error(f"获取植物信息时出错: {e}")
        return jsonify({"error": f"获取植物信息时出错: {e}"}), 500