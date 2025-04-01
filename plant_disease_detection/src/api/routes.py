#src/api/routes.py
"""
API路由模块
提供与前端交互的接口
"""
import os
import logging
import time
import io
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
from src.utils.config_manager import ConfigManager
from src.inference.predictor import DetectionPredictor

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
                        
                        # 获取最可能的类别
                        score, class_idx = torch.max(probabilities, dim=0)
                        
                        # 获取类别映射
                        plant_class_names = getattr(current_app, 'plant_class_names', {})
                        
                        # 将索引转换为字符串以匹配JSON键
                        idx_str = str(class_idx.item())
                        class_name = plant_class_names.get(idx_str, f"类别{class_idx.item()}")
                        
                        # 构建预测列表
                        predictions = []
                        for i, prob in enumerate(probabilities):
                            i_str = str(i)
                            class_n = plant_class_names.get(i_str, f"类别{i}")
                            predictions.append({
                                "class_name": class_n,
                                "confidence": prob.item()
                            })
                        
                        result = {
                            "predictions": sorted(predictions, key=lambda x: x["confidence"], reverse=True),
                            "class_name": class_name,
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
                    # 获取转换函数
                    transform = get_transform(train=False)
                    
                    # 应用转换并添加批次维度
                    # 从detector获取设备，而不是使用self.device
                    device = next(self.detector.parameters()).device
                    image_tensor = transform(image).unsqueeze(0).to(device)
                    
                    # 确保模型处于评估模式
                    self.detector.eval()
                    
                    # 执行预测 - 修改这里，从4D张量[1,C,H,W]提取出3D张量[C,H,W]作为列表元素
                    with torch.no_grad():
                        # 关键修改：传递正确格式的张量列表
                        prediction = self.detector([image_tensor[0]])
                        
                    # 第一个(也是唯一的)图像的预测结果
                    if prediction and len(prediction) > 0:
                        # 添加图像尺寸信息
                        prediction[0]['image_size'] = (original_width, original_height)
                        
                        # 处理预测结果
                        processed_results = self._process_detections(prediction[0], plant_type)
                        return processed_results
                    else:
                        return {"error": "检测模型未返回结果", "detections": []}
                else:
                    return {"error": "检测模型未加载", "detections": []}
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"error": f"检测失败: {str(e)}", "detections": []}
        
        def _process_detections(self, output, plant_type=None):
            """处理检测输出结果为友好格式"""
            
            try:
                # 确保有输出且格式正确
                if not output or not isinstance(output, dict):
                    return {"error": "无效的检测结果格式", "detections": []}
                
                # 获取边界框、置信度和标签
                boxes = output.get('boxes', torch.tensor([]))
                scores = output.get('scores', torch.tensor([]))
                labels = output.get('labels', torch.tensor([]))
                
                # 获取图像尺寸
                image_size = output.get('image_size', (0, 0))
                width, height = image_size
                
                # 获取类别名称映射
                plant_class_names = getattr(current_app, 'plant_class_names', {})
                
                # 构建检测结果
                detections = []
                for i, (box, score, label) in enumerate(zip(boxes.tolist(), scores.tolist(), labels.tolist())):
                    # 获取类别名称
                    class_name = plant_class_names.get(str(label), f"class_{label}")
                    
                    # 将归一化坐标转换为实际像素坐标
                    x, y, x2, y2 = box
                    
                    # 创建检测结果字典
                    detection = {
                        'id': i,
                        'label': label,
                        'class_name': class_name,
                        'score': score,
                        'bbox': {
                            'x': x,
                            'y': y,
                            'width': x2 - x,
                            'height': y2 - y
                        }
                    }
                    
                    # 确定疾病严重程度
                    detection['severity'] = self._assess_severity(score, (x2-x)*(y2-y), width*height)
                    
                    # 添加治疗信息 - 这是新增代码
                    try:
                        if '-' in class_name:
                            plant_disease = class_name.split('-')
                            plant_name = plant_disease[0].strip()
                            disease_name = plant_disease[1].strip()
                            
                            # 如果不是"健康"状态，获取治疗信息
                            if disease_name.lower() != "healthy":
                                # 获取治疗信息
                                treatment_info = treatment_db.get_treatment(plant_name, disease_name)
                                detection['treatment_info'] = treatment_info
                                logger.info(f"添加治疗信息: {plant_name}-{disease_name}")
                    except Exception as e:
                        logger.error(f"获取治疗信息失败: {e}")
                    
                    detections.append(detection)
                
                # 评估整体严重程度
                severity_assessment = {
                    'level': 'unknown',
                    'description': '无法评估'
                }
                
                if detections:
                    # 基于检测结果评估严重程度
                    avg_score = sum(det['score'] for det in detections) / len(detections)
                    max_score = max(det['score'] for det in detections)
                    
                    if len(detections) > 3 and avg_score > 0.7:
                        severity_assessment = {
                            'level': 'severe',
                            'description': '检测到多处高置信度病害，建议立即采取治疗措施'
                        }
                    elif max_score > 0.8 or (len(detections) > 1 and avg_score > 0.6):
                        severity_assessment = {
                            'level': 'moderate',
                            'description': '检测到显著病害迹象，建议尽快采取防治措施'
                        }
                    elif max_score > 0.5:
                        severity_assessment = {
                            'level': 'mild',
                            'description': '检测到轻微病害迹象，建议密切观察并考虑预防性措施'
                        }
                elif plant_type:
                    # 没有检测到病害，植物可能健康
                    severity_assessment = {
                        'level': 'healthy',
                        'description': '未检测到病害，植物可能健康'
                    }
                
                # 构建最终结果
                result = {
                    'detections': detections,
                    'count': len(detections),
                    'image_dimensions': {'width': width, 'height': height},
                    'severity_assessment': severity_assessment
                }
                
                return result
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"error": f"处理检测结果出错: {str(e)}", "detections": []}
        
        def _assess_severity(self, confidence, area, total_area):
            """评估病害严重程度"""
            coverage = area / total_area if total_area > 0 else 0
            
            if confidence > 0.8 and coverage > 0.3:
                return "severe"
            elif confidence > 0.6 and coverage > 0.1:
                return "moderate"
            elif confidence > 0.5:
                return "mild"
            else:
                return "healthy"
    
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
    """病害检测接口 - 修改为侧重分类结果"""
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
    
    # 记录调试信息
    logger.info(f"植物类型参数: {plant_type}")
    
    try:
        # 转换图像为PIL格式供DetectionPredictor使用
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # --- 修改: 直接创建并使用DetectionPredictor ---
        if not hasattr(current_app, 'detector') or current_app.detector is None:
             logger.error("检测模型未加载！")
             return jsonify({"error": "检测模型不可用"}), 500
        if not hasattr(current_app, 'class_id_to_name_map'):
             logger.error("统一类别映射未加载！")
             return jsonify({"error": "系统配置错误"}), 500

        config_manager = ConfigManager()
        score_thresh = config_manager.get('confidence_threshold', 0.3, 'model')

        # 直接创建DetectionPredictor实例
        predictor = DetectionPredictor(
            current_app.detector,
            current_app.class_id_to_name_map, 
            score_threshold=score_thresh
        )

        # 使用映射服务将前端植物名称转换为模型可用格式
        model_plant_type = mapping_service.extract_plant_type_from_class(plant_type)
        logger.info(f"使用植物类型进行检测: {plant_type} (映射为: {model_plant_type})")

        logger.info("Calling DetectionPredictor.predict...")
        results = predictor.predict(image_pil, plant_type=model_plant_type)
        logger.info(f"Received results from DetectionPredictor.predict: {results}")
        # --- 结束修改 ---

        # 提取最高置信度的预测结果
        top_prediction = results.get("top_prediction", {})
        logger.info(f"Extracted top_prediction: {top_prediction}")
        
        class_name = top_prediction.get("class_name", "未知")
        confidence = top_prediction.get("confidence", 0.0)
        logger.info(f"提取的class_name: {class_name}, confidence: {confidence}")
        
        # 提取植物类型和病害名称
        plant_type_from_pred = "未知"
        disease_name_from_pred = "未知"
        if (class_name != "未知" and "-" in class_name):
            parts = class_name.split('-', 1)
            plant_type_from_pred = parts[0]
            disease_name_from_pred = parts[1]
            logger.info(f"从class_name拆分: 植物={plant_type_from_pred}, 病害={disease_name_from_pred}")
        
        # 获取治疗信息
        treatment_info = {}
        if disease_name_from_pred != "未知" and disease_name_from_pred.lower() != "健康":
            treatment_info = treatment_db.get_treatment(plant_type_from_pred, disease_name_from_pred)
            logger.info(f"获取到治疗信息: {bool(treatment_info)}")
        
        # 构建最终响应
        final_response = {
            "class_name": class_name,
            "confidence": confidence,
            "plant_type": plant_type_from_pred,
            "disease_name": disease_name_from_pred,
            "treatment_info": treatment_info if not treatment_info.get('error') else {},
            "detection_mode": "image_level_classification",
            "detections": results.get("detections", []),
            "top_prediction": top_prediction  # 确保包含完整的top_prediction
        }
        
        # 添加植物类型信息到结果中
        final_response['plant_type'] = plant_type
        
        logger.info(f"Final response to be sent: {final_response}")
        logger.info(f"检测完成: {final_response['class_name']}, 置信度: {final_response['confidence']:.4f}")
        return jsonify(final_response)
    
    except Exception as e:
        logger.error(f"检测过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"检测过程中出错: {str(e)}"}), 500

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
                    # 创建病害摘要信息
                    disease_summary = {
                        "name": disease,
                        "symptoms_summary": info.get("symptoms", ["无症状描述"])[0] if isinstance(info.get("symptoms"), list) and info.get("symptoms") else "无症状描述",
                        "severity": info.get("severity", "未知"),
                        "has_treatment": bool(info.get("treatments", []))
                    }
                    diseases_info["diseases"].append(disease_summary)
            
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
        # 如果没有提供disease_type，则返回该植物的所有病害概述
        if not disease_type:
            plant_info = treatment_db.get_treatment(plant_type)
            # 添加植物名称到返回结果
            plant_info['plant_name'] = plant_type
            return jsonify(plant_info)
        
        # 获取特定病害的治疗信息
        treatment_info = treatment_db.get_treatment(plant_type, disease_type)
        
        # 检查是否找到了治疗信息
        if 'error' in treatment_info:
            logger.warning(f"未找到病害信息: {plant_type}/{disease_type}")
            return jsonify(treatment_info), 404
        
        # 确保返回的数据包含植物名称和疾病名称
        if 'plant_name' not in treatment_info:
            treatment_info['plant_name'] = plant_type
        if 'disease_name' not in treatment_info:
            treatment_info['disease_name'] = disease_type
            
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

# --- 新增: Grad-CAM 解释路由 ---
@api_bp.route('/explain_detection', methods=['POST'])
def explain_detection_route():
    """为检测模型的分类结果生成Grad-CAM可视化解释"""
    if 'image' not in request.files:
        return jsonify({"error": "未找到图像文件"}), 400

    file = request.files['image']
    try:
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        logger.error(f"解释请求：图像加载失败: {e}")
        return jsonify({"error": f"无法加载图像: {str(e)}"}), 400

    # 检查模型和映射是否加载
    if not hasattr(current_app, 'detector') or current_app.detector is None:
         return jsonify({"error": "检测模型不可用"}), 500
    if not hasattr(current_app, 'class_id_to_name_map'):
         return jsonify({"error": "系统配置错误"}), 500

    try:
        # 1. 先获取模型的预测结果 (主要是类别 ID)
        config_manager = ConfigManager()
        score_thresh = config_manager.get('confidence_threshold', 0.3, 'model')
        predictor = DetectionPredictor(current_app.detector, current_app.class_id_to_name_map, score_thresh)
        pred_results = predictor.predict(image_pil)
        target_class_id = pred_results.get("top_prediction", {}).get("label_id", None)

        if target_class_id is None or target_class_id == -1:
            return jsonify({"error": "模型未能对此图像进行有效预测，无法生成解释"}), 400

        # 2. 创建 ModelInterpreter 实例
        from src.evaluation.model_interpreter import ModelInterpreter
        interpreter = ModelInterpreter(current_app.detector)

        # 3. 确定 Grad-CAM 目标层 - 可能需要调整为实际模型结构
        # 尝试几个可能的路径格式
        possible_target_layers = [
            'backbone.body.layer4',  # 直接模型结构
            'detector.backbone.body.layer4',  # 封装模型结构
            'model.backbone.body.layer4'  # 另一种可能的封装
        ]
        
        target_layer = None
        for layer_path in possible_target_layers:
            try:
                # 尝试用getattr递归访问各层来验证路径有效性
                parts = layer_path.split('.')
                current = current_app.detector
                for part in parts:
                    if not hasattr(current, part):
                        break
                    current = getattr(current, part)
                else:
                    # 如果没有break，说明路径有效
                    target_layer = layer_path
                    break
            except:
                continue
        
        # 如果找不到有效层，使用默认值
        if not target_layer:
            target_layer = config_manager.get('DETECTOR_GRADCAM_TARGET_LAYER', 'backbone.body.layer4', "model")
            logger.warning(f"无法验证目标层路径，使用配置默认值: {target_layer}")

        # 4. 生成 Grad-CAM 可视化 (不保存文件，返回 numpy 数组)
        visualization_np = interpreter.explain_prediction(
            image=image_pil, # 直接传递 PIL Image
            class_idx=target_class_id,
            target_layer=target_layer,
            output_path=None, # 设置为 None 以获取数组
            class_names=current_app.class_id_to_name_map # 传递类别映射
        )

        # 5. 将 numpy 数组转换为 Base64 编码的图像数据
        import base64
        from io import BytesIO

        # 将 numpy 数组 (0-255, RGB) 转换为 PIL Image
        vis_img_pil = Image.fromarray(visualization_np)
        buffered = BytesIO()
        vis_img_pil.save(buffered, format="PNG") # 保存为 PNG 格式
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{img_str}"

        # 6. 返回包含可视化数据的 JSON
        response_data = {
            "predicted_class_id": target_class_id,
            "predicted_class_name": pred_results.get("top_prediction", {}).get("class_name", "未知"),
            "gradcam_image": data_url # 返回 Base64 Data URL
        }
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"生成Grad-CAM时出错: {e}", exc_info=True)
        return jsonify({"error": f"生成解释时出错: {str(e)}"}), 500