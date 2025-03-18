# src/inference/result_parser.py

import os
import json
from typing import Dict, List, Optional, Union, Any
import numpy as np
from datetime import datetime
from pathlib import Path

from src.utils.config_manager import ConfigManager

"""
结果解析器
处理和增强模型输出结果
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

def parse_classification_results(raw_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析分类模型结果
    
    Args:
        raw_results: 原始分类结果
        
    Returns:
        解析和增强后的结果
    """
    # 确保输出格式一致
    if 'probabilities' in raw_results and isinstance(raw_results['probabilities'], np.ndarray):
        # 转换numpy数组为Python列表
        probs = raw_results['probabilities'].tolist()
        
        # 如果类别名称可用，使用它们
        class_names = raw_results.get('class_names', [f"class_{i}" for i in range(len(probs))])
        
        # 创建预测列表
        predictions = [
            {'class_name': name, 'confidence': prob}
            for name, prob in zip(class_names, probs)
        ]
        
        # 按置信度排序
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'predictions': predictions,
            'top_result': predictions[0]['class_name'],
            'confidence': predictions[0]['confidence']
        }
    
    # 直接返回原始结果
    return raw_results

def parse_detection_results(raw_results: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    """
    解析检测模型结果
    
    Args:
        raw_results: 原始检测结果
        threshold: 检测阈值
        
    Returns:
        解析和增强后的结果
    """
    # 初始化结果字典
    parsed_results = {
        'detections': [],
        'count': 0,
        'severity_assessment': {
            'level': 'Unknown',
            'description': 'No assessment available'
        }
    }
    
    # 检查原始结果格式
    if 'boxes' in raw_results and 'labels' in raw_results and 'scores' in raw_results:
        boxes = raw_results['boxes']
        labels = raw_results['labels']
        scores = raw_results['scores']
        
        # 获取类别名称
        class_names = raw_results.get('class_names', [f"class_{i}" for i in range(100)])
        
        # 过滤低于阈值的检测结果
        filtered_detections = []
        for i, score in enumerate(scores):
            if score >= threshold:
                # 创建检测结果项
                detection = {
                    'bbox': {
                        'x': float(boxes[i][0]),
                        'y': float(boxes[i][1]),
                        'width': float(boxes[i][2] - boxes[i][0]),
                        'height': float(boxes[i][3] - boxes[i][1])
                    },
                    'score': float(score),
                    'label': int(labels[i]),
                    'class_name': class_names[int(labels[i])] if int(labels[i]) < len(class_names) else f"class_{labels[i]}"
                }
                
                # 添加严重程度评估
                detection['severity'] = assess_detection_severity(detection)
                
                filtered_detections.append(detection)
        
        # 更新解析结果
        parsed_results['detections'] = filtered_detections
        parsed_results['count'] = len(filtered_detections)
        
        # 总体严重程度评估
        parsed_results['severity_assessment'] = assess_overall_severity(filtered_detections)
    
    return parsed_results

def assess_detection_severity(detection: Dict[str, Any]) -> str:
    """
    评估单个检测结果的严重程度
    
    Args:
        detection: 检测结果项
        
    Returns:
        严重程度评级
    """
    # 简单基于置信度的严重程度评估
    score = detection['score']
    
    if score > 0.8:
        return 'severe'
    elif score > 0.6:
        return 'moderate'
    elif score > 0.4:
        return 'mild'
    else:
        return 'uncertain'

def assess_overall_severity(detections: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    评估整体病害严重程度
    
    Args:
        detections: 检测结果列表
        
    Returns:
        严重程度评估
    """
    if not detections:
        return {
            'level': 'healthy',
            'description': '未检测到任何病害，植物看起来健康。'
        }
    
    # 计算平均置信度和最高置信度
    avg_score = sum(d['score'] for d in detections) / len(detections)
    max_score = max(d['score'] for d in detections)
    
    # 根据检测数量和置信度评估严重程度
    if len(detections) > 3 and avg_score > 0.7:
        return {
            'level': 'severe',
            'description': '检测到多处高置信度病害，建议立即采取治疗措施。'
        }
    elif max_score > 0.8 or (len(detections) > 1 and avg_score > 0.6):
        return {
            'level': 'moderate',
            'description': '检测到显著病害迹象，建议尽快采取防治措施。'
        }
    elif max_score > 0.5:
        return {
            'level': 'mild',
            'description': '检测到轻微病害迹象，建议密切观察并考虑预防性措施。'
        }
    else:
        return {
            'level': 'uncertain',
            'description': '检测到可能的病害迹象，但置信度较低，建议进一步观察。'
        }

class ResultParser:
    """推理结果解析基类"""
    
    def __init__(self):
        """初始化结果解析器"""
        self.config_manager = ConfigManager()
        
    def parse_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析推理结果
        
        Args:
            result: 模型预测结果
            
        Returns:
            解析后的结果
        """
        # 子类实现具体解析逻辑
        raise NotImplementedError("子类必须实现parse_result方法")
    
    def save_result(self, result: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        保存解析结果到文件
        
        Args:
            result: 解析后的结果
            output_path: 输出路径
        """
        # 确保路径存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


class ClassificationResultParser(ResultParser):
    """分类结果解析类，处理植物分类结果"""
    
    def __init__(self, plant_info_db: Optional[Dict[str, Any]] = None):
        """
        初始化分类结果解析器
        
        Args:
            plant_info_db: 植物信息数据库，包含植物种类的详细信息
        """
        super().__init__()
        self.plant_info_db = plant_info_db or {}
        self.confidence_threshold = self.config_manager.get(
            'CLASSIFICATION_CONFIDENCE_THRESHOLD', 0.7, "model")
    
    def parse_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析分类结果，添加植物信息和建议
        
        Args:
            result: 分类预测结果
            
        Returns:
            增强的分类结果
        """
        # 获取识别的种类信息
        class_name = result['class_name']
        confidence = result['confidence']
        
        # 增强结果
        enhanced_result = {
            'prediction': result,
            'timestamp': datetime.now().isoformat(),
            'plant_info': self._get_plant_info(class_name),
            'confidence_level': self._get_confidence_level(confidence),
            'suggestions': self._get_plant_suggestions(class_name)
        }
        
        return enhanced_result
    
    def _get_plant_info(self, class_name: str) -> Dict[str, Any]:
        """获取植物详细信息"""
        # 先尝试从治疗数据库获取映射
        if hasattr(self, 'treatment_db'):
            plant_name, _ = self.treatment_db.map_dataset_class_to_db(class_name)
            plant_info = self.treatment_db.get_treatment(plant_name)
            if 'error' not in plant_info and 'overview' in plant_info:
                return {
                    'description': plant_info['overview'].get('description', f'关于{plant_name}的信息'),
                    'general_care': plant_info['overview'].get('general_care', [])
                }
        
        # 默认返回
        return {
            'description': f'未找到关于{class_name}的详细信息',
            'general_care': []
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """根据置信度评估识别可靠性"""
        if confidence >= 0.9:
            return "很高"
        elif confidence >= 0.7:
            return "高"
        elif confidence >= 0.5:
            return "中等"
        else:
            return "低"
    
    def _get_plant_suggestions(self, class_name: str) -> List[str]:
        """根据植物种类提供建议"""
        # 这里可以添加针对不同植物种类的具体建议
        return [
            f"已识别为{class_name}，请查看详细信息",
            "如需更准确的识别结果，请提供更清晰的图像"
        ]


class DiseaseResultParser(ResultParser):
    """病害结果解析类，处理植物病害检测结果"""
    
    def __init__(self, disease_info_db: Optional[Dict[str, Any]] = None):
        """
        初始化病害结果解析器
        
        Args:
            disease_info_db: 病害信息数据库，包含各种病害的详细信息
        """
        super().__init__()
        
        # 如果没有提供疾病数据库，则创建一个
        if disease_info_db is None:
            from src.utils.disease_treatments import DiseaseTreatmentDatabase
            self.treatment_db = DiseaseTreatmentDatabase()
        else:
            self.treatment_db = disease_info_db
            
        self.detection_threshold = self.config_manager.get(
            'DETECTION_SCORE_THRESHOLD', 0.5, "model")
    
    def parse_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析病害检测结果，添加病害信息和治疗建议
        
        Args:
            result: 检测预测结果
            
        Returns:
            增强的病害检测结果
        """
        detections = result.get('detections', [])
        
        # 分析检测结果
        disease_summary = self._summarize_diseases(detections)
        severity_assessment = self._assess_severity(detections)
        treatment_recommendations = self._recommend_treatments(disease_summary)
        
        # 增强结果
        enhanced_result = {
            'raw_detection': result,
            'timestamp': datetime.now().isoformat(),
            'disease_summary': disease_summary,
            'severity_assessment': severity_assessment,
            'treatment_recommendations': treatment_recommendations,
            'prevention_tips': self._provide_prevention_tips(disease_summary)
        }
        
        return enhanced_result
    
    def _summarize_diseases(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """总结检测到的病害"""
        if not detections:
            return {"status": "健康", "diseases": []}
        
        # 统计所有检测到的病害
        diseases = {}
        for detection in detections:
            disease_name = detection['class_name']
            if disease_name not in diseases:
                diseases[disease_name] = {
                    'count': 1,
                    'avg_confidence': detection['score'],
                    'max_confidence': detection['score'],
                    'info': self._get_disease_info(disease_name)
                }
            else:
                disease_data = diseases[disease_name]
                disease_data['count'] += 1
                disease_data['avg_confidence'] = (disease_data['avg_confidence'] * 
                                                (disease_data['count'] - 1) + 
                                                detection['score']) / disease_data['count']
                disease_data['max_confidence'] = max(disease_data['max_confidence'], 
                                                  detection['score'])
        
        return {
            "status": "检测到病害" if diseases else "健康",
            "disease_count": len(diseases),
            "diseases": diseases
        }
    
    def _assess_severity(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估病害严重程度"""
        if not detections:
            return {"level": "健康", "description": "未检测到病害"}
        
        # 计算病害覆盖率和严重程度
        total_area = 0
        disease_area = 0
        max_confidence = 0
        
        # 假设图像尺寸信息在原始检测结果中
        if detections and 'image_size' in detections[0]:
            img_width = detections[0]['image_size']['width']
            img_height = detections[0]['image_size']['height']
            total_area = img_width * img_height
        
        for detection in detections:
            box = detection['box']
            score = detection['score']
            # 计算边界框面积
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            disease_area += area
            max_confidence = max(max_confidence, score)
        
        # 计算病害覆盖比例
        coverage_ratio = disease_area / total_area if total_area > 0 else 0
        
        # 基于覆盖率和置信度评估严重程度
        if coverage_ratio < 0.1 and max_confidence < 0.7:
            severity = {"level": "轻微", "description": "病害区域小且可能性较低"}
        elif coverage_ratio < 0.3 and max_confidence >= 0.7:
            severity = {"level": "中等", "description": "确认有病害但面积较小"}
        elif coverage_ratio >= 0.3 and max_confidence >= 0.8:
            severity = {"level": "严重", "description": "大面积病害，建议立即处理"}
        else:
            severity = {"level": "需要进一步检查", "description": "检测结果不确定"}
        
        severity["coverage"] = coverage_ratio
        severity["max_confidence"] = max_confidence
        
        return severity
    
    def _recommend_treatments(self, disease_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据病害提供治疗建议"""
        treatments = []
        plant_type = disease_summary.get('plant_type', '未知植物')
        diseases = disease_summary.get('diseases', {})
        
        for disease_name, disease_data in diseases.items():
            # 从治疗数据库中获取信息
            treatment_info = self.treatment_db.get_treatment(plant_type, disease_name)
            
            # 构建治疗建议
            treatment = {
                'disease': disease_name,
                'treatments': treatment_info.get('treatments', ["请咨询专业农业专家"]),
                'prevention': treatment_info.get('prevention', ["保持植株通风", "定期检查"]),
                'severity': treatment_info.get('severity', "未知"),
                'urgency': self._get_treatment_urgency(disease_data)
            }
            treatments.append(treatment)
        
        return treatments
    
    def _provide_prevention_tips(self, disease_summary: Dict[str, Any]) -> List[str]:
        """提供预防建议"""
        # 通用预防建议
        general_tips = [
            "保持适当间距以提高通风",
            "避免过度浇水，保持适当的土壤湿度",
            "定期检查植物健康状况",
            "使用健康的种子和植物材料"
        ]
        
        # 添加针对特定病害的预防建议
        specific_tips = []
        diseases = disease_summary.get('diseases', {})
        
        for disease_name in diseases:
            tips = self._get_prevention_tips(disease_name)
            specific_tips.extend(tips)
        
        return general_tips + specific_tips
    
    def _get_disease_info(self, disease_name: str) -> Dict[str, Any]:
        """获取病害详细信息"""
        # 先尝试从治疗数据库获取映射
        if hasattr(self, 'treatment_db'):
            plant_name, disease_type = self.treatment_db.map_dataset_class_to_db(disease_name)
            disease_info = self.treatment_db.get_treatment(plant_name, disease_type)
            if 'error' not in disease_info:
                return disease_info
        
        # 如果找不到映射，使用旧方法
        if disease_name in self.disease_info_db:
            return self.disease_info_db[disease_name]
        
        return {
            'name': disease_name,
            'description': f"暂无{disease_name}的详细信息"
        }
    
    def _get_treatment_methods(self, disease_name: str) -> List[str]:
        """获取特定病害的治疗方法"""
        # 这里可以添加针对不同病害的具体治疗方法
        # 实际应用中应该从疾病数据库获取
        return [
            "去除并销毁受感染的植物部分",
            "应用适当的杀菌剂或杀虫剂",
            "改善植物生长环境和通风条件"
        ]
    
    def _get_treatment_urgency(self, disease_data: Dict[str, Any]) -> str:
        """评估治疗紧急程度"""
        confidence = disease_data.get('max_confidence', 0)
        count = disease_data.get('count', 0)
        
        if confidence > 0.9 and count > 3:
            return "高，建议立即处理"
        elif confidence > 0.7:
            return "中，尽快处理"
        else:
            return "低，密切监控"
    
    def _get_prevention_tips(self, disease_name: str) -> List[str]:
        """获取特定病害的预防建议"""
        # 应该从疾病数据库获取针对特定病害的预防措施
        return [f"针对{disease_name}，建议定期检查植物"]


def create_result_parser(task_type: str) -> ResultParser:
    """
    创建适合任务类型的结果解析器
    
    Args:
        task_type: 任务类型('classification' 或 'detection')
        
    Returns:
        结果解析器实例
    """
    if task_type == 'classification':
        return ClassificationResultParser()
    elif task_type == 'detection':
        return DiseaseResultParser()
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")