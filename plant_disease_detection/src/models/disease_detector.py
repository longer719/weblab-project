# src/models/disease_detector.py

import torch
import torch.nn as nn
import torchvision
import torchvision.models.detection as detection
import torchvision.ops as ops
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone  # 添加这行导入
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import logging

from src.models.base_model import BaseModel

# 添加统一配置管理导入
from src.utils.config_manager import ConfigManager
from src.utils.config_helpers import get_config_value

class DiseaseDetector(BaseModel):
    """
    植物病害检测器
    
    基于Faster R-CNN和Mask R-CNN架构，专门针对植物病害检测优化，
    包括小病斑检测增强和病害区域分割功能。
    """
    
    def __init__(self, config):
        """
        初始化病害检测器
        
        Args:
            config: 配置对象、字典或None（如果为None则使用ConfigManager）
        """
        super().__init__(config)
        
        # 创建ConfigManager实例
        self.config_manager = ConfigManager()
        
        # 读取配置（同时支持旧式配置对象和新式ConfigManager）
        if isinstance(config, dict):
            # 从字典获取配置，回退到ConfigManager
            self.num_classes = config.get('num_classes', 
                self.config_manager.get('detector_num_classes', 5, 'model'))
            self.backbone_name = config.get('backbone', 
                self.config_manager.get('detector_backbone', 'resnet50', 'model'))
            self.pretrained = config.get('pretrained', 
                self.config_manager.get('PRETRAINED', True, 'model'))
            self.enable_mask = config.get('enable_mask', 
                self.config_manager.get('enable_mask', False, 'model'))
            self.min_size = config.get('min_size', 
                self.config_manager.get('min_size', 800, 'model'))
            self.max_size = config.get('max_size', 
                self.config_manager.get('max_size', 1333, 'model'))
            
            # 针对植物病害的小物体检测优化参数
            self.rpn_anchor_sizes = config.get('anchor_sizes', 
                self.config_manager.get('anchor_sizes', ((16, 32, 64, 128, 256),), 'model'))
            self.rpn_anchor_ratios = config.get('anchor_ratios', 
                self.config_manager.get('anchor_ratios', ((0.5, 1.0, 2.0),), 'model'))
            
            # 检测阈值
            self.box_score_thresh = config.get('confidence_threshold', 
                self.config_manager.get('confidence_threshold', 0.5, 'model'))
            self.box_nms_thresh = config.get('nms_threshold', 
                self.config_manager.get('nms_threshold', 0.45, 'model'))
            self.box_detections_per_img = config.get('detections_per_img', 
                self.config_manager.get('detections_per_img', 100, 'model'))
                
            # 构建兼容原始实现所需的detector_config
            self.detector_config = {
                'num_classes': self.num_classes,
                'backbone': self.backbone_name,
                'enable_mask': self.enable_mask,
                'min_size': self.min_size,
                'max_size': self.max_size,
                'anchor_sizes': self.rpn_anchor_sizes,
                'anchor_ratios': self.rpn_anchor_ratios,
                'confidence_threshold': self.box_score_thresh,
                'nms_threshold': self.box_nms_thresh,
                'detections_per_img': self.box_detections_per_img
            }
        elif hasattr(config, 'DETECTOR_CONFIG'):
            # 兼容原始实现
            self.detector_config = config.DETECTOR_CONFIG
            self.num_classes = self.detector_config.get('num_classes', 5)
            self.backbone_name = self.detector_config.get('backbone', 'resnet50')
            self.pretrained = getattr(config, 'PRETRAINED', True)
            self.enable_mask = self.detector_config.get('enable_mask', False)
            self.min_size = self.detector_config.get('min_size', 800)
            self.max_size = self.detector_config.get('max_size', 1333)
            
            # 针对植物病害的小物体检测优化参数
            self.rpn_anchor_sizes = self.detector_config.get(
                'anchor_sizes', 
                ((16, 32, 64, 128, 256),)  # 默认包括更小的anchor尺寸
            )
            self.rpn_anchor_ratios = self.detector_config.get(
                'anchor_ratios', 
                ((0.5, 1.0, 2.0),)
            )
            
            # 检测阈值
            self.box_score_thresh = self.detector_config.get('confidence_threshold', 0.5)
            self.box_nms_thresh = self.detector_config.get('nms_threshold', 0.45)
            self.box_detections_per_img = self.detector_config.get('detections_per_img', 100)
        else:
            # 从ConfigManager获取配置
            self.num_classes = self.config_manager.get('detector_num_classes', 5, 'model')
            self.backbone_name = self.config_manager.get('detector_backbone', 'resnet50', 'model')
            self.pretrained = self.config_manager.get('PRETRAINED', True, 'model')
            self.enable_mask = self.config_manager.get('enable_mask', False, 'model')
            self.min_size = self.config_manager.get('min_size', 800, 'model')
            self.max_size = self.config_manager.get('max_size', 1333, 'model')
            
            # 针对植物病害的小物体检测优化参数
            self.rpn_anchor_sizes = self.config_manager.get('anchor_sizes', ((16, 32, 64, 128, 256),), 'model')
            self.rpn_anchor_ratios = self.config_manager.get('anchor_ratios', ((0.5, 1.0, 2.0),), 'model')
            
            # 检测阈值
            self.box_score_thresh = self.config_manager.get('confidence_threshold', 0.5, 'model')
            self.box_nms_thresh = self.config_manager.get('nms_threshold', 0.45, 'model')
            self.box_detections_per_img = self.config_manager.get('detections_per_img', 100, 'model')
            
            # 构建兼容原始实现所需的detector_config
            self.detector_config = {
                'num_classes': self.num_classes,
                'backbone': self.backbone_name,
                'enable_mask': self.enable_mask,
                'min_size': self.min_size,
                'max_size': self.max_size,
                'anchor_sizes': self.rpn_anchor_sizes,
                'anchor_ratios': self.rpn_anchor_ratios,
                'confidence_threshold': self.box_score_thresh,
                'nms_threshold': self.box_nms_thresh,
                'detections_per_img': self.box_detections_per_img
            }
        
        # 构建检测器
        self._build_detector()
    
    def _build_detector(self):
        """构建检测器网络"""
        # 选择检测器类型
        if self.enable_mask:
            # Mask R-CNN (包含分割功能)
            self._build_mask_rcnn()
        else:
            # Faster R-CNN (仅检测)
            self._build_faster_rcnn()
    
    def _build_faster_rcnn(self):
        """构建Faster R-CNN检测器"""
        # 选择权重
        if self.pretrained:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            weights_backbone = None  # 使用默认权重的backbone
        else:
            weights = None
            weights_backbone = None
        
        # 直接使用默认配置创建检测器，不自定义锚框
        self.detector = detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            box_score_thresh=self.box_score_thresh,
            box_nms_thresh=self.box_nms_thresh,
            box_detections_per_img=self.box_detections_per_img,
            min_size=self.min_size,
            max_size=self.max_size
        )
        
        # 修改分类头以适应类别数量
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
            in_features, 
            self.num_classes
        )
    
    def _build_mask_rcnn(self):
        """构建Mask R-CNN检测器，增加分割功能"""
        # 选择权重
        if self.pretrained:
            weights = detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            weights_backbone = None
        else:
            weights = None
            weights_backbone = None
            
        # 创建基本的Mask R-CNN模型
        self.detector = detection.maskrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            box_score_thresh=self.box_score_thresh,
            box_nms_thresh=self.box_nms_thresh,
            box_detections_per_img=self.box_detections_per_img,
            min_size=self.min_size,
            max_size=self.max_size
        )
        
        # 配置RPN部分，优化小物体检测
        anchor_generator = detection.rpn.AnchorGenerator(
            sizes=self.rpn_anchor_sizes,
            aspect_ratios=self.rpn_anchor_ratios
        )
        self.detector.rpn.anchor_generator = anchor_generator
        
        # 优化RPN头部
        self._optimize_rpn_head()
        
        # 替换分类头，使用自定义类别数
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
            in_features, 
            self.num_classes
        )
        
        # 替换掩码预测器
        in_features_mask = self.detector.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.detector.roi_heads.mask_predictor = detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            self.num_classes
        )
    
    def _optimize_rpn_head(self):
        """优化RPN头部，提高小病斑的检测能力"""
        # 使用FPN后的RPN已经有一定能力处理多尺度特征
        # 这里可以进一步优化参数或结构
        
        # 使用更小的NMS阈值，避免小物体被抑制
        self.detector.rpn.nms_thresh = 0.7
        
        # 增加RPN生成的proposal数量
        self.detector.rpn.pre_nms_top_n_train = 2000
        self.detector.rpn.post_nms_top_n_train = 1000
        self.detector.rpn.pre_nms_top_n_test = 1000
        self.detector.rpn.post_nms_top_n_test = 500
    
    def forward(self, x, targets=None):
        """
        模型前向传播
        
        Args:
            x: 输入图像批次
            targets: 训练目标 (训练模式下必需)
            
        Returns:
            检测结果或损失
        """
        if self.training and targets is None:
            raise ValueError("在训练模式下必须提供目标")
            
        return self.detector(x, targets)
    
    @torch.no_grad()
    def predict(self, x):
        """
        预测函数，用于推理阶段
        
        Args:
            x: 输入图像或图像批次
            
        Returns:
            检测结果列表
        """
        self.eval()
        
        # 确保输入是批次形式
        if not isinstance(x, list) and x.dim() == 3:
            x = [x.unsqueeze(0)]
        elif not isinstance(x, list) and x.dim() == 4:
            x = [x[i] for i in range(x.size(0))]
            
        # 执行预测
        result = self.detector(x)
        return result
    
    def predict_with_heatmap(self, x):
        """
        预测并生成注意力热图
        
        Args:
            x: 输入图像或批次
            
        Returns:
            检测结果和热图
        """
        # 目前实现的简化版本，仅返回预测结果
        # 完整的热图生成需要更多代码和可视化工具
        return self.predict(x), None
    
    def get_segmentation_masks(self, result, threshold=0.5):
        """
        从结果中提取分割掩码
        
        Args:
            result: 模型预测结果
            threshold: 掩码阈值
            
        Returns:
            处理后的分割掩码
        """
        if not self.enable_mask:
            return None
            
        masks = []
        for pred in result:
            if 'masks' in pred:
                # 提取所有掩码
                pred_masks = pred['masks']
                if pred_masks.shape[0] > 0:
                    # 二值化掩码
                    binary_masks = (pred_masks > threshold).squeeze(1).cpu().numpy()
                    masks.append(binary_masks)
                else:
                    masks.append(None)
            else:
                masks.append(None)
                
        return masks
    
    def process_results(self, results, image_size=None, min_area=25):
        """
        处理检测结果，过滤小面积病斑
        
        Args:
            results: 模型预测结果
            image_size: 原始图像尺寸
            min_area: 最小病斑面积
            
        Returns:
            处理后的检测结果
        """
        processed_results = []
        
        for result in results:
            # 复制结果
            processed = {k: v.clone().detach() if isinstance(v, torch.Tensor) else v 
                         for k, v in result.items()}
            
            # 过滤小面积框
            if len(processed['boxes']) > 0:
                boxes = processed['boxes']
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                keep = areas >= min_area
                
                processed['boxes'] = boxes[keep]
                processed['scores'] = processed['scores'][keep]
                processed['labels'] = processed['labels'][keep]
                
                if 'masks' in processed and processed['masks'].shape[0] > 0:
                    processed['masks'] = processed['masks'][keep]
            
            processed_results.append(processed)
        
        return processed_results
    
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'num_classes': self.num_classes,
                'backbone': self.backbone_name,
                'enable_mask': self.enable_mask
            }
        }, path)
        
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # 更新配置
        if 'config' in checkpoint:
            self.num_classes = checkpoint['config'].get('num_classes', self.num_classes)
            self.backbone_name = checkpoint['config'].get('backbone', self.backbone_name)
            self.enable_mask = checkpoint['config'].get('enable_mask', self.enable_mask)