# src/evaluation/model_interpreter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

from src.utils.config_manager import ConfigManager

class GradCAM:
    """Grad-CAM实现类，用于可视化模型关注区域"""
    
    def __init__(self, model: nn.Module, target_layer: str = None):
        """初始化GradCAM"""
        self.model = model
        self.model.eval()
        
        config_manager = ConfigManager()
        if target_layer is None:
            target_layer = config_manager.get('GRADCAM_TARGET_LAYER', 'backbone.layer4', "model")
            
        self.target_layer = self._get_layer(model, target_layer)
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _get_layer(self, model: nn.Module, layer_name: str) -> nn.Module:
        """获取模型中指定名称的层"""
        # 实现获取嵌套层级的逻辑
        parts = layer_name.split('.')
        curr_layer = model
        
        for part in parts:
            curr_layer = getattr(curr_layer, part)
            
        return curr_layer
    
    def _forward_hook(self, module, input, output):
        """前向传播钩子"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """反向传播钩子"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, image: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """生成热力图"""
        # 实现热力图生成逻辑
        config_manager = ConfigManager()
        
        # ...热力图生成代码...
        
        return np.zeros((224, 224))  # 占位符，实际实现应返回热力图

class ModelInterpreter:
    """模型解释器类，提供多种模型解释和可视化方法"""
    
    def __init__(self, model: nn.Module):
        """初始化模型解释器"""
        self.model = model
        self.config_manager = ConfigManager()
        
    def explain_prediction(self, image: Union[np.ndarray, torch.Tensor], 
                          class_idx: Optional[int] = None,
                          output_path: Optional[str] = None,
                          class_names: Optional[List[str]] = None) -> np.ndarray:
        """解释预测结果，生成可视化解释"""
        # 确保图像是张量
        if isinstance(image, np.ndarray):
            if image.max() > 1.0:  # 如果是0-255范围
                image = image / 255.0
            image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        
        # 生成热力图
        grad_cam = GradCAM(self.model, 
                          target_layer=self.config_manager.get('GRADCAM_TARGET_LAYER', 
                                                            'backbone.layer4', "model"))
        heatmap = grad_cam.generate_heatmap(image, class_idx)
        
        # 转换图像为numpy数组
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # 应用热力图
        heatmap_alpha = self.config_manager.get('HEATMAP_ALPHA', 0.6, "model")
        result = self._apply_heatmap(image, heatmap, alpha=heatmap_alpha)
        
        # 如果需要保存
        if output_path:
            plt.figure(figsize=self.config_manager.get('FIGURE_SIZE_MEDIUM', (10, 8), "data"))
            plt.imshow(result)
            plt.axis('off')
            
            if class_idx is not None and class_names and class_idx < len(class_names):
                plt.title(f'预测类别: {class_names[class_idx]}')
                
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        
        return result
    
    def _apply_heatmap(self, img: np.ndarray, heatmap: np.ndarray, 
                       alpha: float = 0.6, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """将热力图应用到原始图像上"""
        # 调整热力图尺寸以匹配图像
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # 归一化热力图
        heatmap = np.uint8(255 * heatmap)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # 转换为RGB（如果需要）
        if heatmap.shape[-1] == 3 and img.shape[-1] == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 将热力图叠加到原始图像
        superimposed = heatmap * alpha + img * 255 * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        return superimposed