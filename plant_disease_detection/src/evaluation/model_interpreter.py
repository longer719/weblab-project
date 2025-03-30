# src/evaluation/model_interpreter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.utils.config_manager import ConfigManager

# 导入pytorch-grad-cam相关库
try:
    from pytorch_grad_cam import GradCAM as OfficialGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    logging.warning("pytorch-grad-cam库未安装，某些功能将不可用。运行 'pip install pytorch-grad-cam' 安装。")
    GRADCAM_AVAILABLE = False

class GradCAM:
    """Grad-CAM实现类，用于可视化模型关注区域"""
    
    def __init__(self, model: nn.Module, target_layer: str = None):
        """初始化GradCAM"""
        self.model = model
        self.model.eval()
        
        # 获取配置
        self.config_manager = ConfigManager()
        
        # 如果未指定目标层，使用默认值
        if target_layer is None:
            target_layer = self.config_manager.get('GRADCAM_TARGET_LAYER', 'backbone.layer4', "model")
        
        # 获取目标层
        self.target_layer = self._get_layer(model, target_layer)
        if self.target_layer is None:
            logging.error(f"找不到指定的层: {target_layer}")
            raise ValueError(f"找不到指定的层: {target_layer}")
        
        # 初始化官方GradCAM
        if GRADCAM_AVAILABLE:
            self.cam_executor = OfficialGradCAM(
                model=self.model,
                target_layers=[self.target_layer],
                use_cuda=torch.cuda.is_available()
            )
        else:
            self.cam_executor = None
    
    def _get_layer(self, model: nn.Module, layer_name: str) -> nn.Module:
        """获取模型中指定名称的层"""
        try:
            # 处理嵌套层级
            parts = layer_name.split('.')
            curr_layer = model
            
            for part in parts:
                curr_layer = getattr(curr_layer, part)
            
            return curr_layer
        except AttributeError:
            # 如果找不到指定层，尝试自动查找最后一个卷积层
            logging.warning(f"找不到指定的层 '{layer_name}'，尝试自动查找最后一个卷积层...")
            return self._find_last_conv_layer(model)
    
    def _find_last_conv_layer(self, model: nn.Module) -> Optional[nn.Module]:
        """尝试找到模型中最后一个卷积层"""
        last_conv = None
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                last_conv = module
                break
        
        if last_conv is None:
            logging.error("无法自动找到卷积层")
            return None
        
        logging.info("自动选择了最后一个卷积层作为目标层")
        return last_conv
    
    def generate_heatmap(self, image: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        生成热力图
        
        Args:
            image: 输入图像张量 [1, C, H, W]
            class_idx: 目标类别索引，None则使用预测概率最高的类别
            
        Returns:
            热力图数组 [H, W]，值范围0-1
        """
        if not GRADCAM_AVAILABLE:
            logging.error("无法生成热力图: pytorch-grad-cam未安装")
            return np.zeros((image.shape[2], image.shape[3]), dtype=np.float32)
        
        # 确保输入是正确的形状
        if image.dim() == 3:
            image = image.unsqueeze(0)  # 添加批次维度
        
        # 确保图像在正确的设备上
        device = next(self.model.parameters()).device
        image = image.to(device)
        
        # 定义目标类别
        if class_idx is not None:
            targets = [ClassifierOutputTarget(class_idx)]
        else:
            # 让库自动选择最高概率类别
            targets = None
        
        # 生成CAM
        try:
            grayscale_cam = self.cam_executor(input_tensor=image, targets=targets)
            return grayscale_cam[0]  # 返回第一个批次的CAM
        except Exception as e:
            logging.error(f"生成热力图失败: {str(e)}")
            return np.zeros((image.shape[2], image.shape[3]), dtype=np.float32)


class ModelInterpreter:
    """模型解释器类，提供多种模型解释和可视化方法"""
    
    def __init__(self, model: nn.Module):
        """初始化模型解释器"""
        self.model = model
        self.config_manager = ConfigManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def explain_prediction(self, image: Union[np.ndarray, torch.Tensor, str, Path], 
                          class_idx: Optional[int] = None,
                          target_layer: Optional[str] = None,
                          output_path: Optional[str] = None,
                          class_names: Optional[List[str]] = None) -> np.ndarray:
        """
        解释预测结果，生成可视化解释
        
        Args:
            image: 输入图像(numpy数组、Torch张量或图像路径)
            class_idx: 目标类别索引，None则使用预测概率最高的类别
            target_layer: 要可视化的目标层，None则使用配置中的默认值
            output_path: 输出文件路径，None则仅显示不保存
            class_names: 类别名称列表，用于显示
            
        Returns:
             叠加了热力图的可视化图像(numpy数组)
        """
        # 转换输入图像为标准格式(Tensor和numpy)
        img_tensor, rgb_img = self._prepare_image(image)
        
        # 如果没有指定类别，使用模型预测的最高概率类别
        if class_idx is None:
            class_idx = self._predict_class(img_tensor)
            if class_idx is None:  # 预测失败
                return rgb_img
        
        # 创建GradCAM实例
        try:
            grad_cam = GradCAM(self.model, target_layer)
            # 生成热力图
            heatmap = grad_cam.generate_heatmap(img_tensor, class_idx)
        except Exception as e:
            logging.error(f"生成GradCAM热力图失败: {str(e)}")
            return rgb_img
        
        # 叠加热力图到原始图像
        heatmap_alpha = self.config_manager.get('HEATMAP_ALPHA', 0.6, "model")
        if GRADCAM_AVAILABLE:
            visualization = show_cam_on_image(rgb_img, heatmap, use_rgb=True, 
                                             image_weight=(1-heatmap_alpha))
        else:
            visualization = self._apply_heatmap(rgb_img, heatmap, alpha=heatmap_alpha)
        
        # 创建可视化图像
        plt.figure(figsize=(12, 5))
        
        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img)
        plt.title("原始图像")
        plt.axis('off')
        
        # 显示热力图叠加图像
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        
        # 添加标题和类别信息
        title = "Grad-CAM 热力图"
        if class_names is not None and class_idx < len(class_names):
            title += f"\n类别: {class_names[class_idx]}"
        else:
            title += f"\n类别索引: {class_idx}"
            
        plt.title(title)
        plt.axis('off')
        
        plt.tight_layout()
        
        # 保存或显示结果
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"已将可视化结果保存到: {output_path}")
        else:
            plt.show()
        
        return visualization
        
    def _prepare_image(self, image) -> Tuple[torch.Tensor, np.ndarray]:
        """
        准备输入图像，转换为模型所需格式
        
        Returns:
            Tuple[torch.Tensor, np.ndarray]: (模型输入张量, 显示用的RGB图像数组)
        """
        # 处理不同类型的输入
        if isinstance(image, str) or isinstance(image, Path):
            # 加载图像文件
            try:
                img_pil = Image.open(image).convert('RGB')
                rgb_img = np.array(img_pil) / 255.0
                img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)
            except Exception as e:
                logging.error(f"无法加载图像文件 {image}: {str(e)}")
                raise ValueError(f"无法加载图像文件: {str(e)}")
        
        elif isinstance(image, np.ndarray):
            # 处理numpy数组
            if image.max() > 1.0:  # 假设值范围是0-255
                rgb_img = image.astype(float) / 255.0
            else:
                rgb_img = image.copy()
            
            # 确保是RGB格式
            if len(image.shape) == 2:  # 灰度图
                rgb_img = np.stack([rgb_img, rgb_img, rgb_img], axis=2)
            elif image.shape[2] == 4:  # RGBA图
                rgb_img = rgb_img[:, :, :3]
                
            # 转换为tensor [C,H,W]
            img_tensor = torch.from_numpy(rgb_img.transpose((2, 0, 1))).float().unsqueeze(0)
            
        elif isinstance(image, torch.Tensor):
            # 处理torch张量
            if image.dim() == 3:  # [C,H,W]
                img_tensor = image.unsqueeze(0)
            elif image.dim() == 4:  # [N,C,H,W]
                img_tensor = image
            else:
                raise ValueError(f"不支持的张量维度: {image.dim()}, 应为3或4")
            
            # 转换为numpy数组用于显示
            if img_tensor.dim() == 4:
                rgb_img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
            else:
                rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
                
            # 处理可能的归一化
            if rgb_img.min() < 0 or rgb_img.max() > 1.0:
                # 假设是使用ImageNet均值和标准差归一化的
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                rgb_img = rgb_img * std + mean
                
            rgb_img = np.clip(rgb_img, 0, 1)
        
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
        
        return img_tensor, rgb_img
        
    def _predict_class(self, img_tensor: torch.Tensor) -> Optional[int]:
        """使用模型预测类别索引"""
        try:
            # 确保张量在正确的设备上
            img_tensor = img_tensor.to(self.device)
            
            # 获取模型的归一化参数
            mean = self.config_manager.get('mean', [0.485, 0.456, 0.406], "data")
            std = self.config_manager.get('std', [0.229, 0.224, 0.225], "data")
            
            # 应用归一化（如果尚未应用）
            normalized = img_tensor.clone()
            if normalized.min() >= 0 and normalized.max() <= 1:
                # 图像未归一化，应用归一化
                for i in range(3):
                    normalized[:, i] = (normalized[:, i] - mean[i]) / std[i]
            
            # 进行预测
            with torch.no_grad():
                outputs = self.model(normalized)
                
            # 获取预测的类别
            if isinstance(outputs, tuple):  # 某些模型可能返回多个输出
                outputs = outputs[0]
                
            # 应用softmax获取概率
            probs = F.softmax(outputs, dim=1)
            
            # 获取最高概率的类别
            _, pred_class = torch.max(probs, 1)
            class_idx = pred_class.item()
            
            logging.info(f"预测的类别索引: {class_idx}, 概率: {probs[0][class_idx].item():.4f}")
            return class_idx
            
        except Exception as e:
            logging.error(f"类别预测失败: {str(e)}")
            return None
    
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
        
        # 确保img在0-255范围
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
            
        # 将热力图叠加到原始图像
        superimposed = heatmap * alpha + img * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        return superimposed