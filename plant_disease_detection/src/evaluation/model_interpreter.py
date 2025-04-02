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
    # 导入标准的 ClassifierOutputTarget
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    logging.warning("pytorch-grad-cam库未安装，Grad-CAM 功能将不可用。运行 'pip install pytorch-grad-cam' 安装。")
    GRADCAM_AVAILABLE = False

logger = logging.getLogger(__name__)


# --- 新增：模型包装器 ---
class DetectionModelWrapperForCAM(nn.Module):
    """
    包装检测模型，使其输出类似分类模型的格式，供 GradCAM 使用。
    它提取特定类别的最高分数。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        调用原始模型，然后尝试提取所有检测框的分数和标签，
        并模拟成分类输出格式 [batch_size, num_classes]。
        注意：这种模拟对于 GradCAM 可能不是最优的，但可以尝试。
        更简单的方式是仅关注最高分数的类别。
        """
        # 调用原始模型获取输出列表（假设 batch_size=1）
        outputs = self.model(x) # x 已经是 [1, C, H, W]

        if not outputs or len(outputs) == 0:
            logger.warning("模型包装器：原始模型未返回输出")
            # 需要知道类别数才能返回正确形状的零张量
            num_classes = getattr(self.model, 'num_classes', 38) # 尝试获取类别数，默认38
            device = next(self.model.parameters()).device
            return torch.zeros((1, num_classes), device=device)

        output_dict = outputs[0] # 取第一个样本的结果
        scores = output_dict.get('scores')
        labels = output_dict.get('labels')

        if scores is None or labels is None or len(scores) == 0:
            logger.warning("模型包装器：输出字典缺少scores或labels，或为空")
            num_classes = getattr(self.model, 'num_classes', 38)
            device = next(self.model.parameters()).device
            return torch.zeros((1, num_classes), device=device)

        # 创建一个模拟的分类输出张量 [1, num_classes]
        # 将每个类别的最高分数填入对应位置
        num_classes = getattr(self.model, 'num_classes', 38) # 再次确认类别数
        device = scores.device
        class_scores = torch.zeros(num_classes, device=device)

        unique_labels = torch.unique(labels)
        for label_id in unique_labels:
            if 0 <= label_id < num_classes: # 确保标签有效
                 mask = (labels == label_id)
                 if torch.any(mask):
                     max_score_for_label = torch.max(scores[mask])
                     # 注意：PyTorch的类别索引通常从0开始，但检测模型可能从1开始
                     # 检查你的 num_classes 是否包含背景类
                     # 假设模型输出的 label 是从 0 到 num_classes-1
                     class_scores[label_id] = max_score_for_label
                 else:
                      class_scores[label_id] = 0.0 # 以防万一mask全为False

        # 返回模拟的分类输出，增加批次维度
        return class_scores.unsqueeze(0) # [1, num_classes]

# --- 修改 GradCAM 类 ---
class GradCAM:
    """Grad-CAM实现类"""
    def __init__(self, model: nn.Module, target_layer_name: str = None):
        self.original_model = model # 保存原始模型
        self.original_model.eval()
        self.config_manager = ConfigManager()

        if target_layer_name is None:
            target_layer_name = self.config_manager.get('DETECTOR_GRADCAM_TARGET_LAYER', 'detector.backbone.body.layer4', "model")
            logger.info(f"使用默认目标层: {target_layer_name}")

        # --- 修改：在包装后的模型上查找目标层 ---
        self.model_wrapper = DetectionModelWrapperForCAM(self.original_model)
        self.target_layer_module = self._get_layer(self.original_model, target_layer_name) # 查找层仍然在原始模型上进行
        # --- 修改结束 ---

        if self.target_layer_module is None:
            logging.warning(f"找不到指定的层 '{target_layer_name}'，尝试自动查找...")
            self.target_layer_module = self._find_last_conv_layer(self.original_model) # 在原始模型上查找
            if self.target_layer_module is None:
                raise ValueError("无法找到合适的目标层用于GradCAM")

        if GRADCAM_AVAILABLE:
            try:
                # --- 修改：在包装器上初始化 GradCAM ---
                self.cam_executor = OfficialGradCAM(
                    model=self.model_wrapper, # 使用包装器
                    target_layers=[self.target_layer_module]
                )
                logging.info("官方 pytorch-grad-cam 执行器初始化成功 (使用模型包装器)")
                # --- 修改结束 ---
            except Exception as e:
                logging.error(f"初始化官方 GradCAM 失败: {e}")
                self.cam_executor = None
        else:
            self.cam_executor = None

    def _get_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """获取模型中指定名称的层"""
        try:
            parts = layer_name.split('.')
            curr_layer = model
            name_prefix = ""
            for part in parts:
                if hasattr(curr_layer, part):
                    curr_layer = getattr(curr_layer, part)
                    name_prefix += part + "."
                else:
                    try:
                        idx = int(part)
                        if isinstance(curr_layer, (nn.Sequential, nn.ModuleList)) and 0 <= idx < len(curr_layer):
                            curr_layer = curr_layer[idx]
                            name_prefix += f"[{idx}]."
                        else:
                            raise AttributeError
                    except (ValueError, AttributeError):
                        logging.warning(f"在 {name_prefix[:-1]} 中找不到属性或索引 '{part}'")
                        return None
            logging.debug(f"成功找到层 '{layer_name}' 的模块实例")
            return curr_layer
        except Exception as e:
            logging.error(f"查找层 '{layer_name}' 时出错: {e}")
            return None

    def _find_last_conv_layer(self, model: nn.Module) -> Optional[nn.Module]:
         """尝试找到模型中最后一个卷积层"""
         last_conv_module = None
         last_conv_name = ""
         for name, module in model.named_modules():
             # 查找 Conv2d，或者常见特征提取器的最后阶段
             if isinstance(module, nn.Conv2d):
                 # 优先选择看起来像bottleneck或basicblock内部的最后一个卷积
                 # 但简单的最后一个卷积层通常也有效
                 last_conv_module = module
                 last_conv_name = name
             # 可以添加对特定层名称的检查，例如 'layer4' in name

         if last_conv_module is None:
             logging.error("无法自动找到卷积层")
             return None

         logging.info(f"自动选择了最后一个卷积层 '{last_conv_name}' 作为目标层")
         return last_conv_module

    def generate_heatmap(self, image: torch.Tensor, class_idx: int) -> Optional[np.ndarray]:
        """生成热力图"""
        if not GRADCAM_AVAILABLE or self.cam_executor is None:
            logging.error("无法生成热力图: pytorch-grad-cam 未安装或初始化失败")
            return None

        if image.dim() == 3:
            image = image.unsqueeze(0)

        device = next(self.original_model.parameters()).device # 从原始模型获取设备
        image = image.to(device)

        if class_idx is None:
            logging.error("必须提供目标类别索引 (class_idx)")
            return None

        # --- 修改：使用标准的 ClassifierOutputTarget ---
        targets = [ClassifierOutputTarget(class_idx)]
        # --- 修改结束 ---

        try:
            logging.info(f"正在为类别索引 {class_idx} 生成热力图...")
            # --- 修改： GradCAM 在包装器上运行 ---
            grayscale_cam = self.cam_executor(input_tensor=image, targets=targets)
            # --- 修改结束 ---

            if grayscale_cam is None or len(grayscale_cam) == 0:
                 logging.error("Grad-CAM 执行器未能生成热力图数组")
                 return None

            logging.info(f"热力图生成成功，原始形状: {grayscale_cam.shape}")
            # 提取 CAM 结果并确保是 H x W
            cam_result = grayscale_cam[0, :] # cam_executor 返回 [batch, H, W]
            if cam_result.ndim != 2:
                logger.error(f"预期二维热力图，但得到 {cam_result.ndim} 维，形状: {cam_result.shape}")
                return None

            logger.info(f"最终热力图形状: {cam_result.shape}")
            return cam_result
        except Exception as e:
            logging.error(f"生成热力图过程中失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None

# --- ModelInterpreter 类保持不变，除了其调用 GradCAM 的方式 ---
class ModelInterpreter:
    """模型解释器类"""
    def __init__(self, model: nn.Module):
        self.model = model # 这里仍然是原始模型
        self.config_manager = ConfigManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 注意：我们将原始模型传递给 GradCAM 类，它内部会创建包装器
        self.grad_cam_instance = None
        self.last_target_layer_name = None
        # 确保原始模型在正确的设备上并处于评估模式
        self.model.to(self.device)
        self.model.eval()

    def explain_prediction(self, image: Union[np.ndarray, torch.Tensor, str, Path],
                          class_idx: int,
                          target_layer: Optional[str] = None,
                          output_path: Optional[str] = None,
                          class_names: Optional[Dict[str, str]] = None) -> Optional[np.ndarray]:
        if class_idx is None:
            logging.error("必须提供目标类别索引 (class_idx) 才能生成 Grad-CAM。")
            return None

        try:
            img_tensor, rgb_img_float = self._prepare_image(image)
            if img_tensor is None or rgb_img_float is None:
                return None
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return None

        heatmap = None
        try:
            target_layer_name = target_layer if target_layer else \
                self.config_manager.get('DETECTOR_GRADCAM_TARGET_LAYER', 'detector.backbone.body.layer4', "model")

            # 管理 GradCAM 实例
            if self.grad_cam_instance is None or self.last_target_layer_name != target_layer_name:
                 logging.info(f"为目标层 '{target_layer_name}' 创建 GradCAM 实例...")
                 # 传递原始模型给 GradCAM 类
                 self.grad_cam_instance = GradCAM(self.model, target_layer_name)
                 self.last_target_layer_name = target_layer_name

            if self.grad_cam_instance:
                heatmap = self.grad_cam_instance.generate_heatmap(img_tensor, class_idx)

        except ValueError as ve:
             logger.error(f"初始化 GradCAM 实例失败: {ve}")
             return None
        except Exception as e:
            logging.error(f"生成 GradCAM 热力图过程中出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None

        if heatmap is None:
            logging.error("未能生成有效的 Grad-CAM 热力图。")
            return None

        visualization = None
        try:
            heatmap_alpha = self.config_manager.get('HEATMAP_ALPHA', 0.6, "model")
            if GRADCAM_AVAILABLE:
                visualization = show_cam_on_image(rgb_img_float, heatmap, use_rgb=True,
                                                 image_weight=(1-heatmap_alpha))
            else:
                rgb_img_uint8 = (rgb_img_float * 255).astype(np.uint8)
                visualization = self._apply_heatmap(rgb_img_uint8, heatmap, alpha=heatmap_alpha)

            if visualization is None:
                raise ValueError("热图叠加失败")

            if visualization.dtype != np.uint8:
                visualization = np.clip(visualization, 0, 255).astype(np.uint8)

        except Exception as e:
            logging.error(f"叠加热力图失败: {e}")
            return None

        if output_path is None:
            return visualization
        else:
            try:
                plt.figure(figsize=(8, 8)) # 调整大小以适应单个图像
                plt.imshow(visualization) # 直接显示叠加后的图像

                title = "Grad-CAM 模型关注区域"
                if class_names is not None and isinstance(class_names, dict):
                    name = class_names.get(str(class_idx), f"类别 {class_idx}")
                    title += f"\n预测: {name}"
                else:
                     title += f"\n预测类别索引: {class_idx}"
                plt.title(title)
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(output_path, bbox_inches='tight', dpi=150) # 降低 DPI 可能有助于保存速度
                plt.close()
                logging.info(f"已将可视化结果保存到: {output_path}")
                return visualization
            except Exception as e:
                logging.error(f"保存可视化图像失败: {e}")
                return None

    # --- _prepare_image 和 _apply_heatmap 方法保持上次修改后的版本 ---
    def _prepare_image(self, image: Union[np.ndarray, torch.Tensor, str, Path]) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        """
        准备输入图像。

        Returns:
            Tuple[torch.Tensor, np.ndarray]: (模型输入张量[1,C,H,W], 显示用的RGB图像数组[H,W,C] 范围[0,1] float32)
            如果处理失败，返回 (None, None)
        """
        try:
            # 定义用于可视化的转换（不进行标准化）
            vis_transform = transforms.Compose([
                 transforms.Resize((224, 224)),
                 transforms.ToTensor(), # 输出 [C, H, W], 范围 [0, 1] float32
            ])
            # 定义用于模型输入的转换（包含标准化）
            model_input_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # 处理不同类型的输入
            if isinstance(image, str) or isinstance(image, Path):
                img_pil = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                # 假设输入是 HWC
                if image.shape[2] == 3:
                    img_pil = Image.fromarray(image).convert('RGB')
                elif image.shape[2] == 4: # RGBA
                     img_pil = Image.fromarray(image).convert('RGB')
                elif len(image.shape) == 2: # Grayscale
                     img_pil = Image.fromarray(image).convert('RGB')
                else:
                    raise ValueError(f"无法处理的Numpy数组形状: {image.shape}")
            elif isinstance(image, Image.Image):
                img_pil = image.convert('RGB')
            elif isinstance(image, torch.Tensor):
                 # 如果输入已经是Tensor，我们需要反标准化以获取用于显示的图像
                 if image.dim() == 4: # NCHW
                     img_tensor_chw = image[0].cpu() # 取第一个样本
                 elif image.dim() == 3: # CHW
                     img_tensor_chw = image.cpu()
                 else:
                     raise ValueError(f"不支持的张量维度: {image.dim()}, 应为3或4")

                 # 逆标准化
                 inv_normalize = transforms.Normalize(
                     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                     std=[1/0.229, 1/0.224, 1/0.225]
                 )
                 img_for_vis_tensor = inv_normalize(img_tensor_chw)
                 img_for_vis = img_for_vis_tensor.permute(1, 2, 0).numpy() # CHW -> HWC
                 img_for_vis = np.clip(img_for_vis, 0, 1).astype(np.float32) # 裁剪到 [0, 1], float32

                 # 模型输入tensor直接使用传入的tensor
                 model_input_tensor = img_tensor_chw.unsqueeze(0) if img_tensor_chw.dim() == 3 else img_tensor_chw # 确保有批次维度
                 return model_input_tensor.to(self.device), img_for_vis
            else:
                raise TypeError(f"不支持的图像类型: {type(image)}")

            # 对于 PIL 或 Numpy 输入
            model_input_tensor = model_input_transform(img_pil).unsqueeze(0).to(self.device) # NCHW
            img_for_vis_tensor = vis_transform(img_pil) # CHW, [0, 1] float32
            img_for_vis = img_for_vis_tensor.permute(1, 2, 0).numpy().astype(np.float32) # HWC, [0, 1] float32

            return model_input_tensor, img_for_vis

        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return None, None

    def _apply_heatmap(self, img_uint8: np.ndarray, heatmap: np.ndarray,
                       alpha: float = 0.6, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """将热力图应用到原始图像上 (输入为 uint8 HWC RGB)"""
        if img_uint8.dtype != np.uint8:
            logging.warning(f"_apply_heatmap 预期输入为 uint8，但接收到 {img_uint8.dtype}。尝试转换...")
            if np.issubdtype(img_uint8.dtype, np.floating) and img_uint8.max() <= 1.0 and img_uint8.min() >= 0.0:
                 img_uint8 = (img_uint8 * 255).astype(np.uint8)
            else:
                 try:
                     img_uint8 = np.clip(img_uint8, 0, 255).astype(np.uint8)
                 except ValueError:
                     logging.error("无法将输入图像转换为 uint8")
                     return img_uint8

        if len(img_uint8.shape) != 3 or img_uint8.shape[2] != 3:
            logging.error(f"输入图像形状不正确: {img_uint8.shape}，需要 HWC RGB")
            return img_uint8

        try:
            heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
        except cv2.error as e:
            logging.error(f"调整热力图尺寸失败: {e}")
            return img_uint8

        heatmap_norm = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap) # BGR

        # 将原始图像从 RGB 转换为 BGR
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        superimposed_bgr = cv2.addWeighted(heatmap_colored, alpha, img_bgr, 1 - alpha, 0)

        # 将结果转换回 RGB
        superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)

        return superimposed_rgb