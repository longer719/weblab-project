# src/utils/augmentation.py
# 用于数据增强的模块
# 该模块提供了多种数据增强方法，包括随机翻转、旋转、颜色抖动等

import cv2
import numpy as np
import random
import torch
from typing import Dict, List, Union, Optional, Tuple
from torchvision import transforms
import torchvision.transforms.functional as TF  # 添加这一行导入
from PIL import Image, ImageOps, ImageEnhance

# 添加新的导入
from src.utils.config_manager import ConfigManager
from src.utils.config_helpers import get_config_value

class AugmentationPipeline:
    """
    数据增强管道类
    
    提供多种数据增强策略和配置选项，用于训练和验证过程中的图像预处理。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化增强管道
        
        Args:
            config: 增强配置字典，如果为None则使用默认配置
        """
        # 使用ConfigManager
        self.config_manager = ConfigManager()
        self.config = config or {}
        
        # 使用get_config_value从配置获取默认值
        self.default_mean = get_config_value(config, 'NORMALIZATION_MEAN', 
            [0.485, 0.456, 0.406], "data")
        self.default_std = get_config_value(config, 'NORMALIZATION_STD',
            [0.229, 0.224, 0.225], "data")
    
    def get_train_transforms(self, img_size: tuple = None) -> transforms.Compose:
        """
        获取训练数据增强转换
        
        Args:
            img_size: 目标图像尺寸
            
        Returns:
            数据增强转换组合
        """
        # 从配置获取图像大小
        if img_size is None:
            img_size = get_config_value(self.config, 'IMAGE_SIZE', (224, 224))
        
        # 读取配置参数或使用默认值
        hflip_prob = get_config_value(self.config, 'hflip_prob', 0.5)
        vflip_prob = get_config_value(self.config, 'vflip_prob', 0.0)
        rotation_degrees = get_config_value(self.config, 'rotation_degrees', 15)
        brightness = get_config_value(self.config, 'brightness', 0.35)  # 默认值从0.2改为0.35
        contrast = get_config_value(self.config, 'contrast', 0.3)       # 默认值从0.2改为0.3
        saturation = get_config_value(self.config, 'saturation', 0.3)   # 默认值从0.2改为0.3
        hue = get_config_value(self.config, 'hue', 0.1)
        # 新增三个配置参数
        autocontrast_prob = get_config_value(self.config, 'autocontrast_prob', 0.2)
        equalize_prob = get_config_value(self.config, 'equalize_prob', 0.1) 
        gamma_prob = get_config_value(self.config, 'gamma_prob', 0.3)
        gamma_range = get_config_value(self.config, 'gamma_range', (0.7, 1.3))
        crop_scale = get_config_value(self.config, 'crop_scale', (0.8, 1.0))
        crop_ratio = get_config_value(self.config, 'crop_ratio', (0.75, 1.33))

        # --- 新增几何变换和噪声参数 ---
        affine_prob = get_config_value(self.config, 'affine_prob', 0.5)
        affine_degrees = get_config_value(self.config, 'affine_degrees', 15)
        affine_translate = get_config_value(self.config, 'affine_translate', (0.08, 0.08))
        affine_scale = get_config_value(self.config, 'affine_scale', (0.9, 1.1))
        affine_shear = get_config_value(self.config, 'affine_shear', 10)

        blur_prob = get_config_value(self.config, 'blur_prob', 0.3)
        blur_kernel_size = get_config_value(self.config, 'blur_kernel_size', 5)
        blur_sigma = get_config_value(self.config, 'blur_sigma', (0.1, 2.0))

        noise_prob = get_config_value(self.config, 'noise_prob', 0.2)
        noise_std = get_config_value(self.config, 'noise_std', (0.01, 0.05))

        erasing_prob = get_config_value(self.config, 'erasing_prob', 0.2)
        erasing_scale = get_config_value(self.config, 'erasing_scale', (0.02, 0.15))
        erasing_ratio = get_config_value(self.config, 'erasing_ratio', (0.3, 3.3))
        
        # 是否使用高级增强
        use_advanced = get_config_value(self.config, 'use_advanced', True)
        erasing_prob = get_config_value(self.config, 'erasing_prob', 0.2)
        
        # 构建基础增强流程
        transform_list = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=hflip_prob),
        ]
        
        # 可选垂直翻转
        if vflip_prob > 0:
            transform_list.append(transforms.RandomVerticalFlip(p=vflip_prob))
        
        # 旋转和剪裁
        transform_list.extend([
            transforms.RandomRotation(rotation_degrees),
            transforms.RandomResizedCrop(
                img_size[0] if isinstance(img_size, tuple) else img_size,
                scale=crop_scale,
                ratio=crop_ratio
            ),
            # 新增：随机仿射变换
            transforms.RandomAffine(
                degrees=affine_degrees, 
                translate=affine_translate,
                scale=affine_scale,
                shear=affine_shear,
                fill=0
            ) if random.random() < affine_prob else transforms.Lambda(lambda x: x),
            # 颜色调整（保持现有）
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ),
            # 保持现有的自动对比度、均衡化和Gamma调整
            transforms.RandomAutocontrast(p=autocontrast_prob),
            transforms.RandomEqualize(p=equalize_prob),
            transforms.Lambda(lambda img: TF.adjust_gamma(
                img, 
                random.uniform(gamma_range[0], gamma_range[1])
            ) if random.random() < gamma_prob else img),
            # 新增：高斯模糊
            transforms.GaussianBlur(
                kernel_size=blur_kernel_size,  
                sigma=blur_sigma
            ) if random.random() < blur_prob else transforms.Lambda(lambda x: x),
        ])
        
        # 转换为张量并标准化
        mean = get_config_value(self.config, 'mean', self.default_mean)
        std = get_config_value(self.config, 'std', self.default_std)
        
        transform_list.extend([
            transforms.ToTensor(),
            # 新增：高斯噪声
            transforms.Lambda(lambda x: torch.clamp(
                x + torch.randn_like(x) * random.uniform(noise_std[0], noise_std[1]), 
                0, 1
            ) if random.random() < noise_prob else x),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # 高级增强方法
        if use_advanced:
            # 新增：随机擦除（位置从原来的erasing_prob移到这里，参数更丰富）
            if erasing_prob > 0:
                transform_list.append(
                    transforms.RandomErasing(
                        p=erasing_prob,
                        scale=erasing_scale,
                        ratio=erasing_ratio,
                        value='random'  # 使用随机值填充
                    )
                )
        
        return transforms.Compose(transform_list)
    
    def get_valid_transforms(self, img_size: tuple = None) -> transforms.Compose:
        """
        获取验证数据转换
        
        Args:
            img_size: 目标图像尺寸
            
        Returns:
            用于验证的转换组合
        """
        # 从配置获取图像大小
        if img_size is None:
            img_size = get_config_value(self.config, 'IMAGE_SIZE', (224, 224))
            
        # 验证集只需要基本的大小调整、转换和标准化
        center_crop = get_config_value(self.config, 'center_crop', True)
        
        # 计算调整大小的目标尺寸 - 约为原尺寸的1.14倍（如224→256）
        target_size = int((img_size[0] if isinstance(img_size, tuple) else img_size) * 1.14)
        
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(target_size)  # 修改这里，使用更大的尺寸
        ]
        
        if center_crop:
            transform_list.append(transforms.CenterCrop(img_size[0] if isinstance(img_size, tuple) else img_size))
        
        # 使用配置管理器获取均值和标准差
        mean = get_config_value(self.config, 'mean', self.default_mean)
        std = get_config_value(self.config, 'std', self.default_std)
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        return transforms.Compose(transform_list)
    
    def get_test_transforms(self, img_size: tuple = None) -> transforms.Compose:
        """
        获取测试数据转换
        
        Args:
            img_size: 目标图像尺寸
            
        Returns:
            测试数据的转换组合
        """
        # 从配置获取图像大小
        if img_size is None:
            img_size = get_config_value(self.config, 'IMAGE_SIZE', (224, 224))
        
        # 测试集通常使用和验证集相同的转换
        return self.get_valid_transforms(img_size)

    def get_tta_transforms(self, img_size: tuple = None) -> List[transforms.Compose]:
        """
        获取测试时增强(TTA)的转换列表
        
        Args:
            img_size: 目标图像尺寸
            
        Returns:
            TTA转换列表
        """
        # 从配置获取图像大小
        if img_size is None:
            img_size = get_config_value(self.config, 'IMAGE_SIZE', (224, 224))
        
        # 获取均值和标准差从配置
        mean = get_config_value(self.config, 'mean', self.default_mean)
        std = get_config_value(self.config, 'std', self.default_std)
        
        # 图像尺寸调整量
        resize_padding = get_config_value(self.config, 'tta_resize_padding', 32)
            
        # TTA通常包括原始图像、水平翻转、垂直翻转等多个版本
        basic_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size[0] + resize_padding),  # 稍大的尺寸
            transforms.CenterCrop(img_size[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # 水平翻转
        h_flip = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size[0] + resize_padding),
            transforms.CenterCrop(img_size[0]),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # 获取旋转角度
        rotation_angle = get_config_value(self.config, 'tta_rotation_angle', 90)
        
        # 旋转变换
        rotate = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size[0] + resize_padding),
            transforms.CenterCrop(img_size[0]),
            transforms.RandomRotation((rotation_angle, rotation_angle)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # 亮度调整
        brightness_value = get_config_value(self.config, 'tta_brightness', 0.2)
        brightness = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size[0] + resize_padding),
            transforms.CenterCrop(img_size[0]),
            transforms.ColorJitter(brightness=brightness_value),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        return [basic_transform, h_flip, rotate, brightness]
    
    @staticmethod
    def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        实现MixUp数据增强
        
        Args:
            x: 特征张量 [batch_size, channels, height, width]
            y: 标签张量 [batch_size]
            alpha: 贝塔分布参数
            
        Returns:
            混合后的特征和标签，以及lambda值
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        实现CutMix数据增强
        
        Args:
            x: 特征张量 [batch_size, channels, height, width]
            y: 标签张量 [batch_size]
            alpha: 贝塔分布参数
            
        Returns:
            切割混合后的特征和标签，以及lambda值
        """
        # 生成混合参数
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        
        # 获取批次大小
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        # 计算切割区域
        h, w = x.shape[2:]
        cut_rat = np.sqrt(1. - lam)  # 切割比例
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)
        
        # 随机选择切割区域中心点
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # 确保切割区域在图像内
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # 执行区域替换
        cutmix_images = x.clone()
        cutmix_images[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # 调整混合权重
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
        
        return cutmix_images, y, y[index], lam
    
    def get_transform_by_name(self, name: str, img_size: tuple = None) -> transforms.Compose:
        """
        根据名称获取特定的转换
        
        Args:
            name: 转换名称，可选值: 'train', 'valid', 'test', 'tta'
            img_size: 目标图像尺寸
            
        Returns:
            对应的转换组合
        """
        # 从配置获取图像大小
        if img_size is None:
            img_size = get_config_value(self.config, 'IMAGE_SIZE', (224, 224))
            
        name = name.lower()
        if name == 'train':
            return self.get_train_transforms(img_size)
        elif name in ('valid', 'val'):
            return self.get_valid_transforms(img_size)
        elif name == 'test':
            return self.get_test_transforms(img_size)
        elif name == 'tta':
            return self.get_tta_transforms(img_size)[0]  # 返回基本TTA转换
        else:
            raise ValueError(f"未知的转换名称: {name}")

    @staticmethod
    def random_gaussian_blur(img: np.ndarray, p: float = 0.5, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
        """
        随机应用高斯模糊
        
        Args:
            img: 输入图像
            p: 应用概率
            kernel_size: 高斯核大小
            sigma: 高斯核标准差，0表示自动计算
            
        Returns:
            处理后的图像
        """
        if np.random.random() < p:
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        return img
        
    @staticmethod
    def random_affine(img: np.ndarray, 
                   degrees: float = 10, 
                   translate: Tuple[float, float] = (0.1, 0.1),
                   scale: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        随机仿射变换
        
        Args:
            img: 输入图像
            degrees: 旋转角度范围
            translate: 平移比例范围 (x, y)
            scale: 缩放比例范围
            
        Returns:
            变换后的图像
        """
        height, width = img.shape[:2]
        
        # 随机旋转角度
        angle = np.random.uniform(-degrees, degrees)
        
        # 随机平移
        tx = np.random.uniform(-translate[0], translate[0]) * width
        ty = np.random.uniform(-translate[1], translate[1]) * height
        
        # 随机缩放
        scale_factor = np.random.uniform(scale[0], scale[1])
        
        # 创建变换矩阵
        center = (width / 2, height / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale_factor)
        rot_mat[0, 2] += tx
        rot_mat[1, 2] += ty
        
        # 应用变换
        result = cv2.warpAffine(
            img, rot_mat, (width, height), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return result


class AugmentationConfig:
    """
    数据增强配置类
    
    提供多种预设配置，可用于不同的增强需求
    """
    
    @staticmethod
    def get_default_config() -> Dict:
        """获取默认配置"""
        config_manager = ConfigManager()
        
        # 使用ConfigManager获取默认值
        return {
            'hflip_prob': config_manager.get('hflip_prob', 0.5, "data"),
            'vflip_prob': config_manager.get('vflip_prob', 0.0, "data"),
            'rotation_degrees': config_manager.get('rotation_degrees', 15, "data"),
            'brightness': config_manager.get('brightness', 0.2, "data"),
            'contrast': config_manager.get('contrast', 0.2, "data"),
            'saturation': config_manager.get('saturation', 0.2, "data"),
            'hue': config_manager.get('hue', 0.1, "data"),
            'crop_scale': config_manager.get('crop_scale', (0.8, 1.0), "data"),
            'crop_ratio': config_manager.get('crop_ratio', (0.75, 1.33), "data"),
            'use_advanced': config_manager.get('use_advanced', True, "data"),
            'erasing_prob': config_manager.get('erasing_prob', 0.2, "data"),
            'center_crop': config_manager.get('center_crop', True, "data"),
            'mean': config_manager.get('NORMALIZATION_MEAN', [0.485, 0.456, 0.406], "data"),
            'std': config_manager.get('NORMALIZATION_STD', [0.229, 0.224, 0.225], "data")
        }
    
    @staticmethod
    def get_light_config() -> Dict:
        """获取轻量级增强配置"""
        config_manager = ConfigManager()
        
        return {
            'hflip_prob': config_manager.get('light_hflip_prob', 0.5, "data"),
            'vflip_prob': config_manager.get('light_vflip_prob', 0.0, "data"),
            'rotation_degrees': config_manager.get('light_rotation_degrees', 10, "data"),
            'brightness': config_manager.get('light_brightness', 0.1, "data"),
            'contrast': config_manager.get('light_contrast', 0.1, "data"),
            'saturation': config_manager.get('light_saturation', 0.1, "data"),
            'hue': config_manager.get('light_hue', 0.05, "data"),
            'crop_scale': config_manager.get('light_crop_scale', (0.9, 1.0), "data"),
            'crop_ratio': config_manager.get('light_crop_ratio', (0.9, 1.1), "data"),
            'use_advanced': config_manager.get('light_use_advanced', False, "data"),
            'center_crop': config_manager.get('light_center_crop', True, "data"),
            'mean': config_manager.get('NORMALIZATION_MEAN', [0.485, 0.456, 0.406], "data"),
            'std': config_manager.get('NORMALIZATION_STD', [0.229, 0.224, 0.225], "data")
        }
    
    @staticmethod
    def get_strong_config() -> Dict:
        """获取强增强配置"""
        config_manager = ConfigManager()
        
        return {
            # 保留现有配置
            'hflip_prob': config_manager.get('strong_hflip_prob', 0.5, "data"),
            'vflip_prob': config_manager.get('strong_vflip_prob', 0.3, "data"),
            'rotation_degrees': config_manager.get('strong_rotation_degrees', 30, "data"),
            'brightness': config_manager.get('strong_brightness', 0.4, "data"),
            'contrast': config_manager.get('strong_contrast', 0.4, "data"),
            'saturation': config_manager.get('strong_saturation', 0.4, "data"),
            'hue': config_manager.get('strong_hue', 0.15, "data"),
            'crop_scale': config_manager.get('strong_crop_scale', (0.6, 1.0), "data"),
            'crop_ratio': config_manager.get('strong_crop_ratio', (0.7, 1.43), "data"),
            'use_advanced': config_manager.get('strong_use_advanced', True, "data"),
            'erasing_prob': config_manager.get('strong_erasing_prob', 0.3, "data"),
            'center_crop': config_manager.get('strong_center_crop', False, "data"),
            'autocontrast_prob': config_manager.get('strong_autocontrast_prob', 0.25, "data"),
            'equalize_prob': config_manager.get('strong_equalize_prob', 0.15, "data"),
            'gamma_prob': config_manager.get('strong_gamma_prob', 0.35, "data"),
            'gamma_range': config_manager.get('strong_gamma_range', (0.6, 1.4), "data"),
            
            # 新增强增强配置
            'affine_prob': config_manager.get('strong_affine_prob', 0.6, "data"),
            'affine_degrees': config_manager.get('strong_affine_degrees', 20, "data"),
            'affine_translate': config_manager.get('strong_affine_translate', (0.12, 0.12), "data"),
            'affine_scale': config_manager.get('strong_affine_scale', (0.8, 1.2), "data"),
            'affine_shear': config_manager.get('strong_affine_shear', 15, "data"),
            
            'blur_prob': config_manager.get('strong_blur_prob', 0.4, "data"),
            'blur_kernel_size': config_manager.get('strong_blur_kernel_size', 7, "data"),
            'blur_sigma': config_manager.get('strong_blur_sigma', (0.1, 3.0), "data"),
            
            'noise_prob': config_manager.get('strong_noise_prob', 0.3, "data"),
            'noise_std': config_manager.get('strong_noise_std', (0.02, 0.08), "data"),
            
            'erasing_scale': config_manager.get('strong_erasing_scale', (0.02, 0.2), "data"),
            'erasing_ratio': config_manager.get('strong_erasing_ratio', (0.2, 3.5), "data"),
            
            'mean': config_manager.get('NORMALIZATION_MEAN', [0.485, 0.456, 0.406], "data"),
            'std': config_manager.get('NORMALIZATION_STD', [0.229, 0.224, 0.225], "data")
        }
    
    @staticmethod
    def get_plant_specific_config() -> Dict:
        """获取专为植物图像优化的增强配置"""
        config_manager = ConfigManager()
        
        return {
            'hflip_prob': config_manager.get('plant_hflip_prob', 0.5, "data"),
            'vflip_prob': config_manager.get('plant_vflip_prob', 0.0, "data"),  # 植物通常有朝上的方向性
            'rotation_degrees': config_manager.get('plant_rotation_degrees', 20, "data"),  # 允许适度旋转
            'brightness': config_manager.get('plant_brightness', 0.25, "data"),  # 植物对光照敏感
            'contrast': config_manager.get('plant_contrast', 0.2, "data"),
            'saturation': config_manager.get('plant_saturation', 0.25, "data"),  # 增强叶片颜色差异
            'hue': config_manager.get('plant_hue', 0.05, "data"),  # 植物的色调变化不宜太大
            'crop_scale': config_manager.get('plant_crop_scale', (0.7, 1.0), "data"),
            'crop_ratio': config_manager.get('plant_crop_ratio', (0.8, 1.2), "data"),
            'use_advanced': config_manager.get('plant_use_advanced', True, "data"),
            'erasing_prob': config_manager.get('plant_erasing_prob', 0.15, "data"),  # 模拟叶片缺失或遮挡
            'center_crop': config_manager.get('plant_center_crop', True, "data"),
            'mean': config_manager.get('NORMALIZATION_MEAN', [0.485, 0.456, 0.406], "data"),
            'std': config_manager.get('NORMALIZATION_STD', [0.229, 0.224, 0.225], "data")
        }