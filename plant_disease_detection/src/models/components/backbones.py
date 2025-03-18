# src/models/components/backbones.py

import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple, Optional, Any
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class BackboneBase(nn.Module):
    """
    骨干网络基类
    
    定义骨干网络的基本接口，所有具体骨干网络实现都应继承此类
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.out_channels = None
        self.feature_dims = {}  # 各层特征维度
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回特征"""
        return self.model(x)
    
    def get_feature_dims(self) -> Dict[str, int]:
        """获取各层特征维度"""
        return self.feature_dims
    
    def get_out_channels(self) -> int:
        """获取输出通道数"""
        return self.out_channels
    
    def freeze(self) -> None:
        """冻结所有参数"""
        for param in self.model.parameters():
            param.requires_grad = False
            
    def unfreeze(self) -> None:
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
            
    def freeze_layers(self, layer_names: List[str]) -> None:
        """
        冻结指定层
        
        Args:
            layer_names: 要冻结的层名列表
        """
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = False
                
    def unfreeze_layers(self, layer_names: List[str]) -> None:
        """
        解冻指定层
        
        Args:
            layer_names: 要解冻的层名列表
        """
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = True
                
    def freeze_by_depth(self, depth: int) -> None:
        """
        按深度冻结层
        
        Args:
            depth: 要冻结的深度（具体定义因架构而异）
        """
        # 在子类中实现具体逻辑
        raise NotImplementedError("在子类中实现")


class ResNetBackbone(BackboneBase):
    """ResNet系列骨干网络"""
    
    def __init__(
        self, 
        variant: str = 'resnet50', 
        pretrained: bool = True,
        remove_classifier: bool = True
    ):
        """
        初始化ResNet骨干网络
        
        Args:
            variant: ResNet变体，可选'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            pretrained: 是否加载预训练权重
            remove_classifier: 是否移除分类头
        """
        super().__init__()
        
        self.variant = variant
        self.pretrained = pretrained
        
        # 加载ResNet模型
        try:
            # 使用最新的权重API (PyTorch >= 0.13)
            if variant == 'resnet18':
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.model = models.resnet18(weights=weights)
            elif variant == 'resnet34':
                weights = models.ResNet34_Weights.DEFAULT if pretrained else None
                self.model = models.resnet34(weights=weights)
            elif variant == 'resnet50':
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                self.model = models.resnet50(weights=weights)
            elif variant == 'resnet101':
                weights = models.ResNet101_Weights.DEFAULT if pretrained else None
                self.model = models.resnet101(weights=weights)
            elif variant == 'resnet152':
                weights = models.ResNet152_Weights.DEFAULT if pretrained else None
                self.model = models.resnet152(weights=weights)
            else:
                raise ValueError(f"不支持的ResNet变体: {variant}")
        except (AttributeError, TypeError):
            # 兼容旧版PyTorch
            logger.warning("使用旧版API加载ResNet模型")
            model_fn = getattr(models, variant)
            self.model = model_fn(pretrained=pretrained)
        
        # 记录输出通道数和特征维度
        self.out_channels = self.model.fc.in_features
        
        # 映射各层特征维度
        resnet_dims = {
            'resnet18': {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512},
            'resnet34': {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512},
            'resnet50': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048},
            'resnet101': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048},
            'resnet152': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        }
        self.feature_dims = resnet_dims.get(variant, {})
        
        # 移除分类头
        if remove_classifier:
            self._remove_classifier()
            
        # 保存每层引用，方便冻结/解冻指定层
        self.layer_refs = {
            'conv1': self.model.conv1,
            'bn1': self.model.bn1,
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4
        }

    def _remove_classifier(self) -> None:
        """移除分类头部分"""
        self.model.fc = nn.Identity()
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            return_features: 是否返回中间层特征
            
        Returns:
            如果return_features为False，返回最终特征；否则返回各层特征字典
        """
        if not return_features:
            return self.model(x)
            
        # 返回中间层特征
        features = {}
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        features['layer1'] = x
        
        x = self.model.layer2(x)
        features['layer2'] = x
        
        x = self.model.layer3(x)
        features['layer3'] = x
        
        x = self.model.layer4(x)
        features['layer4'] = x
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        features['final'] = x
        
        return features
    
    def freeze_by_depth(self, depth: int) -> None:
        """
        按深度冻结ResNet的层
        
        Args:
            depth: 冻结深度 (1:conv1+bn1, 2:+layer1, 3:+layer2, 4:+layer3, 5:+layer4)
        """
        layers_to_freeze = []
        if depth >= 1:
            layers_to_freeze.extend(['conv1', 'bn1'])
        if depth >= 2:
            layers_to_freeze.append('layer1')
        if depth >= 3:
            layers_to_freeze.append('layer2')
        if depth >= 4:
            layers_to_freeze.append('layer3')
        if depth >= 5:
            layers_to_freeze.append('layer4')
            
        # 冻结指定层
        for layer in layers_to_freeze:
            for param in self.layer_refs[layer].parameters():
                param.requires_grad = False


class EfficientNetBackbone(BackboneBase):
    """EfficientNet系列骨干网络"""
    
    def __init__(
        self, 
        variant: str = 'efficientnet_b0', 
        pretrained: bool = True,
        remove_classifier: bool = True
    ):
        """
        初始化EfficientNet骨干网络
        
        Args:
            variant: EfficientNet变体，可选'efficientnet_b0'到'efficientnet_b7'
            pretrained: 是否加载预训练权重
            remove_classifier: 是否移除分类头
        """
        super().__init__()
        
        self.variant = variant
        self.pretrained = pretrained
        
        try:
            # 使用最新的权重API
            if variant == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                self.model = models.efficientnet_b0(weights=weights)
            elif variant == 'efficientnet_b1':
                weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
                self.model = models.efficientnet_b1(weights=weights)
            elif variant == 'efficientnet_b2':
                weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
                self.model = models.efficientnet_b2(weights=weights)
            elif variant == 'efficientnet_b3':
                weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
                self.model = models.efficientnet_b3(weights=weights)
            elif variant == 'efficientnet_b4':
                weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
                self.model = models.efficientnet_b4(weights=weights)
            elif variant == 'efficientnet_b5':
                weights = models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
                self.model = models.efficientnet_b5(weights=weights)
            elif variant == 'efficientnet_b6':
                weights = models.EfficientNet_B6_Weights.DEFAULT if pretrained else None
                self.model = models.efficientnet_b6(weights=weights)
            elif variant == 'efficientnet_b7':
                weights = models.EfficientNet_B7_Weights.DEFAULT if pretrained else None
                self.model = models.efficientnet_b7(weights=weights)
            else:
                raise ValueError(f"不支持的EfficientNet变体: {variant}")
        except (AttributeError, ImportError, TypeError):
            # 尝试使用旧版API或第三方实现
            try:
                # 尝试直接使用旧版torchvision
                model_fn = getattr(models, variant)
                self.model = model_fn(pretrained=pretrained)
            except (AttributeError, ImportError):
                try:
                    # 尝试使用timm库
                    import timm
                    self.model = timm.create_model(variant, pretrained=pretrained)
                except (ImportError, ValueError):
                    raise ImportError(f"无法导入{variant}，请安装timm库或升级torchvision")
        
        # 获取特征维度
        try:
            # 标准torchvision结构
            self.out_channels = self.model.classifier[1].in_features
        except (AttributeError, IndexError):
            try:
                # timm结构
                self.out_channels = self.model.classifier.in_features
            except AttributeError:
                # 其他实现
                logger.warning(f"无法自动获取{variant}的特征维度，使用默认值1280")
                self.out_channels = 1280
        
        # 定义各阶段输出维度（近似值，实际因模型而异）
        self.feature_dims = {
            'stage1': 24,
            'stage2': 40,
            'stage3': 80,
            'stage4': 112,
            'stage5': 192,
            'stage6': 320,
            'stage7': 1280
        }
        
        # 移除分类头
        if remove_classifier:
            self._remove_classifier()
    
    def _remove_classifier(self):
        """移除分类头"""
        try:
            # 标准实现
            self.model.classifier = nn.Identity()
        except AttributeError:
            # timm实现
            try:
                self.model.classifier = nn.Identity()
            except AttributeError:
                logger.warning(f"无法移除{self.variant}的分类头")
                
    def freeze_by_depth(self, depth: int) -> None:
        """
        按深度冻结层（具体层的名称取决于模型变体和实现）
        
        Args:
            depth: 冻结深度 (1-7，对应模型的不同阶段)
        """
        if not 1 <= depth <= 7:
            logger.warning(f"无效的冻结深度: {depth}，应当在1-7之间")
            return
            
        # EfficientNet通常按stages划分
        # 冻结所有层，然后根据depth解冻
        self.freeze()
        
        # 根据depth解冻特定层（名称取决于具体实现）
        try:
            # 标准实现
            for i in range(depth, 8):
                for name, param in self.model.features.named_parameters():
                    # 解冻深度之后的层
                    if f"{i}." in name:
                        param.requires_grad = True
        except Exception:
            logger.warning(f"无法按深度冻结{self.variant}，请手动指定层名")


class MobileNetBackbone(BackboneBase):
    """MobileNet系列骨干网络"""
    
    def __init__(
        self, 
        variant: str = 'mobilenet_v3_small', 
        pretrained: bool = True,
        remove_classifier: bool = True
    ):
        """
        初始化MobileNet骨干网络
        
        Args:
            variant: MobileNet变体，可选'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'
            pretrained: 是否加载预训练权重
            remove_classifier: 是否移除分类头
        """
        super().__init__()
        
        self.variant = variant
        self.pretrained = pretrained
        
        try:
            # 使用最新的权重API
            if variant == 'mobilenet_v2':
                weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
                self.model = models.mobilenet_v2(weights=weights)
                self.out_channels = 1280
            elif variant == 'mobilenet_v3_small':
                weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                self.model = models.mobilenet_v3_small(weights=weights)
                self.out_channels = self.model.classifier[3].in_features
            elif variant == 'mobilenet_v3_large':
                weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
                self.model = models.mobilenet_v3_large(weights=weights)
                self.out_channels = self.model.classifier[3].in_features
            else:
                raise ValueError(f"不支持的MobileNet变体: {variant}")
                
        except (AttributeError, TypeError):
            # 兼容旧版
            try:
                model_fn = getattr(models, variant)
                self.model = model_fn(pretrained=pretrained)
                
                # 获取输出通道数
                if variant == 'mobilenet_v2':
                    self.out_channels = 1280
                else:  # v3
                    try:
                        self.out_channels = self.model.classifier[3].in_features
                    except (IndexError, AttributeError):
                        self.out_channels = 1280
                        
            except (AttributeError, ImportError):
                # 尝试使用timm
                try:
                    import timm
                    self.model = timm.create_model(variant, pretrained=pretrained)
                    self.out_channels = self.model.num_features
                except (ImportError, ValueError):
                    raise ImportError(f"无法导入{variant}，请安装timm库或升级torchvision")
        
        # 定义特征维度
        if 'mobilenet_v3_small' in variant:
            self.feature_dims = {
                'block0': 16,
                'block1': 16,
                'block2': 24,
                'block3': 24,
                'block4': 40,
                'block5': 40,
                'block6': 40,
                'block7': 48,
                'block8': 48,
                'block9': 96,
                'block10': 96,
                'block11': 96
            }
        elif 'mobilenet_v3_large' in variant:
            self.feature_dims = {
                'block0': 16,
                'block1': 16,
                'block2': 24,
                'block3': 24,
                'block4': 40,
                'block5': 40,
                'block6': 40,
                'block7': 80,
                'block8': 80,
                'block9': 80,
                'block10': 80,
                'block11': 80,
                'block12': 112,
                'block13': 112,
                'block14': 160,
                'block15': 160,
                'block16': 160
            }
        else:  # v2
            self.feature_dims = {
                'block0': 16,
                'block1': 24,
                'block2': 24,
                'block3': 32,
                'block4': 32,
                'block5': 32,
                'block6': 64,
                'block7': 64,
                'block8': 64,
                'block9': 64,
                'block10': 96,
                'block11': 96,
                'block12': 96,
                'block13': 160,
                'block14': 160,
                'block15': 160,
                'block16': 320
            }
        
        # 移除分类头
        if remove_classifier:
            self._remove_classifier()
    
    def _remove_classifier(self):
        """移除分类头"""
        try:
            self.model.classifier = nn.Identity()
        except AttributeError:
            logger.warning(f"无法移除{self.variant}的分类头")
            
    def freeze_by_depth(self, depth: int) -> None:
        """
        按深度冻结层
        
        Args:
            depth: 冻结深度，1-6对应不同阶段
        """
        if not 1 <= depth <= 6:
            logger.warning(f"无效的冻结深度: {depth}，应当在1-6之间")
            return
        
        # 冻结前depth个blocks
        try:
            if hasattr(self.model, 'features'):
                # 标准实现有features属性
                num_blocks = len(self.model.features)
                freeze_blocks = int(depth * num_blocks / 6)  # 按比例计算
                
                # 冻结指定blocks
                for i in range(freeze_blocks):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = False
            else:
                # 其他实现可能需要具体处理
                logger.warning(f"无法按深度冻结{self.variant}，请手动指定层")
        except Exception as e:
            logger.error(f"冻结层时出错: {e}")


class ConvNextBackbone(BackboneBase):
    """ConvNeXt系列骨干网络"""
    
    def __init__(
        self, 
        variant: str = 'convnext_tiny', 
        pretrained: bool = True,
        remove_classifier: bool = True
    ):
        """
        初始化ConvNeXt骨干网络
        
        Args:
            variant: ConvNeXt变体，可选'convnext_tiny', 'convnext_small', 'convnext_base'等
            pretrained: 是否加载预训练权重
            remove_classifier: 是否移除分类头
        """
        super().__init__()
        
        self.variant = variant
        self.pretrained = pretrained
        
        try:
            # 使用最新的权重API
            if variant == 'convnext_tiny':
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
                self.model = models.convnext_tiny(weights=weights)
            elif variant == 'convnext_small':
                weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
                self.model = models.convnext_small(weights=weights)
            elif variant == 'convnext_base':
                weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
                self.model = models.convnext_base(weights=weights)
            elif variant == 'convnext_large':
                weights = models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None
                self.model = models.convnext_large(weights=weights)
            else:
                raise ValueError(f"不支持的ConvNeXt变体: {variant}")
                
            # 获取输出通道数
            self.out_channels = self.model.classifier[2].in_features
            
        except (AttributeError, ImportError, TypeError):
            # 尝试使用timm
            try:
                import timm
                self.model = timm.create_model(variant, pretrained=pretrained)
                # 获取输出通道
                self.out_channels = self.model.num_features
            except (ImportError, ValueError):
                raise ImportError(f"无法导入{variant}，请安装timm库或升级torchvision")
        
        # 定义特征维度
        # ConvNeXt特征维度
        if 'tiny' in variant:
            self.feature_dims = {
                'stage1': 96,
                'stage2': 192,
                'stage3': 384,
                'stage4': 768
            }
        elif 'small' in variant:
            self.feature_dims = {
                'stage1': 96,
                'stage2': 192,
                'stage3': 384,
                'stage4': 768
            }
        elif 'base' in variant:
            self.feature_dims = {
                'stage1': 128,
                'stage2': 256,
                'stage3': 512,
                'stage4': 1024
            }
        else:  # large
            self.feature_dims = {
                'stage1': 192,
                'stage2': 384,
                'stage3': 768,
                'stage4': 1536
            }
        
        # 移除分类头
        if remove_classifier:
            self._remove_classifier()
    
    def _remove_classifier(self):
        """移除分类头"""
        try:
            self.model.classifier = nn.Identity()
        except AttributeError:
            try:
                self.model.head = nn.Identity()
            except AttributeError:
                logger.warning(f"无法移除{self.variant}的分类头")
                
    def freeze_by_depth(self, depth: int) -> None:
        """
        按深度冻结层
        
        Args:
            depth: 冻结深度，1-4对应不同阶段
        """
        if not 1 <= depth <= 4:
            logger.warning(f"无效的冻结深度: {depth}，应当在1-4之间")
            return
        
        # ConvNeXt通常有4个stage
        try:
            if hasattr(self.model, 'features'):
                for i in range(depth):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = False
            else:
                logger.warning(f"无法按深度冻结{self.variant}，请手动指定层")
        except Exception as e:
            logger.error(f"冻结层时出错: {e}")


def create_backbone(
    model_name: str, 
    pretrained: bool = True, 
    remove_classifier: bool = True
) -> BackboneBase:
    """
    创建骨干网络工厂函数
    
    Args:
        model_name: 模型名称
        pretrained: 是否使用预训练权重
        remove_classifier: 是否移除分类头
    
    Returns:
        骨干网络实例
    
    Examples:
        >>> backbone = create_backbone('resnet50')
        >>> features = backbone(x)
    """
    # ResNet系列
    if model_name.startswith('resnet'):
        return ResNetBackbone(model_name, pretrained, remove_classifier)
    
    # EfficientNet系列
    elif model_name.startswith('efficientnet'):
        return EfficientNetBackbone(model_name, pretrained, remove_classifier)
    
    # MobileNet系列
    elif model_name.startswith('mobilenet'):
        return MobileNetBackbone(model_name, pretrained, remove_classifier)
    
    # ConvNeXt系列
    elif model_name.startswith('convnext'):
        return ConvNextBackbone(model_name, pretrained, remove_classifier)
    
    # 不支持的模型
    else:
        raise ValueError(f"不支持的骨干网络类型: {model_name}")


def get_available_backbones() -> Dict[str, List[str]]:
    """
    获取所有可用的骨干网络
    
    Returns:
        按类型分组的可用骨干网络字典
    """
    return {
        'ResNet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        'EfficientNet': [f'efficientnet_b{i}' for i in range(8)],
        'MobileNet': ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'],
        'ConvNeXt': ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']
    }


def get_feature_extractor(backbone: BackboneBase, output_layers: List[str] = None) -> nn.Module:
    """
    创建特征提取器，用于获取指定层的特征
    
    Args:
        backbone: 骨干网络实例
        output_layers: 要提取的层名列表
    
    Returns:
        可以提取指定层特征的模块
    """
    if output_layers is None:
        # 默认使用最后一层特征
        return backbone
    
    # 创建一个可以返回多层特征的包装器
    class FeatureExtractor(nn.Module):
        def __init__(self, backbone, output_layers):
            super().__init__()
            self.backbone = backbone
            self.output_layers = output_layers
            
        def forward(self, x):
            # 注意：这里假设backbone的forward方法支持return_features参数
            features = self.backbone(x, return_features=True)
            
            if len(self.output_layers) == 1:
                # 只返回单层特征
                return features[self.output_layers[0]]
                
            # 返回指定多层特征
            return {layer: features[layer] for layer in self.output_layers if layer in features}
    
    return FeatureExtractor(backbone, output_layers)