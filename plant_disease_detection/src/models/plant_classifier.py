# src/models/plant_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Union, Tuple, Any
from src.models.base_model import BaseModel

# 添加统一配置管理导入
from src.utils.config_manager import ConfigManager
from src.utils.config_helpers import get_config_value

class PlantClassifier(BaseModel):
    """
    植物分类器模型
    
    支持多种backbone网络，可配置特征提取层和分类头，
    支持单标签和多标签分类，以及精细控制层冻结。
    """
    
    def __init__(self, config):
        """
        初始化植物分类器
        
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
                self.config_manager.get('num_classes', 10, 'model'))
            self.backbone_name = config.get('backbone', 
                self.config_manager.get('backbone', 'resnet50', 'model'))
            self.pretrained = config.get('pretrained', 
                self.config_manager.get('PRETRAINED', True, 'model'))
            self.dropout_rate = config.get('dropout_rate', 
                self.config_manager.get('dropout_rate', 0.5, 'model'))
            self.is_multilabel = config.get('is_multilabel', 
                self.config_manager.get('is_multilabel', False, 'model'))
            self.freeze_layers = config.get('freeze_layers', 
                self.config_manager.get('freeze_layers', None, 'model'))
            
            # 构建兼容原始实现所需的classifier_config
            self.classifier_config = {
                'num_classes': self.num_classes,
                'backbone': self.backbone_name,
                'dropout_rate': self.dropout_rate,
                'is_multilabel': self.is_multilabel,
                'freeze_layers': self.freeze_layers,
                'head_config': config.get('head_config', {})
            }
        elif hasattr(config, 'CLASSIFIER_CONFIG'):
            # 兼容原始实现
            self.classifier_config = config.CLASSIFIER_CONFIG
            self.num_classes = self.classifier_config.get('num_classes', 10)
            self.backbone_name = self.classifier_config.get('backbone', 'resnet50')
            self.pretrained = getattr(config, 'PRETRAINED', True)
            self.dropout_rate = self.classifier_config.get('dropout_rate', 0.3)
            self.is_multilabel = self.classifier_config.get('is_multilabel', False)
            self.freeze_layers = self.classifier_config.get('freeze_layers', None)
        else:
            # 从ConfigManager获取配置
            self.num_classes = self.config_manager.get('num_classes', 10, 'model')
            self.backbone_name = self.config_manager.get('backbone', 'resnet50', 'model')
            self.pretrained = self.config_manager.get('PRETRAINED', True, 'model')
            self.dropout_rate = self.config_manager.get('dropout_rate', 0.3, 'model')
            self.is_multilabel = self.config_manager.get('is_multilabel', False, 'model')
            self.freeze_layers = self.config_manager.get('freeze_layers', None, 'model')
            
            # 构建兼容原始实现所需的classifier_config
            self.classifier_config = {
                'num_classes': self.num_classes,
                'backbone': self.backbone_name,
                'dropout_rate': self.dropout_rate,
                'is_multilabel': self.is_multilabel,
                'freeze_layers': self.freeze_layers,
                'head_config': self.config_manager.get('head_config', {}, 'model')
            }
        
        # 构建网络
        self.build_backbone()
        self.build_classifier_head()
        
        # 冻结指定层
        self.apply_layer_freezing()
        
        # 记录模型架构信息
        self.model_info = {
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'is_multilabel': self.is_multilabel,
            'feature_dim': self.feature_dim
        }
        
    def build_backbone(self):
        """构建特征提取主干网络"""
        # 选择特定的模型系列
        if self.backbone_name.startswith('resnet'):
            self._build_resnet_backbone()
        elif self.backbone_name.startswith('efficientnet'):
            self._build_efficientnet_backbone()
        elif self.backbone_name.startswith('mobilenet'):
            self._build_mobilenet_backbone()
        elif self.backbone_name.startswith('convnext'):
            self._build_convnext_backbone()
        else:
            # 默认使用标准的torchvision模型
            try:
                self.backbone = getattr(models, self.backbone_name)(pretrained=self.pretrained)
                # 获取特征维度
                self.feature_dim = self._get_feature_dim()
                # 移除原始分类头
                self._remove_classifier_head()
            except AttributeError:
                raise ValueError(f"不支持的backbone: {self.backbone_name}")
    
    def _build_resnet_backbone(self):
        """构建ResNet系列主干网络"""
        # 检查是否使用较新的ResNet变体
        if self.backbone_name == 'resnet50_v2':
            # ResNet v2变体具有preactivation设计
            from torchvision.models.resnet import ResNet, Bottleneck
            # 自定义ResNet v2实现，或使用已有库
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])
            if self.pretrained:
                # 加载预训练权重（如果可用）
                pass
        else:
            # 使用标准ResNet模型
            model_fn = getattr(models, self.backbone_name)
            # 新版本torchvision使用weights参数而不是pretrained
            try:
                # 对于torchvision 0.13+
                weights = 'DEFAULT' if self.pretrained else None
                self.backbone = model_fn(weights=weights)
            except TypeError:
                # 对于旧版torchvision
                self.backbone = model_fn(pretrained=self.pretrained)
        
        # 获取特征维度
        self.feature_dim = self.backbone.fc.in_features
        # 移除原始分类头
        self.backbone.fc = nn.Identity()
    
    def _build_efficientnet_backbone(self):
        """构建EfficientNet系列主干网络"""
        try:
            # 尝试使用新版本的EfficientNet实现
            if self.backbone_name == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.DEFAULT if self.pretrained else None
                self.backbone = models.efficientnet_b0(weights=weights)
            elif self.backbone_name == 'efficientnet_b1':
                weights = models.EfficientNet_B1_Weights.DEFAULT if self.pretrained else None
                self.backbone = models.efficientnet_b1(weights=weights)
            elif self.backbone_name == 'efficientnet_b2':
                weights = models.EfficientNet_B2_Weights.DEFAULT if self.pretrained else None
                self.backbone = models.efficientnet_b2(weights=weights)
            else:
                # 其他EfficientNet变体
                model_fn = getattr(models, self.backbone_name)
                self.backbone = model_fn(pretrained=self.pretrained)
                
            # 获取特征维度
            self.feature_dim = self.backbone.classifier[1].in_features
            # 移除原始分类头
            self.backbone.classifier = nn.Identity()
        except (AttributeError, ImportError):
            # 回退到timm库或其他实现
            try:
                import timm
                self.backbone = timm.create_model(
                    self.backbone_name, 
                    pretrained=self.pretrained,
                    num_classes=0  # 去掉分类头
                )
                # 获取特征维度 - timm特定方法
                self.feature_dim = self.backbone.num_features
            except ImportError:
                raise ImportError("无法导入EfficientNet，请安装timm库或升级torchvision")
    
    def _build_mobilenet_backbone(self):
        """构建MobileNet系列主干网络"""
        try:
            # 使用MobileNetV3
            if self.backbone_name == 'mobilenet_v3_small':
                weights = models.MobileNet_V3_Small_Weights.DEFAULT if self.pretrained else None
                self.backbone = models.mobilenet_v3_small(weights=weights)
                self.feature_dim = self.backbone.classifier[3].in_features
            elif self.backbone_name == 'mobilenet_v3_large':
                weights = models.MobileNet_V3_Large_Weights.DEFAULT if self.pretrained else None
                self.backbone = models.mobilenet_v3_large(weights=weights)
                self.feature_dim = self.backbone.classifier[3].in_features
            else:
                # 其他MobileNet变体
                model_fn = getattr(models, self.backbone_name)
                self.backbone = model_fn(pretrained=self.pretrained)
                # 尝试获取特征维度
                self.feature_dim = self._get_feature_dim()
                
            # 移除原始分类头
            self.backbone.classifier = nn.Identity()
        except (AttributeError, ImportError):
            # 回退到timm库
            try:
                import timm
                self.backbone = timm.create_model(
                    self.backbone_name, 
                    pretrained=self.pretrained,
                    num_classes=0
                )
                self.feature_dim = self.backbone.num_features
            except ImportError:
                raise ImportError("无法导入MobileNet，请安装timm库或升级torchvision")
    
    def _build_convnext_backbone(self):
        """构建ConvNeXt系列主干网络"""
        try:
            # 使用ConvNeXt
            if self.backbone_name == 'convnext_tiny':
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT if self.pretrained else None
                self.backbone = models.convnext_tiny(weights=weights)
            elif self.backbone_name == 'convnext_small':
                weights = models.ConvNeXt_Small_Weights.DEFAULT if self.pretrained else None
                self.backbone = models.convnext_small(weights=weights)
            elif self.backbone_name == 'convnext_base':
                weights = models.ConvNeXt_Base_Weights.DEFAULT if self.pretrained else None
                self.backbone = models.convnext_base(weights=weights)
            else:
                # 其他ConvNeXt变体
                model_fn = getattr(models, self.backbone_name)
                self.backbone = model_fn(pretrained=self.pretrained)
                
            # 获取特征维度
            self.feature_dim = self.backbone.classifier[2].in_features
            # 移除原始分类头
            self.backbone.classifier = nn.Identity()
        except (AttributeError, ImportError):
            # 回退到timm库
            try:
                import timm
                self.backbone = timm.create_model(
                    self.backbone_name, 
                    pretrained=self.pretrained,
                    num_classes=0
                )
                self.feature_dim = self.backbone.num_features
            except ImportError:
                raise ImportError("无法导入ConvNeXt，请安装timm库或升级torchvision")
    
    def _get_feature_dim(self):
        """获取主干网络的特征维度"""
        # 尝试不同的属性名称来获取特征维度
        if hasattr(self.backbone, 'fc'):
            return self.backbone.fc.in_features
        elif hasattr(self.backbone, 'classifier'):
            classifier = self.backbone.classifier
            if isinstance(classifier, nn.Sequential):
                for module in reversed(classifier):
                    if isinstance(module, nn.Linear):
                        return module.in_features
            elif isinstance(classifier, nn.Linear):
                return classifier.in_features
        elif hasattr(self.backbone, 'head'):
            if isinstance(self.backbone.head, nn.Linear):
                return self.backbone.head.in_features
            
        # 如果无法确定，使用固定值并发出警告
        print(f"警告: 无法确定{self.backbone_name}的特征维度，使用默认值2048")
        return 2048
    
    def _remove_classifier_head(self):
        """移除原始分类头"""
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Sequential):
                # 寻找序列中的最后一个线性层
                for i in range(len(self.backbone.classifier) - 1, -1, -1):
                    if isinstance(self.backbone.classifier[i], nn.Linear):
                        # 保留之前的层，替换此线性层为Identity
                        modules = list(self.backbone.classifier.children())
                        modules[i] = nn.Identity()
                        self.backbone.classifier = nn.Sequential(*modules)
                        break
            else:
                self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
    
    def build_classifier_head(self):
        """构建分类头"""
        # 获取分类头配置，兼容两种方式
        head_config = {}
        if isinstance(self.classifier_config, dict) and 'head_config' in self.classifier_config:
            head_config = self.classifier_config['head_config']
        else:
            head_config_from_manager = self.config_manager.get('head_config', {}, 'model')
            if isinstance(head_config_from_manager, dict):
                head_config = head_config_from_manager
                
        hidden_dims = head_config.get('hidden_dims', [512])
        activation = head_config.get('activation', 'relu')
        dropout = self.dropout_rate
        
        # 创建激活函数
        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(inplace=True)
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU(inplace=True)
            
        # 构建分类头层
        layers = []
        prev_dim = self.feature_dim
        
        # 添加隐藏层
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                act_fn,
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        # 添加最终分类层
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
        # 如果是多标签分类，不使用softmax
        if not self.is_multilabel:
            # 单标签分类在forward中应用softmax/log_softmax
            pass
        
        # 创建分类头
        self.classifier_head = nn.Sequential(*layers)
        
        # 配置额外的特征层
        self.use_attention = head_config.get('use_attention', False)
        if self.use_attention:
            self.attention = self._build_attention_layer()
    
    def _build_attention_layer(self):
        """构建注意力层"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 16, self.feature_dim),
            nn.Sigmoid()
        )
    
    def apply_layer_freezing(self):
        """应用层冻结策略"""
        # 默认情况：冻结整个主干网络，只训练分类头
        if self.freeze_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = False
            return
            
        # 通过名称冻结特定层
        if isinstance(self.freeze_layers, list):
            for name, param in self.backbone.named_parameters():
                frozen = any(layer in name for layer in self.freeze_layers)
                param.requires_grad = not frozen
        
        # 冻结到指定深度或层级
        elif isinstance(self.freeze_layers, int):
            # 不同网络结构有不同的层次模式
            if self.backbone_name.startswith('resnet'):
                # ResNet有layer1, layer2, layer3, layer4结构
                modules_to_freeze = []
                if self.freeze_layers >= 1:
                    modules_to_freeze.extend([self.backbone.conv1, self.backbone.bn1])
                if self.freeze_layers >= 2:
                    modules_to_freeze.append(self.backbone.layer1)
                if self.freeze_layers >= 3:
                    modules_to_freeze.append(self.backbone.layer2)
                if self.freeze_layers >= 4:
                    modules_to_freeze.append(self.backbone.layer3)
                if self.freeze_layers >= 5:
                    modules_to_freeze.append(self.backbone.layer4)
                    
                for module in modules_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                # 其他网络可能需要特定处理
                print(f"警告：不支持对{self.backbone_name}进行整数级别的层冻结")
    
    def unfreeze_layers(self, layers=None):
        """
        解冻指定层或全部层

        Args:
            layers: 要解冻的层名列表，如果为None则解冻全部
        """
        if layers is None:
            # 解冻全部层
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
            
        # 解冻指定名称的层
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = True
    
    def forward(self, x):
        """
        模型前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, channels, height, width]
            
        Returns:
            如果是单标签分类，返回logits；如果是多标签分类，返回sigmoid激活后的值
        """
        # 特征提取
        features = self.backbone(x)
        
        # 应用注意力机制（如果启用）
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
        # 分类头
        logits = self.classifier_head(features)
        
        # 对于多标签分类，应用sigmoid激活
        if self.is_multilabel:
            return torch.sigmoid(logits)
        
        return logits
    
    def predict(self, x, apply_softmax=True):
        """
        预测函数，方便推理时使用
        
        Args:
            x: 输入张量
            apply_softmax: 是否应用softmax（单标签分类）
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            if self.is_multilabel:
                # 多标签情况下，sigmoid已在forward中应用
                return logits
            elif apply_softmax:
                # 单标签分类，应用softmax
                return F.softmax(logits, dim=1)
            else:
                return logits
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def summary(self):
        """打印模型摘要信息"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"模型架构: {self.backbone_name}")
        print(f"特征维度: {self.feature_dim}")
        print(f"类别数量: {self.num_classes}")
        print(f"是否多标签: {'是' if self.is_multilabel else '否'}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"总参数: {total_params:,}")
        print(f"冻结参数: {total_params - trainable_params:,}")