# src/utils/config_manager.py

import logging
from typing import Any, Dict, Optional
from pathlib import Path
import json
import yaml

# 导入所有配置类
from configs.data_config import DataConfig
try:
    # 修正导入路径 - ModelConfig 实际在 src.models.model_config 中
    from src.models.model_config import ModelConfig
except ImportError:
    ModelConfig = None
try:
    # 修正导入路径 - TrainingConfig 实际在 src.training.train_config 中
    from src.training.train_config import TrainingConfig as TrainConfig
except ImportError:
    TrainConfig = None

class ConfigManager:
    """统一配置管理类，确保所有组件一致访问配置"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        
        # 加载各种配置
        self.data_config = DataConfig()
        
        if ModelConfig:
            self.model_config = ModelConfig()
        else:
            self.model_config = None
            
        if TrainConfig:
            self.train_config = TrainConfig()
        else:
            self.train_config = None
            
        self._override_values = {}
        self._initialized = True
    
    def get(self, key: str, default: Any = None, config_type: str = "data") -> Any:
        """获取配置值"""
        # 检查是否有被覆盖的值
        override_key = f"{config_type}.{key}"
        if override_key in self._override_values:
            return self._override_values[override_key]
        
        # 根据配置类型选择相应的配置对象
        if config_type == "data":
            config_obj = self.data_config
        elif config_type == "model":
            config_obj = self.model_config
        elif config_type == "train":
            config_obj = self.train_config
        else:
            self.logger.warning(f"未知的配置类型: {config_type}，返回默认值")
            return default
        
        # 从配置对象获取值
        if config_obj and hasattr(config_obj, key):
            return getattr(config_obj, key)
        
        return default
    
    def override(self, key: str, value: Any, config_type: str = "data") -> None:
        """覆盖配置值"""
        override_key = f"{config_type}.{key}"
        self._override_values[override_key] = value
    
    def load_from_file(self, file_path: str) -> bool:
        """从文件加载配置"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_path}")
            
            # 更新配置
            self._update_config(config)
            return True
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            return False
    
    def get_dict_compatible(self, config_dict: Dict[str, Any], key: str, default: Any = None, config_type: str = "data") -> Any:
        """兼容字典风格配置访问"""
        if config_dict is not None and key in config_dict:
            return config_dict[key]
        return self.get(key, default, config_type)
    
    def _update_config(self, config):
        """更新配置"""
        for key, value in config.items():
            if isinstance(value, dict):
                # 处理嵌套字典
                for sub_key, sub_value in value.items():
                    config_type = key if key in ["data", "model", "train"] else "data"
                    full_key = sub_key if key in ["data", "model", "train"] else f"{key}_{sub_key}"
                    self.override(full_key, sub_value, config_type)
            else:
                # 处理普通键值
                config_type = "data"  # 默认类型
                if key.startswith("model_") or key == "task":
                    config_type = "model"
                elif key.startswith("train_"):
                    config_type = "train"
                
                self.override(key, value, config_type)