# src/utils/config_helpers.py

from typing import Dict, Any, Optional
from src.utils.config_manager import ConfigManager

def get_config(config=None, config_type="data"):
    """
    获取配置对象，优先使用传入配置，否则使用ConfigManager
    
    如果传入的是字典，则返回ConfigManager以兼容字典访问
    如果传入的是配置对象，则直接返回
    如果未传入配置，则返回ConfigManager
    
    Args:
        config: 输入配置（对象或字典）
        config_type: 配置类型，用于ConfigManager
        
    Returns:
        配置对象或ConfigManager实例
    """
    if config is None:
        return ConfigManager()
    
    if isinstance(config, dict):
        return ConfigManager()
    
    # 否则直接返回配置对象
    return config

def get_config_value(config, key, default=None, config_type="data"):
    """
    从配置中获取值，支持多种配置类型
    
    Args:
        config: 配置（对象、字典或None）
        key: 配置键名
        default: 默认值
        config_type: 配置类型，用于ConfigManager
        
    Returns:
        配置值
    """
    config_manager = ConfigManager()
    
    # 处理None情况
    if config is None:
        return config_manager.get(key, default, config_type)
    
    # 处理字典情况
    if isinstance(config, dict):
        return config.get(key, config_manager.get(key, default, config_type))
    
    # 处理对象情况
    if hasattr(config, key):
        return getattr(config, key)
    
    # 默认使用ConfigManager
    return config_manager.get(key, default, config_type)