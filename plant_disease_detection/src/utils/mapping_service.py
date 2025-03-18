#src/utils/mapping_service.py
"""
统一的类别映射服务

为系统提供中央化的类别映射管理，支持:
1. 植物类别映射 (数字索引 -> 植物名称)
2. 病害类别映射 (数字索引 -> 病害名称)
3. 数据集类名到数据库名称的映射 (例如 "Apple___healthy" -> "苹果", "健康")
"""

import os
import json
import logging
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import re

# 获取logger
logger = logging.getLogger(__name__)

class MappingService:
    """统一的映射服务类，单例模式"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MappingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        
        # 添加映射文件目录
        self.mappings_dir = 'models'
        
        # 植物类别映射 (数字ID -> 植物名称)
        self.plant_classes = {}
        
        # 病害类别映射 (数字ID -> 病害名称)
        self.disease_classes = {}
        
        # 数据集类名到植物和状态的映射 (例如 "Apple___healthy" -> ("苹果", "健康"))
        self.dataset_mapping = {}
        
        # 初始化映射
        self._load_mappings()
        self._initialized = True
    
    def _load_mappings(self):
        """加载所有映射文件"""
        # 加载植物英文名到中文名的映射
        self.plant_mapping = self._load_json_file('models/plant_mappings.json', {})
        
        # 加载病害英文名到中文名的映射
        self.disease_mapping = self._load_json_file('models/disease_mappings.json', {})
        
        # 加载植物类别映射
        self.plant_classes = self._load_json_file('models/plant_classes.json', 
                                                self._get_default_plant_classes())
        
        # 加载病害类别映射
        self.disease_classes = self._load_json_file('models/disease_classes.json',
                                                  self._get_default_disease_classes())
        
        # 创建反向映射（中文 -> 英文）
        self.reverse_plant_mapping = {v: k for k, v in self.plant_mapping.items()}
        self.reverse_disease_mapping = {v: k for k, v in self.disease_mapping.items()}
        
        # 创建数据集类名映射
        self._create_dataset_mapping()
        
        self.logger.info(f"已加载 {len(self.plant_classes)} 个植物类别和 {len(self.disease_classes)} 个病害类别")

    def _load_json_file(self, file_path, default_value=None):
        """从JSON文件加载数据，如果失败则返回默认值"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"文件不存在: {file_path}，将使用默认值")
                return default_value
        except Exception as e:
            self.logger.error(f"加载文件失败 {file_path}: {e}")
            return default_value
    
    def _load_plant_classes(self):
        """加载植物类别映射"""
        try:
            plant_classes_path = os.path.join('models', 'plant_classes.json')
            if os.path.exists(plant_classes_path):
                with open(plant_classes_path, 'r', encoding='utf-8') as f:
                    self.plant_classes = json.load(f)
            else:
                self.logger.warning("植物类别映射文件不存在，将使用默认值")
                self.plant_classes = self._get_default_plant_classes()
        except Exception as e:
            self.logger.error(f"加载植物类别映射失败: {e}")
            self.plant_classes = self._get_default_plant_classes()
    
    def _load_disease_classes(self):
        """加载病害类别映射"""
        try:
            disease_classes_path = os.path.join('models', 'disease_classes.json')
            if os.path.exists(disease_classes_path):
                with open(disease_classes_path, 'r', encoding='utf-8') as f:
                    self.disease_classes = json.load(f)
            else:
                self.logger.warning("病害类别映射文件不存在，将使用默认值")
                self.disease_classes = self._get_default_disease_classes()
        except Exception as e:
            self.logger.error(f"加载病害类别映射失败: {e}")
            self.disease_classes = self._get_default_disease_classes()
    
    def _create_dataset_mapping(self):
        """创建数据集类名到植物和状态的映射"""
        # 使用已经加载的映射，而不是硬编码
        plant_mapping = self.plant_mapping
        disease_mapping = self.disease_mapping
        
        # 为所有植物类别创建映射
        for class_id, class_name in self.plant_classes.items():
            # 尝试解析类名，格式通常为 "植物-病害" 或 "植物-健康"
            if '-' in class_name:
                parts = class_name.split('-', 1)
                plant = parts[0].strip()
                condition = parts[1].strip()
                self.dataset_mapping[class_name] = (plant, condition)
            
            # 为原始数据集格式创建映射（例如"Apple___healthy"）
            for plant_en, plant_zh in plant_mapping.items():
                for disease_en, disease_zh in disease_mapping.items():
                    dataset_class = f"{plant_en}___{disease_en}"
                    self.dataset_mapping[dataset_class] = (plant_zh, disease_zh)
                    
                    # 处理Pepper,_bell的特殊情况
                    if plant_en == 'Pepper':
                        bell_class = f"Pepper,_bell___{disease_en}"
                        self.dataset_mapping[bell_class] = (plant_zh, disease_zh)
    
    def _get_default_plant_classes(self) -> Dict[str, str]:
        """获取默认的植物类别映射"""
        return {
            "0": "苹果-黑星病", "1": "苹果-黑腐病", "2": "苹果-锈病", "3": "苹果-健康", 
            "4": "玉米-叶斑病", "5": "玉米-锈病", "6": "玉米-健康", "7": "玉米-北方叶枯病",
            "8": "葡萄-黑腐病", "9": "葡萄-褐斑病", "10": "葡萄-健康", "11": "葡萄-叶枯病",
            "12": "橙子-黄龙病", "13": "桃子-细菌性斑点", "14": "桃子-健康",
            "15": "土豆-早疫病", "16": "土豆-晚疫病", "17": "土豆-健康",
            "18": "草莓-叶焦病", "19": "草莓-健康", 
            "20": "番茄-细菌斑点", "21": "番茄-早疫病", "22": "番茄-晚疫病", 
            "23": "番茄-叶霉病", "24": "番茄-斑点病", "25": "番茄-花叶病毒", "26": "番茄-健康"
        }
    
    def _get_default_disease_classes(self) -> Dict[str, str]:
        """获取默认的病害类别映射"""
        return {
            "0": "健康",
            "1": "黑星病",
            "2": "黑腐病",
            "3": "锈病",
            "4": "叶斑病"
        }
    
    def get_plant_name(self, class_id: str) -> str:
        """
        获取植物类别名称
        
        Args:
            class_id: 类别ID
            
        Returns:
            植物类别名称
        """
        return self.plant_classes.get(str(class_id), f"未知植物_{class_id}")
    
    def get_disease_name(self, class_id: str) -> str:
        """
        获取病害类别名称
        
        Args:
            class_id: 类别ID
            
        Returns:
            病害类别名称
        """
        return self.disease_classes.get(str(class_id), f"未知病害_{class_id}")
    
    def map_dataset_class_to_db(self, class_name: str) -> Tuple[str, str]:
        """
        将数据集类名映射到数据库中的植物和病害
        
        Args:
            class_name: 数据集类名，可以是"植物-病害"格式或原始的"Apple___healthy"格式
            
        Returns:
            (植物名，病害/状态名)元组
        """
        # 如果是我们定义的 "植物-病害" 格式
        if '-' in class_name:
            parts = class_name.split('-', 1)
            plant_type = parts[0].strip()
            condition = parts[1].strip() if len(parts) > 1 else "健康"
            return plant_type, condition
        
        # 如果是原始数据集格式 "植物___状态"
        elif "___" in class_name:
            parts = class_name.split('___')
            plant_en = parts[0].strip()
            condition_en = parts[1].strip() if len(parts) > 1 else "healthy"
            
            # 映射植物名称（英文->中文）
            plant = self.plant_mapping.get(plant_en, plant_en)
            
            # 映射病害名称（英文->中文）
            disease = self.disease_mapping.get(condition_en, condition_en)
            
            return plant, disease
        
        # 如果是英文植物名
        elif class_name in self.plant_mapping:
            return self.plant_mapping[class_name], "健康"
        
        # 如果是中文植物名
        elif class_name in self.reverse_plant_mapping:
            return class_name, "健康"
        
        # 无法识别的格式，返回原始值
        return class_name, "未知"
    
    def extract_plant_type_from_class(self, class_name: str) -> str:
        """
        从类别名称中提取植物类型
        
        Args:
            class_name: 类别名称，如"苹果-黑星病"或"Apple___Black_rot"
            
        Returns:
            植物类型名称
        """
        # 处理中文格式 "植物-病害"
        if '-' in class_name:
            return class_name.split('-')[0].strip()
            
        # 处理英文数据集格式 "植物___病害"
        if '___' in class_name:
            plant_en = class_name.split('___')[0].strip()
            # 映射到中文
            plant_mapping = {
                'Apple': '苹果',
                'Blueberry': '蓝莓',
                'Cherry': '樱桃',
                'Corn': '玉米',
                'Grape': '葡萄',
                'Orange': '橙子',
                'Peach': '桃子',
                'Pepper': '甜椒',
                'Potato': '土豆',
                'Raspberry': '树莓',
                'Soybean': '大豆',
                'Squash': '西葫芦',
                'Strawberry': '草莓',
                'Tomato': '番茄'
            }
            return plant_mapping.get(plant_en, plant_en)
        
        # 处理英文格式 "Plant_Disease"
        if '_' in class_name:
            # 尝试提取第一个下划线前的部分
            plant_en = class_name.split('_')[0].strip()
            # 映射同上...
            plant_mapping = {
                'Apple': '苹果',
                'Blueberry': '蓝莓',
                'Cherry': '樱桃',
                'Corn': '玉米',
                'Grape': '葡萄',
                'Orange': '橙子',
                'Peach': '桃子',
                'Pepper': '甜椒',
                'Potato': '土豆',
                'Raspberry': '树莓',
                'Soybean': '大豆',
                'Squash': '西葫芦',
                'Strawberry': '草莓',
                'Tomato': '番茄'
            }
            return plant_mapping.get(plant_en, plant_en)
            
        return class_name
    
    def get_all_plant_types(self) -> List[str]:
        """
        获取所有植物类型
        
        Returns:
            植物类型列表
        """
        plant_types = set()
        for class_name in self.plant_classes.values():
            plant_type = self.extract_plant_type_from_class(class_name)
            plant_types.add(plant_type)
        return sorted(list(plant_types))
    
    def save_mappings(self):
        """保存所有映射到文件"""
        try:
            os.makedirs('models', exist_ok=True)
            
            # 保存植物类别映射
            with open(os.path.join('models', 'plant_classes.json'), 'w', encoding='utf-8') as f:
                json.dump(self.plant_classes, f, indent=2, ensure_ascii=False)
            
            # 保存病害类别映射
            with open(os.path.join('models', 'disease_classes.json'), 'w', encoding='utf-8') as f:
                json.dump(self.disease_classes, f, indent=2, ensure_ascii=False)
                
            self.logger.info("映射文件保存成功")
            return True
        except Exception as e:
            self.logger.error(f"保存映射文件失败: {e}")
            return False

    def get_compatible_disease_classes(self, plant_type):
        """
        获取与特定植物兼容的病害类别列表
        
        Args:
            plant_type: 植物类型名称
            
        Returns:
            list: 兼容的病害列表
        """
        # 从JSON文件加载植物-病害映射
        try:
            # 优先加载disease_compatibility.json文件
            compatibility_path = os.path.join('models', 'disease_compatibility.json')
            if os.path.exists(compatibility_path):
                with open(compatibility_path, 'r', encoding='utf-8') as f:
                    plant_disease_map = json.load(f)
                    self.logger.info(f"已从{compatibility_path}加载植物-病害兼容性数据")
            else:
                # 如果文件不存在，使用硬编码的映射
                self.logger.warning(f"未找到植物-病害兼容性文件，使用默认映射")
                plant_disease_map = {
                    "苹果": ["黑星病", "黑腐病", "雪松苹果锈病", "健康"],
                    "樱桃": ["白粉病", "健康"],
                    "玉米": ["灰斑病", "普通锈病", "健康", "北方叶枯病"],
                    "葡萄": ["黑腐病", "黑麻疹病", "健康", "叶枯病"],
                    "橙子": ["黄龙病", "健康"],
                    "桃子": ["细菌性斑点病", "健康"],
                    "甜椒": ["细菌性斑点病", "健康"],
                    "土豆": ["早疫病", "晚疫病", "健康"],
                    "树莓": ["健康"],
                    "大豆": ["健康"],
                    "西葫芦": ["白粉病", "健康"],
                    "草莓": ["叶焦病", "健康"],
                    "番茄": ["细菌性斑点病", "早疫病", "晚疫病", "叶霉病", "斑枯病", "二斑叶螨", "靶斑病", "花叶病毒病", "黄化曲叶病毒病", "健康"]
                }
        except Exception as e:
            self.logger.error(f"加载植物-病害兼容性数据失败: {e}")
            # 如果加载失败，返回默认映射
            plant_disease_map = {"健康": ["健康"]}
        
        # 标准化植物类型名称
        normalized_type = plant_type.lower().strip()
        
        # 如果植物类型在映射中，返回其兼容的病害，否则返回通用病害
        for plant_key in plant_disease_map:
            if normalized_type in plant_key.lower() or plant_key.lower() in normalized_type:
                return plant_disease_map[plant_key]
        
        # 如果没有匹配，至少返回"健康"状态
        return ["健康"]

    def extract_plant_type_from_class(self, class_name):
        """
        从类名中提取植物类型（移除病害部分）
        
        Args:
            class_name: 类别名称，可能是"植物-病害"格式
            
        Returns:
            str: 纯植物类型名称
        """
        if '-' in class_name:
            return class_name.split('-')[0]
        return class_name