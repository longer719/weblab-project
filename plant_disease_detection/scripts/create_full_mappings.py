#scripts/create_full_mappings.py
#!/usr/bin/env python
# filepath: d:\Git_Workspace\weblab-project\plant_disease_detection\scripts\create_full_mappings.py
"""
创建完整的映射文件集合

生成所有需要的映射文件，包括:
1. plant_mappings.json - 植物英文名到中文名的映射
2. disease_mappings.json - 病害英文名到中文名的映射
3. class_mappings.json - 完整的类别索引到类名的映射
"""

import json
import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_plant_mapping() -> Dict[str, str]:
    """创建植物英文名到中文名的映射"""
    return {
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

def create_disease_mapping() -> Dict[str, str]:
    """创建病害英文名到中文名的映射"""
    return {
        'healthy': '健康',
        'Apple_scab': '黑星病',
        'Black_rot': '黑腐病',
        'Cedar_apple_rust': '雪松苹果锈病',
        'Powdery_mildew': '白粉病',
        'Cercospora_leaf_spot Gray_leaf_spot': '灰斑病',
        'Common_rust': '普通锈病',
        'Northern_Leaf_Blight': '北方叶枯病',
        'Esca_(Black_Measles)': '黑麻疹病',
        'Leaf_blight_(Isariopsis_Leaf_Spot)': '叶枯病',
        'Haunglongbing_(Citrus_greening)': '黄龙病',
        'Bacterial_spot': '细菌性斑点病',
        'Early_blight': '早疫病',
        'Late_blight': '晚疫病',
        'Leaf_Mold': '叶霉病',
        'Septoria_leaf_spot': '斑枯病',
        'Spider_mites Two-spotted_spider_mite': '二斑叶螨',
        'Target_Spot': '靶斑病',
        'Tomato_mosaic_virus': '花叶病毒病',
        'Tomato_Yellow_Leaf_Curl_Virus': '黄化曲叶病毒病',
        'Leaf_scorch': '叶焦病'
    }

def create_complete_class_mapping() -> Dict[str, str]:
    """创建完整的类别索引到类名的映射"""
    # 确保包含所有38个类别，与prepare_data.py中的完全一致
    class_map = {
        "0": "苹果-黑星病",
        "1": "苹果-黑腐病",
        "2": "苹果-雪松苹果锈病",  # 修正：由"雪松锈病"改为"雪松苹果锈病"
        "3": "苹果-健康",
        "4": "蓝莓-健康",
        "5": "樱桃-健康",
        "6": "樱桃-白粉病",
        "7": "玉米-灰斑病",
        "8": "玉米-普通锈病",
        "9": "玉米-健康",
        "10": "玉米-北方叶枯病",
        "11": "葡萄-黑腐病",
        "12": "葡萄-黑麻疹病",
        "13": "葡萄-健康",
        "14": "葡萄-叶枯病",
        "15": "橙子-黄龙病",
        "16": "桃子-细菌性斑点病",
        "17": "桃子-健康",
        "18": "甜椒-细菌性斑点病",
        "19": "甜椒-健康",
        "20": "土豆-早疫病",
        "21": "土豆-晚疫病",
        "22": "土豆-健康",
        "23": "树莓-健康",
        "24": "大豆-健康",
        "25": "西葫芦-白粉病",
        "26": "草莓-叶焦病",
        "27": "草莓-健康",
        "28": "番茄-细菌性斑点病",
        "29": "番茄-早疫病",
        "30": "番茄-晚疫病",
        "31": "番茄-叶霉病",
        "32": "番茄-斑枯病",
        "33": "番茄-二斑叶螨",
        "34": "番茄-靶斑病",
        "35": "番茄-花叶病毒病",
        "36": "番茄-黄化曲叶病毒病",
        "37": "番茄-健康"
    }
    return class_map

def create_disease_classes_mapping() -> Dict[str, str]:
    """创建病害类别映射（用于检测器模型）"""
    disease_classes = {
        "0": "健康",
        "1": "黑星病",
        "2": "黑腐病",
        "3": "雪松苹果锈病",
        "4": "白粉病",
        "5": "灰斑病",
        "6": "普通锈病",
        "7": "北方叶枯病", 
        "8": "黑麻疹病",
        "9": "叶枯病",
        "10": "黄龙病",
        "11": "细菌性斑点病",
        "12": "早疫病",
        "13": "晚疫病",
        "14": "叶霉病",
        "15": "斑枯病",
        "16": "二斑叶螨",
        "17": "靶斑病",
        "18": "花叶病毒病",
        "19": "黄化曲叶病毒病",
        "20": "叶焦病"
    }
    return disease_classes

def create_all_mappings():
    """创建所有需要的映射文件"""
    # 1. 创建植物英文名到中文名的映射
    plant_mapping = create_plant_mapping()
    
    # 2. 创建病害英文名到中文名的映射
    disease_mapping = create_disease_mapping()
    
    # 3. 创建完整的类别索引到类名的映射
    class_mapping = create_complete_class_mapping()
    
    # 4. 创建病害类别映射（用于检测器模型）
    disease_classes = create_disease_classes_mapping()
    
    # 创建输出目录
    os.makedirs('models', exist_ok=True)
    
    # 保存映射文件
    with open(os.path.join('models', 'plant_mappings.json'), 'w', encoding='utf-8') as f:
        json.dump(plant_mapping, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join('models', 'disease_mappings.json'), 'w', encoding='utf-8') as f:
        json.dump(disease_mapping, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join('models', 'plant_classes.json'), 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join('models', 'disease_classes.json'), 'w', encoding='utf-8') as f:
        json.dump(disease_classes, f, indent=2, ensure_ascii=False)
    
    logger.info(f"已创建植物映射文件，共{len(plant_mapping)}种植物")
    logger.info(f"已创建病害映射文件，共{len(disease_mapping)}种病害")
    logger.info(f"已创建类别映射文件，共{len(class_mapping)}个类别")
    logger.info(f"已创建病害类别映射文件，共{len(disease_classes)}种病害类型")

def main():
    try:
        create_all_mappings()
        return 0
    except Exception as e:
        logger.error(f"创建映射文件时出错: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())