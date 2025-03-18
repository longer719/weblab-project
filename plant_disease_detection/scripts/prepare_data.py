# scripts/prepare_data.py

import os
import sys
import cv2
import requests
import zipfile
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
import random
import json
import uuid
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
import argparse
from PIL import Image

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 修改导入方式
from src.utils.config_manager import ConfigManager
from src.utils.download import download_file

# 获取配置管理器实例
config_manager = ConfigManager()

# 使用配置管理器获取URL和文件名 (适当时需移到配置文件中)
PLANTVILLAGE_URL = config_manager.get(
    "PLANTVILLAGE_URL",
    "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
)
PLANTVILLAGE_FILENAME = config_manager.get(
    "PLANTVILLAGE_FILENAME", 
    "plantvillage_dataset.zip"
)

def download_plantvillage_dataset(config=None) -> Path:
    """
    下载PlantVillage数据集
    
    Args:
        config: 配置对象（可选）
        
    Returns:
        下载的zip文件路径
    """
    # 使用ConfigManager获取配置值
    raw_data_dir = config_manager.get_dict_compatible(config, "RAW_DATA_DIR", "data/raw")
    
    os.makedirs(raw_data_dir, exist_ok=True)
    zip_path = Path(raw_data_dir) / PLANTVILLAGE_FILENAME
    
    print(f"正在下载PlantVillage数据集到 {zip_path}...")
    download_file(PLANTVILLAGE_URL, zip_path)
    print("下载完成!")
    
    return zip_path

def extract_dataset(zip_path: Path, extract_dir: Path) -> Path:
    """
    解压数据集
    
    Args:
        zip_path: 下载的zip文件路径
        extract_dir: 解压目标目录
        
    Returns:
        解压后的目录路径
    """
    print(f"正在解压数据集到 {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print("解压完成!")
    
    # 找到解压后的主目录
    extracted_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
    if not extracted_dirs:
        raise FileNotFoundError(f"解压后未找到任何目录在 {extract_dir}")
    
    return extracted_dirs[0]  # 通常解压后会有一个主目录

def organize_dataset(source_dir: Path, target_dir: Path, class_mapping: Dict[str, int]) -> List[Dict]:
    """组织数据集到标准格式"""
    print("正在组织数据集...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # 获取图像尺寸，通过ConfigManager
    image_size = tuple(config_manager.get("IMAGE_SIZE", (224, 224)))
    
    # 创建图像目录
    images_dir = target_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # 输出处理信息
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"图像大小: {image_size}")
    print(f"类别映射包含 {len(class_mapping)} 个类别")
    
    # 检查数据集类别和映射是否匹配
    actual_class_dirs = [d.name for d in source_dir.iterdir() if d.is_dir()]
    unknown_classes = [d for d in actual_class_dirs if d not in class_mapping and d != "Background_without_leaves"]
    missing_classes = [c for c in class_mapping if c not in actual_class_dirs]
    
    if unknown_classes:
        print(f"警告: 发现 {len(unknown_classes)} 个未包含在映射中的类别: {unknown_classes}")
        # 为未知类别动态添加映射，而不是跳过它们
        next_id = max(class_mapping.values()) + 1
        for cls in unknown_classes:
            print(f"为未知类别添加映射: {cls} -> {next_id}")
            class_mapping[cls] = next_id
            next_id += 1
            
    if missing_classes:
        print(f"警告: 映射中有 {len(missing_classes)} 个类别在数据目录中未找到: {missing_classes}")
    
    # 使用tqdm显示进度
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob('*.jpg'))) for d in class_dirs if (d.name in class_mapping or d.name == "Background_without_leaves"))
    progress_bar = tqdm(total=total_images, desc="处理图像")
    
    # 为每个类别创建目录并处理图像
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # 跳过背景图像
        if class_name == "Background_without_leaves":
            print(f"跳过背景图像: {class_name}")
            continue
        
        # 检查类别是否在映射中
        if class_name not in class_mapping:
            print(f"跳过未知类别: {class_name}")
            continue
        
        label = class_mapping[class_name]
        print(f"处理类别: {class_name} (标签: {label})")
        
        # 处理该类别中的所有图像
        image_files = list(class_dir.glob('*.jpg'))
        
        for img_path in image_files:
            try:
                # 读取图像
                image = Image.open(img_path)
                
                # 调整大小
                if image_size:
                    image = image.resize(image_size, Image.LANCZOS)
                
                # 生成唯一文件名
                unique_id = uuid.uuid4().hex[:8]
                new_filename = f"{class_name}_{unique_id}.jpg"
                target_path = images_dir / new_filename
                
                # 保存处理后的图像
                image.save(target_path, "JPEG", quality=95)
                
                # 添加到数据列表
                all_data.append({
                    'image_id': new_filename,
                    'image_path': str(target_path.relative_to(target_dir)),
                    'class_name': class_name,
                    'label': label
                })
                
                # 更新进度条
                progress_bar.update(1)
            
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
    
    progress_bar.close()
    print(f"数据组织完成! 总共处理了 {len(all_data)} 张图像")
    return all_data

def split_dataset(data: List[Dict], train_ratio=None, val_ratio=None, test_ratio=None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    将数据集划分为训练、验证和测试集
    
    Args:
        data: 完整数据集
        train_ratio: 训练集比例 (默认从配置获取)
        val_ratio: 验证集比例 (默认从配置获取)
        test_ratio: 测试集比例 (默认从配置获取)
        
    Returns:
        训练集、验证集和测试集
    """
    # 从配置获取比例，如果未指定
    if train_ratio is None:
        train_ratio = config_manager.get("TRAIN_RATIO", 0.7, "data")
    if val_ratio is None:
        val_ratio = config_manager.get("VAL_RATIO", 0.15, "data")
    if test_ratio is None:
        test_ratio = config_manager.get("TEST_RATIO", 0.15, "data")
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必须为1"
    
    # 按类别分组
    class_data = {}
    for item in data:
        class_id = item['label']
        if class_id not in class_data:
            class_data[class_id] = []
        class_data[class_id].append(item)
    
    train_data, val_data, test_data = [], [], []
    
    # 按类别划分，确保每个类别在每个集合中都有代表
    for class_name, items in class_data.items():
        random.shuffle(items)
        n = len(items)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:])
        test_data.extend(items[val_end:])
    
    # 再次打乱
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print(f"数据集划分完成: 训练集 {len(train_data)}张, 验证集 {len(val_data)}张, 测试集 {len(test_data)}张")
    return train_data, val_data, test_data

def split_dataset_with_balance(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                             balance_threshold=0.3):
    """
    平衡类别分布的数据划分
    
    balance_threshold: 允许的最大类别不平衡比率
    """
    # 获取类别分布
    class_counts = {}
    for item in data:
        class_name = item['class_name']
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    # 计算每个类别在每个集合中的目标数量
    min_count = min(class_counts.values())
    
    for class_name, count in class_counts.items():
        if count / min_count > (1 + balance_threshold):
            # 对于过多的类别，采样减少
            logging.info(f"类别 {class_name} 样本过多，将进行下采样")
    
    # 实现分层抽样，确保每个类在各集合中的比例一致且符合要求
    # ...具体实现省略...

def save_labels(data: List[Dict], output_path: Path):
    """保存标签到CSV文件"""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"保存标签文件到 {output_path}")

def create_plant_disease_mapping() -> Dict[str, int]:
    """创建PlantVillage数据集的类别映射"""
    # 完全硬编码映射，确保包含所有38个类别
    return {
        'Apple___Apple_scab': 0,
        'Apple___Black_rot': 1,
        'Apple___Cedar_apple_rust': 2,
        'Apple___healthy': 3,
        'Blueberry___healthy': 4,
        'Cherry___healthy': 5,
        'Cherry___Powdery_mildew': 6,
        'Corn___Cercospora_leaf_spot Gray_leaf_spot': 7,
        'Corn___Common_rust': 8,
        'Corn___healthy': 9,
        'Corn___Northern_Leaf_Blight': 10,
        'Grape___Black_rot': 11,
        'Grape___Esca_(Black_Measles)': 12,
        'Grape___healthy': 13,
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 14,
        'Orange___Haunglongbing_(Citrus_greening)': 15,
        'Peach___Bacterial_spot': 16,
        'Peach___healthy': 17,
        'Pepper,_bell___Bacterial_spot': 18,
        'Pepper,_bell___healthy': 19,
        'Potato___Early_blight': 20,
        'Potato___Late_blight': 21,
        'Potato___healthy': 22,
        'Raspberry___healthy': 23,
        'Soybean___healthy': 24,
        'Squash___Powdery_mildew': 25,
        'Strawberry___Leaf_scorch': 26,
        'Strawberry___healthy': 27,
        'Tomato___Bacterial_spot': 28,
        'Tomato___Early_blight': 29,
        'Tomato___Late_blight': 30,
        'Tomato___Leaf_Mold': 31,
        'Tomato___Septoria_leaf_spot': 32,
        'Tomato___Spider_mites Two-spotted_spider_mite': 33,
        'Tomato___Target_Spot': 34,
        'Tomato___Tomato_mosaic_virus': 35,
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 36,
        'Tomato___healthy': 37
    }

def process_plantvillage_dataset(filter_classes=None):
    """
    处理PlantVillage数据集
    
    Args:
        filter_classes: 要处理的植物类别列表，如None则处理所有类别
    """
    # 使用ConfigManager获取数据目录
    raw_data_dir = config_manager.get("RAW_DATA_DIR", "data/raw")
    processed_dir = config_manager.get("PROCESSED_DATA_DIR", "data/processed")
    
    # 创建目录
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 获取类别映射
    class_mapping = create_plant_disease_mapping()
    
    # 在这里插入日志，输出映射信息
    print(f"【DEBUG】映射中包含 {len(class_mapping)} 个类别")
    print(f"【DEBUG】类别映射的前5个元素: {list(class_mapping.items())[:5]}")
    print(f"【DEBUG】是否包含'Blueberry___healthy': {'Blueberry___healthy' in class_mapping}")
    
    # 修改这里：不再使用默认过滤，我们要处理所有14种植物
    if filter_classes is None:
        # 处理所有植物类别
        filter_classes = [
            "Apple", "Blueberry", "Cherry", "Corn", "Grape", 
            "Orange", "Peach", "Pepper", "Potato", "Raspberry", 
            "Soybean", "Squash", "Strawberry", "Tomato"
        ]
    
    print(f"将处理以下植物类别: {filter_classes}")
    
    # 下载并解压数据集
    extracted_dir = Path(raw_data_dir) / "extracted"
    
    if not extracted_dir.exists():
        zip_path = download_plantvillage_dataset()
        extracted_dir.mkdir(parents=True, exist_ok=True)
        extract_dataset(zip_path, extracted_dir)
    else:
        print(f"使用已解压的数据集: {extracted_dir}")
    
    # 查找PlantVillage数据目录
    source_dirs = list(extracted_dir.glob("**/color"))
    if not source_dirs:
        # 尝试找任何包含数据的目录
        source_dirs = list(extracted_dir.glob("**/*/"))
        if not source_dirs:
            raise FileNotFoundError(f"在 {extracted_dir} 中未找到PlantVillage数据集")
    
    # 检查是否存在可能的数据目录
    valid_source_dirs = []
    for src_dir in source_dirs:
        if src_dir.is_dir() and any(d.is_dir() and d.name.startswith(tuple(filter_classes)) for d in src_dir.iterdir()):
            valid_source_dirs.append(src_dir)
    
    if not valid_source_dirs:
        raise FileNotFoundError(f"在 {extracted_dir} 中未找到含有指定植物类型的目录")
    
    source_dir = valid_source_dirs[0]
    print(f"找到PlantVillage数据目录: {source_dir}")
    
    # 获取类别映射
    class_mapping = create_plant_disease_mapping()
    
    # 过滤所需的植物类别
    if filter_classes:
        filtered_class_mapping = {}
        for class_name, idx in class_mapping.items():
            # 检查原始分类名
            plant_name = class_name.split('___')[0]  # 提取植物名称
            if plant_name in filter_classes:
                filtered_class_mapping[class_name] = idx
        
        # 如果过滤后没有类别，可能是格式问题，尝试更宽松的匹配
        if not filtered_class_mapping:
            for class_name, idx in class_mapping.items():
                plant_name = class_name.split('___')[0]
                if any(filter_plant in plant_name for filter_plant in filter_classes):
                    filtered_class_mapping[class_name] = idx
        
        # 重新编号，确保连续
        new_mapping = {}
        for idx, (class_name, _) in enumerate(filtered_class_mapping.items()):
            new_mapping[class_name] = idx
        
        class_mapping = new_mapping
        print(f"过滤后保留 {len(class_mapping)} 个类别: {list(class_mapping.keys())}")
    
    # 组织数据集
    processed_data_dir = Path(processed_dir)
    all_data = organize_dataset(source_dir, processed_data_dir, class_mapping)
    
    # 保存类别映射
    with open(processed_data_dir / "class_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # 划分并保存数据集
    train_data, val_data, test_data = split_dataset(all_data)
    
    save_labels(train_data, processed_data_dir / "train_labels.csv")
    save_labels(val_data, processed_data_dir / "val_labels.csv")
    save_labels(test_data, processed_data_dir / "test_labels.csv")
    
    print("数据集处理完成！")
    
    return {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "total_size": len(all_data),
        "num_classes": len(class_mapping),
        "classes": list(class_mapping.keys())
    }

def create_sample_data(base_path: str, num_samples: int = 10):
    """创建示例数据集，用于开发和测试"""
    processed_dir = Path(base_path) / 'processed'
    images_dir = processed_dir / 'images'
    
    # 创建目录
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分类
    classes = ['healthy', 'disease_1', 'disease_2']
    class_mapping = {name: idx for idx, name in enumerate(classes)}
    
    # 保存类别映射
    with open(processed_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    # 创建示例图片
    for split in ['train', 'val', 'test']:
        # 为每个类别创建目录
        for class_name in classes:
            (images_dir / class_name).mkdir(parents=True, exist_ok=True)
            
        # 创建数据标签
        data = []
        for i in range(num_samples):
            for class_name in classes:
                img_name = f"{split}_{class_name}_{i}.jpg"
                img_path = images_dir / class_name / img_name
                
                # 创建彩色随机图片
                if class_name == 'healthy':
                    # 健康植物为绿色主导
                    img = np.ones((224, 224, 3), dtype=np.uint8) * np.array([30, 180, 30], dtype=np.uint8)
                else:
                    # 病害植物添加褐色或黄色斑点
                    img = np.ones((224, 224, 3), dtype=np.uint8) * np.array([30, 180, 30], dtype=np.uint8)
                    spots = np.random.rand(224, 224) > 0.8
                    img[spots] = [60, 90, 180] if class_name == 'disease_1' else [30, 100, 220]
                
                # 添加随机噪声
                noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # 保存图片
                cv2.imwrite(str(img_path), img)
                
                # 添加到标签列表
                data.append({
                    'image_path': f"images/{class_name}/{img_name}",
                    'label': class_mapping[class_name],
                    'class_name': class_name
                })
        
        # 保存标签文件
        df = pd.DataFrame(data)
        df.to_csv(processed_dir / f'{split}_labels.csv', index=False)

def download_dataset():
    """通用下载数据集函数，调用具体数据集的下载函数"""
    return download_plantvillage_dataset()

def validate_dataset(processed_dir: Path) -> Dict:
    """验证处理后的数据集质量"""
    validation_results = {
        'file_count': 0,
        'corrupted_images': 0,
        'class_distribution': {},
        'image_sizes': [],
        'errors': []
    }
    
    # 检查所有图像文件
    for img_path in tqdm(list(processed_dir.glob('images/**/*.jpg')), desc="验证图像"):
        validation_results['file_count'] += 1
        
        # 验证图像可读性
        try:
            with Image.open(img_path) as img:
                # 记录图像尺寸
                width, height = img.size
                validation_results['image_sizes'].append((width, height))
                
                # 提取类别名称（从文件名中）
                class_name = img_path.stem.split('_')[0]
                if class_name not in validation_results['class_distribution']:
                    validation_results['class_distribution'][class_name] = 0
                validation_results['class_distribution'][class_name] += 1
                
        except Exception as e:
            validation_results['corrupted_images'] += 1
            validation_results['errors'].append(f"文件 {img_path} 损坏: {str(e)}")
    
    # 保存验证结果
    with open(processed_dir / 'dataset_validation.json', 'w') as f:
        # 使用自定义JSON序列化器处理numpy数组等特殊对象
        json.dump(validation_results, f, indent=2, default=lambda o: str(o))
    
    print(f"验证完成: 共 {validation_results['file_count']} 文件, "
          f"{validation_results['corrupted_images']} 损坏图像")
        
    return validation_results

def process_dataset_with_checkpoints(dataset_name, filter_classes=None, force=False):
    """支持从特定阶段开始的数据处理，完整的数据处理流程"""
    try:
        # 1. 下载数据集
        zip_path = download_plantvillage_dataset()
        
        # 2. 解压数据集
        raw_data_dir = Path(config_manager.get("RAW_DATA_DIR", "data/raw"))
        extract_dir = raw_data_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        dataset_dir = extract_dataset(zip_path, extract_dir)
        
        # 3. 创建类别映射
        class_mapping = create_plant_disease_mapping()
        original_mapping_size = len(class_mapping)
        
        # 如果指定了过滤，只保留相关类别
        if filter_classes:
            # 创建过滤函数，更灵活地匹配类别
            def should_keep(class_name):
                plant = class_name.split('___')[0].replace(',_bell', '') 
                return any(filter_plant == plant or filter_plant in plant for filter_plant in filter_classes)
            
            filtered_mapping = {k: v for k, v in class_mapping.items() if should_keep(k)}
            
            # 检查过滤结果
            if not filtered_mapping:
                print(f"警告: 过滤条件 {filter_classes} 未匹配任何类别，将使用全部类别")
            else:
                # 重新编号
                new_mapping = {}
                for i, (class_name, _) in enumerate(filtered_mapping.items()):
                    new_mapping[class_name] = i
                    
                class_mapping = new_mapping
                print(f"过滤后保留 {len(class_mapping)} 个类别，过滤前有 {original_mapping_size} 个类别")
        
        # 4. 组织数据集
        processed_dir = Path(config_manager.get("PROCESSED_DATA_DIR", "data/processed"))
        all_data = organize_dataset(dataset_dir, processed_dir, class_mapping)
        
        # 5. 划分数据集
        train_data, val_data, test_data = split_dataset(all_data)
        
        # 6. 保存标签
        save_labels(train_data, processed_dir / "train_labels.csv")
        save_labels(val_data, processed_dir / "val_labels.csv")
        save_labels(test_data, processed_dir / "test_labels.csv")
        
        # 保存类别映射
        with open(processed_dir / "class_mapping.json", "w") as f:
            json.dump({k: int(v) for k, v in class_mapping.items()}, f, indent=2)
        
        print("PlantVillage数据集处理完成!")
        return True
    except Exception as e:
        print(f"处理数据集时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def clean_processed_data():
    """清理已处理的数据"""
    processed_dir = Path(config_manager.get("PROCESSED_DATA_DIR", "data/processed"))
    
    if processed_dir.exists():
        print(f"清理数据目录: {processed_dir}")
        
        # 删除标签文件
        for label_file in ["train_labels.csv", "val_labels.csv", "test_labels.csv", "class_mapping.json"]:
            file_path = processed_dir / label_file
            if file_path.exists():
                file_path.unlink()
                print(f"已删除: {file_path}")
        
        # 删除图像目录
        images_dir = processed_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
            print(f"已删除目录: {images_dir}")
            
        return True
    else:
        print(f"目录不存在，无需清理: {processed_dir}")
        return False

def clean_processed_data():
    """清理处理后的数据目录"""
    processed_dir = Path(config_manager.get("PROCESSED_DATA_DIR", "data/processed"))
    
    if not processed_dir.exists():
        print(f"目录不存在: {processed_dir}")
        return
    
    # 询问用户确认
    confirmation = input(f"将删除 {processed_dir} 中的所有文件，确认? (y/n): ")
    if confirmation.lower() != 'y':
        print("操作已取消")
        return
    
    # 删除目录中的文件，但保留目录结构
    print(f"清理目录: {processed_dir}")
    
    # 删除图像文件
    images_dir = processed_dir / "images"
    if images_dir.exists():
        for file in images_dir.glob("*"):
            if file.is_file():
                file.unlink()
        print(f"已清理图像目录: {images_dir}")
    
    # 删除标签文件
    for filename in ["train_labels.csv", "val_labels.csv", "test_labels.csv", "class_mapping.json"]:
        filepath = processed_dir / filename
        if filepath.exists():
            filepath.unlink()
            print(f"已删除: {filepath}")
    
    print("清理完成")

def configure_argument_parser():
    """配置命令行参数解析器"""
    parser = argparse.ArgumentParser(description='植物病害数据集处理工具')
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 下载命令
    download_parser = subparsers.add_parser('download', help='下载数据集')
    
    # 处理命令
    process_parser = subparsers.add_parser('process', help='处理数据集')
    process_parser.add_argument('--filter', type=str, help='要处理的植物类型列表，以逗号分隔 (例如: Corn,Grape,Orange)')
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='验证处理后的数据集')
    
    # 清理命令
    clean_parser = subparsers.add_parser('clean', help='清理已处理的数据')
    
    # 全流程命令
    all_parser = subparsers.add_parser('all', help='执行完整的数据处理流程')
    all_parser.add_argument('--force', action='store_true', help='强制重新执行所有步骤，即使已完成')
    all_parser.add_argument('--filter', type=str, help='要处理的植物类型列表，以逗号分隔 (例如: Corn,Grape,Orange)')
    
    return parser

def main():
    """主函数"""
    args = configure_argument_parser().parse_args()
    
    # 设置随机种子以确保可重现性
    random.seed(42)
    np.random.seed(42)
    
    if args.command == 'clean':
        clean_processed_data()
    elif args.command == 'all' or args.command is None:  # 默认执行全部流程
        # 如果指定了过滤条件则应用，否则处理所有数据
        filter_classes = args.filter.split(',') if hasattr(args, 'filter') and args.filter else None
        force = args.force if hasattr(args, 'force') else False
        
        # 调用处理函数，如果filter_classes为None，则会处理所有植物
        process_dataset_with_checkpoints('plantvillage', filter_classes, force)
        
        # 验证处理后的数据
        processed_dir = Path(config_manager.get("PROCESSED_DATA_DIR", "data/processed"))
        validate_dataset(processed_dir)
    # 其他命令处理...

if __name__ == '__main__':
    main()
    print("数据准备完成!")

def normalize_class_name(class_name: str) -> str:
    """标准化类别名称，处理不同表示方式的差异"""
    # 移除多余的逗号和下划线
    normalized = class_name.replace(',_bell', '')
    
    # 拆分并重组，标准化格式
    if "___" in normalized:
        parts = normalized.split("___")
        return f"{parts[0].strip()}___{parts[1].strip()}"
    
    return normalized