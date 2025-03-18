#scripts/deploy_models.py
#!/usr/bin/env python

"""
模型部署脚本 - 将训练好的实验模型部署到应用程序使用的位置
"""

import os
import shutil
import json
import argparse
from datetime import datetime
import logging
import fnmatch
import torch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加全局args变量
args = None

def deploy_model(experiment_path, target_dir, model_name, config_only=False):
    """
    将模型从实验路径部署到目标目录
    
    Args:
        experiment_path: 实验路径（如disease_detector_v01）
        target_dir: 目标目录（通常是models/）
        model_name: 部署后的模型名称（如disease_detector.pth）
        config_only: 是否仅复制配置文件
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 检查多个可能的模型目录位置
    potential_dirs = [
        os.path.join(experiment_path, 'models'),  # 标准模型目录
        os.path.join(experiment_path, 'checkpoints'),  # 检查点目录
        experiment_path  # 实验根目录
    ]
    
    # 查找有效的模型目录
    model_dir = None
    for dir_path in potential_dirs:
        if os.path.exists(dir_path):
            # 检查此目录中是否有.pth文件
            pth_files = [f for f in os.listdir(dir_path) if f.endswith('.pth')]
            if pth_files:
                model_dir = dir_path
                logger.info(f"找到包含模型文件的目录: {model_dir}")
                break
    
    if model_dir is None:
        logger.error(f"无法在{experiment_path}或其子目录中找到任何模型文件")
        return False
    
    # 查找模型文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    # 选择最佳模型文件（按优先级）
    preferred_files = ['best_model.pth', f'{model_name.split(".")[0]}.pth']
    selected_model = None
    
    for preferred in preferred_files:
        if preferred in model_files:
            selected_model = preferred
            break
    
    # 如果未找到首选文件，选择最新的模型文件
    if selected_model is None:
        selected_model = max(model_files, 
                             key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    
    model_source_path = os.path.join(model_dir, selected_model)
    model_target_path = os.path.join(target_dir, model_name)
    
    # 查找配置文件
    config_found = False
    config_source_path = None
    
    # 在所有可能目录中查找配置文件
    for dir_path in potential_dirs:
        if os.path.exists(dir_path):
            config_files = [f for f in os.listdir(dir_path) if f.endswith('_config.json')]
            if config_files:
                config_source_path = os.path.join(dir_path, config_files[0])
                config_found = True
                logger.info(f"找到配置文件: {config_source_path}")
                break
    
    if not config_found:
        logger.error(f"在{experiment_path}或其子目录中没有找到配置文件")
        return False
    
    config_target_path = os.path.join(target_dir, model_name.replace('.pth', '_config.json'))
    
    # 复制配置文件
    logger.info(f"正在复制配置文件 {config_source_path} -> {config_target_path}")
    shutil.copy2(config_source_path, config_target_path)
    
    # 检查目标目录中是否已有备份文件，如果有则先备份
    if os.path.exists(model_target_path) and not config_only:
        # 创建备份目录
        backup_dir = os.path.join(target_dir, f"backup_{datetime.now().strftime('%Y%m%d')}")
        os.makedirs(backup_dir, exist_ok=True)
        
        # 备份旧模型
        backup_model_path = os.path.join(backup_dir, model_name)
        logger.info(f"备份旧模型 {model_target_path} -> {backup_model_path}")
        shutil.copy2(model_target_path, backup_model_path)
    
    # 复制模型文件
    if not config_only:
        logger.info(f"正在复制模型文件 {model_source_path} -> {model_target_path}")
        shutil.copy2(model_source_path, model_target_path)
    
    # 更新模型映射文件
    if "plant_classifier" in model_name:
        # 更新植物类别映射
        update_class_mappings(target_dir, "plant", experiment_path)
    elif "disease_detector" in model_name:
        # 更新病害类别映射
        update_class_mappings(target_dir, "disease", experiment_path)
    
    # 更新版本信息
    update_version_info(target_dir, model_name.replace('.pth', ''))
    
    return True

def update_class_mappings(target_dir, mapping_type, experiment_path):
    """更新类别映射文件"""
    # 查找实验目录中的类别映射文件
    search_pattern = "*_classes.json" if mapping_type == "plant" else "*_disease_classes.json"
    mapping_files = []
    for root, dirs, files in os.walk(experiment_path):
        for file in files:
            if fnmatch.fnmatch(file, search_pattern):
                mapping_files.append(os.path.join(root, file))
    
    # 如果找到映射文件，复制到目标目录
    if (mapping_files):
        latest_mapping = max(mapping_files, key=os.path.getmtime)
        target_file = os.path.join(target_dir, 
                                  "plant_classes.json" if mapping_type == "plant" else "disease_classes.json")
        logger.info(f"更新{mapping_type}类别映射: {latest_mapping} -> {target_file}")
        shutil.copy2(latest_mapping, target_file)
    else:
        logger.warning(f"在实验目录中未找到{mapping_type}类别映射文件")

def update_version_info(target_dir, model_type):
    """
    更新模型版本信息
    
    Args:
        target_dir: 目标目录
        model_type: 模型类型（如plant_classifier或disease_detector）
    """
    version_file = os.path.join(target_dir, 'version_info.json')
    version_data = {}
    
    # 如果文件已存在，先读取当前版本信息
    if os.path.exists(version_file):
        try:
            # 尝试以不同的编码读取文件
            encodings = ['utf-8', 'cp1252', 'latin-1', 'gbk']
            for encoding in encodings:
                try:
                    with open(version_file, 'r', encoding=encoding) as f:
                        version_data = json.load(f)
                    logger.info(f"使用 {encoding} 编码成功读取版本信息")
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            
            # 如果所有编码都失败了，创建新文件
            if not version_data:
                logger.warning(f"无法读取版本文件: {version_file}，将创建新文件")
        except Exception as e:
            logger.warning(f"读取版本文件时发生错误: {e}，将创建新文件")
    
    # 确定模型键名
    model_key = 'classifier' if model_type == 'plant_classifier' else 'detector'
    
    # 确定版本号
    current_version = version_data.get(model_key, {}).get('version', '0.0.0')
    if current_version.startswith('v'):
        current_version = current_version[1:]  # 移除前缀'v'
    version_parts = current_version.split('.')
    if len(version_parts) != 3:
        version_parts = ['1', '0', '0']  # 默认版本号
    
    try:
        new_version = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}"
    except (ValueError, IndexError):
        new_version = "1.0.1"  # 如果解析失败，使用默认版本号
    
    # 更新版本信息
    version_data[model_key] = {
        'version': f"v{new_version}",
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': f'从实验部署的{model_key}模型',
        'experiment_dir': os.path.abspath(args.classifier_path if model_key == 'classifier' else args.detector_path)
    }
    
    # 写入更新后的版本信息
    try:
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version_data, f, indent=2, ensure_ascii=False)
        logger.info(f"已更新{model_key}模型版本信息为: v{new_version}")
    except Exception as e:
        logger.error(f"保存版本信息失败: {e}")

def convert_model_for_local(model_path, output_path=None):
    """
    将云端训练的模型转换为本地可用格式
    
    Args:
        model_path: 云端训练的模型路径
        output_path: 输出路径
    
    Returns:
        转换后的模型路径
    """
    if output_path is None:
        output_path = model_path.replace('.pth', '_converted.pth')
    
    logger.info(f"转换模型: {model_path} -> {output_path}")
    
    try:
        # 加载模型
        state_dict = torch.load(model_path, map_location='cpu')
        
        # 处理可能的格式差异
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            # 保存的是完整检查点，只提取模型状态
            state_dict = state_dict['model_state_dict']
        
        # 处理可能的DataParallel包装
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除'module.'前缀(如果存在)
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        
        # 保存转换后的模型
        torch.save(new_state_dict, output_path)
        logger.info(f"模型成功转换为本地格式: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"模型转换失败: {e}")
        return None

def main():
    global args  # 在函数开头添加此行
    parser = argparse.ArgumentParser(description='部署训练好的模型到应用程序')
    parser.add_argument('--classifier-path', type=str, default='experiments/plant_classifier_v02',
                      help='分类器实验路径')
    parser.add_argument('--detector-path', type=str, default='experiments/plant_disease_detector_v02',
                      help='检测器实验路径')
    parser.add_argument('--target-dir', type=str, default='models',
                      help='部署目标目录')
    parser.add_argument('--config-only', action='store_true',
                      help='仅复制配置文件，不复制模型文件')
    
    args = parser.parse_args()
    
    logger.info("开始部署模型...")
    
    # 部署分类器模型
    success_classifier = deploy_model(
        args.classifier_path,
        args.target_dir,
        'plant_classifier.pth',
        args.config_only
    )
    
    # 部署检测器模型
    success_detector = deploy_model(
        args.detector_path,
        args.target_dir,
        'disease_detector.pth',
        args.config_only
    )
    
    if success_classifier and success_detector:
        logger.info("模型部署成功！")
    else:
        logger.warning("部分模型部署失败，请查看上方日志")

if __name__ == '__main__':
    main()