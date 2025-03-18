#src/utils/path_utils.py
"""
路径处理工具，提供跨平台路径处理功能，为算力云平台提供支持
"""

import os
import logging
from pathlib import Path
from typing import Union, List, Optional

def normalize_path(path_str: str) -> str:
    """标准化路径，确保路径分隔符兼容当前操作系统"""
    return str(Path(path_str.replace('\\', '/')))

def ensure_dir_exists(directory: Union[str, Path]) -> Path:
    """确保目录存在，如果不存在则创建"""
    dir_path = Path(directory)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"创建目录: {dir_path}")
    return dir_path

def find_file_case_insensitive(directory: Union[str, Path], filename: str) -> Optional[Path]:
    """不区分大小写地查找文件
    
    Args:
        directory: 要搜索的目录
        filename: 要查找的文件名
    
    Returns:
        找到的文件路径，如果未找到则返回None
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    
    # 将文件名转换为小写以进行不区分大小写的比较
    filename_lower = filename.lower()
    
    for file in dir_path.glob('*'):
        if file.name.lower() == filename_lower:
            return file
    
    return None

def list_files(directory: Union[str, Path], pattern: str = '*') -> List[Path]:
    """列出目录中的文件
    
    Args:
        directory: 要列出文件的目录
        pattern: 文件匹配模式
    
    Returns:
        文件路径列表
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    return list(dir_path.glob(pattern))