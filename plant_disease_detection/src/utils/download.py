# src/utils/download.py

import requests
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Union
import logging
import time

def download_file(url: str, destination: Union[str, Path], chunk_size: int = 8192, retries: int = 3, backoff_factor: float = 1.5):
    """
    带有指数退避重试的下载函数
    
    Args:
        url: 下载链接
        destination: 保存路径
        chunk_size: 分块大小
        retries: 重试次数
        backoff_factor: 退避因子
    """
    retry_count = 0
    while retry_count <= retries:
        try:
            # 确保目标目录存在
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果文件已存在且大小不为0，跳过下载
            if destination.exists() and destination.stat().st_size > 0:
                print(f"文件已存在: {destination}")
                return destination
            
            # 创建临时文件
            temp_file = str(destination) + ".tmp"
            
            # 发起请求
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 文件总大小
            total_size = int(response.headers.get('content-length', 0))
            
            # 已下载大小（断点续传）
            initial_pos = 0
            if os.path.exists(temp_file):
                initial_pos = os.path.getsize(temp_file)
                if initial_pos < total_size:
                    # 断点续传
                    header = {"Range": f"bytes={initial_pos}-"}
                    response = requests.get(url, stream=True, headers=header, timeout=30)
                else:
                    # 已经下载完成
                    os.rename(temp_file, destination)
                    return destination
            
            # 显示下载进度条
            mode = 'ab' if initial_pos else 'wb'
            with open(temp_file, mode) as f:
                with tqdm(total=total_size, initial=initial_pos, unit='B', unit_scale=True, desc=f"下载 {Path(destination).name}") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # 下载完成后重命名
            os.rename(temp_file, destination)
            return destination
                
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count > retries:
                print(f"下载失败: {e}")
                if os.path.exists(temp_file):
                    print(f"保留临时文件以便断点续传: {temp_file}")
                raise
            wait_time = backoff_factor * (2 ** (retry_count - 1))
            logging.warning(f"下载失败，{wait_time}秒后重试 ({retry_count}/{retries})")
            time.sleep(wait_time)

def download_with_mirrors(urls: List[str], destination: str, chunk_size: int = 8192) -> Path:
    """尝试从多个镜像下载文件，直到成功"""
    for url in urls:
        try:
            return download_file(url, destination, chunk_size)
        except Exception as e:
            logging.warning(f"从 {url} 下载失败: {e}")
    
    raise RuntimeError("所有下载源都失败")

def verify_file(file_path: Path, expected_hash: str = None, hash_type='md5') -> bool:
    """验证文件完整性"""
    if expected_hash:
        import hashlib
        hash_func = getattr(hashlib, hash_type)()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        actual_hash = hash_func.hexdigest()
        return actual_hash == expected_hash
    return os.path.getsize(file_path) > 0  # 基本验证