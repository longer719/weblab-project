#!/usr/bin/env python
"""
修复数据集CSV文件中的路径格式，使其在各操作系统上兼容。
将Windows风格的反斜杠(\)替换为正斜杠(/)。
"""

import os
import pandas as pd
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_data_paths(data_dir='data/processed'):
    """
    修复CSV文件中的路径分隔符
    
    Args:
        data_dir: 数据目录，默认为'data/processed'
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.error(f"目录不存在: {data_dir}")
        return
    
    # 处理所有CSV文件
    for csv_file in ['train_labels.csv', 'val_labels.csv', 'test_labels.csv']:
        file_path = data_dir / csv_file
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            continue
        
        logger.info(f"处理文件: {file_path}")
        
        # 读取CSV文件
        try:
            df = pd.read_csv(file_path)
            original_count = len(df)
            
            # 修复image_path列中的路径分隔符
            if 'image_path' in df.columns:
                # 备份原始文件
                backup_path = file_path.with_suffix('.csv.bak')
                if not backup_path.exists():
                    df.to_csv(backup_path, index=False)
                    logger.info(f"已备份原始文件到: {backup_path}")
                
                # 替换Windows风格的反斜杠为正斜杠
                df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)
                
                # 保存修改后的文件
                df.to_csv(file_path, index=False)
                logger.info(f"文件已更新: {file_path}, 处理了{original_count}条记录")
            else:
                logger.warning(f"文件{file_path}中没有发现'image_path'列")
                
        except Exception as e:
            logger.error(f"处理文件{file_path}时出错: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='修复数据集CSV文件中的路径格式')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='数据目录路径')
    
    args = parser.parse_args()
    fix_data_paths(args.data_dir)
    logger.info("路径修复完成！")

if __name__ == "__main__":
    main()