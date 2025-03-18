#src/utils/logging_utils.py

import logging
import sys
import os
from pathlib import Path
from typing import Union

def setup_logging(log_file=None):
    """设置日志记录
    
    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # 确保日志文件目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )