import sys
import torch
import platform

print(f"Python 版本: {platform.python_version()} ({sys.version_info})")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '不可用'}")
    print(f"当前设备: {torch.cuda.get_device_name(0)}")
    print(f"设备计算能力: {torch.cuda.get_device_capability(0)}")
    print(f"设备总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 打印其他相关库版本
try:
    import torchvision
    print(f"Torchvision 版本: {torchvision.__version__}")
except ImportError:
    print("Torchvision 未安装")

try:
    import numpy as np
    print(f"NumPy 版本: {np.__version__}")
except ImportError:
    print("NumPy 未安装")

try:
    import pandas as pd
    print(f"Pandas 版本: {pd.__version__}")
except ImportError:
    print("Pandas 未安装")