import json
import os
from pathlib import Path

# 加载类别映射
with open('data/processed/class_mapping.json', 'r') as f:
    class_mapping = json.load(f)

print(f"类别数量: {len(class_mapping)}")
print(f"类别列表: {list(class_mapping.keys())}")

# 检查图像目录
processed_dir = Path('data/processed')
images_dir = processed_dir / 'images'

if images_dir.exists():
    # 统计各类别的图像数量
    class_counts = {}
    for class_dir in images_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')))
            class_counts[class_dir.name] = count
    
    print("\n各类别图像数量:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}张图像")
    
    print(f"\n总图像数量: {sum(class_counts.values())}")
else:
    print(f"图像目录不存在: {images_dir}")