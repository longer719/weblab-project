# scripts/debug_detector.py
import torch
from PIL import Image
import json
import sys
import os
from pathlib import Path

# 添加项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.models import DiseaseDetector
# !!! 注意：确保这里的预处理与 run.py/predictor.py 一致 !!!
# 如果 predictor.py 使用的是 get_transform，这里也应该用它
from src.data_processing.transforms import get_transform
# from src.utils.image_utils import preprocess_image # 或者用这个，取决于推理流程

# --- 配置 ---
MODEL_PATH = "models/disease_detector.pth"
CONFIG_PATH = "models/disease_detector_config.json"
# --- *** 修改这里为你图片的绝对路径 *** ---
# 使用 raw string (r"...") 或者将反斜杠替换为正斜杠或双反斜杠
# 选项 1: Raw String (推荐)
IMAGE_PATH = r"C:\Users\lenovo\Desktop\毕设\毕设测试\test_Cherry_白粉病_标准.jpg"
# 选项 2: 正斜杠
# IMAGE_PATH = "C:/Users/lenovo/Desktop/毕设/毕设测试/test_Cherry_白粉病_标准.jpg"
# 选项 3: 双反斜杠
# IMAGE_PATH = "C:\\Users\\lenovo\\Desktop\\毕设\\毕设测试\\test_Cherry_白粉病_标准.jpg"
# --- *** 结束修改 *** ---
MAPPING_PATH = "models/plant_classes.json"
# --- 结束配置 ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    config = {"num_classes": 38, "backbone": "resnet50"} # 默认
    if os.path.exists(CONFIG_PATH):
        try: # 增加 try-except
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config.update(json.load(f))
            print(f"加载模型配置: {CONFIG_PATH}")
        except Exception as e:
             print(f"加载配置文件 {CONFIG_PATH} 失败: {e}, 使用部分默认配置。")
    else:
        print(f"警告: 找不到配置文件 {CONFIG_PATH}, 使用默认配置。")

    # 加载类别映射 (用于后续解释)
    id_to_name_map = {}
    if os.path.exists(MAPPING_PATH):
         try: # 增加 try-except
              with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
                   id_to_name_map = json.load(f)
         except Exception as e:
              print(f"加载映射文件 {MAPPING_PATH} 失败: {e}")
    else:
         print(f"警告: 找不到映射文件 {MAPPING_PATH}")

    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = DiseaseDetector(config=config)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        # 处理可能的 'module.' 前缀
        if isinstance(state_dict, dict) and all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 加载并预处理图像
    print(f"加载并预处理图像: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
         print(f"错误：图像文件不存在！路径：{IMAGE_PATH}")
         return
    try:
        image = Image.open(IMAGE_PATH).convert('RGB')
        # --- *** 使用与推理完全一致的预处理 *** ---
        
        # 方法1: 使用系统中现有的路由中的预处理函数
        from src.api.routes import get_transform as api_transform
        transform = api_transform(train=False)  # 使用API中的transform函数
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 或者方法2: 直接使用torchvision的transforms
        # from torchvision import transforms
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # image_tensor = transform(image).unsqueeze(0).to(device)
        
        # --- 结束预处理 ---
        print(f"图像预处理完成，Tensor shape: {image_tensor.shape}")
    except Exception as e:
        print(f"图像处理失败: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息
        return

    # 模型推理
    print("开始模型推理...")
    with torch.no_grad():
        # 确保模型是在 eval 模式
        model.eval()
        # torchvision 检测模型在 eval 模式下直接接收图像列表/张量
        outputs = model(image_tensor)

    # --- *** 打印原始输出 *** ---
    print("\n--- 模型原始输出 ---")
    if not outputs or not isinstance(outputs, list) or len(outputs) == 0:
        print("模型没有返回有效的输出列表！")
    else:
        output_dict = outputs[0] # 获取第一个图像的输出字典
        print(f"输出字典包含键: {list(output_dict.keys())}") # 使用 list() 方便查看
        if 'scores' in output_dict:
            scores_tensor = output_dict['scores']
            print(f"预测分数 (scores): {scores_tensor}")
            if len(scores_tensor) > 0:
                 max_score, max_idx = torch.max(scores_tensor, dim=0)
                 print(f"最高分数: {max_score.item():.4f} (在索引 {max_idx.item()})")
            else:
                 print("模型输出了0个预测框!")
        else:
             print("模型输出中缺少 'scores' 键。")

        if 'labels' in output_dict:
            labels_tensor = output_dict['labels']
            print(f"预测标签 (labels): {labels_tensor}")
            if len(labels_tensor) > 0 and 'scores' in output_dict and len(output_dict['scores']) > 0:
                 # 使用与最高分对应的索引
                 max_idx = torch.argmax(output_dict['scores'])
                 top_label = labels_tensor[max_idx].item()
                 print(f"最高分对应的标签: {top_label}")
                 # 尝试解释标签
                 class_id = top_label - 1 # 减 1 得到映射 ID
                 class_name = id_to_name_map.get(str(class_id), f"未知_{class_id}")
                 print(f"对应的类别 ID (减1后): {class_id}, 名称: {class_name}")
            elif len(labels_tensor) == 0:
                 print("模型输出的标签列表为空!")
            else:
                 print("无法确定最高分标签，因为分数列表为空。")
        else:
             print("模型输出中缺少 'labels' 键。")

        if 'boxes' in output_dict:
            boxes_tensor = output_dict['boxes']
            print(f"预测边界框 (boxes) 数量: {len(boxes_tensor)}")
            if len(boxes_tensor) > 0 and 'scores' in output_dict and len(output_dict['scores']) > 0:
                 # 使用与最高分对应的索引
                 max_idx = torch.argmax(output_dict['scores'])
                 print(f"最高分对应的框坐标: {boxes_tensor[max_idx].cpu().numpy()}")
            elif len(boxes_tensor) == 0:
                 print("模型输出的边界框列表为空!")
            else:
                 print("无法显示最高分对应的框，因为分数列表为空。")
        else:
            print("模型输出中缺少 'boxes' 键。")

    print("--- 结束原始输出 ---")

if __name__ == "__main__":
    main()