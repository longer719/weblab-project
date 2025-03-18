# configs/data_config.py

import cv2

class DataConfig:
    """
    数据处理配置类
    
    包含图像处理、数据加载、数据增强和特定领域处理的所有配置项。
    每个配置项都有详细说明、默认值和合理范围。
    """
    
    # ===== 图像基础配置 =====
    # 图像大小 (宽, 高)，用于模型输入
    # 常用值: (224, 224), (256, 256), (299, 299), (384, 384)
    IMAGE_SIZE = (224, 224)
    
    # 批次大小，根据GPU内存和模型大小调整
    # 较大的值可以加速训练，但需要更多的GPU内存
    # 常用值: 8, 16, 32, 64, 128
    BATCH_SIZE = 32
    
    # ===== 数据增强配置 =====
    # 详细的数据增强策略，用于训练阶段
    AUGMENTATION = {
        # 水平翻转，适用于大多数植物病害图像
        # 设为True启用，False禁用
        'horizontal_flip': True,
        
        # 旋转角度范围（度），适当的旋转有助于模型泛化
        # 推荐范围: 10-30度
        'rotation_range': 15,
        
        # 亮度调整范围，模拟不同光照条件
        # [min, max]，其中1.0表示原始亮度
        # 推荐范围: [0.7, 1.3]
        'brightness_range': [0.8, 1.2],
        
        # 随机缩放范围，增加尺度不变性
        # 值表示相对于原始大小的变化比例
        # 推荐范围: 0.05-0.2
        'zoom_range': 0.1,
        
        # 随机裁剪参数，裁剪原始图像的一部分
        # 值表示相对于原始图像的大小比例
        # 推荐范围: 0.7-0.95
        'random_crop_factor': 0.9,
        
        # 颜色抖动参数
        'color_jitter': {
            'brightness': 0.1,  # 亮度变化范围
            'contrast': 0.1,    # 对比度变化范围
            'saturation': 0.1,  # 饱和度变化范围
            'hue': 0.05         # 色调变化范围
        }
    }
    
    # ===== 数据路径配置 =====
    # 原始数据目录，存放下载的数据集文件
    RAW_DATA_DIR = 'data/raw'
    
    # 处理后的数据目录，存放预处理后的图像和标签
    PROCESSED_DATA_DIR = 'data/processed'
    
    # 统计数据目录，存放数据分析结果
    STATS_DIR = 'data/stats'
    
    # ===== 数据集划分比例 =====
    # 训练集比例，通常占总数据的70%-80%
    TRAIN_RATIO = 0.7
    
    # 验证集比例，通常占总数据的10%-15%
    VAL_RATIO = 0.15
    
    # 测试集比例，通常占总数据的10%-15%
    TEST_RATIO = 0.15
    
    # ===== 特定领域处理配置 =====
    # 植物病害增强特定参数
    PLANT_DISEASE_PROCESSING = {
        # 叶脉增强，突出叶脉结构
        'enhance_leaf_veins': {
            'enabled': True,    # 是否启用
            'strength': 0.5,    # 增强强度 (0-1)
            'method': 'clahe'   # 增强方法: 'clahe', 'unsharp_mask', 'gabor'
        },
        
        # 病斑突出，强化病害区域
        'highlight_lesions': {
            'enabled': True,     # 是否启用
            'color_space': 'hsv',# 使用的颜色空间: 'hsv', 'lab', 'ycrcb'
            'sensitivity': 0.7   # 敏感度 (0-1)
        },
        
        # 背景抑制，减弱背景噪声
        'suppress_background': {
            'enabled': False,   # 是否启用 
            'threshold': 0.2    # 背景阈值 (0-1)
        }
    }
    
    # ===== 预处理配置 =====
    # 图像标准化方法
    # 'minmax': 归一化到[0,1]区间
    # 'zscore': 标准化到均值0、方差1
    # 'imagenet': 使用ImageNet预训练模型的均值和标准差
    NORMALIZATION_METHOD = 'imagenet'
    
    # 调整图像大小时使用的插值方法
    # cv2.INTER_NEAREST: 最近邻插值，速度快但质量低
    # cv2.INTER_LINEAR: 双线性插值，速度和质量的平衡
    # cv2.INTER_CUBIC: 双三次插值，质量高但速度慢
    # cv2.INTER_AREA: 区域插值，缩小图像时推荐使用
    RESIZE_INTERPOLATION = cv2.INTER_AREA
    
    # ImageNet预训练模型的均值和标准差
    # 用于'imagenet'标准化方法
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # ===== 类别映射 =====
    # 基础样例类别，用于测试和简单示例
    PLANT_CLASSES = [
        'healthy',    # 健康植物
        'disease_1',  # 病害类型1
        'disease_2'   # 病害类型2
    ]
    
    # PlantVillage数据集类别映射
    # 键: 原始类别名
    # 值: 类别ID (从0开始)
    PLANTVILLAGE_CLASS_MAPPING = {
        'Apple___Apple_scab': 0,
        'Apple___Black_rot': 1,
        'Apple___Cedar_apple_rust': 2,
        'Apple___healthy': 3,
        'Cherry___healthy': 4,
        'Cherry___Powdery_mildew': 5,
        'Corn___Cercospora_leaf_spot': 6,
        'Corn___Common_rust': 7,
        'Corn___healthy': 8,
        'Corn___Northern_Leaf_Blight': 9,
        'Grape___Black_rot': 10,
        'Grape___Esca': 11,
        'Grape___healthy': 12,
        'Grape___Leaf_blight': 13,
        'Peach___Bacterial_spot': 14,
        'Peach___healthy': 15,
        'Pepper,_bell___Bacterial_spot': 16,
        'Pepper,_bell___healthy': 17,
        'Potato___Early_blight': 18,
        'Potato___Late_blight': 19,
        'Potato___healthy': 20,
        'Squash___Powdery_mildew': 21,
        'Strawberry___Leaf_scorch': 22,
        'Strawberry___healthy': 23,
        'Tomato___Bacterial_spot': 24,
        'Tomato___Early_blight': 25,
        'Tomato___Late_blight': 26,
        'Tomato___Leaf_Mold': 27,
        'Tomato___Septoria_leaf_spot': 28,
        'Tomato___Spider_mites Two-spotted_spider_mite': 29,
        'Tomato___Target_Spot': 30,
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 31,
        'Tomato___Tomato_mosaic_virus': 32,
        'Tomato___healthy': 33
    }
    
    # ===== 评估数据配置 =====
    # 评估结果输出目录
    EVALUATION_OUTPUT_DIR = 'results/evaluation'
    
    # 可视化样本数量
    VISUALIZATION_SAMPLES = 5
    
    # 可视化图表尺寸
    FIGURE_SIZE_LARGE = (12, 10)
    FIGURE_SIZE_MEDIUM = (10, 8)
    FIGURE_SIZE_SMALL = (8, 6)

    # 更新默认过滤植物列表，包括国内常见种类
    DEFAULT_PLANT_FILTER = ["Apple", "Corn", "Grape", "Potato", "Tomato", "Strawberry", "Peach", "Pepper_bell", "Squash"]