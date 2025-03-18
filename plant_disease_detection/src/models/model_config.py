# models/model_config.py

class ModelConfig:
    # 通用配置
    PRETRAINED = True
    FREEZE_BACKBONE = True
    
    # 分类器配置
    CLASSIFIER_CONFIG = {
        'backbone': 'resnet50',
        'num_classes': 10,
        'dropout_rate': 0.3,
        'learning_rate': 1e-4
    }
    
    # 检测器配置
    DETECTOR_CONFIG = {
        'backbone': 'yolov5s',
        'num_classes': 5,
        'confidence_threshold': 0.5,
        'nms_threshold': 0.45
    }
    
    # ===== 评估参数配置 =====
    # 检测评估IoU阈值
    DETECTION_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    # 检测结果可视化的分数阈值
    DETECTION_SCORE_THRESHOLD = 0.5
    
    # 是否生成混淆矩阵
    GENERATE_CONFUSION_MATRIX = True
    
    # 是否可视化检测结果
    VISUALIZE_DETECTION_RESULTS = True
    
    # GradCAM目标层
    GRADCAM_TARGET_LAYER = 'backbone.layer4'
    
    # 热图透明度
    HEATMAP_ALPHA = 0.6