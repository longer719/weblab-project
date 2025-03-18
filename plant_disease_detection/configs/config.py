# configs/config.py

class Config:
    # 基础配置
    PROJECT_NAME = "Plant Disease Detection"
    VERSION = "1.0.0"
    
    # 路径配置
    DATA_DIR = "data"
    MODEL_DIR = "models"
    
    # 模型配置
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = {
        "plants": 10,  # 植物种类数量
        "diseases": 5  # 病害类型数量
    }
    
    # API配置
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    DEBUG = True