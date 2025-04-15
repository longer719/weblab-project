# 基于深度学习的植物种类识别与病害检测系统

## 项目概述

本项目是一个基于深度学习技术的植物种类识别与病害检测系统，旨在帮助农业工作者、园艺爱好者和研究人员快速识别植物种类并检测可能存在的病害。系统集成了先进的深度学习模型和直观的Web界面，能够对14种常见植物及其相关的38种病害组合进行识别和分类。

### 主要功能

- **植物种类识别**：上传植物图像，系统能够识别植物种类
- **病害检测与分类**：检测植物上可能存在的病害并进行分类
- **结果可视化**：通过热图和边界框直观展示识别/检测结果
- **Grad-CAM解释**：提供模型关注区域可视化，增强可解释性
- **专业治疗建议**：基于检测结果提供详细的病害治疗和预防建议

### 支持的植物种类

系统基于PlantVillage数据集训练，目前支持14种常见农作物：
苹果、蓝莓、樱桃、玉米、葡萄、橙子、桃子、甜椒、土豆、树莓、大豆、西葫芦、草莓、番茄

### 支持的病害类型

系统可以识别38种植物-病害组合，包括：
黑星病、黑腐病、雪松苹果锈病、白粉病、灰斑病、普通锈病、北方叶枯病、黑麻疹病、叶枯病、黄龙病、细菌性斑点病、早疫病、晚疫病、叶霉病、斑枯病、二斑叶螨、靶斑病、花叶病毒病、黄化曲叶病毒病等。

## 技术架构

### 系统架构

```
plant_disease_detection/
├─ configs/             # 配置文件目录
├─ data/                # 数据目录
│  ├─ processed/        # 处理后的数据
│  └─ raw/              # 原始数据
├─ experiments/         # 模型训练实验结果
├─ logs/                # 日志文件
├─ models/              # 部署模型和映射文件
├─ notebooks/           # Jupyter笔记本
├─ scripts/             # 脚本文件(训练、处理等)
├─ src/                 # 源代码
│  ├─ api/              # API路由和处理
│  ├─ data_processing/  # 数据处理
│  ├─ evaluation/       # 评估工具 
│  ├─ inference/        # 推理相关
│  ├─ models/           # 模型定义
│  ├─ training/         # 训练相关
│  └─ utils/            # 工具函数
├─ static/              # 前端静态文件
├─ templates/           # HTML模板
├─ tests/               # 测试代码
└─ run.py               # 入口文件
```

### 技术栈

- **后端**：Python + Flask
- **深度学习框架**：PyTorch
- **前端**：原生JavaScript + HTML/CSS
- **数据集**：PlantVillage数据集
- **模型**：
  - 植物分类器：基于ResNet101
  - 病害检测器：基于Faster R-CNN架构(ResNet50+FPN骨干网络)，优化为图像级分类任务

## 核心模块

### 1. 数据处理模块

数据处理模块负责下载、处理和加载PlantVillage数据集，包括：

- **数据集下载与解压**：`scripts/prepare_data.py`提供数据集获取功能
- **统一类别映射**：在`data/processed/class_mapping_zh.json`中保存38类统一映射
- **数据预处理**：规范化图像尺寸和格式，生成训练/验证/测试划分
- **高级数据增强**：通过`src/utils/augmentation.py`实现，包括随机翻转、旋转、颜色抖动等

主要文件：
- prepare_data.py：数据准备主脚本
- dataset.py：定义了`ClassificationDataset`和`DetectionDataset`类
- `src/utils/augmentation.py`：提供数据增强功能
- `src/data_processing/transforms.py`：图像变换工具

### 2. 模型模块

模型模块包含两个核心组件：植物分类器和病害检测器。

#### 植物分类器 (PlantClassifier)

基于ResNet101骨干网络，通过迁移学习方式进行微调：

- **灵活的骨干网络**：支持ResNet、EfficientNet等多种骨干网络
- **可配置分类头**：支持多层MLP、Dropout等
- **多标签支持**：可用于单标签或多标签分类任务
- **渐进式微调**：实现了层冻结和差分学习率

主要文件：`src/models/plant_classifier.py`，配置文件：`models/plant_classifier_config.json`

#### 病害检测器 (DiseaseDetector)

基于Faster R-CNN架构，使用ResNet50+FPN作为骨干网络：

- **灵活配置**：支持自定义RPN、锚框设置等
- **创新适配**：优化为图像级分类任务，通过全图边界框方式进行训练
- **统一接口**：与分类器保持一致的预测接口
- **可视化支持**：集成热图生成功能

主要文件：`src/models/disease_detector.py`，配置文件：`models/disease_detector_config.json`

### 3. 训练模块

训练模块实现了模型的训练流程和策略：

- **通用Trainer类**：支持分类和检测任务
- **高级训练策略**：
  - 渐进式微调 (Progressive Unfreezing)
  - 学习率调度 (CosineAnnealingLR)
  - 标签平滑 (Label Smoothing)
  - 梯度累积 (Gradient Accumulation)
  - 混合精度训练 (Mixed Precision)
- **状态监控**：支持TensorBoard可视化、模型检查点保存
- **早停机制**：防止过拟合，自动选择最佳模型

主要文件：
- `src/training/trainer.py`：训练器核心实现
- train_classifier.py：分类器训练脚本
- train_detector.py：检测器训练脚本

### 4. 推理模块

推理模块提供了模型部署和预测功能：

- **统一预测接口**：通过`ClassificationPredictor`和`DetectionPredictor`类
- **批量预测**：支持批量图像处理
- **模型解释**：集成Grad-CAM可视化解释功能
- **配置管理**：灵活的配置加载和管理

主要文件：
- predictor.py：预测器类实现
- model_interpreter.py：模型解释器实现
- deploy_models.py：模型部署工具

### 5. API模块

API模块提供了Web服务接口：

- **植物识别**：`/api/classify` 端点
- **病害检测**：`/api/detect` 端点
- **模型解释**：`/api/explain_detection` 端点
- **治疗建议**：`/api/treatment` 端点
- **植物信息**：`/api/plant_info` 端点

主要文件：`src/api/routes.py`

### 6. 前端界面模块

前端界面提供了用户友好的交互体验：

- **图像上传**：支持拖放和文件选择
- **任务切换**：在植物识别和病害检测模式间切换
- **结果可视化**：显示分类结果和检测结果，包括边界框和热图
- **交互控制**：可调节置信度阈值，切换可视化模式
- **治疗建议**：显示病害治疗和预防方法

主要文件：
- index.html：HTML模板
- main.js：主交互逻辑
- visualization.js：可视化组件
- style.css：样式表

### 7. 病害治疗数据库

详细的病害治疗知识库，为每种病害提供：

- 症状描述
- 病因分析
- 有机/化学治疗方案
- 预防措施
- 严重程度评估

主要文件：`src/utils/disease_treatments.py`

## 使用指南

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

```bash
python scripts/prepare_data.py
```

### 模型训练

```bash
# 训练植物分类器
python scripts/train_classifier.py --config configs/classifier_training.yaml

# 训练病害检测器(分类模式)
python scripts/train_detector.py --config configs/detector_training.yaml
```

### 模型评估

```bash
python scripts/evaluate.py --model_path models/plant_classifier.pth --data_dir data/processed --task_type classification

python scripts/evaluate.py --model_path models/disease_detector.pth --data_dir data/processed --task_type detection
```

### 模型部署

```bash
python scripts/deploy_models.py --classifier-path experiments/plant_classifier_latest --detector-path experiments/disease_detector_latest --target-dir models
```

### 启动服务

```bash
python run.py --mode serve --port 5000
```

### 单图预测

```bash
python run.py --mode predict --task classification --image path/to/image.jpg
```

## 性能指标

- **植物分类器**：
  - 准确率(Accuracy): 94.8%
  - F1分数: 0.943
  
- **病害检测器(分类模式)**：
  - 准确率(Accuracy): 92.3%
  - F1分数: 0.918

## 系统局限性

- 仅限于PlantVillage数据集中的14种植物和38种病害组合
- 对光照条件、背景和图像质量敏感
- 无法精确定位叶片上的病变区域（仅提供图像级分类）

## 未来工作

- 扩展数据集，增加更多植物和病害类型
- 实现真正的病害区域分割和定位
- 开发移动端应用，支持现场拍照识别
- 提供API接口，支持与其他农业系统集成

## 贡献者

- 系统设计与实现: [loong]
- 数据源: [PlantVillage数据集]

## 许可证

本项目采用MIT许可证

## 参考资料

- [PlantVillage数据集](https://github.com/spMohanty/PlantVillage-Dataset)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [Flask官方文档](https://flask.palletsprojects.com/)
- [ResNet论文](https://arxiv.org/abs/1512.03385)
- [Faster R-CNN论文](https://arxiv.org/abs/1506.01497)

## 联系方式

如有问题或建议，请联系: [is_loong@qq.com]