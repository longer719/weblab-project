好的，我完全理解你的要求！在承认之前的回答可能部分过时（特别是关于检测器映射问题你已修复的部分），并明确核心目标是**在最小化代码改动的前提下，显著提升模型的泛化能力和鲁棒性**后，我现在为你提供一个在上一个回答基础上**补充和细化**的优化方案。

这个方案会更具体地说明在哪些文件、哪些函数中进行修改，并优先选择那些对泛化性/鲁棒性影响直接、且实现复杂度在可控范围内的措施。

**再次确认核心问题：** 模型在训练集表现好，但在光照、角度、背景、噪声等变化的**未见过的数据**上表现差。

**优化方案（最小化修改，聚焦泛化性/鲁棒性）：**

**一、 数据增强强化 (最直接提升鲁棒性的方法)**

*   **方案 1.1: 强化颜色与光照增强 (低复杂度)**
    *   **目标:** 提高模型对不同光照强度、色温和对比度的适应能力。
    *   **影响文件:** `src/utils/augmentation.py` (修改 `AugmentationPipeline` 的 `get_train_transforms` 方法)
    *   **实现细节:**
        *   在现有的 `transforms.ColorJitter` 中，**加大** `brightness`, `contrast`, `saturation` 的范围。例如，从 `0.2` 增加到 `0.3` 或 `0.4`。
        *   **增加** `transforms.RandomAutocontrast(p=0.2)` 和 `transforms.RandomEqualize(p=0.2)`，随机应用自动对比度和直方图均衡化。
        *   **增加** `transforms.RandomGamma(scale=(0.8, 1.2), p=0.3)` 来模拟更广泛的光照变化。
        *   可以考虑添加 `transforms.functional.adjust_sharpness` 的随机版本，模拟不同清晰度。
    *   **兼容性:** 只修改数据增强管道，不影响数据加载、模型结构或训练循环主体。

*   **方案 1.2: 增加几何变换与噪声模拟 (低复杂度)**
    *   **目标:** 提高模型对轻微旋转、缩放、平移、遮挡以及传感器噪声的鲁棒性。
    *   **影响文件:** `src/utils/augmentation.py` (修改 `AugmentationPipeline` 的 `get_train_transforms` 方法)
    *   **实现细节:**
        *   **确认/增加 `transforms.RandomAffine`:** 检查是否已使用。如果未使用，添加 `transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, p=0.5)`。如果已使用，可以适当**微调**参数范围。
        *   **增加 `transforms.GaussianBlur`:** 添加 `transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0), p=0.3)` 模拟轻微模糊。
        *   **确认/增加 `transforms.RandomErasing`:** 如果之前没加，添加 `transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)` 模拟遮挡。如果已加，可调整参数。
        *   **增加简单高斯噪声:** 在 `ToTensor` 之后、`Normalize` 之前，添加一个自定义 `lambda` 转换或 `nn.Module` 来给 Tensor 加上标准差较小的高斯噪声，例如 `lambda x: x + torch.randn_like(x) * 0.02`。
    *   **兼容性:** 同上，只修改数据增强管道。

*   **方案 1.3: 引入 MixUp 或 CutMix (中等复杂度，效果显著)**
    *   **目标:** 通过混合样本及其标签，强制模型学习更平滑的决策边界，提升泛化能力。
    *   **影响文件:**
        *   `src/utils/augmentation.py`: 添加 `mixup_data` 和 `cutmix_data` 的静态方法实现（如果尚未存在）。
        *   `src/training/trainer.py`: 修改 `_train_classification_batch` (或检测任务的对应部分)。
        *   `src/training/train_config.py` 或 `yaml` 文件: 添加启用和配置 MixUp/CutMix 的参数（如 `mixup_alpha`, `cutmix_alpha`）。
    *   **实现细节 (`Trainer._train_classification_batch`):**
        ```python
        # 在获取 images, labels 之后
        use_mixup = self.config.get('mixup_alpha', 0.0) > 0
        use_cutmix = self.config.get('cutmix_alpha', 0.0) > 0
        use_mixcut = use_mixup or use_cutmix
        mix_decision = np.random.rand() if use_mixcut else -1

        if use_mixup and mix_decision < 0.5: # 假设50%概率用MixUp
            images, labels_a, labels_b, lam = AugmentationPipeline.mixup_data(images, labels, self.config['mixup_alpha'])
        elif use_cutmix: # 剩下50%概率用CutMix (如果启用)
             images, labels_a, labels_b, lam = AugmentationPipeline.cutmix_data(images, labels, self.config['cutmix_alpha'])
        else: # 不使用 Mixup/Cutmix
             labels_a, labels_b, lam = labels, labels, 1.0 # 保持原始标签

        # ... (模型前向传播)
        outputs = self.model(images)

        # 修改损失计算
        if use_mixcut and mix_decision >= 0: # 如果应用了 MixUp 或 CutMix
            loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
        else: # 否则使用标准损失
            loss = self.criterion(outputs, labels)

        # ... (反向传播和优化器步骤)

        # 注意: 准确率计算可能需要调整或在验证集上评估
        # _, preds = torch.max(outputs, 1)
        # acc = (lam * (preds == labels_a).float().mean() + (1 - lam) * (preds == labels_b).float().mean()).item() if use_mixcut else (preds == labels).float().mean().item()
        # 简单起见，训练时可以只记录loss，在验证时计算标准准确率
        acc = 0.0 # 或者在验证时再计算
        ```
    *   **兼容性:** 主要修改训练循环逻辑和损失计算，模型和数据加载不变。API 接口不受影响。

**二、 模型与训练策略微调 (提升泛化，轻量修改)**

*   **方案 2.1: 调整 Dropout 率 (低复杂度)**
    *   **目标:** 通过增加 Dropout 来提供更强的正则化，减少过拟合。
    *   **影响文件:** `src/models/plant_classifier.py` (初始化时) 或对应的模型配置文件 (`models/plant_classifier_config.json`, 如果 `PlantClassifier` 支持 `from_config_file`)。也可能在训练配置文件 `yaml` 或 `src/training/train_config.py` 中。
    *   **实现细节:** 找到设置 `dropout_rate` 的地方，将其值适当增加，例如从 `0.3` 增加到 `0.4` 或 `0.5`。需要实验找到最佳值。
    *   **兼容性:** 只修改模型定义中的一个参数值，完全兼容。

*   **方案 2.2: 确认并优化渐进式微调 (中等复杂度)**
    *   **目标:** 更好地利用预训练模型的知识，防止在微调早期破坏底层特征。
    *   **影响文件:** `scripts/train_classifier.py`, `scripts/train_detector.py` (或 `src/training/trainer.py` 如果训练循环在那里)。
    *   **实现细节:**
        *   **检查现有实现:** 确认你的 `Trainer` 或训练脚本是否已经实现了类似方案 5 的逻辑。
        *   **如果未实现:** 在训练脚本中实现分阶段训练：
            1.  **阶段 1 (Head Training):**
                *   冻结 `model.backbone` 的所有参数 (`param.requires_grad = False`)。
                *   创建优化器，只包含 `model.classifier_head` (或检测模型的 `roi_heads`) 的参数。
                *   训练少量 epochs (例如 5-10)。
            2.  **阶段 2 (Fine-tuning):**
                *   解冻部分或全部 `backbone` 参数 (`param.requires_grad = True`)。
                *   **关键：** 创建一个新的优化器，包含所有需要训练的参数，并为 `backbone` 参数设置一个**更小**的学习率 (例如，`1e-5` 或 `1e-6`)，而 `head` 参数使用稍大的学习率 (例如，`1e-4`)。这可以通过给 `optimizer` 构造函数传递一个参数组列表实现。
                *   继续训练更多 epochs。
        *   **学习率设置示例 (PyTorch):**
            ```python
            # 假设 model.backbone 和 model.head
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': 1e-5}, # 较小学习率
                {'params': model.head.parameters(), 'lr': 1e-4}     # 较大学习率
            ], weight_decay=0.01)
            ```
    *   **兼容性:** 主要修改训练脚本的设置和循环逻辑，不影响模型结构或 API。

*   **方案 2.3: 尝试不同的优化器或调整参数 (低复杂度)**
    *   **目标:** 找到更适合当前任务和数据的优化算法。
    *   **影响文件:** `scripts/train_classifier.py`, `scripts/train_detector.py`, 以及对应的 `yaml` 配置文件或 `src/training/train_config.py`。
    *   **实现细节:**
        *   在训练脚本中修改优化器的实例化代码。例如，将 `optim.Adam(...)` 改为 `optim.AdamW(...)` 或 `optim.SGD(...)`。
        *   参考 `src/training/train_config.py` 中 `OptimizerConfig` 的参数，调整 `weight_decay`, `momentum` (for SGD), `betas` (for Adam/AdamW)。特别是 `AdamW` 配合合适的 `weight_decay` 通常比 Adam 泛化性更好。
        *   在 `yaml` 文件或 `TrainingConfig` 中更新优化器类型和参数。
    *   **兼容性:** 只修改优化器实例化和配置，完全兼容。

*   **方案 2.4: 调整学习率调度器 (低复杂度)**
    *   **目标:** 使用更现代或更适合的 LR 策略。
    *   **影响文件:** `scripts/train_classifier.py`, `scripts/train_detector.py`, 以及对应的 `yaml` 配置文件或 `src/training/train_config.py`。
    *   **实现细节:**
        *   如果当前使用的是简单的 `StepLR`，尝试改为 `CosineAnnealingLR` 或 `OneCycleLR`（如方案 9）。
        *   调整现有调度器的参数，例如 `CosineAnnealingLR` 的 `T_max` (通常设为总 epochs) 和 `eta_min`，或者 `StepLR` 的 `step_size` 和 `gamma`。
        *   确保在 `Trainer` 中 `scheduler.step()` 的调用位置正确（epoch 级别或 batch 级别，取决于调度器类型）。
    *   **兼容性:** 主要修改调度器实例化和配置，`Trainer` 可能需要微调 `step()` 调用位置。

**三、 系统级优化 (低复杂度)**

*   **方案 3.1: 彻底统一和简化映射管理 (中等复杂度 - 但非常重要)**
    *   **目标:** 消除因映射混乱导致的潜在错误和维护困难，确保系统行为一致。
    *   **影响文件:** `scripts/prepare_data.py`, `run.py`, `src/utils/mapping_service.py`, `scripts/deploy_models.py`。删除 `scripts/create_full_mappings.py`, `models/plant_classes.json`, `models/disease_classes.json`。
    *   **实现细节:**
        1.  **修改 `prepare_data.py`:**
            *   在 `process_plantvillage_dataset` 或类似函数结束时，除了保存 `class_mapping.json` (英文类名 -> ID)，**额外保存**一个中文映射文件 `data/processed/class_mapping_zh.json`。
            *   这个中文映射文件的格式应为 `{"0": "苹果-黑星病", "1": "苹果-黑腐病", ...}`，直接对应 0-37 的索引。你可以通过结合 `class_mapping.json` 和你现有的英->中映射（如 `plant_mappings.json`, `disease_mappings.json`）来生成。
        2.  **修改 `run.py`:**
            *   在 `load_models()` 或应用初始化时，**只加载** `data/processed/class_mapping_zh.json` (或其部署后的副本 `models/class_mapping_zh.json`)。
            *   将加载的内容同时赋值给 `app.plant_class_names` 和 `app.disease_class_names`。
            *   **删除**加载 `models/plant_classes.json` 和 `models/disease_classes.json` 的代码。
        3.  **简化 `MappingService`:**
            *   修改 `MappingService._load_mappings`，不再加载 `plant_classes.json` 和 `disease_classes.json`。
            *   修改或移除 `get_plant_name`, `get_disease_name` 方法，因为 ID 到中文名的映射现在由 `run.py` 加载到 `app` 上下文中，并可能直接传递给 `predictor` 或在 `routes.py` 中使用。
            *   `map_dataset_class_to_db` 仍可用于处理输入（如从 "Apple___healthy" 到 "苹果", "健康"），但不再需要处理 ID。
        4.  **删除冗余:** 删除 `scripts/create_full_mappings.py` 脚本和 `models/` 下的 `plant_classes.json`, `disease_classes.json` 文件。
        5.  **检查/修改 `deploy_models.py`:** 确保部署脚本将训练时生成的 `data/processed/class_mapping_zh.json` 文件复制到 `models/` 目录（如果 `run.py` 从 `models/` 加载），或者修改 `run.py` 直接从 `data/processed/` 加载该文件。
    *   **兼容性:** 需要修改多个文件，但目标是简化和统一，长远看会提高可维护性。API 接口的输入输出格式保持不变。

*   **方案 3.2: 调整检测置信度阈值 (低复杂度)**
    *   **目标:** 平衡检测的召回率和精确率，减少低质量的检测结果。
    *   **影响文件:**
        *   可能在 `src/models/disease_detector.py` 初始化时设置 `box_score_thresh`。
        *   也可能在 `src/inference/predictor.py` 的 `DetectionPredictor` 初始化或 `predict` 方法中应用。
        *   或者在 `src/models/components/detection_utils.py` 的 `PlantDiseasePostProcessor` 或 `NMSUtils` 中。
        *   前端 `visualization.js` 中 `PlantVis.renderDetectionBoxes` 等函数也有 `scoreThreshold` 参数，用于显示。
    *   **实现细节:** 找到应用分数阈值的地方，根据验证集上的 mAP 或 PR 曲线，将阈值从当前的 `0.3` (根据 `first_version.md`) 或 `0.05` (根据 `detector_config.json` 或代码默认值) **提高**到一个更合适的值，例如 `0.5` 或 `0.6`。确保后端过滤和前端显示使用一致或协调的阈值。
    *   **兼容性:** 只修改参数值，完全兼容。

**实施建议:**

1.  **必须执行:** **方案 3.1 (统一映射)** 是基础，务必完成。
2.  **高优先级 (提升泛化/鲁棒性):**
    *   **方案 1.1 & 1.2 (强化增强):** 改动小，效果直接。
    *   **方案 2.2 (渐进式微调):** 效果显著，改动中等。
    *   **方案 2.1 (调整 Dropout):** 改动极小。
    *   **方案 1.3 (MixUp/CutMix):** 效果好，改动中等。
3.  **中优先级 (进一步优化):**
    *   **方案 2.3 & 2.4 (优化器/调度器):** 尝试 AdamW 和 Cosine/OneCycle。
    *   **方案 1.4 (训练/推理一致性):** 仔细检查，改动小。
    *   **方案 3.2 (调整阈值):** 简单有效。
4.  **低优先级 (锦上添花):**
    *   **方案 2.4 (Label Smoothing):** 改动小。
    *   **方案 2.8 (NMS 参数):** 简单尝试。
    *   **方案 11 (Focal Loss):** 仅当类别不平衡严重时。
    *   **方案 10 & 12 (梯度累积/裁剪/AMP):** 主要影响训练速度和稳定性。
    *   **方案 15 (前端可视化):** 根据实际效果决定。

通过实施这些经过筛选和细化的方案，你可以在**最小化代码修改**的前提下，有效提升模型的泛化能力和鲁棒性，并解决关键的系统性问题，使你的毕业设计更加完善。