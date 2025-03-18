#!/bin/bash

# 设置CUDA相关环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置基本路径
BASE_DIR=$(pwd)
DATA_DIR=$BASE_DIR/data
LOG_DIR=$BASE_DIR/logs
EXPERIMENT_DIR=$BASE_DIR/experiments

# 创建必要的目录
mkdir -p $DATA_DIR/processed $LOG_DIR $EXPERIMENT_DIR

echo "===== 准备数据集 ====="
python scripts/prepare_data.py --clean --all

# 训练植物分类器
echo "===== 训练植物分类器 ====="
python scripts/train_classifier.py \
    --config configs/classifier_training.yaml \
    --output_dir $EXPERIMENT_DIR \
    --device cuda

# 训练病害检测器
echo "===== 训练病害检测器 ====="
python scripts/train_detector.py \
    --config configs/detector_training.yaml \
    --output_dir $EXPERIMENT_DIR \
    --device cuda

# 评估两个模型
echo "===== 评估模型 ====="
# 找到最新训练的模型目录
CLASSIFIER_DIR=$(ls -td $EXPERIMENT_DIR/plant_classifier* | head -1)
DETECTOR_DIR=$(ls -td $EXPERIMENT_DIR/disease_detector* | head -1)

CLASSIFIER_MODEL=$(find $CLASSIFIER_DIR -name "*.pth" | head -1)
DETECTOR_MODEL=$(find $DETECTOR_DIR -name "*.pth" | head -1)

if [ -n "$CLASSIFIER_MODEL" ]; then
  python scripts/evaluate.py \
    --model_path $CLASSIFIER_MODEL \
    --data_dir $DATA_DIR/processed \
    --task_type classification \
    --output_dir $EXPERIMENT_DIR/evaluation \
    --visualize
fi

if [ -n "$DETECTOR_MODEL" ]; then
  python scripts/evaluate.py \
    --model_path $DETECTOR_MODEL \
    --data_dir $DATA_DIR/processed \
    --task_type detection \
    --output_dir $EXPERIMENT_DIR/evaluation \
    --visualize
fi

echo "===== 部署模型 ====="
python scripts/deploy_models.py \
    --classifier-path $CLASSIFIER_DIR \
    --detector-path $DETECTOR_DIR \
    --target-dir $BASE_DIR/models

echo "===== 训练完成 ====="