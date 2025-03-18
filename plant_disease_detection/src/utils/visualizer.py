# src/utils/visualizer.py
#这是一个可视化工具类，用于数据和结果可视化。它提供了多种可视化功能，用于数据分析、模型解释和结果展示。

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from pathlib import Path
import cv2
from typing import List, Dict, Optional, Tuple, Union, Any
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import os

class Visualizer:
    """
    数据和结果可视化工具
    
    提供多种可视化功能，用于数据分析、模型解释和结果展示
    """
    
    @staticmethod
    def show_image(img, title=None, figsize=(8, 8), save_path=None):
        """显示单张图像"""
        plt.figure(figsize=figsize)
        
        # 如果是PyTorch张量，转换为numpy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().transpose(1, 2, 0)
            
            # 如果是标准化过的图像，反标准化
            if img.max() <= 1.0 and img.min() < 0:
                # 假设使用ImageNet均值和标准差
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def show_augmentations(image, augmentation_func, n_examples=5, figsize=(15, 5), save_path=None):
        """
        显示数据增强效果
        
        Args:
            image: 原始图像
            augmentation_func: 数据增强函数
            n_examples: 展示的样本数量
            figsize: 图像大小
            save_path: 保存路径
        """
        plt.figure(figsize=figsize)
        
        # 显示原始图像
        plt.subplot(1, n_examples + 1, 1)
        plt.imshow(image)
        plt.title('Original')
        plt.axis('off')
        
        # 显示增强后的图像
        for i in range(n_examples):
            plt.subplot(1, n_examples + 1, i + 2)
            augmented = augmentation_func(image)
            plt.imshow(augmented)
            plt.title(f'Aug #{i+1}')
            plt.axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_class_distribution(data_dir, figsize=(12, 6), save_path=None):
        """绘制类别分布"""
        data_path = Path(data_dir)
        
        # 加载数据标签
        train_df = pd.read_csv(data_path / 'train_labels.csv')
        val_df = pd.read_csv(data_path / 'val_labels.csv')
        test_df = pd.read_csv(data_path / 'test_labels.csv')
        
        # 加载类别映射
        import json
        with open(data_path / 'class_mapping.json', 'r') as f:
            class_mapping = json.load(f)
        
        # 反转映射，用于查找类名
        idx_to_class = {v: k for k, v in class_mapping.items()}
        
        # 准备数据
        train_counts = train_df['label'].value_counts().sort_index()
        val_counts = val_df['label'].value_counts().sort_index()
        test_counts = test_df['label'].value_counts().sort_index()
        
        # 创建DataFrame
        counts_df = pd.DataFrame({
            'Train': train_counts,
            'Validation': val_counts,
            'Test': test_counts
        })
        
        # 替换索引为类名
        counts_df.index = [idx_to_class.get(i, f"Class {i}") for i in counts_df.index]
        
        plt.figure(figsize=figsize)
        ax = counts_df.plot(kind='bar', figsize=figsize)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Dataset')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        
        return counts_df
    
    @staticmethod
    def visualize_detection(image, boxes, labels=None, scores=None, 
                            class_names=None, figsize=(12, 12), save_path=None):
        """可视化目标检测结果"""
        img = image.copy()
        
        # 颜色映射
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (0, 255, 255),  # 黄色
            (255, 0, 255),  # 紫色
        ]
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # 选择颜色
            color_idx = i % len(colors)
            if labels is not None:
                color_idx = int(labels[i]) % len(colors)
            color = colors[color_idx]
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签文本
            label_text = ""
            if class_names is not None and labels is not None:
                label_text = class_names[int(labels[i])]
            if scores is not None:
                score = scores[i]
                if label_text:
                    label_text += f": {score:.2f}"
                else:
                    label_text = f"Score: {score:.2f}"
            
            if label_text:
                cv2.putText(img, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示图像
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        
    @staticmethod
    def plot_training_history(history, figsize=(14, 5), save_path=None):
        """绘制训练历史"""
        plt.figure(figsize=figsize)
        
        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        
        # 绘制准确率
        plt.subplot(1, 2, 2)
        if 'train_acc' in history and 'val_acc' in history:
            plt.plot(history['train_acc'], label='Train Accuracy')
            plt.plot(history['val_acc'], label='Validation Accuracy')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def visualize_confusion_matrix(cm, class_names, normalize=False, figsize=(10, 8), cmap=plt.cm.Blues, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称
            normalize: 是否归一化
            figsize: 图像大小
            cmap: 颜色映射
            save_path: 保存路径
        """
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def from_predictions(y_true, y_pred, class_names=None, save_dir=None):
        """
        从预测结果生成各种可视化
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            save_dir: 保存目录
        """
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 如果没有提供类名，则使用数字标签
        if class_names is None:
            class_names = [str(i) for i in range(len(np.unique(np.concatenate([y_true, y_pred]))))]
        
        # 绘制混淆矩阵
        save_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
        Visualizer.visualize_confusion_matrix(cm, class_names, save_path=save_path)
        
        # 打印分类报告
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # 保存分类报告
        if save_dir:
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(save_dir, 'classification_report.csv'))
        
        return cm
    
    @staticmethod
    def side_by_side_comparison(images, titles=None, figsize=(12, 5), save_path=None):
        """
        并排比较多个图像
        
        Args:
            images: 图像列表
            titles: 标题列表
            figsize: 图像大小
            save_path: 保存路径
        """
        n = len(images)
        
        if titles is None:
            titles = [f"Image {i+1}" for i in range(n)]
            
        # 创建图表
        fig, axes = plt.subplots(1, n, figsize=figsize)
        
        # 显示单个图像
        if n == 1:
            axes.imshow(images[0])
            axes.set_title(titles[0])
            axes.axis('off')
        else:
            for i in range(n):
                axes[i].imshow(images[i])
                axes[i].set_title(titles[i])
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def visualize_gradcam(model, image, target_class, layer_name=None, figsize=(12, 4), save_path=None):
        """
        可视化Grad-CAM激活热图
        
        Args:
            model: 训练好的PyTorch模型
            image: 输入图像 (torch.Tensor或numpy数组)
            target_class: 目标类别索引
            layer_name: 要可视化的层名称（如果为None则使用最后一个卷积层）
            figsize: 图像大小
            save_path: 保存路径
        """
        # 检查是否已安装pytorch-grad-cam
        try:
            import importlib
            if importlib.util.find_spec("pytorch_grad_cam") is None:
                print("请安装pytorch-grad-cam: pip install pytorch-grad-cam")
                print("执行命令: pip install pytorch-grad-cam")
                return
            
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
        except ImportError:
            print("请安装pytorch-grad-cam: pip install pytorch-grad-cam")
            print("执行命令: pip install pytorch-grad-cam")
            return
        
        # 准备输入
        if isinstance(image, np.ndarray):
            img_array = image.copy()
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            
            # 转换为torch tensor [1, 3, H, W]
            if img_array.shape[0] != 3:  # 如果不是 [3, H, W]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0)
            else:
                img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)
        else:
            img_tensor = image.unsqueeze(0) if image.dim() == 3 else image
            # 转换为numpy [H, W, 3]用于显示
            img_array = image.permute(1, 2, 0).cpu().numpy()
            if img_array.max() <= 1.0:
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        
        # 查找目标层
        if layer_name is None:
            # 找到最后一个卷积层
            target_layer = None
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential)):
                    target_layer = module
                    layer_name = name
                    break
                    
            if target_layer is None:
                print("无法自动找到卷积层，请指定layer_name")
                return
        else:
            # 通过名称查找层
            target_layer = None
            for name, module in model.named_modules():
                if name == layer_name:
                    target_layer = module
                    break
                    
            if target_layer is None:
                print(f"找不到指定的层: {layer_name}")
                return
        
        # 创建GradCAM
        cam = GradCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())
        
        # 生成热力图
        grayscale_cam = cam(input_tensor=img_tensor, target_category=target_class)
        grayscale_cam = grayscale_cam[0, :]  # 提取第一个批次
        
        # 创建可视化
        visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
        
        # 显示结果
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_array)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("GradCAM Heatmap")
        plt.imshow(grayscale_cam, cmap='jet')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Overlayed Result")
        plt.imshow(visualization)
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
            
        return grayscale_cam
    
    @staticmethod
    def visualize_plant_segmentation(image, mask, figsize=(12, 4), save_path=None):
        """
        可视化植物分割结果
        
        Args:
            image: 原始图像
            mask: 分割掩码
            figsize: 图像大小
            save_path: 保存路径
        """
        # 确保图像是RGB格式
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            
        # 标准化到[0,1]
        if image.max() > 1.0:
            image = image / 255.0
            
        # 创建叠加图像
        overlay = image.copy()
        overlay[mask > 0] = [0, 1, 0]  # 用绿色表示植物区域
        
        # 创建边界图像
        contour_img = image.copy()
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, (0, 1, 0), 2)
        
        # 显示结果
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Segmentation Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Segmentation Contour")
        plt.imshow(contour_img)
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def visualize_dataset_samples(dataset, num_samples=5, cols=5, figsize=(15, 10), save_path=None):
        """
        可视化数据集样本
        
        Args:
            dataset: PyTorch数据集
            num_samples: 要显示的样本数量
            cols: 每行的图像数
            figsize: 图像大小
            save_path: 保存路径
        """
        # 计算行数
        rows = (num_samples + cols - 1) // cols
        
        # 创建图表
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # 获取随机索引
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        # 遍历样本
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample['image']
            label = sample['label']
            
            # 获取类名(如果数据集有idx_to_class属性)
            class_name = "Unknown"
            if hasattr(dataset, 'idx_to_class'):
                class_name = dataset.idx_to_class.get(label.item(), f"Class {label.item()}")
            
            # 转换图像用于显示
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).cpu().numpy()
                
                # 反标准化
                if image.max() <= 1.0 and image.min() < 0:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image = image * std + mean
                    
                image = np.clip(image, 0, 1)
            
            # 计算索引位置
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # 显示图像
            ax.imshow(image)
            ax.set_title(f"{class_name}")
            ax.axis('off')
        
        # 清空未使用的子图
        for i in range(len(indices), rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()