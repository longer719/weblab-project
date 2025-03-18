# src/evaluation/evaluator.py

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from src.utils.config_manager import ConfigManager
from src.evaluation.metrics import (
    accuracy, precision, recall, f1, confusion_matrix, plot_confusion_matrix,
    classification_report, mean_average_precision, compute_iou_matrix
)
from src.models.components.detection_utils import visualize_detection_results


class BaseEvaluator:
    """评估器基类，提供共享功能"""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 运行设备，默认是自动选择
        """
        self.model = model
        self.config_manager = ConfigManager()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
    def _prepare_output_dir(self, output_dir: Optional[str] = None) -> Path:
        """
        准备输出目录
        
        Args:
            output_dir: 输出目录路径，如果为None则使用默认路径
            
        Returns:
            Path对象，指向准备好的输出目录
        """
        if output_dir is None:
            output_dir = self.config_manager.get('EVALUATION_OUTPUT_DIR', 'evaluation_results', "data")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def _save_evaluation_summary(self, results: Dict[str, Any], output_path: Path):
        """
        保存评估结果摘要
        
        Args:
            results: 评估结果字典
            output_path: 输出路径
        """
        # 创建摘要文本
        summary_lines = ["===== 模型评估结果摘要 =====\n"]
        
        # 添加基本信息
        summary_lines.append(f"模型类型: {type(self.model).__name__}")
        summary_lines.append(f"设备: {self.device}\n")
        
        # 添加性能指标
        summary_lines.append("性能指标:")
        for metric_name, metric_value in results.items():
            if isinstance(metric_value, (int, float)):
                summary_lines.append(f"  {metric_name}: {metric_value:.4f}")
                
        # 保存到文件
        with open(output_path / "evaluation_summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))


class ClassificationEvaluator(BaseEvaluator):
    """分类模型评估器"""
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader, 
                 output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        评估分类模型
        
        Args:
            data_loader: 数据加载器，用于加载评估数据
            output_dir: 输出目录路径
            
        Returns:
            包含各种评估指标的字典
        """
        output_path = self._prepare_output_dir(output_dir)
        
        # 收集所有预测和真实标签
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估"):
                # 获取输入和标签
                inputs = batch['images'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                # 获取预测
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # 收集结果
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_probs.append(probs.cpu())
        
        # 合并所有批次结果
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        # 计算各种指标
        results = self._compute_metrics(all_preds, all_targets, all_probs)
        
        # 生成可视化结果
        if self.config_manager.get('GENERATE_CONFUSION_MATRIX', True, "model"):
            self._plot_confusion_matrix(results['confusion_matrix'], 
                                      data_loader.dataset.class_names if hasattr(data_loader.dataset, 'class_names') else None,
                                      output_path)
        
        # 保存评估摘要
        self._save_evaluation_summary(results, output_path)
        
        return results
    
    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray, 
                        probs: np.ndarray) -> Dict[str, Any]:
        """
        计算各种分类指标
        
        Args:
            preds: 预测类别
            targets: 真实标签
            probs: 预测概率
            
        Returns:
            包含各种指标的字典
        """
        # 基础指标
        acc = accuracy(targets, preds)
        prec = precision(targets, preds)
        rec = recall(targets, preds)
        f1_score = f1(targets, preds)
        cm = confusion_matrix(targets, preds)
        
        # 生成详细报告
        report = classification_report(targets, preds)
        
        # 返回所有指标
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1_score,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': preds,
            'targets': targets,
            'probabilities': probs
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: Optional[List[str]], 
                              output_path: Path):
        """
        绘制并保存混淆矩阵
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称列表
            output_path: 输出路径
        """
        from src.evaluation.metrics import plot_confusion_matrix
        
        if class_names is None:
            class_names = [f"类别{i}" for i in range(cm.shape[0])]
            
        # 绘制标准和归一化混淆矩阵
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(cm, class_names, normalize=False)
        plt.savefig(output_path / "confusion_matrix.png", dpi=100, bbox_inches='tight')
        
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(cm, class_names, normalize=True)
        plt.savefig(output_path / "confusion_matrix_normalized.png", dpi=100, bbox_inches='tight')
    
    def evaluate_sample(self, image: torch.Tensor, 
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        评估单个样本
        
        Args:
            image: 输入图像张量
            output_dir: 输出目录路径
            
        Returns:
            包含评估结果的字典
        """
        if len(image.shape) == 3:
            # 添加批次维度
            image = image.unsqueeze(0)
            
        with torch.no_grad():
            # 获取预测
            image = image.to(self.device)
            output = self.model(image)
            probs = torch.softmax(output, dim=1)
            
            # 获取预测类别和概率
            prob, pred = torch.max(probs, 1)
            
        # 将结果转换为字典
        result = {
            'prediction': pred.item(),
            'confidence': prob.item(),
            'class_probabilities': probs[0].cpu().numpy()
        }
        
        return result


class DetectionEvaluator(BaseEvaluator):
    """检测模型评估器"""
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader, 
                 output_dir: Optional[str] = None,
                 iou_threshold: float = 0.5) -> Dict[str, Any]:
        """
        评估检测模型
        
        Args:
            data_loader: 数据加载器，用于加载评估数据
            output_dir: 输出目录路径
            iou_threshold: IoU阈值
            
        Returns:
            包含各种评估指标的字典
        """
        output_path = self._prepare_output_dir(output_dir)
        
        # 收集所有预测和真实边界框
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_true_boxes = []
        all_true_labels = []
        all_images = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估"):
                # 获取输入和标签
                # 修改这部分代码以处理列表格式的图像
                if isinstance(batch['images'], list):
                    # 列表格式的批次处理
                    images = [img.to(self.device) for img in batch['images']]
                    targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in t.items()} for t in batch['targets']]
                else:
                    # 张量格式的批次处理（原来的代码）
                    images = batch['images'].to(self.device)
                    targets = [{k: v.to(self.device) for k, v in t.items() if isinstance(v, torch.Tensor)}
                            for t in batch['targets']]
                
                # 获取预测
                outputs = self.model(images)
                
                # 收集结果
                for i in range(len(images)):
                    # 获取单个样本预测结果
                    pred_boxes = outputs[i]['boxes'].cpu().numpy()
                    pred_scores = outputs[i]['scores'].cpu().numpy()
                    pred_labels = outputs[i]['labels'].cpu().numpy()
                    
                    # 获取单个样本真实标签
                    true_boxes = targets[i]['boxes'].cpu().numpy()
                    true_labels = targets[i]['labels'].cpu().numpy()
                    
                    all_pred_boxes.append(pred_boxes)
                    all_pred_scores.append(pred_scores)
                    all_pred_labels.append(pred_labels)
                    all_true_boxes.append(true_boxes)
                    all_true_labels.append(true_labels)
                    all_images.append(images[i].cpu())
                    
        # 计算mAP
        iou_thresholds = self.config_manager.get('DETECTION_IOU_THRESHOLDS', 
                                                [0.5, 0.55, 0.6, 0.65, 0.7, 0.75], 
                                                "model")
        
        # 计算检测评估指标
        map_results = mean_average_precision(
            all_pred_boxes, all_pred_scores, all_pred_labels,
            all_true_boxes, all_true_labels,
            iou_thresholds=iou_thresholds
        )
        
        # 计算每个类别的AP
        class_names = data_loader.dataset.class_names if hasattr(data_loader.dataset, 'class_names') else None
        
        # 生成整体结果字典
        results = {
            'mAP': map_results['mAP'],
            'AP_per_class': map_results['AP_per_class'],
            'AP_per_iou': map_results['AP_per_iou'],
            'precision': map_results['precision'],
            'recall': map_results['recall'],
            'pred_boxes': all_pred_boxes,
            'pred_scores': all_pred_scores,
            'pred_labels': all_pred_labels,
            'true_boxes': all_true_boxes,
            'true_labels': all_true_labels
        }
        
        # 可视化部分检测结果
        if self.config_manager.get('VISUALIZE_DETECTION_RESULTS', True, "model"):
            self._visualize_detections(all_images, all_pred_boxes, all_pred_scores, 
                                      all_pred_labels, class_names, output_path)
        
        # 保存评估摘要
        self._save_evaluation_summary(results, output_path)
        
        return results
    
    def _visualize_detections(self, images: List[torch.Tensor], 
                             pred_boxes: List[np.ndarray],
                             pred_scores: List[np.ndarray], 
                             pred_labels: List[np.ndarray],
                             class_names: Optional[List[str]], 
                             output_path: Path):
        """
        可视化检测结果
        
        Args:
            images: 图像列表
            pred_boxes: 预测边界框列表
            pred_scores: 预测分数列表
            pred_labels: 预测标签列表
            class_names: 类别名称列表
            output_path: 输出路径
        """
        # 获取可视化样本数
        num_samples = min(self.config_manager.get('VISUALIZATION_SAMPLES', 5, "model"), len(images))
        
        # 创建可视化目录
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 可视化前n个样本
        for i in range(num_samples):
            # 转换图像格式
            image_np = images[i].permute(1, 2, 0).numpy()
            
            # 如果图像是归一化的，需要反归一化
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            
            # 可视化检测结果
            vis_img = visualize_detection_results(
                image_np, 
                torch.from_numpy(pred_boxes[i]),
                torch.from_numpy(pred_scores[i]),
                torch.from_numpy(pred_labels[i]),
                class_names=class_names,
                score_threshold=self.config_manager.get('DETECTION_SCORE_THRESHOLD', 0.5, "model")
            )
            
            # 保存可视化结果
            plt.figure(figsize=(10, 10))
            plt.imshow(vis_img)
            plt.axis('off')
            plt.savefig(vis_dir / f"detection_sample_{i}.png", dpi=100, bbox_inches='tight')
            plt.close()
    
    def evaluate_sample(self, image: torch.Tensor, 
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        评估单个样本
        
        Args:
            image: 输入图像张量
            output_dir: 输出目录路径
            
        Returns:
            包含评估结果的字典
        """
        if len(image.shape) == 3:
            # 添加批次维度
            image = image.unsqueeze(0)
            
        with torch.no_grad():
            # 获取预测
            image = image.to(self.device)
            outputs = self.model(image)
            
            # 提取第一个样本的结果
            pred_boxes = outputs[0]['boxes'].cpu()
            pred_scores = outputs[0]['scores'].cpu()
            pred_labels = outputs[0]['labels'].cpu()
            
        # 将结果转换为字典
        result = {
            'boxes': pred_boxes.numpy(),
            'scores': pred_scores.numpy(),
            'labels': pred_labels.numpy()
        }
        
        return result


def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, 
                  task_type: str = 'classification', output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    评估模型的便捷函数
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        task_type: 任务类型，'classification'或'detection'
        output_dir: 输出目录
        
    Returns:
        包含评估结果的字典
    """
    # 根据任务类型选择适当的评估器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if task_type == 'classification':
        evaluator = ClassificationEvaluator(model, device)
    elif task_type == 'detection':
        evaluator = DetectionEvaluator(model, device)
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")
    
    # 执行评估
    results = evaluator.evaluate(data_loader, output_dir)
    
    return results