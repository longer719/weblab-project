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
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image

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
    """检测模型评估器 (适配为评估图像级分类性能)"""

    def evaluate(self, data_loader: torch.utils.data.DataLoader,
                 output_dir: Optional[str] = None,
                 iou_threshold: float = 0.5) -> Dict[str, Any]:
        """
        评估检测模型（作为图像级分类器）

        Args:
            data_loader: 数据加载器
            output_dir: 输出目录路径
            iou_threshold: (在此模式下通常不使用)

        Returns:
            包含图像级分类评估指标的字典
        """
        output_path = self._prepare_output_dir(output_dir)

        all_preds_clf = []
        all_targets_clf = []
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_true_boxes = []
        all_true_labels = []
        all_images = []

        with torch.no_grad():
            pbar = tqdm(data_loader, desc="评估检测器(分类模式)")
            for batch in pbar:
                images_list, targets_list = batch
                images = [img.to(self.device) for img in images_list]
                targets_for_eval = [{'labels': t['labels'].to(self.device)} for t in targets_list]

                outputs = self.model(images)

                for i in range(len(outputs)):
                    output = outputs[i]
                    true_label = targets_list[i]['labels'][0].item()
                    all_targets_clf.append(true_label)

                    pred_scores_tensor = output.get('scores')
                    pred_labels_tensor = output.get('labels')

                    if pred_scores_tensor is not None and pred_labels_tensor is not None and len(pred_scores_tensor) > 0:
                        top_idx = torch.argmax(pred_scores_tensor)
                        pred_label = pred_labels_tensor[top_idx].item()
                        all_preds_clf.append(pred_label)
                    else:
                        all_preds_clf.append(-1)

                    all_pred_boxes.append(output.get('boxes', torch.tensor([])).cpu().numpy())
                    all_pred_scores.append(output.get('scores', torch.tensor([])).cpu().numpy())
                    all_pred_labels.append(output.get('labels', torch.tensor([])).cpu().numpy())
                    all_true_boxes.append(targets_list[i]['boxes'].cpu().numpy())
                    all_true_labels.append(targets_list[i]['labels'].cpu().numpy())
                    if len(all_images) < 5:
                        all_images.append(images_list[i].cpu())

        results = {}
        all_preds_clf = np.array(all_preds_clf)
        all_targets_clf = np.array(all_targets_clf)
        valid_mask_clf = all_preds_clf != -1

        if np.any(valid_mask_clf):
             preds_valid = all_preds_clf[valid_mask_clf]
             targets_valid = all_targets_clf[valid_mask_clf]

             if len(preds_valid) > 0:
                 results['accuracy'] = accuracy_score(targets_valid, preds_valid)
                 results['precision'] = precision_score(targets_valid, preds_valid, average='macro', zero_division=0)
                 results['recall'] = recall_score(targets_valid, preds_valid, average='macro', zero_division=0)
                 results['f1_score'] = f1_score(targets_valid, preds_valid, average='macro', zero_division=0)
                 results['confusion_matrix'] = confusion_matrix(targets_valid, preds_valid)
                 logging.info(f"图像级分类评估: Acc={results['accuracy']:.4f}, P={results['precision']:.4f}, R={results['recall']:.4f}, F1={results['f1_score']:.4f}")
             else:
                 logging.warning("有效预测为空，无法计算分类指标。")
                 results = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'confusion_matrix': None}
        else:
             logging.warning("没有任何有效预测，无法计算分类指标。")
             results = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'confusion_matrix': None}

        cm = results.get('confusion_matrix')
        if cm is not None:
            logging.info("尝试绘制混淆矩阵...")
            model_dir = Path('models')
            mapping_path = model_dir / 'plant_classes.json'
            id_to_name_map = {}
            if mapping_path.exists():
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        id_to_name_map = json.load(f)
                    logging.info(f"成功加载类别映射文件: {mapping_path}")
                except Exception as e:
                    logging.error(f"加载类别映射文件失败: {e}")
            else:
                logging.warning(f"类别映射文件未找到: {mapping_path}")

            num_classes_in_cm = cm.shape[0]
            class_names_list = []
            all_possible_dataset_names = data_loader.dataset.class_names if hasattr(data_loader.dataset, 'class_names') and isinstance(data_loader.dataset.class_names, list) else None

            for i in range(num_classes_in_cm):
                class_name = id_to_name_map.get(str(i))

                if not class_name and all_possible_dataset_names and i < len(all_possible_dataset_names):
                    class_name = all_possible_dataset_names[i]
                    logging.debug(f"ID {i} 未在映射中找到，使用数据集名称: {class_name}")

                if not class_name:
                    class_name = f'类别_{i}'
                    logging.warning(f"ID {i} 名称未知，使用默认: {class_name}")

                class_names_list.append(class_name)

            logging.info(f"混淆矩阵使用的类别名称数量: {len(class_names_list)}")

            try:
                # 尝试使用系统已安装的文泉驿微米黑字体和其他回退选项
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', '文泉驛微米黑', '文泉驿微米黑', 
                                                   'SimHei', 'DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                logging.info("尝试设置字体以支持中文显示")
            except Exception as font_e:
                logging.warning(f"设置字体失败: {font_e}. 将使用系统默认字体。")

            try:
                # 增加标签旋转角度并确保有足够的标签间距
                fig_cm = plot_confusion_matrix(cm, class_names_list, normalize=False, 
                                              figsize=(24, 20))  # 增大图表大小
                
                # 在保存前手动旋转X轴标签
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()  # 确保旋转的标签有足够空间
                
                cm_path = output_path / "confusion_matrix_detector_clf.png"
                fig_cm.savefig(cm_path, dpi=100, bbox_inches='tight')
                plt.close(fig_cm)
                logging.info(f"混淆矩阵图已保存: {cm_path}")

                # 归一化混淆矩阵
                fig_cm_norm = plot_confusion_matrix(cm, class_names_list, normalize=True, 
                                                   figsize=(24, 20))
                
                # 同样旋转标签
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                cm_norm_path = output_path / "confusion_matrix_detector_clf_normalized.png"
                fig_cm_norm.savefig(cm_norm_path, dpi=100, bbox_inches='tight')
                plt.close(fig_cm_norm)
                logging.info(f"归一化混淆矩阵图已保存: {cm_norm_path}")
            except Exception as plot_e:
                logging.error(f"绘制混淆矩阵时出错: {plot_e}")
                import traceback
                traceback.print_exc()
                
                # 尝试使用简化方式绘制混淆矩阵
                try:
                    logging.info("尝试使用简化方式绘制混淆矩阵...")
                    plt.figure(figsize=(20, 16))
                    plt.imshow(cm, interpolation='nearest', cmap='Blues')
                    plt.colorbar()
                    plt.title("Confusion Matrix")
                    
                    # 使用数字索引作为标签
                    plt.xticks(np.arange(len(class_names_list)), 
                               [f"#{i}" for i in range(len(class_names_list))], 
                               rotation=45)
                    plt.yticks(np.arange(len(class_names_list)), 
                               [f"#{i}" for i in range(len(class_names_list))])
                    
                    simple_cm_path = output_path / "confusion_matrix_simple.png"
                    plt.tight_layout()
                    plt.savefig(simple_cm_path, bbox_inches='tight')
                    plt.close()
                    logging.info(f"简化混淆矩阵已保存: {simple_cm_path}")
                    
                    # 额外保存一个类别对照文件
                    with open(output_path / "class_index_mapping.txt", "w", encoding="utf-8") as f:
                        for i, name in enumerate(class_names_list):
                            f.write(f"#{i}: {name}\n")
                    logging.info(f"类别索引映射已保存: {output_path / 'class_index_mapping.txt'}")
                except Exception as e:
                    logging.error(f"尝试简化绘图也失败了: {e}")
        else:
            logging.warning("无混淆矩阵数据，跳过绘图。")

        if self.config_manager.get('VISUALIZE_DETECTION_RESULTS', False, "model") and len(all_images) > 0:
            self._visualize_original_detections(
                all_images, all_pred_boxes, all_pred_scores, all_pred_labels,
                id_to_name_map,
                output_path
            )

        self._save_evaluation_summary(results, output_path)

        return results

    def _visualize_original_detections(self, images: List[torch.Tensor],
                             pred_boxes: List[np.ndarray],
                             pred_scores: List[np.ndarray],
                             pred_labels: List[np.ndarray],
                             class_names_map: Optional[Dict[str, str]],
                             output_path: Path,
                             max_images = 5, score_threshold = 0.1):
        """
        (辅助函数) 可视化原始检测框（主要用于调试）
        """
        logging.info("开始可视化原始检测结果（用于调试）...")
        vis_dir = output_path / "visualizations_raw_detection"
        vis_dir.mkdir(exist_ok=True)
        num_samples = min(max_images, len(images))

        for i in range(num_samples):
             image_np = images[i].permute(1, 2, 0).numpy()
             if image_np.max() <= 1.0:
                  mean = np.array([0.485, 0.456, 0.406])
                  std = np.array([0.229, 0.224, 0.225])
                  image_np = image_np * std + mean
                  image_np = np.clip(image_np, 0, 1)
             image_np = (image_np * 255).astype(np.uint8)

             current_class_names = []
             if class_names_map:
                  max_id = 0
                  if pred_labels[i].size > 0:
                       max_id = int(np.max(pred_labels[i]))
                  current_class_names = [class_names_map.get(str(j), f'ID_{j}') for j in range(max_id + 1)]

             vis_img = visualize_detection_results(
                 image_np,
                 torch.from_numpy(pred_boxes[i]),
                 torch.from_numpy(pred_scores[i]),
                 torch.from_numpy(pred_labels[i]),
                 class_names=current_class_names,
                 score_threshold=score_threshold
             )

             save_name = vis_dir / f"raw_detection_sample_{i}.png"
             try:
                  Image.fromarray(vis_img).save(save_name)
             except Exception as e:
                  logging.error(f"保存可视化图像失败 {save_name}: {e}")

        logging.info(f"原始检测结果可视化已保存至: {vis_dir}")


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