# src/evaluation/metrics.py

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple, Optional, Any
from sklearn.metrics import accuracy_score as sk_accuracy
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import recall_score as sk_recall
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config_manager import ConfigManager


# ===== 分类评估指标 =====

def accuracy(y_true: Union[torch.Tensor, np.ndarray], 
             y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """计算分类准确率"""
    # 转换为numpy数组以确保一致性
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return float(sk_accuracy(y_true, y_pred))

def precision(y_true: Union[torch.Tensor, np.ndarray], 
              y_pred: Union[torch.Tensor, np.ndarray],
              average: str = 'macro',
              zero_division: int = 0) -> Union[float, np.ndarray]:
    """计算精确率"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return sk_precision(y_true, y_pred, average=average, zero_division=zero_division)

def recall(y_true: Union[torch.Tensor, np.ndarray], 
           y_pred: Union[torch.Tensor, np.ndarray],
           average: str = 'macro',
           zero_division: int = 0) -> Union[float, np.ndarray]:
    """计算召回率"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return sk_recall(y_true, y_pred, average=average, zero_division=zero_division)

def f1(y_true: Union[torch.Tensor, np.ndarray], 
       y_pred: Union[torch.Tensor, np.ndarray],
       average: str = 'macro',
       zero_division: int = 0) -> Union[float, np.ndarray]:
    """计算F1分数"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return sk_f1_score(y_true, y_pred, average=average, zero_division=zero_division)

def confusion_matrix(y_true: Union[torch.Tensor, np.ndarray],
                     y_pred: Union[torch.Tensor, np.ndarray],
                     normalize: Optional[str] = None) -> np.ndarray:
    """计算混淆矩阵"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return sk_confusion_matrix(y_true, y_pred, normalize=normalize)

def classification_report(y_true: Union[torch.Tensor, np.ndarray],
                          y_pred: Union[torch.Tensor, np.ndarray],
                          class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """生成分类报告（包含多种指标）"""
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1_val = f1(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1_val,
        'confusion_matrix': cm,
        'class_names': class_names
    }


# ===== 检测评估指标 =====

def iou(box1: Union[torch.Tensor, np.ndarray], 
        box2: Union[torch.Tensor, np.ndarray]) -> float:
    """计算两个边界框的IoU"""
    # 确保输入是numpy数组
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
    
    # 获取交集矩形的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # 计算IoU
    iou_value = intersection_area / union_area if union_area > 0 else 0
    
    return float(iou_value)

def compute_iou_matrix(boxes1: Union[torch.Tensor, np.ndarray],
                      boxes2: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """计算两组边界框之间的IoU矩阵"""
    # 确保输入是numpy数组
    if isinstance(boxes1, torch.Tensor):
        boxes1 = boxes1.cpu().numpy()
    if isinstance(boxes2, torch.Tensor):
        boxes2 = boxes2.cpu().numpy()
    
    # 初始化IoU矩阵
    iou_matrix = np.zeros((len(boxes1), len(boxes2)))
    
    # 计算每对边界框之间的IoU
    for i in range(len(boxes1)):
        for j in range(len(boxes2)):
            iou_matrix[i, j] = iou(boxes1[i], boxes2[j])
    
    return iou_matrix

def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """使用PR曲线计算AP值"""
    # 使用精确率和召回率曲线计算AP
    
def compute_ap(pred_boxes: List[np.ndarray], 
              pred_scores: List[np.ndarray], 
              pred_labels: List[np.ndarray],
              true_boxes: List[np.ndarray], 
              true_labels: List[np.ndarray],
              iou_threshold: float = 0.5) -> Dict[str, float]:
    """计算平均精确度AP"""
    # 为每个类别计算AP
    
def mean_average_precision(pred_boxes: List[np.ndarray], 
                          pred_scores: List[np.ndarray], 
                          pred_labels: List[np.ndarray],
                          true_boxes: List[np.ndarray], 
                          true_labels: List[np.ndarray],
                          iou_thresholds: List[float] = None) -> Dict[str, float]:
    """计算mAP"""
    config_manager = ConfigManager()
    
    if iou_thresholds is None:
        iou_thresholds = config_manager.get('DETECTION_IOU_THRESHOLDS', 
                                          [0.5, 0.55, 0.6, 0.65, 0.7, 0.75], 
                                          "model")
    
    # 初始化结果字典
    results = {
        'mAP': 0.0,
        'AP_per_class': {},
        'AP_per_iou': {},
        'precision': {},
        'recall': {}
    }
    
    # 实现mAP计算逻辑
    # 此处需要实现完整的mAP计算，包括精确率-召回率曲线和不同IoU阈值下的计算
    
    return results
    
def precision_recall_curve(pred_scores: np.ndarray, 
                          true_positives: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算PR曲线数据点"""
    # 返回不同阈值下的精确率、召回率值


# ===== 可视化函数 =====

def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         figsize: Tuple[int, int] = (10, 8),
                         cmap: str = 'Blues',
                         normalize: bool = False) -> plt.Figure:
    """绘制混淆矩阵"""
    config_manager = ConfigManager()
    
    if figsize is None:
        figsize = config_manager.get('FIGURE_SIZE_MEDIUM', (10, 8), "data")
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵' + (' (归一化)' if normalize else ''))
    plt.tight_layout()
    
    return plt.gcf()

def plot_precision_recall_curve(precision: np.ndarray,
                               recall: np.ndarray,
                               ap: float,
                               class_name: str = '',
                               figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """绘制PR曲线"""
    # 可视化PR曲线并显示AP值


# ===== 指标跟踪器类 =====

class ClassificationMetricsTracker:
    """分类指标跟踪器，用于累积和计算训练/评估过程中的指标"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """初始化分类指标跟踪器"""
        # 初始化累积变量
        
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """更新指标状态"""
        # 累积预测和真实标签
        
    def compute(self) -> Dict[str, Any]:
        """计算累积的指标值"""
        # 计算并返回各种指标
        
    def reset(self) -> None:
        """重置跟踪器状态"""
        # 清除累积的数据
        
    def get_report(self, include_matrix: bool = True) -> Dict[str, Any]:
        """获取完整的评估报告"""
        # 生成包含所有指标的报告


class DetectionMetricsTracker:
    """检测指标跟踪器，用于累积和计算目标检测评估指标"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None,
                iou_thresholds: List[float] = None):
        """初始化检测指标跟踪器"""
        # 初始化累积变量
        
    def update(self, 
              pred_boxes: List[torch.Tensor], 
              pred_scores: List[torch.Tensor], 
              pred_labels: List[torch.Tensor],
              true_boxes: List[torch.Tensor], 
              true_labels: List[torch.Tensor]) -> None:
        """更新检测结果"""
        # 累积预测和真实检测结果
        
    def compute(self) -> Dict[str, Any]:
        """计算累积的检测指标"""
        # 计算mAP等指标
        
    def reset(self) -> None:
        """重置跟踪器状态"""
        # 清除累积的数据
        
    def get_report(self, include_curves: bool = False) -> Dict[str, Any]:
        """获取完整的检测评估报告"""
        # 生成包含所有指标的报告