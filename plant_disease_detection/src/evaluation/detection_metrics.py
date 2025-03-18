#src/evaluation/detection_metrics.py

import numpy as np
import torch
from typing import List, Dict, Tuple, Union
import logging

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算两组边界框之间的IoU

    Args:
        boxes1: 形状为[N, 4]的边界框
        boxes2: 形状为[M, 4]的边界框

    Returns:
        形状为[N, M]的IoU矩阵
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    根据精度-召回曲线计算平均精度(AP)
    
    Args:
        recalls: 召回率数组
        precisions: 精度数组
        
    Returns:
        AP值
    """
    # 为精度值添加首尾值
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    
    # 从后向前计算最大精度
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # 计算召回率变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # 计算AP
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)

def calculate_metrics_per_class(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    pred_labels: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    gt_labels: List[torch.Tensor],
    iou_threshold: float = 0.5,
    num_classes: int = None
) -> Dict[str, Union[float, List[float]]]:
    """
    计算每个类别的评估指标
    
    Args:
        pred_boxes: 预测边界框列表
        pred_scores: 预测分数列表
        pred_labels: 预测标签列表
        gt_boxes: 真实边界框列表
        gt_labels: 真实标签列表
        iou_threshold: IoU阈值
        num_classes: 类别数量
        
    Returns:
        包含AP和mAP的字典
    """
    if num_classes is None:
        # 收集所有类别ID
        all_labels = set()
        for labels in gt_labels:
            all_labels.update(labels.cpu().numpy())
        for labels in pred_labels:
            all_labels.update(labels.cpu().numpy())
        num_classes = max(all_labels) + 1
    
    # 初始化AP列表
    ap_list = [0] * num_classes
    
    # 对每个类别计算AP
    for class_id in range(num_classes):
        all_detections = []
        all_gt_count = 0
        
        # 整理所有的检测结果和真值
        for i in range(len(pred_boxes)):
            # 获取该类别的预测
            class_mask = pred_labels[i] == class_id
            boxes = pred_boxes[i][class_mask]
            scores = pred_scores[i][class_mask]
            
            # 获取该类别的真值
            gt_mask = gt_labels[i] == class_id
            gts = gt_boxes[i][gt_mask]
            
            all_gt_count += len(gts)
            
            # 将检测结果添加到列表
            all_detections.extend([(score.item(), i, box) for score, box in zip(scores, boxes)])
        
        # 如果没有该类别的真值，跳过
        if all_gt_count == 0:
            continue
            
        # 按置信度排序检测结果
        all_detections.sort(key=lambda x: x[0], reverse=True)
        
        # 计算TP和FP
        tp = np.zeros(len(all_detections))
        fp = np.zeros(len(all_detections))
        
        # 用于跟踪每张图像中已经匹配的真值
        already_matched = {}
        
        # 遍历所有检测
        for i, (score, img_idx, pred_box) in enumerate(all_detections):
            # 获取该图像中该类别的真值
            gt_mask = gt_labels[img_idx] == class_id
            gt_boxes_img = gt_boxes[img_idx][gt_mask]
            
            # 如果没有真值，则为假阳性
            if len(gt_boxes_img) == 0:
                fp[i] = 1
                continue
                
            # 计算与所有真值的IoU
            iou = box_iou(pred_box.unsqueeze(0), gt_boxes_img).squeeze(0)
            
            # 获取最大IoU及其索引
            max_iou, max_idx = iou.max(0)
            
            # 检查是否大于阈值且没有被匹配过
            key = (img_idx, max_idx.item())
            if max_iou > iou_threshold and key not in already_matched:
                tp[i] = 1
                already_matched[key] = True
            else:
                fp[i] = 1
                
        # 计算累积指标
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        recalls = cum_tp / all_gt_count
        precisions = cum_tp / (cum_tp + cum_fp)
        
        # 计算AP
        ap = compute_ap(recalls, precisions)
        ap_list[class_id] = ap
    
    # 计算mAP (忽略没有真值的类别)
    valid_aps = [ap for ap in ap_list if ap > 0]
    mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0
    
    return {
        "AP_per_class": ap_list,
        "mAP": mAP
    }