# src/models/components/detection_utils.py
#这是一个辅助工具类，用于处理植物病害检测任务中的边界框和非极大值抑制(NMS)等操作。

import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict, Union, Optional, Any
import torch.nn.functional as F
from torchvision.ops import nms, box_iou, box_convert


class BoundingBoxUtils:
    """边界框处理工具类"""
    
    @staticmethod
    def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
        """
        将[x1, y1, x2, y2]格式转换为[x, y, w, h]格式
        
        Args:
            boxes: 边界框张量，格式为[x1, y1, x2, y2]
            
        Returns:
            [x, y, w, h]格式的边界框张量
        """
        return box_convert(boxes, 'xyxy', 'xywh')
    
    @staticmethod
    def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """
        将[x, y, w, h]格式转换为[x1, y1, x2, y2]格式
        
        Args:
            boxes: 边界框张量，格式为[x, y, w, h]
            
        Returns:
            [x1, y1, x2, y2]格式的边界框张量
        """
        return box_convert(boxes, 'xywh', 'xyxy')
    
    @staticmethod
    def normalize_boxes(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        将绝对坐标归一化为相对坐标(0-1)
        
        Args:
            boxes: 边界框张量，格式为[x1, y1, x2, y2]
            image_size: 图像尺寸(height, width)
            
        Returns:
            归一化后的边界框张量
        """
        height, width = image_size
        normalized = boxes.clone()
        normalized[:, [0, 2]] /= width
        normalized[:, [1, 3]] /= height
        return normalized
    
    @staticmethod
    def denormalize_boxes(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        将相对坐标(0-1)转换为绝对坐标
        
        Args:
            boxes: 边界框张量，格式为[x1, y1, x2, y2]，值在0-1之间
            image_size: 图像尺寸(height, width)
            
        Returns:
            绝对坐标的边界框张量
        """
        height, width = image_size
        denormalized = boxes.clone()
        denormalized[:, [0, 2]] *= width
        denormalized[:, [1, 3]] *= height
        return denormalized
    
    @staticmethod
    def compute_area(boxes: torch.Tensor) -> torch.Tensor:
        """
        计算边界框面积
        
        Args:
            boxes: 边界框张量，格式为[x1, y1, x2, y2]
            
        Returns:
            面积张量，形状为[N]
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    @staticmethod
    def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        计算两组边界框之间的IoU
        
        Args:
            boxes1: 第一组边界框，格式为[N, 4]
            boxes2: 第二组边界框，格式为[M, 4]
            
        Returns:
            IoU矩阵，形状为[N, M]
        """
        return box_iou(boxes1, boxes2)
    
    @staticmethod
    def filter_small_boxes(boxes: torch.Tensor, scores: torch.Tensor, 
                          labels: torch.Tensor, min_area: float = 25.0,
                          min_width: float = 5.0, min_height: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        过滤太小的边界框
        
        Args:
            boxes: 边界框张量，格式为[x1, y1, x2, y2]
            scores: 置信度分数
            labels: 类别标签
            min_area: 最小面积
            min_width: 最小宽度
            min_height: 最小高度
            
        Returns:
            过滤后的(boxes, scores, labels)
        """
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        keep = (areas >= min_area) & (widths >= min_width) & (heights >= min_height)
        
        return boxes[keep], scores[keep], labels[keep]
    
    @staticmethod
    def crop_boxes_to_image(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        确保边界框不超出图像范围
        
        Args:
            boxes: 边界框张量，格式为[x1, y1, x2, y2]
            image_size: 图像尺寸(height, width)
            
        Returns:
            裁剪后的边界框
        """
        height, width = image_size
        cropped = boxes.clone()
        cropped[:, 0] = torch.clamp(cropped[:, 0], min=0, max=width)
        cropped[:, 1] = torch.clamp(cropped[:, 1], min=0, max=height)
        cropped[:, 2] = torch.clamp(cropped[:, 2], min=0, max=width)
        cropped[:, 3] = torch.clamp(cropped[:, 3], min=0, max=height)
        return cropped


class NMSUtils:
    """非极大值抑制(NMS)工具类"""
    
    @staticmethod
    def apply_nms(boxes: torch.Tensor, scores: torch.Tensor, 
                 iou_threshold: float = 0.5) -> torch.Tensor:
        """
        应用标准NMS
        
        Args:
            boxes: 边界框张量，格式为[x1, y1, x2, y2]
            scores: 置信度分数
            iou_threshold: IoU阈值
            
        Returns:
            保留的边界框索引
        """
        return nms(boxes, scores, iou_threshold)
    
    @staticmethod
    def apply_soft_nms(boxes: torch.Tensor, scores: torch.Tensor,
                      iou_threshold: float = 0.5,
                      sigma: float = 0.5,
                      score_threshold: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用软NMS，对于小病斑检测更友好
        
        Args:
            boxes: 边界框张量，格式为[N, 4]，[x1, y1, x2, y2]
            scores: 置信度分数，形状为[N]
            iou_threshold: IoU阈值
            sigma: 高斯加权系数
            score_threshold: 分数阈值，低于该阈值的框将被去除
            
        Returns:
            (保留的边界框索引, 更新后的分数)
        """
        # 转换为numpy进行处理
        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
        
        N = boxes_np.shape[0]
        indices = np.arange(N)
        
        # 按分数降序排列
        order = scores_np.argsort()[::-1]
        boxes_np = boxes_np[order]
        scores_np = scores_np[order]
        indices = indices[order]
        
        keep = []
        updated_scores = scores_np.copy()
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
                
            # 计算当前框与剩余所有框的IoU
            xx1 = np.maximum(boxes_np[i, 0], boxes_np[1:, 0])
            yy1 = np.maximum(boxes_np[i, 1], boxes_np[1:, 1])
            xx2 = np.minimum(boxes_np[i, 2], boxes_np[1:, 2])
            yy2 = np.minimum(boxes_np[i, 3], boxes_np[1:, 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # 计算IoU
            areas = (boxes_np[1:, 2] - boxes_np[1:, 0]) * (boxes_np[1:, 3] - boxes_np[1:, 1])
            area_i = (boxes_np[i, 2] - boxes_np[i, 0]) * (boxes_np[i, 3] - boxes_np[i, 1])
            union = area_i + areas - inter
            iou = inter / union
            
            # 应用软NMS
            weight = np.exp(-(iou * iou) / sigma) if sigma > 0 else (iou < iou_threshold).astype(np.float32)
            updated_scores[1:] *= weight
            
            # 移除低于阈值的框
            inds = np.where(updated_scores[1:] > score_threshold)[0]
            order = order[inds + 1]
            boxes_np = boxes_np[inds + 1]
            scores_np = scores_np[inds + 1]
            updated_scores = updated_scores[inds + 1]
            indices = indices[inds + 1]
        
        # 转回PyTorch
        keep_indices = torch.tensor(indices[keep], dtype=torch.long, device=boxes.device)
        updated_scores = torch.tensor(updated_scores[keep], dtype=scores.dtype, device=scores.device)
        
        return keep_indices, updated_scores
    
    @staticmethod
    def apply_weighted_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
                          iou_threshold: float = 0.5, score_threshold: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用加权NMS，适用于小密集病斑检测
        
        Args:
            boxes: 边界框张量，格式为[N, 4]
            scores: 置信度分数，形状为[N]
            labels: 类别标签，形状为[N]
            iou_threshold: IoU阈值
            score_threshold: 分数阈值
            
        Returns:
            过滤后的(boxes, scores, labels)
        """
        if boxes.shape[0] == 0:
            return boxes, scores, labels
        
        # 根据类别分组应用NMS
        keep_boxes = []
        keep_scores = []
        keep_labels = []
        
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            class_mask = (labels == label)
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            # 应用标准NMS
            keep_indices = nms(class_boxes, class_scores, iou_threshold)
            
            # 过滤低分数预测
            high_score_mask = class_scores[keep_indices] > score_threshold
            keep_indices = keep_indices[high_score_mask]
            
            keep_boxes.append(class_boxes[keep_indices])
            keep_scores.append(class_scores[keep_indices])
            keep_labels.append(torch.full_like(class_scores[keep_indices], label))
        
        if len(keep_boxes) > 0:
            nms_boxes = torch.cat(keep_boxes)
            nms_scores = torch.cat(keep_scores)
            nms_labels = torch.cat(keep_labels)
            return nms_boxes, nms_scores, nms_labels
        else:
            return torch.zeros((0, 4), device=boxes.device), torch.zeros(0, device=scores.device), torch.zeros(0, device=labels.device)


class PlantDiseasePostProcessor:
    """植物病害检测后处理工具类"""
    
    @staticmethod
    def filter_by_area_ratio(boxes: torch.Tensor, scores: torch.Tensor, 
                            labels: torch.Tensor, plant_mask: torch.Tensor, 
                            min_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于植物区域过滤病害检测结果
        
        Args:
            boxes: 边界框张量，格式为[N, 4]，[x1, y1, x2, y2]
            scores: 置信度分数，形状为[N]
            labels: 类别标签，形状为[N]
            plant_mask: 植物区域掩码，形状为[H, W]，0-1值
            min_ratio: 最小植物区域重叠比例
            
        Returns:
            过滤后的(boxes, scores, labels)
        """
        if boxes.shape[0] == 0:
            return boxes, scores, labels
        
        # 转换为numpy处理
        boxes_np = boxes.detach().cpu().numpy().astype(np.int32)
        plant_mask_np = plant_mask.detach().cpu().numpy()
        
        keep_indices = []
        
        for i, box in enumerate(boxes_np):
            x1, y1, x2, y2 = box
            # 确保坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(plant_mask_np.shape[1] - 1, x2), min(plant_mask_np.shape[0] - 1, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 获取边界框区域的掩码
            box_mask = plant_mask_np[y1:y2, x1:x2]
            
            # 计算边界框内像素中属于植物区域的比例
            plant_pixel_ratio = np.mean(box_mask)
            
            if plant_pixel_ratio >= min_ratio:
                keep_indices.append(i)
        
        if len(keep_indices) > 0:
            keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
            return boxes[keep_tensor], scores[keep_tensor], labels[keep_tensor]
        else:
            return torch.zeros((0, 4), device=boxes.device), torch.zeros(0, device=scores.device), torch.zeros(0, device=labels.device)
    
    @staticmethod
    def adjust_small_lesion_scores(boxes: torch.Tensor, scores: torch.Tensor,
                                  labels: torch.Tensor, boost_factor: float = 1.2,
                                  small_area_threshold: float = 100) -> torch.Tensor:
        """
        提高小型病斑区域的检测分数
        
        Args:
            boxes: 边界框张量，格式为[N, 4]，[x1, y1, x2, y2]
            scores: 置信度分数，形状为[N]
            labels: 类别标签，形状为[N]
            boost_factor: 分数提升因子
            small_area_threshold: 小面积阈值
            
        Returns:
            调整后的分数
        """
        if boxes.shape[0] == 0:
            return scores
        
        # 计算面积
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        # 找出小面积病斑
        small_mask = areas < small_area_threshold
        
        # 对小病斑的分数进行提升
        adjusted_scores = scores.clone()
        adjusted_scores[small_mask] *= boost_factor
        
        # 确保分数不超过1.0
        adjusted_scores = torch.clamp(adjusted_scores, max=1.0)
        
        return adjusted_scores
    
    @staticmethod
    def merge_overlapping_detections(boxes: torch.Tensor, scores: torch.Tensor, 
                                   labels: torch.Tensor, iou_threshold: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        合并高度重叠的同类检测结果
        
        Args:
            boxes: 边界框张量，格式为[N, 4]，[x1, y1, x2, y2]
            scores: 置信度分数，形状为[N]
            labels: 类别标签，形状为[N]
            iou_threshold: 高IoU阈值，超过此阈值的框会被合并
            
        Returns:
            合并后的(boxes, scores, labels)
        """
        if boxes.shape[0] <= 1:
            return boxes, scores, labels
        
        # 计算IoU矩阵
        iou_matrix = box_iou(boxes, boxes)
        n = boxes.shape[0]
        
        # 标记需要合并的框
        merged = torch.zeros(n, dtype=torch.bool)
        merged_boxes = []
        merged_scores = []
        merged_labels = []
        
        for i in range(n):
            if merged[i]:
                continue
                
            # 找到与当前框高IoU且类别相同的所有框
            overlap = (iou_matrix[i] > iou_threshold) & (labels == labels[i])
            overlap[i] = False  # 排除自身
            
            if not overlap.any():
                merged_boxes.append(boxes[i:i+1])
                merged_scores.append(scores[i:i+1])
                merged_labels.append(labels[i:i+1])
                continue
            
            # 获取所有重叠框的索引
            merge_indices = torch.nonzero(overlap).squeeze(1)
            merge_indices = torch.cat([torch.tensor([i], device=boxes.device), merge_indices])
            
            # 标记这些框为已合并
            merged[merge_indices] = True
            
            # 计算加权平均边界框
            weights = scores[merge_indices].unsqueeze(1)
            weighted_boxes = boxes[merge_indices] * weights
            merged_box = weighted_boxes.sum(dim=0) / weights.sum()
            
            # 使用最高分数
            max_score_idx = scores[merge_indices].argmax()
            merged_score = scores[merge_indices][max_score_idx]
            merged_label = labels[merge_indices][max_score_idx]
            
            merged_boxes.append(merged_box.unsqueeze(0))
            merged_scores.append(merged_score.unsqueeze(0))
            merged_labels.append(merged_label.unsqueeze(0))
        
        if merged_boxes:
            return torch.cat(merged_boxes), torch.cat(merged_scores), torch.cat(merged_labels)
        else:
            return torch.zeros((0, 4), device=boxes.device), torch.zeros(0, device=scores.device), torch.zeros(0, device=labels.device)
    
    @staticmethod
    def detect_plant_region(image: np.ndarray) -> np.ndarray:
        """
        检测图像中的植物区域
        
        Args:
            image: RGB或BGR格式图像，形状为[H, W, 3]
            
        Returns:
            植物区域掩码，0-1值，形状为[H, W]
        """
        # 转换为HSV色彩空间，更适合植物分割
        if len(image.shape) == 3 and image.shape[2] == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 提取绿色通道的范围，适用于健康植物
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # 提取黄色通道的范围，适用于部分病害植物
            lower_yellow = np.array([20, 50, 50])
            upper_yellow = np.array([35, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # 提取棕色通道的范围，适用于部分干枯/病害区域
            lower_brown = np.array([10, 80, 30])
            upper_brown = np.array([20, 255, 150])
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # 合并掩码
            plant_mask = cv2.bitwise_or(green_mask, yellow_mask)
            plant_mask = cv2.bitwise_or(plant_mask, brown_mask)
            
            # 形态学操作：先闭操作填充小孔，再开操作去除噪点
            kernel = np.ones((5, 5), np.uint8)
            plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
            plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
            
            return plant_mask / 255.0  # 归一化到0-1
        else:
            # 如果不是3通道图像，返回全1掩码
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
    
    @staticmethod
    def process_detection_results(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
                                 image: Union[torch.Tensor, np.ndarray], 
                                 score_threshold: float = 0.2,  # 降低阈值
                                 nms_type: str = 'weighted',
                                 iou_threshold: float = 0.4,    # 降低IoU阈值
                                 filter_by_plant: bool = True,
                                 boost_small_lesions: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理植物病害检测结果，综合应用多种后处理技术
        
        Args:
            boxes: 边界框张量，格式为[N, 4]，[x1, y1, x2, y2]
            scores: 置信度分数，形状为[N]
            labels: 类别标签，形状为[N]
            image: 原始图像
            score_threshold: 分数阈值
            nms_type: NMS类型，'standard'、'soft'或'weighted'
            iou_threshold: IoU阈值
            filter_by_plant: 是否根据植物区域过滤
            boost_small_lesions: 是否提升小病斑区域的分数
            
        Returns:
            处理后的(boxes, scores, labels)
        """
        if boxes.shape[0] == 0:
            return boxes, scores, labels
        
        # 准备图像
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # [B, C, H, W]
                image = image[0].permute(1, 2, 0).cpu().numpy()
            else:  # [C, H, W]
                image = image.permute(1, 2, 0).cpu().numpy()
                
            # 转换为0-255范围
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        # 1. 过滤低分数预测 - 使用更低阈值
        high_score_mask = scores > score_threshold
        boxes = boxes[high_score_mask]
        scores = scores[high_score_mask]
        labels = labels[high_score_mask]
        
        if boxes.shape[0] == 0:
            return boxes, scores, labels
        
        # 2. 过滤太小的框 - 减少最小面积限制
        boxes, scores, labels = BoundingBoxUtils.filter_small_boxes(
            boxes, scores, labels, min_area=16.0, min_width=4.0, min_height=4.0  # 降低最小尺寸要求
        )
        
        if boxes.shape[0] == 0:
            return boxes, scores, labels
        
        # 3. 根据植物区域过滤（可选）
        if filter_by_plant:
            plant_mask = PlantDiseasePostProcessor.detect_plant_region(image)
            plant_mask_tensor = torch.from_numpy(plant_mask).to(boxes.device)
            boxes, scores, labels = PlantDiseasePostProcessor.filter_by_area_ratio(
                boxes, scores, labels, plant_mask_tensor, min_ratio=0.3
            )
            
        if boxes.shape[0] == 0:
            return boxes, scores, labels
        
        # 4. 提升小病斑的分数（可选）
        if boost_small_lesions:
            scores = PlantDiseasePostProcessor.adjust_small_lesion_scores(
                boxes, scores, labels, boost_factor=1.2, small_area_threshold=100
            )
        
        # 5. 应用NMS
        if nms_type == 'standard':
            keep_indices = NMSUtils.apply_nms(boxes, scores, iou_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
        elif nms_type == 'soft':
            keep_indices, updated_scores = NMSUtils.apply_soft_nms(
                boxes, scores, iou_threshold=iou_threshold
            )
            boxes = boxes[keep_indices]
            scores = updated_scores
            labels = labels[keep_indices]
        elif nms_type == 'weighted':
            boxes, scores, labels = NMSUtils.apply_weighted_nms(
                boxes, scores, labels, iou_threshold=iou_threshold, score_threshold=score_threshold
            )
        
        # 6. 合并重叠检测
        boxes, scores, labels = PlantDiseasePostProcessor.merge_overlapping_detections(
            boxes, scores, labels, iou_threshold=0.7
        )
        
        # 7. 裁剪边界框到图像范围内
        image_size = (image.shape[0], image.shape[1])
        boxes = BoundingBoxUtils.crop_boxes_to_image(boxes, image_size)
        
        return boxes, scores, labels


def visualize_detection_results(image: np.ndarray, boxes: torch.Tensor, scores: torch.Tensor, 
                             labels: torch.Tensor, class_names: Optional[List[str]] = None,
                             score_threshold: float = 0.5, line_thickness: int = 2) -> np.ndarray:
    """
    可视化病害检测结果
    
    Args:
        image: 原始图像 (BGR或RGB格式)
        boxes: 边界框张量，格式为[N, 4]，[x1, y1, x2, y2]
        scores: 置信度分数，形状为[N]
        labels: 类别标签，形状为[N]
        class_names: 类别名称列表
        score_threshold: 显示的分数阈值
        line_thickness: 边界框线宽
        
    Returns:
        可视化结果图像
    """
    # 转换为numpy数组
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # 创建结果图像的副本
    result_img = image.copy()
    
    # 定义颜色映射（为每个类别分配不同颜色）
    colors = [
        (0, 255, 0),     # 绿色 - 健康
        (0, 0, 255),     # 红色 - 严重病害
        (255, 0, 0),     # 蓝色 - 轻微病害
        (0, 255, 255),   # 黄色 - 中度病害
        (255, 0, 255),   # 紫色 - 早期病害
        (255, 255, 0),   # 青色 - 其他类型1
        (128, 0, 0),     # 深蓝色 - 其他类型2
        (0, 128, 0),     # 深绿色 - 其他类型3
        (128, 128, 0),   # 橄榄色 - 其他类型4
        (0, 0, 128),     # 深红色 - 其他类型5
        (128, 0, 128),   # 深紫色 - 其他类型6
    ]
    
    # 设置字体参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # 白色文字
    
    # 过滤低于阈值的检测结果
    valid_indices = scores >= score_threshold
    if not np.any(valid_indices):
        return result_img  # 如果没有有效检测，直接返回原图
    
    valid_boxes = boxes[valid_indices].astype(np.int32)
    valid_scores = scores[valid_indices]
    valid_labels = labels[valid_indices].astype(np.int32)
    
    # 绘制每个检测结果
    for box, score, label in zip(valid_boxes, valid_scores, valid_labels):
        # 获取当前类别对应的颜色
        color = colors[label % len(colors)]
        
        # 绘制边界框
        x1, y1, x2, y2 = box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, line_thickness)
        
        # 准备标签文本
        if class_names is not None and 0 <= label < len(class_names):
            label_text = class_names[label]
        else:
            label_text = f"类别 {label}"
        
        text = f"{label_text}: {score:.2f}"
        
        # 测量文本大小以确定背景矩形尺寸
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # 绘制标签背景（半透明矩形）
        text_bg_x1, text_bg_y1 = x1, y1 - text_height - 5
        text_bg_x2, text_bg_y2 = x1 + text_width + 5, y1
        
        # 确保文本背景矩形在图像内
        if text_bg_y1 < 0:
            text_bg_y1 = y1
            text_bg_y2 = y1 + text_height + 5
        
        # 绘制标签背景矩形
        alpha = 0.6  # 透明度
        overlay = result_img.copy()
        cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), 
                     (text_bg_x2, text_bg_y2), color, -1)  # -1表示填充矩形
        cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0, result_img)
        
        # 添加文字标签（类别和置信度）
        text_position = (x1 + 2, text_bg_y2 - 3)
        cv2.putText(result_img, text, text_position, font, font_scale, text_color, font_thickness)
    
    return result_img


def generate_heatmap(boxes: np.ndarray, scores: np.ndarray, image_shape: Tuple[int, int],
                    sigma: float = 10.0) -> np.ndarray:
    """
    生成病害检测热图，用于可视化检测密度
    
    Args:
        boxes: 边界框数组，格式为[N, 4]，[x1, y1, x2, y2]
        scores: 置信度分数，形状为[N]
        image_shape: 输出热图尺寸 (height, width)
        sigma: 高斯核标准差，控制热点扩散范围
        
    Returns:
        归一化的热图，值范围0-1，形状为[height, width]
    """
    height, width = image_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    if len(boxes) == 0:
        return heatmap
    
    # 将边界框转换为中心点坐标
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    
    # 计算每个中心点处的权重（基于置信度）
    weights = scores
    
    # 生成网格坐标
    x_grid = np.arange(width)
    y_grid = np.arange(height)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # 为每个检测框生成高斯热点
    for cx, cy, weight in zip(centers_x, centers_y, weights):
        # 计算每个像素到中心点的距离的平方
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        
        # 应用高斯函数
        gaussian = weight * np.exp(-dist_sq / (2 * sigma ** 2))
        
        # 累加到热图
        heatmap += gaussian
    
    # 归一化热图到[0, 1]
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, 
                   alpha: float = 0.6, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    将热图叠加到原始图像上
    
    Args:
        image: 原始图像 (BGR格式)
        heatmap: 热图，值范围0-1，形状为[height, width]
        alpha: 热图透明度
        colormap: OpenCV颜色映射类型
        
    Returns:
        叠加热图的图像
    """
    # 将热图转换为彩色图
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # 热图区域掩码
    mask = heatmap > 0.1
    
    # 创建输出图像
    result = image.copy()
    
    # 只在有热图数据的区域应用叠加
    if np.any(mask):
        # 将热图叠加到原始图像
        mask_3d = np.stack([mask, mask, mask], axis=2)
        result[mask_3d] = (alpha * heatmap_colored[mask_3d] + 
                          (1 - alpha) * image[mask_3d]).astype(np.uint8)
    
    return result


def create_detection_summary(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, 
                           labels: np.ndarray, class_names: Optional[List[str]] = None, 
                           score_threshold: float = 0.5) -> Dict[str, Any]:
    """
    创建检测结果摘要，包含检测统计信息
    
    Args:
        image: 原始图像
        boxes: 边界框数组，格式为[N, 4]
        scores: 置信度分数，形状为[N]
        labels: 类别标签，形状为[N]
        class_names: 类别名称列表
        score_threshold: 分数阈值
        
    Returns:
        包含检测摘要信息的字典
    """
    # 过滤低分数检测
    valid_indices = scores >= score_threshold
    valid_boxes = boxes[valid_indices] if len(boxes) > 0 else np.array([])
    valid_scores = scores[valid_indices] if len(scores) > 0 else np.array([])
    valid_labels = labels[valid_indices] if len(labels) > 0 else np.array([])
    
    # 计算检测统计信息
    detection_count = len(valid_boxes)
    
    # 按类别统计检测数量
    class_counts = {}
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    
    for label, count in zip(unique_labels, counts):
        label_idx = int(label)
        if class_names is not None and 0 <= label_idx < len(class_names):
            class_name = class_names[label_idx]
        else:
            class_name = f"类别 {label_idx}"
        
        class_counts[class_name] = int(count)
    
    # 计算平均置信度
    avg_confidence = float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0
    
    # 计算检测区域占比
    if len(valid_boxes) > 0:
        # 创建全零掩码
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 在掩码上绘制所有检测框
        for box in valid_boxes.astype(np.int32):
            x1, y1, x2, y2 = box
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)  # 填充矩形
        
        # 计算检测区域占比
        detection_area_ratio = float(np.sum(mask) / (image.shape[0] * image.shape[1]))
    else:
        detection_area_ratio = 0.0
    
    # 构建摘要字典
    summary = {
        "检测总数": detection_count,
        "类别统计": class_counts,
        "平均置信度": avg_confidence,
        "检测区域占比": detection_area_ratio
    }
    
    return summary