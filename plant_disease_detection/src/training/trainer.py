# src/training/trainer.py


import os
import sys
import time
import logging
import random
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback
from src.evaluation.metrics import accuracy, precision, recall, f1, confusion_matrix, plot_confusion_matrix
from src.evaluation.detection_metrics import calculate_metrics_per_class

class EarlyStopping:
    """早停机制类，监控指标停止改善时提前终止训练"""
    
    def __init__(self, patience=7, delta=0, min_delta=None, mode='min'):
        """初始化早停"""
        self.patience = patience
        # 添加兼容性处理
        self.delta = min_delta if min_delta is not None else delta
        self.mode = mode.lower()
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, metrics: Dict[str, float], model: nn.Module, path: str) -> bool:
        """
        检查是否应该停止训练
        
        Args:
            metrics: 当前的验证指标字典
            model: 当前模型
            path: 保存模型的路径
            
        Returns:
            是否应该停止训练
        """
        # 安全类型检查
        if not isinstance(metrics, dict):
            logging.warning(f"早停接收到非字典类型的指标: {type(metrics)}，将使用该值作为监控指标")
            value = float(metrics)  # 尝试转换为浮点数
            metrics = {"value": value}  # 创建临时字典
            monitor = "value"
        else:
            # 从配置中获取监控指标
            monitor = getattr(self, 'monitor', 'loss')
            
            # 如果指定的监控指标不存在，尝试使用任务特定的默认值
            if not monitor or monitor not in metrics:
                if hasattr(self, 'task_type') and self.task_type == 'detection':
                    monitor = 'mAP' if 'mAP' in metrics else 'loss'
                else:
                    monitor = 'loss' if 'loss' in metrics else next(iter(metrics.keys()))
                
                logging.info(f"指定的监控指标不可用，使用 '{monitor}' 作为替代")
                    
        value = metrics.get(monitor, 0.0)
        
        # mAP和准确率等指标是越高越好，其他如损失是越低越好
        if self.mode == 'max' or monitor in ['mAP', 'acc', 'accuracy', 'f1', 'precision', 'recall']:
            score = value  # 高指标值更好
        else:
            score = -value  # 低指标值更好
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(value, model, path)
        elif score <= self.best_score + self.delta:  # 没有改善或改善太小
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # 有显著改善
            self.best_score = score
            self.save_checkpoint(value, model, path)
            self.counter = 0
            
        return self.early_stop
    
    def save_checkpoint(self, value: float, model: nn.Module, path: str):
        """保存检查点"""
        # 使用正确的指标值进行日志记录
        metric_name = getattr(self, 'monitor', 'loss')
        logging.info(f'Validation {metric_name} improved from {self.val_loss_min:.6f} to {value:.6f}. Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = value
    
    def is_best(self, value: float) -> bool:
        """
        检查是否是最佳模型
        
        Args:
            value: 当前的验证指标值
            
        Returns:
            当前模型是否是最佳模型
        """
        if self.mode == 'min':
            return value < self.val_loss_min
        else:
            return value > self.val_loss_min

class TensorboardLogger:
    """TensorBoard日志记录工具类"""
    
    def __init__(self, log_dir: str):
        """
        初始化TensorBoard
        
        Args:
            log_dir: 日志目录
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        记录标量值
        
        Args:
            tag: 数据标签
            value: 标量值
            step: 步数
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """
        记录多个标量值
        
        Args:
            tag: 数据标签组前缀
            values: 标量值字典
            step: 步数
        """
        for key, value in values.items():
            self.writer.add_scalar(f'{tag}/{key}', value, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int):
        """
        记录图像
        
        Args:
            tag: 图像标签
            image: 图像数组，格式为CHW或NCHW
            step: 步数
        """
        self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """
        记录直方图
        
        Args:
            tag: 直方图标签
            values: 数值数组
            step: 步数
        """
        self.writer.add_histogram(tag, values, step)
    
    def log_pr_curve(self, tag: str, labels: np.ndarray, predictions: np.ndarray, step: int):
        """
        记录PR曲线
        
        Args:
            tag: PR曲线标签
            labels: 真实标签数组
            predictions: 预测概率数组
            step: 步数
        """
        self.writer.add_pr_curve(tag, labels, predictions, step)
    
    def log_figure(self, tag: str, figure, step: int):
        """
        记录matplotlib图表
        
        Args:
            tag: 图表标签
            figure: matplotlib图表对象
            step: 步数
        """
        self.writer.add_figure(tag, figure, step)
    
    def close(self):
        """关闭TensorBoard"""
        self.writer.close()

class WarmupScheduler:
    """学习率预热调度器装饰器"""
    
    def __init__(self, scheduler, warmup_epochs: int, warmup_factor: float = 0.1):
        """
        初始化预热调度器
        
        Args:
            scheduler: 基础学习率调度器
            warmup_epochs: 预热的epoch数
            warmup_factor: 预热起始学习率因子
        """
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.base_lrs = scheduler.base_lrs if hasattr(scheduler, 'base_lrs') else scheduler._get_base_lrs()
        self.current_epoch = 0
        
    def step(self):
        """更新学习率"""
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # 预热阶段
            factor = self.warmup_factor + (1 - self.warmup_factor) * (self.current_epoch / self.warmup_epochs)
            for param_group, base_lr in zip(self.scheduler.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * factor
        else:
            # 正常调度
            self.scheduler.step()
    
    def get_lr(self):
        """
        获取当前学习率
        
        Returns:
            当前各参数组的学习率列表
        """
        return [group['lr'] for group in self.scheduler.optimizer.param_groups]
        
    # 新增方法：保存调度器状态
    def state_dict(self):
        """返回调度器状态"""
        return {
            'scheduler_state': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else {},
            'warmup_epochs': self.warmup_epochs,
            'warmup_factor': self.warmup_factor,
            'current_epoch': self.current_epoch,
            'base_lrs': self.base_lrs
        }
    
    # 新增方法：加载调度器状态
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        if hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_factor = state_dict['warmup_factor']
        self.current_epoch = state_dict['current_epoch']
        self.base_lrs = state_dict['base_lrs']

class Trainer:
    """通用训练器，支持分类和检测任务"""
    
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 config: Dict[str, Any],
                 device: str = 'cuda',
                 task_type: str = 'classification',
                 class_names: List[str] = None):  # 添加 class_names 参数
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            config: 训练配置字典
            device: 训练设备
            task_type: 任务类型，'classification'或'detection'
            class_names: 类别名称列表
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.task_type = task_type
        self.class_names = class_names  # 保存类别名称
        
        # 配置学习率调度器
        self.scheduler = None
        if config.get('scheduler', None):
            scheduler_type = config['scheduler']['type']
            # 使用 get 方法并提供默认空字典，避免 KeyError
            scheduler_args = config['scheduler'].get('args', {})
            
            # 处理直接传入调度器实例的情况
            if 'instance' in config['scheduler'] and config['scheduler']['instance'] is not None:
                self.scheduler = config['scheduler']['instance']
            elif scheduler_type == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, **scheduler_args)
                logging.info(f"使用CosineAnnealingLR学习率调度器，参数: {scheduler_args}")
            elif scheduler_type == 'MultiStepLR':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, **scheduler_args)
            elif scheduler_type == 'ReduceLROnPlateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **scheduler_args)
            elif scheduler_type == 'OneCycleLR':
                # 使用OneCycleLR配置
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, **scheduler_args)
            
            # 添加预热
            if config['scheduler'].get('warmup_epochs', 0) > 0:
                self.scheduler = WarmupScheduler(
                    self.scheduler, 
                    config['scheduler']['warmup_epochs'],
                    config['scheduler'].get('warmup_factor', 0.1)
                )
        
        # 配置早停
        self.early_stopping = None
        if config.get('early_stopping', None):
            early_stopping_config = config['early_stopping'].copy()
            # 确保保存监控指标名称
            self.early_stopping_monitor = early_stopping_config.get('monitor', 'loss')
            if 'monitor' in early_stopping_config:
                early_stopping_config.pop('monitor')  # 移除不兼容的参数
            self.early_stopping = EarlyStopping(**early_stopping_config)
        
        # 配置TensorBoard
        self.tb_logger = None
        if config.get('tensorboard', {}).get('enabled', False):
            log_dir = config['tensorboard'].get('log_dir', 'runs')
            self.tb_logger = TensorboardLogger(log_dir)
        
        # 配置混合精度训练
        self.use_amp = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 初始化训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        # 将模型移动到指定设备
        self.model.to(self.device)
    
    def train(self, epochs: int, save_dir: str) -> Dict[str, List[float]]:
        """
        执行训练流程
        
        Args:
            epochs: 训练的总epoch数
            save_dir: 保存模型的目录
            
        Returns:
            包含训练和验证指标历史的字典
        """
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        
        logging.info("开始训练...")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self._train_epoch(epoch)
            self.train_metrics_history.append(train_metrics)
            
            # 验证一个epoch
            val_metrics = self._validate_epoch(epoch)
            self.val_metrics_history.append(val_metrics)
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau需要指标值
                    monitor_value = val_metrics.get(self.early_stopping_monitor, 0) if self.early_stopping_monitor else val_metrics.get('loss', 0)
                    self.scheduler.step(monitor_value)
                # 新增：排除OneCycleLR，因为它应该在每个batch后调用step()
                elif not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    # 其他调度器（如CosineAnnealingLR, StepLR）在每个epoch结束时调用step()
                    self.scheduler.step()
                    # 记录当前学习率
                    current_lrs = self._get_lr()
                    logging.info(f"Epoch {epoch+1}学习率更新为: {[f'{lr:.6f}' for lr in current_lrs]}")
            
            # 保存检查点
            self.save_checkpoint(epoch, save_dir)
            
            # 检查是否应该早停 - 修复:传入完整的验证指标字典
            if self.early_stopping is not None:
                # 始终传递完整的指标字典，而不是单个值
                if self.early_stopping(val_metrics, self.model, best_model_path):
                    logging.info(f"早停触发，在epoch {epoch+1}停止训练")
                    break
        
        # 关闭TensorBoard
        if self.tb_logger is not None:
            self.tb_logger.close()
        
        return {
            'train_metrics': self.train_metrics_history,
            'val_metrics': self.val_metrics_history
        }
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        metrics = {'loss': 0.0, 'samples': 0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # 根据任务类型选择不同的训练方法
            if self.task_type == 'classification':
                # 直接调用分类批次训练函数，在其内部会调用_prepare_batch
                batch_metrics = self._train_classification_batch(batch)
            else:
                # 直接调用检测批次训练函数，不使用_prepare_batch
                batch_metrics = self._train_detection_batch(batch)
            
            # 更新指标
            batch_size = batch_metrics.pop('batch_size', batch_metrics.get('samples', 0))
            self._update_metrics(metrics, batch_metrics, batch_size)
            
            # 更新进度条
            pbar_postfix = {k: f'{v / metrics["samples"]:.4f}' if k != 'samples' else v 
                           for k, v in metrics.items()}
            pbar.set_postfix(pbar_postfix)
            
            self.global_step += 1

            # 如果使用OneCycleLR，需要在每个批次后调用step()
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
        
        # 计算平均指标
        avg_metrics = {k: v / metrics['samples'] for k, v in metrics.items() if k != 'samples'}
        
        # 记录指标到TensorBoard
        if self.tb_logger is not None:
            self._log_metrics(avg_metrics, epoch, prefix='train/')
            self._log_lr(epoch)
        
        logging.info(f"Epoch {epoch+1}: " + ", ".join([f"train_{k}: {v:.4f}" for k, v in avg_metrics.items()]))
        
        return avg_metrics
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        metrics = {'loss': 0.0, 'samples': 0}
        
        # 新增: 用于收集分类评估指标的列表
        all_preds_clf = []
        all_targets_clf = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.get('epochs', '?')} [Val]")
            
            # 保留一些可视化样本收集的代码，但不再为计算mAP收集数据
            vis_sample_images = []
            vis_sample_targets = []
            vis_sample_outputs = []
            vis_samples_collected = 0
            max_vis_samples = 5
            
            for batch in pbar:
                # 根据任务类型选择不同的验证方法
                if self.task_type == 'classification':
                    # 分类任务的处理逻辑保持不变
                    batch_metrics = self._validate_classification_batch(batch)
                    
                    # 更新指标
                    batch_size = batch_metrics.pop('batch_size', batch_metrics.get('samples', 0))
                    self._update_metrics(metrics, batch_metrics, batch_size)
                    
                    # 新增: 收集分类任务的预测和目标
                    images, labels = self._prepare_batch(batch)
                    outputs = self.model(images)
                    _, preds = torch.max(outputs, 1)
                    all_preds_clf.extend(preds.cpu().numpy())
                    all_targets_clf.extend(labels.cpu().numpy())
                else:  # 检测任务 (现在评估其分类能力)
                    images, targets = batch
                    images = [img.to(self.device) for img in images]
                    
                    # 模型推理
                    outputs = self.model(images)  # eval模式返回检测结果列表
                    
                    # 新增: 提取分类预测和目标
                    for i in range(len(outputs)):
                        output = outputs[i]
                        true_label = targets[i]['labels'][0].item()  # 获取真实图像级标签
                        all_targets_clf.append(true_label)
                        
                        if len(output['scores']) > 0:
                            top_idx = torch.argmax(output['scores'])
                            pred_label = output['labels'][top_idx].item()
                            all_preds_clf.append(pred_label)
                        else:
                            all_preds_clf.append(-1)  # 标记未预测
                    
                    # 可视化样本收集仍然保留
                    for i in range(len(outputs)):
                        if vis_samples_collected < max_vis_samples:
                            vis_sample_images.append(images[i].cpu())
                            vis_sample_targets.append({
                                'boxes': targets[i]['boxes'].cpu(),
                                'labels': targets[i]['labels'].cpu()
                            })
                            vis_sample_outputs.append({
                                'boxes': outputs[i]['boxes'].cpu(),
                                'scores': outputs[i]['scores'].cpu(),
                                'labels': outputs[i]['labels'].cpu()
                            })
                            vis_samples_collected += 1
                    
                    # 更新样本计数
                    batch_size = len(images)
                    metrics['samples'] += batch_size
                    
                    # 更新进度条
                    pbar_postfix = {}
                    for k, v in metrics.items():
                        if k == 'samples':
                            pbar_postfix[k] = v
                        else:
                            # 安全计算平均值
                            avg_value = v / metrics['samples'] if metrics['samples'] > 0 else 0
                            pbar_postfix[k] = f'{avg_value:.4f}'
                    
                    pbar.set_postfix(pbar_postfix)
            
            # 计算平均指标
            avg_metrics = {k: v / metrics['samples'] if metrics['samples'] > 0 else 0.0
                         for k, v in metrics.items() if k != 'samples'}
            
            # 新增: 计算并添加分类指标
            if all_targets_clf:  # 确保有数据
                all_preds_clf = np.array(all_preds_clf)
                all_targets_clf = np.array(all_targets_clf)
                valid_mask_clf = all_preds_clf != -1
                if not np.all(valid_mask_clf):
                    all_preds_clf = all_preds_clf[valid_mask_clf]
                    all_targets_clf = all_targets_clf[valid_mask_clf]
                
                if len(all_preds_clf) > 0:
                    avg_metrics['accuracy'] = accuracy(all_targets_clf, all_preds_clf)
                    # 计算macro F1更鲁棒
                    avg_metrics['f1_score'] = f1(all_targets_clf, all_preds_clf, average='macro', zero_division=0)
                    # 添加precision和recall
                    avg_metrics['precision'] = precision(all_targets_clf, all_preds_clf, average='macro', zero_division=0)
                    avg_metrics['recall'] = recall(all_targets_clf, all_preds_clf, average='macro', zero_division=0)
                    
                    logging.info(f"验证分类指标: Accuracy={avg_metrics['accuracy']:.4f}, F1={avg_metrics['f1_score']:.4f}")
                else:
                    avg_metrics['accuracy'] = 0.0
                    avg_metrics['f1_score'] = 0.0
                    avg_metrics['precision'] = 0.0
                    avg_metrics['recall'] = 0.0
            
            # 仍然保留可视化功能，但不再计算mAP
            if self.task_type == 'detection' and vis_samples_collected > 0 and hasattr(self, '_visualize_detection_results'):
                self._visualize_detection_results(
                    vis_sample_images[:max_vis_samples],
                    vis_sample_targets[:max_vis_samples],
                    vis_sample_outputs[:max_vis_samples],
                    epoch
                )
            
            # 记录指标到TensorBoard
            if self.tb_logger is not None:
                self._log_metrics(avg_metrics, epoch, prefix='val')
            
            # 修改日志输出，重点展示分类指标
            log_metrics = []
            for k, v in avg_metrics.items():
                if k not in ['accuracy', 'f1_score', 'precision', 'recall', 'loss']:
                    continue
                log_metrics.append(f"val_{k}: {v:.4f}")
            
            logging.info(f"Epoch {epoch+1}: " + ", ".join(log_metrics))
            
            return avg_metrics
    
    def _validate_classification_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        验证一个分类任务批次
        
        Args:
            batch: 包含'images'和'labels'的字典
            
        Returns:
            批次验证指标字典
        """
        images, labels = self._prepare_batch(batch)
        
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        # 计算准确率
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        acc = correct / labels.size(0)
        
        return {
            'loss': loss.item(),
            'acc': acc,
            'batch_size': images.size(0)
        }
    
    def _train_classification_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        训练一个分类任务批次
        
        Args:
            batch: 包含'images'和'labels'的字典
            
        Returns:
            批次训练指标字典
        """
        images, labels = self._prepare_batch(batch)
        
        # 添加 MixUp/CutMix 逻辑
        apply_mixup = False
        apply_cutmix = False
        lam = 1.0  # 默认不混合
        labels_a, labels_b = labels, labels
        
        # 检查配置以确定是否应用 MixUp/CutMix
        mixup_alpha = self.config.get('mixup_alpha', 0.0)
        if isinstance(self.config.get('train'), dict):
            mixup_alpha = self.config.get('train', {}).get('mixup_alpha', mixup_alpha)
        
        cutmix_alpha = self.config.get('cutmix_alpha', 0.0)
        if isinstance(self.config.get('train'), dict):
            cutmix_alpha = self.config.get('train', {}).get('cutmix_alpha', cutmix_alpha)
        
        # 随机决定是否应用 MixUp 或 CutMix
        if mixup_alpha > 0 and random.random() < 0.5:
            apply_mixup = True
            from src.utils.augmentation import AugmentationPipeline
            images, labels_a, labels_b, lam = AugmentationPipeline.mixup_data(images, labels, mixup_alpha)
        elif cutmix_alpha > 0:
            apply_cutmix = True
            from src.utils.augmentation import AugmentationPipeline
            images, labels_a, labels_b, lam = AugmentationPipeline.cutmix_data(images, labels, cutmix_alpha)

        # 清零梯度
        self.optimizer.zero_grad()
        
        # 混合精度训练
        if self.use_amp:
            # 确保指定device_type
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device_type):
                outputs = self.model(images)
                
                # 根据是否应用混合增强计算损失
                if apply_mixup or apply_cutmix:
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)
                
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip', None):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 常规训练
            outputs = self.model(images)
            
            # 根据是否应用混合增强计算损失
            if apply_mixup or apply_cutmix:
                loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
            else:
                loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
                
            # 更新参数
            self.optimizer.step()
        
        # 计算准确率（只有当未应用混合增强时才有意义）
        if not apply_mixup and not apply_cutmix:
            _, preds = torch.max(outputs, 1)
            correct = (preds == labels).sum().item()
            acc = correct / labels.size(0)
        else:
            # 当应用混合增强时，准确率计算变得复杂，可以简单地返回0
            acc = 0.0
        
        return {
            'loss': loss.item(),
            'acc': acc,
            'batch_size': images.size(0)
        }
    
    def _train_detection_batch(self, batch):
        """训练一个检测任务批次"""
        # 获取梯度累积步数
        grad_accumulation_steps = self.config.get('grad_accumulation_steps', 1)
        should_update = (self.global_step % grad_accumulation_steps == 0)
        
        try:
            # 直接将batch视为元组解包，因为detection_collate_fn返回的就是元组
            images, targets = batch
            
            # 添加 CutMix 逻辑（只适用于检测任务中的图像混合）
            apply_cutmix = False
            cutmix_alpha = self.config.get('cutmix_alpha', 0.0)
            if isinstance(self.config.get('train'), dict):
                cutmix_alpha = self.config.get('train', {}).get('cutmix_alpha', cutmix_alpha)
            
            mixcut_prob = self.config.get('mixcut_prob', 0.5)
            if isinstance(self.config.get('train'), dict):
                mixcut_prob = self.config.get('train', {}).get('mixcut_prob', mixcut_prob)
            
            # CutMix 只应用于图像，保持目标不变（简化实现）
            if cutmix_alpha > 0 and random.random() < mixcut_prob:
                apply_cutmix = True
                
                # 将图像移到设备上进行混合
                if isinstance(images[0], torch.Tensor):
                    device_images = [img.to(self.device) for img in images]
                    stacked_images = torch.stack(device_images)
                    dummy_labels = torch.zeros(len(images), dtype=torch.long).to(self.device)
                    
                    # 导入并应用 CutMix
                    from src.utils.augmentation import AugmentationPipeline
                    mixed_images, _, _, _ = AugmentationPipeline.cutmix_data(stacked_images, dummy_labels, cutmix_alpha)
                    
                    # 将混合后的图像拆分回列表
                    images = list(torch.unbind(mixed_images))
            
            # 将图像和目标移动到设备上
            if isinstance(images[0], torch.Tensor):
                images = [img.to(self.device) for img in images]
            else:
                logging.warning(f"预期images元素为Tensor，获取到{type(images[0])}")
                
            if isinstance(targets[0], dict):
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in t.items()} for t in targets]
            else:
                logging.warning(f"预期targets元素为字典，获取到{type(targets[0])}")
            
            # 清零梯度
            if should_update:
                self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.use_amp:
                # 获取设备类型
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with autocast(device_type=device_type):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                self.scaler.scale(losses).backward()
                
                if should_update:
                    # 梯度裁剪
                    if 'grad_clip' in self.config and self.config['grad_clip'] > 0:
                        self.scaler.unscale_(self.optimizer)  # 修正：添加下划线
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config['grad_clip']
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # 正常训练
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # 反向传播
                losses.backward()
                
                if should_update:
                    # 梯度裁剪
                    if 'grad_clip' in self.config and self.config['grad_clip'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config['grad_clip']
                        )
                    
                    # 更新参数
                    self.optimizer.step()
            
            # 更新全局步数
            self.global_step += 1
            
            # 转换损失为Python数值
            batch_metrics = {k: v.item() for k, v in loss_dict.items()}
            batch_metrics['loss'] = losses.item()
            batch_metrics['batch_size'] = len(images)
            
            if apply_cutmix:
                batch_metrics['cutmix_applied'] = 1.0
            
            return batch_metrics
            
        except Exception as e:
            logging.error(f"训练批次时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个空指标字典
            return {'loss': 0.0, 'batch_size': 0}
    
    def _validate_detection_batch(self, batch):
        """验证一个检测任务批次"""
        try:
            # 直接将batch视为元组解包
            images, targets = batch
            
            # 将图像和目标移动到设备上
            if isinstance(images[0], torch.Tensor):
                images = [img.to(self.device) for img in images]
            else:
                logging.warning(f"预期images元素为Tensor，获取到{type(images[0])}")
                
            if isinstance(targets[0], dict):
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in t.items()} for t in targets]
            else:
                logging.warning(f"预期targets元素为字典，获取到{type(targets[0])}")
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(images, targets)
                
                # 添加类型检查
                if isinstance(outputs, dict):
                    # 如果是损失字典，计算总损失
                    losses = sum(loss for loss in outputs.values())
                    # 转换损失为Python数值
                    batch_metrics = {k: v.item() for k, v in outputs.items()}
                    batch_metrics['loss'] = losses.item()
                elif isinstance(outputs, list):
                    # 如果是检测结果列表，计算一些评估指标
                    # 但在验证阶段，我们暂时只需前向传播不需要计算损失
                    batch_metrics = {'loss': 0.0}
                else:
                    logging.warning(f"未预期的输出类型: {type(outputs)}")
                    batch_metrics = {'loss': 0.0}
                
                batch_metrics['batch_size'] = len(images)
                
                return batch_metrics
                
        except Exception as e:
            logging.error(f"验证批次时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个空指标字典
            return {'loss': 0.0, 'batch_size': 0}
    
    def save_checkpoint(self, epoch: int, save_dir: str, is_best: bool = False):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            save_dir: 保存目录
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': self.train_metrics_history,
            'val_metrics': self.val_metrics_history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # 每隔一定epoch保存一次检查点
        if self.config.get('save_freq', 0) > 0 and (epoch + 1) % self.config['save_freq'] == 0:
            epoch_path = os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
            logging.info(f"保存epoch检查点到 {epoch_path}")
        
        # 如果是最佳模型，另存一份
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), best_path)
            logging.info(f"保存最佳模型到 {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        logging.info(f"从 {checkpoint_path} 加载检查点")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练状态
        self.current_epoch = checkpoint.get('epoch', -1) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.train_metrics_history = checkpoint.get('train_metrics', [])
        self.val_metrics_history = checkpoint.get('val_metrics', [])
        
        logging.info(f"加载检查点成功，从epoch {self.current_epoch} 继续训练")
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """
        记录指标到TensorBoard
        
        Args:
            metrics: 指标字典
            step: 步数
            prefix: 指标名称前缀
        """
        if self.tb_logger is None:
            return
        
        for name, value in metrics.items():
            tag = f'{prefix}/{name}' if prefix else name
            self.tb_logger.log_scalar(tag, value, step)
    
    def _log_lr(self, step: int):
        """
        记录学习率
        
        Args:
            step: 步数
        """
        if self.tb_logger is None:
            return
        
        # 获取当前学习率
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        # 如果只有一个学习率，直接记录
        if len(lrs) == 1:
            self.tb_logger.log_scalar('lr', lrs[0], step)
        else:
            # 否则记录每个参数组的学习率
            for i, lr in enumerate(lrs):
                name = f'backbone' if i == 0 else f'head'  # 假设第一组是backbone，第二组是head
                self.tb_logger.log_scalar(f'lr/{name}', lr, step)
    
    def _log_images(self, images: torch.Tensor, targets: torch.Tensor, 
                   predictions: torch.Tensor, step: int):
        """
        记录样本图像到TensorBoard
        
        Args:
            images: 图像张量
            targets: 目标张量
            predictions: 预测张量
            step: 步数
        """
        if self.tb_logger is None:
            return
            
        # 只记录部分样本
        n_samples = min(4, images.size(0))
        
        for i in range(n_samples):
            self.tb_logger.log_image(f'samples/{i}/image', images[i].cpu().numpy(), step)
            
            if self.task_type == 'classification':
                # 分类任务，记录标签和预测
                target = targets[i].item()
                pred = torch.argmax(predictions[i]).item()
                self.tb_logger.log_scalar(f'samples/{i}/target', target, step)
                self.tb_logger.log_scalar(f'samples/{i}/prediction', pred, step)
            elif self.task_type == 'detection':
                # 检测任务，这里需要可视化检测结果
                # 由于实现复杂，这里只是示例
                pass
    
    def _warmup_lr(self, epoch: int):
        """
        学习率预热
        
        Args:
            epoch: 当前epoch
        """
        # 如果使用WarmupScheduler，则不需要额外处理
        pass
    
    def _get_lr(self):
        """
        获取当前学习率
        
        Returns:
            当前学习率列表
        """
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预处理批次数据
        
        Args:
            batch: 包含'images'和'labels'的字典
            
        Returns:
            处理后的(images, labels)元组
        """
        images = batch['images'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        return images, labels
    
    def _update_metrics(self, metrics: Dict[str, float], batch_metrics: Dict[str, float], 
                       batch_size: int):
        """
        更新度量指标字典
        
        Args:
            metrics: 累积指标字典
            batch_metrics: 批次指标字典
            batch_size: 批次大小
        """
        # 确保批次大小有效
        if batch_size <= 0:
            batch_size = 1  # 默认至少有1个样本
        
        for k, v in batch_metrics.items():
            if k in metrics:
                metrics[k] += v * batch_size
            else:
                metrics[k] = v * batch_size
                
        metrics['samples'] += batch_size

    # 在trainer.py中添加检测结果可视化方法
    def _visualize_detection_results(self, images, targets, outputs, epoch, max_images=5):
        """
        可视化检测结果
        
        Args:
            images: 图像列表
            targets: 真实标注
            outputs: 模型输出
            epoch: 当前epoch
            max_images: 最大图像数
        """
        if self.tb_logger is None or not hasattr(self.tb_logger, 'writer'):
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # 最多只处理max_images张图像
            n_images = min(len(images), max_images)
            
            for i in range(n_images):
                # 转换图像到numpy用于绘图
                img = images[i].permute(1, 2, 0).cpu().numpy()
                # 标准化到[0, 1]范围
                if img.max() > 1.0:
                    img = img / 255.0
                
                fig, ax = plt.subplots(1, figsize=(12, 9))
                ax.imshow(img)
                
                # 绘制真实边界框
                if 'boxes' in targets[i] and len(targets[i]['boxes']) > 0:
                    boxes = targets[i]['boxes'].cpu()
                    labels = targets[i]['labels'].cpu()
                    
                    for box, label in zip(boxes, labels):
                        x1, y1, x2, y2 = box.tolist()
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                              edgecolor='g', facecolor='none')
                        ax.add_patch(rect)
                        
                        # 添加标签
                        class_name = str(label.item())
                        if hasattr(self, 'class_names') and self.class_names and label.item() < len(self.class_names):
                            class_name = self.class_names[label.item()]
                        ax.text(x1, y1, class_name, bbox=dict(facecolor='green', alpha=0.5))
                
                # 绘制预测边界框
                if 'boxes' in outputs[i] and len(outputs[i]['boxes']) > 0:
                    pred_boxes = outputs[i]['boxes'].cpu()
                    pred_labels = outputs[i]['labels'].cpu()
                    pred_scores = outputs[i]['scores'].cpu()
                    
                    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                        # 过滤低置信度的预测
                        if score < 0.5:
                            continue
                            
                        x1, y1, x2, y2 = box.tolist()
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                              edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        
                        # 添加标签和置信度
                        class_name = str(label.item())
                        if hasattr(self, 'class_names') and self.class_names and label.item() < len(self.class_names):
                            class_name = self.class_names[label.item()]
                        ax.text(x1, y2+10, f'{class_name}: {score:.2f}', 
                                bbox=dict(facecolor='red', alpha=0.5))
                
                plt.axis('off')
                plt.tight_layout()
                
                # 记录到TensorBoard
                self.tb_logger.writer.add_figure(f'detection_results/epoch_{epoch}/image_{i}', fig, epoch)
                plt.close(fig)
        except Exception as e:
            logging.error(f"可视化检测结果时出错: {e}")
            traceback.print_exc()

    def _visualize_detection_results(self, images, targets, outputs, epoch, max_images=5):
        """
        可视化检测结果
        
        Args:
            images: 图像列表
            targets: 真实标注
            outputs: 模型输出
            epoch: 当前epoch
            max_images: 最大图像数
        """
        if self.tb_logger is None or not hasattr(self.tb_logger, 'writer'):
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # 最多只处理max_images张图像
            n_images = min(len(images), max_images)
            
            for i in range(n_images):
                # 转换图像到numpy用于绘图
                img = images[i].permute(1, 2, 0).cpu().numpy()
                
                # 正确归一化图像数据
                if img.min() < 0 or img.max() > 1.0:
                    # 如果是标准化后的图像（均值0），先转回[0,1]范围
                    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
                
                fig, ax = plt.subplots(1, figsize=(12, 9))
                ax.imshow(img)
                
                # 绘制真实边界框和预测框的代码保持不变...
                # ...
                
                plt.tight_layout()
                plt.axis('off')
                
                # 记录到TensorBoard
                self.tb_logger.writer.add_figure(f'detection_results/epoch_{epoch}/image_{i}', fig, epoch)
                plt.close(fig)
        except Exception as e:
            logging.error(f"可视化检测结果时出错: {e}")
            traceback.print_exc()