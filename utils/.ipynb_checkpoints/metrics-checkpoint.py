# utils/metrics.py
import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import precision_recall_curve, average_precision_score
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    @staticmethod
    def calculate_metrics(predictions: torch.Tensor,
                         targets: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            predictions: 模型预测结果 (N, C)
            targets: 真实标签 (N,)
            mask: 掩码 (N,)
        
        Returns:
            Dict: 包含各种评估指标
        """
        # 将张量转换为numpy数组
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        # 计算基本指标
        metrics = {}
        
        # 准确率
        metrics['accuracy'] = np.mean(predictions.argmax(axis=-1) == targets)
        
        # 对每个类别计算指标
        for i in range(3):  # 3个类别
            # 将问题转换为二分类
            binary_preds = (predictions.argmax(axis=-1) == i).astype(np.int32)
            binary_targets = (targets == i).astype(np.int32)
            
            # 计算该类别的指标
            true_positives = np.sum((binary_preds == 1) & (binary_targets == 1))
            false_positives = np.sum((binary_preds == 1) & (binary_targets == 0))
            false_negatives = np.sum((binary_preds == 0) & (binary_targets == 1))
            
            # 计算精确率和召回率
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            
            # F1分数
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            metrics[f'class_{i}_precision'] = float(precision)
            metrics[f'class_{i}_recall'] = float(recall)
            metrics[f'class_{i}_f1'] = float(f1)
        
        # 计算宏平均
        metrics['macro_precision'] = np.mean([
            metrics[f'class_{i}_precision'] for i in range(3)
        ])
        metrics['macro_recall'] = np.mean([
            metrics[f'class_{i}_recall'] for i in range(3)
        ])
        metrics['macro_f1'] = np.mean([
            metrics[f'class_{i}_f1'] for i in range(3)
        ])
        
        return metrics

    @staticmethod
    def calculate_sequence_metrics(predictions: List[List[int]],
                                 targets: List[List[int]],
                                 tolerance: int = 10) -> Dict[str, float]:
        """计算序列级别的评估指标"""
        sequence_tp = 0
        sequence_fp = 0
        sequence_fn = 0
        
        for pred_positions, true_positions in zip(predictions, targets):
            # 对每个预测位点，检查是否在真实位点附近
            matched_positions = set()
            for pred_pos in pred_positions:
                found_match = False
                for true_pos in true_positions:
                    if abs(pred_pos - true_pos) <= tolerance:
                        if true_pos not in matched_positions:
                            sequence_tp += 1
                            matched_positions.add(true_pos)
                            found_match = True
                            break
                if not found_match:
                    sequence_fp += 1
            
            # 统计未匹配的真实位点
            sequence_fn += len(true_positions) - len(matched_positions)
        
        precision = sequence_tp / (sequence_tp + sequence_fp + 1e-10)
        recall = sequence_tp / (sequence_tp + sequence_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            'sequence_precision': precision,
            'sequence_recall': recall,
            'sequence_f1': f1
        }

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, epoch: int, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
        
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0