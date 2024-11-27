import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import pickle
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TranslationSiteDataset(Dataset):
    """翻译位点数据集类"""
    def __init__(
        self,
        sequences: List[Dict[str, torch.Tensor]],
        labels: List[torch.Tensor],
        feature_dims: Optional[Dict[str, int]] = None,
        max_cache_size: int = 1000
    ):
        """初始化数据集
        
        Args:
            sequences: 序列特征列表，每个元素是包含各种特征的字典
            labels: 标签列表
            feature_dims: 特征维度信息
            max_cache_size: 最大缓存大小
        """
        self.sequences = sequences
        self.labels = labels
        self.feature_dims = feature_dims
        self.max_cache_size = max_cache_size
        self.cache = {}
        
        # 计算数据集统计信息
        self.stats = self._calculate_stats()
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            (features, label): 特征和标签对
        """
        if idx in self.cache:
            return self.cache[idx]
        
        # 获取并处理数据
        sequence = {k: v.clone() for k, v in self.sequences[idx].items()}
        label = self.labels[idx].clone()
        
        # 缓存结果
        if len(self.cache) < self.max_cache_size:
            self.cache[idx] = (sequence, label)
        
        return sequence, label
    
    def _calculate_stats(self) -> Dict:
        """计算数据集统计信息"""
        seq_lengths = [len(s['nucleotide']) for s in self.sequences]
        
        # 统计所有位置的标签分布
        total_positions = sum(seq_lengths)
        label_counts = torch.zeros(3)  # [非特殊位点, 起始位点, 终止位点]
        
        for label in self.labels:
            unique, counts = torch.unique(label, return_counts=True)
            for u, c in zip(unique, counts):
                label_counts[u] += c
        
        # 计算比例
        label_distribution = label_counts / total_positions
        
        return {
            'num_sequences': len(self.sequences),
            'max_length': max(seq_lengths),
            'min_length': min(seq_lengths),
            'avg_length': sum(seq_lengths) / len(seq_lengths),
            'label_distribution': label_distribution,
            'total_positions': total_positions,
            'start_sites_count': int(label_counts[1]),  # 起始位点数量
            'stop_sites_count': int(label_counts[2]),   # 终止位点数量
            'sequence_counts': {
                'total': len(self.sequences),
                'with_start': sum(1 for label in self.labels if 1 in label),
                'with_stop': sum(1 for label in self.labels if 2 in label)
            }
        }
    
    def split(self, 
             train_ratio: float = 0.8,
             val_ratio: float = 0.1,
             random_seed: int = 42
             ) -> Tuple['TranslationSiteDataset', 'TranslationSiteDataset', 'TranslationSiteDataset']:
        """拆分数据集为训练集、验证集和测试集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            random_seed: 随机种子
            
        Returns:
            (train_dataset, val_dataset, test_dataset): 三个数据集
        """
        if train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")
        
        np.random.seed(random_seed)
        indices = np.random.permutation(len(self))
        
        # 计算分割点
        train_size = int(len(self) * train_ratio)
        val_size = int(len(self) * val_ratio)
        
        # 拆分索引
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # 创建数据集
        train_data = TranslationSiteDataset(
            [self.sequences[i] for i in train_indices],
            [self.labels[i] for i in train_indices],
            self.feature_dims
        )
        val_data = TranslationSiteDataset(
            [self.sequences[i] for i in val_indices],
            [self.labels[i] for i in val_indices],
            self.feature_dims
        )
        test_data = TranslationSiteDataset(
            [self.sequences[i] for i in test_indices],
            [self.labels[i] for i in test_indices],
            self.feature_dims
        )
        
        return train_data, val_data, test_data
    
    def get_feature_dims(self) -> Dict[str, int]:
        """获取特征维度信息"""
        '''
        if self.feature_dims is None:
            # 从第一个样本推断特征维度
            sample = self.sequences[0]
            self.feature_dims = {
                'nucleotide': 8,  # 固定值
                'structure': 9,   # 固定值
                'kmer': len(sample['kmer']),
                'complexity': len(sample['complexity']),
                'thermodynamic': len(sample['thermodynamic'])
            }
        '''
        return self.feature_dims
    
    def save(self, file_path: str):
        """保存数据集到文件
        
        Args:
            file_path: 保存路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'sequences': self.sequences,
            'labels': self.labels,
            'feature_dims': self.get_feature_dims(),
            'stats': self.stats
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"数据集已保存到: {file_path}")
    
    @classmethod
    def from_file(cls, file_path: str) -> 'TranslationSiteDataset':
        """从文件加载数据集
        
        Args:
            file_path: 文件路径
            
        Returns:
            TranslationSiteDataset: 加载的数据集
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return cls(
            sequences=data['sequences'],
            labels=data['labels'],
            feature_dims=data.get('feature_dims')
        )