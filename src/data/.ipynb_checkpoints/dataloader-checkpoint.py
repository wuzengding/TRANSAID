# src/data/dataloader.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]], 
               max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """自定义数据整理函数
    
    Args:
        batch: 批次数据
        max_length: 最大序列长度。如果为None，则使用batch中的最大长度
    """
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 找出这个batch中最长的序列长度
    max_len = max(len(label) for label in labels)
    if max_length is not None:
        max_len = min(max_len, max_length)
    
    # 准备batch数据
    batch_data = {}
    for key in sequences[0].keys():
        batch_data[key] = []
    
    # 填充序列
    for seq_dict in sequences:
        curr_len = len(seq_dict['nucleotide'])
        pad_len = max_len - curr_len
        
        for key, tensor in seq_dict.items():
            if key == 'nucleotide':
                # 使用0作为padding_idx
                padded = F.pad(tensor, (0, pad_len), value=0)
                batch_data[key].append(padded)
            elif tensor.dim() == 1:  # 其他一维序列特征
                padded = F.pad(tensor, (0, pad_len), value=0)
                batch_data[key].append(padded)
            elif tensor.dim() == 2:  # 二维特征
                padded = F.pad(tensor, (0, 0, 0, pad_len), value=0)
                batch_data[key].append(padded)
            else:  # 全局特征
                batch_data[key].append(tensor)
    
    # 转换为tensor并进行值检查
    batch_tensors = {}
    for k, v in batch_data.items():
        stacked = torch.stack(v)
        if k == 'nucleotide':
            # 确保索引在有效范围内
            stacked = torch.clamp(stacked, min=0, max=4)  # 0-4范围
        batch_tensors[k] = stacked
    
    # 添加标签
    batch_tensors['labels'] = torch.stack([
        F.pad(label, (0, max_len - len(label)), value=0)
        for label in labels
    ])
    
    # 创建注意力掩码
    mask = (batch_tensors['nucleotide'] != 0)
    batch_tensors['attention_mask'] = mask
    
    return batch_tensors

def create_data_loaders(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

class DataLoaderWrapper:
    """数据加载器包装类，提供额外的功能"""
    def __init__(self, 
                 dataloader: DataLoader,
                 device: str = 'cuda'):
        self.dataloader = dataloader
        self.device = device
        self.iterator = None
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        try:
            batch = next(self.iterator)
            return {k: v.to(self.device) for k, v in batch.items()}
        except StopIteration:
            self.iterator = None
            raise StopIteration
    
    def get_batch_size(self) -> int:
        """获取批次大小"""
        return self.dataloader.batch_size
    
    def get_num_batches(self) -> int:
        """获取批次数量"""
        return len(self.dataloader)