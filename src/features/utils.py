# src/features/utils.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_vienna_rna() -> Optional[Any]:
    """加载ViennaRNA包"""
    try:
        import RNA
        return RNA
    except ImportError:
        logger.warning("ViennaRNA package not found. Using simplified prediction.")
        return None

def normalize_sequence(seq: str) -> str:
    """标准化序列"""
    seq = seq.upper()
    seq = seq.replace('U', 'T')
    valid_chars = set('ATGCN')
    return ''.join(c if c in valid_chars else 'N' for c in seq)

def save_features(features: Dict[str, torch.Tensor], 
                 file_path: str):
    """保存特征到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    numpy_features = {
        k: v.cpu().numpy() for k, v in features.items()
    }
    
    with open(file_path, 'wb') as f:
        pickle.dump(numpy_features, f)
        
def load_features(file_path: str) -> Dict[str, torch.Tensor]:
    """从文件加载特征"""
    with open(file_path, 'rb') as f:
        numpy_features = pickle.load(f)
    
    return {
        k: torch.from_numpy(v) for k, v in numpy_features.items()
    }

def calculate_gc_content(seq: str) -> float:
    """计算GC含量"""
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq)

def find_motifs(seq: str, motifs: Dict[str, str]) -> Dict[str, List[int]]:
    """查找序列模体"""
    results = {}
    for name, pattern in motifs.items():
        positions = []
        for match in re.finditer(pattern, seq):
            positions.append(match.start())
        results[name] = positions
    return results

def sliding_window(seq: str, window_size: int, step_size: int) -> List[str]:
    """滑动窗口生成子序列"""
    windows = []
    for i in range(0, len(seq) - window_size + 1, step_size):
        windows.append(seq[i:i+window_size])
    return windows