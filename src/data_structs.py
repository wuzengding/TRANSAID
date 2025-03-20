# src/data_structs.py

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class SequenceFeatures:
    """序列特征数据类"""
    sequence_id: str
    sequence: str
    length: int
    gc_content: float
    features: Optional[Dict] = None

@dataclass
class ModelPrediction:
    """模型预测结果数据类"""
    sequence_id: str
    tis_probs: np.ndarray
    tts_probs: np.ndarray
    kozak_probs: np.ndarray
    
@dataclass
class ValidationMetrics:
    """验证指标数据类"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
