# src/features/structure.py
import torch
import numpy as np
import re
from typing import Dict, List, Optional
from .utils import load_vienna_rna

class StructureFeatureExtractor:
    """结构特征提取器"""
    def __init__(self, config):
        self.config = config
        self.vienna = load_vienna_rna() if config.use_structure else None
        self._init_structure_vocab()
        self.feature_cache = {}

        # 记录特征维度
        self.feature_dims = {
            'structure': len(self.structure_vocab),
            'thermodynamic': 1 if not self.vienna else 4,
            'local_structures': 5  # 五种局部结构特征
        }
        
    def _init_structure_vocab(self):
        """初始化结构词汇表"""
        self.structure_vocab = {
            '.': 0,  # 未配对
            '(': 1,  # 5'配对
            ')': 2,  # 3'配对
            '[': 3,  # 假结构5'端
            ']': 4,  # 假结构3'端
            '{': 5,  # G-quadruplex开始
            '}': 6,  # G-quadruplex结束
            '<': 7,  # 内部环开始
            '>': 8   # 内部环结束
        }
    
    def extract_features(self, seq: str) -> Dict[str, torch.Tensor]:
        """提取结构特征"""
        cache_key = hash(seq)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
            
        features = {}
        
        if self.config.use_structure:
            structure_info = self._predict_structure(seq)
            features['structure'] = structure_info['structure_encoding']
        
        if self.config.use_thermodynamic and self.vienna is not None:
            features.update(self._extract_vienna_features(seq, structure_info))
        
        self.feature_cache[cache_key] = features
        return features
    
    def _predict_structure(self, seq: str) -> Dict:
        """预测RNA二级结构"""
        if self.vienna is not None:
            return self._predict_with_vienna(seq)
        else:
            return self._simplified_prediction(seq)
    
    def _predict_with_vienna(self, seq: str) -> Dict:
        """使用ViennaRNA预测结构"""
        (ss, mfe) = self.vienna.fold(seq)
        fc = self.vienna.fold_compound(seq)
        fc.pf()
        bp_matrix = fc.bpp()
        
        structure_encoding = torch.tensor(
            [self.structure_vocab[s] for s in ss],
            dtype=torch.long
        )
        
        return {
            'dot_bracket': ss,
            'mfe': mfe,
            'bp_probs': bp_matrix,
            'structure_encoding': structure_encoding
        }
    
    def _simplified_prediction(self, seq: str) -> Dict:
        """简化的结构预测"""
        length = len(seq)
        structure = ['.' for _ in range(length)]
        
        for i in range(length - 4):
            if self._can_form_pair(seq[i], seq[i+4]):
                structure[i] = '('
                structure[i+4] = ')'
                
        structure_str = ''.join(structure)
        structure_encoding = torch.tensor(
            [self.structure_vocab[s] for s in structure_str],
            dtype=torch.long
        )
        
        return {
            'dot_bracket': structure_str,
            'mfe': None,
            'bp_probs': None,
            'structure_encoding': structure_encoding
        }
    
    def _can_form_pair(self, base1: str, base2: str) -> bool:
        """检查两个碱基是否能配对"""
        pairs = {
            'A': 'U', 'U': 'A',
            'G': 'C', 'C': 'G',
            'G': 'U', 'U': 'G'
        }
        return base1 in pairs and base2 == pairs[base1]
    
    def _extract_vienna_features(self, seq: str, structure_info: Dict) -> Dict:
        """提取ViennaRNA特征"""
        features = {}
        
        features['thermodynamic'] = torch.tensor(
            [structure_info['mfe'] / len(seq)],
            dtype=torch.float32
        )
        
        if structure_info['bp_probs'] is not None:
            bp_probs = structure_info['bp_probs']
            features['pairing_probs'] = torch.tensor(
                [np.mean(bp_probs), np.max(bp_probs)],
                dtype=torch.float32
            )
            
        local_structures = self._analyze_local_structures(
            structure_info['dot_bracket']
        )
        features['local_structures'] = torch.tensor(
            local_structures,
            dtype=torch.float32
        )
        
        return features
    
    def _extract_simplified_features(self, seq: str) -> Dict:
        """提取简化的结构特征"""
        features = {}
        
        gc_content = (seq.count('G') + seq.count('C')) / len(seq)
        features['thermodynamic'] = torch.tensor(
            [gc_content],
            dtype=torch.float32
        )
        
        dot_bracket = self._simplified_prediction(seq)['dot_bracket']
        local_structures = self._analyze_local_structures(dot_bracket)
        features['local_structures'] = torch.tensor(
            local_structures,
            dtype=torch.float32
        )
        
        return features
    
    def _analyze_local_structures(self, dot_bracket: str) -> List[float]:
        """分析局部结构元件"""
        total_len = len(dot_bracket)
        features = [
            dot_bracket.count('.') / total_len,  # 未配对
            dot_bracket.count('(') / total_len,  # 茎
            self._count_hairpins(dot_bracket) / total_len,  # 发夹环
            self._count_internal_loops(dot_bracket) / total_len,  # 内部环
            self._count_bulges(dot_bracket) / total_len  # 突出环
        ]
        return features
    
    def _count_hairpins(self, dot_bracket: str) -> int:
        """计算发夹环数量"""
        count = 0
        pattern = r'\([.]+\)'
        for match in re.finditer(pattern, dot_bracket):
            count += 1
        return count
    
    def _count_internal_loops(self, dot_bracket: str) -> int:
        """计算内部环数量"""
        count = 0
        pattern = r'\([.]+\([.]+\)[.]+\)'
        for match in re.finditer(pattern, dot_bracket):
            count += 1
        return count
    
    def _count_bulges(self, dot_bracket: str) -> int:
        """计算突出环数量"""
        count = 0
        pattern = r'\([.]+\)'
        for match in re.finditer(pattern, dot_bracket):
            if len(match.group(0)) > 2:
                count += 1
        return count
        
    def get_feature_dims(self) -> Dict[str, int]:
        """获取特征维度信息"""
        dims = {}
        
        if self.config.use_structure:
            dims['structure'] = len(self.structure_vocab)
        if self.config.use_thermodynamic:
            dims['thermodynamic'] = 1 if not self.vienna else 4
        return dims