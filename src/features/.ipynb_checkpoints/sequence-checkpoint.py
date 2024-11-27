# src/features/sequence.py
import torch
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
import itertools
from .utils import normalize_sequence

class SequenceFeatureExtractor:
    """序列特征提取器"""
    def __init__(self, config):
        self.config = config
        self._init_vocabs()
        self.feature_cache = {}
        
    def _init_vocabs(self):
        """初始化词汇表"""
        self.nuc_vocab = { 'A': 1, 'T': 2, 
            'G': 3, 'C': 4
        }
        
        # 生成k-mer词汇表
        self.kmers = self._generate_kmers(self.config.kmer_sizes)
        self.kmer_vocab = {kmer: idx for idx, kmer in enumerate(self.kmers)}
        
        # 记录特征维度
        self.feature_dims = {
            'nucleotide': len(self.nuc_vocab),
            'kmer': len(self.kmer_vocab),
            'complexity': 2,  # entropy和repeat_ratio
            'gc_content': 1
        }
        
        
    def _generate_kmers(self, k_sizes: List[int]) -> List[str]:
        """生成所有可能的k-mer组合"""
        kmers = []
        bases = ['A', 'T', 'G', 'C']
        for k in k_sizes:
            kmers.extend([''.join(p) for p in itertools.product(bases, repeat=k)])
        return kmers
        
    def extract_features(self, seq: str) -> Dict[str, torch.Tensor]:
        """提取序列特征"""
        cache_key = hash(seq)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
            
        seq = normalize_sequence(seq)
        
        #if len(seq) > self.config.max_seq_length:
        #    seq = seq[:self.config.max_seq_length]
            
        features = {}
        
        features['nucleotide'] = torch.tensor(
            [self.nuc_vocab.get(n) for n in seq],
            dtype=torch.long
        )
        
        # 根据配置选择性提取其他特征
        if self.config.use_kmer:
            features['kmer'] = self._extract_kmer_features(seq)
            
        if self.config.use_complexity:
            features['complexity'] = self._calculate_sequence_complexity(seq)
            
        if self.config.use_conservation and hasattr(self, '_calculate_conservation'):
            features['conservation'] = self._calculate_conservation(seq)
        
        self.feature_cache[cache_key] = features
        
        return features
        
    def _extract_kmer_features(self, seq: str) -> torch.Tensor:
        """提取k-mer特征"""
        kmer_counts = np.zeros(len(self.kmer_vocab))
        
        for k in self.config.kmer_sizes:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if kmer in self.kmer_vocab:
                    kmer_counts[self.kmer_vocab[kmer]] += 1
                    
        if kmer_counts.sum() > 0:
            kmer_counts = kmer_counts / kmer_counts.sum()
            
        return torch.tensor(kmer_counts, dtype=torch.float32)
        
    def _calculate_sequence_complexity(self, seq: str) -> torch.Tensor:
        """计算序列复杂度"""
        freq = Counter(seq)
        total = sum(freq.values())
        entropy = -sum((count/total) * np.log2(count/total) 
                      for count in freq.values())
        
        repeats = 0
        for i in range(len(seq)-5):
            if seq[i:i+6] in seq[i+6:]:
                repeats += 1
        repeat_ratio = repeats / len(seq)
        
        return torch.tensor(
            [entropy, repeat_ratio],
            dtype=torch.float32
        )
        
    def _calculate_gc_content(self, seq: str) -> torch.Tensor:
        """计算GC含量"""
        gc_count = seq.count('G') + seq.count('C')
        gc_ratio = gc_count / len(seq)
        return torch.tensor([gc_ratio], dtype=torch.float32)
        
    def get_feature_dims(self) -> Dict[str, int]:
        """获取特征维度信息"""
        dims = {
            'nucleotide': len(self.nuc_vocab),  # 必需特征
        }
        
        if self.config.use_kmer:
            dims['kmer'] = len(self.kmer_vocab)
            
        if self.config.use_complexity:
            dims['complexity'] = 2  # entropy和repeat_ratio
            
        if self.config.use_conservation:
            dims['conservation'] = 1
            
        return dims