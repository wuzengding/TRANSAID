import argparse
import pickle
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import os, sys
from Bio import SeqIO
import logging
from scipy.stats import norm
import math
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class TISFeatures:
    """TIS位点特征类"""
    position: int  # TIS在转录本中的位置
    probs: np.ndarray  # 概率分布(-6:+9)
    kozak_score: float  # Kozak序列得分
    sequence: str  # 周围序列

@dataclass
class TTSFeatures:
    """TTS位点特征类"""
    position: int  # TTS在转录本中的位置
    probs: np.ndarray  # 概率分布(-6:+9)
    sequence: str  # 周围序列

@dataclass
class TISProbsParams:
    """TIS概率参数类，基于统计数据"""
    core_mean: float = 0.94
    core_std: float = 0.07
    core_q1: float = 0.92
    core_q3: float = 0.98
    context_mean: float = 0.0001
    context_std: float = 0.004

@dataclass
class TTSProbsParams:
    """TTS概率参数类，基于统计数据"""
    core_mean: float = 0.95
    core_std: float = 0.06
    core_q1: float = 0.95
    core_q3: float = 0.99
    context_mean: float = 0.0002
    context_std: float = 0.006

class FeatureAnalyzer:
    """特征分析工具类"""
    
    def __init__(self, window_size: int = 6):
        """
        初始化特征分析器
        
        Args:
            window_size: 上下文窗口大小
        """
        self.window_size = window_size
        self._setup_kozak_matrix()
        
    def _setup_kozak_matrix(self):
        """设置Kozak序列矩阵"""
        # 位置权重矩阵
        self.kozak_pwm = {
            -6: {'A': 0.22, 'C': 0.28, 'G': 0.32, 'T': 0.18},
            -5: {'A': 0.20, 'C': 0.30, 'G': 0.30, 'T': 0.20},
            -4: {'A': 0.18, 'C': 0.32, 'G': 0.30, 'T': 0.20},
            -3: {'A': 0.25, 'C': 0.15, 'G': 0.45, 'T': 0.15},  # 重要位点
            -2: {'A': 0.20, 'C': 0.35, 'G': 0.25, 'T': 0.20},
            -1: {'A': 0.20, 'C': 0.35, 'G': 0.25, 'T': 0.20},
            1:  {'A': 0.20, 'C': 0.20, 'G': 0.40, 'T': 0.20},  # ATG后第一个位点
            2:  {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
            3:  {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
        }

    def analyze_kozak_context(self, sequence: str, tis_pos: int) -> Dict:
        """
        分析Kozak序列上下文
        
        Args:
            sequence: 完整序列
            tis_pos: TIS位置
            
        Returns:
            Dict: 分析结果
        """
        if tis_pos < 6 or tis_pos + 4 >= len(sequence):
            return {'score': 0.0, 'strength': 'weak'}
            
        # 提取Kozak区域
        kozak_region = sequence[tis_pos-6:tis_pos+4]
        
        # 计算PWM得分
        score = self._calculate_pwm_score(kozak_region)
        
        # 评估Kozak强度
        strength = self._evaluate_kozak_strength(kozak_region)
        
        return {
            'score': score,
            'strength': strength,
            'sequence': kozak_region
        }
        
    def _calculate_pwm_score(self, kozak_seq: str) -> float:
        """计算PWM得分"""
        score = 1.0
        for i, base in enumerate(kozak_seq):
            pos = i - 6  # 相对于ATG的位置
            if pos in self.kozak_pwm and base in self.kozak_pwm[pos]:
                score *= self.kozak_pwm[pos][base]
        return score
        
    def _evaluate_kozak_strength(self, kozak_seq: str) -> str:
        """评估Kozak序列强度"""
        # 检查关键位点
        minus3 = kozak_seq[3]  # -3位点
        plus4 = kozak_seq[9]   # +4位点
        
        if minus3 in ['A', 'G'] and plus4 == 'G':
            return 'strong'
        elif minus3 in ['A', 'G'] or plus4 == 'G':
            return 'moderate'
        else:
            return 'weak'

    def analyze_cds_features(self, sequence: str, tis_pos: int, tts_pos: int) -> Dict:
        """
        分析CDS特征
        
        Args:
            sequence: 完整序列
            tis_pos: TIS位置
            tts_pos: TTS位置
            
        Returns:
            Dict: CDS特征
        """
        # 提取CDS
        cds = sequence[tis_pos:tts_pos+3]
        
        # 基本特征
        features = {
            'length': len(cds),
            'gc_content': self._calculate_gc_content(cds),
            'codon_composition': self._analyze_codon_composition(cds),
            'phase_pattern': self._analyze_phase_pattern(cds)
        }
        
        return features
        
    def _calculate_gc_content(self, sequence: str) -> float:
        """计算GC含量"""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if sequence else 0
        
    def _analyze_codon_composition(self, sequence: str) -> Dict:
        """分析密码子组成"""
        codon_count = defaultdict(int)
        for i in range(0, len(sequence)-2, 3):
            codon = sequence[i:i+3].upper()
            codon_count[codon] += 1
            
        return dict(codon_count)
        
    def _analyze_phase_pattern(self, sequence: str) -> Dict:
        """分析密码子相位模式"""
        phases = {0: [], 1: [], 2: []}
        
        for i in range(len(sequence)):
            base = sequence[i]
            phase = i % 3
            phases[phase].append(base)
            
        # 计算每个位置的碱基组成
        phase_compositions = {}
        for phase, bases in phases.items():
            composition = defaultdict(int)
            for base in bases:
                composition[base] += 1
            total = len(bases)
            # 转换为频率
            phase_compositions[phase] = {
                base: count/total for base, count in composition.items()
            }
            
        return phase_compositions

    def analyze_sequence_context(self, sequence: str, 
                               position: int,
                               window_size: int = 20) -> Dict:
        """
        分析序列上下文特征
        
        Args:
            sequence: 完整序列
            position: 目标位置
            window_size: 窗口大小
            
        Returns:
            Dict: 上下文特征
        """
        # 获取上下文序列
        start = max(0, position - window_size)
        end = min(len(sequence), position + window_size)
        context = sequence[start:end]
        
        features = {
            'upstream_seq': sequence[start:position],
            'downstream_seq': sequence[position:end],
            'upstream_gc': self._calculate_gc_content(sequence[start:position]),
            'downstream_gc': self._calculate_gc_content(sequence[position:end]),
            'local_complexity': self._calculate_sequence_complexity(context)
        }
        
        return features
        
    def _calculate_sequence_complexity(self, sequence: str) -> float:
        """计算序列复杂度"""
        if not sequence:
            return 0.0
            
        # 使用k-mer熵作为复杂度度量
        k = 3  # 使用3-mer
        kmers = defaultdict(int)
        
        for i in range(len(sequence)-k+1):
            kmer = sequence[i:i+k]
            kmers[kmer] += 1
            
        # 计算Shannon熵
        total_kmers = sum(kmers.values())
        entropy = 0.0
        for count in kmers.values():
            p = count / total_kmers
            entropy -= p * math.log2(p)
            
        return entropy

    def analyze_rna_features(self, sequence: str, tis_pos: int, tts_pos: int) -> Dict:
        """
        分析RNA相关特征
        
        Args:
            sequence: 完整序列
            tis_pos: TIS位置
            tts_pos: TTS位置
            
        Returns:
            Dict: RNA特征
        """
        features = {
            'five_utr_length': tis_pos,
            'cds_length': tts_pos - tis_pos + 3,
            'three_utr_length': len(sequence) - (tts_pos + 3),
            'five_utr_features': self.analyze_sequence_context(sequence, tis_pos),
            'three_utr_features': self.analyze_sequence_context(sequence, tts_pos)
        }
        
        return features
        
    def analyze_frame_compatibility(self, tis_pos: int, tts_pos: int) -> Dict:
        """
        分析阅读框兼容性
        
        Args:
            tis_pos: TIS位置
            tts_pos: TTS位置
            
        Returns:
            Dict: 阅读框分析结果
        """
        # 计算长度
        length = tts_pos - tis_pos + 3
        
        features = {
            'is_in_frame': length % 3 == 0,
            'frame_shift': length % 3,
            'codon_number': length // 3,
            'remaining_bases': length % 3
        }
        
        return features
        
    def generate_feature_report(self, sequence: str, tis_features, tts_features) -> Dict:
        """
        生成完整的特征分析报告
        
        Args:
            sequence: 完整序列
            tis_features: TIS特征
            tts_features: TTS特征
            
        Returns:
            Dict: 特征报告
        """
        tis_pos = tis_features.position
        tts_pos = tts_features.position
        
        report = {
            'kozak_analysis': self.analyze_kozak_context(sequence, tis_pos),
            'cds_features': self.analyze_cds_features(sequence, tis_pos, tts_pos),
            'rna_features': self.analyze_rna_features(sequence, tis_pos, tts_pos),
            'frame_analysis': self.analyze_frame_compatibility(tis_pos, tts_pos),
            'tis_context': self.analyze_sequence_context(sequence, tis_pos),
            'tts_context': self.analyze_sequence_context(sequence, tts_pos)
        }
        
        return report

class BayesianScorer:
    """贝叶斯评分系统"""
    
    def __init__(self, tis_params, tts_params, codon_usage: Dict[str, float]):
        """
        初始化贝叶斯评分器
        
        Args:
            tis_params: TIS概率参数
            tts_params: TTS概率参数
            codon_usage: 密码子使用频率字典
        """
        self.tis_params = tis_params
        self.tts_params = tts_params
        self.codon_usage = codon_usage
        
        # 设置权重
        self.weights = {
            'tis_prob': 0.3,
            'tts_prob': 0.3,
            'kozak': 0.15,
            'cds_len': 0.1,
            'cai': 0.15
        }
        
        # 设置CDS长度分布参数
        self.cds_length_params = {
            'mean': 1000,  # 平均CDS长度
            'std': 500    # 标准差
        }

    def score_pair(self, tis_features, tts_features, sequence: str) -> float:
        """
        对TIS/TTS配对进行综合评分
        
        Args:
            tis_features: TIS特征
            tts_features: TTS特征
            sequence: 完整序列
            
        Returns:
            float: 综合评分(0-1)
        """
        # 计算各个组分的得分
        tis_prob_score = self._score_tis_probs(tis_features.probs)
        tts_prob_score = self._score_tts_probs(tts_features.probs)
        kozak_score = tis_features.kozak_score
        
        # 计算CDS特征
        cds_start = tis_features.position
        cds_end = tts_features.position + 3
        cds = sequence[cds_start:cds_end]
        
        cds_len_score = self._score_cds_length(len(cds))
        cai_score = self._calculate_cai(cds)
        
        # 综合评分
        final_score = (
            self.weights['tis_prob'] * tis_prob_score +
            self.weights['tts_prob'] * tts_prob_score +
            self.weights['kozak'] * kozak_score +
            self.weights['cds_len'] * cds_len_score +
            self.weights['cai'] * cai_score
        )
        
        return final_score

    def _score_tis_probs(self, probs: np.ndarray) -> float:
        """
        评估TIS概率分布得分
        
        Args:
            probs: TIS位点概率分布
            
        Returns:
            float: 概率评分(0-1)
        """
        if len(probs) < 3:
            return 0.0
            
        # 核心区域得分
        core_probs = probs[:3]  # 取前三个位点概率
        core_score = self._score_distribution(
            core_probs,
            self.tis_params.core_mean,
            self.tis_params.core_std
        )
        
        # 上下文区域得分
        context_probs = np.concatenate([probs[3:], probs[:-3]])
        context_score = self._score_distribution(
            context_probs,
            self.tis_params.context_mean,
            self.tis_params.context_std
        )
        
        # 综合得分
        return 0.7 * core_score + 0.3 * context_score

    def _score_tts_probs(self, probs: np.ndarray) -> float:
        """
        评估TTS概率分布得分
        
        Args:
            probs: TTS位点概率分布
            
        Returns:
            float: 概率评分(0-1)
        """
        if len(probs) < 3:
            return 0.0
            
        # 核心区域得分
        core_probs = probs[:3]  # 取前三个位点概率
        core_score = self._score_distribution(
            core_probs,
            self.tts_params.core_mean,
            self.tts_params.core_std
        )
        
        # 上下文区域得分
        context_probs = np.concatenate([probs[3:], probs[:-3]])
        context_score = self._score_distribution(
            context_probs,
            self.tts_params.context_mean,
            self.tts_params.context_std
        )
        
        # 综合得分
        return 0.7 * core_score + 0.3 * context_score

    def _score_distribution(self, values: np.ndarray, 
                          expected_mean: float, 
                          expected_std: float) -> float:
        """
        基于正态分布评估概率值
        
        Args:
            values: 概率值数组
            expected_mean: 期望均值
            expected_std: 期望标准差
            
        Returns:
            float: 分布评分(0-1)
        """
        # 计算与期望分布的匹配度
        z_scores = np.abs((values - expected_mean) / expected_std)
        score = np.mean(norm.cdf(-z_scores))  # 转换为0-1分数
        return score

    def _score_cds_length(self, length: int) -> float:
        """
        评估CDS长度得分
        
        Args:
            length: CDS长度
            
        Returns:
            float: 长度评分(0-1)
        """
        # 使用对数正态分布评估长度
        log_length = np.log(length)
        log_mean = np.log(self.cds_length_params['mean'])
        log_std = np.log(1 + self.cds_length_params['std']**2 / 
                        self.cds_length_params['mean']**2)
        
        z_score = abs((log_length - log_mean) / log_std)
        score = norm.cdf(-z_score)
        
        return score

    def _calculate_cai(self, sequence: str) -> float:
        """
        计算密码子适应指数(CAI)
        
        Args:
            sequence: CDS序列
            
        Returns:
            float: CAI值(0-1)
        """
        if len(sequence) < 3:
            return 0.0
            
        # 计算每个密码子的权重
        w_values = []
        for i in range(0, len(sequence)-2, 3):
            codon = sequence[i:i+3].upper()
            if codon in self.codon_usage:
                w_values.append(np.log(self.codon_usage[codon]))
                
        if not w_values:
            return 0.0
            
        # 计算几何平均值
        cai = np.exp(np.mean(w_values))
        return cai

    def get_optimal_pairs(self, all_pairs: List[Tuple], 
                         sequence: str,
                         threshold: float = 0.5) -> List[Tuple]:
        """
        从所有可能的配对中选择最优的
        
        Args:
            all_pairs: 所有TIS/TTS配对列表
            sequence: 完整序列
            threshold: 最小分数阈值
            
        Returns:
            List[Tuple]: 最优配对列表
        """
        # 计算每个配对的得分
        scored_pairs = []
        for tis, tts in all_pairs:
            score = self.score_pair(tis, tts, sequence)
            if score >= threshold:
                scored_pairs.append((tis, tts, score))
        
        # 按得分排序
        scored_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 解决重叠问题并返回最优配对
        return self._select_non_overlapping_pairs(scored_pairs)

    def _select_non_overlapping_pairs(self, 
                                    scored_pairs: List[Tuple]) -> List[Tuple]:
        """
        选择不重叠的最优配对
        
        Args:
            scored_pairs: 带得分的配对列表
            
        Returns:
            List[Tuple]: 选定的配对列表
        """
        if not scored_pairs:
            return []
            
        selected = []
        used_positions = set()
        
        for tis, tts, score in scored_pairs:
            # 检查是否有重叠
            overlap = False
            for pos in range(tis.position, tts.position + 3):
                if pos in used_positions:
                    overlap = True
                    break
            
            if not overlap:
                selected.append((tis, tts))
                # 标记已使用的位置
                used_positions.update(
                    range(tis.position, tts.position + 3)
                )
        
        return selected

class TranscriptPairAnalyzer:
    """转录本TIS/TTS配对分析器"""
    
    def __init__(self, predictions_file: str, 
                 fasta_file: str = None,
                 gbff_file: str = None,
                 species: str = "human",
                 output_dir: str = "output",
                 min_prob_threshold: float = 0.5):
        """
        初始化分析器
        
        Args:
            predictions_file: 预测结果文件路径(.pkl)
            fasta_file: RNA序列文件路径(.fna)
            gbff_file: 基因组注释文件路径(.gbff)
            species: 物种选择("human" or "mouse")
            output_dir: 输出目录
            min_prob_threshold: 最小概率阈值
        """
        self.min_prob_threshold = min_prob_threshold
        self.species = species
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 加载预测数据
        self.logger.info("Loading prediction data...")
        with open(predictions_file, 'rb') as f:
            self.predictions = pickle.load(f)
        
        # 加载序列数据(如果提供)
        self.sequences = {}
        if fasta_file:
            self.logger.info("Loading sequence data...")
            for record in SeqIO.parse(fasta_file, "fasta"):
                self.sequences[record.id.split('.')[0]] = str(record.seq)
        
        # 初始化统计参数
        self.tis_params = TISProbsParams()
        self.tts_params = TTSProbsParams()
        
        # 设置密码子使用频率
        self._setup_codon_usage()

    def _setup_logging(self):
        """设置日志系统"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器 - 记录所有级别
        fh = logging.FileHandler(
            os.path.join(self.output_dir, 'analysis.log')
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        
        # 控制台处理器 - 只显示INFO以上级别
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _setup_codon_usage(self):
        """设置物种特异的密码子使用频率"""
        if self.species == "human":
            self.codon_usage = {
                'TTT': 0.45, 'TTC': 0.55, 'TTA': 0.07, 'TTG': 0.13,
                'TCT': 0.18, 'TCC': 0.22, 'TCA': 0.15, 'TCG': 0.06,
                'TAT': 0.43, 'TAC': 0.57, 'TAA': 0.28, 'TAG': 0.20,
                'TGT': 0.45, 'TGC': 0.55, 'TGA': 0.52, 'TGG': 1.00,
                'CTT': 0.13, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 0.41,
                'CCT': 0.28, 'CCC': 0.33, 'CCA': 0.27, 'CCG': 0.11,
                'CAT': 0.41, 'CAC': 0.59, 'CAA': 0.25, 'CAG': 0.75,
                'CGT': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21,
                'ATT': 0.36, 'ATC': 0.48, 'ATA': 0.16, 'ATG': 1.00,
                'ACT': 0.24, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.12,
                'AAT': 0.46, 'AAC': 0.54, 'AAA': 0.42, 'AAG': 0.58,
                'AGT': 0.15, 'AGC': 0.24, 'AGA': 0.20, 'AGG': 0.20,
                'GTT': 0.18, 'GTC': 0.24, 'GTA': 0.11, 'GTG': 0.47,
                'GCT': 0.26, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11,
                'GAT': 0.46, 'GAC': 0.54, 'GAA': 0.42, 'GAG': 0.58,
                'GGT': 0.16, 'GGC': 0.34, 'GGA': 0.25, 'GGG': 0.25
            }
        elif self.species == "mouse":
            # TODO: 添加小鼠的密码子使用频率
            self.codon_usage = {}
        else:
            raise ValueError(f"Unsupported species: {self.species}")

    def find_potential_pairs(self, transcript_id: str) -> List[Tuple[TISFeatures, TTSFeatures]]:
        """
        找出一个转录本中所有潜在的TIS/TTS配对
        
        Args:
            transcript_id: 转录本ID
            
        Returns:
            List of (TIS, TTS) feature pairs
        """
        # 获取转录本数据
        transcript_data = self._get_transcript_data(transcript_id)
        if not transcript_data:
            return []
            
        # 找出所有潜在的TIS和TTS
        tis_candidates = self._find_tis_candidates(transcript_data)
        tts_candidates = self._find_tts_candidates(transcript_data)
        
        # 生成有效的配对
        valid_pairs = self._generate_valid_pairs(tis_candidates, tts_candidates)
        
        # 处理重叠情况
        final_pairs = self._resolve_overlaps(valid_pairs)
        
        return final_pairs

    def _get_transcript_data(self, transcript_id: str) -> Dict:
        """获取转录本数据"""
        for item in self.predictions:
            if item['transcript_id'] == transcript_id:
                return item
        return None

    def _find_tis_candidates(self, transcript_data: Dict) -> List[TISFeatures]:
        """找出所有潜在的TIS位点"""
        candidates = []
        probs = transcript_data['predictions_probs']
        sequence = self.sequences.get(transcript_data['transcript_id'], '')
        
        for i in range(len(probs)-2):
            # 检查是否为潜在TIS(ATG)
            if (i+2 < len(sequence) and 
                sequence[i:i+3].upper() == 'ATG' and
                sum([probs[i+j][0] for j in range(3)]) >0.1 ):
                # 获取周围区域的概率
                prob_window = self._get_prob_window(probs, i, window_size=6)
                # 计算Kozak得分
                kozak_score = self._calculate_kozak_score(sequence, i)
                
                candidates.append(TISFeatures(
                    position=i,
                    probs=prob_window,
                    kozak_score=kozak_score,
                    sequence=sequence[i-6:i+9] if i >= 6 else sequence[:i+9]
                ))
                
        return candidates

    def _find_tts_candidates(self, transcript_data: Dict) -> List[TTSFeatures]:
        """找出所有潜在的TTS位点"""
        candidates = []
        probs = transcript_data['predictions_probs']
        sequence = self.sequences.get(transcript_data['transcript_id'], '')
        
        for i in range(len(probs)-2):
            # 检查是否为潜在TTS(TAA/TAG/TGA)
            if (i+2 < len(sequence) and 
                sequence[i:i+3].upper() in ['TAA', 'TAG', 'TGA'] and
                sum([probs[i+j][1] for j in range(3)]) >0.1 ):
                # 获取周围区域的概率
                prob_window = self._get_prob_window(probs, i, window_size=6)
                
                candidates.append(TTSFeatures(
                    position=i,
                    probs=prob_window,
                    sequence=sequence[i-6:i+9] if i >= 6 else sequence[:i+9]
                ))
                
        return candidates

    def _get_prob_window(self, probs: np.ndarray, center: int, 
                        window_size: int = 6) -> np.ndarray:
        """获取指定位置周围的概率窗口"""
        start = max(0, center - window_size)
        end = min(len(probs), center + window_size + 3)
        return probs[start:end]

    def _calculate_kozak_score(self, sequence: str, tis_pos: int) -> float:
        """计算Kozak序列得分"""
        if tis_pos < 6 or tis_pos + 4 >= len(sequence):
            return 0.0
            
        kozak_region = sequence[tis_pos-6:tis_pos+4]
        kozak_weights = {
            -3: {'A': 0.8, 'G': 1.0, 'C': 0.4, 'T': 0.4},
            +4: {'G': 1.0, 'A': 0.8, 'C': 0.4, 'T': 0.4}
        }
        
        score = 1.0
        # Check -3 position
        score *= kozak_weights[-3].get(kozak_region[3], 0.2)
        # Check +4 position
        score *= kozak_weights[+4].get(kozak_region[9], 0.2)
        
        return score

    def _generate_valid_pairs(self, tis_candidates: List[TISFeatures],
                            tts_candidates: List[TTSFeatures]) -> List[Tuple[TISFeatures, TTSFeatures]]:
        """生成有效的TIS/TTS配对"""
        valid_pairs = []
        
        for tis in tis_candidates:
            for tts in tts_candidates:
                # 检查基本条件
                if (tts.position > tis.position and  # TTS在TIS之后
                    (tts.position - tis.position) % 3 == 0):  # 长度是3的倍数
                    valid_pairs.append((tis, tts))
                    
        return valid_pairs

    def _resolve_overlaps(self, pairs: List[Tuple[TISFeatures, TTSFeatures]]) -> List[Tuple[TISFeatures, TTSFeatures]]:
        """解决重叠配对"""
        if not pairs:
            return []
            
        # 按TIS位置排序
        pairs = sorted(pairs, key=lambda x: x[0].position)
        
        resolved_pairs = []
        current_group = [pairs[0]]
        
        for i in range(1, len(pairs)):
            current_pair = pairs[i]
            last_pair = current_group[-1]
            
            # 检查是否有重叠
            if self._has_overlap(last_pair, current_pair):
                # 处理重叠情况
                if current_pair[0] == last_pair[0]:  # 共同TIS
                    # 保留靠5'端的TTS
                    if current_pair[1].position < last_pair[1].position:
                        current_group[-1] = current_pair
                else:  # 其他重叠情况
                    current_group.append(current_pair)
            else:
                # 无重叠，开始新组
                resolved_pairs.extend(current_group)
                current_group = [current_pair]
        
        # 添加最后一组
        resolved_pairs.extend(current_group)
        
        return resolved_pairs

    def _has_overlap(self, pair1: Tuple[TISFeatures, TTSFeatures],
                    pair2: Tuple[TISFeatures, TTSFeatures]) -> bool:
        """检查两个配对是否有重叠"""
        tis1, tts1 = pair1
        tis2, tts2 = pair2
        
        # 检查是否有重叠区域
        return not (tts1.position < tis2.position or tis1.position > tts2.position)

    def _save_results(self, results: Dict):
        """保存分析结果"""
        # 保存为pickle文件
        output_file = os.path.join(self.output_dir, 'tis_tts_pairs.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
            
        # 生成统计报告
        self._generate_report(results)
        
    def _generate_report(self, results: Dict):
        """生成分析报告"""
        report_file = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("TIS/TTS Pair Analysis Report\n")
            f.write("==========================\n\n")
            
            # 基本统计
            total_transcripts = len(results)
            transcripts_with_pairs = sum(1 for r in results.values() 
                                    if r['optimal_pairs'])
            
            f.write(f"Total transcripts analyzed: {total_transcripts}\n")
            f.write(f"Transcripts with valid pairs: {transcripts_with_pairs}\n")
            f.write(f"Success rate: {transcripts_with_pairs/total_transcripts*100:.2f}%\n\n")
            
            # 转录本类型统计
            transcript_types = defaultdict(int)
            for transcript_id in results:
                t_type = transcript_id.split('_')[0]
                transcript_types[t_type] += 1
                
            f.write("Transcript Type Distribution:\n")
            f.write("--------------------------\n")
            for t_type, count in transcript_types.items():
                f.write(f"{t_type}: {count} ")
                f.write(f"({count/total_transcripts*100:.2f}%)\n")
            f.write("\n")
            
            # 配对统计
            total_pairs = sum(len(r['optimal_pairs']) for r in results.values())
            avg_pairs = total_pairs / transcripts_with_pairs if transcripts_with_pairs > 0 else 0
            
            f.write("Pair Statistics:\n")
            f.write("---------------\n")
            f.write(f"Total pairs found: {total_pairs}\n")
            f.write(f"Average pairs per transcript: {avg_pairs:.2f}\n")
    
            # 详细的CSV报告
            self._generate_csv_report(results)

    def _extract_true_positions(self, transcript_data: Dict) -> Tuple[int, int]:
        """
        从true_labels中提取真实的TIS和TTS位置
        
        Args:
            transcript_data: 转录本数据
            
        Returns:
            Tuple[int, int]: (TIS位置, TTS位置)
        """
        true_labels = transcript_data['true_labels']
        
        # 找到连续的TIS (label=0)
        tis_pos = -1
        for i in range(len(true_labels)-2):
            if (true_labels[i] == 0 and 
                true_labels[i+1] == 0 and 
                true_labels[i+2] == 0):
                tis_pos = i
                break
        
        # 找到连续的TTS (label=1)
        tts_pos = -1
        for i in range(len(true_labels)-2):
            if (true_labels[i] == 1 and 
                true_labels[i+1] == 1 and 
                true_labels[i+2] == 1):
                tts_pos = i
                break
                
        return tis_pos, tts_pos
    
    def _generate_csv_report(self, results: Dict):
        """生成CSV格式的详细报告"""
        output_file = os.path.join(self.output_dir, 'tis_tts_pairs.csv')
        
        headers = [
            'transcript_id',
            'tis_position',
            'tts_position',
            'TIS_True',
            'TTS_True',
            'tis_sequence',
            'tts_sequence',
            'tis_prob_score',
            'tts_prob_score',
            'kozak_score',
            'cds_length',
            'is_in_frame'
        ]
        
        match_count = 0
        total_count = 0
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for transcript_id, result in tqdm(results.items(), 
                                            desc="Generating CSV report"):
                # 获取原始数据以提取真实位置
                transcript_data = self._get_transcript_data(transcript_id)
                if not transcript_data:
                    continue
                    
                true_tis, true_tts = self._extract_true_positions(transcript_data)
                
                for tis, tts in result['optimal_pairs']:
                    total_count += 1
                    # 检查是否与真实位置匹配
                    is_tis_match = tis.position == true_tis
                    is_tts_match = tts.position == true_tts
                    
                    if is_tis_match and is_tts_match:
                        match_count += 1
                    
                    row = {
                        'transcript_id': transcript_id,
                        'tis_position': tis.position,
                        'tts_position': tts.position,
                        'TIS_True': true_tis,
                        'TTS_True': true_tts,
                        'tis_sequence': tis.sequence,
                        'tts_sequence': tts.sequence,
                        'tis_prob_score': np.mean(tis.probs[:3]),
                        'tts_prob_score': np.mean(tts.probs[:3]),
                        'kozak_score': tis.kozak_score,
                        'cds_length': tts.position - tis.position + 3,
                        'is_in_frame': (tts.position - tis.position + 3) % 3 == 0
                    }
                    writer.writerow(row)
        
        # 记录统计信息
        self.logger.info(f"Total pairs analyzed: {total_count}")
        self.logger.info(f"Matching pairs: {match_count}")
        self.logger.info(f"Matching rate: {match_count/total_count*100:.2f}%")

    def analyze_transcripts(self) -> Dict:
        """
        分析所有转录本中的TIS/TTS配对
        
        Returns:
            Dict: 分析结果
        """
        results = {}
        scorer = BayesianScorer(self.tis_params, self.tts_params, self.codon_usage)
        
        # 使用tqdm包装predictions迭代
        for item in tqdm(self.predictions, desc="Analyzing transcripts"):
            transcript_id = item['transcript_id']
            #self.logger.debug(f"Analyzing transcript: {transcript_id}")  # 改为debug级别
            
            # 找出所有可能的配对
            potential_pairs = self.find_potential_pairs(transcript_id)
            if not potential_pairs:
                self.logger.debug(f"No valid pairs found for {transcript_id}")
                continue
                
            # 获取序列
            sequence = self.sequences.get(transcript_id, '')
            if not sequence:
                self.logger.debug(f"No sequence found for {transcript_id}")
                continue
                
            # 选择最优配对
            optimal_pairs = scorer.get_optimal_pairs(
                potential_pairs,
                sequence,
                self.min_prob_threshold
            )
            
            # 保存结果
            results[transcript_id] = {
                'optimal_pairs': optimal_pairs,
                'all_pairs': potential_pairs,
                'sequence_length': len(sequence)
            }
            
        # 保存分析结果
        self._save_results(results)
        
        return results
        
class ResultVisualizer:
    """结果可视化类"""
    
    def __init__(self, output_dir: str):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self._setup_plot_style()

    def _setup_plot_style(self):
        """设置绘图风格"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        try:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        except:
            sns.set_style("darkgrid")  
        
        
        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

    def plot_results(self, results: Dict):
        """
        生成结果可视化
        
        Args:
            results: 分析结果字典
        """
        self._plot_pair_distribution(results)
        self._plot_length_distribution(results)
        self._plot_pair_scores(results)
        self._plot_feature_importance(results)
        
    def _plot_pair_distribution(self, results: Dict):
        """绘制配对分布图"""
        import matplotlib.pyplot as plt
        
        # 统计每个转录本的配对数
        pair_counts = [len(r['optimal_pairs']) for r in results.values()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(pair_counts, bins='auto', density=True, alpha=0.7)
        plt.xlabel('Number of TIS/TTS pairs per transcript')
        plt.ylabel('Density')
        plt.title('Distribution of TIS/TTS Pairs per Transcript')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, 'pair_distribution.png'))
        plt.close()
        
    def _plot_length_distribution(self, results: Dict):
        """绘制长度分布图"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 获取CDS长度
        cds_lengths = []
        for r in results.values():
            for tis, tts in r['optimal_pairs']:
                length = tts.position - tis.position + 3
                cds_lengths.append(length)
        
        plt.figure(figsize=(10, 6))
        
        # 使用对数刻度
        plt.hist(np.log10(cds_lengths), bins='auto', density=True, alpha=0.7)
        plt.xlabel('log10(CDS Length)')
        plt.ylabel('Density')
        plt.title('Distribution of CDS Lengths')
        plt.grid(True, alpha=0.3)
        
        # 添加原始刻度
        ax = plt.gca()
        ticks = ax.get_xticks()
        ax.set_xticklabels([f'{10**x:.0f}' for x in ticks])
        
        plt.savefig(os.path.join(self.output_dir, 'length_distribution.png'))
        plt.close()
        
    def _plot_pair_scores(self, results: Dict):
        """绘制配对得分分布"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 收集TIS和TTS的概率得分
        tis_scores = []
        tts_scores = []
        
        for r in results.values():
            for tis, tts in r['optimal_pairs']:
                tis_scores.append(np.mean(tis.probs[:3]))
                tts_scores.append(np.mean(tts.probs[:3]))
        
        plt.figure(figsize=(12, 5))
        
        # TIS得分分布
        plt.subplot(1, 2, 1)
        sns.kdeplot(data=tis_scores, fill=True)
        plt.xlabel('TIS Score')
        plt.ylabel('Density')
        plt.title('TIS Score Distribution')
        
        # TTS得分分布
        plt.subplot(1, 2, 2)
        sns.kdeplot(data=tts_scores, fill=True)
        plt.xlabel('TTS Score')
        plt.ylabel('Density')
        plt.title('TTS Score Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_distribution.png'))
        plt.close()
        
    def _plot_feature_importance(self, results: Dict):
        """绘制特征重要性图"""
        import matplotlib.pyplot as plt
        
        # 特征权重
        features = {
            'TIS Probability': 0.3,
            'TTS Probability': 0.3,
            'Kozak Sequence': 0.15,
            'CDS Length': 0.1,
            'CAI Score': 0.15
        }
        
        plt.figure(figsize=(10, 6))
        
        # 创建条形图
        plt.bar(features.keys(), features.values())
        plt.xlabel('Features')
        plt.ylabel('Weight')
        plt.title('Feature Importance in Scoring Model')
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.close()


def save_feature_report(report: Dict, output_file: str):
    """
    保存特征分析报告
    
    Args:
        report: 特征报告字典
        output_file: 输出文件路径
    """
    with open(output_file, 'w') as f:
        f.write("Feature Analysis Report\n")
        f.write("=====================\n\n")
        
        # Kozak分析
        f.write("Kozak Sequence Analysis:\n")
        f.write("-----------------------\n")
        kozak = report['kozak_analysis']
        f.write(f"Score: {kozak['score']:.4f}\n")
        f.write(f"Strength: {kozak['strength']}\n")
        f.write(f"Sequence: {kozak['sequence']}\n\n")
        
        # CDS特征
        f.write("CDS Features:\n")
        f.write("-------------\n")
        cds = report['cds_features']
        f.write(f"Length: {cds['length']}bp\n")
        f.write(f"GC content: {cds['gc_content']*100:.2f}%\n\n")
        
        # RNA特征
        f.write("RNA Features:\n")
        f.write("-------------\n")
        rna = report['rna_features']
        f.write(f"5'-UTR length: {rna['five_utr_length']}bp\n")
        f.write(f"CDS length: {rna['cds_length']}bp\n")
        f.write(f"3'-UTR length: {rna['three_utr_length']}bp\n\n")
        
        # 阅读框分析
        f.write("Reading Frame Analysis:\n")
        f.write("----------------------\n")
        frame = report['frame_analysis']
        f.write(f"In frame: {frame['is_in_frame']}\n")
        f.write(f"Number of codons: {frame['codon_number']}\n")
        f.write(f"Remaining bases: {frame['remaining_bases']}\n")

'''
def validate_parameters(args):
    """验证输入参数"""
    # 检查文件是否存在
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")
    
    if args.fasta and not os.path.exists(args.fasta):
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    
    if args.gbff and not os.path.exists(args.gbff):
        raise FileNotFoundError(f"GBFF file not found: {args.gbff}")
    
    # 检查概率阈值
    if not 0 <= args.min_prob <= 1:
        raise ValueError("Minimum probability threshold must be between 0 and 1")
    
    # 检查线程数
    if args.threads < 1:
        raise ValueError("Number of threads must be positive")
'''
def setup_logging():
    """设置全局日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='Analyze TIS/TTS pairs in transcripts')
    
    # 输入文件
    parser.add_argument('--predictions', required=True,
                      help='Path to predictions pickle file')
    parser.add_argument('--fasta',
                      help='Path to RNA sequence FASTA file')
    parser.add_argument('--gbff',
                      help='Path to genome annotation GBFF file')
                      
    # 分析参数
    parser.add_argument('--min-score', type=float, default=0.5,
                      help='Minimum score threshold')
    parser.add_argument('--species', choices=['human', 'mouse'],
                      default='human', help='Species selection')
                      
    # 输出控制
    parser.add_argument('--output-dir', default='output',
                      help='Output directory')
    parser.add_argument('--plot', action='store_true',
                      help='Generate visualization plots')
                      
    # 性能参数
    parser.add_argument('--threads', type=int, default=1,
                      help='Number of threads for parallel processing')
    
    args = parser.parse_args()
    
    # 创建分析器 - 修改这里
    analyzer = TranscriptPairAnalyzer(
        predictions_file=args.predictions,
        fasta_file=args.fasta,
        gbff_file=args.gbff,
        species=args.species,
        output_dir=args.output_dir,
        min_prob_threshold=args.min_score
    )
    
    # 执行分析
    results = analyzer.analyze_transcripts()
    
    # 如果需要，生成可视化
    if args.plot:
        visualizer = ResultVisualizer(args.output_dir)
        visualizer.plot_results(results)


if __name__ == '__main__':
    # 设置日志
    setup_logging()
    
    try:
        main()
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        sys.exit(1)
        