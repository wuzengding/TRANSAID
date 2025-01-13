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
    probs: np.ndarray  # 概率分布
    kozak_score: float  # Kozak序列得分
    sequence: str  # 周围序列
    prob_offset: int  # 添加这个字段来记录概率数组中的实际起始位置

@dataclass
class TTSFeatures:
    """TTS位点特征类"""
    position: int  # TTS在转录本中的位置
    probs: np.ndarray  # 概率分布
    sequence: str  # 周围序列
    prob_offset: int  # 添加这个字段来记录概率数组中的实际起始位置

class BayesianScorer:
    """贝叶斯评分系统"""
    
    def __init__(self, min_cds_len: int = 50):
        """
        初始化贝叶斯评分器
        
        Args:
            min_cds_len: 最小CDS长度阈值
        """
        self.min_cds_len = min_cds_len
        
        # 设置权重
        self.weights = {
            'tis_prob': 0.45,
            'tts_prob': 0.45,
            'kozak': 0.04,
            'cai': 0.04,
            'gc_score': 0.02
        }
        
        # 设置Kozak序列PWM矩阵
        self.kozak_pwm = {
            -6: {'A': 0.22, 'C': 0.28, 'G': 0.32, 'T': 0.18},
            -5: {'A': 0.20, 'C': 0.30, 'G': 0.30, 'T': 0.20},
            -4: {'A': 0.18, 'C': 0.32, 'G': 0.30, 'T': 0.20},
            -3: {'A': 0.25, 'C': 0.15, 'G': 0.45, 'T': 0.15},  # 重要位点
            -2: {'A': 0.20, 'C': 0.35, 'G': 0.25, 'T': 0.20},
            -1: {'A': 0.20, 'C': 0.35, 'G': 0.25, 'T': 0.20},
            0:  {'A': 1.00, 'C': 0.00, 'G': 0.00, 'T': 0.00},  # A of ATG
            1:  {'A': 0.00, 'C': 0.00, 'G': 0.00, 'T': 1.00},  # T of ATG
            2:  {'A': 0.00, 'C': 0.00, 'G': 1.00, 'T': 0.00},  # G of ATG
            3:  {'A': 0.20, 'C': 0.20, 'G': 0.40, 'T': 0.20}   # +4位点
        }
        
        # 设置密码子使用频率
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

    def calculate_kozak_score(self, sequence: str, tis_pos: int) -> float:
        """
        计算完整的Kozak序列得分
        
        Args:
            sequence: 完整序列
            tis_pos: TIS位置
            
        Returns:
            float: Kozak序列得分(0-1)
        """
        if tis_pos < 6 or tis_pos + 4 >= len(sequence):
            return 0.0
        
        # 提取Kozak区域 (-6 到 +4)
        kozak_region = sequence[tis_pos-6:tis_pos+4].upper()
        if len(kozak_region) != 10:
            return 0.0
        
        # 计算PWM得分
        score = 1.0
        for i, base in enumerate(kozak_region):
            pos = i - 6  # 相对于ATG的位置
            if pos in self.kozak_pwm and base in self.kozak_pwm[pos]:
                score *= self.kozak_pwm[pos][base]
        
        return score*10000
    
    def calculate_gc_score(self, sequence: str) -> float:
        """
        计算序列的GC含量
        
        Args:
            sequence: 输入序列
            
        Returns:
            float: GC含量(0-1)
        """
        if not sequence:
            return 0.0
        
        gc_count = sequence.count('G') + sequence.count('C')
        gc_content = gc_count / len(sequence)
        gc_score = 2 * np.exp(-0.5 * ((gc_content- 0.42) / 0.2) ** 2) - 1
        return gc_score
    
    def calculate_cai(self, sequence: str) -> float:
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
        return min(cai, 1.0)  # 确保CAI不超过1
    
    def score_pair(self, tis_features: TISFeatures, 
                  tts_features: TTSFeatures, 
                  sequence: str) -> Tuple[float, str]:
        """
        对TIS/TTS配对进行综合评分
        
        Args:
            tis_features: TIS特征
            tts_features: TTS特征
            sequence: 完整序列
            
        Returns:
            Tuple[float, str]: (综合评分, 过滤状态)
        """
        # 检查CDS长度
        cds_length = tts_features.position - tis_features.position + 3
        if cds_length < self.min_cds_len:
            return 0.0, f"failed_short_cds_{cds_length}bp"

        # 使用正确的偏移量计算概率得分
        tis_start = tis_features.prob_offset
        tis_end = tis_start + 3
        if tis_end > len(tis_features.probs):
            return 0.0, "failed_invalid_tis_window"
    
        tts_start = tts_features.prob_offset
        tts_end = tts_start + 3
        if tts_end > len(tts_features.probs):
            return 0.0, "failed_invalid_tts_window"
        
        # 计算TIS和TTS的概率得分
        tis_prob_score = np.mean([probs[0] for probs in tis_features.probs[tis_start:tis_end]])
        tts_prob_score = np.mean([probs[1] for probs in tts_features.probs[tts_start:tts_end]])
        
        # 计算Kozak得分
        kozak_score = tis_features.kozak_score
        
        # 提取CDS区域并计算特征
        cds = sequence[tis_features.position:tts_features.position+3]
        cai_score = self.calculate_cai(cds)
        gc_score = self.calculate_gc_score(cds)
        
        # 综合评分
        final_score = (
            self.weights['tis_prob'] * tis_prob_score +
            self.weights['tts_prob'] * tts_prob_score +
            self.weights['kozak'] * kozak_score +
            self.weights['cai'] * cai_score +
            self.weights['gc_score'] * gc_score
        )
        
        return final_score, "passed"
    
    def get_optimal_pairs(self, all_pairs: List[Tuple[TISFeatures, TTSFeatures]], 
                         sequence: str,
                         threshold: float = 0.5) -> List[Tuple[TISFeatures, TTSFeatures, float, str]]:
        """
        从所有可能的配对中选择最优的并返回得分信息
        
        Args:
            all_pairs: 所有TIS/TTS配对列表
            sequence: 完整序列
            threshold: 最小分数阈值
            
        Returns:
            List[Tuple]: [(TIS, TTS, score, filter_status)] 包含得分和过滤状态的配对列表
        """
        if not all_pairs:
            return []
        
        # 计算每个配对的得分
        scored_pairs = []
        for tis, tts in all_pairs:
            score, status = self.score_pair(tis, tts, sequence)
            scored_pairs.append((tis, tts, score, status))
        
        # 标记低分配对
        final_pairs = []
        for tis, tts, score, status in scored_pairs:
            if status == "passed" and score < threshold:
                status = f"failed_low_score_{score:.3f}"
            final_pairs.append((tis, tts, score, status))
        
        # 按得分排序
        final_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return final_pairs

class TranscriptPairAnalyzer:
    """转录本TIS/TTS配对分析器"""
    
    def __init__(self, predictions_file: str, 
                 fasta_file: str = None,
                 output_dir: str = "output",
                 prefix: str = "",
                 min_prob_threshold: float = 0.5,
                 min_cds_length: int = 50):
        """
        初始化分析器
        
        Args:
            predictions_file: 预测结果文件路径(.pkl)
            fasta_file: RNA序列文件路径(.fna)
            output_dir: 输出目录
            prefix: 文件前缀
            min_prob_threshold: 最小概率阈值
            min_cds_length: 最小CDS长度阈值
        """
        self.prefix = prefix
        self.min_prob_threshold = min_prob_threshold
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
        
        # 初始化评分器
        self.scorer = BayesianScorer(min_cds_len=min_cds_length)

    def _setup_logging(self):
        """设置日志系统"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        fh = logging.FileHandler(
            os.path.join(self.output_dir, 'analysis.log')
        )
        fh.setFormatter(formatter)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def find_potential_pairs_(self, transcript_id: str) -> List[Tuple[TISFeatures, TTSFeatures]]:
        """找出一个转录本中所有潜在的TIS/TTS配对"""
        transcript_data = None
        for item in self.predictions:
            if item['transcript_id'] == transcript_id:
                transcript_data = item
                break
        
        if not transcript_data:
            return []
        
        probs = transcript_data['predictions_probs']
        sequence = self.sequences.get(transcript_id, '')
        
        # 找出所有潜在的TIS
        tis_candidates = []
        for i in range(len(probs)-2):
            '''
            if transcript_id == "NM_001171622":
                if i ==496:
                    print(transcript_id, i, sequence[i:i+3])
                if i ==1036:
                    print(transcript_id, i, sequence[i:i+3])
                if i == 1447:
                    print(transcript_id, i, sequence[i:i+3])
            '''
            if (i+2 < len(sequence) and 
                sequence[i:i+3].upper() == 'ATG' and
                sum([probs[i+j][0] for j in range(3)]) > 0.1):
                
                prob_window = probs[max(0, i-6):min(len(probs), i+9)]
                kozak_score = self.scorer.calculate_kozak_score(sequence, i)
                
                tis_candidates.append(TISFeatures(
                    position=i,
                    probs=prob_window,
                    kozak_score=kozak_score,
                    sequence=sequence[max(0, i-6):min(len(sequence), i+9)]
                ))
        
        # 找出所有潜在的TTS
        tts_candidates = []
        for i in range(len(probs)-2):
            if (i+2 < len(sequence) and 
                sequence[i:i+3].upper() in ['TAA', 'TAG', 'TGA'] and
                sum([probs[i+j][1] for j in range(3)]) > 0.1):
                
                prob_window = probs[max(0, i-6):min(len(probs), i+9)]
                
                tts_candidates.append(TTSFeatures(
                    position=i,
                    probs=prob_window,
                    sequence=sequence[max(0, i-6):min(len(sequence), i+9)]
                ))
        
        # 生成有效配对
        valid_pairs = []
        for tis in tis_candidates:
            for tts in tts_candidates:
                if tts.position > tis.position and (tts.position - tis.position) % 3 == 0:
                    valid_pairs.append((tis, tts))
        
        return valid_pairs

    def find_potential_pairs(self, transcript_id: str) -> List[Tuple[TISFeatures, TTSFeatures]]:
        """找出一个转录本中所有潜在的TIS/TTS配对，并处理重叠情况"""
        transcript_data = None
        for item in self.predictions:
            if item['transcript_id'] == transcript_id:
                transcript_data = item
                break
        
        if not transcript_data:
            return []
        
        probs = transcript_data['predictions_probs']
        sequence = self.sequences.get(transcript_id, '')
        
        # 找出所有潜在的TIS
        tis_candidates = []
        for i in range(len(probs)-2):
            if (i+2 < len(sequence) and 
                sequence[i:i+3].upper() == 'ATG' and
                sum([probs[i+j][0] for j in range(3)]) > 0.1):

                # 计算窗口的实际范围和偏移量
                start = max(0, i-6)
                end = min(len(probs), i+9)
                prob_offset = i - start  # 计算实际偏移量
                
                prob_window = probs[start:end]
                kozak_score = self.scorer.calculate_kozak_score(sequence, i)
                
                tis_candidates.append(TISFeatures(
                    position=i,
                    probs=prob_window,
                    kozak_score=kozak_score,
                    sequence=sequence[max(0, i-6):min(len(sequence), i+9)],
                    prob_offset=prob_offset
                ))
        
        # 找出所有潜在的TTS
        tts_candidates = []
        for i in range(len(probs)-2):
            if (i+2 < len(sequence) and 
                sequence[i:i+3].upper() in ['TAA', 'TAG', 'TGA'] and
                sum([probs[i+j][1] for j in range(3)]) > 0.1):
                
                # 计算窗口的实际范围和偏移量
                start = max(0, i-6)
                end = min(len(probs), i+9)
                prob_offset = i - start  # 计算实际偏移量
                
                prob_window = probs[start:end]

                tts_candidates.append(TTSFeatures(
                    position=i,
                    probs=prob_window,
                    sequence=sequence[max(0, i-6):min(len(sequence), i+9)],
                    prob_offset=prob_offset
                ))
        
        # 生成所有可能的配对
        potential_pairs = []
        for tis in tis_candidates:
            for tts in tts_candidates:
                if tts.position > tis.position and (tts.position - tis.position) % 3 == 0:
                    potential_pairs.append((tis, tts))
        
        # 处理重叠情况
        non_overlapping_pairs = []
        
        # 按TIS位置排序
        potential_pairs.sort(key=lambda x: (x[0].position, x[1].position))
        
        while potential_pairs:
            current_pair = potential_pairs.pop(0)
            current_tis, current_tts = current_pair
            
            # 检查与剩余pairs的重叠
            overlapping_pairs = []
            non_overlapping = []
            
            for pair in potential_pairs:
                if self._is_overlapping(current_pair, pair):
                    overlapping_pairs.append(pair)
                else:
                    non_overlapping.append(pair)
            
            # 如果有重叠，选择最优的pair
            if overlapping_pairs:
                # 将当前pair加入比较
                overlapping_pairs.append(current_pair)
                best_pair = self._select_best_pair(overlapping_pairs, sequence)
                non_overlapping_pairs.append(best_pair)
            else:
                non_overlapping_pairs.append(current_pair)
            
            # 更新剩余的pairs
            potential_pairs = non_overlapping
        
        return non_overlapping_pairs
    
    def _is_overlapping(self, pair1: Tuple[TISFeatures, TTSFeatures], 
                    pair2: Tuple[TISFeatures, TTSFeatures]) -> bool:
        """检查两个TIS/TTS对是否重叠"""
        tis1, tts1 = pair1
        tis2, tts2 = pair2
        
        # 如果一个pair完全在另一个pair之前或之后，则不重叠
        return not (tts1.position < tis2.position or tis1.position > tts2.position)
    
    def _select_best_pair(self, pairs: List[Tuple[TISFeatures, TTSFeatures]], 
                        sequence: str) -> Tuple[TISFeatures, TTSFeatures]:
        """在重叠的pairs中选择最优的一个"""
        best_score = -float('inf')
        best_pair = None
        
        for pair in pairs:
            score, _ = self.scorer.score_pair(pair[0], pair[1], sequence)
            if score > best_score:
                best_score = score
                best_pair = pair
        
        return best_pair
    
    
    def analyze_transcripts(self):
        """分析所有转录本并输出结果"""
        # 基本列定义
        headers = [
            'transcript_id',
            'tis_position',
            'tts_position',
            'tis_sequence',
            'tts_sequence',
            'tis_prob_score',
            'tts_prob_score',
            'kozak_score',
            'cai_score',
            'gc_score',
            'cds_length',
            'final_score',
            'filter_status',
            'true_tis',
            'true_tts'
        ]
        
        # wrong文件的额外列
        wrong_headers = headers + ['wrong_reason']
        
        # 初始化统计计数
        total_transcripts = len(self.predictions)
        correct_transcripts = 0
        prediction_stats = {
            'multiple_predictions': 0,
            'wrong_position': 0,
            'false_positive': 0,
            'missing_prediction': 0
        }

        # 打开两个输出文件
        with open(os.path.join(self.output_dir, "_".join([self.prefix, 'tis_tts_pairs.csv'])), 'w', newline='') as f_all, \
             open(os.path.join(self.output_dir, "_".join([self.prefix, 'tis_tts_pairs_wrong.csv'])), 'w', newline='') as f_wrong:
            
            writer_all = csv.DictWriter(f_all, fieldnames=headers)
            writer_wrong = csv.DictWriter(f_wrong, fieldnames=wrong_headers)
            
            writer_all.writeheader()
            writer_wrong.writeheader()
            
            # 分析每个转录本
            for transcript_data in tqdm(self.predictions, desc="Analyzing transcripts"):
                transcript_id = transcript_data['transcript_id']
                sequence = self.sequences.get(transcript_id, '')
                
                # 获取真实TIS/TTS位置
                true_tis, true_tts = self._get_true_positions(transcript_data['true_labels'])
                
                # 找出所有可能的配对
                potential_pairs = self.find_potential_pairs(transcript_id)
                if potential_pairs is None:  # 明确检查是否为 None
                    self.logger.warning(f"No potential pairs found for transcript {transcript_id}")
                    continue
                all_rows = []
                passed_pairs = []

                '''
                if transcript_id == "NM_001171622":
                    print(potential_pairs)
                    for i, probs in enumerate(transcript_data['predictions_probs']):
                        if probs[2] < 0.95:
                            print(i,probs)
                    break
                '''    
                if not potential_pairs:
                    row = {field: '' for field in headers}
                    row['transcript_id'] = transcript_id
                    row['filter_status'] = 'failed_no_pairs'
                    writer_all.writerow(row)
                    
                    # 检查是否应该记录到wrong文件
                    if true_tis is not None or true_tts is not None:
                        wrong_row = {**row, 'true_tis': true_tis, 'true_tts': true_tts, 
                                   'wrong_reason': 'missing_prediction'}
                        writer_wrong.writerow(wrong_row)
                        prediction_stats['missing_prediction'] += 1
                    continue
                
                # 评分和过滤
                #print(transcript_id, potential_pairs)
                scored_pairs = self.scorer.get_optimal_pairs(
                    potential_pairs,
                    sequence,
                    self.min_prob_threshold
                )
                
                # 处理每个配对
                is_wrong = False
                wrong_reason = None
                
                # 记录所有配对信息并判断是否有错误
                for tis, tts, score, status in scored_pairs:
                    cds = sequence[tis.position:tts.position+3]
                    row = {
                        'transcript_id': transcript_id,
                        'tis_position': tis.position,
                        'tts_position': tts.position,
                        'tis_sequence': tis.sequence,
                        'tts_sequence': tts.sequence,
                        'tis_prob_score': np.mean([probs[0] for probs in tis.probs[tis.prob_offset:tis.prob_offset+3]]),
                        'tts_prob_score': np.mean([probs[1] for probs in tts.probs[tts.prob_offset:tts.prob_offset+3]]),
                        'kozak_score': tis.kozak_score,
                        'cai_score': self.scorer.calculate_cai(cds),
                        'gc_score': self.scorer.calculate_gc_score(cds),
                        'cds_length': len(cds),
                        'final_score': score,
                        'filter_status': status,
                        'true_tis': true_tis,
                        'true_tts': true_tts
                    }
                    all_rows.append(row)
                    writer_all.writerow(row)
                    
                    if status == 'passed':
                        passed_pairs.append((tis.position, tts.position))
                
                # 根据passed的结果判断是否需要写入wrong文件
                if true_tis is None and true_tts is None:
                    if passed_pairs:
                        is_wrong = True
                        wrong_reason = 'false_positive'
                        prediction_stats['false_positive'] += 1
                else:
                    if not passed_pairs:
                        is_wrong = True
                        wrong_reason = 'missing_prediction'
                        prediction_stats['missing_prediction'] += 1
                    elif len(passed_pairs) > 1:
                        is_wrong = True
                        wrong_reason = 'multiple_predictions'
                        prediction_stats['multiple_predictions'] += 1
                    elif len(passed_pairs) == 1 and \
                         (passed_pairs[0][0] != true_tis or passed_pairs[0][1] != true_tts):
                        is_wrong = True
                        wrong_reason = 'wrong_position'
                        prediction_stats['wrong_position'] += 1
                
                # 如果有错误，将所有结果写入wrong文件
                if is_wrong:
                    for row in all_rows:
                        wrong_row = {**row, 
                                   'wrong_reason': wrong_reason}
                        writer_wrong.writerow(wrong_row)
                else:
                    correct_transcripts += 1

        # 打印统计信息
        self.logger.info("\nPrediction Statistics:")
        self.logger.info("-" * 50)
        self.logger.info(f"Total transcripts analyzed: {total_transcripts}")
        self.logger.info(f"Correctly predicted transcripts: {correct_transcripts}")
        accuracy = (correct_transcripts / total_transcripts) * 100
        self.logger.info(f"Prediction accuracy: {accuracy:.2f}%")
        
        self.logger.info("\nError Type Distribution:")
        self.logger.info("-" * 50)
        for error_type, count in prediction_stats.items():
            percentage = (count / total_transcripts) * 100
            self.logger.info(f"{error_type}: {count} ({percentage:.2f}%)")

    def _get_true_positions(self, true_labels):
        """从true_labels中提取真实的TIS和TTS位置"""
        tis_pos = None
        tts_pos = None
        
        # 寻找连续的3个TIS标签(0)
        for i in range(len(true_labels)-2):
            if (true_labels[i] == 0 and 
                true_labels[i+1] == 0 and 
                true_labels[i+2] == 0):
                tis_pos = i
                break
        
        # 寻找连续的3个TTS标签(1)
        for i in range(len(true_labels)-2):
            if (true_labels[i] == 1 and 
                true_labels[i+1] == 1 and 
                true_labels[i+2] == 1):
                tts_pos = i
                break
        
        return tis_pos, tts_pos

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Analyze TIS/TTS pairs in transcripts')
    
    parser.add_argument('--predictions', required=True,
                      help='Path to predictions pickle file')
    parser.add_argument('--fasta', required=True,
                      help='Path to RNA sequence FASTA file')
    parser.add_argument('--output-dir', default='output',
                      help='Output directory')
    parser.add_argument('--min-prob', type=float, default=0.5,
                      help='Minimum probability threshold')
    parser.add_argument('--min-cds-len', type=int, default=50,
                      help='Minimum CDS length threshold')
    parser.add_argument('--prefix', type=str,
                       help='prefix of output file')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = TranscriptPairAnalyzer(
        predictions_file=args.predictions,
        fasta_file=args.fasta,
        output_dir=args.output_dir,
        prefix=args.prefix,
        min_prob_threshold=args.min_prob,
        min_cds_length=args.min_cds_len
    )
    
    # 执行分析
    analyzer.analyze_transcripts()

if __name__ == '__main__':
    main()