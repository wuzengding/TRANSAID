import numpy as np
from typing import List, Tuple, Dict
import math

class TISPredictor:
    def __init__(self, sequence: str, softmax_probs: np.ndarray, prob_threshold: float = 0.05):
        """
        初始化预测器
        
        Args:
            sequence: 转录本序列
            softmax_probs: 每个位置的概率值 shape=(seq_len, 3)
            prob_threshold: 概率阈值
        """
        self.sequence = sequence.upper()
        self.probs = softmax_probs
        self.prob_threshold = prob_threshold
        self.seq_len = len(sequence)
        
        # 标准Kozak序列PWM
        self.kozak_pwm = {
            -6: {'G':0.3, 'A':0.2, 'C':0.3, 'T':0.2},
            -5: {'G':0.3, 'A':0.2, 'C':0.3, 'T':0.2}, 
            -4: {'C':0.5, 'G':0.3, 'A':0.1, 'T':0.1},
            -3: {'G':0.6, 'A':0.3, 'C':0.1, 'T':0.0},
            -2: {'C':0.6, 'G':0.2, 'A':0.1, 'T':0.1},
            -1: {'C':0.6, 'G':0.2, 'A':0.1, 'T':0.1},
            1:  {'G':0.5, 'A':0.3, 'C':0.1, 'T':0.1},
            2:  {'G':0.3, 'A':0.3, 'C':0.2, 'T':0.2},
            3:  {'G':0.3, 'A':0.3, 'C':0.2, 'T':0.2}
        }
        
        # 密码子使用频率表(人类)
        self.codon_usage = self._load_codon_usage()
    
    def find_potential_pairs(self) -> List[Tuple[int, int]]:
        """找到所有潜在的TIS-TTS对"""
        potential_pairs = []
        
        # 找到所有可能的TIS位点
        tis_positions = self._find_potential_tis()
        # 找到所有可能的TTS位点
        tts_positions = self._find_potential_tts()
        
        # 组合配对并检查生物学规则
        for tis_pos in tis_positions:
            for tts_pos in tts_positions:
                if self._check_pair_validity(tis_pos, tts_pos):
                    potential_pairs.append((tis_pos, tts_pos))
        
        return potential_pairs
    
    def _find_potential_tis(self) -> List[int]:
        """找到所有可能的TIS位点"""
        tis_positions = []
        
        for i in range(0, self.seq_len-2):
            # 检查序列是否为ATG
            if self.sequence[i:i+3] == 'ATG':
                # 检查连续3个位置的概率值
                probs = [self.probs[j][0] for j in range(i, i+3)]
                # 判断是否有至少2个位置概率值超过阈值
                if sum(prob >= self.prob_threshold for prob in probs) >= 2:
                    tis_positions.append(i)
        
        return tis_positions
    
    def _find_potential_tts(self) -> List[int]:
        """找到所有可能的TTS位点"""
        tts_positions = []
        stop_codons = {'TAA', 'TAG', 'TGA'}
        
        for i in range(0, self.seq_len-2):
            # 检查序列是否为终止密码子
            if self.sequence[i:i+3] in stop_codons:
                # 检查TTS概率是否超过阈值
                if self.probs[i][1] >= self.prob_threshold:
                    tts_positions.append(i)
        
        return tts_positions
    
    def _check_pair_validity(self, tis_pos: int, tts_pos: int) -> bool:
        """检查TIS-TTS对是否符合生物学规则"""
        # 1. TIS必须在TTS之前
        if tis_pos >= tts_pos:
            return False
            
        # 2. 距离必须是3的整数倍
        if (tts_pos - tis_pos) % 3 != 0:
            return False
            
        # 3. 中间不能有提前终止密码子
        if self._check_premature_stop(tis_pos, tts_pos):
            return False
            
        return True
    
    def _check_premature_stop(self, tis_pos: int, tts_pos: int) -> bool:
        """检查是否存在提前终止密码子"""
        stop_codons = {'TAA', 'TAG', 'TGA'}
        
        for i in range(tis_pos+3, tts_pos, 3):
            if self.sequence[i:i+3] in stop_codons:
                return True
        return False




class BayesianEvaluator:
    def __init__(self, sequence: str, softmax_probs: np.ndarray):
        self.sequence = sequence
        self.probs = softmax_probs
        
    def evaluate_pairs(self, potential_pairs: List[Tuple[int, int]], 
                      confidence_threshold: float = 0.8) -> Dict:
        """评估所有潜在的TIS-TTS对"""
        evaluated_pairs = {}
        
        for tis_pos, tts_pos in potential_pairs:
            # 计算特征向量
            features = self._calculate_features(tis_pos, tts_pos)
            
            # 计算后验概率
            posterior_prob = self._calculate_posterior(features)
            
            if posterior_prob >= confidence_threshold:
                evaluated_pairs[(tis_pos, tts_pos)] = {
                    'posterior_probability': posterior_prob,
                    'features': features
                }
        
        return evaluated_pairs
    
    def _calculate_features(self, tis_pos: int, tts_pos: int) -> Dict:
        """计算特征向量"""
        features = {
            'tis_prob': self.probs[tis_pos][0],
            'tts_prob': self.probs[tts_pos][1],
            'kozak_score': self._calculate_kozak_score(tis_pos),
            'cai_score': self._calculate_cai(tis_pos, tts_pos),
            'length_score': self._calculate_length_score(tis_pos, tts_pos)
        }
        return features
    
    def _calculate_kozak_score(self, tis_pos: int) -> float:
        """计算Kozak序列得分"""
        if tis_pos < 6 or tis_pos + 4 >= len(self.sequence):
            return 0.0
            
        kozak_region = self.sequence[tis_pos-6:tis_pos+4]
        score = 1.0
        
        for i, base in enumerate(kozak_region):
            pos = i - 6
            if pos in self.kozak_pwm:
                score *= self.kozak_pwm[pos].get(base, 0.1)
        
        return score
    
    def _calculate_cai(self, tis_pos: int, tts_pos: int) -> float:
        """计算CAI值"""
        # 提取CDS序列
        cds = self.sequence[tis_pos:tts_pos+3]
        cai_score = 1.0
        codon_count = len(cds) // 3
        
        for i in range(0, len(cds)-2, 3):
            codon = cds[i:i+3]
            if codon in self.codon_usage:
                cai_score *= self.codon_usage[codon]
        
        return math.pow(cai_score, 1.0/codon_count)
    
    def _calculate_length_score(self, tis_pos: int, tts_pos: int) -> float:
        """计算长度得分(基于对数正态分布)"""
        length = tts_pos - tis_pos + 3
        mu = np.log(1000)  # 典型CDS长度中位数
        sigma = 0.5        # 分布宽度参数
        
        score = np.exp(-(np.log(length) - mu)**2 / (2 * sigma**2))
        return score
    
    def _calculate_posterior(self, features: Dict) -> float:
        """计算后验概率"""
        # 特征权重
        weights = {
            'tis_prob': 0.3,
            'tts_prob': 0.3,
            'kozak_score': 0.2,
            'cai_score': 0.1,
            'length_score': 0.1
        }
        
        # 计算加权得分
        posterior = 0
        for feature, weight in weights.items():
            posterior += features[feature] * weight
            
        return posterior
        
def main():
    # 示例数据
    sequence = "您的转录本序列"
    softmax_probs = np.array([...])  # 您提供的softmax概率数据
    
    # 第一步：找到潜在TIS-TTS对
    predictor = TISPredictor(sequence, softmax_probs)
    potential_pairs = predictor.find_potential_pairs()
    
    # 第二步：贝叶斯评估
    evaluator = BayesianEvaluator(sequence, softmax_probs)
    high_confidence_pairs = evaluator.evaluate_pairs(potential_pairs)
    
    # 输出结果
    for (tis_pos, tts_pos), result in high_confidence_pairs.items():
        print(f"TIS-TTS pair ({tis_pos}, {tts_pos}):")
        print(f"Posterior probability: {result['posterior_probability']:.4f}")
        print("Features:")
        for feature, value in result['features'].items():
            print(f"- {feature}: {value:.4f}")
        print()

if __name__ == "__main__":
    main()