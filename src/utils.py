# src/utils.py

import torch
import numpy as np
from typing import Dict, List, Union
from pathlib import Path
import logging

# 核苷酸编码字典
NUCLEOTIDE_DICT = {
    'A': 0, 'T': 1, 'G': 2, 'C': 3,
    'N': 4, 'R': 4, 'Y': 4, 'M': 4,
    'K': 4, 'S': 4, 'W': 4, 'H': 4,
    'B': 4, 'V': 4, 'D': 4
}

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s'
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def sequence_to_tensor(sequence: str, max_len: int = 27109) -> torch.Tensor:
    """将核苷酸序列转换为定长tensor
    
    Args:
        sequence: 输入核苷酸序列
        max_len: 固定长度,默认27109
        
    Returns:
        torch.Tensor: 形状为(max_len,)的编码tensor
    """
    encoding = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4, "N":0}
    sequence = sequence.upper()
    encoded_seq = np.zeros(max_len, dtype=np.int64)
    
    # 只处理max_len长度的序列
    for i, nucleotide in enumerate(sequence[:max_len]):
        encoded_seq[i] = encoding.get(nucleotide, 0)
        
    return torch.tensor(encoded_seq, dtype=torch.long)

def save_checkpoint(state: Dict, 
                   is_best: bool,
                   checkpoint_dir: Union[str, Path],
                   best_model_name: str = 'model_best.pth'):
    """保存检查点
    
    Args:
        state: 要保存的状态字典
        is_best: 是否是最佳模型
        checkpoint_dir: 检查点保存目录
        best_model_name: 最佳模型的文件名
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存最新检查点
    latest_path = checkpoint_dir / 'checkpoint_latest.pth'
    torch.save(state, latest_path)
    
    # 如果是最佳模型,同时保存一份
    if is_best:
        best_path = checkpoint_dir / best_model_name
        torch.save(state, best_path)

def load_checkpoint(checkpoint_path: Union[str, Path]) -> Dict:
    """加载检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        
    Returns:
        Dict: 加载的状态字典
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        
    return torch.load(checkpoint_path, map_location='cpu')

def save_results(probs,
                results,
                csv_path: Path,
                faa_path: Path,
                save_raw: bool = False,
                save_only_passed: bool = True):
    """保存预测结果
    
    Args:
        probs: 预测的概率数据
        results: 预测结果列表
        csv_path: CSV输出路径
        faa_path: FASTA格式蛋白质序列输出路径
        save_raw: 是否保存原始预测结果
        save_only_passed: 是否只在FAA文件中保存通过过滤的结果
    """
    # 保存CSV格式结果，包含所有结果和过滤状态
    with open(csv_path, 'w') as f:
        f.write("Sequence_ID,Start,Stop,TIS_Score,TTS_Score,Kozak_Score,"
                "CAI_Score,GC_Score,Integrated_Score,Protein_Length,"
                "Passed_Filter,Filter_Reason\n")
        for r in results:
            f.write(f"{r.sequence_id},{r.start_position},{r.stop_position},"
                   f"{r.tis_score:.3f},{r.tts_score:.3f},{r.kozak_score:.3f},"
                   f"{r.cai_score:.3f},{r.gc_score:.3f},{r.integrated_score:.3f},"
                   f"{len(r.protein_sequence)},"
                   f"{str(r.passed_filter)},\"{r.filter_reason}\"\n")
    
    # 保存蛋白质序列，可选择只保存通过过滤的结果
    with open(faa_path, 'w') as f:
        for r in results:
            if not save_only_passed or r.passed_filter:
                filter_status = "PASS" if r.passed_filter else "FAIL"
                f.write(f">{r.sequence_id}_{r.start_position}_{r.stop_position}_{filter_status}_{r.integrated_score:.3f}\n")
                f.write(f"{r.protein_sequence}\n")
            
    # 保存原始预测结果
    if save_raw and probs is not None:
        import pickle
        pickle_path = csv_path.with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(probs, f)



