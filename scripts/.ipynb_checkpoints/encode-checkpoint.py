# scripts/encode.py
import argparse
from pathlib import Path
import logging
from src.features import SequenceFeatureExtractor, StructureFeatureExtractor
from src.data import TranslationSiteDataset
from utils.logger import setup_logger
from configs import ModelConfig
import torch
from Bio import SeqIO
from typing import Dict, List, Tuple
import json
from src.features.utils import normalize_sequence

def get_encode_args():
    parser = argparse.ArgumentParser(description='RNA序列编码')
    parser.add_argument('--fasta_file', type=str, required=True,
                      help='输入FASTA文件路径')
    parser.add_argument('--gbff_file', type=str, required=True,
                      help='输入GBFF注释文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出目录')
    parser.add_argument('--max_seq_length', type=int, default=10000,
                      help='最大序列长度')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                      help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                      help='验证集比例')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    return parser.parse_args()

def process_sequences(fasta_file: str,
                     gbff_file: str,
                     seq_extractor: SequenceFeatureExtractor,
                     struct_extractor: StructureFeatureExtractor,
                     logger: logging.Logger) -> Dict:
    """处理序列数据
    
    Args:
        fasta_file: FASTA文件路径
        gbff_file: GBFF注释文件路径
        seq_extractor: 序列特征提取器
        struct_extractor: 结构特征提取器
        logger: 日志记录器
    
    Returns:
        encoded_sequences: 编码后的序列特征列表
        labels: 标签列表
    """
    # 初始化统计计数
    stats = {
        'total_reads': 0,            # 总reads数
        'not_in_gbff': 0,           # 不在gbff文件中的reads数
        'too_long': 0,              # 超过最大长度的reads数
        'filtered_abnormal': 0,      # 因为异常碱基过滤掉的reads数
        'process_error': 0,          # 处理出错的reads数
        'success': 0                 # 成功处理的reads数
    }
    
    # 解析GBFF文件获取CDS位置
    logger.info("解析注释文件...")
    cds_positions = {}
    for record in SeqIO.parse(gbff_file, "genbank"):
        acc = record.id
        for feature in record.features:
            if feature.type == "CDS":
                start = int(feature.location.start)
                end = int(feature.location.end)
                cds_positions[acc] = (start, end)
    
    # 处理序列
    logger.info("处理序列...")
    encoded_sequences = []
    labels = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        stats['total_reads'] += 1
        
        if record.id not in cds_positions:
            stats['not_in_gbff'] += 1
            logger.warning(f"序列 {record.id} 不在gbff文件中")
            continue
            
        start, end = cds_positions[record.id]
        seq = str(record.seq)
        
        if len(seq) > seq_extractor.config.max_seq_length:
            stats['too_long'] += 1
            continue
            
        try:
            # 检查序列中是否包含N或其他异常碱基
            normalized_seq = normalize_sequence(seq)
            if 'N' in normalized_seq:
                stats['filtered_abnormal'] += 1
                logger.debug(f"序列 {record.id} 包含异常碱基，已过滤")
                continue
            
            # 提取特征
            seq_features = seq_extractor.extract_features(seq)
            struct_features = struct_extractor.extract_features(seq)
            
            # 合并特征
            features = {**seq_features, **struct_features}
            
            # 创建标签
            label = torch.zeros(len(seq), dtype=torch.long)
            label[start:start+3] = 1  # 起始位点
            label[end-3:end] = 2      # 终止位点
            
            encoded_sequences.append(features)
            labels.append(label)
            stats['success'] += 1
            
        except Exception as e:
            stats['process_error'] += 1
            logger.warning(f"处理序列 {record.id} 时出错: {str(e)}")
    
    # 记录统计信息
    logger.info("\n数据处理统计:")
    logger.info(f"总reads数: {stats['total_reads']}")
    logger.info(f"不在gbff文件中的reads数: {stats['not_in_gbff']} ({stats['not_in_gbff']/stats['total_reads']*100:.2f}%)")
    logger.info(f"超过最大长度的reads数: {stats['too_long']} ({stats['too_long']/stats['total_reads']*100:.2f}%)")
    logger.info(f"包含异常碱基被过滤的reads数: {stats['filtered_abnormal']} ({stats['filtered_abnormal']/stats['total_reads']*100:.2f}%)")
    logger.info(f"处理出错的reads数: {stats['process_error']} ({stats['process_error']/stats['total_reads']*100:.2f}%)")
    logger.info(f"成功处理的reads数: {stats['success']} ({stats['success']/stats['total_reads']*100:.2f}%)")
    
    # 获取特征维度信息
    feature_dims = {
        **seq_extractor.get_feature_dims(),
        **struct_extractor.get_feature_dims()
    }
    
    return {
        'sequences': encoded_sequences,
        'labels': labels,
        'feature_dims': feature_dims,
        'stats': stats  # 添加统计信息到返回值
    }
    
def main():
    args = get_encode_args()
    logger = setup_logger('encoder')
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 初始化配置和特征提取器
        config = ModelConfig(max_seq_length=args.max_seq_length)
        seq_extractor = SequenceFeatureExtractor(config)
        struct_extractor = StructureFeatureExtractor(config)
        
        # 提取特征
        logger.info("开始提取特征...")
        encoded_data = process_sequences(
            args.fasta_file,
            args.gbff_file,
            seq_extractor,
            struct_extractor,
            logger
        )
        
        # 创建数据集
        dataset = TranslationSiteDataset(
            encoded_data['sequences'],  # sequences
            encoded_data['labels'],   # labels
            encoded_data['feature_dims']
        )
        
        # 拆分数据集
        logger.info("拆分数据集...")
        train_data, val_data, test_data = dataset.split(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_seed=args.seed
        )
        
        # 保存数据集
        logger.info("保存数据集...")
        for name, data in [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]:
            save_path = output_dir / f'{name}_data.pkl'
            data.save(save_path)
            logger.info(f"保存{name}数据集到: {save_path}")
        
        # 保存数据集统计信息
        stats = {
            'total_sequences': len(dataset),
            'train_sequences': len(train_data),
            'val_sequences': len(val_data),
            'test_sequences': len(test_data),
            'max_sequence_length': dataset.stats['max_length'],
            'avg_sequence_length': dataset.stats['avg_length'],
            'class_distribution': dataset.stats['label_distribution'].tolist(),
            'split_ratios': {
                'train': args.train_ratio,
                'val': args.val_ratio,
                'test': 1 - args.train_ratio - args.val_ratio
            }
        }
        
        with open(output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        # 保存特征提取器配置
        config_dict = {
            'model_config': config.__dict__,
            'preprocessing': {
                'max_seq_length': args.max_seq_length,
                'train_ratio': args.train_ratio,
                'val_ratio': args.val_ratio,
                'seed': args.seed
            }
        }
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        logger.info("\n特征提取完成!")
        logger.info(f"总序列数: {len(dataset)}")
        logger.info(f"训练集: {len(train_data)} 序列")
        logger.info(f"验证集: {len(val_data)} 序列")
        logger.info(f"测试集: {len(test_data)} 序列")
        logger.info(f"最大序列长度: {stats['max_sequence_length']}")
        logger.info(f"平均序列长度: {stats['avg_sequence_length']:.2f}")
        logger.info(f"类别分布: {stats['class_distribution']}")
        logger.info(f"所有文件已保存到: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\n处理被用户中断")
    except Exception as e:
        logger.error(f"\n处理过程中出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()