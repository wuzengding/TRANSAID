# scripts/predict.py
import argparse
import torch
from pathlib import Path
from src.features import SequenceFeatureExtractor, StructureFeatureExtractor
from src.models import MultiModalTransformer
from utils.logger import setup_logger
from Bio import SeqIO
import json
from tqdm import tqdm

def get_predict_args():
    parser = argparse.ArgumentParser(description='预测RNA翻译位点')
    parser.add_argument('--model', type=str, required=True,
                      help='模型文件路径')
    parser.add_argument('--input_file', type=str, required=True,
                      help='输入FASTA文件路径')
    parser.add_argument('--output_file', type=str, required=True,
                      help='输出文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='预测阈值')
    return parser.parse_args()

def predict_sequence(seq: str,
                    model: MultiModalTransformer,
                    seq_extractor: SequenceFeatureExtractor,
                    struct_extractor: StructureFeatureExtractor,
                    device: torch.device,
                    threshold: float = 0.5):
    """预测单个序列的翻译位点
    
    Args:
        seq: 输入序列
        model: 模型
        seq_extractor: 序列特征提取器
        struct_extractor: 结构特征提取器
        device: 计算设备
        threshold: 预测阈值
    
    Returns:
        dict: 预测结果
    """
    # 提取特征
    seq_features = seq_extractor.extract_features(seq)
    struct_features = struct_extractor.extract_features(seq)
    
    # 合并特征
    features = {
        **seq_features,
        **struct_features
    }
    
    # 移动到设备
    features = {k: v.to(device) for k, v in features.items()}
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=-1)
        predictions = outputs.argmax(dim=-1)
    
    # 找到起始和终止位点
    start_positions = (predictions == 1).nonzero().flatten().cpu().tolist()
    stop_positions = (predictions == 2).nonzero().flatten().cpu().tolist()
    
    # 获取概率值
    start_probs = [float(probabilities[i, 1]) for i in start_positions]
    stop_probs = [float(probabilities[i, 2]) for i in stop_positions]
    
    # 过滤低于阈值的位点
    filtered_starts = [
        (pos, prob) for pos, prob in zip(start_positions, start_probs)
        if prob >= threshold
    ]
    filtered_stops = [
        (pos, prob) for pos, prob in zip(stop_positions, stop_probs)
        if prob >= threshold
    ]
    
    return {
        'start_sites': [
            {'position': pos, 'probability': prob}
            for pos, prob in filtered_starts
        ],
        'stop_sites': [
            {'position': pos, 'probability': prob}
            for pos, prob in filtered_stops
        ]
    }

def process_file(args, model, seq_extractor, struct_extractor, device, logger):
    """处理输入文件"""
    results = []
    
    # 读取并处理序列
    for record in tqdm(SeqIO.parse(args.input_file, "fasta"),
                      desc="Processing sequences"):
        seq = str(record.seq)
        
        try:
            # 预测位点
            prediction = predict_sequence(
                seq,
                model,
                seq_extractor,
                struct_extractor,
                device,
                args.threshold
            )
            
            # 构建结果
            result = {
                'id': record.id,
                'length': len(seq),
                'start_sites': prediction['start_sites'],
                'stop_sites': prediction['stop_sites']
            }
            
            # 添加可能的ORF信息
            result['orfs'] = find_possible_orfs(
                prediction['start_sites'],
                prediction['stop_sites'],
                min_length=30  # 最小ORF长度
            )
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"处理序列 {record.id} 时出错: {str(e)}")
    
    # 保存结果
    save_results(results, args.output_file, logger)
    
    return results

def find_possible_orfs(start_sites, stop_sites, min_length=30):
    """查找可能的ORFs"""
    orfs = []
    
    # 将位点转换为位置列表
    starts = [(site['position'], site['probability']) for site in start_sites]
    stops = [(site['position'], site['probability']) for site in stop_sites]
    
    # 对每个起始位点
    for start_pos, start_prob in starts:
        # 寻找下一个终止位点
        valid_stops = [(pos, prob) for pos, prob in stops if pos > start_pos]
        if valid_stops:
            stop_pos, stop_prob = min(valid_stops, key=lambda x: x[0])
            length = stop_pos - start_pos + 3
            
            if length >= min_length:
                orfs.append({
                    'start': start_pos,
                    'stop': stop_pos,
                    'length': length,
                    'start_probability': start_prob,
                    'stop_probability': stop_prob,
                    'combined_probability': (start_prob + stop_prob) / 2
                })
    
    # 按长度排序
    orfs.sort(key=lambda x: x['length'], reverse=True)
    return orfs

def save_results(results, output_file, logger):
    """保存预测结果"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON格式的详细结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 保存BED格式的结果
    bed_file = output_file.with_suffix('.bed')
    with open(bed_file, 'w') as f:
        for result in results:
            seq_id = result['id']
            # 保存起始位点
            for site in result['start_sites']:
                f.write(f"{seq_id}\t{site['position']}\t{site['position']+3}\t"
                       f"start\t{site['probability']:.3f}\t+\n")
            # 保存终止位点
            for site in result['stop_sites']:
                f.write(f"{seq_id}\t{site['position']}\t{site['position']+3}\t"
                       f"stop\t{site['probability']:.3f}\t+\n")
    
    # 保存统计信息
    stats = {
        'total_sequences': len(results),
        'sequences_with_predictions': len([r for r in results if r['start_sites'] or r['stop_sites']]),
        'total_start_sites': sum(len(r['start_sites']) for r in results),
        'total_stop_sites': sum(len(r['stop_sites']) for r in results),
        'total_orfs': sum(len(r['orfs']) for r in results),
    }
    
    stats_file = output_file.with_name(output_file.stem + '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    logger.info(f"结果已保存到: {output_file}")
    logger.info(f"BED格式结果: {bed_file}")
    logger.info(f"统计信息: {stats_file}")

def main():
    args = get_predict_args()
    logger = setup_logger('predictor')
    
    try:
        # 加载模型
        logger.info("加载模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, checkpoint = MultiModalTransformer.load_checkpoint(args.model, device)
        config = checkpoint['config']
        
        # 初始化特征提取器
        seq_extractor = SequenceFeatureExtractor(config)
        struct_extractor = StructureFeatureExtractor(config)
        
        # 处理输入文件
        logger.info("开始预测...")
        results = process_file(
            args,
            model,
            seq_extractor,
            struct_extractor,
            device,
            logger
        )
        
        logger.info(f"预测完成! 处理了 {len(results)} 个序列")
        
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()