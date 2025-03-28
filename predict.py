#!/usr/bin/env python
# predict.py

import argparse
import sys
from pathlib import Path
import logging
from src.predictor import TranslationPredictor
from src.utils import save_results, setup_logger

def get_args():
    parser = argparse.ArgumentParser(description='Predict translation products from RNA sequences')
    
    # 基本输入输出
    parser.add_argument('--input', type=str, required=True, 
                        help='Input FASTA file')
    parser.add_argument('--output', type=str, required=True,
                        help='Base path for output files (CSV and FASTA)')

    # 模型相关                   
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained model')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU device ID, -1 for CPU')

    # 结果过滤相关
    #parser.add_argument('--probs_cutoff', type=float, default=0.1,
    #                    help='Cutoff value for probability score')
    parser.add_argument('--integrated_cutoff', type=float, default=0.5,
                        help='Cutoff value for integrated score')
    parser.add_argument('--filter_mode', choices=['all', 'best'], default='best',
                        help='Keep all ORFs above cutoff or only the best one')
                       
    # 结果输出控制
    parser.add_argument('--save_raw_predictions', action='store_true',
                        help='Whether to save raw prediction results in PKL format')

    # 预测参数调整
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction')
    parser.add_argument('--max_seq_len', type=int, default=27109,
                        help='Maximum sequence length to process')

    # 结果过滤细节
    parser.add_argument('--orf_length_cutoff', type=int, default=30,
                        help='Minimum ORF length in amino acids')
    parser.add_argument('--kozak_cutoff', type=float, default=0.05,
                        help='Minimum Kozak sequence score')
    parser.add_argument('--tis_cutoff', type=float, default=0.1,
                        help='Minimum TIS prediction score')
    parser.add_argument('--tts_cutoff', type=float, default=0.1,
                        help='Minimum TTS prediction score')
    
    return parser.parse_args()

def main():
    # 获取参数
    args = get_args()

    # 检查输出路径
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    output_base = Path(args.output)
    log_file = output_base.with_suffix('.log') 

    if log_file.exists():
        with open(log_file, "w") as f:
            f.truncate(0)
    logger = setup_logger(name="translation_predictor",log_file=str(log_file))
    
    try:
        # 检查输入文件
        input_path = Path(args.input)
        #if not input_path.exists():
        #    raise FileNotFoundError(f"Input file not found: {args.input}")
        
        # 初始化预测器
        logger.info("Initializing predictor...")
        predictor = TranslationPredictor(
            model_path=args.model_path,
            device=args.gpu,
            batch_size=args.batch_size,
            sequence_length=args.max_seq_len,
            tis_cutoff=args.tis_cutoff,
            tts_cutoff=args.tts_cutoff,
            orf_length_cutoff=args.orf_length_cutoff
        )
        
        # 执行预测
        logger.info("Starting prediction...")
        if input_path.is_file() and input_path.suffix.lower() in {'.fasta', '.fa', '.fna'}:
                logger.info(f"Processing FASTA file: {input_path}")
                results,probs = predictor.predict_batch(input_path)  # 传入Path对象
        else:
            logger.error(f"Invalid file type: {input_path.suffix} (expected .fasta/.fa/.fna)")
            exit(1)

        
        #print("results", results)
        # 过滤结果
        logger.info("Applying filters and marking results...")
        all_results = results  # 保留所有结果
        
        # 对所有结果应用过滤条件并标记原因
        for r in all_results:
            reasons = []
            if r.integrated_score <= args.integrated_cutoff:
                reasons.append(f"integrated_score({r.integrated_score:.3f}) <= {args.integrated_cutoff}")
            #if r.kozak_score <= args.kozak_cutoff:
            #    reasons.append(f"kozak_score({r.kozak_score:.3f}) <= {args.kozak_cutoff}")
            if len(r.protein_sequence) < args.orf_length_cutoff:
                reasons.append(f"protein_length({len(r.protein_sequence)}) < {args.orf_length_cutoff}")
            
            if reasons:
                r.passed_filter = False
                r.filter_reason = "; ".join(reasons)
              
        # 如果filter_mode为best，标记除最佳结果外的所有通过过滤的结果
        if args.filter_mode == 'best':
            # 按转录本ID分组
            transcript_groups = {}
            for r in all_results:
                if r.sequence_id not in transcript_groups:
                    transcript_groups[r.sequence_id] = []
                transcript_groups[r.sequence_id].append(r)
            
            # 对每个转录本，只保留通过过滤的最佳结果
            for transcript_id, transcript_results in transcript_groups.items():
                # 找出该转录本中通过过滤的结果
                passed_results = [r for r in transcript_results if r.passed_filter]
                if passed_results:
                    # 保留得分最高的，将其他标记为"not best for transcript"
                    best_result = max(passed_results, key=lambda x: x.integrated_score)
                    for r in passed_results:
                        if r != best_result:
                            r.passed_filter = False
                            r.filter_reason += "; not the best result for this transcript"


        # 保存结果
        logger.info("Saving results...")
        #保存结果前对结果进行排序，使输出文件中结果按得分降序排列
        sorted_results = sorted(all_results, key=lambda x: x.integrated_score, reverse=True)
        output_base = Path(args.output)
        csv_path = output_base.with_suffix('.csv')
        faa_path = output_base.with_suffix('.faa')
        
        # 修改调用save_results函数
        save_results(
            probs=probs,
            results=sorted_results,  # 保存所有结果
            csv_path=csv_path,
            faa_path=faa_path,
            save_raw=args.save_raw_predictions,
            save_only_passed=True  # 在FAA文件中只保存通过过滤的结果
        )
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
