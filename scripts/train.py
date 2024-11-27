# scripts/train.py
import argparse
from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.data import collate_fn
from src.models import MultiModalTransformer
from src.data import create_data_loaders, TranslationSiteDataset
from utils.logger import setup_logger
from utils.metrics import MetricsCalculator, EarlyStopping
from configs import ModelConfig, TrainingConfig
import json
import wandb
import pickle
from tqdm import tqdm
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_train_args():
    parser = argparse.ArgumentParser(description='训练RNA翻译位点预测模型')
    # 数据参数
    parser.add_argument('--train_data', type=str, required=True,
                      help='训练数据文件路径')
    parser.add_argument('--val_data', type=str, required=True,
                      help='验证数据文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                      help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='数据加载进程数')
    
    # GPU相关
    parser.add_argument('--fp16', action='store_true',
                      help='使用混合精度训练')
    parser.add_argument('--local_rank', type=int, default=-1,
                      help='分布式训练的本地排名')
    parser.add_argument('--gpus', type=str, default='0,1',
                      help='使用的GPU ID，例如0,1')
    parser.add_argument('--gpu_memory_fraction', type=str, default=None,
                      help='每个GPU的显存比例，例如0.4,0.4')
    parser.add_argument('--master_gpu', type=int, default=None,
                      help='主GPU ID')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    parser.add_argument('--use_wandb', action='store_true',
                      help='使用Weights & Biases记录')
    
    return parser.parse_args()

def train_epoch(model, data_loader, optimizer, criterion, device, scaler,
                metrics_calculator, logger, accumulation_steps: int = 8):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(data_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # 准备数据
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        mask = batch.get('attention_mask', None)
        if mask is not None:
            mask = mask.to(device)
        
        # 前向传播
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs, mask=mask)
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    labels.reshape(-1)
                )
                loss = loss / accumulation_steps
            
            # 反向传播
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(inputs, mask=mask)
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                labels.reshape(-1)
            )
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # 收集预测结果
        total_loss += loss.item() * accumulation_steps
        with torch.no_grad():
            predictions = outputs.argmax(dim=-1)
            valid_mask = labels != 0
            if valid_mask.any():
                all_predictions.append(outputs[valid_mask])
                all_labels.append(labels[valid_mask])
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    # 计算整体指标
    metrics = {'loss': total_loss / len(data_loader)}
    if all_predictions and all_labels:
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics.update(metrics_calculator.calculate_metrics(
            all_predictions,
            all_labels
        ))
    else:
        metrics.update({
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'class_0_f1': 0.0,
            'class_1_f1': 0.0,
            'class_2_f1': 0.0
        })
    
    return metrics

def validate_dataset(dataset: TranslationSiteDataset, logger: logging.Logger):
    """验证数据集的正确性"""
    logger.info("验证数据集...")
    
    invalid_sequences = []
    for idx in range(len(dataset)):
        features, _ = dataset[idx]
        nuc_seq = features['nucleotide']
        max_idx = nuc_seq.max().item()
        if max_idx >= 5:
            invalid_sequences.append({
                'index': idx,
                'max_value': max_idx,
                'shape': nuc_seq.shape
            })
    
    if invalid_sequences:
        logger.warning(f"Found {len(invalid_sequences)} sequences with invalid indices")
        for seq in invalid_sequences[:5]:  # 只显示前5个
            logger.warning(f"Invalid sequence at index {seq['index']}: max_value={seq['max_value']}, shape={seq['shape']}")

def validate(model, data_loader, criterion, device, metrics_calculator, logger):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Validation')
        for batch in pbar:
            # 准备数据
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            mask = batch.get('attention_mask', None)
            if mask is not None:
                mask = mask.to(device)
            
            # 前向传播
            outputs = model(inputs, mask=mask)
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                labels.reshape(-1)
            )
            
            # 收集预测和标签
            total_loss += loss.item()
            predictions = outputs.argmax(dim=-1)
            valid_mask = labels != 0
            if valid_mask.any():
                all_predictions.append(outputs[valid_mask])
                all_labels.append(labels[valid_mask])
    
    # 计算指标
    metrics = {'loss': total_loss / len(data_loader)}
    if all_predictions and all_labels:
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics.update(metrics_calculator.calculate_metrics(
            all_predictions,
            all_labels
        ))
    else:
        metrics.update({
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'class_0_f1': 0.0,
            'class_1_f1': 0.0,
            'class_2_f1': 0.0
        })
    
    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_score, path):
    """保存检查点"""
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_score': best_score,
        'feature_dims': model.feature_dims,  # 保存特征维度信息
        'config': model.config.__dict__      # 保存完整配置
    }
    torch.save(save_dict, path)

def setup_gpu_environment(args, logger):
    """设置GPU环境"""
    if args.gpus:
        gpus = [int(gpu) for gpu in args.gpus.split(',')]
        memory_fractions = None
        
        # 如果指定了显存使用比例
        if args.gpu_memory_fraction:
            memory_fractions = [float(f) for f in args.gpu_memory_fraction.split(',')]
            if len(memory_fractions) != len(gpus):
                raise ValueError("GPU数量和显存比例数量不匹配")
            
            for gpu, fraction in zip(gpus, memory_fractions):
                torch.cuda.set_per_process_memory_fraction(fraction, gpu)
        
        # 设置可见的GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        
        # 设置主GPU
        master_gpu = args.master_gpu if args.master_gpu is not None else gpus[0]
        torch.cuda.set_device(master_gpu)
        
        logger.info(f"GPU设置:")
        logger.info(f"- 使用的GPU: {gpus}")
        if memory_fractions:
            logger.info(f"- GPU显存比例: {memory_fractions}")
        logger.info(f"- 主GPU: {master_gpu}")
        
        return master_gpu
    else:
        logger.info("未指定GPU，使用CPU训练")
        return None

def setup_ddp(args, logger):
    """设置分布式训练"""
    if args.gpus:
        gpus = [int(gpu) for gpu in args.gpus.split(',')]
        n_gpu = len(gpus)
        
        if n_gpu > 1:
            # 初始化分布式环境
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(args.local_rank)
            logger.info(f"使用 {n_gpu} 个GPU训练, 当前GPU rank: {args.local_rank}")
            return True
    return False

def cleanup_ddp():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        
def train(args, logger):    
    """训练模型"""
    master_gpu = setup_gpu_environment(args, logger)
    # 初始化配置
    model_config = ModelConfig()
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        num_workers=args.num_workers,
        fp16=args.fp16
    )
    
    # 加载和创建数据集
    logger.info("加载数据集...")
    train_dataset = TranslationSiteDataset.from_file(args.train_data)
    val_dataset = TranslationSiteDataset.from_file(args.val_data)

    # 添加数据验证
    validate_dataset(train_dataset, logger)
    validate_dataset(val_dataset, logger)
    
    feature_dims = train_dataset.get_feature_dims()
    logger.info(f"特征维度: {feature_dims}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_length=model_config.max_seq_length)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_length=model_config.max_seq_length)
    )
    
    # 初始化模型和训练组件
    logger.info("初始化模型...")
    model = MultiModalTransformer(model_config, feature_dims)
    logger.info(f"模型特征维度(包含padding): {model.feature_dims}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    #if is_distributed:
    #    model = DDP(model, device_ids=[args.local_rank])
    #    logger.info(f"模型已包装为DDP，设备ID: {args.local_rank}")
    
    # 初始化优化器、调度器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.cuda.amp.GradScaler() if training_config.fp16 else None
    
    # 初始化训练组件
    early_stopping = EarlyStopping(
        patience=training_config.early_stopping_patience,
        mode='max',
        min_delta=1e-4
    )
    
    metrics_calculator = MetricsCalculator()
    best_score = float('-inf')
    best_model_path = Path(args.output_dir) / 'best_model.pt'
    
    if args.use_wandb:
        wandb.init(
            project="rna-translation-sites",
            config={
                "model_config": model_config.__dict__,
                "training_config": training_config.__dict__,
                "args": vars(args)
            }
        )
    
    # 训练循环
    logger.info("开始训练...")
    try:
        for epoch in range(training_config.epochs):
            # 训练epoch
            #if train_sampler is not None:
            #    train_sampler.set_epoch(epoch)
                
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, metrics_calculator, logger
            )
            
            # 验证
            val_metrics = validate(
                model, val_loader, criterion,
                device, metrics_calculator, logger
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # 记录指标
            logger.info(f"\nEpoch {epoch+1}/{training_config.epochs}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"Train Macro F1: {train_metrics['macro_f1']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
            
            # 记录到wandb
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'learning_rate': current_lr,
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
            
            # 保存最佳模型
            current_score = val_metrics['macro_f1']
            if current_score > best_score:
                best_score = current_score
                logger.info(f"发现更好的模型 (score: {best_score:.4f})")
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, val_metrics, best_score,
                    best_model_path
                )
            
            # 早停检查
            if early_stopping(epoch, current_score):
                logger.info(f"触发早停! 最佳分数: {best_score:.4f}")
                break
    
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise
    
    logger.info(f"训练完成! 最佳验证分数: {best_score:.4f}")
    return best_score

def main():
    args = get_train_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        'trainer',
        log_file=output_dir / 'training.log'
    )
    
    try:
        train(args, logger)
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    main()