# scripts/train.py
import argparse
from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader
from src.data import collate_fn
from src.models import MultiModalTransformer
from src.data import TranslationSiteDataset
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
    parser.add_argument('--accumulation_steps', type=int, default=8,
                      help='梯度累积步数')
    parser.add_argument('--gpu', type=str, default='0',
                      help='使用的GPU ID，例如0或1')
    
    # 其他参数
    parser.add_argument('--fp16', action='store_true',
                      help='使用混合精度训练')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    parser.add_argument('--use_wandb', action='store_true',
                      help='使用Weights & Biases记录')
    
    return parser.parse_args()

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
        for seq in invalid_sequences[:5]:
            logger.warning(f"Invalid sequence at index {seq['index']}: max_value={seq['max_value']}, shape={seq['shape']}")

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_score, path):
    """保存检查点"""
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_score': best_score,
        'feature_dims': model.feature_dims,
        'config': model.config.__dict__
    }
    torch.save(save_dict, path)

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
        
        total_loss += loss.item() * accumulation_steps
        with torch.no_grad():
            predictions = outputs.argmax(dim=-1)
            valid_mask = labels != 0
            if valid_mask.any():
                all_predictions.append(outputs[valid_mask])
                all_labels.append(labels[valid_mask])
        
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
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

def validate(model, data_loader, criterion, device, metrics_calculator, logger):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Validation')
        for batch in pbar:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            mask = batch.get('attention_mask', None)
            if mask is not None:
                mask = mask.to(device)
            
            outputs = model(inputs, mask=mask)
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                labels.reshape(-1)
            )
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=-1)
            valid_mask = labels != 0
            if valid_mask.any():
                all_predictions.append(outputs[valid_mask])
                all_labels.append(labels[valid_mask])
    
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

def train(args, logger):
    """训练函数"""
    # 设置GPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 限制GPU内存使用
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%的GPU内存
    
    # 初始化配置
    model_config = ModelConfig()
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        num_workers=args.num_workers,
        fp16=args.fp16
    )
    
    # 加载和验证数据集
    logger.info("加载数据集...")
    train_dataset = TranslationSiteDataset.from_file(args.train_data)
    val_dataset = TranslationSiteDataset.from_file(args.val_data)
    validate_dataset(train_dataset, logger)
    validate_dataset(val_dataset, logger)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=lambda x: collate_fn(x, max_length=model_config.max_seq_length)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=lambda x: collate_fn(x, max_length=model_config.max_seq_length)
    )
    
    # 初始化模型
    feature_dims = train_dataset.get_feature_dims()
    logger.info(f"特征维度: {feature_dims}")
    model = MultiModalTransformer(model_config, feature_dims)
    model = model.to(device)
    
    # 初始化优化器等
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.cuda.amp.GradScaler() if training_config.fp16 else None
    
    # 初始化训练组件
    metrics_calculator = MetricsCalculator()
    best_score = float('-inf')
    best_model_path = Path(args.output_dir) / 'best_model.pt'
    early_stopping = EarlyStopping(
        patience=training_config.early_stopping_patience,
        mode='max',
        min_delta=1e-4
    )
    
    # 训练循环
    try:
        for epoch in range(training_config.epochs):
            # 训练和验证
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, metrics_calculator, logger,
                accumulation_steps=args.accumulation_steps
            )
            
            val_metrics = validate(
                model, val_loader, criterion,
                device, metrics_calculator, logger
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # 记录日志
            logger.info(f"\nEpoch {epoch+1}/{training_config.epochs}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"Train Macro F1: {train_metrics['macro_f1']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
            
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
            
            if early_stopping(epoch, current_score):
                logger.info(f"触发早停! 最佳分数: {best_score:.4f}")
                break
        
        logger.info(f"训练完成! 最佳验证分数: {best_score:.4f}")
        
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
    
    return best_score

def main():
    args = get_train_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger('trainer', log_file=output_dir / 'training.log')
    
    try:
        train(args, logger)
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    main()