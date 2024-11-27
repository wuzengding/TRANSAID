# src/models/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional
from .modules import PositionalEncoding, ConvBlock, AttentionPooling, LayerNorm, FeedForward

class MultiModalTransformer(nn.Module):
    """多模态Transformer模型"""
    def __init__(self, config, feature_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims.copy() 
        self.feature_dims['nucleotide'] += 1 #为PAD(0)增加一个维度
        
        # nucleotide特征是必需的
        self.nuc_embedding = nn.Embedding(
            num_embeddings=self.feature_dims['nucleotide'],
            embedding_dim=config.embedding_dim,
            padding_idx=0,  # 明确指定padding_idx=0
        )
        
        # 根据配置创建structure embedding
        if config.use_structure and 'structure' in feature_dims:
            self.struct_embedding = nn.Embedding(
                feature_dims['structure'], 
                config.embedding_dim
            )
        
        # 动态创建特征投影层
        projection_features = {
            'kmer': config.use_kmer,
            'complexity': config.use_complexity,
            'thermodynamic': config.use_thermodynamic,
            'conservation': config.use_conservation
        }
        
        self.feature_projections = nn.ModuleDict()
        for feat_name, use_feature in projection_features.items():
            if use_feature and feat_name in feature_dims:
                self.feature_projections[feat_name] = nn.Sequential(
                    nn.Linear(feature_dims[feat_name], config.embedding_dim),
                    nn.LayerNorm(config.embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout)
                )
        
        # CNN层
        if config.use_cnn:
            self.cnn_layers = nn.ModuleList([
                ConvBlock(
                    config.embedding_dim,
                    config.embedding_dim,
                    dropout=config.dropout
                )
                for _ in range(3)
            ])
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(
            config.embedding_dim,
            dropout=config.dropout
        )
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.num_layers)
        ])
        
        # 注意力池化
        if config.use_attention_pooling:
            self.attention_pooling = AttentionPooling(config.embedding_dim)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim // 2, 3)  # 3类输出
        )
    
    def forward(self, 
                features: Dict[str, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # nucleotide特征是必需的
        combined = self.nuc_embedding(features['nucleotide'])
    
        # 添加structure特征（如果启用）
        if hasattr(self, 'struct_embedding') and 'structure' in features:
            struct_emb = self.struct_embedding(features['structure'])
            combined = combined + struct_emb
    
        # 处理其他特征
        for name, projection in self.feature_projections.items():
            if name in features:
                feat = projection(features[name])
                if feat.dim() == 2:
                    feat = feat.unsqueeze(1).expand(-1, combined.size(1), -1)
                combined = combined + feat
    
        # CNN处理
        if hasattr(self, 'cnn_layers'):
            seq_features = combined.transpose(1, 2)
            for cnn in self.cnn_layers:
                seq_features = cnn(seq_features)
            combined = seq_features.transpose(1, 2)
    
        # 位置编码和后续处理
        encoded = self.pos_encoder(combined)
    
        hidden_states = encoded
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, mask=mask)
    
        if hasattr(self, 'attention_pooling'):
            pooled = self.attention_pooling(hidden_states, mask)
            output = self.output_projection(pooled)
            output = output.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        else:
            output = self.output_projection(hidden_states)
    
        return output

    def save_checkpoint(self, path: str, optimizer=None, epoch=None, metrics=None):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device='cuda'):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'], checkpoint['feature_dims'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model, checkpoint
    
    def save_checkpoint(self, path: str, optimizer=None, epoch=None, metrics=None):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'feature_dims': self.feature_dims
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
        torch.save(checkpoint, path)

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            config.embedding_dim, 
            config.num_heads,
            dropout=config.dropout
        )
        self.feed_forward = FeedForward(
            config.embedding_dim,
            config.hidden_dim,
            dropout=config.dropout
        )
        self.norm1 = LayerNorm(config.embedding_dim)
        self.norm2 = LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 自注意力
        attn_output = self.self_attn(
            x, x, x,
            mask=mask,
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 分割成多头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # 调整mask维度以匹配scores: [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e4)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output