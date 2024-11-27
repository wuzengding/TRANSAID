# src/models/modules.py
import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv_module(x)
        x = x + residual
        return self.dropout(x)

class AttentionPooling(nn.Module):
    """注意力池化层"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (batch_size, seq_len, hidden_size)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(
                mask.unsqueeze(-1) == 0, float('-inf')
            )
        
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_sum = torch.bmm(attention_weights.transpose(1, 2), x)  # (batch_size, 1, hidden_size)
        return weighted_sum.squeeze(1)  # (batch_size, hidden_size)

class LayerNorm(nn.Module):
    """层归一化"""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))