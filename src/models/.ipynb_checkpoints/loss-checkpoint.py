# src/models/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    """Focal Loss实现"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, 
                inputs: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, seq_len, num_classes)
            targets: (batch_size, seq_len)
            mask: (batch_size, seq_len)
        """
        ce_loss = F.cross_entropy(
            inputs.view(-1, inputs.size(-1)),
            targets.view(-1),
            reduction='none'
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if mask is not None:
            focal_loss = focal_loss * mask.view(-1)
        
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """Dice Loss实现"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.softmax(inputs, dim=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, 
                 focal_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        focal_loss = self.focal(inputs, targets, mask)
        dice_loss = self.dice(inputs, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss