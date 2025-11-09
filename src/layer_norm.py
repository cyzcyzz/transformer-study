"""
层归一化和残差连接
Layer Normalization and Residual Connection
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """层归一化"""
    
    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: 模型维度
            eps: 防止除零的小值
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        """
        层归一化: LN(x) = γ * (x - μ) / (σ + ε) + β
        
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            [batch_size, seq_len, d_model]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def residual_connection(x, sublayer_output):
    """
    残差连接: x + sublayer(x)
    
    Args:
        x: 输入
        sublayer_output: 子层输出
    
    Returns:
        残差连接后的结果
    """
    return x + sublayer_output

