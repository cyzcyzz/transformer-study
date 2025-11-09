"""
位置前馈网络实现
Position-wise Feed-Forward Network Implementation
"""
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络（FFN）"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super(PositionwiseFeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            [batch_size, seq_len, d_model]
        """
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        # 这里使用GELU替代ReLU
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

