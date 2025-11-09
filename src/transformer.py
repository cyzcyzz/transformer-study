"""
完整的Transformer模型实现
Complete Transformer Model Implementation
"""
import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .positional_encoding import PositionalEncoding
from .layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 掩码矩阵
        
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        
        Returns:
            [batch_size, tgt_seq_len, d_model]
        """
        # 掩码自注意力
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 交叉注意力
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Transformer(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_ff=2048, max_seq_length=5000, dropout=0.1):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            d_ff: 前馈网络隐藏层维度
            max_seq_length: 最大序列长度
            dropout: dropout比率
        """
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt=None):
        """生成掩码矩阵"""
        # 源序列掩码（padding掩码）
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        if tgt is not None:
            # 目标序列掩码（padding掩码 + 因果掩码）
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
            seq_length = tgt.size(1)
            causal_mask = torch.tril(torch.ones(seq_length, seq_length)).bool().to(tgt.device)
            tgt_mask = tgt_mask & causal_mask.unsqueeze(0).unsqueeze(0)
            return src_mask, tgt_mask
        
        return src_mask
    
    def forward(self, src, tgt=None):
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]（训练时提供）
        
        Returns:
            输出logits [batch_size, tgt_seq_len, vocab_size]
        """
        # 生成掩码
        if tgt is not None:
            src_mask, tgt_mask = self.generate_mask(src, tgt)
        else:
            src_mask = self.generate_mask(src)
            tgt_mask = None
        
        # 编码器
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        encoder_output = self.dropout(src_emb)
        
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)
        
        # 解码器（如果有目标序列）
        if tgt is not None:
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoding(tgt_emb)
            decoder_output = self.dropout(tgt_emb)
            
            for decoder_layer in self.decoder_layers:
                decoder_output = decoder_layer(decoder_output, encoder_output, src_mask, tgt_mask)
            
            # 输出投影
            output = self.output_projection(decoder_output)
            return output
        
        return encoder_output


# 为了简化，我们也提供一个只有编码器的版本用于语言建模
class TransformerLM(nn.Module):
    """用于语言建模的Transformer（仅编码器）"""
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_seq_length=5000, dropout=0.1):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            d_ff: 前馈网络隐藏层维度
            max_seq_length: 最大序列长度
            dropout: dropout比率
        """
        super(TransformerLM, self).__init__()
        
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, x):
        """生成因果掩码（用于语言建模）"""
        seq_length = x.size(1)
        mask = torch.tril(torch.ones(seq_length, seq_length)).bool().to(x.device)
        padding_mask = (x != 0).unsqueeze(1).unsqueeze(2)
        return mask.unsqueeze(0).unsqueeze(0) & padding_mask
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len]
        
        Returns:
            输出logits [batch_size, seq_len, vocab_size]
        """
        # 生成掩码
        mask = self.generate_mask(x)
        
        # 词嵌入 + 位置编码
        x_emb = self.embedding(x) * math.sqrt(self.d_model)
        x_emb = self.pos_encoding(x_emb)
        x = self.dropout(x_emb)
        
        # 编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output

