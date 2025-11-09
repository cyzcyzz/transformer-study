"""
数据处理工具
Data Processing Utilities
"""
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, texts, vocab, max_length=128):
        """
        Args:
            texts: 文本列表
            vocab: 词汇表字典
            max_length: 最大序列长度
        """
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 将文本转换为token ID序列
        tokens = [self.vocab.get(char, self.vocab['<UNK>']) for char in text]
        
        # 截断或填充到固定长度
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_length - len(tokens))
        
        # 输入是除了最后一个token的所有token
        # 目标是从第二个token开始的所有token
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids


def build_vocab(texts, min_freq=1):
    """
    构建词汇表
    
    Args:
        texts: 文本列表
        min_freq: 最小词频
    
    Returns:
        词汇表字典和反向词汇表
    """
    # 统计字符频率
    counter = Counter()
    for text in texts:
        counter.update(text)
    
    # 构建词汇表
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for char, freq in counter.items():
        if freq >= min_freq:
            vocab[char] = idx
            idx += 1
    
    # 反向词汇表
    idx2char = {idx: char for char, idx in vocab.items()}
    
    return vocab, idx2char


def load_text_data(file_path):
    """
    加载文本数据
    
    Args:
        file_path: 文本文件路径
    
    Returns:
        文本列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    # 简单清理：去除换行符和空行
    texts = [text.strip() for text in texts if text.strip()]
    
    return texts


def create_dataloader(texts, vocab, batch_size=32, max_length=128, shuffle=True):
    """
    创建数据加载器
    
    Args:
        texts: 文本列表
        vocab: 词汇表
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱
    
    Returns:
        DataLoader
    """
    dataset = TextDataset(texts, vocab, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

