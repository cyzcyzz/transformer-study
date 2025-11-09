"""
使用示例
Usage Example
"""
import torch
from src.transformer import TransformerLM
from src.data_utils import load_text_data, build_vocab, create_dataloader

# 加载数据
print("加载数据...")
train_texts = load_text_data('data/train.txt')
print(f"训练样本数: {len(train_texts)}")

# 构建词汇表
print("\n构建词汇表...")
vocab, idx2char = build_vocab(train_texts)
vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}")

# 创建数据加载器
print("\n创建数据加载器...")
train_loader = create_dataloader(train_texts, vocab, batch_size=2, max_length=32, shuffle=False)

# 创建模型
print("\n创建模型...")
model = TransformerLM(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    max_seq_length=32,
    dropout=0.1
)

# 测试一个批次
print("\n测试一个批次...")
for input_ids, target_ids in train_loader:
    print(f"输入形状: {input_ids.shape}")
    print(f"目标形状: {target_ids.shape}")
    
    # 前向传播
    output = model(input_ids)
    print(f"输出形状: {output.shape}")
    break

print("\n示例运行成功！")

