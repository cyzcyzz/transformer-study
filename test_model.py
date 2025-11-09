"""
简单的模型测试脚本
Simple model test script
"""
import torch
from src.transformer import TransformerLM

# 测试模型创建
print("测试模型创建...")
vocab_size = 1000
d_model = 256
num_heads = 8
num_layers = 4
d_ff = 1024
max_length = 128

model = TransformerLM(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_length,
    dropout=0.1
)

print(f"模型创建成功！")
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 测试前向传播
print("\n测试前向传播...")
batch_size = 2
seq_length = 10
x = torch.randint(0, vocab_size, (batch_size, seq_length))

with torch.no_grad():
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出应该是: [{batch_size}, {seq_length}, {vocab_size}]")
    print("前向传播测试通过！")

print("\n所有测试通过！")

