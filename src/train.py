"""
训练脚本
Training Script
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from .transformer import TransformerLM
from .data_utils import load_text_data, build_vocab, create_dataloader


def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for input_ids, target_ids in tqdm(dataloader, desc="Training"):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # 前向传播
        logits = model(input_ids)
        
        # 计算损失（只计算非padding位置）
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def save_model(model, optimizer, epoch, loss, filepath):
    """保存模型"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_model(model, optimizer, filepath, device):
    """加载模型"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Model')
    parser.add_argument('--data_path', type=str, default='data/train.txt',
                      help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default='data/val.txt',
                      help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--d_model', type=int, default=256,
                      help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                      help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=1024,
                      help='Feed-forward dimension')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                      help='Maximum gradient norm for clipping')
    parser.add_argument('--save_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--model_save_path', type=str, default='results/model.pt',
                      help='Path to save model')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    print('Loading data...')
    train_texts = load_text_data(args.data_path)
    val_texts = load_text_data(args.val_data_path)
    
    # 构建词汇表
    print('Building vocabulary...')
    vocab, idx2char = build_vocab(train_texts + val_texts)
    vocab_size = len(vocab)
    print(f'Vocabulary size: {vocab_size}')
    
    # 保存词汇表
    with open(os.path.join(args.save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # 创建数据加载器
    train_loader = create_dataloader(train_texts, vocab, args.batch_size, args.max_length, shuffle=True)
    val_loader = create_dataloader(val_texts, vocab, args.batch_size, args.max_length, shuffle=False)
    
    # 创建模型
    print('Creating model...')
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_length,
        dropout=args.dropout
    ).to(device)
    
    # 统计参数
    num_params = count_parameters(model)
    print(f'Model parameters: {num_params:,}')
    
    # 保存参数统计
    param_stats = {
        'total_parameters': num_params,
        'model_config': {
            'vocab_size': vocab_size,
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'd_ff': args.d_ff,
            'max_length': args.max_length,
            'dropout': args.dropout
        }
    }
    with open(os.path.join(args.save_dir, 'param_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(param_stats, f, indent=2)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print('Starting training...')
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.max_grad_norm)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss, args.model_save_path)
            print(f'Model saved with validation loss: {val_loss:.4f}')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, os.path.join(args.save_dir, 'training_curves.png'))
    
    # 保存训练结果
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }
    with open(os.path.join(args.save_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print('\nTraining completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()

