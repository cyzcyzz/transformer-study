#!/bin/bash

# Transformer训练脚本
# Training script for Transformer

echo "开始训练Transformer模型..."
echo "Starting Transformer training..."

python -m src.train \
    --data_path data/train.txt \
    --val_data_path data/val.txt \
    --batch_size 32 \
    --epochs 10 \
    --lr 1e-4 \
    --d_model 256 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 1024 \
    --max_length 128 \
    --dropout 0.1 \
    --max_grad_norm 1.0 \
    --save_dir results \
    --seed 42

echo "训练完成！"
echo "Training completed!"

