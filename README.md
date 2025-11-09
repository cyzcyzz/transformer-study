# Transformer 实现作业

这是一个完整的Transformer模型实现，用于大模型课程作业。

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── attention.py       # 多头自注意力机制
│   ├── feed_forward.py    # 位置前馈网络
│   ├── positional_encoding.py  # 位置编码
│   ├── layer_norm.py      # 层归一化
│   ├── transformer.py     # Transformer模型
│   ├── data_utils.py      # 数据处理工具
│   └── train.py           # 训练脚本
├── data/                   # 数据目录
│   ├── train.txt          # 训练数据
│   └── val.txt            # 验证数据
├── results/                # 结果目录（训练后生成）
│   ├── model.pt           # 保存的模型
│   ├── vocab.json         # 词汇表
│   ├── training_curves.png # 训练曲线图
│   ├── training_results.json  # 训练结果
│   └── param_stats.json   # 参数统计
├── scripts/                # 脚本目录
│   └── run.sh             # 运行脚本
├── requirements.txt        # 依赖包
└── README.md              # 本文件
```

## 环境要求

- Python 3.12
- PyTorch >= 2.0.0
- 其他依赖见 `requirements.txt`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 硬件要求

- **最低要求**: CPU即可运行（训练速度较慢）
- **推荐配置**: 具有CUDA支持的GPU（NVIDIA GPU，显存 >= 2GB）

## 运行方法

### 方法1: 使用运行脚本（Linux/Mac）

```bash
bash scripts/run.sh
```

### 方法2: 直接运行Python脚本（Windows）

```bash
python -m src.train --data_path data/train.txt --val_data_path data/val.txt --epochs 10 --batch_size 32
```

### 方法3: 使用命令行参数

```bash
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
```

## 主要参数说明

- `--data_path`: 训练数据路径
- `--val_data_path`: 验证数据路径
- `--batch_size`: 批次大小（默认32）
- `--epochs`: 训练轮数（默认10）
- `--lr`: 学习率（默认1e-4）
- `--d_model`: 模型维度（默认256）
- `--num_heads`: 注意力头数（默认8）
- `--num_layers`: 编码器层数（默认4）
- `--d_ff`: 前馈网络隐藏层维度（默认1024）
- `--max_length`: 最大序列长度（默认128）
- `--dropout`: Dropout比率（默认0.1）
- `--max_grad_norm`: 梯度裁剪阈值（默认1.0）
- `--save_dir`: 结果保存目录（默认results）
- `--seed`: 随机种子（默认42）

## 实现的功能

### 基础功能（60-70分）
- ✅ 多头自注意力机制（Multi-Head Self-Attention）
- ✅ 位置前馈网络（Position-wise FFN）
- ✅ 残差连接 + LayerNorm
- ✅ 位置编码（Positional Encoding）

### 进阶功能（70-80分）
- ✅ 完整的Transformer编码器
- ✅ 训练稳定性技术：
  - ✅ 学习率调度（CosineAnnealingLR）
  - ✅ 梯度裁剪（Gradient Clipping）
  - ✅ AdamW优化器
- ✅ 参数统计
- ✅ 模型保存/加载
- ✅ 训练曲线可视化

### 高级功能（80-90分）
- ✅ Transformer解码器实现
- ✅ 完整的Encoder-Decoder架构

## 实验结果

训练完成后，结果将保存在 `results/` 目录下：

- `model.pt`: 保存的最佳模型
- `vocab.json`: 词汇表
- `training_curves.png`: 训练和验证损失曲线
- `training_results.json`: 训练结果统计
- `param_stats.json`: 模型参数统计

## 代码说明

### 核心组件

1. **多头自注意力** (`src/attention.py`): 实现了缩放点积注意力机制
2. **前馈网络** (`src/feed_forward.py`): 实现了位置前馈网络
3. **位置编码** (`src/positional_encoding.py`): 实现了正弦位置编码
4. **层归一化** (`src/layer_norm.py`): 实现了层归一化和残差连接
5. **Transformer模型** (`src/transformer.py`): 完整的Transformer实现

### 训练流程

1. 加载和预处理数据
2. 构建词汇表
3. 创建数据加载器
4. 初始化模型
5. 训练循环（包含梯度裁剪、学习率调度）
6. 验证和保存模型
7. 绘制训练曲线

## 复现实验

使用以下命令可以完全复现实验结果：

```bash
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
```

## 注意事项

1. 数据格式：每行一个文本样本
2. 如果使用GPU，确保已安装CUDA版本的PyTorch
3. 训练时间取决于硬件配置和数据集大小
4. 建议先用小数据集验证代码正确性

## 作者

大模型课程作业

## 许可证

本项目仅用于学习目的。

