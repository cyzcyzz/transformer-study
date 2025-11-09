# 快速开始指南

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 运行训练（Windows）

```bash
python -m src.train --data_path data/train.txt --val_data_path data/val.txt --epochs 10
```

或者使用脚本：
```bash
scripts\run.bat
```

## 3. 运行训练（Linux/Mac）

```bash
bash scripts/run.sh
```

## 4. 查看结果

训练完成后，结果将保存在 `results/` 目录下：

- `model.pt`: 保存的最佳模型
- `vocab.json`: 词汇表
- `training_curves.png`: 训练和验证损失曲线
- `training_results.json`: 训练结果统计
- `param_stats.json`: 模型参数统计

## 5. 测试模型

```bash
python test_model.py
```

## 6. 编译LaTeX报告

使用XeLaTeX编译（推荐）：
```bash
xelatex report.tex
```

或使用pdfLaTeX：
```bash
pdflatex report.tex
```

## 注意事项

1. 确保Python版本为3.12
2. 如果使用GPU，确保已安装CUDA版本的PyTorch
3. 训练时间取决于硬件配置
4. 建议先用小数据集验证代码正确性

