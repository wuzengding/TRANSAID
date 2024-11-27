# RNA Translation Sites Predictor

基于深度学习的RNA翻译起始位点和终止位点预测工具。

## 功能特点

- 预测RNA序列中的翻译起始位点和终止位点
- 考虑序列和结构特征的多模态分析
- 使用Transformer架构进行序列建模
- 支持GPU加速和分布式训练
- 提供完整的数据处理、训练和预测流程

## 安装要求

```bash
# 克隆项目
git clone https://github.com/yourusername/translation_site_predictor.git
cd translation_site_predictor

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备和特征提取

```bash
python scripts/encode.py \
    --fasta_file /path/to/sequences.fasta \
    --gbff_file /path/to/annotation.gbff \
    --output_dir /path/to/output \
    --max_seq_length 10000
```

### 2. 模型训练  

```bash
python scripts/train.py \
    --train_data /path/to/train_data.pkl \
    --val_data /path/to/val_data.pkl \
    --output_dir /path/to/model_output \
    --batch_size 32 \
    --epochs 50 \
    --fp16
```

### 3. 预测新序列

```bash
python scripts/predict.py \
    --model /path/to/best_model.pt \
    --input_file /path/to/new_sequences.fasta \
    --output_file /path/to/predictions.json
```

## 项目结构

```
translation_site_predictor/
├── configs/           # 配置文件
├── src/              # 源代码
│   ├── features/     # 特征提取
│   ├── models/       # 模型定义
│   └── data/         # 数据处理
├── scripts/          # 运行脚本
├── utils/            # 工具函数
└── tests/            # 单元测试
```

## 模型架构

- 多模态特征融合
- Transformer编码器
- 注意力机制
- CNN特征提取

## 性能指标

- 起始位点预测准确率：XX%
- 终止位点预测准确率：XX%
- F1分数：XX
- 平均精度：XX

## 引用

如果您使用了这个项目，请引用：

```bibtex
@software{translation_site_predictor,
  author = {Your Name},
  title = {RNA Translation Sites Predictor},
  year = {2024},
  url = {https://github.com/yourusername/translation_site_predictor}
}
```

## 许可证

MIT License

## 贡献

欢迎提交Pull Request或创建Issue。