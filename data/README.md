# 📊 数据目录

本目录按数据集组织，存放训练、验证、测试数据。

## 目录结构

```
data/
├── cdsd/                 # CDSD 中文构音障碍数据集
│   ├── 1h/              # 1小时子集
│   ├── 10h/             # 10小时子集
│   ├── full/            # 完整数据集 (待添加)
│   └── list/            # 数据列表文件
│
├── uaspeech/            # UASpeech 英文数据集 (待添加)
│   ├── train/
│   └── test/
│
└── README.md            # 本文件
```

## 数据格式

### 音频格式
- 采样率: 16kHz
- 格式: WAV (PCM 16-bit)
- 声道: 单声道

### 标注格式
使用 Kaldi 风格的文件:
- `wav.scp`: 音频路径索引
- `text`: 转录文本
- `utt2spk`: 句子到说话人映射
- `spk2utt`: 说话人到句子映射

## 数据集详情

参考 [研究文档 - 数据集](../research/papers/datasets/README.md)
