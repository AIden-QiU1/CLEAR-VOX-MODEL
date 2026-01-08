# 🗣️ CLEAR-VOX-MODEL

> **构音障碍语音识别专用仓库** | Dysarthria Speech Recognition Repository
>
> 基于 [FunASR](https://github.com/modelscope/FunASR) 构建的构音障碍语音识别研究与训练平台

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

---

## 🎯 项目目标

为构音障碍患者打造**最专业、最实用**的语音识别系统：

- 📊 **系统性研究文档**：34+篇论文的深度分析与实验计划
- 🧠 **高效微调方法**：基于LoRA的低成本个性化适配
- 🔧 **端到端训练流程**：从数据处理到模型部署
- 📈 **持续实验追踪**：可复现的实验记录

---

## 📁 仓库结构

```
CLEAR-VOX-MODEL/
│
├── 📊 data/                    # 数据目录
│   └── cdsd/                  # CDSD中文构音障碍数据集
│       ├── 1h/               # 1小时子集
│       ├── 10h/              # 10小时子集
│       └── list/             # 数据列表
│
├── 📖 docs/                    # 文档
│   ├── tutorials/            # 训练教程
│   ├── benchmarks/           # 性能基准
│   ├── model_zoo/            # 模型列表
│   └── funasr/               # FunASR原始文档
│
├── 🔬 research/                # 研究文档 ⭐
│   ├── papers/               # 论文分析（5大主题）
│   ├── experiments/          # 实验计划与记录
│   ├── insights/             # 核心发现
│   └── resources/            # 工具资源
│
├── 🧠 funasr/                  # 核心ASR代码（FunASR）
│   ├── models/               # 模型定义
│   ├── train_utils/          # 训练工具
│   └── ...
│
├── 🧩 modules/                 # 扩展模块 ⭐
│   ├── tts/                  # TTS语音合成
│   ├── vc/                   # 声音转换
│   ├── dsr/                  # 语音重建
│   └── enhancement/          # 语音增强
│
├── 📜 scripts/                 # 训练脚本
│   ├── prepare_*.py          # 数据预处理
│   ├── finetune_*.sh         # 微调脚本
│   └── inference_*.py        # 推理脚本
│
├── 🚀 runtime/                 # 部署运行时
│   ├── onnxruntime/          # ONNX推理
│   ├── triton_gpu/           # GPU服务
│   └── websocket/            # 流式识别
│
├── 🔤 fun_text_processing/     # 文本后处理
│   ├── inverse_text_normalization/  # ITN
│   └── text_normalization/          # TN
│
└── 🧪 tests/                   # 单元测试
```

---

## 🚀 快速开始

### 1. 环境安装
```bash
pip install -e .
pip install peft deepspeed
```

### 2. 数据准备
```bash
# 处理10小时CDSD数据
python scripts/prepare_10h_dataset.py
```

### 3. 开始训练
```bash
# LoRA微调 Paraformer-large
bash scripts/finetune_paraformer_10h_optimized.sh
```

### 4. 推理测试
```bash
python scripts/inference_finetuned.py --checkpoint outputs/best
```

---

## 📚 研究文档导航

| 主题 | 描述 | 链接 |
|------|------|------|
| **数据增强** | TTS/VC合成、SpecAugment | [→](research/papers/data_augmentation/README.md) |
| **ASR适配** | LoRA、MoE、Perceiver-Prompt | [→](research/papers/asr_adaptation/README.md) |
| **LLM融合** | N-best重排、多模态 | [→](research/papers/llm_integration/README.md) |
| **语音重建** | DiffDSR、TTS增益 | [→](research/papers/speech_reconstruction/README.md) |
| **数据集** | CDSD、UASpeech | [→](research/papers/datasets/README.md) |

**核心发现**：[insights/key_findings.md](research/insights/key_findings.md)

---

## 📊 实验进度

| 实验 | 描述 | 状态 |
|------|------|------|
| EXP-001 | 基线测试 | 🔄 计划中 |
| EXP-002 | LoRA微调 | 🔄 计划中 |
| EXP-003 | 数据增强 | 🔄 计划中 |
| EXP-004 | LLM重排 | 🔄 计划中 |

详见 [research/experiments/](research/experiments/)

---

## 🔧 技术栈

| 组件 | 方案 |
|------|------|
| 基础ASR | Paraformer-large (220M) |
| 微调方法 | LoRA (rank=8) |
| 训练框架 | FunASR + DeepSpeed |
| 数据增强 | F5-TTS + SpecAugment |
| 后处理 | LLM N-best重排 |

---

## 📄 许可证

本项目代码采用 [CC BY-NC-SA 4.0](LICENSE) 许可证。

**使用限制**:
- ✅ 学术研究 / Academic research
- ✅ 个人学习 / Personal learning  
- ✅ 教育目的 / Educational purposes
- ❌ 商业产品 / Commercial products
- ❌ 付费服务 / Paid services

**模型许可**: 使用的 FunASR 模型遵循 [FunASR Model License](docs/funasr/MODEL_LICENSE)

---

## 🙏 致谢

本项目基于以下优秀开源项目构建：

### 核心框架
- [FunASR](https://github.com/modelscope/FunASR) - 阿里巴巴达摩院语音识别框架
  - Paraformer-large 预训练模型
  - 训练与推理工具链
  - [FunASR 原始文档](docs/funasr/)

### 数据集
- [CDSD](https://arxiv.org/pdf/2310.15930) - 中文构音障碍语音数据库 (Interspeech 2023)
- [UASpeech](http://www.isle.illinois.edu/sst/data/UASpeech/) - 英文构音障碍数据集

### 参考框架
- [ESPnet](https://github.com/espnet/espnet) - 端到端语音处理
- [SpeechBrain](https://github.com/speechbrain/speechbrain) - 语音AI工具包
- [Kaldi](https://github.com/kaldi-asr/kaldi) - 数据处理工具

### 研究论文
感谢 34+ 篇构音障碍语音识别领域论文的作者们，详见 [研究文档](research/)

---

## �� 联系

如有问题或建议，请提交 Issue 或 PR。
