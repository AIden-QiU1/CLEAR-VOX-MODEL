# CLEAR-VOX-MODEL

> **构音障碍语音识别专用仓库** | Dysarthria Speech Recognition Repository
>
> 基于 [FunASR](https://github.com/modelscope/FunASR) 构建的构音障碍语音识别研究与训练平台

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

---

## 项目目标

为构音障碍患者打造**最专业、最实用**的语音识别系统：

- 系统性研究文档：34+篇论文的深度分析与实验计划
- 高效微调方法：基于LoRA的低成本个性化适配
- 端到端训练流程：从数据处理到模型部署
- 持续实验追踪：可复现的实验记录

---

## 仓库结构

```
CLEAR-VOX-MODEL/
│
├── data/                    # 数据目录
│   └── cdsd/                  # CDSD中文构音障碍数据集
│       ├── 1h/               # 1小时子集
│       ├── 10h/              # 10小时子集
│       └── list/             # 数据列表
│
├── docs/                    # 文档
│   ├── tutorials/            # 训练教程
│   ├── benchmarks/           # 性能基准
│   ├── model_zoo/            # 模型列表
│   └── funasr/               # FunASR原始文档
│
├── research/                # 研究文档 [重点]
│   ├── papers/               # 论文分析（5大主题）
│   ├── experiments/          # 实验计划与记录
│   ├── insights/             # 核心发现
│   └── resources/            # 工具资源
│
├── funasr/                  # 核心ASR代码（FunASR）
│   ├── models/               # 模型定义
│   ├── train_utils/          # 训练工具
│   └── ...
│
├── modules/                 # 扩展模块 [重点]
│   ├── tts/                  # TTS语音合成
│   ├── vc/                   # 声音转换
│   ├── dsr/                  # 语音重建
│   └── enhancement/          # 语音增强
│
├── scripts/                 # 训练脚本
│   ├── prepare_*.py          # 数据预处理
│   ├── finetune_*.sh         # 微调脚本
│   └── inference_*.py        # 推理脚本
│
├── runtime/                 # 部署运行时
│   ├── onnxruntime/          # ONNX推理
│   ├── triton_gpu/           # GPU服务
│   └── websocket/            # 流式识别
│
├── fun_text_processing/     # 文本后处理
│   ├── inverse_text_normalization/  # ITN
│   └── text_normalization/          # TN
│
└── tests/                   # 单元测试
```

---

## 快速开始

### 环境依赖

在开始使用之前，请确保已经安装以下基础环境：

#### 基础要求

```text
python >= 3.8
torch >= 1.13
torchaudio >= 0.13
```

#### 推荐的安装方式

**方式一：直接安装（推荐）**

```bash
# 安装本项目（包含FunASR及其依赖）
pip install -e .

# 安装微调相关依赖
pip install peft deepspeed
```

说明：运行 `pip install -e .` 会根据 setup.py 自动安装以下内容：
- FunASR核心库及其依赖（scipy, librosa, soundfile, modelscope等）
- 命令行工具（funasr, funasr-train, funasr-export等）
- 数据处理工具（scp2jsonl, jsonl2scp等）

**方式二：分步安装**

```bash
# 先安装FunASR
pip3 install -U funasr

# 再安装项目依赖
pip install -e .

# 安装微调相关依赖
pip install peft deepspeed
```

#### GPU 支持

如果需要使用 GPU 加速，请根据你的 CUDA 版本安装对应的 PyTorch：

```bash
# CUDA 11.8
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 验证安装

```python
# 验证 FunASR 安装
python3 -c "import funasr; print(funasr.__version__)"

# 验证 PyTorch 和 CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 数据准备

```bash
# 处理10小时CDSD数据
python scripts/prepare_10h_dataset.py

# 或者处理1小时CDSD数据（快速实验）
python scripts/prepare_1h_dataset.py
```

### 开始训练

```bash
# LoRA微调 Paraformer-large（10小时数据）
bash scripts/finetune_paraformer_10h_optimized.sh

# 或使用1小时数据快速实验
bash scripts/finetune_paraformer_optimized.sh
```

### 推理测试

```bash
# 使用微调后的模型进行推理
python scripts/inference_finetuned.py --checkpoint outputs/best

# 或使用测试脚本
python scripts/inference_test.py
```

---

## 研究文档导航

| 主题 | 描述 | 链接 |
|------|------|------|
| **数据增强** | TTS/VC合成、SpecAugment | [查看](research/papers/data_augmentation/README.md) |
| **ASR适配** | LoRA、MoE、Perceiver-Prompt | [查看](research/papers/asr_adaptation/README.md) |
| **LLM融合** | N-best重排、多模态 | [查看](research/papers/llm_integration/README.md) |
| **语音重建** | DiffDSR、TTS增益 | [查看](research/papers/speech_reconstruction/README.md) |
| **数据集** | CDSD、UASpeech | [查看](research/papers/datasets/README.md) |

**核心发现**：[insights/key_findings.md](research/insights/key_findings.md)

---

## 实验进度

| 实验 | 描述 | 状态 |
|------|------|------|
| EXP-001 | 基线测试 | 计划中 |
| EXP-002 | LoRA微调 | 计划中 |
| EXP-003 | 数据增强 | 计划中 |
| EXP-004 | LLM重排 | 计划中 |

详见 [research/experiments/](research/experiments/)

---

## 技术栈

| 组件 | 方案 |
|------|------|
| 基础ASR | Paraformer-large (220M) |
| 微调方法 | LoRA (rank=8) |
| 训练框架 | FunASR + DeepSpeed |
| 数据增强 | F5-TTS + SpecAugment |
| 后处理 | LLM N-best重排 |

---

## FunASR 核心功能

FunASR 是一个基础语音识别工具包，提供以下核心功能：

- **语音识别（ASR）**：支持中英文及多语言语音识别
- **语音端点检测（VAD）**：自动检测语音的起始和结束位置
- **标点恢复**：为识别结果自动添加标点符号
- **说话人验证/分离**：识别和区分不同说话人
- **多人对话识别**：支持多说话人场景的语音识别
- **实时流式识别**：支持低延迟的实时语音识别

FunASR 在 [ModelScope](https://www.modelscope.cn/models?page=1&tasks=auto-speech-recognition) 和 [Hugging Face](https://huggingface.co/FunASR) 上发布了大量工业级预训练模型。

---

## 许可证

本项目代码采用 [CC BY-NC-SA 4.0](LICENSE) 许可证。

**使用限制**:
- 允许：学术研究、个人学习、教育目的
- 禁止：商业产品、付费服务

**模型许可**: 使用的 FunASR 模型遵循 [FunASR Model License](docs/funasr/MODEL_LICENSE)

---

## 致谢

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

## 联系

如有问题或建议，请提交 Issue 或 PR。
