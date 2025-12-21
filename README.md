CLEAR-VOX-MODEL
ClearVox: Accessible ASR for Chinese Dysarthric Speech

CLEAR-VOX-MODEL 是一个面向 中文构音障碍（Dysarthria） 场景的语音识别研究与工程项目，基于 阿里 FunASR / FunASR-Nano，通过阶段化训练策略，逐步构建 高可用、可扩展的无障碍语音识别系统。

1. 项目目标

🎯 提升构音障碍语音的 自动语音识别（ASR）准确率

🎯 支持 CDSD / MDSC 等障碍语音数据集

🎯 采用 轻量模型（FunASR-Nano）→ 高性能模型 的渐进式路线

🎯 可扩展到 ASR + GER（二阶段纠错） 架构

🎯 支持 单卡 RTX 3090 / 4090 本地训练

2. 推荐目录结构（CLEAR-VOX-MODEL）
CLEAR-VOX-MODEL/
│
├── README.md                      # 本文档
│
├── data/
│   └── cdsd/
│       ├── raw/                   # 原始 CDSD 数据（只读）
│       │   ├── audio/
│       │   └── text/
│       │
│       └── list/                  # 训练清单（脚本生成）
│           ├── transcripts.tsv
│           ├── train.jsonl
│           ├── dev.jsonl
│           ├── train_wav.scp
│           ├── train_text.txt
│           ├── train_utt2spk
│           ├── dev_wav.scp
│           ├── dev_text.txt
│           └── dev_utt2spk
│
├── scripts/
│   ├── prepare_cdsd_from_audiotext.py   # CDSD 接入脚本
│   └── utils/                           # 可选：文本清洗、统计等
│
├── exp/
│   ├── stage1_baseline/          # 阶段一：通用 ASR 基线
│   ├── stage2_domain_adapt/      # 阶段二：构音障碍适配
│   └── stage3_asr_ger/           # 阶段三：ASR + GER
│
└── tools/
    ├── infer_asr.py              # 推理脚本（可选）
    └── eval_cer.py               # CER 评测脚本（可选）

3. 环境与硬件要求
3.1 硬件建议
组件	推荐
GPU	RTX 4090 24GB × 1（Nano 可单卡）
CPU	≥ 8 核
内存	≥ 32GB（建议 64GB）
存储	≥ 200GB

FunASR-Nano（~0.8B）支持 单卡 LoRA / 小 batch 全量微调，无需 A100。

3.2 软件环境
conda create -n clearvox python=3.10 -y
conda activate clearvox

pip install torch torchaudio
pip install -U funasr modelscope huggingface_hub

4. 数据准备（CDSD）
4.1 原始数据结构（audio / text 分离）
data/cdsd/raw/
  audio/
    speaker_0001_xxx.wav
  text/
    speaker_0001_xxx.txt

4.2 生成训练清单（关键一步）
python scripts/prepare_cdsd_from_audiotext.py \
  --cdsd_root data/cdsd/raw \
  --out_dir data/cdsd/list \
  --dev_ratio 0.05 \
  --seed 42 \
  --make_jsonl \
  --match_mode relpath


生成的 data/cdsd/list/ 是 FunASR 唯一依赖的数据入口。

5. 模型训练的三阶段路线（核心设计）

CLEAR-VOX 采用 逐阶段演进 的训练策略，而不是“一步到位”。

阶段一：Baseline ASR（通用能力对齐）

目标
验证训练流程 & 数据是否正确，不追求最终指标。

模型

FunASR-Nano-2512（不做或少量微调）

训练配置重点

batch_type: token
batch_size: 400
max_epoch: 5~10
learning_rate: 2e-4


输出目录

exp/stage1_baseline/


你关注的指标

loss 是否正常下降

dev CER 是否 < 原始模型 CER

阶段二：构音障碍领域适配（最关键）

目标
让模型真正“听懂”构音障碍语音。

模型

FunASR-Nano + LoRA（推荐）

或 Nano 全量微调（显存允许时）

关键配置调整（非常重要）

项	建议	原因
batch_size	↓ 200~300	发音差异大，梯度不稳定
max_epoch	↑ 30~50	需要充分适应
learning_rate	1e-4 ~ 3e-4	防止灾难性遗忘
dev split	speaker-level	防止说话人泄漏

输出目录

exp/stage2_domain_adapt/

阶段三：ASR + GER（二阶段纠错）

目标
用语言模型修正 ASR 的结构性错误。

结构

audio
  → ASR（ClearVox-ASR）
    → N-best hypotheses
      → GER（文本纠错模型）
        → final transcript


GER 模型建议

Chinese T5 / BART / Qwen-7B（LoRA）

输入：ASR 输出

输出：修正文本

配置关注点

ASR 阶段：输出 N-best

GER 阶段：文本 max_length、beam size

输出目录

exp/stage3_asr_ger/

6. 不同阶段你“主要调什么”
快速对照表
阶段	你最常调的参数
Stage 1	batch_size, epoch
Stage 2	lr, epoch, speaker split
Stage 3	N-best size, GER 模型大小
7. 推理与评测

推理：FunASR AutoModel.generate

评测：CER（Character Error Rate）

建议：按 speaker 统计 CER 分布

8. 命名规范（推荐）

模型：

ClearVox-Nano-ASR

ClearVox-ASR-v2

实验：

stage2_cdsd_lora_lr2e4

论文系统名：

ClearVox: An Accessible ASR System for Chinese Dysarthric Speech

9. 项目状态

 数据接入（CDSD）

 FunASR-Nano 微调

 ASR + GER 集成

 多模型对比实验

 用户端应用

10. 下一步可以继续做的事

⬜ 自动 <NOISE> 策略对比实验

⬜ FireRedASR vs FunASR 对比

⬜ GER 模型蒸馏

⬜ Web / App 推理接口