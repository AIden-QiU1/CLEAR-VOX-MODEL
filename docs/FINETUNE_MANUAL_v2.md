# 🎯 FunASR 构音障碍语音识别微调手册 v2.0

> **硬件**: RTX 3090 24GB | **框架**: FunASR | **目标**: 中文构音障碍 ASR
> 
> **更新**: 基于官方配置校验 + CDSD论文 (INTERSPEECH 2024) 优化

---

## 📋 目录
1. [模型选择](#1-模型选择)
2. [环境配置](#2-环境配置)
3. [数据准备](#3-数据准备)
4. [训练配置](#4-训练配置) ⚠️ **已更新**
5. [执行训练](#5-执行训练)
6. [模型评测](#6-模型评测)
7. [常见问题](#7-常见问题)
8. [参考文献](#8-参考文献) 🆕

---

## 1. 模型选择

### 🏆 推荐模型对比

| 模型 | 参数量 | 3090显存占用 | 特点 | 推荐度 |
|------|--------|--------------|------|--------|
| **Paraformer-large** | 220M | ~8GB(batch4000) | 非自回归，速度快，精度高 | ⭐⭐⭐⭐⭐ |
| SenseVoice-Small | 330M | ~12GB | 多功能(ASR+情感) | ⭐⭐⭐⭐ |
| Conformer-12e6d | ~100M | ~6GB | 经典架构，易调优 | ⭐⭐⭐ |
| Fun-ASR-Nano | 800M | >20GB | 最新最强，但无训练代码 | ❌ |

### ✅ 最终选择：Paraformer-large

**理由**：
- 非自回归架构，推理速度快10倍
- 220M参数，3090可全量微调
- Aishell1 test CER: 1.94%（SOTA水平）
- 60000小时中文预训练
- 完整的微调代码支持

```bash
# ModelScope 模型ID
model_id="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
```

---

## 2. 环境配置

### 2.1 创建虚拟环境
```bash
conda create -n funasr python=3.10 -y
conda activate funasr
```

### 2.2 安装依赖
```bash
# PyTorch with CUDA
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# FunASR 核心
pip install -U funasr modelscope

# 可选加速
pip install deepspeed
```

### 2.3 验证安装
```python
from funasr import AutoModel
model = AutoModel(model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
print("Installation OK!")
```

### 2.4 模型下载（可选，加速后续训练）
```bash
# 自动下载到 ~/.cache/modelscope/
python -c "from funasr import AutoModel; AutoModel(model='iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')"
```

---

## 3. 数据准备

### 3.1 数据格式
FunASR 支持两种格式：

**JSONL格式（推荐）**：
```json
{"key": "utt_001", "source": "/path/to/audio.wav", "target": "转录文本"}
```

**Kaldi格式**：
```
# wav.scp
utt_001 /path/to/audio.wav

# text.txt  
utt_001 转录文本
```

### 3.2 执行数据准备
```bash
cd /root/CLEAR-VOX-MODEL
python scripts/prepare_1h_dataset.py
```

### 3.3 验证数据
```bash
# 检查生成的文件
ls -la data/1h_dataset/

# 查看统计信息
cat data/1h_dataset/data_statistics.txt
```

### 3.4 1h 数据集统计
| 划分 | 说话人数 | 语音条数 |
|------|----------|----------|
| 训练集 | 35 | 45,327 |
| 验证集 | 4 | 4,460 |
| 测试集 | 5 | 6,064 |
| **总计** | **44** | **55,851** |

---

## 4. 训练配置 ⚠️ 已校验更新

### 4.1 关键参数对比（官方 vs 3090优化）

| 参数 | 官方默认 | 3090优化 | 说明 |
|------|----------|----------|------|
| batch_size | 6000 tokens | 4000-6000 | 3090单卡可尝试6000 |
| **learning_rate** | **0.0002** | **0.0002** | ⚠️ 官方推荐，比之前0.0001更高 |
| max_epoch | 50 | 50-100 | 小数据集可适当增加 |
| validate_interval | 2000 | 2000 | 每2000步验证 |
| keep_nbest_models | 20 | 10 | 3090存储优化 |
| **avg_nbest_model** | **10** | **10** | 🆕 最佳N个模型平均 |
| **sort_size** | **1024** | **1024** | 🆕 排序缓冲区大小 |
| **data_split_num** | **1** | **1** | 数据切片数（大数据集可增大） |

### 4.2 完整训练脚本 (v2.0)

```bash
#!/bin/bash
# /root/CLEAR-VOX-MODEL/scripts/finetune_paraformer.sh
# Version 2.0 - 基于官方配置校验

export CUDA_VISIBLE_DEVICES="0"

workspace=/root/CLEAR-VOX-MODEL
model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

train_data="${workspace}/data/1h_dataset/train.jsonl"
val_data="${workspace}/data/1h_dataset/val.jsonl"
output_dir="${workspace}/exp/paraformer_finetune_1h"

mkdir -p ${output_dir}

torchrun --nproc_per_node=1 \
${workspace}/funasr/bin/train_ds.py \
++model="${model}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset="AudioDataset" \
++dataset_conf.index_ds="IndexDSJsonl" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=6000 \
++dataset_conf.sort_size=1024 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=50 \
++train_conf.log_interval=50 \
++train_conf.resume=true \
++train_conf.validate_interval=2000 \
++train_conf.save_checkpoint_interval=2000 \
++train_conf.keep_nbest_models=10 \
++train_conf.avg_nbest_model=10 \
++train_conf.use_deepspeed=false \
++optim_conf.lr=0.0002 \
++output_dir="${output_dir}" \
2>&1 | tee ${output_dir}/train.log
```

### 4.3 与官方配置差异说明

| 配置项 | 官方值 | 我们的值 | 原因 |
|--------|--------|----------|------|
| CUDA_VISIBLE_DEVICES | "0,1" | "0" | 3090单卡 |
| batch_size | 6000 | 6000 | 官方推荐 |
| learning_rate | 0.0002 | 0.0002 | ✅ 已修正，官方推荐 |
| keep_nbest_models | 20 | 10 | 节省存储 |
| resume | true | true | 支持断点续训 |

---

## 5. 执行训练

### 5.1 启动训练
```bash
cd /root/CLEAR-VOX-MODEL
bash scripts/finetune_paraformer.sh
```

### 5.2 监控训练
```bash
# 实时查看日志
tail -f exp/paraformer_finetune_1h/train.log

# 查看GPU使用
watch -n 1 nvidia-smi
```

### 5.3 训练输出
```
exp/paraformer_finetune_1h/
├── model.pt.ep0           # epoch 0 模型
├── model.pt.ep1           # epoch 1 模型
├── ...
├── model.pt.avg_10        # 最佳10个模型平均
└── train.log              # 训练日志
```

---

## 6. 模型评测

### 6.1 推理测试
```bash
python scripts/inference_test.py \
  --model exp/paraformer_finetune_1h/model.pt.avg_10 \
  --test data/1h_dataset/test.jsonl \
  --output exp/paraformer_finetune_1h/test_results.json
```

### 6.2 基线对比
```bash
# 测试原始模型
python scripts/inference_test.py \
  --model "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
  --test data/1h_dataset/test.jsonl \
  --output exp/baseline_results.json
```

### 6.3 预期效果（基于CDSD论文参考）

| 模型 | 预期 CER | 说明 |
|------|----------|------|
| Paraformer-large (原始) | 25-35% | 未适应构音障碍 |
| Paraformer-large (微调) | 16-22% | 微调后 |
| **CDSD论文最佳** | **16.4%** | Hybrid CTC/Attention |
| 人类评估者 | 20.45% | CDSD论文人类baseline |

> **注意**: 微调后 CER 优于人类评估者 (20.45%) 是合理目标

---

## 7. 常见问题

### Q1: 显存不足 (OOM)
```bash
# 降低batch_size
++dataset_conf.batch_size=4000
# 或降到2000
++dataset_conf.batch_size=2000
```

### Q2: 模型下载失败
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
pip install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

### Q3: 训练不收敛
```bash
# 降低学习率（但先尝试官方0.0002）
++optim_conf.lr=0.0001
# 增加warmup
++scheduler_conf.warmup_steps=1000
```

### Q4: 想使用LoRA微调
目前 FunASR 原生不支持 LoRA，需使用 PEFT 库：
```python
from peft import LoraConfig, get_peft_model
# 需要自定义训练脚本
```

### Q5: 断点续训
```bash
# 已配置 resume=true，自动从最新checkpoint继续
# 如需指定checkpoint:
++init_param="${output_dir}/model.pt.ep10"
```

---

## 8. 参考文献 🆕

### 8.1 CDSD 数据集论文
> **CDSD: Chinese Dysarthria Speech Database**
> - 会议: INTERSPEECH 2024
> - 规模: 133 小时，44位说话人
> - 最佳结果: CER 16.4% (Hybrid CTC/Attention)
> - 人类基线: CER 20.45%
> - arXiv: https://arxiv.org/abs/2310.15930

### 8.2 FunASR 框架
> - GitHub: https://github.com/modelscope/FunASR
> - 文档: https://funasr.readthedocs.io
> - ModelScope: https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

### 8.3 Paraformer 论文
> **Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition**
> - arXiv: https://arxiv.org/abs/2206.08317

---

## 📊 训练流程图

```
┌────────────────────────────────────────────────────────────────┐
│              FunASR 构音障碍 ASR 微调 v2.0                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. 数据准备                                                    │
│     Audio + Text → prepare_1h_dataset.py → train/val/test.jsonl│
│     └─ 45,327 训练 / 4,460 验证 / 6,064 测试                    │
│                                                                │
│  2. 模型选择                                                    │
│     Paraformer-large (220M) ← 推荐                             │
│                                                                │
│  3. 训练配置 (v2.0 校验后)                                       │
│     batch=6000, lr=0.0002, epoch=50, avg_nbest=10              │
│                                                                │
│  4. 执行训练                                                    │
│     torchrun → train_ds.py → model.pt.avg_10                   │
│                                                                │
│  5. 评测                                                        │
│     inference_test.py → CER% (目标 < 20.45% 人类基线)           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**作者**: GitHub Copilot  
**日期**: 2025-12-23  
**版本**: v2.0 (基于官方配置校验 + CDSD论文优化)
