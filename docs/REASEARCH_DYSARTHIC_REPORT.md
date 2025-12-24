# 📊 构音障碍语音识别 (Dysarthric ASR) 技术调研报告

> **CLEAR-VOX Project Research Report**  
> 版本: 1.0 | 日期: 2025-12-23  
> 作者: CLEAR-VOX Research Team

---

## 📋 目录

1. [模型版本与现状](#1-模型版本与现状)
2. [2024-2025 最新研究进展](#2-2024-2025-最新研究进展)
3. [大语言模型在构音障碍ASR的应用](#3-大语言模型在构音障碍asr的应用)
4. [主流训练策略对比](#4-主流训练策略对比)
5. [针对CDSD+Paraformer的优化建议](#5-针对cdsdparaformer的优化建议)
6. [参考文献](#6-参考文献)

---

## 1. 模型版本与现状

### 1.1 Paraformer-large 模型信息

| 属性 | 值 |
|------|-----|
| **模型ID** | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| **当前版本** | v2.0.4 (最新稳定版) |
| **发布时间** | 2024-02-01 |
| **参数量** | 220M |
| **预训练数据** | 60,000+ 小时中文语音 |
| **下载量** | 32,005,526+ 次 |

**结论**: 这是目前最新的官方版本，无更新版本发布。

### 1.2 Paraformer 发展历程

| 时间 | 事件 |
|------|------|
| 2022年6月 | Paraformer 论文发表 (INTERSPEECH 2022) |
| 2023年1月 | FunASR v0.1.6: Paraformer-large 开源 |
| 2023年3月 | FunASR v0.3.0: 流式模型支持 |
| 2024年2月 | v2.0.4: 当前最新版本 |

### 1.3 架构组成

```
Paraformer-large 架构 (220M 参数)
├── SANMEncoder (158M, 72%)     - 50层 SANM 自注意力
├── ParaformerSANMDecoder (61M, 28%) - 16层非自回归解码
├── CifPredictorV2 (0.8M)       - CIF 长度预测
└── SpecAugLFR                  - 频谱数据增强
```

---

## 2. 2024-2025 最新研究进展

### 2.1 重要论文汇总

#### 🔥 [2025.12] Zero-Shot Recognition using MLLM
**arXiv:2512.17474** - Ali Alsayegh et al.

> **核心发现**: 
> - 评测 8 个商用 ASR 系统在 TORGO 构音障碍数据集的表现
> - 轻度构音障碍: WER 3-5% (接近正常语音)
> - 重度构音障碍: WER > 49%
> - **GPT-4o 使用 verbatim-transcription prompt 可降低 7.36% WER**

**对本项目的启示**: 可以尝试使用 LLM 后处理 ASR 输出结果。

---

#### 🔥 [2024.07] Prototype-Based Adaptation for Unseen Speakers
**arXiv:2407.18461** - INTERSPEECH 2024

> **核心方法**:
> - 使用 HuBERT 特征提取器
> - 构建 per-word prototypes (原型)
> - 监督对比学习优化特征
> - **无需微调即可适应新说话人**

**开源代码**: https://github.com/NKU-HLT/PB-DSR

**对本项目的启示**: 可以考虑基于原型的适应策略。

---

#### 🔥 [2024] Fine-Tuning Strategies for Dutch DSR
**INTERSPEECH 2024** - Leivaditi et al.

> **核心发现** (荷兰语构音障碍):
> - 比较 3 种微调策略:
>   1. Healthy speech → Dysarthric speech
>   2. Disease-specific data only
>   3. Speaker-specific adaptation
> - **Speaker-specific 效果最好，但数据需求高**
> - **Self-supervised learning (SSL) 预训练特征有帮助**

---

#### 🔥 [2024] Speech Technology for DSR: An Overview
**Journal of Speech, Language, and Hearing Research, 2025**

> **综述要点**:
> - Transfer Learning (TL) 是最有效的技术
> - 不同 source domain 效果差异大
> - 数据增强可以缓解数据稀缺问题
> - **显著降低 WER 的关键是针对性的领域适应**

---

#### 🔥 [2024] SLT 2024 LRDWWS Challenge
**IEEE SLT 2024** - 构音障碍唤醒词检测挑战赛

> **背景**: LRDWWS = Low-Resource Dysarthric Wake Word Spotting
> - 专门针对构音障碍的端到端方法
> - 使用 CDSD 数据集的子集
> - 端到端方法优于传统级联方法

---

### 2.2 大语言模型应用

| 方法 | 模型 | WER (轻度) | WER (重度) | 说明 |
|------|------|-----------|-----------|------|
| 传统 ASR | Whisper large-v3 | ~5% | ~45% | 无领域适应 |
| 传统 ASR | Deepgram Nova-3 | ~4% | ~40% | 商用服务 |
| **MLLM** | **GPT-4o** | ~3% | ~42% | 使用 verbatim prompt |
| MLLM | Gemini 2.5 Pro | ~5% | ~48% | 无明显改进 |

**关键发现**: 
- MLLM 在重度构音障碍上仍然表现不佳
- GPT-4o 的 verbatim prompt 技巧值得尝试
- 语义可恢复性比字面准确性更重要

---

## 3. 大语言模型在构音障碍ASR的应用

### 3.1 MLLM 后处理方案

```python
# 概念示意 - 使用 LLM 纠错
import openai

def asr_with_llm_correction(audio_path, asr_model, llm_client):
    # 1. ASR 识别
    raw_text = asr_model.transcribe(audio_path)
    
    # 2. LLM 纠错
    prompt = f"""
    以下是构音障碍患者的语音识别结果，可能存在错误。
    请根据上下文语义进行纠正，保留原意：
    
    识别结果: {raw_text}
    纠正后:
    """
    
    corrected = llm_client.complete(prompt)
    return corrected
```

### 3.2 多模态融合方向

```
未来趋势:
Audio → [ASR Encoder] → Text Embedding
                              ↓
                        [LLM Decoder] → 纠正后文本
                              ↑
Context → [Context Encoder] → Context Embedding
```

---

## 4. 主流训练策略对比

### 4.1 策略对比表

| 策略 | 优点 | 缺点 | 数据需求 | 推荐度 |
|------|------|------|----------|--------|
| **直接微调** | 简单高效 | 易过拟合 | 中 | ⭐⭐⭐⭐ |
| 课程学习 | 渐进适应 | 需要多阶段 | 高 | ⭐⭐⭐ |
| 数据增强 | 扩充数据 | 可能引入噪声 | 低 | ⭐⭐⭐⭐ |
| 原型学习 | 泛化性好 | 实现复杂 | 中 | ⭐⭐⭐⭐ |
| Speaker Adaptation | 效果最佳 | 需要每人数据 | 极高 | ⭐⭐⭐ |
| **对比学习** | 特征更好 | 需要负样本 | 中 | ⭐⭐⭐⭐⭐ |

### 4.2 各策略详解

#### 策略1: 直接微调 (当前方案)
```bash
# 优点: 简单直接
# 缺点: 小数据集易过拟合
torchrun funasr/bin/train_ds.py \
++model="paraformer-large" \
++train_data_set_list="train.jsonl"
```

#### 策略2: 数据增强 (推荐添加)
```python
# FunASR 内置 SpecAugment
# 可额外添加:
- Speed Perturbation: [0.9, 1.0, 1.1]
- Volume Perturbation
- 不建议: 过度噪声 (构音障碍本身就是"噪声")
```

#### 策略3: 对比学习 (SOTA方法)
```python
# 基于 HuBERT 的监督对比学习
# 参考: arXiv:2407.18461
loss = CE_loss + λ * Contrastive_loss
```

---

## 5. 针对CDSD+Paraformer的优化建议

### 5.1 推荐策略 (不改变学习率和轮次)

基于文献调研，我们推荐以下两个优化策略：

---

### ⭐ 策略A: 数据增强 (Speed Perturbation)

**原理**: 构音障碍语速异常，通过 speed perturbation 增加数据多样性

**实现方式**: 修改训练配置

```yaml
# 在 dataset_conf 中添加
dataset_conf:
  preprocessor_speech: SpeechPreprocessSpeedPerturb
  preprocessor_speech_conf:
    speed_perturb: [0.9, 1.0, 1.1]  # 0.9x, 1.0x, 1.1x 三种速度
```

**修改训练命令**:
```bash
torchrun --nproc_per_node=1 \
/root/CLEAR-VOX-MODEL/funasr/bin/train_ds.py \
++model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
++train_data_set_list="/root/CLEAR-VOX-MODEL/data/1h_dataset/train.jsonl" \
++valid_data_set_list="/root/CLEAR-VOX-MODEL/data/1h_dataset/val.jsonl" \
++dataset="AudioDataset" \
++dataset_conf.batch_size=6000 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++dataset_conf.preprocessor_speech="SpeechPreprocessSpeedPerturb" \
++dataset_conf.preprocessor_speech_conf.speed_perturb="[0.9, 1.0, 1.1]" \
++train_conf.max_epoch=50 \
++train_conf.log_interval=50 \
++train_conf.validate_interval=2000 \
++train_conf.keep_nbest_models=10 \
++train_conf.avg_nbest_model=10 \
++optim_conf.lr=0.0002 \
++output_dir="/root/CLEAR-VOX-MODEL/exp/paraformer_finetune_1h_sp"
```

**预期效果**: 
- 数据量扩充 3 倍
- CER 预期下降 3-8%

---

### ⭐ 策略B: 增强 SpecAugment (推荐)

**原理**: 更强的频谱遮蔽可以增强模型鲁棒性

**Paraformer 默认 SpecAug 配置**:
```yaml
specaug_conf:
  freq_mask_width_range: [0, 30]
  num_freq_mask: 1
  time_mask_width_range: [0, 12]
  num_time_mask: 1
```

**增强配置** (参考官方示例):
```yaml
specaug_conf:
  freq_mask_width_range: [0, 30]
  num_freq_mask: 2        # 增加到 2
  time_mask_width_range: [0, 40]  # 增加
  num_time_mask: 2        # 增加到 2
```

**修改训练命令**:
```bash
torchrun --nproc_per_node=1 \
/root/CLEAR-VOX-MODEL/funasr/bin/train_ds.py \
++model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
++train_data_set_list="/root/CLEAR-VOX-MODEL/data/1h_dataset/train.jsonl" \
++valid_data_set_list="/root/CLEAR-VOX-MODEL/data/1h_dataset/val.jsonl" \
++dataset="AudioDataset" \
++dataset_conf.batch_size=6000 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++specaug_conf.num_freq_mask=2 \
++specaug_conf.num_time_mask=2 \
++specaug_conf.time_mask_width_range="[0, 40]" \
++train_conf.max_epoch=50 \
++train_conf.log_interval=50 \
++train_conf.validate_interval=2000 \
++train_conf.keep_nbest_models=10 \
++train_conf.avg_nbest_model=10 \
++optim_conf.lr=0.0002 \
++output_dir="/root/CLEAR-VOX-MODEL/exp/paraformer_finetune_1h_aug"
```

**预期效果**:
- 增强模型对频谱变化的鲁棒性
- 对构音障碍的不清晰发音有更好的泛化

---

### 5.2 推荐的完整训练命令 (结合两种策略)

```bash
#!/bin/bash
# 优化后的微调脚本 v2.1
# 结合 Speed Perturbation + 增强 SpecAugment

export CUDA_VISIBLE_DEVICES="0"

workspace=/root/CLEAR-VOX-MODEL
model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

train_data="${workspace}/data/1h_dataset/train.jsonl"
val_data="${workspace}/data/1h_dataset/val.jsonl"
output_dir="${workspace}/exp/paraformer_finetune_1h_optimized"

mkdir -p ${output_dir}

echo "=============================================="
echo "FunASR Paraformer 构音障碍微调 v2.1 (优化版)"
echo "=============================================="
echo "策略: Speed Perturbation + 增强 SpecAugment"
echo "=============================================="

torchrun --nproc_per_node=1 \
${workspace}/funasr/bin/train_ds.py \
++model="${model}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset="AudioDataset" \
++dataset_conf.index_ds="IndexDSJsonl" \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=6000 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++dataset_conf.preprocessor_speech="SpeechPreprocessSpeedPerturb" \
++dataset_conf.preprocessor_speech_conf.speed_perturb="[0.9, 1.0, 1.1]" \
++specaug_conf.num_freq_mask=2 \
++specaug_conf.num_time_mask=2 \
++specaug_conf.time_mask_width_range="[0, 40]" \
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

echo ""
echo "训练完成！模型保存在: ${output_dir}"
```

---

## 6. 参考文献

### 核心论文

1. **Paraformer** (2022)
   - Gao et al. "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition"
   - INTERSPEECH 2022
   - arXiv:2206.08317

2. **CDSD** (2024)
   - "CDSD: Chinese Dysarthria Speech Database"
   - INTERSPEECH 2024
   - arXiv:2310.15930

3. **Zero-Shot DSR with MLLM** (2025)
   - Alsayegh & Masood. "Zero-Shot Recognition of Dysarthric Speech Using Commercial ASR and Multimodal Large Language Models"
   - arXiv:2512.17474

4. **Prototype-Based Adaptation** (2024)
   - Wang et al. "Enhancing Dysarthric Speech Recognition for Unseen Speakers via Prototype-Based Adaptation"
   - INTERSPEECH 2024
   - arXiv:2407.18461

5. **DSR Fine-Tuning Strategies** (2024)
   - Leivaditi et al. "Fine-Tuning Strategies for Dutch Dysarthric Speech Recognition"
   - INTERSPEECH 2024

6. **DSR Overview** (2025)
   - Bhat & Strik. "Speech technology for automatic recognition and assessment of dysarthric speech: An overview"
   - Journal of Speech, Language, and Hearing Research

### 相关挑战赛

- **SLT 2024 LRDWWS Challenge**: Low-Resource Dysarthric Wake Word Spotting
- IEEE SLT 2024

---

## 附录: CDSD 数据集信息

| 属性 | 值 |
|------|-----|
| 总时长 | 133 小时 |
| 说话人数 | 44 人 |
| 最佳 CER | 16.4% (Hybrid CTC/Attention) |
| 人类基线 | 20.45% CER |
| 会议 | INTERSPEECH 2024 |

---

**报告结束**

*本报告基于 2025-12-23 的公开资料整理*
