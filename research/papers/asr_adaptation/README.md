# 🎯 ASR模型适配 (ASR Adaptation)

> **核心问题**: 如何让预训练ASR模型适配构音障碍语音？

---

## 📋 论文索引

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 | 重要性 |
|---|------|-----------|------|----------|--------|
| 1 | [Perceiver-Prompt: Flexible Speaker Adaptation](#1-perceiver-prompt) | Interspeech | 2024 | 可学习说话人Prompt | ⭐⭐⭐⭐⭐ |
| 2 | [On-the-fly MoE Routing](#2-moe-routing) | Interspeech | 2025 | 严重度自动路由 | ⭐⭐⭐⭐⭐ |
| 3 | [Two-step Acoustic Model Adaptation](#3-two-step-adaptation) | ICASSP | 2020 | 两阶段LoRA微调 | ⭐⭐⭐⭐⭐ |
| 4 | [Prototype-Based Adaptation](#4-prototype-adaptation) | Interspeech | 2024 | 冻结解码器策略 | ⭐⭐⭐⭐ |
| 5 | [Dysarthric Speech Conformer](#5-conformer-adaptation) | ICASSP | 2025 | Conformer适配 | ⭐⭐⭐⭐ |
| 6 | [Householder Transformation Adapter](#6-householder-adapter) | Interspeech | 2023 | 极致参数压缩 | ⭐⭐⭐ |
| 7 | [Curriculum Learning + Articulatory](#7-curriculum-learning) | Interspeech | 2024 | 课程学习策略 | ⭐⭐⭐⭐ |
| 8 | [Cross-Etiology Speaker-Independent](#8-cross-etiology) | ICASSP | 2025 | 跨病因泛化 | ⭐⭐⭐⭐ |
| 9 | [Raw Waveform with PCNN](#9-raw-waveform-pcnn) | Interspeech | 2023 | 双流特征融合 | ⭐⭐⭐ |
| 10 | [Wav2vec2 Speaker Adaptation](#10-wav2vec2-adaptation) | Interspeech | 2022 | Adapter替代方案 | ⭐⭐⭐ |

---

## 📖 论文详解

### 1. Perceiver-Prompt: Flexible Speaker Adaptation in Whisper
**Interspeech 2024** | [论文链接](https://www.isca-archive.org/interspeech_2024/jiang24b_interspeech.pdf)

#### 核心创新
用可训练的 **Perceiver** 模块把可变长度输入语音编码成**固定长度的 Speaker Prompt**，注入到 Whisper 中实现说话人适配。

#### 架构图
```
输入语音 → Whisper Encoder → Perceiver (可变→固定) → Speaker Prompt
                                                          ↓
                               Whisper Decoder ← 拼接 ← Encoder输出
```

#### 关键结果
- 在中文构音障碍数据上 CER 相对降低 **13.04%**
- Prompt长度: 32~64 tokens 效果最佳

#### 移植方案
```python
# 在Paraformer中实现Perceiver-Prompt
import torch.nn as nn

class PerceiverPrompt(nn.Module):
    def __init__(self, d_model=512, num_latents=32):
        super().__init__()
        # 可学习的查询向量
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, d_model))
        # Cross-attention: 从音频特征提取说话人信息
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8)
        
    def forward(self, audio_features):
        # audio_features: (B, T, D) 可变长度
        B = audio_features.size(0)
        queries = self.latent_queries.expand(B, -1, -1)
        
        # 输出固定长度的speaker prompt
        speaker_prompt, _ = self.cross_attn(
            queries.transpose(0, 1),
            audio_features.transpose(0, 1),
            audio_features.transpose(0, 1)
        )
        return speaker_prompt.transpose(0, 1)  # (B, num_latents, D)
```

#### 实验计划
- [ ] EXP-101: Perceiver-Prompt在Paraformer中的实现
- [ ] EXP-102: Prompt长度消融实验 (16/32/64/128)

---

### 2. On-the-fly MoE Routing for Dysarthric ASR
**Interspeech 2025** | [论文链接](https://arxiv.org/pdf/2412.18832)

#### 核心思想
> 不同严重程度的构音障碍需要不同的模型参数 → **MoE (Mixture of Experts) 路由**

#### 架构
```
                    ┌─────────────┐
                    │ Router MLP  │ ← 输入音频特征
                    └──────┬──────┘
                           │ 权重分配
            ┌──────────────┼──────────────┐
            ↓              ↓              ↓
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │ LoRA-轻 │    │ LoRA-中 │    │ LoRA-重 │
      └────┬────┘    └────┬────┘    └────┬────┘
           └──────────────┴──────────────┘
                          ↓
                    加权融合输出
```

#### 实现步骤
1. 按严重度分别训练 3~5 个 LoRA
2. 训练一个小型 Router 网络（几层MLP）
3. Router 根据输入自动分配 LoRA 权重

#### 移植方案
```python
class MoELoRARouter(nn.Module):
    def __init__(self, d_model=512, num_experts=3):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, audio_features):
        # 全局平均池化
        pooled = audio_features.mean(dim=1)  # (B, D)
        weights = self.router(pooled)  # (B, num_experts)
        return weights

# 使用时
router_weights = router(audio_features)  # [0.7, 0.2, 0.1]
output = sum(w * lora_i(x) for w, lora_i in zip(router_weights, loras))
```

#### 实验计划
- [ ] EXP-103: 按严重度分组训练多个LoRA
- [ ] EXP-104: Router网络架构探索
- [ ] EXP-105: MoE vs 单一LoRA对比

---

### 3. Two-step Acoustic Model Adaptation
**ICASSP 2020** | [论文链接](https://ieeexplore.ieee.org/abstract/document/9053735)

#### 核心策略
> **通用病理适配 → 个人定制微调** 双阶段LoRA

#### 两阶段流程
```
Stage 1: 通用构音障碍适配
┌─────────────────────────────────────────┐
│ 数据: 所有患者的混合数据                   │
│ 目标: 学习构音障碍的通用特征               │
│ 输出: Base-LoRA (通用病理)                │
└─────────────────────────────────────────┘
                    ↓
Stage 2: 个人定制微调  
┌─────────────────────────────────────────┐
│ 数据: 特定患者的15-20句语音               │
│ 目标: 适配个体发音特点                    │
│ 输出: User-LoRA (个人定制)                │
└─────────────────────────────────────────┘
                    ↓
推理: 基础模型 + Base-LoRA + User-LoRA
```

#### 实现代码
```python
from peft import LoraConfig, get_peft_model

# Stage 1: 通用病理LoRA
base_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["encoder.layers.*.self_attn.q_proj", "encoder.layers.*.self_attn.v_proj"],
    lora_dropout=0.1,
)

# Stage 2: 个人定制LoRA (在Base-LoRA基础上)
user_lora_config = LoraConfig(
    r=8,  # 更小的rank，避免过拟合
    lora_alpha=16,
    target_modules=["encoder.layers.*.self_attn.q_proj"],
    lora_dropout=0.05,
)
```

#### 实验计划
- [ ] EXP-106: 两阶段LoRA vs 单阶段LoRA
- [ ] EXP-107: 个人定制最少数据量探索 (5/10/15/20句)

---

### 4. Prototype-Based Adaptation for Unseen Speakers
**Interspeech 2024** | [论文链接](https://arxiv.org/abs/2407.18461)

#### 核心策略
> **冻结解码器，仅微调编码器** → 保留语言建模能力

#### 实验发现
| 微调策略 | CER | 分析 |
|----------|-----|------|
| 全参数微调 | 较差 | 遗忘语言知识 |
| 冻结编码器 | 较差 | 声学适配不足 |
| **冻结解码器** | **最佳** | 平衡声学和语言 |

#### 移植建议
```python
# Paraformer微调时冻结解码器
for name, param in model.named_parameters():
    if "decoder" in name:
        param.requires_grad = False
    elif "encoder" in name:
        param.requires_grad = True
```

---

### 5. Dysarthric Speech Conformer Adaptation
**ICASSP 2025** | [论文链接](https://ieeexplore.ieee.org/document/10889046)

#### 关键配置
- **损失函数**: 70% KL散度 + 30% CTC
- **数据增强**: SpecAugment + 时频扰动 + 语速变化
- **冻结策略**: 冻结Decoder，微调Encoder

#### 数据增强配置
```python
augment_config = {
    "spec_augment": {
        "time_mask_max": 80,
        "freq_mask_max": 40,
        "n_time_masks": 2,
        "n_freq_masks": 2,
    },
    "speed_perturb": [0.9, 1.0, 1.1],
    "pitch_shift": [-2, 0, 2],  # 半音
}
```

---

### 6. Householder Transformation Adapter
**Interspeech 2023** | [论文链接](https://arxiv.org/html/2306.07090v1)

#### 核心创新
用**反射正交矩阵**代替全连接层，极致压缩参数量。

#### 数学原理
```
Householder变换: H = I - 2vv^T / (v^T v)
其中 v 是可学习向量，H 是正交矩阵
参数量: O(d) vs 全连接 O(d²)
```

#### 适用场景
- 极致边缘端部署（单片机、超低功耗）
- 不支持LoRA算子加速的硬件
- 参数预算极度受限

---

### 7. Curriculum Learning with Articulatory Features
**Interspeech 2024** | [论文链接](https://www.isca-archive.org/interspeech_2024/hsieh24_interspeech.pdf)

#### 课程学习策略
```
阶段1: 健康语音预训练 (易)
    ↓
阶段2: 轻度构音障碍 (中)
    ↓
阶段3: 中度构音障碍 (难)
    ↓
阶段4: 重度构音障碍 (最难) + 个性化微调
```

#### 神经元冻结策略
- 阶段2: 冻结前6层
- 阶段3: 冻结前3层
- 阶段4: 全部解冻

---

### 8. Cross-Etiology and Speaker-Independent Recognition
**ICASSP 2025** | [论文链接](https://arxiv.org/html/2501.14994v1)

#### 核心问题
> 模型总是倾向于**记住具体的人**，而不是学习**通用的病理模式**

#### 解决方案
1. **遗忘分支**: 显式遗忘说话人特定信息
2. **说话人对抗损失**: 让模型无法区分说话人

```python
# 说话人对抗训练
class SpeakerAdversarialLoss(nn.Module):
    def __init__(self, d_model, num_speakers):
        self.speaker_classifier = nn.Linear(d_model, num_speakers)
        self.gradient_reversal = GradientReversal()
        
    def forward(self, features, speaker_ids):
        reversed_features = self.gradient_reversal(features)
        speaker_logits = self.speaker_classifier(reversed_features)
        return F.cross_entropy(speaker_logits, speaker_ids)
```

---

### 9. Raw Waveform with Parametric CNNs (SincNet)
**Interspeech 2023** | [论文链接](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/176300344/INTERSPEECH_2022.pdf)

#### 双流架构
```
Stream A (传统): Fbank → Paraformer Encoder → 特征
Stream B (波形): Raw Waveform → SincNet → Linear → 特征
                              ↓
                          特征拼接
                              ↓
                          联合预测
```

#### 核心思想
SincNet 是生物仿生听觉前端，能捕捉被传统Fbank遗漏的**病理高频细节**。

---

### 10. Wav2vec2 Speaker Adaptation
**Interspeech 2022** | [论文链接](https://arxiv.org/pdf/2204.00770)

#### LoRA vs Adapter
| 方法 | 参数量 | 效果 | 推荐 |
|------|--------|------|------|
| Adapter | 更多 | 稍好 | 有足够数据时 |
| **LoRA** | 更少 | 相当 | **数据稀缺时** |

---

## 🧪 实验计划总览

### EXP-1XX: ASR适配实验系列

| ID | 实验名称 | 假设 | 优先级 |
|----|----------|------|--------|
| EXP-101 | Perceiver-Prompt实现 | Speaker Prompt提升适配性 | P1 |
| EXP-102 | Prompt长度消融 | 32-64最佳 | P2 |
| EXP-103 | 多LoRA按严重度训练 | 不同严重度需要不同参数 | P1 |
| EXP-104 | MoE Router设计 | 自动路由优于固定选择 | P1 |
| EXP-105 | MoE vs 单一LoRA | MoE更鲁棒 | P0 |
| EXP-106 | 两阶段LoRA策略 | 两阶段优于单阶段 | P0 |
| EXP-107 | 个性化最少数据量 | 15-20句足够 | P1 |
| EXP-108 | 冻结策略对比 | 冻结Decoder最优 | P0 |
| EXP-109 | 课程学习流程 | 渐进训练优于直接微调 | P1 |
| EXP-110 | 说话人对抗训练 | 提升跨说话人泛化 | P2 |

---

## 💡 核心结论与建议

### ✅ 最佳实践路线
```
1. 基础模型: Paraformer-large (非自回归，速度快)
2. 微调策略: 冻结Decoder + LoRA微调Encoder
3. 个性化: 两阶段LoRA (通用病理 → 个人定制)
4. 进阶: MoE路由 (按严重度自动选择专家)
```

### 📊 策略对比

| 策略 | 效果 | 复杂度 | 推荐场景 |
|------|------|--------|----------|
| 全参数微调 | ⭐⭐ | 低 | 不推荐 |
| 冻结Decoder + LoRA | ⭐⭐⭐⭐ | 中 | 通用场景 |
| 两阶段LoRA | ⭐⭐⭐⭐⭐ | 中 | 需要个性化 |
| MoE路由 | ⭐⭐⭐⭐⭐ | 高 | 多说话人 |
| Perceiver-Prompt | ⭐⭐⭐⭐ | 高 | 研究探索 |

---

## 📚 相关资源

- [PEFT (LoRA) 库](https://github.com/huggingface/peft)
- [Paraformer 微调文档](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/finetune.md)
- [Whisper 微调指南](https://huggingface.co/blog/fine-tune-whisper)
