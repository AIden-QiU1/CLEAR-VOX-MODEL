# 🧠 ASR模型适配 (ASR Adaptation)

> 构音障碍语音识别的模型适配技术：LoRA、MoE、Adapter、迁移学习等

---

## 📋 论文列表（按时间倒序 + 重要性）

### 🔥 2025年 顶会论文

| # | 论文 | 会议 | 重要性 |
|---|------|------|--------|
| 1 | On-the-fly MoE Routing for Zero-shot Speaker Adaptation | Interspeech 2025 | ⭐⭐⭐⭐⭐ |
| 2 | Bridging ASR and LLMs for Dysarthric Speech | Interspeech 2025 | ⭐⭐⭐⭐⭐ |
| 3 | Cross-lingual VC for Inclusive ASR | Interspeech 2025 | ⭐⭐⭐⭐ |
| 4 | Unsupervised Rhythm and Voice Conversion | Interspeech 2025 | ⭐⭐⭐⭐ |
| 5 | Comparison of Acoustic+Textual for Severity Classification | Interspeech 2025 | ⭐⭐⭐⭐ |
| 6 | Robust Cross-Etiology Speaker-Independent DSR | ICASSP 2025 | ⭐⭐⭐⭐ |
| 7 | Dysarthric Speech Conformer Adaptation | ICASSP 2025 | ⭐⭐⭐⭐⭐ |
| 8 | Phone-purity Guided Discrete Tokens | ICASSP 2025 | ⭐⭐⭐ |

### 📚 2024年 论文

| # | 论文 | 会议 | 重要性 |
|---|------|------|--------|
| 9 | Perceiver-Prompt for Whisper Adaptation | Interspeech 2024 | ⭐⭐⭐⭐⭐ |
| 10 | Prototype-Based Adaptation for Unseen Speakers | Interspeech 2024 | ⭐⭐⭐⭐ |
| 11 | Curriculum Learning with Articulatory Features | Interspeech 2024 | ⭐⭐⭐⭐ |
| 12 | Pre-trained Model for Articulatory Feature Extraction | Interspeech 2024 | ⭐⭐⭐ |
| 13 | CoLM-DSR: Neural Codec Language Modeling | Interspeech 2024 | ⭐⭐⭐⭐ |
| 14 | Self-Supervised ASR for Dysarthric and Elderly | ACM 2024 | ⭐⭐⭐⭐ |

### 📖 2023年及更早

| # | 论文 | 会议 | 重要性 |
|---|------|------|--------|
| 15 | Householder Transformation Adapter | Interspeech 2023 | ⭐⭐⭐ |
| 16 | Raw Waveform with SincNet (PCNN) | Interspeech 2023 | ⭐⭐⭐ |
| 17 | Wav2vec2 Speaker Adaptation | Interspeech 2022 | ⭐⭐⭐⭐ |
| 18 | Two-step Acoustic Model Adaptation | ICASSP 2020 | ⭐⭐⭐⭐⭐ |

### 📊 综合性工作 (CUHK系列)

| # | 论文 | 来源 |
|---|------|------|
| 19 | Recent Progress in CUHK DSR System | TASLP 2021 |
| 20 | Self-Supervised ASR Models and Features | TASLP 2024 |
| 21 | Speaker Adaptation Using Spectro-Temporal Features | TASLP 2022 |
| 22 | Homogeneous Speaker Features for On-the-Fly | TASLP 2025 |
| 23 | Cross-Utterance Speech Contexts for Conformer | TASLP 2025 |
| 24 | MOPSA: Mixture of Prompt-Experts | arXiv 2025 |
| 25 | Federated Learning for Privacy-Preserving DSR | arXiv 2025 |
| 26 | Structured Speaker-Deficiency Adaptation | arXiv 2024 |

---

## 📖 核心论文详解

### 1. On-the-fly MoE Routing for Zero-shot Speaker Adaptation ⭐⭐⭐⭐⭐
**Interspeech 2025** | [论文](https://arxiv.org/pdf/2412.18832)

#### 核心创新
> 根据用户语音特征**自动路由**到合适的LoRA专家

#### 技术方案
```python
class MoELoRARouter(nn.Module):
    """MoE路由器: 根据音频特征选择LoRA专家"""
    def __init__(self, n_experts=5, hidden_dim=256):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, audio_features):
        # 计算专家权重
        weights = self.router(audio_features.mean(dim=1))
        return weights  # [batch, n_experts]
```

#### 迁移建议
- 按**严重度**(轻度/中度/重度)或**类型**(痉挛型/迟缓型)训练不同LoRA
- 训练一个轻量级路由网络(几层MLP)

---

### 2. Dysarthric Speech Conformer Adaptation ⭐⭐⭐⭐⭐
**ICASSP 2025** | [论文](https://ieeexplore.ieee.org/document/10889046)

#### 核心策略
> **冻结Decoder，只微调Encoder**

#### 训练配置
```yaml
model:
  encoder: trainable (with LoRA)
  decoder: frozen

loss:
  kl_divergence: 0.7
  ctc: 0.3

data_augmentation:
  - SpecAugment (时频遮挡)
  - 时频扰动 (语速/音高)
  - 语音去噪增强
```

---

### 3. Perceiver-Prompt: Flexible Speaker Adaptation in Whisper ⭐⭐⭐⭐⭐
**Interspeech 2024** | [论文](https://www.isca-archive.org/interspeech_2024/jiang24b_interspeech.pdf)

#### 核心创新
> 用Perceiver将**可变长度语音**编码为**固定长度prompt**

#### 技术实现
```python
class PerceiverPrompt(nn.Module):
    """Perceiver编码说话人特征为固定长度prompt"""
    def __init__(self, prompt_length=16, d_model=512):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(prompt_length, d_model))
        self.cross_attention = nn.MultiheadAttention(d_model, 8)
        
    def forward(self, speech_features):
        # speech_features: [T, B, D] 可变长度
        prompt, _ = self.cross_attention(
            self.latents.unsqueeze(1).expand(-1, speech_features.size(1), -1),
            speech_features,
            speech_features
        )
        return prompt  # [prompt_length, B, D] 固定长度
```

#### 效果
- 中文构音障碍数据CER**相对降低13.04%**

---

### 4. Curriculum Learning with Articulatory Features ⭐⭐⭐⭐
**Interspeech 2024** | [论文](https://www.isca-archive.org/interspeech_2024/hsieh24_interspeech.pdf)

#### 核心策略
```
阶段1: 健康对照语音 → 词汇层面预适配
阶段2: 患者语音 → 个体化微调
```

#### 关键技术
- **神经元冻结策略**防止过拟合
- 结合数据增强

---

### 5. Prototype-Based Adaptation for Unseen Speakers ⭐⭐⭐⭐
**Interspeech 2024** | [论文](https://arxiv.org/abs/2407.18461)

#### 核心策略
> **冻结Decoder + 微调Encoder + 两阶段迁移**

#### 迁移路径
```
通用ASR预训练 → 构音障碍通用适配 → 个体化微调
```

---

### 6. Two-step Acoustic Model Adaptation ⭐⭐⭐⭐⭐
**ICASSP 2020** | [论文](https://ieeexplore.ieee.org/abstract/document/90537)

#### 经典方法
> **双阶段LoRA微调**: 通用病理 → 个人定制

---

### 7. Wav2vec2 Speaker Adaptation ⭐⭐⭐⭐
**Interspeech 2022** | [论文](https://arxiv.org/pdf/2204.00770)

#### 建议
> 用**LoRA替代Adapter**可以获得更好的参数效率

---

### 8. Cross-Etiology and Speaker-Independent DSR ⭐⭐⭐⭐
**ICASSP 2025** | [论文](https://arxiv.org/html/2501.14994v1)

#### 核心发现
> 模型会**记住特定说话人**而非学习通用特征

#### 解决方案
- 加入**遗忘分支**
- 添加**说话人识别loss**进行对抗训练

---

### 9. Raw Waveform with Parametric CNNs (SincNet) ⭐⭐⭐
**Interspeech 2023** | [论文](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/176300344/INTERSPEECH_2022.pdf)

#### 双流架构
```
Stream A: Fbank → Paraformer Encoder (主路)
Stream B: Raw Waveform → SincNet → Linear (旁路)
                 ↓
            特征拼接
```

#### 核心价值
- 捕捉被传统特征遗漏的**病理高频细节**
- 生物仿生听觉前端

---

### 10. Unsupervised Rhythm and Voice Conversion ⭐⭐⭐⭐
**Interspeech 2025** | [论文](https://arxiv.org/abs/2506.01618)

#### 核心创新
> ASR前的**语速正骨**模块

#### 技术
- 无监督节奏建模
- 针对性压缩**元音与停顿时长**
- 消除拖沓卡顿

---

### 11. Cross-lingual VC for Inclusive ASR ⭐⭐⭐⭐
**Interspeech 2025** | [论文](https://arxiv.org/abs/2505.14874)

#### 核心创新
> 用**英语构音障碍语音**训练VC，迁移到其他语言

#### 中文迁移建议
```python
# 中文特色适配
策略 = {
    "声调保真": "只注入节奏/能量异常，不破坏四声轮廓",
    "方言覆盖": "利用CosyVoice迁移病态特征至方言",
    "数据扩充": "AISHELL/WenetSpeech → 障碍风格",
}
```

---

## 🔬 实验计划

| 实验ID | 描述 | 优先级 |
|--------|------|--------|
| EXP-101 | 冻结Decoder + LoRA Encoder | P0 |
| EXP-102 | Perceiver-Prompt实现 | P1 |
| EXP-103 | MoE按严重度路由 | P1 |
| EXP-104 | 两阶段迁移学习 | P0 |
| EXP-105 | SincNet双流融合 | P2 |
| EXP-106 | 语速正骨前处理 | P2 |

---

## ✅ 最佳实践路线

```
Step 1: 冻结Decoder，Encoder添加LoRA (rank=8)
    ↓
Step 2: 通用构音障碍数据预适配
    ↓
Step 3: 目标用户个性化微调
    ↓
Step 4: 可选：MoE路由增强
```

## 📊 策略对比

| 策略 | 参数量 | 效果 | 适用场景 |
|------|--------|------|----------|
| Full Fine-tune | 100% | ⭐⭐⭐⭐ | 大数据量 |
| LoRA (r=8) | ~1% | ⭐⭐⭐⭐ | 推荐 |
| Adapter | ~2% | ⭐⭐⭐ | 备选 |
| Freeze Decoder | -50% | ⭐⭐⭐⭐⭐ | 小数据必选 |
