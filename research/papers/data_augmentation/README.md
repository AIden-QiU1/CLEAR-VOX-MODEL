# 📈 数据增强 (Data Augmentation)

> 通过TTS合成、语音转换、SpecAugment等技术扩充构音障碍训练数据

---

## 📋 论文列表（按时间倒序）

| # | 论文 | 会议/期刊 | 年份 | 重要性 |
|---|------|----------|------|--------|
| 1 | Synthetic Dysarthric Speech: Supplement Not Substitute | Interspeech | 2025 | ⭐⭐⭐⭐⭐ |
| 2 | Data Augmentation for Severity Classification | Interspeech | 2025 | ⭐⭐⭐⭐ |
| 3 | Training Data Augmentation by TTS | Interspeech | 2024 | ⭐⭐⭐⭐⭐ |
| 4 | Personalized Adversarial Data Augmentation | TASLP | 2024 | ⭐⭐⭐⭐ |
| 5 | Enhancing ASR Fine-tuning with Adversarial DA | ICASSP | 2024 | ⭐⭐⭐⭐ |
| 6 | Towards Automatic Data Augmentation | ICASSP | 2024 | ⭐⭐⭐ |
| 7 | Accurate Synthesis with Severity Control | Speech Comm | 2024 | ⭐⭐⭐⭐ |
| 8 | VC-based Augmentation (StarGAN-VC) | IEEE TNSR | 2023 | ⭐⭐⭐ |
| 9 | Adversarial Data Augmentation Using VAE-GAN | ICASSP | 2023 | ⭐⭐⭐ |
| 10 | Few-shot DSR with TTS Augmentation | Interspeech | 2023 | ⭐⭐⭐ |
| 11 | Two-stage Data Augmentation | Interspeech | 2022 | ⭐⭐⭐⭐⭐ |
| 12 | Synthesis of New Words for Expanded Vocabulary | ICASSP | 2021 | ⭐⭐⭐ |

---

## 📖 论文详解

### 1. Synthetic Dysarthric Speech: A Supplement, Not a Substitute ⭐⭐⭐⭐⭐
**Interspeech 2025** | [论文](https://www.isca-archive.org/interspeech_2025/li25n_interspeech.pdf)

#### 核心发现
> 合成数据仅适合作为预训练底座，**绝不可替代真实患者数据**进行最终对齐

#### 关键洞察
- 合成构音数据（TTS/VC）存在**过度平滑/缺乏类内变异性**问题
- 模型会学习到错误的规律性偏差
- **混合训练**是提升识别率的最佳路径

#### 迁移建议
```yaml
训练策略:
  阶段1_预训练: 合成数据 (TTS/VC生成)
  阶段2_微调: 真实患者数据
  比例: 合成:真实 = 3:1 到 1:1
```

---

### 2. Data Augmentation using Speech Synthesis for Severity Classification ⭐⭐⭐⭐
**Interspeech 2025** | [论文](https://www.isca-archive.org/interspeech_2025/kim25w_interspeech.pdf)

#### 核心贡献
- 利用可控TTS合成**不同严重等级**的构音障碍语音
- 解决真实病理分级数据稀缺问题
- **逆严重度加权**的数据混合策略

#### 关键技术
```python
# 逆严重度加权策略
def get_synthesis_ratio(severity):
    """重度样本需要更多合成数据"""
    ratios = {
        'severe': 3.0,    # 合成:真实 = 3:1
        'moderate': 2.0,
        'mild': 1.0
    }
    return ratios.get(severity, 1.0)
```

#### 课程学习策略
- 训练后期**逐步剔除合成数据**
- 迫使模型适配真实病理特征

---

### 3. Training Data Augmentation by Text-to-Dysarthric-Speech Synthesis ⭐⭐⭐⭐⭐
**Interspeech 2024** | [论文](https://arxiv.org/abs/2406.08568)

#### 核心贡献
- 建立**构音数据工厂**
- 利用F5-TTS/CosyVoice低步数推理合成含糊语音
- **One-Shot音色迁移**解决无数据冷启动

#### 实现方案
```python
from f5_tts import F5TTS

def synthesize_dysarthric(text, reference_audio):
    """
    使用F5-TTS合成构音障碍风格语音
    reference_audio: 患者参考音频（用于音色克隆）
    """
    tts = F5TTS()
    # 低推理步数保留一定的"含糊"特征
    audio = tts.generate(
        text=text,
        reference=reference_audio,
        inference_steps=10  # 低步数
    )
    return audio
```

---

### 4. Personalized Adversarial Data Augmentation ⭐⭐⭐⭐
**TASLP 2024** | CUHK

#### 核心思想
- 对抗训练生成**个性化增强样本**
- 针对每个患者的特定错误模式

---

### 5. Enhancing Pre-trained ASR Fine-tuning with Adversarial DA ⭐⭐⭐⭐
**ICASSP 2024** | CUHK | [论文](https://ieeexplore.ieee.org/document/xxxx)

#### 核心思想
- 结合对抗训练与预训练模型微调
- 生成更具挑战性的训练样本

---

### 6. Towards Automatic Data Augmentation for Disordered Speech ⭐⭐⭐
**ICASSP 2024** | CUHK

#### 核心思想
- **自动化**选择最优增强策略
- 无需人工调参

---

### 7. Accurate Synthesis of Dysarthric Speech for ASR Data Augmentation ⭐⭐⭐⭐
**Speech Communication 2024** | [论文](https://www.sciencedirect.com/science/article/abs/pii/S0167639324000839)

#### 核心贡献
- 加入**严重程度系数**控制合成语音
- **停顿插入模型**模拟病理特征

#### 技术架构
```
Severity-Controlled FastSpeech 2 (Acoustic Model)
         ↓
    Severity Embedding (轻度/中度/重度)
         ↓
    HiFi-GAN (Vocoder)
         ↓
    构音障碍风格语音
```

---

### 8. Improving VC for Dysarthria Voice Conversion ⭐⭐⭐
**IEEE TNSR 2023** | [论文](https://ieeexplore.ieee.org/document/10313325)

#### 核心贡献
- CycleGAN/Diff-GAN/StarGAN-VC 对比
- **StarGAN-VC最优**: 无需配对语料

#### 数据策略
```
真实 + 合成 混合数据模式
├── 少量目标患者真实语音
└── 大量类构音障碍合成语音
```

---

### 9. Adversarial Data Augmentation Using VAE-GAN ⭐⭐⭐
**ICASSP 2023** | CUHK

---

### 10. Few-shot Dysarthric Speech Recognition with TTS Data Augmentation ⭐⭐⭐
**Interspeech 2023** | [论文](https://publications.idiap.ch/attachments/papers/2023/Hermann_INTERSPEECH_2023.pdf)

#### 关键发现
> 合成语音在已见说话人场景有效，但在unseen speaker的few-shot场景**质量/多样性是瓶颈**

---

### 11. Improved ASR with Two-stage Data Augmentation ⭐⭐⭐⭐⭐
**Interspeech 2022** | [论文](https://www.sciencedirect.com/science/article/pii/S0010482525003051)

#### 核心贡献（极重要！）
**定制化SpecAugment掩码**模拟构音障碍特征：

```python
def stutter_mask(spectrogram, repeat_count=3):
    """口吃掩码: 在频谱上随机复制几帧（模仿卡顿）"""
    t = random.randint(0, spectrogram.shape[1] - 5)
    frame = spectrogram[:, t:t+1]
    repeated = frame.repeat(1, repeat_count)
    spectrogram[:, t:t+repeat_count] = repeated
    return spectrogram

def hypernasal_mask(spectrogram, high_boost=0.3, low_cut=0.2):
    """鼻音化掩码: 高频增强 + 低频衰减"""
    spectrogram[:int(spectrogram.shape[0]*0.3), :] *= (1 - low_cut)
    spectrogram[int(spectrogram.shape[0]*0.7):, :] *= (1 + high_boost)
    return spectrogram

def breathiness_mask(spectrogram, noise_level=0.1):
    """气息音掩码: 注入高斯噪声（模仿漏气）"""
    noise = torch.randn_like(spectrogram) * noise_level
    return spectrogram + noise
```

#### 核心价值
- **极低成本**增强模型鲁棒性
- 不需要外部TTS/VC模型

---

### 12. Synthesis of New Words for Improved Dysarthric Speech Recognition ⭐⭐⭐
**ICASSP 2021** | [论文](https://ieeexplore.ieee.org/abstract/document/9414869)

#### 核心贡献
- **已见词/未见词**区分训练
- 针对性扩展词汇覆盖

---

## 🔬 实验计划

| 实验ID | 描述 | 优先级 |
|--------|------|--------|
| EXP-201 | SpecAugment症状掩码实验 | P0 |
| EXP-202 | F5-TTS合成增强 | P1 |
| EXP-203 | CosyVoice合成增强 | P1 |
| EXP-204 | 混合数据比例实验 | P1 |
| EXP-205 | 逆严重度加权策略 | P2 |
| EXP-206 | StarGAN-VC增强 | P2 |

---

## ✅ 推荐技术路线

```
第一阶段: SpecAugment症状掩码（零成本）
    ↓
第二阶段: F5-TTS合成增强（中等成本）
    ↓
第三阶段: 混合训练优化比例
    ↓
第四阶段: 课程学习策略
```

## ❌ 避免的做法

1. ❌ **不要**只用合成数据训练
2. ❌ **不要**忽略真实数据的微调阶段
3. ❌ **不要**对所有严重度使用相同增强比例
