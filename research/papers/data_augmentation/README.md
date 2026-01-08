# 🔊 数据增强 (Data Augmentation)

> **核心问题**: 构音障碍语音数据极度稀缺，如何有效扩充训练数据？

---

## 📋 论文索引

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 | 重要性 |
|---|------|-----------|------|----------|--------|
| 1 | [Synthetic Dysarthric Speech: A Supplement, Not a Substitute](#1-synthetic-dysarthric-speech) | Interspeech | 2025 | 合成vs真实数据的权衡 | ⭐⭐⭐⭐⭐ |
| 2 | [Training Data Augmentation by TTS](#2-tts-data-augmentation) | Interspeech | 2024 | TTS合成构音障碍语音 | ⭐⭐⭐⭐⭐ |
| 3 | [Two-stage Data Augmentation](#3-two-stage-data-augmentation) | Interspeech | 2022 | 低成本频谱增强策略 | ⭐⭐⭐⭐ |
| 4 | [Improving Dysarthria VC System](#4-vc-based-augmentation) | IEEE TNSR | 2023 | GAN语音转换增强 | ⭐⭐⭐⭐ |
| 5 | [Accurate Synthesis for ASR Augmentation](#5-accurate-synthesis) | Speech Comm | 2024 | 严重度可控TTS | ⭐⭐⭐ |
| 6 | [Synthesis of New Words](#6-new-words-synthesis) | ICASSP | 2021 | 扩展词汇覆盖 | ⭐⭐⭐ |

---

## 📖 论文详解

### 1. Synthetic Dysarthric Speech: A Supplement, Not a Substitute
**Interspeech 2025** | [论文链接](https://www.isca-archive.org/interspeech_2025/li25n_interspeech.pdf)

#### 核心发现
> ⚠️ **关键结论**: 合成构音障碍数据（TTDS/VC）存在**过度平滑**和**缺乏类内变异性**问题，会导致模型学习到错误的规律性偏差。

#### 实验证据
- 仅用合成数据训练 → 模型在真实患者语音上CER反而上升
- 合成数据作为预训练 + 真实数据微调 → 最佳效果
- 最佳混合比例: 合成:真实 ≈ 3:1（预训练阶段）

#### 移植建议
```
📌 实践指南:
1. 不要用合成数据替代真实数据做最终微调
2. 合成数据适合作为预训练底座
3. 建议流程: 正常语音预训练 → 合成构音语音适配 → 真实患者数据对齐
```

#### 代码实现思路
```python
# 训练流程伪代码
# Stage 1: 使用合成数据预热
model.train(synthetic_dysarthric_data, epochs=5, lr=1e-4)

# Stage 2: 使用真实数据微调对齐
model.train(real_dysarthric_data, epochs=10, lr=1e-5)
```

---

### 2. Training Data Augmentation by TTS
**Interspeech 2024** | [论文链接](https://arxiv.org/abs/2406.08568)

#### 核心思想
利用 F5-TTS/CosyVoice 等现代TTS模型的**低步数推理**能力合成含糊语音，通过**One-Shot音色迁移**解决无数据冷启动问题。

#### 关键技术
1. **低步数推理**: 减少扩散步数 → 语音更模糊
2. **音色克隆**: 用患者少量语音（5-10秒）克隆音色
3. **批量生成**: 对同一文本生成多个变体

#### 移植方案
```python
# 使用 CosyVoice 生成构音障碍风格语音
from cosyvoice import CosyVoice

tts = CosyVoice(model_dir="CosyVoice-300M")

# 关键参数: 减少扩散步数使语音更模糊
synthetic_audio = tts.generate(
    text="打开空调",
    reference_audio="patient_sample.wav",  # 患者参考音频
    diffusion_steps=5,  # 低步数 → 更模糊
)
```

#### 实验计划
- [ ] EXP-201: CosyVoice低步数合成效果评估
- [ ] EXP-202: 合成数据量 vs CER下降曲线

---

### 3. Two-stage Data Augmentation
**Interspeech 2022** | [论文链接](https://www.sciencedirect.com/science/article/pii/S0010482525003051)

#### 核心创新
**低成本模拟病理语音特征**，无需复杂模型：

| 增强方法 | 模拟症状 | 实现方式 |
|----------|----------|----------|
| **Stutter Mask** | 口吃/卡顿 | 频谱上随机复制几帧 |
| **Hypernasal Mask** | 鼻音过重 | 高频/低频能量衰减 |
| **Breathiness Mask** | 漏气/气息 | 注入随机高斯噪声 |
| **静态降速** | 语速慢 | 时间轴拉伸 |

#### 代码实现
```python
import torch
import torchaudio

def stutter_mask(spectrogram, n_repeats=2, frame_range=(3, 8)):
    """模拟口吃：随机复制帧"""
    B, F, T = spectrogram.shape
    repeat_start = torch.randint(0, T-10, (1,)).item()
    repeat_len = torch.randint(*frame_range, (1,)).item()
    
    segment = spectrogram[:, :, repeat_start:repeat_start+repeat_len]
    repeated = segment.repeat(1, 1, n_repeats)
    
    return torch.cat([
        spectrogram[:, :, :repeat_start],
        repeated,
        spectrogram[:, :, repeat_start:]
    ], dim=2)

def hypernasal_mask(spectrogram, high_freq_decay=0.7):
    """模拟鼻音过重：高频衰减"""
    B, F, T = spectrogram.shape
    decay = torch.linspace(1, high_freq_decay, F).view(1, F, 1)
    return spectrogram * decay

def breathiness_mask(spectrogram, noise_level=0.1):
    """模拟漏气：添加高斯噪声"""
    noise = torch.randn_like(spectrogram) * noise_level
    return spectrogram + noise
```

#### 实验计划
- [ ] EXP-203: 各Mask策略单独效果对比
- [ ] EXP-204: Mask组合策略探索

---

### 4. VC-based Augmentation (StarGAN-VC/CycleGAN)
**IEEE TNSR 2023** | [论文链接](https://ieeexplore.ieee.org/document/10313325)

#### 核心思想
利用**无配对数据**的语音转换模型，将正常语音转换为构音障碍风格。

#### 技术对比
| 模型 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| **StarGAN-VC** | 多对多转换，无需配对 | 训练不稳定 | 多说话人 |
| **CycleGAN-VC** | 训练稳定 | 只能一对一 | 单说话人 |
| **DiffGAN** | 生成质量高 | 计算量大 | 质量优先 |

#### 最佳实践
```
📌 "真实 + 合成" 混合数据模式:
- 少量真实患者语音（保证相关性）
- 大量VC生成的类构音障碍语音（扩充多样性）
- 混合比例建议: 真实:合成 = 1:5 ~ 1:10
```

---

### 5. Accurate Synthesis with Severity Control
**Speech Communication 2024** | [论文链接](https://www.sciencedirect.com/science/article/abs/pii/S0167639324000839)

#### 核心创新
在TTS合成中加入**严重程度系数**和**停顿插入模型**：
- Severity-Controlled FastSpeech 2 (声学模型)
- HiFi-GAN (声码器)

#### 架构
```
文本 → FastSpeech2 → 严重度嵌入 → 停顿预测 → Mel谱 → HiFi-GAN → 波形
                        ↓
                   severity ∈ [0, 1]
                   0=正常, 1=重度
```

#### 移植思路
- 在 F5-TTS 中添加 severity embedding
- 使用对比学习把不同严重度的 style 拉开

---

### 6. Synthesis of New Words for Vocabulary Expansion
**ICASSP 2021** | [论文链接](https://ieeexplore.ieee.org/abstract/document/9414869)

#### 关键洞察
> 构音障碍ASR存在**已见词/未见词**的严重性能差距

#### 解决方案
1. 识别词汇表中的**低频词**和**未见词**
2. 使用TTS为这些词生成构音障碍风格语音
3. 针对性扩充训练数据

---

## 🧪 实验计划总览

### EXP-2XX: 数据增强实验系列

| ID | 实验名称 | 假设 | 优先级 |
|----|----------|------|--------|
| EXP-201 | CosyVoice低步数合成 | 低扩散步数产生更模糊语音 | P1 |
| EXP-202 | 合成数据量曲线 | 合成数据存在边际效益递减 | P1 |
| EXP-203 | 频谱Mask单策略 | Stutter Mask效果 > Hypernasal | P2 |
| EXP-204 | Mask组合策略 | 组合优于单独使用 | P2 |
| EXP-205 | 真实:合成混合比 | 最佳比例在1:5附近 | P1 |
| EXP-206 | 三阶段训练流程 | 正常→合成→真实 优于直接微调 | P0 |

---

## 💡 核心结论与建议

### ✅ 推荐做法
1. **三阶段训练**: 正常语音预训练 → 合成构音语音 → 真实患者数据
2. **低成本增强优先**: Stutter/Hypernasal Mask 成本低效果好
3. **混合训练**: 真实+合成数据混合，比例1:5~1:10
4. **课程学习**: 逆严重度加权，重度样本用更多合成数据

### ❌ 避免的做法
1. 不要用合成数据完全替代真实数据
2. 不要过度依赖单一增强策略
3. 避免合成数据过于"规律"（添加随机性）

---

## 📚 相关资源

- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [F5-TTS 论文](https://arxiv.org/abs/2410.06885)
- [StarGAN-VC 实现](https://github.com/kamepong/StarGAN-VC)
