# 🔊 语音重建/TTS (Speech Reconstruction)

> 构音障碍语音的重建与转换：TTS增强、Voice Conversion、语音恢复

---

## 📋 论文列表（按时间倒序 + 重要性）

### 🔥 2025年 论文

| # | 论文 | 会议 | 重要性 |
|---|------|------|--------|
| 1 | DiffDSR: Latent Diffusion for Dysarthric Speech Reconstruction | ICASSP 2025 | ⭐⭐⭐⭐⭐ |
| 2 | Cross-lingual VC for Inclusive ASR | Interspeech 2025 | ⭐⭐⭐⭐ |
| 3 | Unsupervised Rhythm and Voice Conversion | Interspeech 2025 | ⭐⭐⭐⭐ |
| 4 | F5-TTS Fairness and Bias Study | ICASSP 2025 | ⭐⭐⭐⭐ |
| 5 | Phone-purity Guided Discrete Tokens for VC | ICASSP 2025 | ⭐⭐⭐ |

### 📚 2024年 论文

| # | 论文 | 会议 | 重要性 |
|---|------|------|--------|
| 6 | CoLM-DSR: Neural Codec Language Modeling | Interspeech 2024 | ⭐⭐⭐⭐⭐ |
| 7 | Zero-shot TTS for Atypical Speech | Interspeech 2024 | ⭐⭐⭐⭐ |
| 8 | CosyVoice: Scalable Multi-lingual TTS | arXiv 2024 | ⭐⭐⭐⭐⭐ |

### 📖 2023年及更早 论文

| # | 论文 | 会议 | 重要性 |
|---|------|------|--------|
| 9 | F5-TTS: Flow-based Zero-shot TTS | arXiv 2024 | ⭐⭐⭐⭐⭐ |
| 10 | Parrotron: End-to-End Speech Conversion | arXiv 2021 | ⭐⭐⭐⭐⭐ |
| 11 | VoiceLoop: Neural TTS for Speech Disorders | 2018 | ⭐⭐⭐ |
| 12 | Tacotron-based Dysarthric Speech Synthesis | 2019 | ⭐⭐⭐ |

---

## 📖 核心论文详解

### 1. DiffDSR: Latent Diffusion for Dysarthric Speech Reconstruction ⭐⭐⭐⭐⭐
**ICASSP 2025** | [论文](https://arxiv.org/abs/2501.xxxxx)

#### 核心创新
> 使用**潜在扩散模型**将病态语音重建为清晰语音

#### 技术架构
```
病态语音 → Encoder → 潜在空间 → Diffusion → 清晰语音
                         ↓
                    噪声调度器
                         ↓
                  保留语义，修复发音
```

#### 实现框架
```python
import torch
import torch.nn as nn

class DiffDSR(nn.Module):
    """潜在扩散语音重建"""
    def __init__(self, latent_dim=512, time_steps=1000):
        super().__init__()
        self.encoder = SpeechEncoder(out_dim=latent_dim)
        self.decoder = SpeechDecoder(in_dim=latent_dim)
        self.diffusion = GaussianDiffusion(
            denoise_fn=UNet1D(latent_dim),
            timesteps=time_steps
        )
        
    def forward(self, dysarthric_audio, target_audio=None):
        # 编码到潜在空间
        z_d = self.encoder(dysarthric_audio)
        
        if target_audio is not None:  # 训练模式
            z_t = self.encoder(target_audio)
            loss = self.diffusion(z_d, z_t)
            return loss
        else:  # 推理模式
            z_clean = self.diffusion.sample(z_d)
            return self.decoder(z_clean)
```

#### 关键技术
- **内容-韵律解耦**: 保留说话人身份
- **语义保持约束**: 确保转录一致
- **渐进去噪**: 1000步扩散过程

---

### 2. CoLM-DSR: Neural Codec Language Modeling ⭐⭐⭐⭐⭐
**Interspeech 2024** | [论文](https://arxiv.org/abs/2406.xxxxx)

#### 核心创新
> 使用**神经编解码器语言模型**进行语音重建

#### 技术方案
```python
class CoLMDSR:
    """Codec Language Model for DSR"""
    def __init__(self):
        self.codec = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.lm = TransformerLM(vocab_size=1024, d_model=512)
        
    def encode(self, audio):
        """编码为离散tokens"""
        return self.codec.encode(audio)
        
    def reconstruct(self, dysarthric_tokens):
        """自回归重建清晰tokens"""
        clean_tokens = self.lm.generate(dysarthric_tokens)
        return self.codec.decode(clean_tokens)
```

#### 优势
- 利用大规模预训练编解码器
- 离散token便于语言模型建模
- 可融合文本先验

---

### 3. Parrotron: End-to-End Speech Conversion ⭐⭐⭐⭐⭐
**Google 2021** | [论文](https://arxiv.org/abs/1904.04169)

#### 核心设计
> **端到端Seq2Seq**: 病态语音 → 清晰语音

#### 架构
```
Input: 病态语音频谱
   ↓
Encoder (Conformer)
   ↓
Attention
   ↓
Decoder (Autoregressive)
   ↓
Vocoder (HiFi-GAN)
   ↓
Output: 清晰语音波形
```

#### 训练策略
```python
# 多任务学习
losses = {
    "reconstruction": F.mse_loss(pred_mel, target_mel),
    "asr_ctc": ctc_loss(pred_text, target_text),
    "speaker_similarity": cosine_loss(spk_emb_pred, spk_emb_target)
}
total_loss = sum(losses.values())
```

---

### 4. F5-TTS: Flow-based Zero-shot TTS ⭐⭐⭐⭐⭐
**arXiv 2024** | [论文](https://arxiv.org/abs/2410.06885)

#### 核心优势
> **无需fine-tune即可克隆声音**

#### 应用于构音障碍
```python
class F5TTSDysarthricAugmentation:
    """使用F5-TTS生成病态语音"""
    def __init__(self):
        self.f5tts = F5TTS.from_pretrained("...")
        
    def augment(self, text, healthy_audio, style="dysarthric"):
        """生成带构音障碍风格的语音"""
        # 方案1: 用健康参考生成，再加病态扰动
        clean_audio = self.f5tts.synthesize(text, ref=healthy_audio)
        return self.add_dysarthric_style(clean_audio, style)
        
    def add_dysarthric_style(self, audio, style):
        """添加病态特征"""
        if style == "slow":
            return librosa.effects.time_stretch(audio, rate=0.7)
        elif style == "breathy":
            return self.add_breathiness(audio)
        elif style == "slurred":
            return self.add_formant_shift(audio)
```

#### 公平性研究发现
- 原始F5-TTS对非典型语音存在**偏见**
- 需要针对性数据增强改善

---

### 5. CosyVoice: Scalable Multi-lingual TTS ⭐⭐⭐⭐⭐
**阿里巴巴 2024** | [论文](https://arxiv.org/abs/2407.xxxxx)

#### 核心能力
- **零样本声音克隆**
- **跨语言合成**
- **情感控制**

#### 构音障碍应用
```python
from cosyvoice import CosyVoice

class CosyVoiceAugmenter:
    """CosyVoice数据增强"""
    def __init__(self):
        self.model = CosyVoice.from_pretrained("CosyVoice-300M")
        
    def generate_dysarthric_parallel(self, text, patient_audio, healthy_audio):
        """生成配对数据"""
        # 提取患者声音特征
        patient_spk = self.model.extract_speaker(patient_audio)
        
        # 用健康人发音风格 + 患者声音 = 理想目标
        # 这样可以生成 (patient_dysarthric, patient_ideal) 配对
        ideal = self.model.synthesize(
            text=text,
            speaker=patient_spk,
            style="clear"  # 清晰发音风格
        )
        return ideal
```

---

### 6. Cross-lingual VC for Inclusive ASR ⭐⭐⭐⭐
**Interspeech 2025** | [论文](https://arxiv.org/abs/2505.14874)

#### 核心思想
> 跨语言迁移病态特征，扩充低资源语言数据

#### 技术流程
```
英语病态语音 → 特征提取 → 病态风格编码
                              ↓
中文健康语音 → 内容提取 → + 病态风格 → 中文病态语音
```

---

### 7. Unsupervised Rhythm and Voice Conversion ⭐⭐⭐⭐
**Interspeech 2025** | [论文](https://arxiv.org/abs/2506.01618)

#### 核心贡献
> **无监督学习韵律转换**

#### 应用场景
- 将正常语速映射到病态语速（数据增强）
- 将病态语速规整为正常语速（预处理）

---

## 🔬 实验计划

| 实验ID | 描述 | 优先级 | 模型 | 预期收益 |
|--------|------|--------|------|----------|
| EXP-401 | F5-TTS零样本克隆 + 病态扰动 | P0 | F5-TTS | 10倍数据 |
| EXP-402 | CosyVoice生成理想配对 | P0 | CosyVoice | 配对数据 |
| EXP-403 | DiffDSR语音重建 | P1 | Diffusion | 清晰化 |
| EXP-404 | Parrotron端到端转换 | P1 | Seq2Seq | 实时转换 |
| EXP-405 | CoLM离散token建模 | P2 | Codec LM | 新范式 |
| EXP-406 | 跨语言病态迁移 | P2 | VC | 数据扩充 |

---

## ✅ 推荐技术路线

### 数据增强路线
```
健康语音语料 (AISHELL/WenetSpeech)
         ↓
    F5-TTS / CosyVoice
         ↓
    零样本声音克隆
         ↓
    添加病态特征扰动
         ↓
    大规模伪病态语音
```

### 语音重建路线
```
病态语音输入
     ↓
DiffDSR / Parrotron
     ↓
清晰语音输出
     ↓
ASR识别
```

### 配对数据生成
```
患者语音 + 文本标注
         ↓
CosyVoice (患者声音 + 清晰风格)
         ↓
(病态,理想) 配对数据
         ↓
训练语音重建模型
```

---

## 📊 TTS/VC模型对比

| 模型 | 类型 | 零样本 | 中文支持 | 开源 | 推荐度 |
|------|------|--------|----------|------|--------|
| F5-TTS | Flow | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| CosyVoice | AR+NAR | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| VALL-E | AR | ✅ | ❌ | ❌ | ⭐⭐⭐ |
| XTTS | AR | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ |
| Parrotron | Seq2Seq | ❌ | ❌ | ❌ | ⭐⭐⭐ |

---

## 🎯 关键代码片段

### 病态特征注入
```python
import librosa
import numpy as np

def inject_dysarthric_features(audio, sr=16000, severity="mild"):
    """向健康语音注入构音障碍特征"""
    params = {
        "mild": {"speed": 0.9, "jitter": 0.02, "breathiness": 0.1},
        "moderate": {"speed": 0.75, "jitter": 0.05, "breathiness": 0.2},
        "severe": {"speed": 0.6, "jitter": 0.1, "breathiness": 0.3},
    }[severity]
    
    # 1. 语速变慢
    audio = librosa.effects.time_stretch(audio, rate=params["speed"])
    
    # 2. 添加颤抖 (jitter)
    jitter = np.random.randn(len(audio)) * params["jitter"]
    audio = audio + jitter
    
    # 3. 添加气息音
    noise = np.random.randn(len(audio)) * params["breathiness"]
    audio = audio + noise * 0.1
    
    return audio

def add_stutter(audio, sr=16000, repeat_prob=0.1):
    """添加结巴/重复"""
    # 随机选择音节重复
    chunks = librosa.effects.split(audio, top_db=20)
    result = []
    for start, end in chunks:
        chunk = audio[start:end]
        if np.random.rand() < repeat_prob:
            result.extend([chunk] * np.random.randint(2, 4))
        else:
            result.append(chunk)
    return np.concatenate(result)
```

---

## 📚 相关资源

- [F5-TTS 官方仓库](https://github.com/SWivid/F5-TTS)
- [CosyVoice 官方仓库](https://github.com/FunAudioLLM/CosyVoice)
- [Parrotron 论文](https://arxiv.org/abs/1904.04169)
