# 🧩 扩展模块 (Modules)

本目录存放除ASR外的其他语音处理模块。

## 📁 目录结构

```
modules/
├── tts/              # 语音合成模块
│   ├── f5_tts/      # F5-TTS (计划中)
│   └── cosyvoice/   # CosyVoice (计划中)
│
├── vc/               # 声音转换模块
│   └── stargan_vc/  # StarGAN-VC (计划中)
│
├── dsr/              # 语音重建模块
│   └── diffdsr/     # DiffDSR (计划中)
│
├── enhancement/      # 语音增强模块
│   ├── denoising/   # 降噪 (计划中)
│   └── dereverberation/  # 去混响 (计划中)
│
└── README.md         # 本文件
```

## 🎯 模块用途

| 模块 | 主要用途 | 优先级 |
|------|----------|--------|
| **tts** | 数据增强 - 合成模拟语音 | P1 |
| **vc** | 数据增强 - 声音转换 | P2 |
| **dsr** | 语音重建 - 提升可懂度 | P3 |
| **enhancement** | 预处理 - 提升音质 | P2 |

## 📦 安装依赖

各模块的依赖安装方式：

```bash
# TTS模块
pip install f5-tts
# 或
pip install cosyvoice

# VC模块
pip install stargan-vc

# 增强模块
pip install denoiser
```

## 🔗 与ASR的集成

```python
# 示例：使用TTS进行数据增强
from modules.tts.f5_tts import synthesize
from funasr import AutoModel

# 1. 使用TTS生成增强数据
augmented_audio = synthesize(text="测试文本", speaker_id="spk01")

# 2. 使用ASR识别
model = AutoModel(model="paraformer-large")
result = model.generate(input=augmented_audio)
```

## 📚 相关研究

- [TTS数据增强论文](../research/papers/data_augmentation/)
- [语音重建论文](../research/papers/speech_reconstruction/)
