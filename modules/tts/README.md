# 🔊 TTS 语音合成模块

用于数据增强的TTS模块。

## 支持的TTS引擎

| 引擎 | 特点 | 状态 |
|------|------|------|
| **F5-TTS** | 零样本克隆 | 📝 计划中 |
| **CosyVoice** | 中文效果好 | 📝 计划中 |

## 用途

1. **数据增强** - 从少量样本合成更多训练数据
2. **说话人克隆** - 保持构音障碍特征
3. **文本扩展** - 使用更丰富的文本内容

## 使用示例

```python
from modules.tts import synthesize

# 合成语音
audio = synthesize(
    text="你好世界",
    speaker_embedding=speaker_emb,
    add_dysarthria_effects=True
)
```
