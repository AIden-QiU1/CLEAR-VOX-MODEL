# 🛠️ 工具与资源

> 构音障碍语音识别研究常用工具

---

## 📦 核心框架

| 工具 | 用途 | 链接 |
|------|------|------|
| **FunASR** | ASR训练与推理 | [GitHub](https://github.com/modelscope/FunASR) |
| **ESPnet** | 端到端语音 | [GitHub](https://github.com/espnet/espnet) |
| **SpeechBrain** | 语音AI工具包 | [GitHub](https://github.com/speechbrain/speechbrain) |
| **Kaldi** | 传统ASR | [GitHub](https://github.com/kaldi-asr/kaldi) |

---

## 🎤 TTS/VC 工具

### 推荐用于数据增强

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **F5-TTS** | 零样本克隆 | 快速生成多样化语音 |
| **CosyVoice** | 阿里开源 | 中文TTS |
| **VITS** | 端到端 | 高质量合成 |
| **StarGAN-VC** | 声音转换 | 模拟发音问题 |

### 安装示例
```bash
# F5-TTS
pip install f5-tts

# CosyVoice
pip install cosyvoice
```

---

## 📊 评估工具

| 工具 | 用途 | 命令 |
|------|------|------|
| **jiwer** | CER/WER计算 | `pip install jiwer` |
| **whisper** | 音频转录 | `pip install openai-whisper` |
| **torchaudio** | 音频处理 | `pip install torchaudio` |

### 评估示例
```python
from jiwer import cer, wer

# 计算CER
error_rate = cer(reference, hypothesis)
print(f"CER: {error_rate:.2%}")
```

---

## 🔧 音频处理

| 工具 | 用途 |
|------|------|
| **librosa** | 特征提取 |
| **pydub** | 音频格式转换 |
| **sox** | 命令行音频处理 |
| **ffmpeg** | 视频/音频转码 |

---

## 🧠 LLM 工具

| 工具 | 用途 | API |
|------|------|-----|
| **OpenAI** | GPT-4重排 | openai |
| **Qwen** | 中文LLM | dashscope |
| **vLLM** | 高效推理 | vllm |

### N-best重排示例
```python
from openai import OpenAI

def llm_rerank(candidates, context=""):
    client = OpenAI()
    prompt = f"""
    对以下ASR候选结果按语义合理性排序:
    {candidates}
    输出最可能正确的结果。
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

---

## 📁 数据处理

| 工具 | 用途 |
|------|------|
| **pandas** | 数据分析 |
| **tqdm** | 进度条 |
| **jsonlines** | JSONL处理 |
| **webdataset** | 大规模数据加载 |

---

## 🖥️ 训练工具

| 工具 | 用途 |
|------|------|
| **DeepSpeed** | 分布式训练 |
| **PEFT** | LoRA/QLoRA |
| **wandb** | 实验追踪 |
| **tensorboard** | 可视化 |

### LoRA配置示例
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)
```

---

## 🔗 推荐资源

- [Hugging Face ASR Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition)
- [ModelScope 语音模型](https://modelscope.cn/models?page=1&tasks=auto-speech-recognition)
- [OpenSLR 数据集](https://www.openslr.org/)
