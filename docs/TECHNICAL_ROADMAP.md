# 🎯 构音障碍ASR系统技术路线图

> **项目**: CLEAR-VOX (构音障碍语音识别)
> **目标**: 低资源 ASR + Agent + TTS 三阶段迭代开发

---

## 📊 流式 vs 非流式 ASR 深度对比

### 1. 架构差异

| 特性 | 非流式 (Offline) | 流式 (Streaming) |
|------|------------------|------------------|
| **输入方式** | 完整音频 | 音频块 (chunk) |
| **延迟** | 音频结束后出结果 | 实时输出 (600ms/480ms) |
| **上下文** | 全局上下文 | 有限前瞻/回看 |
| **精度** | ✅ 更高 | ⚠️ 略低 (1-3% CER 差距) |
| **应用场景** | 转写、字幕后处理 | 实时对话、会议 |
| **资源占用** | 可批处理优化 | 需要持续计算 |

### 2. FunASR 流式实现

```python
# 流式模型配置
chunk_size = [0, 10, 5]  # 600ms 延迟配置
# [0, 10, 5] = 600ms 输出粒度, 300ms 未来上下文
# [0, 8, 4]  = 480ms 输出粒度, 240ms 未来上下文

encoder_chunk_look_back = 4  # encoder 回看4个chunk
decoder_chunk_look_back = 1  # decoder 回看1个chunk

model = AutoModel(model="paraformer-zh-streaming")

# 流式处理循环
cache = {}
for chunk in audio_chunks:
    res = model.generate(
        input=chunk, 
        cache=cache,  # 状态缓存
        is_final=is_last_chunk,
        chunk_size=chunk_size
    )
```

### 3. 延迟分析

| 配置 | 输出延迟 | 理论RTF | 适用场景 |
|------|----------|---------|----------|
| [0,10,5] | 600ms | 0.1-0.2 | 标准实时 |
| [0,8,4] | 480ms | 0.15-0.25 | 低延迟实时 |
| Offline | N/A | 0.05-0.1 | 后处理转写 |

---

## 🔬 SenseVoice vs Paraformer 对比

### 1. 模型规格对比

| 特性 | Paraformer-large | SenseVoice-Small |
|------|------------------|------------------|
| **参数量** | 220M | 330M |
| **架构** | 非自回归 NAR | 自回归 AR |
| **支持语言** | 中文为主 | 中/英/日/韩/粤 |
| **额外功能** | ❌ | ✅ 情感/事件/语言检测 |
| **推理速度** | ⚡ 快 (10x+) | 较慢 |
| **流式支持** | ✅ 原生 | ❌ 无流式版 |
| **微调代码** | ✅ 完整 | ✅ 完整 |
| **3090显存** | ~8GB | ~12GB |

### 2. 功能对比

**Paraformer-large**:
```python
# 纯ASR + 标点
model = AutoModel(model="paraformer-zh", punc_model="ct-punc")
res = model.generate(input="audio.wav")
# 输出: {"text": "这是转录结果，带标点。"}
```

**SenseVoice-Small**:
```python
# 多功能: ASR + 语言 + 情感 + 事件
model = AutoModel(model="iic/SenseVoiceSmall")
res = model.generate(input="audio.wav", language="auto")
# 输出: {"text": "这是转录结果<|NEUTRAL|><|Speech|>", "language": "zh"}

# 情感标签: <|HAPPY|>, <|SAD|>, <|ANGRY|>, <|NEUTRAL|>...
# 事件标签: <|Speech|>, <|BGM|>, <|Applause|>, <|Laughter|>...
```

### 3. 构音障碍场景选择建议

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| **第一版 (低资源)** | **Paraformer-large** | 速度快，微调完善，显存友好 |
| 实时对话 | Paraformer-streaming | 原生支持流式 |
| 情感分析 | SenseVoice | 内置情感检测 |
| 多语言 | SenseVoice | 支持5种语言 |

---

## 📈 三阶段开发路线图

### 🚀 第一阶段：低资源基础版

**目标**: ASR + Agent + TTS 基础链路打通

**技术选型**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  ASR        │────▶│  Agent      │────▶│  TTS        │
│  Paraformer │     │  LLM API    │     │  VITS/GPT   │
│  (非流式)    │     │  (低延迟)    │     │  (非流式)    │
└─────────────┘     └─────────────┘     └─────────────┘
```

| 组件 | 技术选型 | 延迟容忍 |
|------|----------|----------|
| ASR | Paraformer-large + 微调 | 2-5s (允许) |
| Agent | GPT-4o / Claude / 讯飞星火 | 1-3s |
| TTS | VITS / GPT-SoVITS | 1-2s |

**开发任务**:
- [x] 数据准备 (prepare_1h_dataset.py)
- [x] 训练脚本 (finetune_paraformer.sh)
- [ ] 基线测试
- [ ] 模型微调
- [ ] Agent 集成
- [ ] TTS 集成
- [ ] 端到端测试

**预期指标**:
- ASR CER: < 20% (优于人类基线20.45%)
- 端到端延迟: 5-10s (可接受)
- 准确性: 允许不精确

---

### ⚡ 第二阶段：流式实时优化

**目标**: Agent + TTS 流式适配，实时性优化

**技术升级**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  ASR        │────▶│  Agent      │────▶│  TTS        │
│  Paraformer │     │  LLM        │     │  GPT-SoVITS │
│  (流式)     │     │  (流式输出)  │     │  (流式)     │
└─────────────┘     └─────────────┘     └─────────────┘
      ↓                   ↓                   ↓
   600ms              逐token             流式合成
```

**关键优化**:

1. **ASR 流式改造**:
```python
# 从非流式
model = AutoModel(model="paraformer-zh")

# 改为流式
model = AutoModel(model="paraformer-zh-streaming")
chunk_size = [0, 10, 5]  # 600ms
```

2. **Agent 流式输出**:
```python
# OpenAI streaming
for chunk in client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    stream=True
):
    yield chunk.choices[0].delta.content
```

3. **TTS 流式合成**:
```python
# GPT-SoVITS 流式模式
for audio_chunk in tts.stream_generate(text):
    yield audio_chunk
```

**预期指标**:
- ASR 首字延迟: 600ms
- Agent 首字延迟: 500ms
- TTS 首音延迟: 300ms
- **端到端首响应**: < 1.5s

---

### 🎯 第三阶段：精度与能力优化

**目标**: ASR准确性提升 + Agent能力增强

**优化方向**:

1. **ASR 精度提升**:
   - 数据增强 (SpecAugment, Speed Perturbation)
   - 更大数据集 (133h CDSD 全量)
   - 模型集成/平均
   - 语言模型重打分 (LM Rescoring)

2. **Agent 能力增强**:
   - 构音障碍特定提示词优化
   - 知识库检索增强 (RAG)
   - 多轮对话管理
   - 意图理解优化

3. **多说话人支持 (后续)**:
   - 会议室场景 (麦克风标注speaker ID)
   - 在野场景 (模型说话人分离)

**技术储备**:
```python
# FunASR 说话人分离
model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++"  # 说话人聚类
)
```

**目标指标**:
- ASR CER: < 16.4% (达到CDSD SOTA)
- Agent 意图准确率: > 90%
- 多说话人 DER: < 20%

---

## 🛠️ 当前优先级

### 立即执行 (第一阶段)

| 优先级 | 任务 | 状态 | 说明 |
|--------|------|------|------|
| P0 | 基线测试 | ⬜ | 原始Paraformer在测试集的CER |
| P0 | 模型微调 | ⬜ | bash finetune_paraformer.sh |
| P1 | Agent集成 | ⬜ | OpenAI/Claude API |
| P1 | TTS选型 | ⬜ | VITS vs GPT-SoVITS |
| P2 | 端到端测试 | ⬜ | 完整链路延迟测试 |

### 推荐执行顺序

```bash
# 1. 测试基线
python scripts/inference_test.py \
  --model "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
  --test data/1h_dataset/test.jsonl \
  --output exp/baseline_results.json

# 2. 开始微调
bash scripts/finetune_paraformer.sh

# 3. 测试微调后
python scripts/inference_test.py \
  --model exp/paraformer_finetune_1h/model.pt.avg_10 \
  --test data/1h_dataset/test.jsonl \
  --output exp/finetune_results.json
```

---

## 📚 参考资料

### FunASR 官方文档
- GitHub: https://github.com/modelscope/FunASR
- 文档: https://funasr.readthedocs.io
- ModelScope: https://modelscope.cn

### 模型资源
| 模型 | ModelScope ID |
|------|---------------|
| Paraformer-large | iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch |
| Paraformer-streaming | iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online |
| SenseVoice-Small | iic/SenseVoiceSmall |
| FSMN-VAD | iic/speech_fsmn_vad_zh-cn-16k-common-pytorch |
| CT-Punc | iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch |

### CDSD 论文
- 标题: CDSD: Chinese Dysarthria Speech Database
- 会议: INTERSPEECH 2024
- arXiv: https://arxiv.org/abs/2310.15930

---

**作者**: GitHub Copilot  
**日期**: 2025-12-23  
**版本**: v1.0
