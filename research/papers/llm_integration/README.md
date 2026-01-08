# 🤖 LLM融合 (LLM Integration)

> **核心问题**: 如何利用大语言模型的语义理解能力提升构音障碍语音识别准确率？

---

## 📋 论文索引

| # | 论文 | 会议/期刊 | 年份 | 核心贡献 | 重要性 |
|---|------|-----------|------|----------|--------|
| 1 | [Bridging ASR and LLMs](#1-bridging-asr-llms) | Interspeech | 2025 | N-best重排序 | ⭐⭐⭐⭐⭐ |
| 2 | [Zero-Shot Recognition with MLLM](#2-zero-shot-mllm) | arXiv | 2024 | 多模态零样本 | ⭐⭐⭐⭐⭐ |
| 3 | [Acoustic + Textual Features](#3-multimodal-severity) | Interspeech | 2025 | 多模态严重度分类 | ⭐⭐⭐⭐ |
| 4 | [Data Augmentation using Speech Synthesis for Severity](#4-severity-classification) | Interspeech | 2025 | 严重度分类增强 | ⭐⭐⭐ |

---

## 📖 论文详解

### 1. Bridging ASR and LLMs for Dysarthric Speech Recognition
**Interspeech 2025** | [论文链接](https://arxiv.org/abs/2508.08027)

#### 核心思想
利用LLM的**上下文推理能力**对ASR输出的N-best候选列表进行**重排序与修复**。

#### 架构
```
音频 → ASR → N-best候选列表 → LLM重排序 → 最终输出
              ↓
        [候选1: "打开空挑"]
        [候选2: "打开空条"]
        [候选3: "打开空调"]  ← LLM选择最合理的
```

#### 关键发现

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Q-Former信息压缩 | 细节丢失 | 改用**线性投影**保留时序特征 |
| 泛化性瓶颈 | 领域差异 | 全链路LoRA (Whisper端+LLM端) |

#### 移植方案
```python
import openai

def llm_rerank(n_best_list, context="构音障碍患者日常指令"):
    """使用LLM对N-best候选进行重排序"""
    
    prompt = f"""你是一个构音障碍语音识别助手。
患者说话可能不清楚，ASR可能识别错误。

场景: {context}
ASR候选列表:
{chr(10).join([f'{i+1}. {c}' for i, c in enumerate(n_best_list)])}

请选择最可能是患者真实意图的选项，只回复数字序号。"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    selected_idx = int(response.choices[0].message.content.strip()) - 1
    return n_best_list[selected_idx]

# 使用示例
candidates = ["打开空挑", "打开空条", "打开空调"]
result = llm_rerank(candidates)  # → "打开空调"
```

#### 全链路LoRA策略
```python
# Whisper端: 修正声学特征
whisper_lora = LoraConfig(
    r=8,
    target_modules=["encoder.layers.*.self_attn"],
    task_type="FEATURE_EXTRACTION"
)

# LLM端: 适配病理语义
llm_lora = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
```

#### 实验计划
- [ ] EXP-301: N-best重排序基线实验
- [ ] EXP-302: 不同LLM对比 (GPT-4 vs Qwen vs Llama)
- [ ] EXP-303: 线性投影 vs Q-Former
- [ ] EXP-304: 全链路LoRA微调

---

### 2. Zero-Shot Recognition with Multimodal LLMs
**arXiv 2024** | [论文链接](https://arxiv.org/abs/2512.17474)

#### 核心思想
> 采用本地Paraformer处理**高置信度语音**，低置信度样本触发**云端GPT-5/Gemini-3**

#### 分层架构
```
                    ┌─────────────┐
输入音频 ──────────→│ 本地ASR      │
                    │ (Paraformer) │
                    └──────┬──────┘
                           │
                    置信度判断
                    ↙         ↘
              高置信度      低置信度
                ↓              ↓
          直接输出        ┌─────────────┐
                         │ 云端MLLM     │
                         │ (GPT-5/Gemini)│
                         └─────────────┘
                               │
                         结合用户历史
                         In-Context Learning
                               ↓
                          最终输出
```

#### 关键技术
1. **置信度阈值**: 通过CTC解码的路径概率判断
2. **In-Context Learning**: 利用用户历史发音作为few-shot示例
3. **多模态输入**: 直接传音频给MLLM，而非文本

#### 移植方案
```python
class HybridASR:
    def __init__(self, local_model, cloud_api, threshold=0.7):
        self.local = local_model
        self.cloud = cloud_api
        self.threshold = threshold
        self.user_history = []  # 用户历史 (音频, 文本) 对
        
    def recognize(self, audio):
        # 本地识别
        result, confidence = self.local.recognize(audio, return_confidence=True)
        
        if confidence > self.threshold:
            return result
        else:
            # 调用云端MLLM
            return self.cloud_correction(audio, result)
    
    def cloud_correction(self, audio, local_result):
        # 构建few-shot prompt
        examples = self.get_similar_history(audio, k=3)
        
        prompt = f"""这是一位构音障碍患者的语音。
本地ASR识别为: {local_result}

以下是该患者的历史发音示例:
{self.format_examples(examples)}

请根据上下文推断患者的真实意图:"""
        
        return self.cloud.complete(audio, prompt)
```

#### 实验计划
- [ ] EXP-305: 置信度阈值调优
- [ ] EXP-306: 用户历史对In-Context效果的影响
- [ ] EXP-307: 本地+云端混合延迟评估

---

### 3. Acoustic + Textual Features for Severity Classification
**Interspeech 2025** | [论文链接](https://www.isca-archive.org/interspeech_2025/ys25_interspeech.pdf)

#### 核心思想
融合**声学特征分析**和**ASR转写的文本特征**（词性/句法分析），构建多模态严重度分类器。

#### 洞察
> 患者不仅发音不准，语言组织能力也可能退化

#### 特征类型
| 类型 | 特征 | 分析维度 |
|------|------|----------|
| 声学 | MFCC, F0, 能量 | 发音物理特性 |
| 文本 | 词频, 句长, 复杂度 | 语言能力 |
| 语法 | 词性分布, 依存树深度 | 认知能力 |

#### 应用场景
- 自动评估严重程度
- 为用户匹配合适的ASR模型
- 提供多维度康复建议

#### 移植方案
```python
class SeverityClassifier:
    def __init__(self):
        self.acoustic_encoder = AcousticEncoder()
        self.text_encoder = TextEncoder()  # BERT-based
        self.classifier = nn.Linear(768 + 256, 4)  # 4级严重度
        
    def forward(self, audio, transcript):
        acoustic_feat = self.acoustic_encoder(audio)
        text_feat = self.text_encoder(transcript)
        
        fused = torch.cat([acoustic_feat, text_feat], dim=-1)
        return self.classifier(fused)
```

#### 实验计划
- [ ] EXP-308: 声学 vs 文本 vs 融合 对比
- [ ] EXP-309: 严重度分类引导ASR模型选择

---

### 4. Data Augmentation for Severity Classification
**Interspeech 2025** | [论文链接](https://www.isca-archive.org/interspeech_2025/kim25w_interspeech.pdf)

#### 核心策略
利用可控TTS合成不同严重等级的构音障碍语音，采用**逆严重度加权**的数据混合策略。

#### 逆严重度加权
```
重度样本 (CER高): 合成:真实 = 3:1  (合成更多)
轻度样本 (CER低): 合成:真实 = 1:1  (合成较少)
```

#### 课程学习
```
训练初期: 合成数据占比高
    ↓
训练后期: 逐步剔除合成数据，迫使模型适配真实病理特征
```

---

## 🧪 实验计划总览

### EXP-3XX: LLM融合实验系列

| ID | 实验名称 | 假设 | 优先级 |
|----|----------|------|--------|
| EXP-301 | N-best重排序基线 | LLM可提升10%+准确率 | P0 |
| EXP-302 | LLM模型对比 | GPT-4 > Qwen > Llama | P1 |
| EXP-303 | 线性投影vs Q-Former | 线性投影保留更多信息 | P2 |
| EXP-304 | 全链路LoRA | ASR+LLM联合微调效果最佳 | P1 |
| EXP-305 | 置信度阈值调优 | 0.7附近最优 | P1 |
| EXP-306 | 用户历史ICL | 历史示例提升5%+ | P2 |
| EXP-307 | 混合架构延迟 | 云端调用增加200-500ms | P1 |
| EXP-308 | 多模态严重度分类 | 融合优于单模态 | P2 |
| EXP-309 | 严重度引导模型选择 | 自动路由提升泛化性 | P2 |

---

## 💡 核心结论与建议

### ✅ 推荐的LLM融合策略

```
最小可行方案 (MVP):
1. 本地Paraformer识别
2. 置信度低时调用GPT-4 API纠错
3. 简单prompt工程，无需训练

进阶方案:
1. N-best重排序 + 领域prompt
2. 全链路LoRA (ASR端 + LLM端)
3. 用户历史In-Context Learning
```

### 📊 成本-效果权衡

| 方案 | 效果提升 | 延迟增加 | API成本 | 推荐 |
|------|----------|----------|---------|------|
| 无LLM | 基线 | 0 | 0 | - |
| GPT-4纠错 | +15% | +500ms | $$$ | MVP |
| 开源LLM本地 | +10% | +200ms | $$ | 成本敏感 |
| 全链路LoRA | +20% | +100ms | $ | 最佳平衡 |

---

## 📚 相关资源

- [OpenAI API](https://platform.openai.com/)
- [通义千问 API](https://help.aliyun.com/zh/dashscope/)
- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) - 开源多模态音频LLM
