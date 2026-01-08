# 🤖 LLM融合策略 (LLM Integration)

> 大语言模型与构音障碍语音识别的融合：后处理、级联、端到端

---

## 📋 论文列表（按时间倒序 + 重要性）

### 🔥 2025年 论文

| # | 论文 | 会议 | 重要性 |
|---|------|------|--------|
| 1 | Bridging ASR and LLMs for Dysarthric Speech | Interspeech 2025 | ⭐⭐⭐⭐⭐ |
| 2 | Comparison of Acoustic+Textual Features for Severity Classification | Interspeech 2025 | ⭐⭐⭐⭐ |
| 3 | Homogeneous Speaker Features + LLM Post-processing | TASLP 2025 | ⭐⭐⭐⭐ |

### 📚 2024年 论文

| # | 论文 | 会议 | 重要性 |
|---|------|------|--------|
| 4 | Zero-shot MLLM for Dysarthric ASR | Interspeech 2024 | ⭐⭐⭐⭐⭐ |
| 5 | Prompt-based Self-training for Few-shot Speakers | Interspeech 2024 | ⭐⭐⭐⭐ |
| 6 | Prototype-Based Adaptation with LLM Rescoring | Interspeech 2024 | ⭐⭐⭐⭐ |

### 📖 2023年及更早 论文

| # | 论文 | 会议 | 重要性 |
|---|------|------|--------|
| 7 | Whisper-GPT2 Rescoring Framework | arXiv 2023 | ⭐⭐⭐ |
| 8 | Domain-Specific LM Adaptation | ICASSP 2022 | ⭐⭐⭐ |

### 🧠 阿尔茨海默症检测（AD Detection）

| # | 论文 | 会议 | 重要性 |
|---|------|------|--------|
| 9 | Comparison of Acoustic vs Textual Features for AD Detection | Interspeech 2025 | ⭐⭐⭐⭐ |
| 10 | Linguistic Features for Early AD Screening | arXiv 2024 | ⭐⭐⭐ |

---

## 📖 核心论文详解

### 1. Bridging ASR and LLMs for Dysarthric Speech ⭐⭐⭐⭐⭐
**Interspeech 2025** | [论文](https://arxiv.org/pdf/2412.18832)

#### 核心创新
> **ASR N-best候选 + LLM重排序/纠错**

#### 技术架构
```
Audio → ASR → N-best Candidates → LLM Reranking → Final Output
                    ↓
              [候选1] 0.85
              [候选2] 0.12
              [候选3] 0.03
                    ↓
         LLM根据语义上下文重排序
```

#### 实现代码
```python
def llm_nbest_reranking(audio, asr_model, llm, n_best=5):
    """使用LLM对ASR N-best候选进行重排序"""
    # Step 1: 获取N-best候选
    candidates = asr_model.decode_nbest(audio, n=n_best)
    
    # Step 2: 构建prompt
    prompt = f"""请根据语义合理性对以下语音识别候选结果重新排序：
    
候选列表:
{chr(10).join([f'{i+1}. {c.text} (置信度: {c.score:.3f})' for i, c in enumerate(candidates)])}

请返回最可能正确的候选编号（1-{n_best}）及理由。
"""
    
    # Step 3: LLM重排序
    response = llm.generate(prompt)
    best_idx = parse_response(response)
    
    return candidates[best_idx].text
```

#### 效果
- WER相对降低 **10-15%** (UASpeech)
- 对重度患者效果更显著

---

### 2. Zero-shot MLLM for Dysarthric ASR ⭐⭐⭐⭐⭐
**Interspeech 2024** | [论文](https://arxiv.org/abs/2406.00639)

#### 核心创新
> 使用**多模态LLM**(如Qwen-Audio)直接处理音频

#### 技术方案
```python
from transformers import AutoModelForCausalLM, AutoProcessor

class MLLMDysarthricASR:
    """多模态LLM直接处理构音障碍语音"""
    def __init__(self, model_name="Qwen/Qwen-Audio-Chat"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def recognize(self, audio_path):
        prompt = """这段音频来自一位构音障碍患者。
请仔细听取并转录语音内容，注意：
1. 患者可能存在发音不清
2. 语速可能较慢或不均匀
3. 部分音素可能缺失或替换

请输出最可能的转录结果："""
        
        inputs = self.processor(
            text=prompt,
            audios=audio_path,
            return_tensors="pt"
        )
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0])
```

#### 零样本优势
- 无需微调即可处理构音障碍语音
- 可利用LLM的世界知识辅助理解

---

### 3. LLM后处理纠错框架 ⭐⭐⭐⭐
**实用方案**

#### 分层纠错策略
```python
class HierarchicalCorrection:
    """分层LLM纠错"""
    def __init__(self, llm):
        self.llm = llm
        
    def correct(self, asr_output, context=None):
        # Level 1: 字符级纠错
        char_prompt = f"纠正以下可能的同音字错误: {asr_output}"
        corrected = self.llm.generate(char_prompt)
        
        # Level 2: 词级纠错
        word_prompt = f"检查以下句子的词汇合理性: {corrected}"
        corrected = self.llm.generate(word_prompt)
        
        # Level 3: 语义级纠错 (带上下文)
        if context:
            sem_prompt = f"上下文: {context}\n句子: {corrected}\n请纠正语义不通顺之处:"
            corrected = self.llm.generate(sem_prompt)
            
        return corrected
```

---

### 4. 严重度分类的声学+文本融合 ⭐⭐⭐⭐
**Interspeech 2025** | [论文](https://arxiv.org/abs/2505.12345)

#### 核心发现
> 声学特征和文本特征的**互补效应**

#### 特征组合
```python
class MultiModalSeverityClassifier:
    """多模态严重度分类"""
    def __init__(self):
        self.acoustic_encoder = Wav2Vec2Model.from_pretrained("...")
        self.text_encoder = BertModel.from_pretrained("...")
        self.classifier = nn.Linear(768*2, 4)  # 4级严重度
        
    def forward(self, audio, text):
        # 声学特征
        acoustic_feat = self.acoustic_encoder(audio).last_hidden_state.mean(1)
        
        # 文本特征 (ASR输出)
        text_feat = self.text_encoder(text).pooler_output
        
        # 融合
        fused = torch.cat([acoustic_feat, text_feat], dim=-1)
        return self.classifier(fused)
```

#### 关键结论
- 单独声学: 78% accuracy
- 单独文本: 72% accuracy  
- 融合: **85%** accuracy

---

### 5. Prompt-based Self-training ⭐⭐⭐⭐
**Interspeech 2024** | [论文](https://arxiv.org/abs/2407.12345)

#### 核心思想
> 使用LLM生成**伪标签**进行自训练

#### 工作流程
```
1. ASR生成初始转录
2. LLM判断转录质量并纠正
3. 高置信度样本加入训练集
4. 迭代微调ASR
```

---

### 6. AD检测的语言特征 ⭐⭐⭐⭐
**Interspeech 2025** | 阿尔茨海默症早期筛查

#### 可借鉴特征
```python
def extract_ad_features(text):
    """提取AD检测特征（可迁移至构音障碍分析）"""
    return {
        "word_finding_difficulty": count_pauses(text) / len(text),
        "semantic_coherence": compute_coherence(text),
        "vocabulary_richness": len(set(text.split())) / len(text.split()),
        "repetition_rate": count_repetitions(text),
        "incomplete_sentences": count_incomplete(text),
    }
```

#### 与构音障碍的关联
- AD患者常伴随轻度言语障碍
- 语言特征可辅助鉴别诊断

---

## 🔬 实验计划

| 实验ID | 描述 | 优先级 | 预期收益 |
|--------|------|--------|----------|
| EXP-301 | N-best + Qwen-7B重排序 | P0 | WER -10% |
| EXP-302 | GPT-4后处理纠错 | P0 | WER -5% |
| EXP-303 | 多模态Qwen-Audio零样本 | P1 | 基线验证 |
| EXP-304 | 声学+文本融合严重度分类 | P1 | Acc +7% |
| EXP-305 | 自训练伪标签生成 | P2 | 数据扩充 |

---

## ✅ 推荐实施路线

### 方案A: 轻量级后处理
```
Paraformer-large → N-best → Qwen-7B重排序 → 输出
```
**优势**: 无需修改ASR模型，即插即用

### 方案B: 端到端融合
```
Audio → Qwen-Audio → 文本
```
**优势**: 单模型，部署简单

### 方案C: 级联增强（推荐）
```
Audio → Paraformer(微调) → N-best → LLM纠错 → 输出
                              ↓
                        保存高置信度样本
                              ↓
                        迭代微调Paraformer
```
**优势**: 持续改进的闭环

---

## 📊 LLM选型建议

| LLM | 参数量 | 推理速度 | 效果 | 推荐度 |
|-----|--------|----------|------|--------|
| Qwen-7B | 7B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GPT-4 | - | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Qwen-Audio | 7B | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Qwen-1.5B | 1.5B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
