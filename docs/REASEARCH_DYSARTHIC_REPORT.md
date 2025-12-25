# 📊 构音障碍语音识别 (Dysarthric ASR) 技术调研报告

> **CLEAR-VOX Project Research Report**  
> 版本: 2.0 | 日期: 2025-12-23 (更新)  
> 作者: CLEAR-VOX Research Team

---

## 📋 目录

1. [模型版本与现状](#1-模型版本与现状)
2. [2025年最新研究进展 (重大更新)](#2-2025年最新研究进展)
3. [INTERSPEECH 2025 重要论文](#3-interspeech-2025-重要论文)
4. [大语言模型在构音障碍ASR的应用](#4-大语言模型在构音障碍asr的应用)
5. [开源代码与模型资源](#5-开源代码与模型资源)
6. [主流训练策略对比](#6-主流训练策略对比)
7. [针对CDSD+Paraformer的优化建议](#7-针对cdsdparaformer的优化建议)
8. [重要数据集](#8-重要数据集)
9. [参考文献](#9-参考文献)
10. [GitHub开源项目详细分析](#10-github开源项目详细分析)
11. [补充GitHub仓库资源](#11-补充-github-仓库资源-2025-12-24-更新)

---

## 1. 模型版本与现状

### 1.1 Paraformer-large 模型信息

| 属性 | 值 |
|------|-----|
| **模型ID** | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| **当前版本** | v2.0.4 (最新稳定版) |
| **发布时间** | 2024-02-01 |
| **参数量** | 220M |
| **预训练数据** | 60,000+ 小时中文语音 |
| **下载量** | 32,005,526+ 次 |

### 1.2 2025年基础模型现状

| 模型 | 参数量 | 特点 | 构音障碍表现 |
|------|--------|------|------------|
| **Whisper Large-v3** | 1.5B | 多语言强 | WER~45% (重度) |
| **HuBERT Large** | 300M | 自监督特征 | 最佳特征提取器 |
| **WavLM Large** | 300M | 多任务预训练 | 鲁棒性强 |
| **Paraformer-large** | 220M | 中文专精 | 微调后效果好 |

---

## 2. 2025年最新研究进展

### 🔥 2.1 顶级会议/期刊论文汇总

#### 📌 [ICASSP 2025] Phone-purity Guided Discrete Tokens
**arXiv:2501.04379** - 香港中文大学 Xunying Liu 团队

> **核心贡献**:
> - 提出 Phone-purity Guided (PPG) 离散token提取方法
> - 使用音素标签监督来正则化K-means和VAE-VQ
> - 在UASpeech上获得 **23.25% WER** (最佳系统融合)
> - **WER降低 0.99%~1.77%** (统计显著)

**技术细节**:
```
传统方法: Audio → HuBERT → K-means/VAE-VQ → Discrete Tokens
PPG方法:  Audio → HuBERT → PPG-guided K-means → Better Tokens
                                ↓
                   Phoneme Label Supervision (正则化)
```

**对本项目启示**: 可以探索使用离散token而非直接微调。

---

#### 📌 [ICASSP 2025] Robust Cross-Etiology Speaker-Independent DSR
**arXiv:2501.14994** - Singh et al.

> **核心贡献**:
> - 首个跨病因(cross-etiology)的构音障碍ASR研究
> - 说话人无关(speaker-independent)系统设计
> - 在多个数据集上验证泛化性

**对本项目启示**: 考虑跨说话人泛化能力评估。

---

#### 📌 [NAACL 2025 Main] DyPCL: Dynamic Phoneme-level Contrastive Learning
**arXiv:2501.19010** - 韩国POSTECH Lee et al.

> **核心贡献**:
> - **动态音素级对比学习**: 将语音分解为音素片段
> - **课程学习**: 从简单负样本逐渐过渡到困难负样本
> - 在UASpeech上 **WER相对降低 22.10%**
> - 解决了说话人内部变异性问题

**技术亮点**:
```python
# DyPCL 核心思想
1. 音素分割: 使用CTC对齐将语音分解为音素单元
2. 音素对比: 同一音素的不同说话人表示应接近
3. 动态课程: 基于音素相似度选择负样本
   - 初期: 容易区分的负样本 (如 /a/ vs /k/)
   - 后期: 困难区分的负样本 (如 /b/ vs /p/)
```

**对本项目启示**: 可以考虑音素级特征对齐。

---

#### 📌 [INTERSPEECH 2025] MoE Speaker Adaptation (SOTA)
**arXiv:2505.22072** - 香港中文大学

> **核心贡献**:
> - **零样本MoE说话人适应**: 无需针对新说话人微调
> - 实时处理 + 领域知识融合
> - **最低发表WER: 16.35%** (UASpeech) 🏆
> - 极低可懂度: 46.77% WER
> - **RTF加速7倍** vs batch-mode adaptation

**技术架构**:
```
Speech Input
    ↓
[Severity Predictor] → Severity Embedding
    ↓
[Gender Predictor] → Gender Embedding
    ↓
[MoE Router] → 动态组合专家网络
    ↓
[HuBERT/WavLM Backbone]
    ↓
ASR Output
```

**对本项目启示**: MoE架构可显著提升泛化性。

---

#### 📌 [INTERSPEECH 2025] Self-Training for Long Dysarthric Speech
**arXiv:2506.22810** - 中科大 Zhao et al.

> **核心贡献**:
> - 针对长句构音障碍语音的自训练方法
> - 增加训练数据 + 适应不完整语音片段
> - **SAP Challenge 第二名** 🥈

**SAP Challenge背景**: Speech Accessibility Project 是由 UIUC 联合 Amazon、Apple、Google、Meta、Microsoft 发起的大规模构音障碍数据收集项目。

**对本项目启示**: 自训练是扩充数据的有效方法。

---

#### 📌 [INTERSPEECH 2025] Voice Conversion for Low-Resource Languages
**arXiv:2505.14874** - CMU et al.

> **核心贡献**:
> - 使用语音转换将健康语音转为"类构音障碍"语音
> - 支持低资源语言: Spanish (PC-GITA), Italian (EasyCall), Tamil (SSNCE)
> - **显著优于Speed/Tempo Perturbation传统增强**

**技术流程**:
```
1. 在英文构音障碍数据(UASpeech)上训练VC模型
2. 将健康语音(FLEURS)转换为类构音障碍语音
3. 用生成数据微调MMS多语言ASR模型
```

**对本项目启示**: 可以用VC生成更多中文构音障碍数据。

---

#### 📌 [INTERSPEECH 2025] Generative Error Correction (GER4Dys)
**arXiv:2505.20163** - 意大利都灵理工

> **核心贡献**:
> - 使用生成式模型进行ASR后处理错误纠正
> - 结合语言模型修复构音障碍导致的识别错误

**开源代码**: https://github.com/MorenoLaQuatra/GER4Dys

**对本项目启示**: LLM后处理是低成本提升方案。

---

#### 📌 [2025.09] MetaICL for On-the-fly Personalization (Google)
**arXiv:2509.15516** - Google Research

> **核心贡献**:
> - 元学习 + In-Context Learning 实现即时个性化
> - **无需存储个人adapter，单模型服务所有用户**
> - Euphonia: **13.9% WER** (vs baseline 17.5%)
> - SAP Test 1: **5.3% WER** (vs personalized adapter 8%)

**关键发现**:
- 5个精选样本 ≈ 19个随机样本的效果
- 样本选择策略至关重要

**对本项目启示**: 少样本适应是实际部署的关键。

---

### 2.2 2025年研究趋势总结

| 趋势 | 代表工作 | 关键技术 |
|------|----------|----------|
| **零样本/少样本适应** | MoE, MetaICL | MoE、元学习、ICL |
| **音素级特征学习** | DyPCL, PPG-Tokens | 对比学习、离散化 |
| **数据增强** | Voice Conversion | VC生成合成数据 |
| **LLM后处理** | GER4Dys | 生成式错误纠正 |
| **隐私保护** | Federated Learning | 联邦学习 |

---

## 3. INTERSPEECH 2025 重要论文

### 完整论文列表 (按发表时间)

| 月份 | 论文 | 核心贡献 | 数据集 |
|------|------|----------|--------|
| 5月 | Personalized Fine-Tuning with Controllable Synthetic Speech (arXiv:2505.12991) | LLM生成转录+可控TTS | SAP |
| 5月 | Towards Inclusive ASR: VC for Low-Resource (arXiv:2505.14874) | 语音转换跨语言增强 | PC-GITA等 |
| 5月 | Robust fine-tuning via Model Merging (arXiv:2505.20477) | 模型融合鲁棒微调 | UASpeech |
| 5月 | GER4Dys: Generative Error Correction (arXiv:2505.20163) | 生成式后处理纠错 | UASpeech |
| 5月 | MoE Speaker Adaptation (arXiv:2505.22072) | 零样本MoE适应 | UASpeech |
| 6月 | Towards Temporally Explainable Assessment (arXiv:2506.00454) | 可解释清晰度评估 | 自建 |
| 6月 | Unsupervised Rhythm and Voice Conversion (arXiv:2506.01618) | 韵律+声音转换增强 | UASpeech |
| 6月 | Federated Learning for DSR (arXiv:2506.11069) | 隐私保护联邦学习 | UASpeech |
| 6月 | Self-Training for Whisper (arXiv:2506.22810) | 长句自训练方法 | SAP |
| 8月 | Diffusion-Based Enhancement (arXiv:2508.17980) | 扩散模型语音增强 | UASpeech |

---

## 4. 大语言模型在构音障碍ASR的应用

### 4.1 2025年LLM应用进展

#### 📌 [2025.08] Bridging ASR and LLMs Benchmark
**arXiv:2508.08027**

> **系统性对比**:
> - 评测模型: Wav2Vec2, HuBERT, Whisper
> - 解码策略: CTC, Seq2Seq, LLM-enhanced
> - LLM: BART, GPT-2, Vicuna

**关键发现**:
- **LLM解码可利用语言约束修复音素错误**
- **语法纠正对构音障碍特别有效**

---

#### 📌 [2025.10] Multilingual Framework
**arXiv:2510.03986**

> **统一框架包含**:
> 1. 二分类检测 (97%准确率)
> 2. 严重程度分类 (97%准确率)
> 3. 干净语音生成
> 4. 语音转文字 (WER 13.67%)
> 5. 情感检测
> 6. 声音克隆

**技术亮点**: 跨语言迁移学习 (俄语→英语)

---

### 4.2 LLM后处理最佳实践

```python
# 2025年推荐的LLM后处理方案
def asr_with_llm_correction(asr_output, severity="moderate"):
    """
    使用LLM进行构音障碍ASR结果纠正
    
    参考: arXiv:2508.08027, arXiv:2505.20163
    """
    
    # 根据严重程度调整prompt
    if severity == "severe":
        prompt = f"""
        以下是重度构音障碍患者的语音识别结果，可能存在严重的音素替换和省略。
        请根据上下文语义和常见表达进行智能纠正，尽量保留原意：
        
        识别结果: {asr_output}
        
        注意:
        1. 常见错误: 声母混淆 (b/p, d/t, g/k)
        2. 可能有音节省略
        3. 优先考虑语义完整性
        
        纠正后:
        """
    else:
        prompt = f"""
        以下是构音障碍患者的语音识别结果，请纠正明显的错误：
        
        识别结果: {asr_output}
        纠正后:
        """
    
    # 调用LLM
    corrected = llm_client.generate(prompt)
    return corrected
```

---

## 5. 开源代码与模型资源

### 5.1 2025年发布的开源代码

| 项目 | 论文 | 链接 | 语言 |
|------|------|------|------|
| **GER4Dys** | INTERSPEECH 2025 | [GitHub](https://github.com/MorenoLaQuatra/GER4Dys) | Python |
| **PB-DSR** | INTERSPEECH 2024 | [GitHub](https://github.com/NKU-HLT/PB-DSR) | Python |
| **Multitask-DSA** | ASRU 2025 | [GitHub](https://github.com/th-nuernberg/multitask-dysarthric-speech-analysis) | Python |
| **SpeechVision** | 2024 | [GitHub](https://github.com/rshahamiri/SpeechVision) | Keras |
| **asr-dysarthria** | 研究项目 | [GitHub](https://github.com/jmaczan/asr-dysarthria) | Jupyter |
| **Dys_Locate** | 2024 | [GitHub](https://github.com/abhinavbammidi1401/Dys_Locate) | Python |

### 5.2 预训练模型资源

| 模型 | 平台 | 用途 | 推荐度 |
|------|------|------|--------|
| **HuBERT-large** | HuggingFace | 特征提取 | ⭐⭐⭐⭐⭐ |
| **WavLM-large** | HuggingFace | 特征提取 | ⭐⭐⭐⭐⭐ |
| **Whisper-large-v3** | OpenAI | 直接推理 | ⭐⭐⭐⭐ |
| **Paraformer-large** | ModelScope | 中文ASR | ⭐⭐⭐⭐ |
| **MMS** | Meta | 多语言ASR | ⭐⭐⭐⭐ |

### 5.3 相关工具链

```bash
# 推荐安装的工具
pip install funasr          # 阿里达摩院ASR框架
pip install transformers    # HuggingFace模型加载
pip install openai-whisper  # Whisper模型
pip install speechbrain     # 语音处理工具箱
pip install kaldi-io        # Kaldi格式支持
```

---

## 6. 主流训练策略对比

### 6.1 2025年策略排名

| 排名 | 策略 | 代表论文 | 效果 | 实现难度 |
|------|------|----------|------|----------|
| 🥇 | **MoE说话人适应** | arXiv:2505.22072 | 最佳 | 高 |
| 🥈 | **音素级对比学习** | arXiv:2501.19010 | 极佳 | 中 |
| 🥉 | **PPG离散Token** | arXiv:2501.04379 | 优秀 | 中 |
| 4 | 语音转换增强 | arXiv:2505.14874 | 优秀 | 中 |
| 5 | 自训练 | arXiv:2506.22810 | 良好 | 低 |
| 6 | LLM后处理 | arXiv:2505.20163 | 良好 | 低 |
| 7 | 直接微调 | 基线方法 | 基线 | 最低 |

### 6.2 策略详解

#### 🏆 策略1: MoE说话人适应 (SOTA)
```
优点: 零样本泛化、实时处理、效果最佳
缺点: 实现复杂、需要专家网络设计
适用: 需要部署给大量用户的产品

核心思想:
1. 预测说话人严重程度 → 选择对应专家
2. 预测说话人性别 → 微调专家组合
3. KL散度约束 → 增强专家多样性
```

#### 🥈 策略2: 音素级对比学习 (DyPCL)
```
优点: 细粒度特征学习、22% WER相对降低
缺点: 需要音素对齐、训练较复杂
适用: 有音素标注或可以用CTC对齐的场景

核心思想:
1. CTC对齐 → 音素片段
2. 同音素不同说话人 → 正样本对
3. 不同音素 → 负样本 (按相似度难度递增)
```

#### �� 策略3: PPG离散Token
```
优点: 特征更可解释、可与下游任务解耦
缺点: 需要音素标签、额外训练步骤
适用: 有音素标注的数据集

核心思想:
1. HuBERT提取连续特征
2. K-means聚类 + 音素纯度正则化
3. 离散token输入下游ASR
```

---

## 7. 针对CDSD+Paraformer的优化建议

### 7.1 当前方案评估

| 配置项 | 当前值 | 评价 |
|--------|--------|------|
| 学习率 | 0.0002 | ✅ 合适 |
| Epoch | 50 | ✅ 足够 |
| Batch Size | 24 | ✅ GPU限制下合理 |
| 数据增强 | 无 | ⚠️ 建议添加 |
| LLM后处理 | 无 | ⚠️ 建议添加 |

### 7.2 推荐优化路线图

```
阶段1: 基线微调 (当前)
├── 直接微调Paraformer
├── 预期CER: 50-60%
└── 时间: 1天

阶段2: 数据增强 (下一步)
├── Speed Perturbation [0.9, 1.0, 1.1]
├── SpecAugment增强
├── 预期CER: 40-50%
└── 时间: 1天

阶段3: LLM后处理 (推荐)
├── 使用Qwen/ChatGLM进行纠错
├── 参考GER4Dys实现
├── 预期CER: 35-45%
└── 时间: 0.5天

阶段4: 高级策略 (可选)
├── 选项A: 对比学习 (DyPCL风格)
├── 选项B: 语音转换增强
├── 预期CER: 30-40%
└── 时间: 3-5天
```

### 7.3 立即可用的优化配置

#### 配置A: Speed Perturbation
```bash
# 添加到训练命令
++dataset_conf.preprocessor_speech="SpeechPreprocessSpeedPerturb" \
++dataset_conf.preprocessor_speech_conf.speed_perturb="[0.9, 1.0, 1.1]"
```

#### 配置B: 增强SpecAugment
```bash
# 添加到训练命令
++specaug_conf.num_freq_mask=2 \
++specaug_conf.num_time_mask=2 \
++specaug_conf.time_mask_width_range="[0, 40]"
```

#### 配置C: LLM后处理脚本
```python
#!/usr/bin/env python3
"""
构音障碍ASR后处理 - 基于GER4Dys思想
"""
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

class DysarthricASRPostProcessor:
    def __init__(self, model_name="Qwen/Qwen2-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def correct(self, asr_text, context=None):
        prompt = f"""你是一个专业的构音障碍语音识别纠错专家。
以下是ASR系统识别的构音障碍患者语音，可能存在错误。
请纠正明显的错误，保留原意。只输出纠正后的文本。

识别结果: {asr_text}
纠正后:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected.split("纠正后:")[-1].strip()

# 使用示例
if __name__ == "__main__":
    processor = DysarthricASRPostProcessor()
    result = processor.correct("我像去一晚")  # 应为"我想去医院"
    print(f"纠正结果: {result}")
```

---

## 8. 重要数据集

### 8.1 公开数据集汇总

| 数据集 | 语言 | 时长 | 说话人 | 最佳CER/WER | 链接 |
|--------|------|------|--------|------------|------|
| **UASpeech** | 英语 | 10h | 15 | 16.35% WER | [论文](http://www.isle.illinois.edu/sst/data/UASpeech/) |
| **TORGO** | 英语 | 23h | 15 | ~20% WER | [链接](http://www.cs.toronto.edu/~complingweb/data/TORGO/) |
| **CDSD** | 中文 | 133h | 44 | 16.4% CER | INTERSPEECH 2024 |
| **SAP** | 英语 | 大规模 | 2000+ | 进行中 | [链接](https://speechaccessibilityproject.beckman.illinois.edu/) |
| **PC-GITA** | 西班牙语 | - | - | - | 帕金森患者 |
| **EasyCall** | 意大利语 | - | - | - | 构音障碍 |

### 8.2 Speech Accessibility Project (SAP) 重要更新

> **2025年9月数据**: 约2000名参与者已提交录音

**行业支持**: Amazon, Apple, Google, Meta, Microsoft

**微软成果**: 使用SAP数据实现 **18%-60%准确率提升** (Azure ASR)

**数据获取**: 需提交研究提案申请使用权

---

## 9. 参考文献

### 2025年核心论文 (按重要性排序)

1. **[INTERSPEECH 2025] MoE Speaker Adaptation**
   - Hu et al. "On-the-fly Routing for Zero-shot MoE Speaker Adaptation"
   - arXiv:2505.22072
   - **最低WER: 16.35%** 🏆

2. **[NAACL 2025] DyPCL**
   - Lee et al. "DyPCL: Dynamic Phoneme-level Contrastive Learning"
   - arXiv:2501.19010
   - **22% WER相对降低**

3. **[ICASSP 2025] PPG Discrete Tokens**
   - Wang et al. "Phone-purity Guided Discrete Tokens"
   - arXiv:2501.04379
   - **最低WER: 23.25%**

4. **[INTERSPEECH 2025] Self-Training for Long DSR**
   - Wang et al. "A Self-Training Approach for Whisper"
   - arXiv:2506.22810
   - **SAP Challenge第二名** 🥈

5. **[INTERSPEECH 2025] Voice Conversion for Low-Resource**
   - Li et al. "Towards Inclusive ASR"
   - arXiv:2505.14874

6. **[INTERSPEECH 2025] GER4Dys**
   - La Quatra et al. "Exploring Generative Error Correction"
   - arXiv:2505.20163
   - **开源代码**

7. **[2025.09] MetaICL Personalization**
   - Agarwal et al. "State-of-the-Art DSR with MetaICL"
   - arXiv:2509.15516
   - **Google Research**

8. **[2025.08] CDSD Cross-Learning**
   - Xiao et al. "Cross-Learning Fine-Tuning Strategy"
   - arXiv:2508.18732
   - **针对CDSD数据集**

9. **[2025.10] Multilingual Framework**
   - Raghu et al. "A Multilingual Framework for Dysarthria"
   - arXiv:2510.03986
   - **97%检测准确率**

10. **[2025.06] Federated Learning**
    - Zhong et al. "Regularized Federated Learning"
    - arXiv:2506.11069
    - **隐私保护**

### 2024年基础论文

11. **Paraformer** (2022)
    - Gao et al. "Paraformer: Fast and Accurate Parallel Transformer"
    - INTERSPEECH 2022, arXiv:2206.08317

12. **CDSD Dataset** (2024)
    - "CDSD: Chinese Dysarthria Speech Database"
    - INTERSPEECH 2024, arXiv:2310.15930

13. **Prototype-Based Adaptation** (2024)
    - Wang et al. "Enhancing DSR for Unseen Speakers"
    - INTERSPEECH 2024, arXiv:2407.18461

---

## 附录: CDSD专项研究

### A.1 CDSD数据集详情

| 属性 | 值 |
|------|-----|
| 总时长 | 133 小时 |
| 说话人数 | 44 人 |
| 病因 | 脑瘫、中风、帕金森等 |
| 最佳CER | 16.4% (Hybrid CTC/Attention) |
| 人类基线 | 20.45% CER |
| 会议 | INTERSPEECH 2024 |

### A.2 CDSD专项论文 (arXiv:2508.18732)

**Cross-Learning Fine-Tuning Strategy for CDSD**

> **核心发现**:
> - 多说话人联合微调 > 单说话人微调
> - WER降低达 **13.15%**
> - 跨说话人学习增强泛化

**对本项目直接启示**: 
- 不要单独微调每个说话人
- 使用全部44人数据联合训练
- 交叉学习可以防止过拟合

---

**报告结束**

*本报告基于 2025-12-23 最新公开资料整理*
*下次更新: 关注ICASSP 2026和INTERSPEECH 2026*

---

## 10. GitHub开源项目详细分析 (新增)

> 基于 https://github.com/topics/dysarthric-speech 的9个开源项目分析

### 10.1 训练相关项目

#### 🌟🌟��🌟🌟 [jmaczan/asr-dysarthria](https://github.com/jmaczan/asr-dysarthria)
**评级: ⭐⭐⭐⭐⭐ 强烈推荐**

| 属性 | 值 |
|------|-----|
| Stars | 16 |
| 语言 | Jupyter Notebook |
| 许可 | MIT |
| 更新 | 2024 |

**核心功能**:
- 基于 **wav2vec2-large-xls-r-300m** 的构音障碍ASR微调
- 提供完整训练pipeline + 预训练模型
- 支持 **浏览器端部署** (ONNX转换)
- 预训练模型WER: **18.2%** (UASpeech+TORGO)

**预训练模型**:
- HuggingFace: https://huggingface.co/jmaczan/wav2vec2-large-xls-r-300m-dysarthria-big-dataset
- 在线Demo: https://asr-dysarthria-preliminary.pages.dev/

**对本项目的价值**:
1. ✅ **Web应用参考**: 提供了浏览器端ASR的完整实现
2. ✅ **训练参考**: wav2vec2微调流程可借鉴
3. ✅ **预训练权重**: 可用于英语构音障碍迁移学习

**可借鉴代码**:
```bash
# 目录结构
├── web-app/          # 浏览器端ASR (TypeScript)
├── inference/        # 推理脚本 (Python)
├── to_onnx/          # ONNX转换脚本
└── training/         # 训练notebook
```

---

#### ��🌟🌟🌟🌟 [th-nuernberg/multitask-dysarthric-speech-analysis](https://github.com/th-nuernberg/multitask-dysarthric-speech-analysis)
**评级: ⭐⭐⭐⭐⭐ 强烈推荐 (ASRU 2025)**

| 属性 | 值 |
|------|-----|
| Stars | 0 (新项目) |
| 语言 | Python |
| 许可 | Apache-2.0 |
| 更新 | 2025年8月 |

**核心功能**:
- **Phi-4-Multimodal** 多模态语言模型微调
- 联合任务: ASR + 感知属性预测 (可懂度、自然度等)
- SAP Challenge官方评估脚本
- 完整的训练/推理pipeline

**技术亮点**:
```python
# 多任务训练命令
accelerate launch train_sap.py \
  --model_name_or_path "microsoft/Phi-4-multimodal-instruct" \
  --tasks asr intelligibility naturalness consonants \
  --weights 0.14 0.14 0.14 0.14
```

**对本项目的价值**:
1. ✅ **多任务学习参考**: ASR + 严重程度评估
2. ✅ **LLM微调范例**: Phi-4在语音任务的应用
3. ✅ **评估脚本**: 官方WER和SemScore计算

---

#### 🌟🌟🌟 [rshahamiri/SpeechVision](https://github.com/rshahamiri/SpeechVision)
**评级: ⭐⭐⭐ 有参考价值**

| 属性 | 值 |
|------|-----|
| Stars | 7 |
| 语言 | Jupyter Notebook |
| 许可 | MIT |
| 更新 | 2024 |

**核心功能**:
- **视觉方法**: 从语音提取视觉特征 (频谱图)
- CNN + 迁移学习
- 使用UASpeech数据集

**对本项目的价值**:
1. ⚠️ **方法较旧**: 不如wav2vec2等自监督方法
2. ✅ **数据增强思路**: 视觉特征可作为补充

---

### 10.2 Web应用相关项目

#### 🌟🌟🌟🌟 [abhinavbammidi1401/Dys_Locate](https://github.com/abhinavbammidi1401/Dys_Locate)
**评级: ⭐⭐⭐⭐ 推荐用于Web应用参考**

| 属性 | 值 |
|------|-----|
| Stars | 0 |
| 语言 | Jupyter + Python |
| 许可 | MIT |
| 更新 | 2024年12月 |

**核心功能**:
- 构音障碍检测 (95%准确率)
- 语音转写
- **Streamlit Web界面** ⭐

**对Web应用开发的价值**:
```python
# Streamlit应用示例
streamlit run app.py

# 功能:
# 1. 上传音频文件
# 2. 实时检测是否为构音障碍语音
# 3. 语音转文字
```

**可直接借鉴**:
- `app.py` - Streamlit界面实现
- `dysarthria_detection_model.h5` - 检测模型
- `proj.ipynb` - 完整训练流程

---

#### 🌟🌟 [thirionjwf/speak4me--app](https://github.com/thirionjwf/speak4me--app)
**评级: ⭐⭐ 可作为移动端参考**

| 属性 | 值 |
|------|-----|
| Stars | 1 |
| 语言 | Dart (Flutter) |
| 许可 | GPL-3.0 |
| 更新 | 2021 |

**核心功能**:
- Flutter跨平台AAC应用
- 支持手机、平板、手表
- 辅助替代沟通 (AAC)

**对Web应用开发的价值**:
1. ⚠️ **代码较旧**: 4年未更新
2. ✅ **产品思路参考**: AAC应用设计理念

---

### 10.3 项目整合建议

#### 立即可用的资源

| 资源 | 用途 | 链接 |
|------|------|------|
| **wav2vec2预训练权重** | 迁移学习 | [HuggingFace](https://huggingface.co/jmaczan/wav2vec2-large-xls-r-300m-dysarthria-big-dataset) |
| **浏览器端ASR** | Web应用参考 | [Demo](https://asr-dysarthria-preliminary.pages.dev/) |
| **Streamlit模板** | Web界面 | [Dys_Locate](https://github.com/abhinavbammidi1401/Dys_Locate) |
| **多任务训练** | 高级训练 | [multitask-dsa](https://github.com/th-nuernberg/multitask-dysarthric-speech-analysis) |

#### 推荐的Web应用技术栈

```
前端选择:
├── 方案A: Streamlit (快速原型) ⭐推荐
│   └── 参考: Dys_Locate/app.py
├── 方案B: Gradio (ML演示专用)
│   └── HuggingFace Spaces集成
└── 方案C: Next.js + WebSocket (生产级)
    └── 参考: jmaczan/asr-dysarthria/web-app

后端选择:
├── FastAPI + FunASR (当前方案)
├── Flask + Whisper
└── WebSocket + 流式推理
```

#### CLEAR-VOX Web应用开发路线图

```
阶段1: Streamlit原型 (1天)
├── 复用Dys_Locate的界面设计
├── 集成训练好的Paraformer模型
└── 基本功能: 上传音频 → ASR → 显示结果

阶段2: 增强功能 (2天)
├── 添加构音障碍检测
├── 添加LLM后处理纠错
└── 添加严重程度评估

阶段3: 生产部署 (3天)
├── ONNX模型转换 (参考asr-dysarthria)
├── 浏览器端推理 (可选)
└── Docker部署
```

---

### 10.4 完整代码参考索引

| 功能 | 最佳参考项目 | 文件路径 |
|------|------------|---------|
| Wav2Vec2微调 | jmaczan/asr-dysarthria | `training/*.ipynb` |
| ONNX转换 | jmaczan/asr-dysarthria | `to_onnx/` |
| 浏览器ASR | jmaczan/asr-dysarthria | `web-app/` |
| Streamlit界面 | Dys_Locate | `app.py` |
| 多任务训练 | multitask-dsa | `train_sap.py` |
| Phi-4微调 | multitask-dsa | `model/` |
| 检测模型 | Dys_Locate | `dysarthria_detection_model.h5` |

---

**注**: 以上分析基于2025年12月的GitHub仓库状态

---

## 11. 补充 GitHub 仓库资源 (2025-12-24 更新)

> 基于 GitHub 全站搜索 "dysarthric speech recognition" 结果，补充更多高质量开源项目

### 11.1 论文复现项目 (强烈推荐)

#### ⭐⭐⭐⭐⭐ [NKU-HLT/PB-DSR](https://github.com/NKU-HLT/PB-DSR)
**INTERSPEECH 2024 官方代码 - 原型基适应方法**

| 属性 | 值 |
|------|-----|
| Stars | 12 |
| 语言 | Python |
| 更新 | 2024年9月 |
| 会议 | INTERSPEECH 2024 |
| 论文 | arXiv:2407.18461 |

**核心贡献**:
- **原型基说话人适应 (Prototype-Based Adaptation)**
- 无需额外微调即可适应新说话人
- 在 UASpeech 上获得 SOTA 结果

**技术亮点**:
```
1. 从训练集说话人中学习原型表示
2. 测试时通过说话人嵌入检索最近原型
3. 使用原型引导解码
```

**直接价值**:
- ✅ 完整训练代码，可直接复现
- ✅ 说话人适应方法可迁移到 Paraformer

---

#### ⭐⭐⭐⭐⭐ [tan90xx/CBA-Whisper](https://github.com/tan90xx/CBA-Whisper)
**课程学习 + AdaLoRA 微调 Whisper**

| 属性 | 值 |
|------|-----|
| Stars | 6 |
| 语言 | Python |
| 更新 | 2024 |
| 方法 | Curriculum Learning + AdaLoRA |

**核心贡献**:
- **课程学习**: 按难度递增顺序训练
- **AdaLoRA**: 自适应低秩适应，参数高效微调
- 基于 OpenAI Whisper 模型

**技术架构**:
```python
# CBA-Whisper 训练流程
1. 按 CER 难度排序训练样本
2. Easy → Medium → Hard 逐步训练
3. AdaLoRA 微调 (仅更新 ~5% 参数)
```

**直接价值**:
- ✅ 课程学习策略可迁移到 FunASR
- ✅ AdaLoRA 方法可减少计算资源需求

---

#### ⭐⭐⭐⭐ [idiap/torgo_asr](https://github.com/idiap/torgo_asr)
**Kaldi 构音障碍 ASR Recipe**

| 属性 | 值 |
|------|-----|
| Stars | 17 |
| 语言 | Shell + Perl (Kaldi) |
| 更新 | 2021 |
| 数据集 | TORGO |

**核心贡献**:
- 完整的 **Kaldi recipe** for TORGO 数据集
- 包含数据准备、特征提取、训练脚本
- 经典基线参考

**直接价值**:
- ✅ Kaldi 数据处理流程可参考
- ⚠️ 传统方法，不如端到端模型

---

### 11.2 数据增强与预处理项目

#### ⭐⭐⭐⭐ [superkailang/VASR](https://github.com/superkailang/VASR)
**语音增强预处理**

| 属性 | 值 |
|------|-----|
| Stars | 3 |
| 语言 | Python |
| 更新 | 2023 |

**核心功能**:
- 语音质量增强预处理
- 噪声消除 + 语速归一化
- 可作为 ASR 前处理模块

---

#### ⭐⭐⭐ [MorenoLaQuatra/GER4Dys](https://github.com/MorenoLaQuatra/GER4Dys)
**INTERSPEECH 2025 - 生成式错误纠正**

| 属性 | 值 |
|------|-----|
| Stars | 4 |
| 语言 | Python |
| 更新 | 2025年5月 |
| 会议 | INTERSPEECH 2025 |

**核心贡献**:
- 使用 LLM 进行 ASR 后处理错误纠正
- 专门针对构音障碍语音设计
- 开源训练/推理代码

**直接价值**:
- ✅ LLM 后处理方案可直接集成
- ✅ 低成本提升方案 (无需重新训练 ASR)

---

### 11.3 评估与工具项目

#### ⭐⭐⭐ [akulepan/LRDMS](https://github.com/akulepan/LRDMS)
**低资源构音障碍多说话人 ASR**

| 属性 | 值 |
|------|-----|
| Stars | 2 |
| 语言 | Python |
| 更新 | 2024 |

**核心功能**:
- 低资源场景下的多说话人 ASR
- 数据增强策略
- 跨说话人泛化

---

#### ⭐⭐ [DISHA-research/Dysarthric-Dataset](https://github.com/DISHA-research/Dysarthric-Dataset)
**印地语构音障碍数据集**

| 属性 | 值 |
|------|-----|
| Stars | 1 |
| 语言 | - |
| 更新 | 2023 |

**核心功能**:
- 印地语 (Hindi) 构音障碍语音数据集
- 30名说话人
- 研究用途免费

---

### 11.4 项目推荐优先级汇总

| 优先级 | 项目 | 理由 | 适用场景 |
|--------|------|------|----------|
| ⭐⭐⭐⭐⭐ | **NKU-HLT/PB-DSR** | INTERSPEECH 2024 官方代码 | 说话人适应 |
| ⭐⭐⭐⭐⭐ | **tan90xx/CBA-Whisper** | 课程学习 + AdaLoRA | 参数高效微调 |
| ⭐⭐⭐⭐⭐ | **jmaczan/asr-dysarthria** | 完整训练+部署 | Web 应用参考 |
| ⭐⭐⭐⭐ | **MorenoLaQuatra/GER4Dys** | LLM 后处理 | 低成本提升 |
| ⭐⭐⭐⭐ | **idiap/torgo_asr** | 经典 Kaldi baseline | 数据处理参考 |
| ⭐⭐⭐⭐ | **multitask-dsa** | 多任务学习 | 高级训练策略 |
| ⭐⭐⭐ | **superkailang/VASR** | 语音增强 | 预处理 |
| ⭐⭐⭐ | **Dys_Locate** | Streamlit 界面 | Web UI |

---

### 11.5 CLEAR-VOX 项目集成建议

#### 立即可用 (本周)

```bash
# 1. 克隆最有价值的仓库
cd /root/CLEAR-VOX-MODEL

# PB-DSR - 说话人适应
git clone https://github.com/NKU-HLT/PB-DSR.git external/PB-DSR

# GER4Dys - LLM 后处理
git clone https://github.com/MorenoLaQuatra/GER4Dys.git external/GER4Dys

# CBA-Whisper - 课程学习
git clone https://github.com/tan90xx/CBA-Whisper.git external/CBA-Whisper
```

#### 技术迁移路线

| 来源 | 技术 | 目标 | 优先级 |
|------|------|------|--------|
| CBA-Whisper | 课程学习 | 训练脚本 | 高 |
| GER4Dys | LLM 纠错 | 后处理模块 | 高 |
| PB-DSR | 原型适应 | 推理阶段 | 中 |
| asr-dysarthria | ONNX 部署 | Web 应用 | 中 |

---

**更新日期**: 2025-12-24  
**更新内容**: 补充 GitHub 仓库资源，增加技术迁移建议
