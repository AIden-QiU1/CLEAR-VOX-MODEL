# 📊 数据集 (Datasets)

> 构音障碍语音识别领域的主要数据集汇总

---

## 📋 数据集总览

| 数据集 | 语言 | 说话人数 | 词汇量 | 总时长 | 公开性 | 推荐度 |
|--------|------|----------|--------|--------|--------|--------|
| **CDSD** | 中文 | 多说话人 | 开放 | ~10h+ | 需申请 | ⭐⭐⭐⭐⭐ |
| **UASpeech** | 英文 | 19人 | 765词 | ~10h | 公开 | ⭐⭐⭐⭐⭐ |
| **TORGO** | 英文 | 8人 | ~500词 | ~5h | 公开 | ⭐⭐⭐⭐ |
| **Nemours** | 英文 | 11人 | - | ~3h | 需申请 | ⭐⭐⭐ |

---

## 📖 数据集详解

### 1. CDSD (Chinese Dysarthria Speech Database)
**Interspeech 2023** | [论文链接](https://arxiv.org/pdf/2310.15930)

#### 基本信息
| 属性 | 值 |
|------|---|
| 语言 | 普通话 |
| 说话人 | 多名构音障碍患者 |
| 文本来源 | AISHELL-1语料 + 小学/中学课文 |
| 严重度标注 | 有 |

#### 关键发现
> 1. 普通话语音识别模型**难以泛化**到构音障碍语音
> 2. 个性化微调可以**显著提升**识别准确率
> 3. 不同构音障碍个体之间**难以泛化**

#### 数据划分建议
```python
# 按说话人划分，确保测试说话人不出现在训练集
train_speakers = ["spk01", "spk02", "spk03", ...]
val_speakers = ["spk10", "spk11"]
test_speakers = ["spk12", "spk13", "spk14"]

# 不建议按句子随机划分（会导致说话人信息泄露）
```

#### 使用注意
- 需要向作者申请获取
- 包含儿童说话人（语料选自小学课文）
- 严重度分布不均

---

### 2. UASpeech
**经典基准数据集**

#### 基本信息
| 属性 | 值 |
|------|---|
| 语言 | 英语 |
| 说话人 | 19名构音障碍者 + 13名正常对照 |
| 词汇量 | 765个独立词汇 |
| 严重度 | 4级 (Very Low / Low / Mid / High) |

#### 严重度分布
| 级别 | 可懂度 | 说话人数 |
|------|--------|----------|
| Very Low | 2-15% | 4人 |
| Low | 15-28% | 5人 |
| Mid | 43-58% | 5人 |
| High | 62-95% | 5人 |

#### 常用划分
```
Block 1, 2: 训练集
Block 3: 测试集
```

#### 使用建议
- 英文基准测试首选
- 词汇量有限，适合孤立词识别
- 说话人差异大，需要个性化

---

### 3. TORGO
**同时包含声学和发音特征**

#### 基本信息
| 属性 | 值 |
|------|---|
| 语言 | 英语 |
| 说话人 | 8名构音障碍者 + 7名正常对照 |
| 特点 | 同时录制了EMA（电磁发音测量）数据 |

#### 独特价值
- 可用于研究发音器官运动与声学的关系
- 适合虚拟发音特征（Articulatory Feature）研究

---

### 4. 辅助数据集

#### 正常语音数据（预训练用）

| 数据集 | 语言 | 规模 | 用途 |
|--------|------|------|------|
| **AISHELL-1** | 中文 | 178h | 中文预训练 |
| **WenetSpeech** | 中文 | 10000h+ | 大规模预训练 |
| **LibriSpeech** | 英文 | 1000h | 英文预训练 |
| **VCTK** | 英文 | 44h/109人 | 多说话人TTS |
| **LibriTTS** | 英文 | 585h | TTS训练 |

---

## 🔬 数据处理最佳实践

### 1. 数据划分原则
```
✅ 按说话人划分（防止信息泄露）
✅ 保持严重度分布平衡
❌ 不要按句子随机划分
❌ 不要让同一说话人出现在训练和测试集
```

### 2. 数据增强策略
参考 [数据增强主题](../data_augmentation/README.md)

### 3. 质量控制
```python
def quality_check(audio, transcript):
    # 1. 检查音频时长
    if len(audio) < 0.5 * sr:  # 少于0.5秒
        return False
        
    # 2. 检查信噪比
    if compute_snr(audio) < 10:  # SNR < 10dB
        return False
        
    # 3. 检查标注匹配
    if not validate_alignment(audio, transcript):
        return False
        
    return True
```

---

## �� CDSD 与 UASpeech 对比

| 维度 | CDSD | UASpeech |
|------|------|----------|
| 语言 | 中文 | 英文 |
| 任务 | 连续语音 | 孤立词 |
| 声调 | 有（四声） | 无 |
| 主要挑战 | 声调损伤、方言 | 严重度差异大 |
| 适用场景 | 中文产品研发 | 英文基准对比 |

---

## 💡 研究洞察

### 数据稀缺问题
> 构音障碍数据天然稀缺（患者少、录制难、隐私敏感）

#### 解决方案
1. **数据增强**: TTS/VC合成
2. **迁移学习**: 从正常语音预训练迁移
3. **个性化**: 少量数据快速适配
4. **跨语言**: 利用其他语言数据

### 个体差异问题
> 不同患者的错误模式差异极大

#### 解决方案
1. **两阶段训练**: 通用 → 个性化
2. **MoE路由**: 按特征自动选择
3. **原型学习**: 学习患者聚类中心

---

## 📚 相关论文

1. [CDSD: Chinese Dysarthria Speech Database](https://arxiv.org/pdf/2310.15930) - Interspeech 2023
2. [UASpeech: A Speech Database for Dysarthric Speakers](https://www.isca-archive.org/interspeech_2008/kim08c_interspeech.pdf) - Interspeech 2008
3. [TORGO Database](https://www.cs.toronto.edu/~frank/torgo.html)

---

## 🔗 数据获取链接

| 数据集 | 获取方式 |
|--------|----------|
| CDSD | 联系论文作者申请 |
| UASpeech | [官方网站](http://www.isle.illinois.edu/sst/data/UASpeech/) |
| TORGO | [官方网站](https://www.cs.toronto.edu/~frank/torgo.html) |
| AISHELL-1 | [OpenSLR](https://www.openslr.org/33/) |
| LibriSpeech | [OpenSLR](https://www.openslr.org/12/) |
