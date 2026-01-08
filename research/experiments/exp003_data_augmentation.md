# EXP-003: 数据增强实验

> **状态**: 🔄 计划中  
> **优先级**: P1  
> **依赖**: EXP-002  
> **预计时间**: 5-7天

---

## 假设

使用TTS合成的模拟构音障碍语音进行数据增强，可以显著提升模型在真实构音障碍数据上的表现。

## 方法

### 方案A: SpecAugment症状掩码
针对构音障碍特征设计定制化SpecAugment掩码：
- **口吃掩码**: 时间轴重复小段
- **鼻音化掩码**: 高频增强+低频衰减
- **气息音掩码**: 添加白噪声

### 方案B: TTS合成增强
使用F5-TTS/CosyVoice生成模拟语音：
- 从CDSD提取说话人特征
- 合成新的文本内容
- 添加症状变换

## 配置

```yaml
augmentation:
  # SpecAugment症状掩码
  specaugment:
    stutter_mask:
      enabled: true
      repeat_prob: 0.3
      repeat_count: [2, 4]
    
    hypernasal_mask:
      enabled: true
      high_freq_boost: 0.3
      low_freq_cut: 0.2
    
    breathiness_mask:
      enabled: true
      noise_level: 0.1

  # TTS增强
  tts_augment:
    engine: f5-tts  # or cosyvoice
    speakers: ["spk01", "spk02"]  # CDSD说话人
    texts_source: aishell  # 文本来源
    num_samples: 5000  # 生成样本数

training:
  # 继承 EXP-002 配置
  base_config: exp002_lora_finetune
  
  data:
    train: 
      - data/cdsd/10h/train  # 原始数据
      - data/augmented/      # 增强数据
```

## 执行命令

```bash
# Step 1: 生成增强数据
python scripts/generate_augmented_data.py \
    --method specaugment \
    --output data/augmented/

# Step 2: 训练（使用增强数据）
bash scripts/finetune_with_augmentation.sh
```

## 消融实验

| 实验ID | 增强方法 | 增强比例 | 预期CER |
|--------|----------|----------|---------|
| 003a | 无增强 (EXP-002) | 0% | baseline |
| 003b | SpecAugment-症状 | +50% | - |
| 003c | TTS合成 | +100% | - |
| 003d | SpecAug + TTS | +150% | - |

## 预期结果

| 指标 | EXP-002基线 | 目标 | 相对提升 |
|------|-------------|------|----------|
| CER | ~30% | ~25% | 15-20% |

## 实际结果

（待实验完成后填写）

## 关键发现

（待实验完成后填写）

## 下一步

- [ ] 完成增强数据生成
- [ ] 对比不同增强策略
- [ ] 进入 EXP-004 LLM重排
