# EXP-002: LoRA微调

> **状态**: 🔄 计划中  
> **优先级**: P0  
> **依赖**: EXP-001  
> **预计时间**: 3-5天

---

## 假设

使用LoRA对Paraformer-large的Encoder进行微调，同时冻结Decoder，可以在CDSD数据上获得显著的CER提升。

## 方法

1. 在Encoder的self-attention层添加LoRA适配器
2. 冻结Decoder所有参数
3. 使用CDSD 10h数据训练
4. 对比不同LoRA rank的效果

## 配置

```yaml
model:
  name: paraformer-large
  source: modelscope
  
lora:
  enabled: true
  rank: 8
  alpha: 16
  target_modules:
    - encoder.encoders.*.self_attn.linear_q
    - encoder.encoders.*.self_attn.linear_v
  dropout: 0.1

training:
  epochs: 30
  batch_size: 8
  gradient_accumulation: 4
  learning_rate: 1e-4
  warmup_steps: 500
  
  freeze:
    decoder: true
    encoder: false
    
  optimizer: AdamW
  scheduler: cosine
  
data:
  train: data/cdsd/10h/train
  val: data/cdsd/10h/val
  test: data/cdsd/10h/test

device:
  gpu: 0
  mixed_precision: fp16
```

## 执行命令

```bash
# 训练
bash scripts/finetune_paraformer_10h_optimized.sh

# 评估
python scripts/inference_finetuned.py \
    --model_path outputs/exp002/checkpoint_best \
    --test_data data/cdsd/10h/test/wav.scp
```

## 消融实验

| 实验ID | LoRA rank | 冻结策略 | 预期CER |
|--------|-----------|----------|---------|
| 002a | 4 | 冻结Decoder | - |
| 002b | 8 | 冻结Decoder | - |
| 002c | 16 | 冻结Decoder | - |
| 002d | 8 | 全量训练 | - |

## 预期结果

| 指标 | 基线 | 目标 | 提升 |
|------|------|------|------|
| CER | ~50% | ~30% | 40% rel. |

## 实际结果

| 实验ID | CER | 训练时间 | 显存 |
|--------|-----|----------|------|
| 002a | - | - | - |
| 002b | - | - | - |

## 分析

（待实验完成后填写）

## 下一步

- [ ] 完成LoRA微调
- [ ] 选择最佳rank
- [ ] 进入 EXP-003 数据增强
