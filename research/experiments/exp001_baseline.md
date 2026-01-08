# EXP-001: 基线测试

> **状态**: 🔄 计划中  
> **优先级**: P0  
> **预计时间**: 2-3天

---

## 假设

直接使用Paraformer-large在CDSD测试集上推理，建立基线CER。

## 方法

1. 下载预训练Paraformer-large模型
2. 准备CDSD测试集（标准格式）
3. 运行推理并计算CER

## 配置

```yaml
model:
  name: paraformer-large
  source: modelscope
  
data:
  test_set: data/cdsd/10h/test
  format: kaldi
  
inference:
  batch_size: 16
  beam_size: 5
  device: cuda:0
```

## 执行命令

```bash
# 下载模型
python -c "from funasr import AutoModel; AutoModel(model='paraformer-large')"

# 运行推理
python scripts/inference_test.py \
    --model paraformer-large \
    --test_data data/cdsd/10h/test/wav.scp \
    --output results/exp001/
```

## 预期结果

| 指标 | 预期范围 | 说明 |
|------|----------|------|
| CER | 40-60% | 未微调，预期较高 |
| RTF | < 0.1 | 实时性良好 |

## 实际结果

| 指标 | 数值 | 备注 |
|------|------|------|
| CER | - | 待填写 |
| RTF | - | 待填写 |

## 分析

（待实验完成后填写）

## 下一步

- [ ] 完成基线测试
- [ ] 分析错误模式
- [ ] 进入 EXP-002 LoRA微调
