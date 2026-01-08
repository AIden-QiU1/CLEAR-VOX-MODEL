# EXP-004: LLM N-best重排

> **状态**: 🔄 计划中  
> **优先级**: P1  
> **依赖**: EXP-003  
> **预计时间**: 3-5天

---

## 假设

使用大语言模型对ASR的N-best候选结果进行语义重排，可以在后处理阶段进一步降低CER 10-20%。

## 方法

1. ASR生成 Top-K (K=5~10) 候选结果
2. 使用LLM对候选进行语义打分
3. 选择语义最合理的结果作为最终输出

### LLM选型
- **GPT-4**: 效果最好，成本高
- **Qwen-72B**: 中文效果好，成本中等
- **Qwen-7B**: 本地部署，成本低

## 配置

```yaml
asr:
  model: outputs/exp003/checkpoint_best  # 最佳增强模型
  beam_size: 10  # 生成10个候选
  
llm_rerank:
  enabled: true
  model: qwen-72b  # or gpt-4, qwen-7b
  
  prompt_template: |
    请从以下构音障碍患者的语音识别候选结果中，选择语义最通顺的一个：
    
    候选结果：
    {candidates}
    
    请直接输出最合理的结果，不要解释。
    
  scoring:
    method: perplexity  # or ranking
    threshold: 0.8
    
  # 本地部署配置（可选）
  local_deployment:
    enabled: false
    engine: vllm
    model_path: /models/qwen-7b
```

## 执行命令

```bash
# Step 1: 生成N-best候选
python scripts/generate_nbest.py \
    --model outputs/exp003/checkpoint_best \
    --test_data data/cdsd/10h/test \
    --beam_size 10 \
    --output results/exp004/nbest/

# Step 2: LLM重排
python scripts/llm_rerank.py \
    --nbest_dir results/exp004/nbest/ \
    --llm qwen-72b \
    --output results/exp004/reranked/

# Step 3: 评估
python scripts/evaluate.py \
    --hypothesis results/exp004/reranked/output.txt \
    --reference data/cdsd/10h/test/text
```

## 消融实验

| 实验ID | LLM模型 | 候选数K | 预期CER |
|--------|---------|---------|---------|
| 004a | 无LLM (Top-1) | 1 | baseline |
| 004b | Qwen-7B | 5 | - |
| 004c | Qwen-72B | 5 | - |
| 004d | GPT-4 | 5 | - |
| 004e | Qwen-72B | 10 | - |

## 预期结果

| 指标 | EXP-003基线 | 目标 | 相对提升 |
|------|-------------|------|----------|
| CER | ~25% | ~22% | 10-15% |

## 成本估算

| LLM | 单次调用 | 测试集(1000条) | 月度预算 |
|-----|----------|----------------|----------|
| GPT-4 | $0.03 | $30 | - |
| Qwen-72B | ¥0.02 | ¥20 | - |
| Qwen-7B (本地) | 免费 | 免费 | GPU成本 |

## 实际结果

（待实验完成后填写）

## 关键发现

（待实验完成后填写）

## 下一步

- [ ] 完成N-best生成
- [ ] 对比不同LLM效果
- [ ] 考虑本地部署方案
- [ ] 进入 EXP-005 个性化适配
