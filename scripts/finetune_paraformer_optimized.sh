#!/bin/bash
# FunASR Paraformer 构音障碍微调脚本 v2.4 (显存优化 + 缓存清理版)
# 策略: 增强 SpecAugment + PyTorch 内存优化
# 显存优化: batch_type=example, batch_size=24 (适合 24GB GPU)
# 作者: CLEAR-VOX Team
# 日期: 2025-12-24

export CUDA_VISIBLE_DEVICES="0"

# ============ 关键：PyTorch 显存优化配置 ============
# 限制内存碎片，防止 cache 无限增长
export PYTORCH_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
# 启用 cudnn benchmark 优化
export CUDNN_BENCHMARK=1

workspace=/root/CLEAR-VOX-MODEL
model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
train_data="${workspace}/data/1h_dataset/train.jsonl"
val_data="${workspace}/data/1h_dataset/val.jsonl"
output_dir="/root/autodl-tmp/exp/paraformer_finetune_1h_optimized"

mkdir -p ${output_dir}

echo "=============================================="
echo "FunASR Paraformer 构音障碍微调 v2.4 (显存优化)"
echo "=============================================="
echo "训练数据: ${train_data}"
echo "验证数据: ${val_data}"
echo "输出目录: ${output_dir}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "优化策略:"
echo "  1. 增强 SpecAugment (频率/时间遮蔽)"
echo "  2. PyTorch 内存碎片优化 (max_split_size_mb=128)"
echo "  3. 可扩展内存段 (expandable_segments=True)"
echo ""
echo "显存优化:"
echo "  - batch_type: example (按样本数计数)"
echo "  - batch_size: 24"
echo "  - max_token_length: 1800 (限制超长音频)"
echo ""
echo "模型保存策略:"
echo "  - 保留效果最好的 5 个模型"
echo "  - 自动生成 5 模型融合版本"
echo "=============================================="

if [ ! -f "${train_data}" ]; then
    echo "错误: 训练数据不存在"
    exit 1
fi

echo "开始训练..."

torchrun --nproc_per_node=1 ${workspace}/funasr/bin/train_ds.py \
    ++model="${model}" \
    ++train_data_set_list="${train_data}" \
    ++valid_data_set_list="${val_data}" \
    ++dataset="AudioDataset" \
    ++dataset_conf.index_ds="IndexDSJsonl" \
    ++dataset_conf.data_split_num=1 \
    ++dataset_conf.batch_sampler="BatchSampler" \
    ++dataset_conf.batch_size=20 \
    ++dataset_conf.sort_size=1024 \
    ++dataset_conf.batch_type="example" \
    ++dataset_conf.max_token_length=1800 \
    ++dataset_conf.num_workers=2 \
    ++specaug_conf.apply_time_warp=false \
    ++specaug_conf.num_freq_mask=2 \
    ++specaug_conf.freq_mask_width_range="[0,30]" \
    ++specaug_conf.num_time_mask=2 \
    ++specaug_conf.time_mask_width_range="[0,40]" \
    ++train_conf.max_epoch=50 \
    ++train_conf.log_interval=100 \
    ++train_conf.resume=true \
    ++train_conf.validate_interval=2000 \
    ++train_conf.save_checkpoint_interval=2000 \
    ++train_conf.keep_nbest_models=5 \
    ++train_conf.avg_nbest_model=5 \
    ++train_conf.use_deepspeed=false \
    ++optim_conf.lr=0.0002 \
    ++output_dir="${output_dir}" \
    2>&1 | tee ${output_dir}/train.log

echo ""
echo "=============================================="
echo "训练完成！"
echo "=============================================="
echo ""
echo "保存的模型文件:"
ls -la ${output_dir}/*.pt 2>/dev/null || echo "尚无模型文件"
echo ""
echo "融合模型: ${output_dir}/model.pt.avg5 (自动生成)"
echo "最佳单模型: ${output_dir}/model.pt"
echo "=============================================="
