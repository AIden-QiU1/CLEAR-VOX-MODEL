#!/bin/bash
# FunASR Paraformer EXP-003: 10h数据微调 (梯度累积+BF16优化版)

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate funasr

export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"
export CUDNN_BENCHMARK=1

workspace=/root/CLEAR-VOX-MODEL
model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
train_data="${workspace}/data/10h_dataset/train.jsonl"
val_data="${workspace}/data/10h_dataset/val.jsonl"
output_dir="/root/autodl-tmp/exp/paraformer_finetune_10h"

rm -rf ${output_dir}
mkdir -p ${output_dir}

echo "=============================================="
echo "EXP-003: 10h微调 (梯度累积accum_grad=2+BF16)"
echo "=============================================="
echo "配置: batch=16, accum_grad=2, dtype=bfloat16"
echo "每个epoch保存一次checkpoint (约5216个batch)"
echo "预期显存: ~8GB, 时间: +5-10% (可能还快)"
echo "=============================================="

python ${workspace}/funasr/bin/train_ds.py \
    ++model="${model}" \
    ++train_data_set_list="${train_data}" \
    ++valid_data_set_list="${val_data}" \
    ++dataset="AudioDataset" \
    ++dataset_conf.index_ds="IndexDSJsonl" \
    ++dataset_conf.data_split_num=1 \
    ++dataset_conf.batch_sampler="BatchSampler" \
    ++dataset_conf.batch_size=16 \
    ++dataset_conf.sort_size=512 \
    ++dataset_conf.batch_type="example" \
    ++dataset_conf.max_token_length=500 \
    ++dataset_conf.num_workers=8 \
    ++specaug_conf.apply_time_warp=false \
    ++specaug_conf.num_freq_mask=2 \
    ++specaug_conf.freq_mask_width_range="[0,30]" \
    ++specaug_conf.num_time_mask=2 \
    ++specaug_conf.time_mask_width_range="[0,40]" \
    ++train_conf.max_epoch=30 \
    ++train_conf.log_interval=400 \
    ++train_conf.resume=true \
    ++train_conf.accum_grad=2 \
    ++train_conf.use_bf16=true \
    ++train_conf.validate_interval=4000 \
    ++train_conf.save_checkpoint_interval=4000 \
    ++train_conf.keep_nbest_models=5 \
    ++train_conf.avg_nbest_model=5 \
    ++train_conf.use_deepspeed=false \
    ++optim_conf.lr=0.0002 \
    ++output_dir="${output_dir}" \
    2>&1 | tee ${output_dir}/train_optimized.log

echo "训练完成！"
