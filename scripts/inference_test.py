#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理测试脚本 - 用于评测微调后的模型
作者：GitHub Copilot
日期：2025-12-23
"""

import os
import json
import argparse
from pathlib import Path
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def calculate_cer(reference: str, hypothesis: str) -> float:
    """计算字符错误率 (CER)"""
    # 删除空格进行中文CER计算
    ref = list(reference.replace(" ", ""))
    hyp = list(hypothesis.replace(" ", ""))
    
    # 动态规划计算编辑距离
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n] / max(len(ref), 1) * 100


def inference_test(
    model_dir: str,
    test_jsonl: str,
    output_file: str = None,
    device: str = "cuda:0",
    batch_size: int = 1,
    max_samples: int = None
):
    """运行推理测试"""
    
    print(f"Loading model from: {model_dir}")
    
    # 加载模型
    model = AutoModel(
        model=model_dir,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
    )
    
    # 读取测试数据
    print(f"Loading test data from: {test_jsonl}")
    test_samples = []
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
    
    if max_samples:
        test_samples = test_samples[:max_samples]
    
    print(f"Total samples: {len(test_samples)}")
    
    # 运行推理
    results = []
    total_cer = 0
    
    for i, sample in enumerate(test_samples):
        audio_path = sample['source']
        reference = sample['target']
        
        try:
            # 推理
            res = model.generate(input=[audio_path], cache={}, batch_size_s=0)
            hypothesis = res[0]["text"] if res else ""
            
            # 计算CER
            cer = calculate_cer(reference, hypothesis)
            total_cer += cer
            
            results.append({
                'key': sample.get('key', ''),
                'reference': reference,
                'hypothesis': hypothesis,
                'cer': cer
            })
            
            if (i + 1) % 100 == 0:
                avg_cer = total_cer / (i + 1)
                print(f"Processed {i+1}/{len(test_samples)}, Average CER: {avg_cer:.2f}%")
                
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            results.append({
                'key': sample.get('key', ''),
                'reference': reference,
                'hypothesis': '[ERROR]',
                'cer': 100.0
            })
            total_cer += 100.0
    
    # 计算总体指标
    avg_cer = total_cer / len(test_samples) if test_samples else 0
    
    print("\n" + "=" * 60)
    print(f"Test Results Summary")
    print("=" * 60)
    print(f"Total samples: {len(test_samples)}")
    print(f"Average CER: {avg_cer:.2f}%")
    
    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_samples': len(test_samples),
                    'average_cer': avg_cer
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {output_file}")
    
    return avg_cer, results


def main():
    parser = argparse.ArgumentParser(description='ASR Inference Test')
    parser.add_argument('--model', type=str, required=True, help='Model directory')
    parser.add_argument('--test', type=str, required=True, help='Test JSONL file')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples')
    
    args = parser.parse_args()
    
    inference_test(
        model_dir=args.model,
        test_jsonl=args.test,
        output_file=args.output,
        device=args.device,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
