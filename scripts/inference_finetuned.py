#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微调模型推理测试脚本
用于评测 FunASR 微调后的 Paraformer 模型
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def calculate_cer(reference: str, hypothesis: str) -> float:
    """计算字符错误率 (CER)"""
    ref = list(reference.replace(" ", ""))
    hyp = list(hypothesis.replace(" ", ""))
    
    m, n = len(ref), len(hyp)
    if m == 0:
        return 100.0 if n > 0 else 0.0
    
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
    
    return dp[m][n] / m * 100


def inference_test(
    model_dir: str,
    init_param: str,
    test_jsonl: str,
    output_file: str = None,
    device: str = "cuda:0",
    max_samples: int = None
):
    """运行推理测试"""
    
    from funasr import AutoModel
    
    print(f"Loading finetuned model...")
    print(f"  Model dir: {model_dir}")
    print(f"  Init param: {init_param}")
    
    # 关键：使用原始模型路径 + init_param 指向微调权重
    model = AutoModel(
        model=model_dir,  # 原始预训练模型路径
        init_param=init_param,  # 微调后的权重
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        disable_update=True
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
    error_count = 0
    
    for i, sample in enumerate(tqdm(test_samples, desc="推理中")):
        audio_path = sample['source']
        reference = sample['target']
        
        try:
            res = model.generate(input=audio_path)
            hypothesis = res[0]["text"] if res else ""
            
            cer = calculate_cer(reference, hypothesis)
            total_cer += cer
            
            results.append({
                'key': sample.get('key', ''),
                'reference': reference,
                'hypothesis': hypothesis,
                'cer': round(cer, 2)
            })
                
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            error_count += 1
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
    print(f"测试结果汇总")
    print("=" * 60)
    print(f"测试样本数: {len(test_samples)}")
    print(f"错误样本数: {error_count}")
    print(f"平均 CER: {avg_cer:.2f}%")
    print("=" * 60)
    
    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_samples': len(test_samples),
                    'error_samples': error_count,
                    'average_cer': round(avg_cer, 2)
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存: {output_file}")
    
    return avg_cer, results


def main():
    parser = argparse.ArgumentParser(description='微调模型推理测试')
    parser.add_argument('--model', type=str, 
                       default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                       help='原始预训练模型名称或路径')
    parser.add_argument('--init_param', type=str, required=True,
                       help='微调后的模型权重路径 (model.pt.best)')
    parser.add_argument('--test', type=str, required=True,
                       help='测试数据 JSONL 文件')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大测试样本数')
    
    args = parser.parse_args()
    
    inference_test(
        model_dir=args.model,
        init_param=args.init_param,
        test_jsonl=args.test,
        output_file=args.output,
        device=args.device,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
