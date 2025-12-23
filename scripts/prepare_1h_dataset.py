#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本：将1h数据集转换为FunASR训练格式
作者：GitHub Copilot
日期：2025-12-23
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetPreparator:
    """1h数据集预处理器"""
    
    def __init__(
        self,
        audio_root: str = "/root/autodl-tmp/1h/Audio",
        text_root: str = "/root/autodl-tmp/1h/Text",
        output_dir: str = "/root/CLEAR-VOX-MODEL/data/1h_dataset",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        self.audio_root = Path(audio_root)
        self.text_root = Path(text_root)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        random.seed(seed)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_transcripts(self, speaker_id: str) -> Dict[str, str]:
        """加载某个说话人的转录文本"""
        label_file = self.text_root / f"{speaker_id}_label.txt"
        transcripts = {}
        
        if not label_file.exists():
            logger.warning(f"Label file not found: {label_file}")
            return transcripts
            
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    transcripts[utt_id] = text.strip()
                    
        return transcripts
    
    def collect_all_data(self) -> List[Dict]:
        """收集所有音频文件和对应的文本"""
        all_data = []
        
        # 遍历所有说话人目录
        speaker_dirs = sorted([d for d in self.audio_root.iterdir() if d.is_dir()])
        logger.info(f"Found {len(speaker_dirs)} speakers")
        
        for speaker_dir in speaker_dirs:
            speaker_id = speaker_dir.name
            logger.info(f"Processing speaker: {speaker_id}")
            
            # 加载该说话人的转录文本
            transcripts = self.load_transcripts(speaker_id)
            if not transcripts:
                logger.warning(f"No transcripts found for speaker {speaker_id}, skipping")
                continue
            
            # 遍历该说话人的所有音频文件
            audio_files = sorted(speaker_dir.glob("*.wav"))
            matched_count = 0
            
            for audio_file in audio_files:
                # 从文件名提取 utterance ID（去掉.wav后缀）
                utt_id = audio_file.stem
                
                # 查找对应的文本
                if utt_id in transcripts:
                    text = transcripts[utt_id]
                    all_data.append({
                        'utt_id': utt_id,
                        'audio_path': str(audio_file.absolute()),
                        'text': text,
                        'speaker': speaker_id
                    })
                    matched_count += 1
                else:
                    logger.debug(f"No transcript found for {utt_id}")
            
            logger.info(f"Speaker {speaker_id}: {matched_count}/{len(audio_files)} files matched")
        
        logger.info(f"Total collected: {len(all_data)} utterances")
        return all_data
    
    def split_data_by_speaker(self, all_data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """按说话人划分数据集（保证同一说话人不会跨集合）"""
        # 按说话人分组
        speaker_data = {}
        for item in all_data:
            speaker = item['speaker']
            if speaker not in speaker_data:
                speaker_data[speaker] = []
            speaker_data[speaker].append(item)
        
        # 随机打乱说话人列表
        speakers = list(speaker_data.keys())
        random.shuffle(speakers)
        
        # 计算划分点
        total_speakers = len(speakers)
        train_count = int(total_speakers * self.train_ratio)
        val_count = int(total_speakers * self.val_ratio)
        
        train_speakers = speakers[:train_count]
        val_speakers = speakers[train_count:train_count + val_count]
        test_speakers = speakers[train_count + val_count:]
        
        # 收集各集合的数据
        train_data = []
        val_data = []
        test_data = []
        
        for speaker in train_speakers:
            train_data.extend(speaker_data[speaker])
        for speaker in val_speakers:
            val_data.extend(speaker_data[speaker])
        for speaker in test_speakers:
            test_data.extend(speaker_data[speaker])
        
        # 在集合内部打乱
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        logger.info(f"Split summary:")
        logger.info(f"  Train: {len(train_speakers)} speakers, {len(train_data)} utterances")
        logger.info(f"  Val:   {len(val_speakers)} speakers, {len(val_data)} utterances")
        logger.info(f"  Test:  {len(test_speakers)} speakers, {len(test_data)} utterances")
        
        return train_data, val_data, test_data
    
    def write_kaldi_format(self, data: List[Dict], split_name: str):
        """写入Kaldi格式文件（wav.scp, text, utt2spk）"""
        wav_scp_file = self.output_dir / f"{split_name}_wav.scp"
        text_file = self.output_dir / f"{split_name}_text.txt"
        utt2spk_file = self.output_dir / f"{split_name}_utt2spk"
        
        with open(wav_scp_file, 'w', encoding='utf-8') as f_wav, \
             open(text_file, 'w', encoding='utf-8') as f_text, \
             open(utt2spk_file, 'w', encoding='utf-8') as f_utt2spk:
            
            for item in data:
                utt_id = item['utt_id']
                audio_path = item['audio_path']
                text = item['text']
                speaker = item['speaker']
                
                f_wav.write(f"{utt_id} {audio_path}\n")
                f_text.write(f"{utt_id} {text}\n")
                f_utt2spk.write(f"{utt_id} {speaker}\n")
        
        logger.info(f"Written Kaldi format files for {split_name}")
    
    def write_jsonl_format(self, data: List[Dict], split_name: str):
        """写入JSONL格式文件（FunASR新版本推荐格式）"""
        jsonl_file = self.output_dir / f"{split_name}.jsonl"
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in data:
                json_obj = {
                    'key': item['utt_id'],
                    'source': item['audio_path'],
                    'target': item['text'],
                    'speaker': item['speaker']
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        
        logger.info(f"Written JSONL file for {split_name}: {jsonl_file}")
    
    def write_statistics(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """写入统计信息"""
        stats_file = self.output_dir / "data_statistics.txt"
        
        def get_speaker_count(data):
            return len(set(item['speaker'] for item in data))
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("1h Dataset Statistics\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Train Set:\n")
            f.write(f"  Speakers: {get_speaker_count(train_data)}\n")
            f.write(f"  Utterances: {len(train_data)}\n\n")
            
            f.write(f"Validation Set:\n")
            f.write(f"  Speakers: {get_speaker_count(val_data)}\n")
            f.write(f"  Utterances: {len(val_data)}\n\n")
            
            f.write(f"Test Set:\n")
            f.write(f"  Speakers: {get_speaker_count(test_data)}\n")
            f.write(f"  Utterances: {len(test_data)}\n\n")
            
            f.write(f"Total:\n")
            f.write(f"  Speakers: {get_speaker_count(train_data + val_data + test_data)}\n")
            f.write(f"  Utterances: {len(train_data) + len(val_data) + len(test_data)}\n")
        
        logger.info(f"Written statistics to {stats_file}")
    
    def prepare(self):
        """执行完整的数据准备流程"""
        logger.info("Starting data preparation...")
        
        # 1. 收集所有数据
        all_data = self.collect_all_data()
        if not all_data:
            logger.error("No data collected. Please check your data paths.")
            return
        
        # 2. 划分数据集
        train_data, val_data, test_data = self.split_data_by_speaker(all_data)
        
        # 3. 写入Kaldi格式
        self.write_kaldi_format(train_data, "train")
        self.write_kaldi_format(val_data, "val")
        self.write_kaldi_format(test_data, "test")
        
        # 4. 写入JSONL格式
        self.write_jsonl_format(train_data, "train")
        self.write_jsonl_format(val_data, "val")
        self.write_jsonl_format(test_data, "test")
        
        # 5. 写入统计信息
        self.write_statistics(train_data, val_data, test_data)
        
        logger.info(f"Data preparation completed! Output directory: {self.output_dir}")
        logger.info(f"Please check the generated files and statistics.")


def main():
    """主函数"""
    preparator = DatasetPreparator(
        audio_root="/root/autodl-tmp/1h/Audio",
        text_root="/root/autodl-tmp/1h/Text",
        output_dir="/root/CLEAR-VOX-MODEL/data/1h_dataset",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    
    preparator.prepare()


if __name__ == "__main__":
    main()
