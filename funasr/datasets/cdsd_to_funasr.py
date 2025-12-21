#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare CDSD when audio/ and text/ are separated.

Inputs:
  CDSD_ROOT/audio/**/*.wav
  CDSD_ROOT/text/**/*.txt

Outputs:
  out_dir/transcripts.tsv (wav_path \t text \t spk_id)
  out_dir/{train,dev}_wav.scp
  out_dir/{train,dev}_text.txt
  out_dir/{train,dev}_utt2spk
  optional out_dir/{train,dev}.jsonl

Pairing:
  - default: match by relative path + stem:
      audio/a/b/001.wav <-> text/a/b/001.txt
  - fallback: match by stem only (requires global uniqueness)

Speaker ID:
  - try to extract from audio path by regex (default)
  - else spk_unk

Usage:
  python prepare_cdsd_from_audiotext.py \
    --cdsd_root /abs/path/CDSD_ROOT \
    --out_dir data/cdsd/list \
    --dev_ratio 0.05 \
    --seed 42 \
    --make_jsonl \
    --spk_regex "(speaker_\\d+|spk\\d+|S\\d+)"
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

def read_text_file(txt_path: Path) -> str:
    # 读取整文件并做简单清洗：去 BOM / 合并空白
    s = txt_path.read_text(encoding="utf-8", errors="ignore")
    s = s.replace("\ufeff", "").strip()
    # 将多行合成一行（如果你的标注一行一句，这样也兼容）
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_index(files: List[Path], root: Path) -> Dict[str, Path]:
    """
    Build index by relative path without suffix:
      rel_no_suffix = str(relpath).replace(suffix,'')
    Example:
      audio/a/b/001.wav -> key "a/b/001"
    """
    idx = {}
    for p in files:
        rel = p.relative_to(root).as_posix()
        rel_no_suffix = str(Path(rel).with_suffix("")).replace("\\", "/")
        idx[rel_no_suffix] = p
    return idx

def build_stem_index(files: List[Path]) -> Dict[str, List[Path]]:
    stem2paths = defaultdict(list)
    for p in files:
        stem2paths[p.stem].append(p)
    return stem2paths

def extract_spk_id(audio_path: Path, spk_regex: str) -> str:
    m = re.search(spk_regex, audio_path.as_posix())
    if not m:
        return "spk_unk"
    return m.group(1)

def speaker_split(items: List[Tuple[str, str, str]], dev_ratio: float, seed: int):
    spk2items = defaultdict(list)
    for wav, text, spk in items:
        spk2items[spk].append((wav, text, spk))

    spks = list(spk2items.keys())
    random.Random(seed).shuffle(spks)

    # 若全是 spk_unk，就退化为随机按条目拆分（但不推荐）
    if len(spks) == 1 and spks[0] == "spk_unk":
        random.Random(seed).shuffle(items)
        n_dev = max(1, int(len(items) * dev_ratio))
        return items[n_dev:], items[:n_dev]

    dev_spk_count = max(1, int(len(spks) * dev_ratio))
    dev_spks = set(spks[:dev_spk_count])

    train, dev = [], []
    for spk, lst in spk2items.items():
        (dev if spk in dev_spks else train).extend(lst)
    return train, dev

def make_utt_id(wav_path: str, spk: str, idx: int) -> str:
    stem = Path(wav_path).stem
    return f"{spk}_{stem}_{idx:06d}"

def write_kaldi_lists(items, prefix: str, out_dir: Path):
    wav_scp = out_dir / f"{prefix}_wav.scp"
    text_txt = out_dir / f"{prefix}_text.txt"
    utt2spk = out_dir / f"{prefix}_utt2spk"

    with wav_scp.open("w", encoding="utf-8") as fwav, \
         text_txt.open("w", encoding="utf-8") as ftxt, \
         utt2spk.open("w", encoding="utf-8") as fspk:
        for i, (wav, text, spk) in enumerate(items):
            utt = make_utt_id(wav, spk, i)
            fwav.write(f"{utt}\t{wav}\n")
            ftxt.write(f"{utt}\t{text}\n")
            fspk.write(f"{utt}\t{spk}\n")

def write_jsonl(items, prefix: str, out_dir: Path):
    out_path = out_dir / f"{prefix}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, (wav, text, spk) in enumerate(items):
            key = make_utt_id(wav, spk, i)
            f.write(json.dumps({"key": key, "source": wav, "target": text}, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdsd_root", required=True)
    ap.add_argument("--audio_dir", default="audio", help="relative to cdsd_root")
    ap.add_argument("--text_dir", default="text", help="relative to cdsd_root")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dev_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--make_jsonl", action="store_true")
    ap.add_argument("--spk_regex", default=r"(speaker_\d+|spk\d+|S\d+)",
                    help="regex to extract speaker id from audio path")
    ap.add_argument("--match_mode", choices=["relpath", "stem"], default="relpath",
                    help="relpath: match by relative path+stem; stem: match by stem only")
    args = ap.parse_args()

    cdsd_root = Path(args.cdsd_root).expanduser().resolve()
    audio_root = cdsd_root / args.audio_dir
    text_root = cdsd_root / args.text_dir
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(audio_root.rglob("*.wav"))
    txt_files = sorted(text_root.rglob("*.txt"))

    if not wav_files:
        raise RuntimeError(f"No wav found under: {audio_root}")
    if not txt_files:
        raise RuntimeError(f"No txt found under: {text_root}")

    items = []

    if args.match_mode == "relpath":
        wav_idx = build_index(wav_files, audio_root)
        txt_idx = build_index(txt_files, text_root)

        common = sorted(set(wav_idx.keys()) & set(txt_idx.keys()))
        missing_txt = sorted(set(wav_idx.keys()) - set(txt_idx.keys()))
        missing_wav = sorted(set(txt_idx.keys()) - set(wav_idx.keys()))

        if len(common) == 0:
            raise RuntimeError(
                "No matches by relpath. Try --match_mode stem or check directory structure."
            )

        for k in common:
            wav_p = wav_idx[k]
            txt_p = txt_idx[k]
            text = read_text_file(txt_p)
            if not text:
                continue
            spk = extract_spk_id(wav_p, args.spk_regex)
            items.append((str(wav_p.resolve()), text, spk))

        print(f"[relpath] matched={len(common)} | wav_only={len(missing_txt)} | txt_only={len(missing_wav)}")
        if missing_txt[:5]:
            print("Example wav_only keys:", missing_txt[:5])
        if missing_wav[:5]:
            print("Example txt_only keys:", missing_wav[:5])

    else:  # stem
        wav_stem = build_stem_index(wav_files)
        txt_stem = build_stem_index(txt_files)
        common_stems = sorted(set(wav_stem.keys()) & set(txt_stem.keys()))
        if len(common_stems) == 0:
            raise RuntimeError("No matches by stem. Check your files.")
        # stem-only 要求唯一，否则会歧义
        ambiguous = [s for s in common_stems if len(wav_stem[s]) != 1 or len(txt_stem[s]) != 1]
        if ambiguous:
            raise RuntimeError(
                f"Stem matching ambiguous for {len(ambiguous)} stems (not unique). "
                f"Example: {ambiguous[:5]}. Use relpath mode or rename files."
            )

        for s in common_stems:
            wav_p = wav_stem[s][0]
            txt_p = txt_stem[s][0]
            text = read_text_file(txt_p)
            if not text:
                continue
            spk = extract_spk_id(wav_p, args.spk_regex)
            items.append((str(wav_p.resolve()), text, spk))

        print(f"[stem] matched_stems={len(common_stems)}")

    # 写 transcripts.tsv
    tsv_path = out_dir / "transcripts.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        for wav, text, spk in items:
            f.write(f"{wav}\t{text}\t{spk}\n")
    print("Wrote:", tsv_path, "items=", len(items))

    # split
    train, dev = speaker_split(items, dev_ratio=args.dev_ratio, seed=args.seed)
    print(f"Split: train={len(train)} dev={len(dev)} | unique_spk={len(set([x[2] for x in items]))}")

    # write kaldi-style lists
    write_kaldi_lists(train, "train", out_dir)
    write_kaldi_lists(dev, "dev", out_dir)

    if args.make_jsonl:
        write_jsonl(train, "train", out_dir)
        write_jsonl(dev, "dev", out_dir)

    print("Done. Output dir:", out_dir)

if __name__ == "__main__":
    main()
