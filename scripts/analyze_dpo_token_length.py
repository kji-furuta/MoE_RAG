#!/usr/bin/env python3
"""
DPOデータセットのトークン長分布を分析するスクリプト

使用方法:
    python scripts/analyze_dpo_token_length.py data/dpo/preference_dataset.jsonl
"""

import sys
import json
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer
import numpy as np

def load_dataset(filepath: str) -> List[Dict]:
    """JSONLファイルからデータセットを読み込む"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def analyze_token_lengths(dataset: List[Dict], tokenizer) -> Dict:
    """トークン長の統計を計算"""
    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []
    total_lengths = []

    for item in dataset:
        # Tokenize each field
        prompt_tokens = tokenizer.encode(item.get('prompt', ''), add_special_tokens=False)
        chosen_tokens = tokenizer.encode(item.get('chosen', ''), add_special_tokens=False)
        rejected_tokens = tokenizer.encode(item.get('rejected', ''), add_special_tokens=False)

        prompt_lengths.append(len(prompt_tokens))
        chosen_lengths.append(len(chosen_tokens))
        rejected_lengths.append(len(rejected_tokens))

        # Total = prompt + max(chosen, rejected) for worst-case memory
        total_lengths.append(len(prompt_tokens) + max(len(chosen_tokens), len(rejected_tokens)))

    def stats(lengths: List[int], name: str) -> Dict:
        arr = np.array(lengths)
        return {
            'name': name,
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': int(np.min(arr)),
            'max': int(np.max(arr)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
        }

    return {
        'prompt': stats(prompt_lengths, 'Prompt'),
        'chosen': stats(chosen_lengths, 'Chosen'),
        'rejected': stats(rejected_lengths, 'Rejected'),
        'total': stats(total_lengths, 'Total (prompt + max response)'),
    }

def print_analysis(stats: Dict, max_prompt: int = 256, max_total: int = 512):
    """統計結果を表示"""
    print("=" * 80)
    print("DPO Dataset Token Length Analysis")
    print("=" * 80)
    print()

    for key in ['prompt', 'chosen', 'rejected', 'total']:
        s = stats[key]
        print(f"{s['name']}:")
        print(f"  Mean:   {s['mean']:.1f} tokens")
        print(f"  Median: {s['median']:.1f} tokens")
        print(f"  Std:    {s['std']:.1f} tokens")
        print(f"  Min:    {s['min']} tokens")
        print(f"  Max:    {s['max']} tokens")
        print(f"  95%ile: {s['p95']:.1f} tokens")
        print(f"  99%ile: {s['p99']:.1f} tokens")
        print()

    # Check truncation impact
    print("=" * 80)
    print(f"Truncation Impact (max_prompt_length={max_prompt}, max_length={max_total})")
    print("=" * 80)
    print()

    # Calculate truncation percentage
    prompt_truncated = sum(1 for x in stats['prompt'] if x > max_prompt)
    total_truncated = sum(1 for x in stats['total'] if x > max_total)

    # This is a workaround - we need actual data
    print(f"⚠️  To get exact truncation percentages, run this script with your dataset:")
    print(f"    python scripts/analyze_dpo_token_length.py <dataset_path>")
    print()
    print(f"Estimated based on statistics:")
    if stats['prompt']['p95'] > max_prompt:
        print(f"  ⚠️  WARNING: 95th percentile prompt length ({stats['prompt']['p95']:.0f}) exceeds limit ({max_prompt})")
    else:
        print(f"  ✅ Prompt: 95% of samples fit within {max_prompt} tokens")

    if stats['total']['p95'] > max_total:
        print(f"  ⚠️  WARNING: 95th percentile total length ({stats['total']['p95']:.0f}) exceeds limit ({max_total})")
    else:
        print(f"  ✅ Total: 95% of samples fit within {max_total} tokens")
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_dpo_token_length.py <dataset_path>")
        print("\nExample:")
        print("  python scripts/analyze_dpo_token_length.py data/dpo/preference_dataset.jsonl")
        sys.exit(1)

    dataset_path = sys.argv[1]

    if not Path(dataset_path).exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Load tokenizer (using Qwen2.5-32B tokenizer as default)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")
    print()

    # Analyze
    print("Analyzing token lengths...")
    stats = analyze_token_lengths(dataset, tokenizer)

    # Print results
    print_analysis(stats, max_prompt=256, max_total=512)

    # Recommendations
    print("=" * 80)
    print("Recommendations")
    print("=" * 80)
    print()

    if stats['total']['p95'] <= 512:
        print("✅ Current settings (max_prompt_length=256, max_length=512) should work well.")
        print("   95% of your data fits within these limits.")
    elif stats['total']['p95'] <= 768:
        print("⚠️  Consider increasing max_length to 768 if memory allows.")
        print("   This would accommodate 95% of your data.")
    else:
        print("⚠️  Your dataset has long sequences:")
        print("   Option 1: Filter out samples exceeding 512 tokens")
        print("   Option 2: Increase max_length to 1024 and reduce batch size to 1")
        print("   Option 3: Use smaller model (7B/8B instead of 32B)")
    print()

if __name__ == "__main__":
    main()
