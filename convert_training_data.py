#!/usr/bin/env python3
"""
convert_training_data.py

Converts the simple training data format (from prepare_reranker_training.py) to the grouped format required by HuggingFace-based fine-tuning (simple_train_reranker.py).

- Input: reranker_training_data.json (list of {"query", "passage", "label"})
- Output: flagembedding_training_data.json (list of {"query", "pos": [...], "neg": [...], ...})

Usage:
    python convert_training_data.py

This step is REQUIRED for HuggingFace-based fine-tuning (simple_train_reranker.py).
It is OPTIONAL for legacy FlagEmbedding .fit() training.
"""

import json
from collections import defaultdict
import random

def convert_training_data(input_file, output_file):
    """Convert training data to FlagEmbedding format."""
    
    print(f"📖 Reading training data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Processing {len(data)} examples...")
    
    # Group by query
    query_groups = defaultdict(lambda: {"pos": [], "neg": []})
    
    for item in data:
        query = item["query"]
        passage = item["passage"]
        label = item["label"]
        
        if label == 1:
            query_groups[query]["pos"].append(passage)
        else:
            query_groups[query]["neg"].append(passage)
    
    # Collect all negative passages for synthetic negatives
    all_negative_passages = []
    for groups in query_groups.values():
        all_negative_passages.extend(groups["neg"])
    
    print(f"📊 Found {len(all_negative_passages)} total negative passages for synthetic sampling")
    
    # Convert to FlagEmbedding format
    converted_data = []
    total_pos = 0
    total_neg = 0
    
    for query, groups in query_groups.items():
        pos_passages = groups["pos"]
        neg_passages = groups["neg"]
        
        # Skip if no positive examples
        if not pos_passages:
            continue
        
        # If no negative examples, create synthetic ones
        if not neg_passages:
            if all_negative_passages:
                # Sample 2-4 negative passages from other queries
                num_synthetic = min(3, len(all_negative_passages))
                synthetic_negatives = random.sample(all_negative_passages, num_synthetic)
                neg_passages = synthetic_negatives
                print(f"✅ Created {num_synthetic} synthetic negatives for query: {query[:50]}...")
            else:
                print(f"⚠️  Warning: No negative examples available, skipping: {query[:50]}...")
                continue
        
        converted_item = {
            "query": query,
            "pos": pos_passages,
            "neg": neg_passages,
            "pos_scores": [1.0] * len(pos_passages),  # Default scores
            "neg_scores": [0.0] * len(neg_passages),  # Default scores
        }
        
        converted_data.append(converted_item)
        total_pos += len(pos_passages)
        total_neg += len(neg_passages)
    
    print(f"✅ Converted {len(converted_data)} query groups")
    print(f"   - Total positive passages: {total_pos}")
    print(f"   - Total negative passages: {total_neg}")
    print(f"   - Average positive per query: {total_pos/len(converted_data):.1f}")
    print(f"   - Average negative per query: {total_neg/len(converted_data):.1f}")
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted data saved to: {output_file}")
    
    # Show sample
    if converted_data:
        print("\n📝 Sample converted data:")
        sample = converted_data[0]
        print(f"Query: {sample['query'][:100]}...")
        print(f"Positive passages: {len(sample['pos'])}")
        print(f"Negative passages: {len(sample['neg'])}")

if __name__ == "__main__":
    convert_training_data("reranker_training_data.json", "flagembedding_training_data.json") 