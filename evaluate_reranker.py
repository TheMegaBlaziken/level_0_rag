#!/usr/bin/env python3
"""
evaluate_reranker.py

Evaluates the accuracy of the original and fine-tuned reranker on your training data.

- Input: reranker_training_data.json (produced by prepare_reranker_training.py)

Usage:
    python evaluate_reranker.py

This script will print accuracy for both the original and fine-tuned reranker, and report any improvement.
"""

import os
import json
from FlagEmbedding import FlagReranker
import numpy as np

def load_training_data(json_path):
    """Load training data for evaluation."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_reranker(reranker, test_data, model_name):
    """Evaluate a reranker on test data."""
    correct = 0
    total = len(test_data)
    
    print(f"🔍 Evaluating {model_name}...")
    
    for i, example in enumerate(test_data):
        query = example['query']
        passage = example['passage']
        true_label = example['label']
        
        # Get reranker score
        scores = reranker.compute_score([[query, passage]])
        
        # Handle different return types from compute_score
        if isinstance(scores, list):
            score = scores[0]  # Extract first score from list
        else:
            score = scores  # Single score
        
        # Convert score to prediction (higher score = more relevant)
        predicted_label = 1 if score > 0.5 else 0
        
        if predicted_label == true_label:
            correct += 1
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{total} examples...")
    
    accuracy = correct / total
    print(f"✅ {model_name} Accuracy: {accuracy:.3f} ({correct}/{total})")
    return accuracy

def main():
    # Load test data (use a subset of training data for evaluation)
    test_data = load_training_data("reranker_training_data.json")
    
    # Use 20% for evaluation
    test_size = len(test_data) // 5
    test_data = test_data[:test_size]
    
    print(f"📊 Evaluating on {len(test_data)} test examples")
    
    # Load original model
    print("🔄 Loading original reranker...")
    original_reranker = FlagReranker('BAAI/bge-reranker-large')
    
    # Evaluate original
    original_accuracy = evaluate_reranker(original_reranker, test_data, "Original Reranker")
    
    # Check if fine-tuned model exists
    fine_tuned_path = "fine_tuned_reranker"
    if os.path.exists(fine_tuned_path):
        print("🔄 Loading fine-tuned reranker...")
        fine_tuned_reranker = FlagReranker(fine_tuned_path)
        
        # Evaluate fine-tuned
        fine_tuned_accuracy = evaluate_reranker(fine_tuned_reranker, test_data, "Fine-tuned Reranker")
        
        # Compare
        improvement = fine_tuned_accuracy - original_accuracy
        print(f"\n📈 Improvement: {improvement:+.3f}")
        
        if improvement > 0:
            print("🎉 Fine-tuned model performs better!")
        elif improvement < 0:
            print("⚠️ Original model performs better. Consider adjusting training parameters.")
        else:
            print("🤔 Models perform similarly.")
    else:
        print("❌ Fine-tuned model not found. Run training first.")

if __name__ == "__main__":
    main() 