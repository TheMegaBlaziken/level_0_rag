#!/usr/bin/env python3
"""
prepare_reranker_training.py

Converts feedback CSV from auto_feedback_collector.py to FlagEmbedding reranker training format.
Outputs a JSON file that can be used directly with FlagEmbedding training scripts.
"""

import csv
import json
import os
from datetime import datetime

def convert_feedback_to_training_data(feedback_csv_path, output_json_path):
    """
    Convert feedback CSV to FlagEmbedding training format.
    
    Expected CSV format (no headers):
    timestamp, paper_id, heading, chunk_text, question, feedback_type
    
    Output JSON format for FlagEmbedding:
    [
        {
            "query": "What is the methodology...?",
            "passage": "The experimental methodology used...",
            "label": 1  # 1 for useful, 0 for not useful
        },
        ...
    ]
    """
    
    training_data = []
    
    if not os.path.exists(feedback_csv_path):
        print(f"❌ Feedback CSV not found: {feedback_csv_path}")
        print("Run auto_feedback_collector.py first to generate feedback data.")
        return
    
    print(f"📖 Reading feedback data from: {feedback_csv_path}")
    
    with open(feedback_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        for row in reader:
            # Skip empty rows
            if not row or len(row) < 6:
                continue
                
            # Extract data from CSV by position (no headers)
            timestamp = row[0].strip()
            paper_id = row[1].strip()
            heading = row[2].strip()
            chunk_text = row[3].strip()
            question = row[4].strip()
            feedback_type = row[5].strip().lower()
            
            # Skip empty entries
            if not question or not chunk_text:
                continue
            
            # Convert feedback type to label
            if feedback_type == 'useful':
                label = 1
            elif feedback_type == 'not useful':
                label = 0
            else:
                print(f"⚠️ Unknown feedback type: {feedback_type}, skipping...")
                continue
            
            # Create training example
            training_example = {
                "query": question,
                "passage": chunk_text,
                "label": label
            }
            
            training_data.append(training_example)
    
    print(f"📊 Processed {len(training_data)} training examples")
    
    # Count labels
    useful_count = sum(1 for ex in training_data if ex['label'] == 1)
    not_useful_count = sum(1 for ex in training_data if ex['label'] == 0)
    
    print(f"   - Useful chunks: {useful_count}")
    print(f"   - Not useful chunks: {not_useful_count}")
    
    # Save to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Training data saved to: {output_json_path}")
    
    # Print some examples
    print("\n📝 Sample training examples:")
    for i, example in enumerate(training_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Query: {example['query'][:100]}...")
        print(f"  Passage: {example['passage'][:100]}...")
        print(f"  Label: {example['label']} ({'useful' if example['label'] == 1 else 'not useful'})")

def main():
    """Main function to prepare training data."""
    
    print("🚀 Preparing FlagEmbedding Reranker Training Pipeline")
    print("=" * 60)
    
    # Convert feedback to training data
    feedback_csv = "feedback_log.csv"
    training_json = "reranker_training_data.json"
    
    convert_feedback_to_training_data(feedback_csv, training_json)
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. (Optional) Convert to FlagEmbedding format: python convert_training_data.py")
    print("2. Fine-tune reranker: python simple_train_reranker.py")
    print("3. Evaluate reranker: python evaluate_reranker.py")
    print("4. Update rag_chat.py to use the fine-tuned model: python update_rag_with_fine_tuned.py")
    print("5. (Optional) A/B test reranker: python ab_test_reranker.py")
    print("\n📁 Available files:")
    print("   - simple_train_reranker.py (training script)")
    print("   - convert_training_data.py (optional conversion script)")
    print("   - evaluate_reranker.py (evaluation script)")
    print("   - ab_test_reranker.py (A/B testing script)")
    print("   - update_rag_with_fine_tuned.py (update RAG pipeline)")

if __name__ == "__main__":
    main() 