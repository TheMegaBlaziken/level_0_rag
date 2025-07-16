#!/usr/bin/env python3
"""
simple_train_reranker.py

Recommended reranker fine-tuning script for this pipeline.
Uses HuggingFace Transformers to fine-tune the BAAI/bge-reranker-large model.

- Input: flagembedding_training_data.json (produced by convert_training_data.py)
- Output: fine_tuned_reranker/ (directory with model and tokenizer)

Usage:
    python simple_train_reranker.py

This script is robust and recommended for most users. It bypasses distributed training issues in FlagEmbedding's .fit().
"""

import json
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np

def load_training_data(data_path):
    """Load and prepare training data."""
    print(f"📖 Loading training data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to the format expected by HuggingFace
    training_examples = []
    
    for item in data:
        query = item['query']
        pos_passages = item['pos']
        neg_passages = item['neg']
        
        # Create positive examples (label = 1.0 for relevant)
        for passage in pos_passages:
            training_examples.append({
                'text': f"{query} [SEP] {passage}",
                'label': 1.0  # Use float for regression
            })
        
        # Create negative examples (label = 0.0 for not relevant)
        for passage in neg_passages:
            training_examples.append({
                'text': f"{query} [SEP] {passage}",
                'label': 0.0  # Use float for regression
            })
    
    print(f"✅ Created {len(training_examples)} training examples")
    print(f"   - Positive examples: {sum(1 for ex in training_examples if ex['label'] == 1.0)}")
    print(f"   - Negative examples: {sum(1 for ex in training_examples if ex['label'] == 0.0)}")
    
    return training_examples

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the examples."""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

def main():
    # Configuration
    MODEL_NAME = "BAAI/bge-reranker-large"
    TRAIN_DATA_PATH = "flagembedding_training_data.json"
    OUTPUT_DIR = "fine_tuned_reranker"
    NUM_EPOCHS = 3
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512
    
    print("🚀 Starting simplified reranker fine-tuning...")
    print(f"Model: {MODEL_NAME}")
    print(f"Training data: {TRAIN_DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load training data
    training_examples = load_training_data(TRAIN_DATA_PATH)
    
    # Create dataset
    dataset = Dataset.from_list(training_examples)
    
    # Load tokenizer and model
    print("📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load the model with regression head (1 output) - this is what the model was trained for
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,  # Single output for regression scoring
        problem_type="regression"  # This is the key change
    )
    
    # Add special tokens if needed
    if tokenizer.sep_token is None:
        tokenizer.sep_token = "[SEP]"
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    
    # Tokenize dataset
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Add labels back to the tokenized dataset
    tokenized_dataset = tokenized_dataset.add_column("labels", dataset["label"])
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="no",  # Changed from evaluation_strategy
        load_best_model_at_end=False,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_pin_memory=False,  # Avoid issues on Windows
        gradient_checkpointing=True,  # Save memory
        overwrite_output_dir=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save the model
    print(f"💾 Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Training completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nTo use the fine-tuned model in your RAG pipeline:")
    print(f"flag_reranker = FlagReranker('{OUTPUT_DIR}')")

if __name__ == "__main__":
    main() 