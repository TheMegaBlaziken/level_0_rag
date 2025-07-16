#!/usr/bin/env python3
"""
update_rag_with_fine_tuned.py

Updates rag_chat.py to use the fine-tuned reranker model.
Creates a backup of the original file and modifies the FlagReranker initialization.
"""

import os
import shutil
from datetime import datetime

def backup_original_file(file_path):
    """Create a backup of the original file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    
    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)
        print(f"📋 Backup created: {backup_path}")
        return backup_path
    else:
        print(f"❌ File not found: {file_path}")
        return None

def update_rag_chat_with_fine_tuned(fine_tuned_path="fine_tuned_reranker"):
    """
    Update rag_chat.py to use the fine-tuned reranker.
    
    Args:
        fine_tuned_path: Path to the fine-tuned model directory
    """
    
    rag_chat_path = "rag_chat.py"
    
    # Check if fine-tuned model exists
    if not os.path.exists(fine_tuned_path):
        print(f"❌ Fine-tuned model not found: {fine_tuned_path}")
        print("Run the training script first: bash train_reranker.sh")
        return False
    
    # Check if rag_chat.py exists
    if not os.path.exists(rag_chat_path):
        print(f"❌ rag_chat.py not found: {rag_chat_path}")
        return False
    
    # Create backup
    backup_path = backup_original_file(rag_chat_path)
    if not backup_path:
        return False
    
    print(f"🔧 Updating {rag_chat_path} to use fine-tuned model...")
    
    # Read the file
    with open(rag_chat_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the FlagReranker initialization
    original_line = "flag_reranker = FlagReranker('BAAI/bge-reranker-large')"
    new_line = f"flag_reranker = FlagReranker('{fine_tuned_path}')"
    
    if original_line in content:
        # Replace the line
        updated_content = content.replace(original_line, new_line)
        
        # Write the updated content
        with open(rag_chat_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {rag_chat_path}")
        print(f"   Original: {original_line}")
        print(f"   Updated:  {new_line}")
        
        return True
    else:
        print("❌ Could not find FlagReranker initialization line in rag_chat.py")
        print("   Expected: flag_reranker = FlagReranker('BAAI/bge-reranker-large')")
        return False

def create_ab_test_script():
    """
    Create a script to A/B test the original vs fine-tuned reranker.
    """
    
    script_content = '''#!/usr/bin/env python3
"""
ab_test_reranker.py

A/B test script to compare original vs fine-tuned reranker performance.
"""

import os
import json
import time
from rag_chat import ask_question
from FlagEmbedding import FlagReranker

def test_reranker_performance(questions, model_name, reranker_path):
    """
    Test reranker performance on a set of questions.
    
    Args:
        questions: List of test questions
        model_name: Name for logging
        reranker_path: Path to reranker model
    
    Returns:
        dict: Performance metrics
    """
    
    print(f"🧪 Testing {model_name}...")
    
    # Temporarily update rag_chat.py to use this reranker
    original_content = None
    try:
        with open("rag_chat.py", 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Replace reranker path
        updated_content = original_content.replace(
            "flag_reranker = FlagReranker('fine_tuned_reranker')",
            f"flag_reranker = FlagReranker('{reranker_path}')"
        )
        
        with open("rag_chat.py", 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        # Test questions
        results = []
        start_time = time.time()
        
        for i, question in enumerate(questions):
            print(f"   Question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Get answer and chunk info
                (
                    answer, snippet_scores, sources, images, threshold,
                    img_scores, debug_logs, debug_img_list,
                    answer_structured, useful_chunks, not_useful_chunks, raw_llm_output
                ) = ask_question(question)
                
                # Calculate metrics
                total_chunks = len(useful_chunks) + len(not_useful_chunks)
                useful_ratio = len(useful_chunks) / max(1, total_chunks)
                
                results.append({
                    "question": question,
                    "useful_chunks": len(useful_chunks),
                    "not_useful_chunks": len(not_useful_chunks),
                    "useful_ratio": useful_ratio,
                    "threshold": threshold
                })
                
            except Exception as e:
                print(f"   ❌ Error on question {i+1}: {e}")
                results.append({
                    "question": question,
                    "error": str(e)
                })
        
        end_time = time.time()
        
        # Calculate overall metrics
        successful_results = [r for r in results if "error" not in r]
        
        if successful_results:
            avg_useful_ratio = sum(r["useful_ratio"] for r in successful_results) / len(successful_results)
            avg_threshold = sum(r["threshold"] for r in successful_results) / len(successful_results)
            total_time = end_time - start_time
            
            metrics = {
                "model_name": model_name,
                "total_questions": len(questions),
                "successful_questions": len(successful_results),
                "avg_useful_ratio": avg_useful_ratio,
                "avg_threshold": avg_threshold,
                "total_time": total_time,
                "avg_time_per_question": total_time / len(questions)
            }
        else:
            metrics = {
                "model_name": model_name,
                "error": "No successful questions"
            }
        
        return metrics
        
    finally:
        # Restore original content
        if original_content:
            with open("rag_chat.py", 'w', encoding='utf-8') as f:
                f.write(original_content)

def main():
    """Run A/B test between original and fine-tuned reranker."""
    
    print("🧪 A/B Testing Reranker Performance")
    print("=" * 50)
    
    # Test questions (you can modify these)
    test_questions = [
        "What is the methodology used in the CeH9 study?",
        "How do the critical currents compare between CeH9 and Bi-2223?",
        "What are the key findings about n-values in the research?",
        "What experimental conditions were used in the study?",
        "How do the results compare to traditional superconductors?"
    ]
    
    print(f"📝 Testing with {len(test_questions)} questions")
    
    # Test original model
    original_metrics = test_reranker_performance(
        test_questions, 
        "Original Reranker", 
        "BAAI/bge-reranker-large"
    )
    
    print()
    
    # Test fine-tuned model
    fine_tuned_metrics = test_reranker_performance(
        test_questions, 
        "Fine-tuned Reranker", 
        "fine_tuned_reranker"
    )
    
    # Compare results
    print("\\n" + "=" * 50)
    print("📊 A/B Test Results")
    print("=" * 50)
    
    print(f"Original Reranker:")
    for key, value in original_metrics.items():
        if key != "model_name":
            print(f"  {key}: {value}")
    
    print(f"\\nFine-tuned Reranker:")
    for key, value in fine_tuned_metrics.items():
        if key != "model_name":
            print(f"  {key}: {value}")
    
    # Calculate improvements
    if "avg_useful_ratio" in original_metrics and "avg_useful_ratio" in fine_tuned_metrics:
        useful_improvement = fine_tuned_metrics["avg_useful_ratio"] - original_metrics["avg_useful_ratio"]
        print(f"\\n📈 Useful Chunk Ratio Improvement: {useful_improvement:+.3f}")
        
        if useful_improvement > 0:
            print("🎉 Fine-tuned model shows improvement!")
        elif useful_improvement < 0:
            print("⚠️ Original model performs better.")
        else:
            print("🤔 Models perform similarly.")
    
    # Save results
    results = {
        "original": original_metrics,
        "fine_tuned": fine_tuned_metrics,
        "test_questions": test_questions,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("ab_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n💾 Results saved to: ab_test_results.json")

if __name__ == "__main__":
    main()
'''
    
    script_path = "ab_test_reranker.py"
    with open(script_path, 'w', encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"🧪 A/B test script created: {script_path}")
    print("   Run: python ab_test_reranker.py")
    
    return script_path

def main():
    """Main function to update rag_chat.py and create A/B test script."""
    
    print("🔧 Updating RAG Pipeline with Fine-tuned Reranker")
    print("=" * 60)
    
    # Update rag_chat.py
    success = update_rag_chat_with_fine_tuned()
    
    if success:
        print("\n" + "=" * 60)
        
        # Create A/B test script
        create_ab_test_script()
        
        print("\n" + "=" * 60)
        print("🎯 Next Steps:")
        print("1. Test your updated RAG pipeline")
        print("2. Run A/B test: python ab_test_reranker.py")
        print("3. Compare performance metrics")
        print("4. If needed, revert: cp rag_chat.py.backup_* rag_chat.py")
    else:
        print("\n❌ Failed to update rag_chat.py")

if __name__ == "__main__":
    main() 