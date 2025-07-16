# Large Files and Data Management

This document explains the large files required by the SciFy RAG pipeline and how to obtain them.

## Why Large Files Are Not in the Repository

The following files and directories are intentionally excluded from this git repository to:
- **Keep the repository small and fast** (under 100MB)
- **Avoid GitHub's file size limits** (100MB per file, 1GB total recommended)
- **Protect sensitive information** (API keys, tokens)
- **Allow users to generate fresh data** using the provided scripts

## Required Large Files

### 1. Model Files
**Location:** `fine_tuned_reranker/`
**Size:** ~50-200MB
**Purpose:** Fine-tuned reranker model for improved search results
**How to obtain:** Run the fine-tuning pipeline (see README.md section 8)

### 2. PDF Papers
**Location:** `papers/pdfs/`
**Size:** Varies (typically 1-50MB per paper)
**Purpose:** Source documents for the RAG system
**How to obtain:** Run `download_arxiv_pdfs.py` (see README.md section 2)

### 3. Weaviate Database
**Location:** `weaviate_data/`
**Size:** Varies (typically 100MB-1GB+)
**Purpose:** Vector database storing document embeddings and metadata
**How to obtain:** Run `ingest_markdown.py` (see README.md section 4)

### 4. Training Data
**Files:** 
- `flagembedding_training_data.json`
- `reranker_training_data.json`
- `feedback_log.csv`
- `ab_test_results.json`
**Purpose:** Data for fine-tuning and evaluation
**How to obtain:** Generated during feedback collection and training (see README.md sections 7-8)

### 5. FlagEmbedding Library
**Location:** `FlagEmbedding/`
**Size:** ~100MB
**Purpose:** Core library for embedding and reranking
**How to obtain:** 
```bash
git clone https://github.com/FlagOpen/FlagEmbedding.git
# or
pip install FlagEmbedding
```

## Setup Instructions

### For New Users
1. **Clone this repository**
2. **Follow the README.md** step by step
3. **Each script will generate/download** the required files automatically

### For Development
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Set up environment variables:** Copy `.env.example` to `.env` and fill in your API keys
3. **Run the pipeline stages** in order (see README.md)

### For Production
- Consider using cloud storage (AWS S3, Google Drive) for large files
- Use Weaviate Cloud instead of local Weaviate
- Store models on Hugging Face Hub or similar platforms

## File Size Estimates

| Component | Typical Size | Notes |
|-----------|-------------|-------|
| PDF Papers | 10-500MB | Depends on number of papers |
| Weaviate DB | 100MB-2GB | Grows with document count |
| Fine-tuned Model | 50-200MB | Depends on model size |
| FlagEmbedding | 100MB | Library files |
| **Total** | **260MB-2.8GB** | **Not included in repo** |

## Troubleshooting

### "File not found" errors
- Ensure you've run the prerequisite scripts
- Check that large files are in the correct locations
- Verify environment variables are set correctly

### Out of disk space
- Large files require significant disk space
- Consider using external storage for production
- Clean up old checkpoints and temporary files

### Performance issues
- Large files can slow down operations
- Consider using SSDs for better performance
- Monitor disk I/O during heavy operations

## Security Notes

- **Never commit `.env` files** with real API keys
- **Use environment variables** for sensitive configuration
- **Keep large files local** unless using secure cloud storage
- **Regularly rotate API keys** and tokens

## Support

If you encounter issues with large files:
1. Check the README.md for detailed setup instructions
2. Verify all prerequisites are installed
3. Ensure sufficient disk space is available
4. Check that environment variables are correctly set 