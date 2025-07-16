# SciFy RAG Pipeline: End-to-End Guide

A comprehensive Retrieval-Augmented Generation (RAG) pipeline for scientific papers, supporting everything from PDF download to optional reranker fine-tuning. This guide covers every step, file, and configuration needed to run and extend the system.

---

## Pipeline Overview

1. **Dockerized Weaviate Setup**
2. **Download PDFs from arXiv**
3. **Convert PDFs to Markdown**
4. **Ingest Markdown into Weaviate**
5. **Run the RAG Chatbot**
6. **(Optional) Collect Feedback for Fine-tuning**
7. **(Optional) Prepare and Convert Training Data**
8. **(Optional) Fine-tune the Reranker**
9. **(Optional) Evaluate and A/B Test Reranker**
10. **(Optional) Update Pipeline to Use Fine-tuned Reranker**

---

## 1. Dockerized Weaviate Setup

**File:** `docker-compose.yml`

- **Purpose:** Launches a local Weaviate vector database with OpenAI vectorizer support.
- **Usage:**
  1. Install [Docker](https://docs.docker.com/get-docker/) if not already installed.
  2. In your project root, run:
     ```sh
     docker-compose up -d
     ```
  3. This will start Weaviate on ports 8080 (REST) and 50051 (gRPC).
- **Configuration:**
  - The container uses OpenAI for vectorization. Set your OpenAI API key in the environment section of `docker-compose.yml` or via `.env`.
  - Data is persisted in `./weaviate_data`.

---

## 2. Download PDFs from arXiv

**File:** `download_arxiv_pdfs.py`

- **Purpose:** Download PDFs from arXiv by ID or search query.
- **Dependencies:** `arxiv`, `requests`
- **Usage:**
  - Download by IDs:
    ```sh
    python download_arxiv_pdfs.py --ids 2301.00001,2105.12345 --output ./pdfs
    ```
  - Download by search:
    ```sh
    python download_arxiv_pdfs.py --query "machine learning" --max-results 20 --output ./pdfs
    ```
- **How it works:** Fetches PDFs from arXiv and saves them to the specified directory.

---

## 3. Convert PDFs to Markdown

**File:** `pdf_to_markdown.py`

- **Purpose:** Convert PDFs to Markdown, extract images, and generate AI captions.
- **Dependencies:** `marker-pdf`, `openai`, `pillow`
- **Usage:**
  - Single PDF:
    ```sh
    python pdf_to_markdown.py --input paper.pdf --output paper.md --use-llm
    ```
  - Directory:
    ```sh
    python pdf_to_markdown.py --input pdf_folder --output md_folder --use-llm
    ```
- **How it works:**
  - Extracts text and images from PDFs.
  - Saves images to a folder.
  - Uses OpenAI to generate scientific captions for each image.
  - Embeds captions in the Markdown file.

---

## 4. Ingest Markdown into Weaviate

**File:** `ingest_markdown.py`

- **Purpose:** Chunk Markdown, embed with OpenAI, upload to Weaviate, and push images to GitHub.
- **Dependencies:** `weaviate`, `openai`, `transformers`, `Pillow`, `nltk`, `requests`, `pyyaml`, `dotenv`
- **Environment Variables:**
  - `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `OPENAI_API_KEY`, `GITHUB_TOKEN`, `GITHUB_REPO`
  - Optionally: `GITHUB_BRANCH`, `GITHUB_PATH_PREFIX`
- **Usage:**
  ```sh
  python ingest_markdown.py --input path/to/markdowns
  ```
- **How it works:**
  - Reads Markdown files, splits into sections and chunks.
  - Embeds text chunks using OpenAI.
  - Uploads text and metadata to Weaviate.
  - Uploads images to GitHub and indexes them in Weaviate with CLIP embeddings.

---

## 5. Run the RAG Chatbot

**Files:** `app.py`, `rag_chat.py`

- **Purpose:** Provide a Streamlit UI for question answering over your ingested papers.
- **Dependencies:** `streamlit`, `weaviate`, `openai`, `FlagEmbedding`, `transformers`, `Pillow`, `nltk`
- **Usage:**
  ```sh
  streamlit run app.py
  ```
- **How it works:**
  - User asks a question.
  - Retrieves relevant chunks from Weaviate.
  - Reranks with FlagReranker (fine-tuned or base).
  - Generates an answer with OpenAI, including inline citations and images.
  - Shows debug logs and progress in the UI.

---

## 6. (Optional) Collect Feedback for Fine-tuning

**File:** `auto_feedback_collector.py`

- **Purpose:** Automatically generate questions for each paper, run them through the RAG pipeline, and log which chunks were useful or not for each answer.
- **Usage:**
  ```sh
  python auto_feedback_collector.py
  ```
- **How it works:**
  - For each Markdown file, generates technical questions using OpenAI.
  - Runs each question through the RAG pipeline.
  - Logs feedback (useful/not useful) for each chunk in `feedback_log.csv`.

---

## 7. (Optional) Prepare and Convert Training Data

**Files:**
- `prepare_reranker_training.py`: Converts feedback CSV to FlagEmbedding training format (`reranker_training_data.json`).
- `convert_training_data.py`: Converts to the format expected by FlagEmbedding's fine-tuning scripts (`flagembedding_training_data.json`).

- **Usage:**
  1. Prepare from feedback:
     ```sh
     python prepare_reranker_training.py
     ```
  2. Convert to FlagEmbedding format:
     ```sh
     python convert_training_data.py
     ```
- **How it works:**
  - Reads feedback from `feedback_log.csv`.
  - Outputs a JSON file for training.
  - Converts to the grouped format required for FlagEmbedding or HuggingFace training.

---

## 8. (Optional) Fine-tune the Reranker

**Recommended Fine-tuning Pipeline:**

1. **Prepare training data:**
   ```sh
   python prepare_reranker_training.py
   ```
   - Produces `reranker_training_data.json` from feedback.
2. **Convert to grouped format:**
   ```sh
   python convert_training_data.py
   ```
   - Produces `flagembedding_training_data.json` (required for HuggingFace-based training).
3. **Fine-tune reranker (recommended):**
   ```sh
   python simple_train_reranker.py
   ```
   - Uses HuggingFace Transformers. Output: `fine_tuned_reranker/`.
4. **Evaluate reranker:**
   ```sh
   python evaluate_reranker.py
   ```
   - Prints accuracy for both original and fine-tuned reranker.
5. **Update pipeline to use fine-tuned model:**
   ```sh
   python update_rag_with_fine_tuned.py
   ```
6. **(Optional) A/B test reranker:**
   ```sh
   python ab_test_reranker.py
   ```
   - Compares both rerankers on a set of questions.

---

## 9. (Optional) Evaluate and A/B Test Reranker

**Files:**
- `evaluate_reranker.py`: Evaluates accuracy of original and fine-tuned reranker.
- `ab_test_reranker.py`: Runs a set of questions through both rerankers and compares metrics.

- **Usage:**
  - Evaluate:
    ```sh
    python evaluate_reranker.py
    ```
  - A/B test:
    ```sh
    python ab_test_reranker.py
    ```
- **How it works:**
  - Evaluates accuracy and improvement.
  - Saves results to `ab_test_results.json`.

---

## 10. (Optional) Update Pipeline to Use Fine-tuned Reranker

**File:** `update_rag_with_fine_tuned.py`

- **Purpose:** Update `rag_chat.py` to use your fine-tuned reranker.
- **Usage:**
  ```sh
  python update_rag_with_fine_tuned.py
  ```
- **How it works:**
  - Backs up your original `rag_chat.py`.
  - Modifies the reranker initialization to use your fine-tuned model.

---

## File-by-File Summary

| File                        | Purpose/Usage                                                                 |
|-----------------------------|-------------------------------------------------------------------------------|
| `docker-compose.yml`        | Launches Weaviate vector DB with OpenAI support via Docker.                   |
| `download_arxiv_pdfs.py`    | Download PDFs from arXiv by ID or search query.                              |
| `pdf_to_markdown.py`        | Convert PDFs to Markdown, extract images, generate AI captions.               |
| `ingest_markdown.py`        | Chunk and embed Markdown, upload to Weaviate, push images to GitHub.          |
| `app.py`                    | Streamlit UI for the chatbot.                                                 |
| `rag_chat.py`               | Core RAG pipeline: retrieval, reranking, LLM, image handling, debug.          |
| `auto_feedback_collector.py`| Collects feedback for fine-tuning by running questions and logging chunk utility. |
| `prepare_reranker_training.py` | Converts feedback to training data for reranker.                          |
| `convert_training_data.py`  | Converts training data to grouped format for HuggingFace fine-tuning.         |
| `simple_train_reranker.py`  | Fine-tune the reranker using HuggingFace Transformers.                        |
| `evaluate_reranker.py`      | Evaluate reranker accuracy.                                                   |
| `ab_test_reranker.py`       | A/B test reranker performance and save results.                               |
| `update_rag_with_fine_tuned.py` | Update the pipeline to use your fine-tuned reranker.                    |

---

## Environment Variables

Set these in a `.env` file or your shell:
```
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-weaviate-api-key
OPENAI_API_KEY=your-openai-api-key
GITHUB_TOKEN=your-github-token
GITHUB_REPO=yourusername/your-repo
GITHUB_BRANCH=main
GITHUB_PATH_PREFIX=papers
```

---

## Troubleshooting & Tips

- **Weaviate/OpenAI errors:** Check your API keys and URLs. Ensure Docker is running.
- **Image upload issues:** Ensure your GitHub token has repo access.
- **Schema changes:** Re-ingest your data if you change metadata fields.
- **Dependency issues:** Reinstall with `pip install -r requirements.txt`.
- **Training issues:** Try both FlagEmbedding and HuggingFace scripts for fine-tuning.
- **Best practice:** Always backup your data and scripts before running updates or training.

---

## License

MIT (or your chosen license)
