# Research SciFy RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for ingesting arXiv papers into a Weaviate vector database and powering an OpenAI-backed chat interface.

---

## 📁 Repo Structure

```
.
├── download_arxiv_pdfs.py      # 1️⃣ Download PDFs from arXiv
├── pdf_to_markdown.py         # 2️⃣ Convert PDFs → Markdown + image captioning
├── ingest_markdown.py         # 3️⃣ Ingest Markdown & images into Weaviate
├── rag_chat.py                # 4️⃣ RAG-powered chatbot
└── delete_cluster_data.py     # 5️⃣ Wipe / reset Weaviate data
```

---

## 🔍 File Descriptions

### 1. download_arxiv_pdfs.py  
**Purpose:** Automatically fetch PDF files from the arXiv repository for downstream processing.

**How it works:**
- Uses the `arxiv` Python package to query papers by keyword, author, or category.
- Parses command-line arguments:
  - `--query`: search terms (e.g., "graph neural networks").
  - `--max-results`: maximum number of papers to retrieve.
  - `--output-dir`: directory to save downloaded PDFs.
- Downloads each PDF as `<arxiv_id>.pdf` and logs progress with `tqdm`.

**Usage:**
```bash
python download_arxiv_pdfs.py   --query "graph neural networks"   --max-results 50   --output-dir ./pdfs
```

---

### 2. pdf_to_markdown.py  
**Purpose:** Convert downloaded PDFs into Markdown, extract images, and generate captions.

**How it works:**
- Extracts text via `pdfminer.six` and converts to Markdown using `markdownify`.
- Extracts figures with `Pillow`, saves to `images/`, and captions via OpenAI.
- Embeds images and captions in Markdown files.

**Usage:**
```bash
python pdf_to_markdown.py   --input-dir ./pdfs   --output-dir ./markdown
```

---

### 3. ingest_markdown.py  
**Purpose:** Chunk Markdown & images, embed content, and ingest into Weaviate.

**How it works:**
- Splits `.md` files into semantic chunks.
- Requests embeddings via the OpenAI Embeddings API.
- Constructs objects with:
  - `text`: chunk or image caption.
  - `metadata`: source filename, chunk index, section header.
  - `image_url` (if applicable).
- Uploads objects to the Weaviate `DocumentChunk` class using `weaviate-client`.

**Usage:**
```bash
python ingest_markdown.py   --markdown-dir ./markdown   --weaviate-class DocumentChunk
```

---

### 4. rag_chat.py  
**Purpose:** Interactive chat interface that retrieves relevant content from Weaviate and generates answers via OpenAI.

**How it works:**
- Embeds user queries via the OpenAI Embeddings API.
- Queries Weaviate for top-K similar chunks.
- Builds a prompt with retrieved context and calls `openai.ChatCompletion`.
- Displays answers along with source citations.

**Usage:**
```bash
python rag_chat.py   --weaviate-class DocumentChunk   --openai-model gpt-4o-mini
```

---

### 5. delete_cluster_data.py  
**Purpose:** Reset or wipe data from the Weaviate instance.

**How it works:**
- Parses `--classes`; if omitted, deletes all objects.
- Uses `weaviate-client` to purge specified classes or the entire dataset.

**Usage:**
```bash
python delete_cluster_data.py   --classes DocumentChunk,Image
```

---

## 🔧 Installation

1. **Clone** the repo:
   ```bash
   git clone https://github.com/your-username/level_0_rag.git
   cd level_0_rag
   ```
2. **Create** & activate a Python 3.9+ virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate   # Windows
   ```
3. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure** environment variables in a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_key
   WEAVIATE_URL=https://your-weaviate-instance.com
   WEAVIATE_API_KEY=your_weaviate_api_key
   ```

---

## 🛠 Contributing & License

MIT © 2025 Your Name
