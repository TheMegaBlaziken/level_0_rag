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
- For each result, downloads the PDF and names it `<arxiv_id>.pdf` in the output directory.
- Logs progress with `tqdm` and handles network errors gracefully.

**Usage:**
```bash
python download_arxiv_pdfs.py   --query "graph neural networks"   --max-results 50   --output-dir ./pdfs
```

---

### 2. pdf_to_markdown.py  
**Purpose:** Convert downloaded PDFs into Markdown format, extract images, and generate captions.

**How it works:**
- Iterates over all PDFs in `--input-dir`.
- Uses `pdfminer.six` (or similar) to extract text and layout.
- Converts PDF text blocks to Markdown using `markdownify`.
- Extracts figures and plots via `Pillow` or a PDF toolkit:
  - Saves each image to `<output-dir>/images/`.
  - Calls OpenAI’s image-caption API for descriptive captions.
  - Embeds captions under image links in the Markdown.
- Writes out `.md` files preserving section headings and figure references.

**Usage:**
```bash
python pdf_to_markdown.py   --input-dir ./pdfs   --output-dir ./markdown
```

---

### 3. ingest_markdown.py  
**Purpose:** Chunk markdown documents and images, embed content, and ingest into Weaviate.

**How it works:**
- Traverses all `.md` files in `--markdown-dir`.
- Splits text into semantic chunks (e.g., paragraphs or fixed token windows).
- Reads associated image-caption pairs.
- For each chunk or image:
  - Requests embeddings via the OpenAI embeddings API.
  - Constructs a Weaviate object with:
    - `text`: the chunk or caption.
    - `metadata`: source filename, chunk index, section header.
    - `image_url`: (when applicable).
- Uses `weaviate-client` to batch-upload objects into the `DocumentChunk` class.
- Handles schema creation if needed and logs ingestion stats.

**Usage:**
```bash
python ingest_markdown.py   --markdown-dir ./markdown   --weaviate-class DocumentChunk
```

---

### 4. rag_chat.py  
**Purpose:** Provide an interactive chat interface that retrieves relevant content from Weaviate and responds using OpenAI.

**How it works:**
- Parses arguments:
  - `--weaviate-class`: name of the class storing chunks.
  - `--openai-model`: name of the GPT model to use.
- On startup, connects to Weaviate and verifies the schema.
- In a REPL loop:
  1. Accepts user input question.
  2. Embeds the question via OpenAI.
  3. Queries Weaviate for top-k similar chunks by vector similarity.
  4. Constructs a chat prompt combining retrieved chunks as context.
  5. Calls `openai.ChatCompletion` to generate an answer.
  6. Displays the response and optionally sources with citations.

**Usage:**
```bash
python rag_chat.py   --weaviate-class DocumentChunk   --openai-model gpt-4o-mini
```

---

### 5. delete_cluster_data.py  
**Purpose:** Reset or wipe data from the Weaviate instance, useful for re-ingesting fresh content.

**How it works:**
- Parses `--classes`: comma-separated list of class names to delete.
- If no classes specified, deletes **all** objects from the cluster.
- Uses `weaviate-client` to:
  - Iterate over objects in each class.
  - Call the delete API for each object or purge the entire class.
- Confirms deletion and logs counts removed.

**Usage:**
```bash
python delete_cluster_data.py   --classes DocumentChunk,Image
```
> Omitting `--classes` will remove every object in all classes.

---

## 🔧 Installation

1. **Clone** & enter the repo:
   ```bash
   git clone https://github.com/your-username/research-scify.git
   cd research-scify
   ```

2. **Create** & activate a Python 3.9+ virtualenv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate    # Windows
   ```

3. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up** your environment variables in a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_key
   WEAVIATE_URL=https://your-weaviate-instance.com
   WEAVIATE_API_KEY=your_weaviate_api_key
   ```

---

## 🛠 Contributing & License

- See [CONTRIBUTING.md] for guidelines.  
- MIT © 2025 Your Name
