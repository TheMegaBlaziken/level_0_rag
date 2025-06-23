# Research SciFy RAG Pipeline

A Retrieval‑Augmented Generation (RAG) pipeline that:

1. **Downloads** arXiv PDFs by ID or search query.  
2. **Converts** PDFs to Markdown with extracted figures and AI‑generated captions.  
3. **Ingests** Markdown text chunks and image embeddings into a Weaviate vector database.  
4. **Provides** an interactive RAG chatbot for querying the indexed content.  
5. **Resets** your Weaviate data with a one‑click utility.

---

## 📁 Repo Structure

```
.
├── download_arxiv_pdfs.py      # Fetch PDFs from arXiv
├── pdf_to_markdown.py         # Convert PDFs → Markdown + captions
├── ingest_markdown.py         # Chunk & ingest Markdown + images into Weaviate
├── rag_chat.py                # Interactive RAG chatbot interface
└── delete_cluster_data.py     # Clear all schema & data from Weaviate
```

---

## Prerequisites

- **Python 3.9+**  
- **Environment Variables** (required before running):
  - `OPENAI_API_KEY` – OpenAI API key for captioning and embeddings  
  - `WEAVIATE_URL` – URL of your Weaviate instance  
  - `WEAVIATE_API_KEY` – API key for Weaviate (if applicable)  
  - `GITHUB_TOKEN` – GitHub Personal Access Token for uploading extracted images  
  - `GITHUB_REPO` – GitHub repo in `owner/name` form for image hosting  

Install dependencies (using your activated virtual environment):

```bash
pip install -r requirements.txt
```

---

## Scripts & Usage

### 1. `download_arxiv_pdfs.py`

**Description:** Download PDF files from arXiv by explicit IDs or by search term.

**Arguments:**

- `--ids, -i` &nbsp;Comma‑separated arXiv IDs (e.g., `2301.00001,2105.12345`)  
- `--query, -q` &nbsp;Search query string (e.g., `"graph neural networks"`)  
- `--max-results, -m` &nbsp;(with `--query`) number of papers to fetch (default: 10)  
- `--output, -o` &nbsp;Output directory for downloaded PDFs  

**Examples:**

```bash
# By IDs
python download_arxiv_pdfs.py -i 2301.00001,2105.12345 -o ./pdfs

# By search
python download_arxiv_pdfs.py -q "machine learning" -m 20 -o ./pdfs
```

---

### 2. `pdf_to_markdown.py`

**Description:** Convert PDFs into Markdown, extract figures, and generate OpenAI-powered captions.

**Arguments:**

- `--input, -i` &nbsp;Path to a single PDF or a directory of PDFs  
- `--output, -o` &nbsp;Path to a single Markdown file or a directory for output `.md` files  
- `--use-llm` &nbsp;Optional flag: enable LLM-enhanced parsing mode  

**Examples:**

```bash
# Single PDF → single Markdown
python pdf_to_markdown.py -i paper.pdf -o paper.md

# Folder of PDFs → folder of Markdown files (with images in subfolders)
python pdf_to_markdown.py -i ./pdfs -o ./markdowns --use-llm
```

Output:

- `<base>.md` files with text and image links  
- `<base>_images/` directories with extracted figure images  
- Captions inserted under each image in Markdown as `**Caption:** ...`

---

### 3. `ingest_markdown.py`

**Description:** Split Markdown into chunks, compute embeddings, and ingest text chunks and images into Weaviate.

**Arguments:**

- `--input, -i` &nbsp;Directory containing `.md` files and their corresponding `_images/` folders  

**Behavior:**

1. Reads each Markdown file and its extracted images.  
2. Splits text into ~1000‑word sections; embeds with OpenAI embeddings.  
3. Uploads text sections as `DocumentChunk` objects (properties: `text`, `paper_id`, `heading`).  
4. Computes image+caption embeddings (CLIP fusion), uploads image files to the specified GitHub repo, and stores as `PaperImage` objects (properties: `filename`, `paper_id`, `caption`, `url`).  

**Example:**

```bash
python ingest_markdown.py -i ./markdowns
```

---

### 4. `rag_chat.py`

**Description:** Interactive RAG chatbot that retrieves relevant content (text + images) and answers queries.

**Usage:**

```bash
python rag_chat.py
```

**Flow:**

1. Prompts `Q>` for user query.  
2. Embeds query and retrieves top candidates from `DocumentChunk` & `PaperImage`.  
3. (If available) Reranks with a cross-encoder for better relevance.  
4. Calls `openai.ChatCompletion` with retrieved context, prints answer with `[Source: <paper_id>]` citations.  
5. Displays up to 3 most relevant figure URLs.  

Type `exit` or `quit` to end the session.

---

### 5. `delete_cluster_data.py`

**Description:** Wipe your entire Weaviate schema and data for a fresh start.

**Usage:**

```bash
python delete_cluster_data.py
```

Be cautious: this uses `client.schema.delete_all()` and will permanently remove all classes and objects.

---

## Requirements

All core dependencies are listed in `requirements.txt`. Review and pin versions as needed.

---

## License

MIT © 2025 Your Name
