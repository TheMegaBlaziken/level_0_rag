# Research SciFy RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for scientific papers, built with **Streamlit**, **Weaviate**, **OpenAI**, and **FlagReranker**. The system supports rich metadata, robust reranking, and a modern debug UI for transparency and troubleshooting.

## Features

- **Streamlit UI**: Modern, user-friendly interface with answer, debug, and progress tabs.
- **Weaviate Vector Database**: Stores paper chunks, metadata, and images for fast semantic and hybrid search.
- **OpenAI LLM**: Generates answers with inline citations, using only provided context.
- **FlagReranker**: Reranks retrieved chunks for relevance using BAAI/bge-reranker-large.
- **Rich Metadata**: Each chunk stores paper ID, title, section heading, and more for smarter retrieval and reranking.
- **Image Handling**: Relevant images are scored and displayed with answers; image metadata is stored and searchable.
- **Debug Tab**: Shows retrieval, reranking, LLM prompt/response, image scoring, and errors for transparency.
- **Subject-Aware Boosting**: Chunks from papers/sections matching the query subject are boosted for better relevance.

## Setup

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2. (Recommended) Create and Activate a Conda Environment
```sh
conda create -n rag-chatbot python=3.9
conda activate rag-chatbot
```

### 3. Install Dependencies
Install Python 3.9+ and run:
```sh
pip install -r requirements.txt
```

### 4. Environment Variables
Set the following environment variables (e.g., in a `.env` file):
```
WEAVIATE_URL=your-weaviate-url
WEAVIATE_API_KEY=your-weaviate-api-key
OPENAI_API_KEY=your-openai-api-key
GITHUB_TOKEN=your-github-token
GITHUB_REPO=yourusername/your-repo
GITHUB_BRANCH=main
GITHUB_PATH_PREFIX=papers
```

### 5. Ingest Papers
Use the ingestion script to process markdown files and images:
```sh
python ingest_markdown.py --input path/to/markdowns
```
- This will split papers into chunks, extract metadata, embed with OpenAI, and upload to Weaviate.
- Images are uploaded to GitHub and indexed in Weaviate with CLIP embeddings.

### 6. Run the Chatbot
```sh
streamlit run app.py
```
- Open the provided local URL in your browser.
- Ask questions about your paper collection!

## Main Files
- `app.py`: Streamlit UI and main app logic.
- `rag_chat.py`: RAG pipeline, retrieval, reranking, LLM, and debug logic.
- `ingest_markdown.py`: Ingestion pipeline for markdowns and images.
- `requirements.txt`: Python dependencies.

## Troubleshooting
- **Weaviate connection errors**: Check your `WEAVIATE_URL` and API key.
- **OpenAI errors**: Ensure your OpenAI API key is valid and has quota.
- **Image upload issues**: Make sure your GitHub token has repo access.
- **Dependency issues**: Reinstall with `pip install -r requirements.txt`.
- **Schema changes**: If you change metadata fields, re-ingest your data.

## Extending
- Add new metadata fields in `ingest_markdown.py` and update queries in `rag_chat.py`.
- Tune subject boosting or reranker logic for your use case.
- Add new UI features in `app.py` as needed.

## License
MIT (or your chosen license)
