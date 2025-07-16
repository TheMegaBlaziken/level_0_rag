#!/usr/bin/env python3
"""
ingest_markdown.py

A script to read Markdown file(s), split content by headings into sections,
chunk each section into 1000-word pieces, embed using OpenAI,
and ingest into a Weaviate Cloud instance with rich metadata, including:
  - paper ID (from filename)
  - section headings
  - input text chunks
  - images with fused caption+visual embeddings and GitHub-hosted URL links

Configuration:
    Ensure the following environment variables are set:
      WEAVIATE_URL         Your Weaviate Cloud REST URL
      WEAVIATE_API_KEY     Your Weaviate Cloud Admin API Key
      OPENAI_API_KEY       Your OpenAI API Key
      GITHUB_TOKEN         A GitHub personal access token with repo scopes
      GITHUB_REPO          The target repo in "owner/repo" format
      GITHUB_BRANCH        The branch to commit images to (default: main)
      GITHUB_PATH_PREFIX   Optional path prefix in the repo for images (e.g. "papers")

Usage:
    export WEAVIATE_URL=https://YOUR-CLOUD-ID.weaviate.network
    export WEAVIATE_API_KEY=YOUR_WEAVIATE_API_KEY
    export OPENAI_API_KEY=YOUR_OPENAI_KEY
    export GITHUB_TOKEN=YOUR_GITHUB_TOKEN
    export GITHUB_REPO=username/repo
    # Optional:
    export GITHUB_BRANCH=main
    export GITHUB_PATH_PREFIX=papers

    python ingest_markdown.py --input path/to/file_or_directory
"""
import argparse
import os
import re
import sys
import base64
import requests
import uuid
import weaviate
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import weaviate.classes.config as wc
from weaviate.connect import ConnectionParams
import nltk
import time
from datetime import datetime
import yaml
from dotenv import load_dotenv

# find .env and load all the variables into os.environ
load_dotenv()

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
import weaviate.classes.query as wvcq

# Chunk size in words
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 50
MIN_CHUNK_SIZE = 100

# Global clients (initialized in main)
openai_client = None
clip_processor = None
clip_model = None

# Load configuration from environment
env = os.environ
WEAVIATE_URL       = env.get('WEAVIATE_URL')
WEAVIATE_API_KEY   = env.get('WEAVIATE_API_KEY')
OPENAI_API_KEY     = env.get('OPENAI_API_KEY')
GITHUB_TOKEN       = env.get('GITHUB_TOKEN')
GITHUB_REPO        = env.get('GITHUB_REPO')
GITHUB_BRANCH      = env.get('GITHUB_BRANCH', 'main')
GITHUB_PATH_PREFIX = env.get('GITHUB_PATH_PREFIX', '').strip('/')

print("WEAVIATE_URL =", WEAVIATE_URL)

# Validate required environment variables
def validate_config():
    missing = []
    for var in ('WEAVIATE_URL','WEAVIATE_API_KEY','OPENAI_API_KEY','GITHUB_TOKEN','GITHUB_REPO'):
        if not env.get(var):
            missing.append(var)
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

# Initialize Weaviate client and collections (v4 API)
def init_weaviate_client():
    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_url("http://localhost:8080", grpc_port=50051)
    )
    client.connect()

    # Create Paper collection if missing
    if not client.collections.exists('Paper'):
        print('[INFO] Creating Weaviate collection: Paper')
        client.collections.create(
            name               = 'Paper',
            vectorizer_config  = wc.Configure.Vectorizer.text2vec_openai(),
            properties         = [
                wc.Property(name='paper_id', data_type=wc.DataType.TEXT, description='paper ID'),
                wc.Property(name='title',    data_type=wc.DataType.TEXT, description='paper title'),
                wc.Property(name='authors',  data_type=wc.DataType.TEXT, description='authors'),
                wc.Property(name='year',     data_type=wc.DataType.INT,  description='year'),
                wc.Property(name='source_file', data_type=wc.DataType.TEXT, description='source filename'),
                wc.Property(name='abstract', data_type=wc.DataType.TEXT, description='abstract'),
            ]
        )

    # Create DocumentChunk collection if missing
    if not client.collections.exists('DocumentChunk'):
        print('[INFO] Creating Weaviate collection: DocumentChunk')
        client.collections.create(
            name               = 'DocumentChunk',
            vectorizer_config  = wc.Configure.Vectorizer.text2vec_openai(),
            properties         = [
                wc.Property(name='text',    data_type=wc.DataType.TEXT, description='chunk text'),
                wc.Property(name='paper',   data_type=wc.DataType.TEXT, description='paper ID'),
                wc.Property(name='heading', data_type=wc.DataType.TEXT, description='section heading'),
                wc.Property(name='paper_title', data_type=wc.DataType.TEXT, description='paper title'),
            ]
        )

    # Create PaperImage collection if missing
    if not client.collections.exists('PaperImage'):
        print('[INFO] Creating Weaviate collection: PaperImage')
        client.collections.create(
            name               = 'PaperImage',
            vectorizer_config  = wc.Configure.Vectorizer.text2vec_openai(),
            properties         = [
                wc.Property(name='filename', data_type=wc.DataType.TEXT, description='image filename'),
                wc.Property(name='paper',    data_type=wc.DataType.TEXT, description='paper ID'),
                wc.Property(name='caption',  data_type=wc.DataType.TEXT, description='image caption'),
                wc.Property(name='image_url',data_type=wc.DataType.TEXT, description='raw GitHub URL'),
            ]
        )

    return client

# Read markdown content
def load_markdown(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Split markdown into (heading, section_body)
def split_into_sections(md: str):
    parts = re.split(r'(?m)^(#{1,6}\s+.+)$', md)
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body    = parts[i+1].strip()
        if body:
            yield heading, body

# Clean up markdown artifacts (LaTeX, HTML tags)
def clean_text(text):
    # Remove LaTeX math
    text = re.sub(r'\$.*?\$', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Sentence-aware chunking with overlap
def sentence_chunker(text, max_words=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = []
    count = 0
    for sent in sentences:
        words = sent.split()
        if count + len(words) > max_words:
            if len(current) >= MIN_CHUNK_SIZE:
                chunks.append(' '.join(current))
            # Overlap: keep last N words
            current = current[-overlap:] if overlap > 0 else []
            count = len(current)
        current.extend(words)
        count += len(words)
    if len(current) >= MIN_CHUNK_SIZE:
        chunks.append(' '.join(current))
    return chunks

# Chunk text into size-limited pieces
def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE):
    words = text.split()
    for i in range(0, len(words), size):
        yield ' '.join(words[i:i+size])

# Extract paper-level metadata from markdown (YAML frontmatter or first lines)
def extract_paper_metadata(md_text, source):
    import re
    lines = md_text.splitlines()
    # Title: first non-empty line starting with '#'
    title = None
    for line in lines:
        if line.strip().startswith('#'):
            title = line.strip('# ').strip()
            break
    if not title:
        title = os.path.splitext(source)[0]
    # Authors: look for names/emails in first 20 lines
    authors = None
    author_lines = []
    for line in lines[:20]:
        # Look for lines with emails or multiple names
        if re.search(r'@|[A-Z][a-z]+,? [A-Z][a-z]+', line):
            author_lines.append(line.strip())
    if author_lines:
        authors = ' '.join(author_lines)
    # Year: search whole file for 4-digit year (prefer near top)
    year = None
    for line in lines[:40]:
        m = re.search(r'(19|20)\d{2}', line)
        if m:
            year = int(m.group())
            break
    if not year:
        m = re.search(r'(19|20)\d{2}', md_text)
        if m:
            year = int(m.group())
    # Abstract: find line with 'abstract', then capture all lines after until next heading or blank line
    abstract = None
    for i, line in enumerate(lines):
        if 'abstract' in line.lower():
            abstract_lines = []
            for l in lines[i+1:]:
                if l.strip().startswith('#') or l.strip() == '':
                    break
                abstract_lines.append(l.strip())
            if abstract_lines:
                abstract = ' '.join(abstract_lines)
            break
    return {
        'paper_id': os.path.splitext(source)[0],
        'title': title,
        'authors': authors,
        'year': year,
        'source_file': source,
        'abstract': abstract
    }

# Ingest markdown sections (text) into Weaviate
def ingest_markdown_file(client, md_text: str, source: str, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP, log_file=None):
    print(f"[INFO] Ingesting text for paper {source}")
    md_text = re.sub(r"!\[[^]]*\]\([^)]*\)\s*\n\s*\*\*Caption:\*\*.+", "", md_text)
    paper_id = os.path.splitext(source)[0]
    # --- NEW: Insert Paper record ---
    paper_meta = extract_paper_metadata(md_text, source)
    paper_collection = client.collections.get("Paper")
    # Only insert if not already present
    results = paper_collection.query.fetch_objects(
        filters=wvcq.Filter.by_property("paper_id").equal(paper_id),
        limit=1
    )
    if not results.objects:
        print(f"[INFO] Inserting Paper record for {paper_id}")
        paper_collection.data.insert(properties=paper_meta, uuid=str(uuid.uuid4()))
    # --- END NEW ---
    seen_chunks = set()
    timestamp = datetime.utcnow().isoformat()
    for heading, section in split_into_sections(md_text):
        # Filter out low-value sections
        low_value_sections = [
            'author contribution', 'author contributions', 'acknowledgement', 'acknowledgements',
            'conflict of interest', 'conflicts of interest', 'funding', 'disclosure', 'ethics statement',
            'competing interests', 'data availability', 'supplementary material', 'appendix', 'references', 'bibliography'
        ]
        heading_lower = heading.lower()
        if any(lvs in heading_lower for lvs in low_value_sections):
            print(f"[INFO] Skipping low-value section: {heading}")
            continue
        if 'reference' in heading.lower() or 'bibliograph' in heading.lower():
            print(f"[INFO] Skipping section: {heading}")
            continue
        print(f"[DEBUG] Processing heading: {heading}")
        # Extract heading level (number of #)
        heading_level = len(re.match(r'#+', heading).group()) if re.match(r'#+', heading) else 0
        section_clean = clean_text(section)
        chunks = sentence_chunker(section_clean, max_words=chunk_size, overlap=overlap)
        buffer = ""
        for idx, chunk in enumerate(chunks):
            chunk_clean = clean_text(chunk)
            # Prepend buffer if exists
            if buffer:
                chunk_clean = buffer + " " + chunk_clean
                buffer = ""
            if len(chunk_clean.split()) < MIN_CHUNK_SIZE:
                print(f"[INFO] Buffering small chunk (words: {len(chunk_clean.split())}) to prepend to next chunk")
                buffer = chunk_clean
                continue
            if chunk_clean in seen_chunks:
                print(f"[INFO] Skipping duplicate chunk")
                continue
            seen_chunks.add(chunk_clean)
            try:
                resp = openai_client.embeddings.create(
                    model='text-embedding-ada-002', input=chunk_clean
                )
                vector = resp.data[0].embedding
                client.collections.get("DocumentChunk").data.insert(
                    properties={
                        'text': chunk_clean,
                        'paper': paper_id,
                        'heading': heading,
                        'heading_level': heading_level,
                        'chunk_index': idx,
                        'source_file': source,
                        'timestamp': timestamp,
                        'paper_title': paper_meta['title']
                    },
                    vector=vector,
                    uuid=str(uuid.uuid4())
                )
                print(f"[DEBUG] Inserted chunk for heading: {heading}")
            except Exception as e:
                print(f"[ERROR] Exception: {e}")
                if log_file:
                    with open(log_file, 'a') as lf:
                        lf.write(f"[ERROR] {source} heading: {heading} idx: {idx} error: {e}\n")
        # If buffer still has content after all chunks, skip it (or optionally append to last chunk if desired)

# Ingest images: fuse caption+visual and push with GitHub upload
def ingest_images_for_paper(client, md_path: str, source: str):
    print(f"[INFO] Ingesting images for paper {source}")
    paper_id = os.path.splitext(source)[0]
    base     = os.path.splitext(md_path)[0]
    img_dir  = f"{base}_images"
    md_text  = load_markdown(md_path)
    pattern  = r"!\[[^]]*\]\(([^)]+)\)\s*\n\s*\*\*Caption:\*\*\s*(.+)"

    for fname, caption in re.findall(pattern, md_text):
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            print(f"[WARN] Image file not found: {img_path}")
            continue

        # Compute CLIP features
        image   = Image.open(img_path).convert('RGB')
        max_len = clip_processor.tokenizer.model_max_length
        inputs  = clip_processor(
            text=[caption], images=image,
            padding=True, truncation=True, max_length=max_len,
            return_tensors='pt'
        )
        with torch.no_grad():
            text_feats  = clip_model.get_text_features(**{k: inputs[k] for k in ['input_ids','attention_mask']})
            image_feats = clip_model.get_image_features(**{k: inputs[k] for k in ['pixel_values']})
        fused = ((text_feats + image_feats) / 2).detach()[0].cpu().numpy().tolist()

        # GitHub upload
        rel_path = f"{GITHUB_PATH_PREFIX}/{paper_id}_images/{fname}" if GITHUB_PATH_PREFIX else f"{paper_id}_images/{fname}"
        raw_url  = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{rel_path}"
        api_url  = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{rel_path}?ref={GITHUB_BRANCH}"
        headers  = {'Authorization': f'token {GITHUB_TOKEN}'}

        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.status_code == 404:
            print(f"[INFO] Uploading image {rel_path}")
            with open(img_path, 'rb') as f:
                content_b64 = base64.b64encode(f.read()).decode()
            payload = {
                'message': f'Add image {rel_path}',
                'content': content_b64,
                'branch':  GITHUB_BRANCH
            }
            put_url = api_url.replace(f'?ref={GITHUB_BRANCH}', '')
            r2 = requests.put(put_url, json=payload, headers=headers, timeout=10)
            if r2.ok:
                print(f"[INFO] Uploaded {rel_path}")
            else:
                print(f"[WARN] GitHub upload failed ({r2.status_code}): {r2.text}")
        else:
            print(f"[INFO] Image {rel_path} already exists, skipping upload")

        # v4: insert into PaperImage via collections API
        client.collections.get("PaperImage").data.insert(
            properties={
                'filename': fname,
                'paper': paper_id,
                'caption': caption,
                'image_url': raw_url
            },
            vector=fused,
            uuid=str(uuid.uuid4())
        )

# Process input file or directory
def process_input(input_path: str, client, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP, log_file=None):
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in sorted(os.listdir(input_path)) if f.lower().endswith('.md')]
    else:
        files = [input_path]

    for md_path in files:
        source = os.path.basename(md_path)
        ingest_markdown_file(client, load_markdown(md_path), source, chunk_size=chunk_size, overlap=overlap, log_file=log_file)
        ingest_images_for_paper(client, md_path, source)
        print(f"[INFO] Ingested '{source}' (text + images)")

# Main entrypoint
def main():
    parser = argparse.ArgumentParser(description='Ingest Markdown + images into Weaviate')
    parser.add_argument('--input', '-i', required=True, help='Markdown file or directory')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE, help='Max chunk size in words (default: 1000)')
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP, help='Chunk overlap in words (default: 50)')
    parser.add_argument('--log-file', type=str, default=None, help='File to log errors to (optional)')
    args = parser.parse_args()

    validate_config()
    global openai_client, clip_processor, clip_model
    openai_client  = OpenAI(api_key=OPENAI_API_KEY)
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    clip_model.eval()
    client = init_weaviate_client()
    try:
        process_input(args.input, client, chunk_size=args.chunk_size, overlap=args.overlap, log_file=args.log_file)
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        if args.log_file:
            with open(args.log_file, 'a') as lf:
                lf.write(f"[ERROR] {args.input} error: {e}\n")
        raise
    client.close()

if __name__ == '__main__':
    main()
