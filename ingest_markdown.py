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
      OPENAI_API_KEY       Your OpenAI API Key
      GITHUB_TOKEN         A GitHub personal access token with repo scopes
      GITHUB_REPO          The target repo in "owner/repo" format
      GITHUB_BRANCH        The branch to commit images to (default: main)
      GITHUB_PATH_PREFIX   Optional path prefix in the repo for images (e.g. "papers")

Usage:
    export WEAVIATE_URL=https://YOUR-CLOUD-ID.weaviate.network
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
import weaviate
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI

# Chunk size in words
CHUNK_SIZE = 1000

# Global clients (initialized in main)
openai_client = None
clip_processor = None
clip_model = None

# Load configuration from environment
env = os.environ
WEAVIATE_URL       = env.get('WEAVIATE_URL')
OPENAI_API_KEY     = env.get('OPENAI_API_KEY')
GITHUB_TOKEN       = env.get('GITHUB_TOKEN')
GITHUB_REPO        = env.get('GITHUB_REPO')
GITHUB_BRANCH      = env.get('GITHUB_BRANCH', 'main')
GITHUB_PATH_PREFIX = env.get('GITHUB_PATH_PREFIX', '').strip('/')

# Validate required environment variables
def validate_config():
    missing = []
    for var in ('WEAVIATE_URL', 'OPENAI_API_KEY', 'GITHUB_TOKEN', 'GITHUB_REPO'):
        if not env.get(var):
            missing.append(var)
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

# Initialize Weaviate client and schema
def init_weaviate_client():
    from weaviate import Client
    from weaviate.auth import AuthApiKey, AuthClientPassword

    url = WEAVIATE_URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    auth = None
    api_key = env.get('WEAVIATE_API_KEY')
    if api_key:
        auth = AuthApiKey(api_key=api_key)
    else:
        user = env.get('WEAVIATE_USERNAME'); pwd = env.get('WEAVIATE_PASSWORD')
        if user and pwd:
            auth = AuthClientPassword(username=user, password=pwd)

    client = Client(url=url, auth_client_secret=auth) if auth else Client(url=url)
    schema = client.schema.get()
    classes = [c['class'] for c in schema.get('classes', [])]

    # Document chunks (1536-D embeddings)
    if 'DocumentChunk' not in classes:
        print("[INFO] Creating Weaviate class DocumentChunk")
        client.schema.create_class({
            'class': 'DocumentChunk',
            'properties': [
                {'name': 'text',    'dataType': ['text']},
                {'name': 'paper',   'dataType': ['string']},
                {'name': 'heading', 'dataType': ['string']}
            ],
            'vectorizer': 'none'
        })

    # PaperImage (512-D fused embeddings + URL)
    if 'PaperImage' not in classes:
        print("[INFO] Creating Weaviate class PaperImage")
        client.schema.create_class({
            'class': 'PaperImage',
            'properties': [
                {'name': 'filename',  'dataType': ['string']},
                {'name': 'paper',     'dataType': ['string']},
                {'name': 'caption',   'dataType': ['text']},
                {'name': 'image_url', 'dataType': ['string']}
            ],
            'vectorizer': 'none'
        })

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
        body = parts[i+1].strip()
        if body:
            yield heading, body

# Chunk text into size-limited pieces
def chunk_text(text: str, size: int = CHUNK_SIZE):
    words = text.split()
    for i in range(0, len(words), size):
        yield ' '.join(words[i:i+size])

# Ingest markdown sections, with captions filtered out
def ingest_markdown_file(client, md_text: str, source: str):
    print(f"[INFO] Ingesting text for paper {source}")
    md_text = re.sub(r"!\[[^\]]*\]\([^)]*\)\s*\n\s*\*\*Caption:\*\*.+", "", md_text)
    paper_id = os.path.splitext(source)[0]
    for heading, section in split_into_sections(md_text):
        for chunk in chunk_text(section):
            resp = openai_client.embeddings.create(
                model='text-embedding-ada-002', input=chunk
            )
            vector = resp.data[0].embedding
            client.data_object.create(
                data_object={
                    'text':    chunk,
                    'paper':   paper_id,
                    'heading': heading
                },
                class_name='DocumentChunk',
                vector=vector
            )

# Ingest images: fuse caption+visual and push with GitHub upload
def ingest_images_for_paper(client, md_path: str, source: str):
    print(f"[INFO] Ingesting images for paper {source}")
    paper_id = os.path.splitext(source)[0]
    base = os.path.splitext(md_path)[0]
    img_dir = f"{base}_images"
    md_text = load_markdown(md_path)
    pattern = r"!\[[^\]]*\]\(([^)]+)\)\s*\n\s*\*\*Caption:\*\*\s*(.+)"

    for fname, caption in re.findall(pattern, md_text):
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            print(f"[WARN] Image file not found: {img_path}")
            continue

        # Fuse features
        image = Image.open(img_path).convert("RGB")
        # Truncate long captions to model max length
        max_len = clip_processor.tokenizer.model_max_length
        inputs = clip_processor(
            text=[caption],
            images=image,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_feats  = clip_model.get_text_features(**{k: inputs[k] for k in ['input_ids','attention_mask']})
            image_feats = clip_model.get_image_features(**{k: inputs[k] for k in ['pixel_values']})
        fused = ((text_feats + image_feats) / 2).detach()[0].cpu().numpy().tolist()

        # Determine GitHub path and raw URL
        rel_path = f"{GITHUB_PATH_PREFIX}/{paper_id}_images/{fname}" if GITHUB_PATH_PREFIX else f"{paper_id}_images/{fname}"
        raw_url  = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{rel_path}"
        api_url   = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{rel_path}?ref={GITHUB_BRANCH}"
        headers   = {'Authorization': f'token {GITHUB_TOKEN}'}

        # Check existence
        try:
            resp = requests.get(api_url, headers=headers, timeout=10)
        except Exception as e:
            print(f"[ERROR] Checking GitHub for {rel_path} failed: {e}")
            continue

        if resp.status_code == 404:
            print(f"[INFO] Uploading image {rel_path} to GitHub...")
            with open(img_path, 'rb') as f:
                data = f.read()
            content_b64 = base64.b64encode(data).decode()
            put_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{rel_path}"
            payload = {'message': f'Add image {rel_path}', 'content': content_b64, 'branch': GITHUB_BRANCH}
            r2 = requests.put(put_url, json=payload, headers=headers, timeout=10)
            if r2.ok:
                print(f"[INFO] Uploaded {rel_path} -> {raw_url}")
            else:
                print(f"[WARN] GitHub upload failed for {rel_path}: {r2.status_code} {r2.text}")
        else:
            print(f"[INFO] Image {rel_path} already exists, skipping upload")

        # Ingest into Weaviate
        client.data_object.create(
            data_object={
                'filename':  fname,
                'paper':     paper_id,
                'caption':   caption,
                'image_url': raw_url
            },
            class_name='PaperImage',
            vector=fused
        )

# Process input file or directory
def process_input(input_path: str, client):
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in sorted(os.listdir(input_path)) if f.lower().endswith('.md')]
    else:
        files = [input_path]

    for md_path in files:
        source = os.path.basename(md_path)
        ingest_markdown_file(client, load_markdown(md_path), source)
        ingest_images_for_paper(client, md_path, source)
        print(f"[INFO] Ingested '{source}' (text + images)")

# Main entrypoint
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest Markdown + images into Weaviate')
    parser.add_argument('--input','-i',required=True,help='Markdown file or directory')
    args = parser.parse_args()

    validate_config()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    client = init_weaviate_client()
    process_input(args.input, client)
