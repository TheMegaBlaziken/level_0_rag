#!/usr/bin/env python3
"""
download_arxiv_pdfs.py

A utility to fetch PDF files from arXiv automatically, given either:
  - A list of arXiv IDs
  - A search query string

Dependencies:
    pip install arxiv requests

Usage:
    # Download by ID list:
    python download_arxiv_pdfs.py --ids 2301.00001,2105.12345 --output ./pdfs

    # Download by search term:
    python download_arxiv_pdfs.py --query "machine learning" --max-results 20 --output ./pdfs
"""

import argparse
import os
import arxiv
import requests

PDF_URL = "https://arxiv.org/pdf/{id}.pdf"


def download_by_id(arxiv_id: str, out_dir: str):
    """Fetch a single PDF from arXiv by its ID."""
    url = PDF_URL.format(id=arxiv_id)
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        # Sanitize ID for filename (replace any slashes) and ensure directory exists
        safe_id = arxiv_id.replace('/', '_')
        filename = f"{safe_id}.pdf"
        path = os.path.join(out_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded {arxiv_id} -> {path}")
    else:
        print(f"[ERROR] Failed to download {arxiv_id}: HTTP {resp.status_code}")


def download_by_query(query: str, max_results: int, out_dir: str):
    """Search arXiv and download PDFs for the top results."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    for result in search.results():
        aid = result.get_short_id()
        download_by_id(aid, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Download PDFs from arXiv automatically.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ids', '-i',
                       help="Comma-separated arXiv IDs (e.g. 2301.00001,2105.12345)")
    group.add_argument('--query', '-q',
                       help="Search query for arXiv",
                       metavar="QUERY")
    parser.add_argument('--max-results', '-m', type=int, default=10,
                        help="Max papers to fetch when using --query")
    parser.add_argument('--output', '-o', required=True,
                        help="Directory to save PDFs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.ids:
        for aid in args.ids.split(','):
            download_by_id(aid.strip(), args.output)
    else:
        download_by_query(args.query, args.max_results, args.output)


if __name__ == '__main__':
    main()
