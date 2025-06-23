#!/usr/bin/env python3
"""
pdf_to_markdown.py

A script that reads a PDF or directory of PDFs, extracts structured content using Marker,
writes Markdown files with embedded images, and appends AI-generated captions for all images
using the OpenAI Chat Completions API.

Usage:
    pip install marker-pdf openai pillow
    export OPENAI_API_KEY=your_key_here

    # Single PDF
    python pdf_to_markdown.py --input paper.pdf --output paper.md [--use-llm]

    # Directory of PDFs
    python pdf_to_markdown.py --input pdf_folder --output md_folder [--use-llm]
"""
import argparse
import os
import re
import json
from openai import OpenAI
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from PIL import Image

# Model configuration
def default_caption_model():
    """Return the OpenAI model name for captioning."""
    return "gpt-4o-mini"

# System prompt for captioning
SYSTEM_PROMPT = (
    "You are an expert in analyzing scientific figures and generating accurate, detailed captions. "
    "Please produce captions close to 70 tokens that describe what the figure shows (data & visualization), "
    "key trends, experimental parameters, and scientific significance."
)

# Initialize OpenAI client
oai = None

def init_openai():
    global oai
    if oai is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        oai = OpenAI(api_key=api_key)
    return oai


def generate_captions(prompt: str) -> dict:
    """Call the OpenAI Chat API to generate captions (stripping any Markdown fences)."""
    client = init_openai()
    resp = client.chat.completions.create(
        model=default_caption_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + " Please output only raw JSON without any markdown fences or extra text."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.1
    )
    text = resp.choices[0].message.content.strip()
    # Strip Markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def pdf_to_markdown(input_path: str, output_path: str, use_llm: bool = False):
    base, _ = os.path.splitext(output_path)
    img_dir = f"{base}_images"

    # Skip if already processed
    if os.path.exists(output_path) and os.path.isdir(img_dir):
        print(f"[SKIP] Existing Markdown & images found: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        images = [fn for fn in os.listdir(img_dir) if not fn.startswith('.')]
    else:
        # Convert PDF to markdown + images
        artifacts = create_model_dict()
        converter = PdfConverter(artifact_dict=artifacts)
        if use_llm:
            try:
                converter.enable_llm()
            except AttributeError:
                print("[WARN] LLM mode not supported.")
        rendered = converter(input_path)
        md_text, _, img_map = text_from_rendered(rendered)

        # Write markdown
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_text)
        print(f"[INFO] Markdown written: {output_path}")

        # Save images
        os.makedirs(img_dir, exist_ok=True)
        for name, data in img_map.items():
            path = os.path.join(img_dir, name)
            if isinstance(data, (bytes, bytearray)):
                with open(path, 'wb') as out:
                    out.write(data)
            elif Image and isinstance(data, Image.Image):
                data.save(path)
        images = list(img_map.keys())
        print(f"[INFO] Saved {len(images)} images to {img_dir}")

    # Generate captions
    if not images:
        return
    image_list = ', '.join(images)
    prompt = (
        "Generate a JSON object mapping each image filename to a concise scientific caption. "
        f"Images: [{image_list}]. Full Markdown content follows:\n```markdown"
        + md_text + "\n```"
    )
    try:
        captions = generate_captions(prompt)
    except Exception as e:
        print(f"[ERROR] Caption generation failed: {e}")
        return

    # Insert captions
    updated = md_text
    for name, cap in captions.items():
        updated = re.sub(
            rf"(!\[[^\]]*\]\({re.escape(name)}\))",
            rf"\1\n\n**Caption:** {cap}",
            updated
        )
        print(f"[INFO] Caption for {name}: {cap}")

    # Save updated markdown
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(updated)
    print(f"[INFO] Captions appended: {output_path}")


def process_directory(input_dir: str, output_dir: str, use_llm: bool):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith('.pdf'):
            pdf_to_markdown(
                os.path.join(input_dir, fname),
                os.path.join(output_dir, fname.replace('.pdf', '.md')),
                use_llm
            )


def main():
    parser = argparse.ArgumentParser(description="Convert PDF(s) to Markdown & caption figures.")
    parser.add_argument('--input','-i',required=True, help='PDF file or directory')
    parser.add_argument('--output','-o',required=True, help='Markdown file or directory')
    parser.add_argument('--use-llm',action='store_true', help='Enable LLM-enhanced conversion')
    args = parser.parse_args()
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.use_llm)
    else:
        pdf_to_markdown(args.input, args.output, args.use_llm)

if __name__ == '__main__':
    main()
