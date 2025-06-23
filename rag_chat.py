#!/usr/bin/env python3
"""
rag_chat.py

A Retrieval-Augmented Generation chat loop using Weaviate:
- Fetch text chunks via vector search
- Fetch images via CLIP-text query
- Rerank both text and image candidates with a late-interaction cross-encoder
- Refine image selection by requiring high relevance to both query and generated answer
- Display only genuinely relevant, deduplicated figures
- Pass top-K text chunks into an OpenAI ChatCompletion with concise, cited responses
- Supports 'exit'/'quit' or Ctrl+C/EOF to end the session
"""
import os
import weaviate
from openai import OpenAI

# Transformers support
torch_available = True
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        CLIPProcessor,
        CLIPModel,
    )
except ImportError:
    torch_available = False

# Configurable constants
RETRIEVE_TEXT_K     = 10   # initial text chunks
FINAL_TEXT_K        = 5    # after reranking
IMAGE_TOP_K         = 10   # initial image fetch
IMAGE_REL_THRESHOLD = 0.8  # fraction of top reranker score
MAX_IMAGES          = 3    # max figures to show
MAX_RESPONSE_TOKENS = 300  # LLM cap

# Environment
env = os.environ
WEAVIATE_URL     = env.get('WEAVIATE_URL')
WEAVIATE_API_KEY = env.get('WEAVIATE_API_KEY')
OPENAI_API_KEY   = env.get('OPENAI_API_KEY')

# Init clients
def init_clients():
    auth = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth)
    oa = OpenAI(api_key=OPENAI_API_KEY)
    reranker = None
    if torch_available:
        tok = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2', use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        model.eval()
        reranker = (tok, model)
    return client, oa, reranker

# Globals
openai_client = None
clip_processor = None
clip_model     = None

# Embedding helpers
def embed_text(text: str):
    resp = openai_client.embeddings.create(model='text-embedding-ada-002', input=text)
    return resp.data[0].embedding

def embed_clip_text(text: str):
    inputs = clip_processor(text=[text], images=None, return_tensors='pt', padding=True)
    feats = clip_model.get_text_features(**{k: inputs[k] for k in ['input_ids','attention_mask']}).detach()
    return feats[0].cpu().numpy().tolist()

# Retrieve text and images
def retrieve(client, query: str):
    texts = client.query.get('DocumentChunk', ['text','paper']) \
        .with_near_vector({'vector': embed_text(query)}) \
        .with_limit(RETRIEVE_TEXT_K).do()["data"]["Get"]["DocumentChunk"]
    imgs = client.query.get('PaperImage', ['image_url','paper','caption']) \
        .with_near_vector({'vector': embed_clip_text(query)}) \
        .with_additional('certainty') \
        .with_limit(IMAGE_TOP_K).do()["data"]["Get"]["PaperImage"]
    return texts, imgs

# Cross-encoder reranking
def rerank(query: str, candidates: list, reranker):
    tok, model = reranker
    texts = [snippet if snippet else item.get('caption','') for snippet,item,is_img,conf in candidates]
    inputs = tok([query]*len(texts), texts,
                 padding=True, truncation=True, max_length=512,
                 return_tensors='pt')
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1).cpu().tolist()
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [(snippet,item,is_img,conf,score) for score,(snippet,item,is_img,conf) in scored]

# Main loop
if __name__ == '__main__':
    client, oa, reranker = init_clients()
    openai_client = oa
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    clip_model.eval()

    print("RAG Chat ready. Type 'exit' or Ctrl+C to quit.")
    while True:
        try:
            query = input('Q> ')
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break
        if query.strip().lower() in ('exit','quit','q'):
            print('Goodbye!')
            break

        # 1) Retrieve candidates
        text_hits, img_hits_weav = retrieve(client, query)
        mixed = []
        for t in text_hits:
            mixed.append((f"{t['text']} [Source: {t['paper']}]", t, False, None))
        for im in img_hits_weav:
            mixed.append((None, im, True, im.get('_additional',{}).get('certainty',0)))

        # 2) Rerank via query
        enriched = rerank(query, mixed, reranker) if reranker else [(sn, it, img, conf, 0.0) for (sn,it,img,conf) in mixed]

        # 3) Select unique top images by query-based rerank score
        img_scores = [(it['image_url'], score) for (sn,it,is_img,conf,score) in enriched if is_img]
        display_images = []
        if img_scores:
            max_score = max(score for _, score in img_scores)
            threshold = IMAGE_REL_THRESHOLD * max_score
            for url, score in sorted(img_scores, key=lambda x: x[1], reverse=True):
                if score >= threshold and url not in display_images:
                    display_images.append(url)
                    if len(display_images) >= MAX_IMAGES:
                        break

        # 4) Generate answer
        contexts = [sn for (sn,_,is_img,_,_) in enriched if not is_img][:FINAL_TEXT_K]
        prompt = (
            'You are a concise assistant that cites sources. Use snippets to answer and cite [Source: PAPER_ID]:\n\n'
            + '\n---\n'.join(contexts)
            + f"\n\nUser Query: {query}"
        )
        resp = oa.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role':'system','content':'Provide a brief answer.'}, {'role':'user','content':prompt}],
            max_tokens=MAX_RESPONSE_TOKENS
        )
        answer = resp.choices[0].message.content
        print(answer)

        # 5) Refine via answer-based rerank with dedupe
        display_final = display_images.copy()
        if reranker and display_images:
            ans_candidates = [(None, it, True, None) for (sn,it,is_img,conf,score) in enriched if is_img]
            enriched_ans = rerank(answer, ans_candidates, reranker)
            ans_scores = [(it['image_url'], score) for (sn,it,is_img,conf,score) in enriched_ans]
            if ans_scores:
                max_ans = max(score for _, score in ans_scores)
                ans_thresh = IMAGE_REL_THRESHOLD * max_ans
                recall = []
                for url in display_images:
                    score = dict(ans_scores).get(url, 0)
                    if score >= ans_thresh:
                        recall.append(url)
                display_final = recall

        # 6) Display final figures
        if display_final:
            print('\n=== Relevant Figures ===')
            for i, url in enumerate(display_final, 1):
                print(f'![Figure {i}]')
                print(f'({url})')
            print('========================\n')
