#!/usr/bin/env python3
"""
rag_chat.py

Retrieval-Augmented Generation (RAG) chat client using Weaviate and OpenAI, with full debug logging:
 - Weaviate v4 Python client (GraphQL via REST)
 - Hybrid and fallback text search with detailed request/response logs
 - Optional HuggingFace Cross-Encoder reranking
 - OpenAI answer generation with inline clickable citations
 - Dual-stage CLIP-based image scoring: first by query, then by response
 - Section-level citation grouping
 - Returns retrieval logs and image debug logs for UI inspection
"""
import os
import re
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import weaviate
from weaviate.classes.init import Auth
import openai
import time
import json
from dotenv import load_dotenv

# find .env and load all the variables into os.environ
load_dotenv()

# Optional CLIP and reranker imports
torch = None
CLIPProcessor = None
CLIPModel = None
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    pass

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

FlagReranker = None
try:
    from FlagEmbedding import FlagReranker
except ImportError:
    pass



# Initialize the reranker (do this once, e.g., at the top of your script)
flag_reranker = None
reranker_status_msg = None
try:
    # Use the fine-tuned model if it exists, otherwise fall back to the original
    if os.path.exists("fine_tuned_reranker"):
        flag_reranker = FlagReranker('fine_tuned_reranker')
        reranker_status_msg = "✅ FlagReranker initialized successfully with fine-tuned model: fine_tuned_reranker"
    else:
        flag_reranker = FlagReranker('fine_tuned_reranker')
        reranker_status_msg = "✅ FlagReranker initialized successfully with original model: BAAI/bge-reranker-large"
except Exception as e:
    reranker_status_msg = f"❌ Warning: Could not initialize FlagReranker: {e}"
    flag_reranker = None

# ── Configuration ───────────────────────────────────────────────────────────
RETRIEVE_TEXT_K      = 20
RERANK_TEXT_K        = 5
RERANK_ALPHA         = 0.5
MAX_IMAGES           = 3
MAX_RESPONSE_TOKENS  = 1000
IMAGE_CONF_THRESHOLD = 0.5
HYBRID_ALPHA         = 0.5
# New configuration for simplified relevance filtering
MIN_RELEVANCE_SCORE  = -2.0  # or even lower, depending on your data
MAX_CHUNKS_PER_PAPER = 4    # Maximum chunks to use from any single paper

# Environment vars
WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
SCI_CLIP_MODEL   = os.getenv("SCI_CLIP_MODEL", "openai/clip-vit-base-patch32")

# Ensure WEAVIATE_URL includes a scheme
if WEAVIATE_URL and not WEAVIATE_URL.startswith(("http://", "https://")):
    WEAVIATE_URL = "https://" + WEAVIATE_URL

# ── Client initialization ───────────────────────────────────────────────────
def init_clients():
    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        raise RuntimeError("Set WEAVIATE_URL and WEAVIATE_API_KEY in env vars.")
    client = weaviate.connect_to_local(
        grpc_port=50051,
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
    )
    if not client.is_ready():
        raise RuntimeError(f"Cannot connect to Weaviate at {WEAVIATE_URL}")
    oa = openai.OpenAI(api_key=OPENAI_API_KEY)
    proc = None
    mdl_clip = None
    if torch:
        proc = CLIPProcessor.from_pretrained(SCI_CLIP_MODEL, use_fast=False)
        mdl_clip = CLIPModel.from_pretrained(SCI_CLIP_MODEL)
        mdl_clip.eval()
    return client, oa, proc, mdl_clip

# ── Embedding helpers ───────────────────────────────────────────────────────
def embed_text_clip(text, proc, mdl_clip):
    inputs = proc(text=[text], return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad(): feats = mdl_clip.get_text_features(**inputs)
    return feats.cpu().numpy()[0]

def embed_image_clip(url, proc, mdl_clip):
    resp = requests.get(url, timeout=10)
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad(): feats = mdl_clip.get_image_features(**inputs)
    return feats.cpu().numpy()[0]

# ── Text retrieval ──────────────────────────────────────────────────────────
def retrieve_text_with_debug(client, oa, query, k, set_status=None):
    logs = []
    if set_status:
        set_status("🔍 Starting hybrid search...", progress=10)
    graphql = f"{WEAVIATE_URL}/v1/graphql"
    headers = {"Content-Type": "application/json"}
    uq = query.replace('"', '\"')
    # Hybrid search
    hq = {"query": f"{{ Get {{ DocumentChunk(hybrid: {{query:\"{uq}\",alpha:{HYBRID_ALPHA}}},limit:{k}){{text paper paper_title heading _additional{{certainty distance score}}}} }} }}"}
    rh = requests.post(graphql, json=hq, headers=headers); rh.raise_for_status(); jh = rh.json()
    docs_hybrid = jh.get("data",{}).get("Get",{}).get("DocumentChunk",[]) or []
    if set_status:
        set_status("🔍 Starting vector search...", progress=12)
    # Vector search
    emb = oa.embeddings.create(model="text-embedding-ada-002", input=query).data[0].embedding
    emb_str = json.dumps(emb)
    fq = {"query": f"{{ Get {{ DocumentChunk(nearVector:{{vector:{emb_str}}},limit:{k}){{text paper paper_title heading _additional{{certainty distance score}}}} }} }}"}
    rf = requests.post(graphql, json=fq, headers=headers); rf.raise_for_status(); jf = rf.json()
    docs_vector = jf.get("data",{}).get("Get",{}).get("DocumentChunk",[]) or []
    if set_status:
        set_status("🔍 Combining and deduplicating results...", progress=15)
    # Combine and deduplicate (by text+paper)
    seen = set()
    docs = []
    for d in docs_hybrid + docs_vector:
        key = (d.get("text",""), d.get("paper",""))
        if key not in seen:
            docs.append(d)
            seen.add(key)
    # Build snips with certainty or score
    snips = []
    for d in docs:
        text = d.get("text", "")
        paper = d.get("paper", "")
        paper_title = d.get("paper_title", "")
        heading = d.get("heading", "")
        certainty = d.get("_additional", {}).get("certainty")
        score = d.get("_additional", {}).get("score")
        # Use certainty if available, else use score (converted to float), else 0.0
        if certainty is not None:
            final_score = certainty
        elif score is not None:
            try:
                final_score = float(score)
            except Exception:
                final_score = 0.0
        else:
            final_score = 0.0
        snips.append((text, paper, paper_title, heading, final_score))
    if set_status:
        set_status("🔍 Retrieval complete.", progress=20)
    return docs, snips, logs

# ── Image retrieval ─────────────────────────────────────────────────────────
def retrieve_images_for_paper(client, pid, limit=100):
    gql = {"query": f"{{ Get {{ PaperImage(where:{{path:[\"paper\"],operator:Equal,valueText:\"{pid}\"}},limit:{limit}){{image_url caption paper _additional{{certainty}}}} }} }}"}
    r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json=gql); r.raise_for_status()
    items = []
    for e in r.json().get("data",{}).get("Get",{}).get("PaperImage",[]) or []:
        if isinstance(e,dict): items.append({"url":e.get("image_url"),"caption":e.get("caption"),"paper":e.get("paper"),"certainty":(e.get("_additional") or {}).get("certainty",0.0)})
    return items

# ── Utilities ─────────────────────────────────────────────────────────────
def split_sections(txt): return [s.strip() for s in txt.split("\n\n") if s.strip()]

# ── Main ────────────────────────────────────────────────────────────────────
def ask_question(query, set_status=None):
    debug_logs = []
    if set_status:
        set_status("🔍 Initializing clients...", progress=0)
    if 'reranker_status_msg' in globals() and reranker_status_msg:
        debug_logs.append(reranker_status_msg)
    else:
        debug_logs.append("⚠️ FlagReranker status unknown")
    t0 = time.time()
    if not query or not query.strip(): return "",[],[],[],0.0,[],[],[],"",[],[], ""
    try:
        if set_status:
            set_status("🔍 Connecting to database...", progress=5)
        t_init = time.time()
        client, oa, proc, mdl = init_clients()
    except Exception as e:
        debug_logs.append(f"Error initializing clients: {e}")
        if set_status:
            set_status(f"❌ Error initializing clients: {e}", progress=0)
        return f"Error: {e}",[],[],[],0.0,[],[],[],"",[],[], ""
    # text retrieval + rerank
    if set_status:
        set_status("🔍 Retrieving relevant documents...", progress=10)
    t_retrieve = time.time()
    docs, snips, logs = retrieve_text_with_debug(client, oa, query, RETRIEVE_TEXT_K, set_status=set_status)
    debug_logs.append(f"[Retrieval] Retrieved {len(docs)} documents from Weaviate in {time.time() - t_retrieve:.2f} seconds. These are the candidate chunks for answering the query.")
    if set_status:
        set_status("🔍 Retrieval complete.", progress=20)
    if not docs:
        debug_logs.append("[Retrieval] No documents found during retrieval. This means the query did not match any content in the database.")
        if set_status:
            set_status("❌ Error during retrieval.", progress=20)
        return "Error during retrieval. See debug logs.", [], [], [], 0.0, [], logs, [], "", [], ""
    debug_logs += logs
    if set_status:
        set_status("🏷️ Preparing for reranking...", progress=30)
    if not snips: 
        debug_logs.append("[Retrieval] No document snippets found for reranking. This usually means retrieval returned empty or malformed results.")
        if set_status:
            set_status("❌ No docs found for query.", progress=30)
        return f"No docs for '{query}'",[],[],[],0.0,[],[],[],"",[],[], ""
    t_rerank = time.time()
    if flag_reranker:
        # Use paper_title and heading in reranker input
        pairs = [(query, f"Title: {paper_title}\nSection: {heading}\nText: {text}") for text, paper, paper_title, heading, _ in snips]
        total = len(pairs)
        if set_status:
            set_status(f"🏷️ Reranking {total} chunks...", progress=35)
        batch_size = max(1, min(5, total // 10))
        scores = []
        for i in range(0, total, batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = flag_reranker.compute_score(batch_pairs)
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
            if set_status:
                progress = 35 + int(35 * (i + len(batch_pairs)) / total)
                set_status(f"🏷️ Reranking batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}...", progress=progress)
        debug_logs.append(f"[Reranking] Reranking complete. Each score below shows the relevance of a chunk to the query (higher is better, negative means less relevant):")
        for idx, ((text, paper, paper_title, heading, _), score) in enumerate(zip(snips, scores)):
            debug_logs.append(f"    Chunk {idx+1} from paper {paper} (title: {paper_title}, section: {heading}): score = {score:.3f}")
        paired = [((text, paper, paper_title, heading), score) for (text, paper, paper_title, heading, _), score in zip(snips, scores)]
    else:
        paired = []
        for s in snips:
            if len(s) == 5:
                text, paper, paper_title, heading, certainty = s
                score = certainty if certainty is not None else 0.0
                paired.append(((text, paper, paper_title, heading), score))
            elif len(s) == 4:
                text, paper, paper_title, heading = s
                paired.append(((text, paper, paper_title, heading), 1.0))
        debug_logs.append(f"[Reranking] No reranker available. Used certainty scores for reranking. Higher certainty means more relevant.")
    # TODO: Optionally filter/boost chunks that mention the query subject in paper_title or heading
    valid_chunks = [(chunk, score) for chunk, score in paired if score is not None]
    if not valid_chunks:
        debug_logs.append("[Reranking] No valid chunks after reranking. No content passed the relevance threshold.")
        if set_status:
            set_status("❌ No valid chunks after reranking.", progress=70)
        return "No relevant documents found.", [], [], [], 0.0, [], debug_logs, [], "", [], ""
    # After reranking, apply subject-based boosting if a clear subject is found
    def extract_subject(query):
        import re
        # Find capitalized words/phrases of length > 4 (e.g., DNACloud, DeepLearning)
        matches = re.findall(r'([A-Z][a-zA-Z0-9]{4,})', query)
        # Return the longest match if any
        if matches:
            return max(matches, key=len).lower()
        return None

    subject = extract_subject(query)
    boost_value = 1.0
    boosted_chunks = []
    if subject:
        debug_logs.append(f"[Boosting] Applying subject boost for: '{subject}'")
        for (chunk, score) in valid_chunks:
            text, paper, paper_title, heading = chunk
            if subject in paper_title.lower() or subject in heading.lower():
                debug_logs.append(f"[Boosting] Boosted chunk from paper {paper} (title: {paper_title}, section: {heading}) by +{boost_value}")
                boosted_chunks.append((chunk, score + boost_value))
            else:
                boosted_chunks.append((chunk, score))
    else:
        boosted_chunks = valid_chunks
        debug_logs.append("[Boosting] No clear subject found in query; no boost applied.")

    sorted_chunks = sorted(boosted_chunks, key=lambda x: x[1], reverse=True)
    relevant_chunks = sorted_chunks[:RERANK_TEXT_K]
    used_ids = list(set(paper for (_, paper, _, _), _ in relevant_chunks))
    debug_logs.append(f"[Reranking] Text of the {RERANK_TEXT_K} selected chunks (shown in order):")
    for idx, ((text, paper, paper_title, heading), score) in enumerate(relevant_chunks):
        debug_logs.append(f"    Chunk {idx+1} (paper {paper}, title: {paper_title}, section: {heading}, score={score:.3f}): {text}")
    debug_logs.append(f"[Reranking] FINAL SELECTED CHUNKS (top {RERANK_TEXT_K}):")
    for idx, ((_, paper, paper_title, heading), score) in enumerate(relevant_chunks):
        debug_logs.append(f"    Rank {idx+1}: paper {paper} (title: {paper_title}, section: {heading}, score = {score:.3f})")
    if set_status:
        set_status("🤖 Generating answer with LLM...", progress=75)
    t_llm = time.time()
    system_prompt = (
        "You are a helpful assistant answering questions based on scientific papers from arXiv. "
        "First, review the provided context snippets. List which snippets are most useful for answering the question, and which are not useful or relevant. "
        "Then answer the question, referencing only the useful snippets. "
        "Always include references to the source papers in your answer. "
        "Only reference paper IDs that are provided in the context below. Do not make up any paper IDs. "
        "When referencing information from a paper, mention the paper ID (e.g., 'as shown in paper 2403.01944v2'). "
        "If an image is provided, reference it in your answer and explain how it relates to the question. "
        "Be concise and clear. If the answer is uncertain, say so. If possible, answer in bullet points."
    )
    allowed_paper_ids = ', '.join([paper for (_, paper, _, _), _ in relevant_chunks])
    user_prompt = "Context from the following papers:\n\n"
    for i, ((text, paper, paper_title, heading), _) in enumerate(relevant_chunks):
        user_prompt += f"Paper {paper} (title: {paper_title}, section: {heading}):\n{text}\n\n"
    user_prompt += (
        f"Question: {query}\n\n"
        f"Only reference the following paper IDs in your answer: {allowed_paper_ids}.\n"
        "Do not make up any paper IDs.\n"
        "\n"
        "---\n"
        "Respond ONLY with a JSON object in the following format. The 'answer' field should be a readable markdown answer for the user. The 'useful_chunks' and 'not_useful_chunks' fields should be lists of objects, each with only 'text', 'paper', 'heading', and 'score'. If a list is empty, use an empty list [].\n"
        "Example:\n"
        "{\n"
        "  \"answer\": \"Your markdown answer here, referencing the chunks as needed.\",\n"
        "  \"useful_chunks\": [\n"
        "    {\"text\": \"...\", \"paper\": \"arXiv_2401.00823\", \"heading\": \"Abstract\", \"score\": 0.95}\n"
        "  ],\n"
        "  \"not_useful_chunks\": [\n"
        "    {\"text\": \"...\", \"paper\": \"arXiv_1305.1269\", \"heading\": \"Author contributions\", \"score\": 0.10}\n"
        "  ]\n"
        "}\n"
        "---\n"
    )
    debug_logs.append(f"[LLM] Prompt sent to LLM. The prompt includes {len(relevant_chunks)} context snippets and allows only these paper IDs: {allowed_paper_ids}. First 300 chars shown below:")
    debug_logs.append(user_prompt[:300] + ("..." if len(user_prompt) > 300 else ""))
    try:
        answer = oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=MAX_RESPONSE_TOKENS
        ).choices[0].message.content
        debug_logs.append(f"[LLM] LLM answer received. First 300 chars shown below:")
        debug_logs.append(answer[:300] + ("..." if len(answer) > 300 else ""))
    except Exception as e:
        debug_logs.append(f"[Error] Error in LLM call: {e}. This usually means the LLM API failed or the prompt was too long.")
        if set_status:
            set_status(f"❌ Error in LLM call: {e}", progress=75)
        return "", [], [], [], 0.0, [], debug_logs, [], "", [], ""

    # --- JSON parsing for answer and feedback ---
    import json
    try:
        parsed = json.loads(answer)
        answer_structured = parsed.get('answer', '').strip()
        useful_chunks = parsed.get('useful_chunks', [])
        not_useful_chunks = parsed.get('not_useful_chunks', [])
    except Exception:
        answer_structured = answer.strip()
        useful_chunks = []
        not_useful_chunks = []

    # --- Markdown parsing for answer and feedback ---
    import re
    def parse_markdown_answer(ans):
        useful = []
        not_useful = []
        answer_text = ""
        m = re.search(r"## Useful Chunks(.*?)## Not Useful Chunks(.*?)## Answer(.*)", ans, re.DOTALL)
        if m:
            useful_section = m.group(1).strip()
            not_useful_section = m.group(2).strip()
            answer_text = m.group(3).strip()
            # Parse useful chunks
            for line in useful_section.split('\n'):
                m2 = re.match(r"\d+\.\s*\[(.*?)\] \(section: (.*?), score: ([\d\.]+)\): (.*)", line)
                if m2:
                    useful.append({
                        "paper": m2.group(1),
                        "heading": m2.group(2),
                        "score": float(m2.group(3)),
                        "text": m2.group(4)
                    })
            # Parse not useful chunks
            for line in not_useful_section.split('\n'):
                m2 = re.match(r"\d+\.\s*\[(.*?)\] \(section: (.*?), score: ([\d\.]+)\): (.*)", line)
                if m2:
                    not_useful.append({
                        "paper": m2.group(1),
                        "heading": m2.group(2),
                        "score": float(m2.group(3)),
                        "text": m2.group(4)
                    })
        else:
            answer_text = ans.strip()
        return answer_text, useful, not_useful

    # answer_structured, useful_chunks, not_useful_chunks = parse_markdown_answer(answer) # This line is no longer needed

    if set_status:
        set_status("🖼️ Scoring and gathering images...", progress=80)
    t_img = time.time()
    uniq = {}
    for pid in used_ids:
        for img in retrieve_images_for_paper(client,pid): 
            uniq[img['url']]=img
    img_scores_query=[]
    if proc and mdl:
        qv=embed_text_clip(query,proc,mdl)
        total_imgs = len(uniq)
        for idx, img in enumerate(uniq.values()):
            iv=embed_image_clip(img['url'],proc,mdl)
            scq=float(np.dot(qv,iv)/(np.linalg.norm(qv)*np.linalg.norm(iv)))
            img_scores_query.append((img,scq))
            if set_status:
                frac = 80 + int(5 * (idx+1) / max(1, total_imgs))
                set_status(f"🖼️ Scoring image {idx+1}/{total_imgs} (query)...", progress=frac)
        debug_logs.append(f"[Images] Image query scores (top 3 shown, higher means more relevant to the query):")
        for img, scq in img_scores_query[:3]:
            debug_logs.append(f"    Image {img['url']} (paper {img['paper']}): query score = {scq:.3f}")
    else:
        for img in uniq.values(): 
            img_scores_query.append((img,img.get('certainty',0.0)))
    cand=[img for img,_ in sorted(img_scores_query,key=lambda x:x[1],reverse=True)][:MAX_IMAGES]
    img_scores_final=[]
    if proc and mdl:
        av=embed_text_clip(answer_structured,proc,mdl)
        total_cand = len(cand)
        for idx, img in enumerate(cand):
            iv=embed_image_clip(img['url'],proc,mdl)
            sca=float(np.dot(av,iv)/(np.linalg.norm(av)*np.linalg.norm(iv)))
            img_scores_final.append((img,sca))
            if set_status:
                frac = 85 + int(10 * (idx+1) / max(1, total_cand))
                set_status(f"🖼️ Scoring image {idx+1}/{total_cand} (answer)...", progress=frac)
        debug_logs.append(f"[Images] Image answer scores (top 3 shown, higher means more relevant to the LLM answer):")
        for img, sca in img_scores_final[:3]:
            debug_logs.append(f"    Image {img['url']} (paper {img['paper']}): answer score = {sca:.3f}")
    else:
        for img in cand: 
            img_scores_final.append((img,img.get('certainty',0.0)))
    display=[]
    seen_urls = set()
    for img, sca in img_scores_final:
        scq = 0.0
        for img2, scq2 in img_scores_query:
            if img['url'] == img2['url']:
                scq = scq2
                break
        if scq > 0.25 and sca > 0.25 and img['url'] not in seen_urls:
            display.append(img)
            seen_urls.add(img['url'])
    if display:
        debug_logs.append(f"[Images] Top image candidate selected: {display[0]['url']} (from paper {display[0]['paper']})")
    else:
        debug_logs.append("[Images] No image candidates passed the relevance threshold.")
    
    # Deduplicate images by URL for display
    display=[]
    seen_urls = set()
    for img, sca in img_scores_final:
        scq = 0.0
        for img2, scq2 in img_scores_query:
            if img['url'] == img2['url']:
                scq = scq2
                break
        if scq > 0.25 and sca > 0.25 and img['url'] not in seen_urls:
            display.append(img)
            seen_urls.add(img['url'])
    
    # Prepare image info and append to answer if available
    if display:
        img = display[0]  # Use the top image
        image_url = img['url']
        image_caption = img.get('caption', 'Relevant figure')
        # Append the image as markdown at the bottom of the answer
        answer_structured += f"\n\n![{image_caption}]({image_url})"

    debug_img_list=[]
    for (img, scq) in img_scores_query:
        sca = 0.0
        for (img2, sca2) in img_scores_final:
            if img['url'] == img2['url']:
                sca = sca2
                break
        debug_img_list.append((img['url'], img['paper'], scq, sca, img.get('caption','')))
    
    # Add arXiv links for all referenced papers
    # First, find any paper references in the answer (e.g., "paper 2403.01944v2")
    # Use a more specific pattern for arXiv paper IDs: YYYY.MMDDNNNNvN format
    paper_refs = re.findall(r'paper\s+(\d{4}\.\d{4,5}v?\d*)', answer_structured, re.IGNORECASE)
    
    # Add links for all used papers and any mentioned in the answer
    all_papers = set(used_ids + paper_refs)
    
    # Create a references section
    if all_papers:
        references_section = "\n\n**References:**\n"
        for paper_id in sorted(all_papers):
            references_section += f"- [{paper_id}](https://arxiv.org/abs/{paper_id})\n"
        answer_structured += references_section
    
    # Also replace any paper references in the text with clickable links
    for paper_id in all_papers:
        # Replace "paper 2403.01944v2" with "paper [2403.01944v2](https://arxiv.org/abs/2403.01944v2)"
        answer_structured = re.sub(
            rf'paper\s+{re.escape(paper_id)}',
            f'paper [{paper_id}](https://arxiv.org/abs/{paper_id})',
            answer_structured,
            flags=re.IGNORECASE
        )
    
    pat=r"\[.*?\]\(https://arxiv\\.org/abs/(.*?)\)"; parts=[]
    for sec in split_sections(answer_structured):
        ids = re.findall(pat, sec)
        txt = re.sub(pat, '', sec).strip()
        valid_ids = [pid for pid in ids if pid in used_ids]
        links = ' '.join(f"[{pid}](https://arxiv.org/abs/{pid})" for pid in valid_ids)
        parts.append(f"{txt} {links}" if valid_ids else txt)
    final="\n\n".join(parts)
    if relevant_chunks:
        thr = sum(score for _, score in relevant_chunks) / len(relevant_chunks)
    else:
        thr = MIN_RELEVANCE_SCORE
    if set_status:
        set_status("✅ Complete!", progress=100)
    return (
        final,                # processed answer text
        relevant_chunks,      # top reranked chunks
        used_ids,             # paper IDs used
        [],                   # images (if any)
        thr,                  # threshold
        img_scores_final,     # image scores
        debug_logs,           # debug logs
        debug_img_list,       # image debug info
        answer_structured,    # parsed markdown answer
        useful_chunks,        # parsed useful chunks
        not_useful_chunks,    # parsed not useful chunks
        answer                # raw LLM output
    )

if __name__=="__main__":
    print("RAG Chat ready")
    while True:
        q=input("Q> ")
        if not q or q.lower() in ("exit","quit"): break
        ans,sn,src,imgs,thr,im_fs,ret,im_dbg = ask_question(q)
        print("\n-- Retrieval Logs --"); [print(l) for l in ret]
        print("\n-- Image Logs --"); [print(f"{u}, {p}, q={qsc:.3f}, a={asc:.3f}, cap={cap}") for (u,p,qsc,asc,cap) in im_dbg]
        print(f"\nAnswer (thr={thr:.3f}):\n{ans}\nSources: {src}")
