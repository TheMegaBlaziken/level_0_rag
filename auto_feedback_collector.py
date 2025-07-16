import os
import json
import csv
from datetime import datetime
from rag_chat import ask_question  # Import your RAG function
from openai import OpenAI

def generate_questions(md_text, n=3):
    prompt = (
        f"Given the following scientific paper in markdown, generate {n} technical questions "
        "that require information from different sections of the paper to answer. "
        "Respond ONLY with a JSON list of questions, e.g.:\n"
        "[\"Question 1?\", \"Question 2?\", \"Question 3?\"]\n\n"
        f"{md_text[:4000]}"
    )
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )
    # Try to extract the first JSON list from the response
    import re, json
    content = resp.choices[0].message.content.strip()
    try:
        # Try direct JSON parse
        return json.loads(content)
    except Exception:
        # Try to extract JSON list from within text/code block
        match = re.search(r'(\[.*\])', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse questions as JSON. LLM output was:\n{content}")

def log_feedback(feedback_file, question, chunk, paper_id, heading, feedback_type):
    with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            paper_id,
            heading,
            chunk[:100],  # Truncate for brevity
            question,
            feedback_type
        ])

def main(markdown_dir, feedback_file='feedback_log.csv'):
    for fname in os.listdir(markdown_dir):
        if not fname.endswith('.md'):
            continue
        path = os.path.join(markdown_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        paper_id = os.path.splitext(fname)[0]
        questions = generate_questions(md_text)
        for question in questions:
            # Use your RAG pipeline to get answer and chunk info
            (
                answer,
                snippet_scores,
                sources,
                images,
                threshold,
                img_scores,
                debug_logs,
                debug_img_list,
                answer_structured,
                useful_chunks,
                not_useful_chunks,
                raw_llm_output
            ) = ask_question(question)
            # Log feedback for useful chunks
            for chunk in useful_chunks:
                log_feedback(
                    feedback_file,
                    question,
                    chunk.get('text', ''),
                    chunk.get('paper', paper_id),
                    chunk.get('heading', ''),
                    'useful'
                )
            # Log feedback for not useful chunks
            for chunk in not_useful_chunks:
                log_feedback(
                    feedback_file,
                    question,
                    chunk.get('text', ''),
                    chunk.get('paper', paper_id),
                    chunk.get('heading', ''),
                    'not useful'
                )

if __name__ == "__main__":
    main("papers/markdowns")
