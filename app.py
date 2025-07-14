#!/usr/bin/env python3
import os
import streamlit as st
import time
from dotenv import load_dotenv
from rag_chat import ask_question
import json

# Load environment variables
load_dotenv()

# Debug: Show environment variables in Streamlit
with st.expander("Debug: Environment Variables", expanded=False):
    st.write("WEAVIATE_URL =", os.getenv("WEAVIATE_URL"))
    st.write("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
    st.write("WEAVIATE_API_KEY =", os.getenv("WEAVIATE_API_KEY"))

# Set up OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit setup
st.set_page_config(page_title="Research SciFy RAG Chatbot", layout="wide")
st.title("🔍 Research SciFy RAG Chatbot")

# Place status/progress indicator at the top of the main page
status_placeholder = st.empty()
progress_bar = st.empty()

# State for results
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'debug' not in st.session_state:
    st.session_state['debug'] = None
if 'processing' not in st.session_state:
    st.session_state['processing'] = False

def set_status(msg, progress=None):
    status_placeholder.markdown(f"### {msg}")
    if progress is not None:
        progress_bar.progress(progress)
    else:
        progress_bar.empty()

# Query form (only show if not processing)
if not st.session_state['processing']:
    with st.form("query_form"):
        query = st.text_input(
            "Ask a question about your arXiv collection:",
            placeholder="e.g. What are the 3 main modules of DNACloud?"
        )
        submitted = st.form_submit_button("Submit")
        if submitted and query:
            st.session_state['processing'] = True
            set_status("🔍 Retrieving relevant documents...", progress=0)
            start_time = time.time()
            try:
                with st.spinner("Processing your query..."):
                    (
                        answer,
                        snippet_scores,
                        sources,
                        images,  # This will now be empty list
                        threshold,
                        img_scores,
                        retrieval_logs,
                        debug_img_list
                    ) = ask_question(query, set_status=set_status)
                elapsed = time.time() - start_time
                set_status(f"✅ Complete! (Total time: {elapsed:.1f} seconds)", progress=100)
                st.session_state['result'] = (answer, threshold, debug_img_list, retrieval_logs, snippet_scores)
                st.session_state['debug'] = retrieval_logs
            except Exception as e:
                set_status("")
                st.session_state['processing'] = False
                st.session_state['result'] = None
                st.session_state['debug'] = None
                st.error(f"❌ Error running RAG pipeline: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.write("WEAVIATE_URL =", os.getenv("WEAVIATE_URL"))
                st.write("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
                st.write("WEAVIATE_API_KEY =", os.getenv("WEAVIATE_API_KEY"))
            else:
                st.session_state['processing'] = False

# While processing, show spinner and progress bar
if st.session_state['processing']:
    set_status("Processing...", progress=0)
    st.spinner("Processing your query...")

# After completion, show answer, new query button, and debug info
if st.session_state['result']:
    answer, threshold, debug_img_list, retrieval_logs, snippet_scores = st.session_state['result']
    tab_answer, tab_debug = st.tabs(["Answer", "Debug Info"])
    with tab_answer:
        st.markdown("**Answer:**")
        st.write(answer)
        st.markdown(f"_Threshold used: {threshold:.3f}_")
    with tab_debug:
        def short(log):
            if isinstance(log, str) and (len(log) > 300 or 'payload' in log or 'resp' in log or 'embedding' in log):
                return log[:300] + ' ... [truncated]'
            return log
        retrieval_logs_grouped = [log for log in retrieval_logs if log.startswith("[Retrieval]")]
        rerank_logs_grouped = [
            log for log in retrieval_logs
            if log.startswith("[Reranking]") or log.lstrip().startswith("Chunk")
        ]
        llm_logs_grouped = [log for log in retrieval_logs if log.startswith("[LLM]")]
        image_logs_grouped = [log for log in retrieval_logs if log.startswith("[Images]")]
        error_logs_grouped = [log for log in retrieval_logs if log.startswith("[Error]")]
        if retrieval_logs_grouped:
            st.markdown("### Retrieval")
            st.info("Shows how many documents were retrieved from the database and how long it took. If no documents are found, the query may not match your collection.")
            for log in retrieval_logs_grouped:
                st.code(short(log))
        if rerank_logs_grouped:
            st.markdown("### Reranking")
            st.info("Shows how the system scored and selected the most relevant document chunks for your query. Higher scores mean more relevant.")
            for log in rerank_logs_grouped:
                if log.lstrip().startswith("Chunk"):
                    chunk_label, chunk_text = log.split(":", 1)
                    with st.expander(chunk_label.strip()):
                        st.markdown(chunk_text.strip())
                else:
                    st.code(short(log))
        if llm_logs_grouped:
            st.markdown("### LLM")
            st.info("Shows the prompt sent to the language model and the answer received. Only the first 300 characters are shown for readability.")
            for log in llm_logs_grouped:
                st.code(short(log))
        if image_logs_grouped:
            st.markdown("### Images")
            st.info("Shows how images were scored for relevance to your query and the answer. Only the top candidates are shown.")
            for log in image_logs_grouped:
                st.code(short(log))
        if error_logs_grouped:
            st.markdown("### Errors")
            st.info("Any errors or warnings encountered during the pipeline will appear here.")
            for log in error_logs_grouped:
                st.code(short(log))

# Footer
st.markdown("---")
st.caption("Powered by Weaviate & OpenAI")

headers = {
    "X-OpenAI-Api-Key": OPENAI_API_KEY,
    "X-Openai-Api-Key": OPENAI_API_KEY
}
