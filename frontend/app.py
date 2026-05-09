import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="DocuMind", page_icon="🔬", layout="wide")
st.title("🔬 DocuMind")
st.caption("Data Science Research Paper Intelligence")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def api_get(path: str):
    return requests.get(f"{API_BASE_URL}{path}", timeout=30)


def api_post(path: str, payload: dict):
    return requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=180)


with st.sidebar:
    st.header("DocuMind")
    try:
        health = api_get("/health").json()
        st.success("Ollama online" if health.get("ollama_available") else "Ollama offline")
        st.write(f"LLM: `{health.get('llm_model', '-')}`")
        st.write(f"Embedding: `{health.get('embedding_model', '-')}`")
        stats = health.get("collection_stats", {})
        st.metric("Papers", stats.get("paper_count", 0))
        st.metric("Chunks", stats.get("total_chunks", 0))
    except Exception as exc:
        st.error(f"API unavailable: {exc}")

tabs = st.tabs(["🔍 Ask Papers", "📄 Upload Paper", "🌐 Fetch from ArXiv", "📚 Paper Library"])

mode_mapping = {
    "General Q&A": "general",
    "Compare Methods": "compare",
    "Methodology Deep Dive": "methodology",
    "Dataset Finder": "datasets",
    "Reproduce Results": "reproduce",
}

with tabs[0]:
    mode_label = st.radio("Query Mode", list(mode_mapping.keys()), horizontal=True)
    section = st.selectbox(
        "Section Filter",
        ["All Sections", "abstract", "introduction", "methodology", "experiments", "results", "conclusion"],
    )
    top_k = st.slider("Top K", 3, 24, 6)
    query = st.text_area(
        "Question", placeholder="e.g. What datasets were used for tabular fraud detection?"
    )
    if st.button("Ask DocuMind", use_container_width=True) and query.strip():
        payload = {
            "query": query.strip(),
            "top_k": top_k,
            "query_mode": mode_mapping[mode_label],
            "section_filter": None if section == "All Sections" else section,
        }
        with st.spinner("Reasoning over your paper library..."):
            resp = api_post("/api/v1/query", payload)
        if resp.status_code == 200:
            data = resp.json()
            st.markdown(
                f"<div style='border-left: 4px solid #22c55e; padding: 12px;'>{data['answer']}</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(data.get("confidence", 0.0)))
            if not data.get("has_answer"):
                st.warning("No strong answer found in current library.")
            for i, src in enumerate(data.get("sources", []), start=1):
                with st.expander(
                    f"📄 Source {i}: {src.get('paper_title', 'Unknown')} ({src.get('year', '-')}) — {src.get('section', 'body')}"
                ):
                    st.write(src.get("content_preview", ""))
            st.session_state.chat_history.append(
                {"query": query, "answer": data["answer"], "sources": data.get("sources", []), "mode": payload["query_mode"]}
            )
        else:
            st.error(resp.text)

    for idx, exchange in enumerate(reversed(st.session_state.chat_history[-5:]), start=1):
        with st.expander(f"History {idx}: {exchange['query'][:80]}"):
            st.write(exchange["answer"])

with tabs[1]:
    files = st.file_uploader("Upload papers", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if files and st.button("Upload Selected Files", use_container_width=True):
        for item in files:
            files_payload = {"file": (item.name, item.getvalue(), item.type or "application/octet-stream")}
            response = requests.post(f"{API_BASE_URL}/api/v1/ingest", files=files_payload, timeout=180)
            if response.status_code == 200:
                payload = response.json()
                st.success(f"Uploaded {payload['filename']}")
                st.write(
                    f"Title: {payload['title']} | Authors: {payload['authors']} | Year: {payload['year']} | Chunks: {payload['chunks_created']} | {payload['processing_time_ms']:.1f} ms"
                )
            else:
                st.error(f"{item.name}: {response.text}")

with tabs[2]:
    arxiv_id = st.text_input("ArXiv Paper ID", placeholder="e.g. 1706.03762 or 2401.12345")
    if st.button("Fetch from ArXiv", use_container_width=True) and arxiv_id.strip():
        response = api_post("/api/v1/fetch-arxiv", {"arxiv_id": arxiv_id.strip()})
        if response.status_code == 200:
            payload = response.json()
            st.success(f"Fetched {payload['title']}")
            st.write(
                f"Authors: {payload['authors']} | Year: {payload['year']} | Chunks: {payload['chunks_created']}"
            )
            clean_id = arxiv_id.strip().replace("arXiv:", "")
            st.markdown(f"[Open on arXiv](https://arxiv.org/abs/{clean_id})")
        else:
            st.error(response.text)

with tabs[3]:
    response = api_get("/api/v1/papers")
    if response.status_code != 200:
        st.error(response.text)
    else:
        papers = response.json()
        if not papers:
            st.info("Your library is empty. Upload a paper or fetch one from ArXiv.")
        else:
            total_chunks = sum(p["chunk_count"] for p in papers)
            st.write(f"Total papers: **{len(papers)}** | Total chunks: **{total_chunks}**")
            for paper in papers:
                cols = st.columns([10, 2])
                with cols[0]:
                    st.subheader(f"📄 {paper['title']}")
                    arxiv_id = paper.get("arxiv_id", "")
                    if arxiv_id:
                        st.markdown(
                            f"{paper['authors']} — {paper['year']} — [arXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})"
                        )
                    else:
                        st.write(f"{paper['authors']} — {paper['year']}")
                    st.caption(f"Chunks: {paper['chunk_count']}")
                with cols[1]:
                    if st.button("🗑️ Delete", key=paper["doc_id"]):
                        requests.delete(f"{API_BASE_URL}/api/v1/papers/{paper['doc_id']}", timeout=60)
                        st.rerun()
