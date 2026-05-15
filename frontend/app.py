import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8001"
LIBRARY = "public"

st.set_page_config(page_title="DocuMind · Public corpus", page_icon="📚", layout="wide")
st.title("DocuMind — public index")
st.caption(
    "Streamlit mirror of the operator path: queries and ingest target the **public** Chroma collection only. "
    "Bulk Wikipedia-scale loads: `scripts/bulk_index_public.py`."
)

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
        lib = api_get("/api/v1/libraries")
        if lib.status_code == 200:
            data = lib.json()
            pub = data.get("public", {})
            st.metric("Public articles", pub.get("paper_count", 0))
            st.metric("Public vectors", pub.get("total_chunks", 0))
            st.caption(f"Collection `{pub.get('collection_name', '')}`")
        else:
            stats = health.get("collection_stats", {})
            st.metric("Articles (health view)", stats.get("paper_count", 0))
            st.metric("Chunks", stats.get("total_chunks", 0))
    except Exception as exc:
        st.error(f"API unavailable: {exc}")

tabs = st.tabs(["Ask (public index)", "Upload to public", "Articles in index"])

mode_mapping = {
    "General Q&A": "general",
    "Compare across articles": "compare",
    "Topic deep dive": "methodology",
    "Entity & fact inventory": "datasets",
    "Chronology / provenance": "reproduce",
}

with tabs[0]:
    mode_label = st.radio("Query mode", list(mode_mapping.keys()), horizontal=True)
    top_k = st.slider("Top K", 3, 24, 10)
    use_flare = st.checkbox(
        "FLARE-style active retrieval (draft + possible 2nd search)",
        value=False,
        disabled=(mode_label == "Entity & fact inventory"),
        help="Ignored for entity-inventory mode.",
    )
    query = st.text_area(
        "Question",
        placeholder="Grounded question — answers use only retrieved passages from the public index.",
    )
    if st.button("Run query", use_container_width=True) and query.strip():
        payload = {
            "query": query.strip(),
            "library": LIBRARY,
            "top_k": top_k,
            "query_mode": mode_mapping[mode_label],
            "section_filter": None,
            "use_flare": bool(use_flare),
        }
        with st.spinner("Retrieving from public index…"):
            resp = api_post("/api/v1/query", payload)
        if resp.status_code == 200:
            data = resp.json()
            st.markdown(
                f"<div style='border-left: 4px solid #22c55e; padding: 12px;'>{data['answer']}</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(data.get("confidence", 0.0)))
            if data.get("flare_followup_retrieval"):
                st.caption("Retrieval: FLARE follow-up pass merged into context.")
            if not data.get("has_answer"):
                st.warning("No strong answer from the current public index.")
            for i, src in enumerate(data.get("sources", []), start=1):
                with st.expander(
                    f"Source {i}: {src.get('paper_title', 'Unknown')} ({src.get('year', '-')}) — {src.get('section', 'body')}"
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
    files = st.file_uploader("Upload to public index", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if files and st.button("Upload selected files", use_container_width=True):
        for item in files:
            files_payload = {"file": (item.name, item.getvalue(), item.type or "application/octet-stream")}
            response = requests.post(
                f"{API_BASE_URL}/api/v1/ingest",
                files=files_payload,
                data={"library": LIBRARY},
                timeout=180,
            )
            if response.status_code == 200:
                payload = response.json()
                st.success(f"Indexed {payload['filename']}")
                st.write(
                    f"Title: {payload['title']} | Chunks: {payload['chunks_created']} | {payload['processing_time_ms']:.1f} ms"
                )
            else:
                st.error(f"{item.name}: {response.text}")

with tabs[2]:
    response = api_get(f"/api/v1/papers?library={LIBRARY}")
    if response.status_code != 200:
        st.error(response.text)
    else:
        papers = response.json()
        if not papers:
            st.info("Public index is empty. Use bulk_index_public.py or the Upload tab.")
        else:
            total_chunks = sum(p["chunk_count"] for p in papers)
            st.write(f"Articles: **{len(papers)}** | Vectors: **{total_chunks}**")
            for paper in papers:
                cols = st.columns([10, 2])
                with cols[0]:
                    st.subheader(paper["title"])
                    st.write(f"{paper.get('authors', '')} — {paper.get('year', '')}")
                    st.caption(f"Chunks: {paper['chunk_count']}")
                with cols[1]:
                    if st.button("Delete", key=paper["doc_id"]):
                        requests.delete(
                            f"{API_BASE_URL}/api/v1/papers/{paper['doc_id']}?library={LIBRARY}",
                            timeout=60,
                        )
                        st.rerun()
