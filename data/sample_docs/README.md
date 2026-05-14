# `data/sample_docs/` — legacy demo corpus (papers library only)

**Product stance:** DocuMind is **Wikipedia-primary** for scale (`library=public` + offline bulk index). This directory is a **small synthetic / curated bundle** for optional **papers** demos—not a resume-grade “millions of documents” knowledge base.

## Committed metrics (counted from git tree)

| Metric | Count |
|--------|------:|
| Total `*.txt` files | **463** |
| Synthetic `sample_corpus_p7_*.txt` | **400** |
| Other named / hand-authored `.txt` | **63** |

**Chunk row count** is **not** stored in git: it is whatever Chroma holds after `SEED_SAMPLE_DOCS=true` startup indexing (`CHUNK_SIZE`, `CHUNK_OVERLAP`, and text length). Order of magnitude for the full bundle: **low thousands to ~10k+** chunks (estimate only).

**Contrast:** English Wikipedia is on the order of **millions** of articles (CC BY-SA). That is the credible production RAG scale story; use `scripts/stream_wikipedia_to_txt.py` + `scripts/bulk_index_public.py` into `CHROMA_COLLECTION_PUBLIC`.

## Tests

CI and unit tests use **fake** corpora (e.g. `tests/ranking_fake_embedding.py` — six deterministic chunks), not this directory.
