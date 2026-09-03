"""Meridian Academy RAG — grounded, cited Q&A over a heterogeneous document corpus.

The specification for this package is DESIGN.md at the repository root. Where
code and that document disagree, the document is the reference and the code is
the bug.

Module map (DESIGN.md §a.1):

    config      settings, one source of truth for every tunable parameter
    models      RawDoc / Chunk / RetrievedChunk / AnswerResult contracts
    loaders     per-file-type parsing, authority stamping          (§b.4)
    chunker     structure-aware splitting, breadcrumb headers      (§b.1-b.3)
    embeddings  pluggable Embedder: fastembed | local | openai     (§c.1)
    store       SQLite + FTS5 + NumPy vector matrix                (§c.2)
    retriever   hybrid dense + BM25, RRF fusion, authority order   (§c.3-c.6)
    llm         provider-detecting client, structured output       (§a.3)
    answerer    grounded prompt, citation verification, abstention (§d)
    tracing     one structured trace per query                     (§e)
    pipeline    ingest() and ask() orchestration
    api         FastAPI surface and minimal UI
"""

__version__ = "0.1.0"
