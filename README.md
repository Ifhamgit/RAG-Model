# Meridian Academy RAG — grounded Q&A with source attribution

A retrieval-augmented Q&A service over a mock edtech company's policies, FAQs, and course
details. A support agent asks a question in plain language and gets a short, factual answer with
every claim attributed to the document, section, and page it came from — or an explicit "the
corpus does not cover this", which is treated as a correct answer rather than a failure.

Built without a RAG framework: plain Python, FastAPI, SQLite + FTS5, and a NumPy vector matrix.
The reasoning behind every design decision — chunking strategy, retrieval design, the exact system
prompt, instrumentation, and evaluation — is in **[DESIGN.md](DESIGN.md)**. The step-by-step build
sequence is in **[BUILD_PLAN.md](BUILD_PLAN.md)**.

> **Status: scaffolding.** Step 0 of 14 complete. The commands below are the target interface and
> do not work yet; each becomes live at the step noted.

## Quickstart

```bash
pip install -r requirements.txt

cp .env.example .env          # then set SCALER_LLM_API_KEY
```

```bash
python main.py --ingest                       # build the index from corpus/   (Step 6)
python main.py --query "What is the refund policy?"   # ask a question         (Step 9)
python main.py --serve                        # POST /ask + browser UI at :8000 (Step 11)
```

## Corpus

`corpus/` holds six documents in four formats — 2 PDFs, 3 plain-text policy documents, and 1 YAML
structured document, ~19k tokens total. The PDFs are reproducible from source with
`python tools/build_corpus_pdfs.py`.

## Layout

```
corpus/     source documents (read-only)
rag/        the pipeline — see rag/__init__.py for the module map
eval/       evaluation suite (Step 12)
tools/      corpus generation
index/      generated: rag.db + vectors.npy   (gitignored, rebuildable)
logs/       generated: traces.jsonl           (gitignored)
```
