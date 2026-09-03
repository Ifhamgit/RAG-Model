# Meridian Academy RAG — grounded Q&A with source attribution

A retrieval-augmented Q&A service over a mock edtech company's policies, FAQs, and course details.
A support agent asks in plain language and gets a short, factual answer with every claim attributed
to the document, section, and page it came from. When the corpus does not cover a question, the
system says so and flags the ticket for a human — abstention is a correct answer here, not a
failure.

Built without a RAG framework: plain Python, FastAPI, SQLite + FTS5, and a NumPy vector matrix.
The reasoning behind every decision is in **[DESIGN.md](DESIGN.md)**, written before the code; the
build sequence is in **[BUILD_PLAN.md](BUILD_PLAN.md)**.

---

## Quickstart

From a clean clone:

```bash
pip install -r requirements.txt

cp .env.example .env          # Windows: copy .env.example .env
# then set SCALER_LLM_API_KEY in .env

python main.py --ingest                                        # build the index (~80 s first run)
python main.py --query "If someone withdraws on day 45, do they get their GST back?"
python main.py --serve                                         # UI + API on http://127.0.0.1:8000
```

The first `--ingest` downloads the embedding model (~130 MB, once, then cached). No other setup:
no Docker, no vector database, no external services.

`--ingest` and `--search` work **without an API key** — only answering needs one.

---

## CLI

| Command | What it does |
|---|---|
| `python main.py --ingest [--force]` | Build the index from `corpus/`. Re-runs reuse unchanged embeddings (~1 s); `--force` rebuilds. |
| `python main.py --search "..."` | **Retrieval only, no LLM.** Ranked chunks with per-arm scores *and* ranks. |
| `python main.py --query "..."` | Ask a question; prints the answer, cited sources, and every verification flag. |
| `python main.py --check-llm` | One round-trip to confirm the key, provider, and structured output resolve. |
| `python main.py --traces [N]` | Recent query traces. |
| `python main.py --trace <id>` | One trace in full, including per-arm ranks. |
| `python main.py --serve` | HTTP API and browser UI. |
| `python tools/sql.py --tables` | Read-only SQL against the index and traces. |

`--search` exists because retrieval failures are the majority of RAG failures and no prompt work
rescues them. Comparing `dense_rank` against `bm25_rank` localises a failure to one arm:

```
$ python main.py --search "What is the refund policy?"

#  doc                      section                     auth            dns    cos  bm    bm25      rrf
1  refund_terms.txt         1. SCOPE AND DEFINITIONS    authoritative    19  0.701   7   3.420  0.02758
2  faq.txt                  SECTION 6: REFUNDS...       summary           1  0.732   3   4.690  0.02581*
...
  * = demoted by the §c.6 authority rule (x0.8)
  dense_miss=False  sparse_miss=False  HARD_MISS=False
```

Note rows 1 and 2: raw semantic similarity ranks the FAQ *summary* above the authoritative refund
schedule. The authority rule reorders them — see §c.6.

---

## API

`POST /ask`

```json
{ "question": "If someone withdraws on day 45, do they get their GST back?" }
```

```json
{
  "answer": "No, GST is not refunded for a Day 45 withdrawal [S3]. Day 45 falls in the Day 31 to Day 60 band, where GST is explicitly excluded from the refund [S2]. ... tuition refund is 40% of INR 2,10,000 = INR 84,000, and GST refund = INR 0 [S3].",
  "citations": ["S1", "S2", "S3"],
  "sources": [
    {
      "id": "S1", "doc": "refund_terms.txt", "doc_id": "MA-REFUND-2026-01",
      "section": "4. GST TREATMENT", "page": null, "authority": "authoritative",
      "dense_rank": 1, "dense_score": 0.755, "bm25_rank": 2, "bm25_score": 25.888,
      "rrf_score": 0.03252, "authority_demoted": false, "cited": true,
      "text": "4.1 GST is charged at 18% on the tuition component ..."
    }
  ],
  "sufficient_context": true,
  "escalate": false,
  "no_context": false,
  "citation_integrity": "ok",
  "invalid_citations": [],
  "unsupported_claim": false,
  "trace_id": "0d022197-f5a5-4fcf-a621-b21b9906f7c2",
  "latency_ms": 4561.0,
  "latency_breakdown": { "expand_ms": 0.0, "embed_ms": 73.0, "retrieve_ms": 3.0, "llm_ms": 4480.0, "verify_ms": 0.1 },
  "input_tokens": 2423, "output_tokens": 286,
  "model": "anthropic/claude-sonnet-5",
  "embedding_backend": "fastembed:BAAI/bge-small-en-v1.5"
}
```

**An abstention is HTTP 200 with `escalate: true`**, never an error — it is the correct answer to an
unanswerable question, and `escalate` is the field a real support tool routes on. Only a genuine
fault (index missing, LLM unreachable) is a 5xx.

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Ask a question |
| `GET /health` | Chunk count, which embedding backend actually loaded, LLM provider/model, and whether the corpus fingerprint still matches the index |
| `GET /traces?limit=N` | Recent traces |
| `GET /traces/{id}` | One trace in full |
| `GET /` | Single self-contained page — no build step, no CDN |

---

## Architecture

```
corpus/ ─► loaders ─► chunker ─► embedder ─► SQLite + FTS5 + vectors.npy
           per-type   structure  fastembed        (one file, no services)
           + authority  -aware    bge-small
                        + breadcrumbs
                                                        ▲
question ─► [expansion] ─► dense (cosine) ─┐             │
                        ─► sparse (BM25)  ─┴─► RRF ─► authority reorder ─► top-5
                                                                    │
                                    [S1..S5] blocks ─► LLM ─► verify citations ─► trace
```

- **Loaders** normalise four file types into one shape and stamp document **authority**, because
  this corpus deliberately contains summaries that say "quote the other document, not me".
- **Chunker** splits on structure the author already wrote (clauses, Q&A pairs, headings), then
  prefixes each chunk with a `[title | doc_id | §section]` breadcrumb *inside the indexed text*.
- **Retrieval is hybrid** because the corpus is full of exact tokens (`ASD-501`, `Day 31 to Day 60`)
  where BM25 wins, and policy vocabulary the user never types ("Withdrawal Request" for "can I
  quit") where embeddings win. Fused by Reciprocal Rank Fusion — rank only, so the two
  incomparable score scales never need weights.
- **Grounding is enforced, not requested**: the prompt is one of four mechanisms; the others are
  labelled provenance blocks, server-side citation verification, and a structural check that an
  answer claiming sufficient context actually cited something.

Every claim above is argued in [DESIGN.md](DESIGN.md); read it for the trade-offs and the
alternatives rejected.

---

## Evaluation

```bash
python eval/run_eval.py --retrieval-only   # free, no LLM: recall@k and miss routing
python eval/run_eval.py --no-judge         # answers generated, judges skipped
python eval/run_eval.py                    # all five metrics
python eval/run_eval.py --case refund-day-45
```

Twelve cases across eight categories, including a cross-document synthesis, an authority conflict,
a numeric case with a **negative** required fact, and two out-of-corpus questions that must be
refused. Five metrics (DESIGN.md §f.2): retrieval recall@k and abstention accuracy are
deterministic; answer correctness, faithfulness, and context precision are LLM-judged.

Reading the output: a case passes only if **every** metric it ran clears its bar — all required
facts present, zero contradicted claims. Full results, including every judge justification, land in
`eval/results/<timestamp>.json` so a surprising score can be inspected rather than trusted. The
suite exits non-zero on any failure, so it works as a CI gate.

Current results (12/12 passing):

| Metric | Score | |
|---|---|---|
| Retrieval recall@5 | **100%** | expected document *and* section in the top-5, all 10 answerable cases |
| Abstention accuracy | **100%** | abstained on exactly the 2 unanswerable cases, and on no others |
| Answer correctness | **100%** | every required fact present, judged by meaning not string match |
| Faithfulness | **100%** | 0 contradicted claims across all cases |
| Context precision | **45%** | ← the one weak number; see below |

Also reported: p50 latency 5.1 s, p95 8.9 s; ~2,600 tokens and ~$0.008 per query.

**Context precision at 45% is the honest signal, and it is not a bug.** It means roughly 2 of every
5 chunks sent to the model earned their place. With recall at 100% that is the expected shape of a
`TOP_K` set slightly too wide: we are buying complete coverage with some noise. Lowering `TOP_K`
would raise precision and risk recall, and DESIGN.md §f.3 is explicit that n=12 cannot distinguish
a real gain from noise — so it stays as measured, and a principled sweep on a larger case set is
§g.3's job rather than a number tuned to look good here.

**The eval already changed the design once.** DESIGN.md §c.5 argues for LLM query expansion. Measured
against the suite, it lost on every metric at once — recall@5 80% vs 100%, correctness 89% vs 92%,
context precision 42% vs 48% — while adding ~2.5 s per query. It now ships disabled behind a flag,
and §c.5 records the reversal above its original argument rather than quietly rewriting it.

---

## Known limitations

Stated plainly, because a metric whose weaknesses are unstated invites over-trust.

- **Twelve cases is a smoke test, not a measurement.** At n=12 one case flipping moves the headline
  by 8 points. This suite can catch a *broken* change and cannot adjudicate a *marginal* one. I
  would not tune a parameter on a 4-point difference here.
- **The test set was written by the system's author, from the corpus.** Its vocabulary is
  unconsciously closer to the documents than a real support ticket would be — the most common way
  RAG evals flatter themselves. The fix is real logged questions, not more author-written ones.
- **The judge shares a model family with the system under test**, so a misread clause both make
  identically is invisible. Mitigated by low effort, strict schemas, and logged justifications;
  properly fixed only by a different judge model plus human agreement sampling.
- **No reranker.** Bi-encoder retrieval scores query and chunk independently and cannot model their
  interaction. A cross-encoder over the top-20 is the standard next step and the single
  highest-value addition — deliberately deferred, not overlooked.
- **The offline LSA fallback has a real ceiling.** If the embedding model cannot be downloaded, the
  system falls back to corpus-fitted TF-IDF/SVD. Measured on "can I quit and get my money back" vs
  a Withdrawal Request clause, it scored the *unrelated* sentence higher (0.079 vs 0.041) where
  bge-small separated them cleanly (0.425 vs 0.726). It keeps the system running offline; it does
  not keep it good. Every trace records `embedding_backend` so a degraded run is explained.
- **Single-node, in-process, no auth.** No queue, no worker pool, no horizontal scaling story, and
  CORS is open for local use. The store is the right choice at ~183 chunks and the wrong one past
  ~100k.
- **Nothing here measures what actually matters commercially** — whether the agent resolved the
  ticket faster and whether the answer avoided creating a complaint. Offline metrics are a loose
  proxy for that.

---

## Layout

```
corpus/     six source documents in four formats (~19k tokens), read-only
rag/        the pipeline — see rag/__init__.py for the module map
eval/       12 cases, five metrics, LLM judge
tools/      corpus generation, read-only SQL console
index/      generated: rag.db + vectors.npy       (gitignored, rebuildable)
logs/       generated: traces.jsonl               (gitignored)
```
