# DESIGN.md — Grounded Q&A over the Meridian Academy corpus

**Author:** Ifham Siddiqui
**Scope:** A retrieval-augmented Q&A service that answers natural-language questions about a
mock edtech company's policies, FAQs, and course details, and attributes every answer to the
source document and chunk it came from.

---

## 0. The problem, stated precisely

A support agent types *"If someone withdraws on day 45, do they get their GST back?"* and needs a
correct, citable answer in seconds.

The failure that matters here is not "no answer". It is **a confident wrong answer about a refund
amount**. A support agent who repeats a hallucinated refund figure to a learner creates a financial
and reputational liability. So the design target is not *maximise answer rate*; it is:

> Maximise the number of questions answered correctly **with a verifiable citation**, while driving
> confidently-wrong answers as close to zero as possible — and abstaining loudly when the corpus is
> silent.

Every decision below is downstream of that sentence. Abstention is a feature, not a fallback.

Three properties of this specific corpus shape the whole design:

1. **It is small — ~19k tokens across 6 documents.** That rules *out* heavyweight vector
   infrastructure as over-engineering, and rules *in* techniques that are too slow at 10M chunks.
2. **It is heterogeneous** — 2 PDFs, 3 plain-text policy documents, 1 YAML structured document.
   Each needs different parsing and, critically, different chunking.
3. **It is adversarially cross-referential.** The FAQ gives a *summary* of the refund policy and
   explicitly says "do not quote this, quote MA-REFUND-2026-01". Pricing appears in the brochure
   *and* the YAML. Placement Assurance eligibility is stated in four documents. A naive retriever
   will happily return the summary and produce a subtly wrong answer. Handling document authority
   is a first-class requirement, not a nicety.

---

## a. System Architecture

### a.1 Component diagram

```
╔══════════════════ OFFLINE — ingestion (idempotent, re-runnable) ═══════════════════╗
║                                                                                     ║
║  corpus/                                                                            ║
║   ├ course_brochure.pdf ──┐                                                         ║
║   ├ placement_policy.pdf ─┤    ┌──────────┐    ┌───────────┐    ┌──────────┐        ║
║   ├ faq.txt ──────────────┼───►│ Loaders  │───►│  Chunker  │───►│ Embedder │        ║
║   ├ refund_terms.txt ─────┤    │ per-type │    │ structure │    │ pluggable│        ║
║   ├ eligibility_*.txt ────┤    └──────────┘    │  -aware   │    └─────┬────┘        ║
║   └ program_pricing.yaml ─┘          │         └─────┬─────┘          │             ║
║                                      ▼               ▼                ▼             ║
║                              RawDoc(text,      Chunk(text,       float32[n,d]       ║
║                               metadata)      + rich metadata)                       ║
╚═════════════════════════════════════════════════════════╪═══════════════════════════╝
                                                          ▼
                              ╔═══════════════════════════════════════════╗
                              ║   INDEX  (index/rag.db  +  index/vecs.npy)║
                              ║  ┌─────────────────────────────────────┐  ║
                              ║  │ chunks     text + metadata + hash   │  ║
                              ║  │ chunks_fts FTS5/BM25 keyword index  │  ║
                              ║  │ vectors    dense matrix (.npy)      │  ║
                              ║  │ traces     one row per query        │  ║
                              ║  └─────────────────────────────────────┘  ║
                              ╚═══════════════════════════════════════════╝
                                                          ▲
╔═════════════════ ONLINE — query (per request) ══════════╪═══════════════════════════╗
║                                                          │                          ║
║  CLI  ──┐                                                │                          ║
║         ├──► ┌───────────┐   ┌──────────────────────────┴──────────────────┐        ║
║  API ───┤    │  Query    │   │  RETRIEVER                                  │        ║
║  (POST  │    │ Expansion │──►│   ├─ dense arm : cosine over embeddings     │        ║
║  /ask)  │    │  (LLM,    │   │   ├─ sparse arm: BM25 over FTS5             │        ║
║         │    │  cached)  │   │   ├─ fuse: Reciprocal Rank Fusion           │        ║
║  UI ────┘    └───────────┘   │   └─ authority-aware re-ordering            │        ║
║                              └───────────────────┬─────────────────────────┘        ║
║                                                  ▼  top-k chunks                    ║
║                              ┌──────────────────────────────────────────┐           ║
║                              │  ANSWERER                                │           ║
║                              │   • build [S1..Sk] labelled context      │           ║
║                              │   • system prompt (§d.1)                 │           ║
║                              │   • LLM w/ structured output             │           ║
║                              │   • post-hoc citation verification       │           ║
║                              └───────────────────┬──────────────────────┘           ║
║                                                  ▼                                  ║
║                    Answer{text, citations[], sufficient_context, ...}               ║
║                                                  │                                  ║
║                              ┌───────────────────▼──────────────────────┐           ║
║                              │  TRACER → traces table + traces.jsonl    │           ║
║                              └──────────────────────────────────────────┘           ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
```

### a.2 How a document becomes an answer — the full path

1. **Load.** A per-type loader turns each file into one or more `RawDoc(text, metadata)`. PDFs are
   parsed page-by-page with `pypdf` so page numbers survive as citation anchors. Text files are
   read whole. YAML is *not* dumped as a string — it is walked and rendered into readable
   sentences (§b.4), because raw YAML embeds and BM25-matches poorly.
2. **Chunk.** The chunker splits on document structure (headings, numbered clauses, Q&A pairs)
   and only falls back to size-based splitting inside an oversized structural unit (§b).
3. **Contextualise.** Every chunk is prefixed with a breadcrumb — `document title > section > page`
   — *inside the embedded text*. A chunk that reads "70% refund, GST fully refunded" is nearly
   meaningless alone; "Refund Terms > §2 The Refund Schedule > 70% refund…" is retrievable.
4. **Embed & index.** Chunks go to SQLite (text + metadata + content hash), to an FTS5 virtual
   table (BM25), and their vectors to a NumPy matrix on disk. Ingestion is idempotent: a chunk
   whose content hash is unchanged is not re-embedded. (This applies to the `fastembed` and
   `openai` backends, whose vectors depend only on the chunk text. The TF-IDF/LSA fallback is
   corpus-derived, so it refits and re-vectorises everything on every ingest — cheap at this size.)
5. **Expand.** At query time the question is optionally rewritten by the LLM into a small set of
   corpus-vocabulary keywords (§c.5). "Can I pay monthly?" → "EMI instalment monthly payment plan
   tenure financing".
6. **Retrieve.** Dense and sparse arms each return a ranked list; RRF fuses them; an authority
   adjustment demotes chunks that self-identify as non-authoritative summaries (§c.6).
7. **Generate.** Top-k chunks are formatted as `[S1]…[Sk]` blocks with provenance headers and sent
   with the system prompt in §d.1, using structured output.
8. **Verify.** Returned citation IDs are checked against the IDs actually supplied. Unknown IDs
   are stripped and the answer is flagged. An answer that claims `sufficient_context: true` but
   cites nothing is downgraded.
9. **Trace.** Everything above — inputs, per-stage latency, scores, prompt size, output, verdicts —
   is written to one trace row (§e).

### a.3 Key design decisions and trade-offs

| # | Decision | Alternative rejected | Why |
|---|---|---|---|
| 1 | **No framework** — plain Python + FastAPI | LangChain / LlamaIndex | The interesting logic here *is* chunking, fusion, and prompt construction. A framework hides exactly what should be inspectable and debuggable, and adds a large dependency tree for ~400 lines of orchestration I can write directly. |
| 2 | **SQLite + NumPy** as the store | Chroma / FAISS / pgvector / Pinecone | At ~150 chunks a brute-force cosine over a NumPy matrix takes microseconds. A vector DB adds a service to run, a schema to sync, and a failure mode — for zero measurable gain. FTS5 also gives me BM25 in the same file for free. Guarded by a `VectorStore` interface so a swap is one file. |
| 3 | **Hybrid retrieval** (BM25 + dense, RRF) | Dense only | The corpus is dense with exact tokens that carry the whole meaning: `ASD-501`, `INR 7,00,000`, `Day 31 to Day 60`, `MA-REFUND-2026-01`. Pure semantic search is notoriously weak on these; BM25 is excellent. The reverse is true for "can I quit and get money back". Both arms are load-bearing. |
| 4 | **Pluggable embedder: open-weight `bge-small` via `fastembed` by default, zero-dependency TF-IDF/LSA fallback, OpenAI opt-in** | Hard-code one embedding API | The provider field in the brief is unfilled (`[OpenAI / Anthropic]`), and **Anthropic ships no embeddings endpoint**, so a hosted embedder cannot be the default. `bge-small` is free, runs on CPU without torch, and understands paraphrase ("quit" ≈ "withdraw") — the one thing a corpus-derived embedder cannot do. The fallback guarantees the system still runs if the one-time model download is blocked. See §c.1. |
| 5 | **LLM query expansion** | Nothing, or a cross-encoder reranker | The sparse arm (BM25) only matches vocabulary that appears in the corpus, and users ask in everyday words while policies are written in defined terms ("Refund Day Count", "Withdrawal Request"). Expansion rewrites the question into corpus vocabulary before BM25 sees it. It also fully protects the fallback embedder, which has the same vocabulary limitation. It costs one small LLM call, is cached, and degrades gracefully to a no-op on failure. |
| 6 | **Authority-aware retrieval** | Treat all chunks equally | The corpus *deliberately* contains non-authoritative summaries that contradict-by-omission the authoritative source. Ignoring this produces exactly the confidently-wrong refund answer the design target forbids. |
| 7 | **Structured output + server-side citation verification** | Parse citations out of prose with a regex | Makes "the answer is grounded" a property the *system* enforces, not a promise the model makes. Turns `sufficient_context` into a machine-readable escalation signal a support tool can route on. |
| 8 | **Provider-detecting LLM client** | Assume Anthropic | Same reason as #4 — it must actually run on the grader's machine. Anthropic (`claude-opus-5`) is the primary, documented path; an OpenAI adapter sits behind the same interface. |

**The trade-offs I am consciously accepting:**

- The dense arm depends on a **one-time ~130 MB model download** (`bge-small`, cached after the
  first run). If that download is blocked, the system silently degrades to the TF-IDF/LSA
  fallback, which captures term co-occurrence, not semantics. The trace records which backend
  actually ran, so a degraded run is visible, not mysterious. **The fallback is the weakest link
  in the system and I say so in §f.3.**
- `bge-small` is a 384-dimensional bi-encoder, chosen for size over ceiling. A larger model
  (`bge-base`, or a hosted embedder) would score higher on paraphrase; on a 150-chunk corpus with
  BM25 alongside it, I do not expect that gap to be measurable at n=12, so I take the smaller
  download.
- Brute-force cosine is O(n·d) per query. Correct here; wrong above ~10⁵ chunks.
- No cross-encoder reranker. It is the standard next step and the single highest-value addition
  (§g.1) — deliberately deferred, not overlooked.
- Everything is in-process and single-node. No queue, no worker pool, no horizontal scaling story.

---

## b. Chunking Strategy

### b.1 The principle

Chunk on **meaning boundaries the author already wrote**, not on a character count. These documents
are heavily structured — numbered clauses, `Q1.1`-style FAQ pairs, `### Module 3` headings, YAML
keys. Those markers are free, high-quality segmentation signals. A blind 1000-character window
would cut §2.2's worked refund example in half, which is precisely the content a support agent
needs whole.

### b.2 The algorithm

```
for each RawDoc:
    units = split_on_structure(doc)          # per-type splitter, see b.4
    for unit in units:
        if size(unit) <= MAX:  emit(unit)
        else:                  emit(*window(unit, MAX, OVERLAP))   # paragraph-aligned
    merge_forward(units, MIN)                # glue runt units to their neighbour
```

Then every emitted chunk is prefixed with a **breadcrumb header** built from its metadata:

```
[Refund Terms and Cancellation Policy | MA-REFUND-2026-01 | §2. THE REFUND SCHEDULE]
2.1 The refund payable on an approved Withdrawal Request is determined solely by …
```

This header is part of the embedded and BM25-indexed text. It is the single highest-leverage line
in the chunker: it repairs the context that splitting destroyed, and it means a query mentioning
"refund policy" matches a chunk whose body never says the word "policy".

### b.3 Size and overlap, and why

| Parameter | Value | Reasoning |
|---|---|---|
| `MAX_CHARS` | **1200** (~300 tokens) | Big enough to hold one complete clause plus its worked example — the atomic unit of a policy answer. Small enough that a top-5 context stays ~1,500 tokens, keeping the model's attention on the relevant span and keeping cost and latency low. |
| `OVERLAP_CHARS` | **180** (15%) | Only applied when a structural unit *must* be force-split. Enough to carry the subject of a sentence across the seam so a fact spanning the boundary is recoverable from either side. More overlap inflates the index and returns near-duplicate chunks; 15% is the usual sweet spot, and below it I expect facts that straddle a seam to be orphaned. That expectation is the first thing §g.3's sweep would test. |
| `MIN_CHARS` | **220** | Below this a chunk is usually a stray heading or a one-line clause that embeds noisily. Such units are merged forward into the next unit rather than indexed alone. FAQ Q&A pairs are exempt: a short pair is indexed as-is rather than glued to its neighbour, because the question line is the retrieval signal and merging two questions blurs it. |
| `target chunks` | est. ~140–180 → **183 measured** | An estimate for this corpus from the structural unit counts (≈50 FAQ pairs, ≈45 refund clauses, ≈30 eligibility blocks, ≈15 YAML records, ≈45 PDF sections). The ingest command prints the real count; the eval run records it. **Measured: 183.** The estimate ran low on the brochure, whose syllabus gives every module its own heading: 39 chunks land at 120–200 chars, each a complete, individually-labelled unit such as `§Module 4 — Databases and SQL`. These sit below `MIN_CHARS` and are deliberately *not* merged — each occupies its own section, so gluing two modules together to satisfy a length floor would cost retrieval precision and buy nothing. `MIN_CHARS` exists to suppress noise, and a labelled module description is not noise. |

These values are **reasoned, not tuned** — this document is written before the code. The eval set
(§f) is the regression net that will tell me if they are wrong, and chunk size is the first thing I
would sweep with more time (§g.3).

### b.4 Handling heterogeneous document types

One `Loader` protocol, four implementations, **one output shape** (`RawDoc`). After the loader
boundary nothing downstream knows or cares what the source format was — that is what keeps the
chunker simple.

| Type | Loader behaviour | Structural split unit | Type-specific concern |
|---|---|---|---|
| **PDF** (`course_brochure.pdf`, `placement_policy.pdf`) | `pypdf`, one `RawDoc` per page; whitespace normalised (including the non-breaking spaces the rendered tables contain — without this, BM25 cannot tokenise figures like `6,40,000`), hyphenated line-breaks rejoined | **Heading lines detected by shape, not by markup** — the PDF text has no `##` markers; those existed only in the source that generated it. A line is a heading if it is short (< 80 chars, no terminal full stop) and matches one of: a program code (`^[A-Z]{3,4}-\d{3}\b`), `^Module \d`, a numbered clause (`^\d+(\.\d+)?\s`), or a known title word (`Syllabus`, `Capstone projects`, `Career outcomes`, `Instructors`, `Certification`). | Page number is captured as citation metadata. A section spanning a page break is stitched by carrying the last-seen heading forward, so a module list that runs onto the next page keeps its program title. Heading detection is a regex list, so a missed heading is a one-line fix, and the ingest command prints the section titles it found per page for eyeballing. |
| **Plain text policy** (`refund_terms.txt`, `eligibility_criteria.txt`) | Read whole, single `RawDoc` | `===` banner sections, then numbered clauses (`2.1`, `5.3`) | Clause numbers are preserved *in* the chunk text — agents cite "§5.3", so the number must be retrievable. Banner rules are stripped from the body but kept as section metadata. |
| **FAQ text** (`faq.txt`) | Read whole, single `RawDoc` | **One chunk per Q&A pair** (`Q3.6 …` through to the next `Q`) | This is the clearest case for structure-aware chunking: a Q&A pair is already a perfectly-sized, self-contained retrieval unit. The question text is a natural paraphrase of what users will ask, so it embeds and BM25-matches extremely well. Never split a Q from its A. |
| **YAML** (`program_pricing.yaml`) | Walked as a tree, **rendered to prose**, one `RawDoc` per top-level program / section | One chunk per program record or per config block | Raw YAML is bad retrieval input — `list_tuition_inr: 210000` shares no tokens with "how much does the foundations course cost". So each record is flattened into a sentence-shaped rendering: *"Program SEF-101, Software Engineering Foundations. Duration 9 months. Level beginner. List tuition INR 210000 exclusive of GST. GST INR 37800…"* The original YAML fragment is kept in metadata for exactness, and the numbers are preserved verbatim so BM25 can still hit `210000`. |

**The cross-cutting concern: document authority.** Each loader stamps its `RawDoc` with an
`authority` level (`authoritative` / `summary`) and a `supersedes` note, derived from the document's
own header (e.g. refund_terms.txt declares *"AUTHORITATIVE. Where this document conflicts with any
summary… this document governs"*, while the FAQ says *"support agents must quote that document, not
this summary"*). That flag flows into retrieval (§c.6) and into the prompt (§d.1 rule 5).

---

## c. Retrieval Design

### c.1 Embedding model choice and rationale

**The constraint that drives this:** the brief specifies one key, `SCALER_LLM_API_KEY`, with the
provider left as `[OpenAI / Anthropic — fill in]`. **Anthropic does not offer an embeddings
endpoint.** So a design that hard-codes a hosted embedder has a coin-flip chance of not running at
all on the grading machine. "Must actually run on this machine without additional setup" is an
explicit requirement, so I treat *runs unconditionally* as a hard constraint and *embedding quality*
as the thing to maximise within it.

**What the corpus needs from an embedder.** The documents are written in defined policy terms
("Withdrawal Request", "Refund Day Count", "Placement Readiness Gate"); users ask in everyday words
("can I quit and get my money back?"). Exact codes and figures (`ASD-501`, `INR 2,10,000`) are
handled by BM25, so the dense arm's one job is **paraphrase**: knowing that "quit" is near
"withdraw" and "money back" is near "refund" *before* seeing this corpus. Only a pretrained model
can do that. That rules out a corpus-derived embedder as the primary and rules in a small
open-weight model.

Resolution: an `Embedder` protocol with three implementations, selected by `EMBEDDING_BACKEND`
(default `auto`, which tries them in the order below and takes the first that loads).

| Backend | Model | Dim | Selected when | Trade-off |
|---|---|---|---|---|
| `fastembed` **(primary)** | `BAAI/bge-small-en-v1.5` via ONNX Runtime | 384 | `fastembed` imports and the model loads | **Free, open-weight, CPU-only, no torch.** ~60 MB of packages plus a one-time ~130 MB model download, cached locally thereafter. Embeds the whole corpus in seconds. Strong pretrained paraphrase understanding, which is exactly the gap this corpus has. |
| `local` **(fallback)** | TF-IDF → truncated SVD (LSA), pure NumPy | `min(128, n_chunks − 1)` | `fastembed` is absent or the model download fails | **Zero dependencies, zero network, deterministic, ~40 ms to fit the whole corpus.** Captures term co-occurrence within *this* corpus only; cannot resolve a paraphrase that shares no vocabulary with the source. |
| `openai` **(opt-in)** | `text-embedding-3-small` | 1536 | `EMBEDDING_BACKEND=openai` is set *and* the key is an OpenAI key | Marginally better than `bge-small`; **paid**, and a network call on every ingest and query. Never auto-selected, because the key may be an Anthropic key. |

**Why `bge-small` is the right primary.** It is a general-purpose English embedding model that
already encodes synonymy and topical similarity, so "can I quit?" lands near "Withdrawal Request"
without any help. It is small enough (384 dimensions, ~130 MB) that the download is a one-time
minute rather than a setup step, and it runs through ONNX Runtime on CPU, so it needs neither
torch nor a GPU. On a 150-chunk corpus it is well past the point of diminishing returns; a larger
model would cost more download for a gain I could not measure at n=12 (§f.3). `fastembed` ships in
`requirements.txt`, so a single `pip install -r requirements.txt` covers it.

**Why the LSA fallback exists and why it is defensible.** The first run needs internet to fetch the
model. If the grading machine is offline or the download is blocked, the system must still answer
questions. TF-IDF → SVD is fitted from the corpus itself in ~40 ms with no dependencies beyond
NumPy. Collapsing to a low-rank space puts "fee", "tuition", "cost" and "INR" on shared axes
*because this corpus uses them together*, which is a real, if shallow, semantic signal. (The
dimension is capped at `n_chunks − 1` because the rank of a ~150-row matrix cannot exceed ~150;
128 is the practical value.) It is also fully deterministic, which makes the eval reproducible.

**The ceiling of the fallback, stated plainly:** it cannot bridge a vocabulary gap. When it is the
active backend, the system leans on the other two mechanisms: BM25 handles exact tokens, and **LLM
query expansion (§c.5) rewrites the user's vocabulary into the corpus's vocabulary before
retrieval**. The trace records `embedding_backend` on every query, so a run on the fallback is
visible and its weaker recall is explained rather than mysterious.

The interface is the point: switching backends is one environment variable and no code change.

### c.2 Vector store choice and rationale

**Chosen: SQLite (chunks + metadata + FTS5) alongside a single NumPy `.npy` matrix, both on local
disk.**

Reasoning, in order of weight:

1. **Scale says so.** ~150 chunks × 384 dims is a ~230 KB matrix. A brute-force `matrix @ query`
   cosine scan is tens of microseconds. An ANN index would be *slower* here (index build + approximation error)
   and is pure operational overhead. Choosing infrastructure the data doesn't need is a real
   engineering error, not a safe default.
2. **Hybrid comes free.** SQLite's FTS5 module is a well-tested BM25 implementation built into the
   standard library. Same file, same transaction, no second system to keep in sync with the vectors.
3. **"Runs without additional setup."** Both are stdlib-or-NumPy. No Docker, no server, no
   migration step, no API key. The index is two files you can delete and rebuild in seconds.
4. **Debuggability.** When a retrieval fails I can open the `.db` and read the exact chunk text and
   metadata. That is worth a lot during a 2.5-hour build and, honestly, in production too.
5. **Traces live in the same database** (§e), so a failure investigation is a single SQL join from
   trace → retrieved chunk → chunk text.

**When this becomes wrong:** past roughly 100k chunks, or when multiple processes need concurrent
writes, or when the index no longer fits comfortably in RAM. The `VectorStore` interface
(`add / search / count`) is deliberately narrow so pgvector or a hosted store is a single-file
change. I would not make that change before measuring it.

### c.3 Similarity metric and why

**Cosine similarity**, on L2-normalised vectors — implemented as a plain dot product, since after
normalisation the two are identical and the dot product is one BLAS call.

Why cosine:

- **It measures direction, not magnitude.** Chunk lengths here vary by ~5× (a one-line YAML record
  vs. a full clause with a worked example). Raw dot product would systematically favour longer
  chunks simply for having more terms — a length bias with no semantic justification. Cosine
  removes it.
- **It is the metric these representations are built for.** `bge-small` is trained with a
  contrastive objective on normalised vectors, so cosine is the similarity it was optimised for.
  TF-IDF/LSA vectors are conventionally compared by cosine, and every mainstream embedding API
  (OpenAI, Voyage) documents cosine as the intended metric. Using anything else discards the
  geometry the representation was fit under.
- **Euclidean distance is equivalent-but-worse here.** On normalised vectors
  `‖a−b‖² = 2(1−cos)`, so it produces the *same ranking* while being harder to threshold and
  interpret — cosine's `[-1, 1]` range makes a score readable in a trace at a glance.

Scores are surfaced in the trace and the API response, so a low-confidence retrieval is visible
rather than silently averaged away.

**Fusion metric — Reciprocal Rank Fusion.** The dense arm returns cosine in `[0,1]`; BM25 returns
an unbounded, corpus-dependent score. These are not comparable, and normalising them requires
weights I have no principled way to pick and no data to fit. RRF sidesteps the problem by
discarding magnitudes and using only **rank**:

```
RRF(chunk) = Σ over arms  1 / (K + rank_in_arm)      with K = 60
```

It is parameter-light (`K=60` is the standard published value), robust to one arm being badly
calibrated, and rewards chunks that both arms agree on — exactly the behaviour I want. Each arm
contributes its top 20; the fused list is truncated to `TOP_K = 5`.

### c.4 Retrieval parameters

| Parameter | Value | Why |
|---|---|---|
| `CANDIDATES_PER_ARM` | 20 | Wide enough that RRF has genuine signal to fuse; cheap at this corpus size. |
| `TOP_K` | 5 | ~1,500 tokens of context. Beyond ~6 chunks I expect precision to fall and the model to start citing marginally-relevant sources. The context-precision metric in §f is the signal I would tune it against. |
| `RRF_K` | 60 | Standard value from the original RRF paper; not tuned, and I would not tune it without more eval data than 12 cases. |
| `MISS_DENSE_COSINE` | **0.70 measured** | Part of miss detection (below). Set by inspecting the top cosine for the abstention cases (§f cases 10–11) against the answerable cases; logged per query as `top_score` so it can be re-calibrated from traces. **Measured on the eval set: answerable `min 0.6463 / median 0.7465 / max 0.8608`, abstention `min 0.6405 / median 0.6577 / max 0.6750`. The two clusters overlap, so no dense floor separates them alone** — bge-small compresses cosines into a narrow band. This is not a failure of the parameter but the reason miss detection is an AND: the *precise* signal is the sparse one, and the dense floor only has to avoid vetoing it. 0.70 clears the one abstention case that also trips `sparse_miss` (0.675). Two answerable cases sit just above at 0.706/0.708 and one falls below at 0.646 — all three are protected by matching content terms, so they can never satisfy the AND. `eval/run_eval.py --no-judge` recomputes the safe range on every run. |

**Why there is no floor on the fused score.** An obvious design — "drop chunks whose RRF score is
below a threshold" — does not work, and I want to say why rather than ship it. Both arms *always*
return their top 20, and RRF uses only rank positions, so a question the corpus cannot answer
produces the same score distribution as one it can. The fused score carries no information about
absolute relevance. Miss detection therefore has to look at the arms' **raw** signals before
fusion:

- **Dense arm:** the top cosine is below `MISS_DENSE_COSINE`.
- **Sparse arm:** FTS5 matched **no** non-stopword term of the (expanded) query, or its top BM25
  score is below a small floor. "Stopword" here includes a short **domain stopword list**
  (`meridian`, `academy`, `program`, `course`, `learner`, `offer`, `support`) — otherwise almost
  every question matches *something*, since nearly every chunk contains "program" or "Meridian",
  and the sparse condition would never fire.

A query is a **hard miss** only when *both* conditions hold — one arm alone is not trusted to
declare a miss, because each has known blind spots (§c.1). Everything else goes to the model with
the scores attached, and the prompt's abstention clause handles the marginal cases (§d.3).

### c.5 Query expansion

One cheap LLM call rewrites the question into corpus vocabulary before retrieval:

> `"can I get my money back if I drop out after two months"`
> → `withdrawal refund cancellation Refund Day Count tuition refunded Day 61 no refund`

The expanded string is used for the **sparse arm only** (BM25 benefits most from vocabulary
overlap); the dense arm sees the original question, so an expansion that goes off-target cannot
poison both arms at once. The call is cached by question hash, has a short timeout, and **fails
open** — on any error, expansion is skipped and retrieval proceeds with the raw query. It is an
optimisation, never a dependency.

**Two things expansion must not touch, both found by measurement.**

*It must not feed miss detection.* The sparse miss-condition asks whether the user's words appear
in the corpus. Expansion's entire job is to inject corpus vocabulary, so testing the *expanded*
query against the corpus is circular — it always matches, and miss detection silently stops working
the moment expansion is enabled. Measured: *"Does Meridian offer a cybersecurity program?"*
hard-misses on the raw query and does not on the expanded one. §c.4's conditions therefore read the
original question only.

*It must not run on the hard-miss path.* Expansion is itself an LLM call, measured at ~3.8 s. Since
miss detection reads the raw query, the hard-miss verdict is already exact after the dense search
and the term probe — before any expansion. Passing the expansion into the retriever as a *callable*
rather than a value lets it be skipped entirely when the verdict is already known. Without this,
short-circuiting spent an LLM call to avoid an LLM call, inverting the whole point of §d.3:
measured, the hard-miss path fell from **4,272 ms to 105 ms** and from one billed call to none.

### c.6 Authority-aware ordering

**The rule, stated so it can be implemented.** "Overlapping topic" is not something the code can
judge, so I do not ask it to. Instead I use a signal the corpus hands me for free: **every summary
in this corpus names the document it defers to.** FAQ Q6.1 says "the authoritative and complete
terms… are in the Refund Terms document (MA-REFUND-2026-01)"; Q3.1 defers to `MA-PRICING-2026-03`;
Q5.1 and Q3.8 to `MA-PLACE-2026-02`; Q2.2 to `MA-ELIG-2026-01`; the YAML `refund_reference` block
to `MA-REFUND-2026-01`. At ingest, the loader extracts any document ID of the form
`MA-[A-Z]+-\d{4}-\d{2}` from a chunk's text and stores it as `defers_to`. Then:

```
for chunk in fused_results:
    if chunk.defers_to and any(other.doc_id == chunk.defers_to for other in fused_results):
        chunk.rrf_score *= AUTHORITY_PENALTY      # 0.8 — reorders, never filters
```

A summary is demoted **only** when the document it explicitly points at is already in the result
set. A FAQ chunk that is the sole source for its fact (support hours, MAT retake rules, cohort
size) carries no `defers_to`, or its target is not retrieved, so it is never penalised. Both
chunks remain in context; the penalty reorders, and the `authority` flag and `defers_to` ID are
printed in the `[Sn]` provenance header so the model can apply prompt rule 5.

**Document-level defaults.** Not every document declares its own status, so the loader assigns
one: `refund_terms.txt`, `placement_policy.pdf`, and `program_pricing.yaml` say "authoritative" /
"single source of truth" in their headers and are stamped `authoritative`. `eligibility_criteria.txt`
has no such line but is the only source for its content, so it is `authoritative` too.
`course_brochure.pdf` is stamped `reference` and, per its own "Next steps" section, carries a
document-level `defers_to` of the pricing, eligibility, and placement documents — so a brochure
pricing line yields to the pricing YAML when both are retrieved. FAQ chunks are `summary` only when
they carry a `defers_to`; otherwise `reference`.

**The failure this is built to prevent** (a prediction, to be confirmed once the index exists):
*"what is the refund policy?"* is lexically closest to FAQ Q6.1, whose question line literally
contains "refund policy", so I expect it to outrank Refund Terms §2 in both arms. Q6.1 omits the
GST treatment that §2 carries. Both are "relevant"; only one is safe to quote. Eval case 7 (§f.1)
is the regression test for exactly this.

---

## d. Answer Generation

### d.1 Prompt design

Context is assembled as numbered blocks, each with a full provenance header — this is what makes
citation possible *and* verifiable:

```
[S1] course_brochure.pdf | Meridian Academy Course Brochure 2026 | MA-BROCHURE-2026-02
     § SEF-101 — Software Engineering Foundations | page 3 | authority: authoritative
SEF-101 is our entry program and the only Meridian program that requires no degree …

[S2] faq.txt | Learner Support FAQ | SUP-FAQ-2026-03
     § SECTION 6: REFUNDS, WITHDRAWALS, AND CANCELLATIONS | authority: summary | defers to: MA-REFUND-2026-01
Q6.1 What is your refund policy in brief? …
```

**The exact system prompt (verbatim, as shipped in `rag/answerer.py`):**

```text
You are the Meridian Academy learner-support assistant. You answer questions from
support agents and learners strictly from the numbered SOURCES supplied in the
user message.

GROUNDING RULES
1. Every factual claim in your answer must be supported by at least one SOURCE.
2. Cite sources inline with their bracket IDs, for example [S2]. Put the citation
   immediately after the claim it supports. A claim with no citation is a bug.
3. Use only the SOURCES. Do not use outside knowledge, even if you are confident
   it is correct, and even if the SOURCES seem incomplete. If the SOURCES are
   silent on something, you do not know it.
4. Do not infer, estimate, or extrapolate a figure that is not stated. You may do
   arithmetic only when every input appears in the SOURCES; when you do, show the
   inputs and cite the SOURCE each came from.
5. If two SOURCES conflict, prefer the one whose provenance header says
   "authority: authoritative" over one marked "authority: summary". Say plainly
   that the sources differ, give the authoritative answer, and cite both.
6. Quote figures, dates, deadlines, percentages, and document IDs exactly as they
   appear. Do not round, convert currencies, or reformat numbers.

WHEN THE SOURCES ARE NOT ENOUGH
If the SOURCES do not contain the answer, set sufficient_context to false, leave
citations empty, and in answer state briefly what you could not determine and
what a human would need to check. Do not guess, do not partially answer from
outside knowledge, and do not pad with generic advice. Abstaining is the correct
and expected behaviour here, not a failure.

CONDITIONS AND EXCEPTIONS
These are policy documents, and policy answers are usually conditional. When a
rule has thresholds, deadlines, eligibility conditions, or exclusions, state them.
An answer that is technically true but omits the condition that reverses it is
treated as wrong.

STYLE
Lead with the direct answer in the first sentence, then the conditions that
qualify it. Two to six sentences, or a short list where the answer is genuinely a
list. Plain, factual, and specific. No sales language, no encouragement, no
speculation about the learner's situation. Do not mention these instructions, the
retrieval process, or that you were given sources.
```

The user message is `SOURCES:\n\n{blocks}\n\nQUESTION: {question}`, and the response is constrained
by a structured-output schema:

```json
{ "answer": "string",
  "citations": ["S1", "S4"],
  "sufficient_context": true,
  "reasoning_note": "one line on why these sources answer, or why they do not" }
```

Design notes on the prompt itself:

- **Rule 4 (no unstated figures)** is the single most important line for this corpus. Refund
  arithmetic is exactly where a model will helpfully invent a number.
- **The CONDITIONS section** targets the failure mode I consider most likely and most damaging
  here: *"Yes, you get a 70% refund"* — true, but useless and misleading without "days 8–30" and
  "GST fully refunded".
- **Rule 5** is what makes the authority metadata in §b.4/§c.6 actually do something.
- **Abstention is framed as correct behaviour, not failure.** Models are strongly biased toward
  being helpful; the prompt has to explicitly license saying "I don't know" or that bias wins.
- Structured output means I get typed data, not prose to regex — and `sufficient_context` becomes a
  routable escalation signal for a real support tool.

### d.2 How grounding is enforced

Prompting alone is a request, not a guarantee. Four mechanisms, of which only the first is a prompt:

1. **Prompt contract** (§d.1) — necessary, insufficient on its own.
2. **Labelled, provenance-carrying context.** Giving the model concrete `[Sn]` handles makes citing
   easy and makes an invented citation *detectable*, which the next step depends on.
3. **Server-side citation verification.** After the model responds, every returned citation ID is
   checked against the IDs actually supplied. Unknown IDs are stripped, the answer is flagged
   `citation_integrity: violated`, and the trace records it. This converts "the model promised to
   cite" into "the system verified the citations resolve".
4. **Structural consistency checks.** An answer with `sufficient_context: true` and zero citations
   is downgraded to `sufficient_context: false` and marked `unsupported_claim` — the model has
   asserted something it did not attribute, which by rule 2 is a bug.

Plus, at the boundary: the raw-signal miss detection in §c.4 means a question with no real match
never reaches the model at all, so the *retriever* forces abstention before the generator gets a
chance to improvise.

**What this does not catch:** an answer that cites `[S1]` where `S1` exists and is topically
related but does not actually support the specific claim. That is a semantic check, and it is
exactly what the LLM-judge faithfulness metric in §f measures offline. Verification catches
structural hallucination cheaply and synchronously; the judge catches semantic hallucination
thoroughly and asynchronously. Both are needed.

### d.3 When no relevant context is found

Three distinct cases, handled distinctly — because "we have nothing" and "we have something
marginal" deserve different behaviour:

| Case | Trigger | Behaviour |
|---|---|---|
| **Hard miss** | Top dense cosine below `MISS_DENSE_COSINE` **and** BM25 matched no non-stopword query term (§c.4) | Short-circuit. **No LLM call at all** — saves latency and cost, and removes any chance of improvisation. Return a fixed, honest message with `sufficient_context: false`, `citations: []`, `no_context: true`. |
| **Weak retrieval** | Chunks retrieved, all below a confidence floor | Call the model normally. The prompt's abstention clause usually produces a correct "the sources don't cover this". The low scores are in the trace and the response, so the caller can see the answer rests on weak retrieval. |
| **Model abstains** | Model returns `sufficient_context: false` | Pass through verbatim, still return the retrieved sources so a human can judge for themselves, and mark the response for escalation. |

In every case the API returns HTTP 200 with a well-formed body — an abstention is a valid,
successful answer, not an error — and sets `escalate: true`. That field is the product feature this
whole section exists to produce: in a real deployment it is what routes the ticket to a human agent
instead of leaving a learner with a confident fabrication about their refund.

---

## e. Instrumentation Design

### e.1 What is logged per query, and why each field earns its place

One row per query in the `traces` SQLite table **and** one JSON object per line in
`logs/traces.jsonl`. SQLite for querying and joining; JSONL for grepping, tailing, and shipping to
a log pipeline unchanged.

| Field | Why it is there |
|---|---|
| `trace_id` (uuid), `timestamp_utc` | Correlate a user complaint to a specific execution. The first question in any debug is "which request?" |
| `query` (verbatim) | The only way to reproduce. Never normalised before logging — the normalisation is itself a suspect. |
| `expanded_query` | Query expansion is a common silent failure. If expansion drifted off-topic, the bad retrieval is explained immediately. |
| `retrieved[]`: `chunk_id`, `doc`, `section`, `page`, `dense_score`, `dense_rank`, `bm25_score`, `bm25_rank`, `rrf_score`, `final_rank`, `text_preview` | **The core diagnostic payload.** Per-arm scores *and* ranks are logged separately, not just the fused result — that is what makes it possible to tell *which arm* failed (§e.2). |
| `retrieval_mode`, `embedding_backend` | Which code path actually ran. Essential when behaviour differs between machines because the embedder auto-selected differently. |
| `prompt_token_estimate`, `prompt_char_len`, `prompt_sha256`, `n_sources` | The full prompt can be large, so I log a size, a hash, and the source list by default; `TRACE_FULL_PROMPT=1` logs it verbatim. The hash alone proves whether two runs sent identical prompts — which settles "is this a retrieval bug or a model nondeterminism bug" in one query. |
| `answer`, `citations[]`, `sufficient_context`, `reasoning_note` | The output, and the model's own account of it. |
| `citation_integrity` (`ok`/`violated`), `invalid_citations[]`, `unsupported_claim` | The §d.2 verifier verdicts. A rising rate here is a live hallucination alarm. |
| `latency_ms` **total**, plus `latency_expand_ms`, `latency_embed_ms`, `latency_retrieve_ms`, `latency_llm_ms`, `latency_verify_ms` | End-to-end latency answers "is it slow"; the breakdown answers "which stage". Without the split, a p99 regression is an unbounded investigation. |
| `input_tokens`, `output_tokens`, `model`, `est_cost_usd` | Cost per query, attributable per query. Needed the moment anyone asks whether this is affordable at 500 tickets/day. |
| `no_context`, `escalate` | The abstention path. **The escalation rate is the single most important product metric here** — it is the honest measure of corpus coverage. |
| `error`, `error_stage` | Failures are traced with the same schema as successes, so a failure is never a gap in the log. |
| `top_score`, `score_gap` (top1 − top2) | A cheap, powerful confidence proxy. A high top score with a large gap is a clean hit; a flat distribution across five chunks means the retriever is guessing. Alertable without an LLM judge. |

### e.2 Debugging a retrieval failure in production

Concretely: a support agent reports that *"Is GST refunded if I withdraw after 40 days?"* got a
wrong answer.

1. **Find the trace.** `SELECT * FROM traces WHERE query LIKE '%GST%' ORDER BY timestamp DESC` —
   or straight by `trace_id` if the UI surfaced it (it does; it's in the response body).
2. **Split the failure in two: retrieval or generation?** Read `retrieved[]`. Did Refund Terms §4.3
   ("GST is not refundable for a Refund Day Count of 31 or more") make it into the top-5?
   - **It is in the sources, but the answer is still wrong → generation failure.** Check
     `citation_integrity` and which `[Sn]` was cited. If it cited the FAQ summary over the
     authoritative clause, that is a prompt rule-5 failure or an authority-flag failure. If it cited
     the right chunk and still misread it, that is a prompt problem — reproduce offline with
     `prompt_sha256` and iterate on the prompt with the eval suite as a regression net.
   - **It is not in the sources → retrieval failure.** Continue to 3.
3. **Isolate which arm failed**, using the per-arm ranks:
   - Good `bm25_rank`, bad `dense_rank` → the embedder missed. First read `embedding_backend`
     in the same trace. If it says `local`, the fallback was running (the `bge-small` download
     failed on that machine) and the fix is restoring `fastembed`, not touching retrieval. If it
     says `fastembed`, the miss is a genuine bi-encoder limitation and the fix is a reranker
     (§g.1) or a larger model.
   - Good `dense_rank`, bad `bm25_rank` → vocabulary mismatch. Look at `expanded_query`: did
     expansion fire, and did it produce the right terms? Often the fix is in expansion, not
     retrieval.
   - **Bad in both arms** → the chunk itself is the problem. Go to 4.
4. **Inspect the chunk.** Join the trace to the `chunks` table.
   `SELECT text, section, char_len FROM chunks WHERE doc='refund_terms.txt' AND text LIKE '%GST%'`.
   The usual culprits, in the order I would check them: the clause got split across two chunks so
   neither contains the full statement; the breadcrumb header is missing or wrong so the chunk lost
   its topic; or the chunk is a 1200-char wall in which the GST sentence is diluted below the
   retrieval threshold. Each has a different fix, and the chunk text tells you which.
5. **Turn it into a regression test.** Add the question to `eval/test_cases.json` with the expected
   source, re-run `eval/run_eval.py`, fix, and confirm the metric moves — and that nothing else
   regressed. **A retrieval bug that does not become a test case will come back.**
6. **Check whether it is systemic.** `SELECT AVG(top_score), AVG(escalate) FROM traces WHERE
   timestamp > date('now','-1 day')` against the prior week. A corpus update that broke ingestion
   shows up here as a fleet-wide drop long before individual complaints do.

The design principle: **the trace must contain enough to reproduce the failure without the user.**
Per-arm ranks and `prompt_sha256` are in the schema specifically so that step 3 is a lookup rather
than a re-run.

---

## f. Evaluation Design

### f.1 How I know the system works

Answer quality is a product of two independently-failing stages, so measuring only the final answer
tells me *that* something broke, not *what*. **The evaluation therefore measures retrieval and
generation separately, then end-to-end.** If end-to-end accuracy drops, the component metrics
localise it immediately.

`eval/test_cases.json` holds **12 cases** (the brief asks for ≥8), hand-written to cover the corpus
and, deliberately, its hard edges:

| # | Case type | Example | What it probes |
|---|---|---|---|
| 1–5 | Single-document factual lookup | "What is the minimum MAT score for DSML-301?" | Baseline retrieval + extraction |
| 6 | **Cross-document synthesis** | "I'm on a 35% scholarship for FSWD — am I eligible for Placement Assurance?" | Must join pricing YAML (stacking cap) with placement policy §5.3(e) |
| 7 | **Authority conflict** | "What's the refund if I withdraw on day 45?" | Must prefer Refund Terms §2 over the FAQ summary, and must include the GST exclusion |
| 8 | **Numeric / computational** | "I paid the full SEF-101 fee upfront at list price, no discount, and I'm withdrawing on day 22 — how much do I get back?" | Arithmetic strictly from stated inputs (the worked example in Refund Terms §2.2: INR 1,84,800). The question pins "no discount" deliberately: elsewhere in the corpus "paid upfront" implies the 10% upfront discount (FAQ Q3.5, Refund Terms §6.3), which would give a different figure. A question with two defensible answers cannot fairly pass or fail, so the eval removes the ambiguity — and a separate `expected_facts` entry checks the answer does *not* silently apply the discount. |
| 9 | **Negative / exclusion** | "Does ASD-501 include placement support?" | Must answer *no* confidently — models are biased toward yes |
| 10 | **Out-of-corpus (abstention)** | "Does Meridian offer a cybersecurity program?" | **Must abstain.** `expect_abstain: true` |
| 11 | **Out-of-corpus, adjacent (abstention)** | "What is the placement rate for the 2026 cohort?" | Corpus has 2025 figures only — must not silently substitute them |
| 12 | **Conditional / multi-clause** | "Can I get a refund if I'm expelled for plagiarism?" | Must find §8.1's override of the normal schedule |

Cases 10 and 11 matter disproportionately: a system that answers everything scores well on
naive accuracy and is dangerous in production. **Abstention accuracy is a first-class metric.**

### f.2 Metrics, and how each is computed

Five metrics, run by `eval/run_eval.py`, reported per-case (pass/fail) and aggregated.

**1. Retrieval Recall@k — deterministic, no LLM.**
Each case names the `expected_source` document (and where meaningful, the section). Pass if any
top-k chunk comes from it.
`recall@5 = |cases where expected doc ∈ retrieved| / |cases|`
Cheap, fast, fully reproducible. It is the first thing to check on any regression because
generation quality is capped by it.

**2. Answer Correctness — LLM judge.**
Each case carries `expected_facts`: a list of atomic facts the answer must contain (e.g.
`["40%", "GST is not refunded"]`). The judge is asked, per fact, whether the answer asserts it.
Pass requires all required facts present. Scored by a judge rather than string matching because
"40%" may legitimately appear as "forty percent" or "0.4 of tuition".

**3. Faithfulness — LLM judge.** *(the brief's suggested metric)*
Given **only** the retrieved context and the answer — not the expected answer, not the question's
ground truth — the judge classifies each claim in the answer as `supported`,
`contradicted`, or `not_in_context`.
`faithfulness = supported_claims / total_claims`, and any `contradicted` claim fails the case
outright. This measures hallucination *relative to what the model was actually shown*, which is the
property §d.2 cannot check structurally. A system can be faithful and wrong (good retrieval is a
separate problem) — which is precisely why this is separate from metric 1.

**4. Context Precision — LLM judge.** *(the brief's other suggested metric)*
For each retrieved chunk, the judge answers: *was this chunk actually useful in answering the
question?*
`context_precision = useful_chunks / retrieved_chunks`
This is my tuning signal for `TOP_K` and chunk size. Recall@k tells me I *found* the answer;
precision tells me how much noise I paid to get it. Rising recall with falling precision means
`TOP_K` is too high and I am diluting the model's attention.

**5. Abstention Accuracy — deterministic.**
For `expect_abstain: true` cases, pass iff `sufficient_context == false`. For all others, pass iff
`sufficient_context == true`. Catches both failure directions: hallucinating on an unanswerable
question, and over-abstaining on an answerable one.

Also reported, not pass/fail: p50 and p95 latency, mean tokens and estimated cost per query,
mean `top_score` and `score_gap`. A quality gain paid for with a 10× latency increase is a
trade-off someone should get to make explicitly.

**Judge design.** The judge uses the same `SCALER_LLM_API_KEY`, with structured output and the
most deterministic settings the provider allows (current Anthropic models reject sampling
parameters such as temperature, so determinism comes from a fixed rubric, a low effort setting, and
a strict output schema rather than from `temperature=0`), and is given a rubric with a worked
example per metric. Critically, the judge is given
**only what it needs**: the faithfulness judge never sees the expected answer, so it cannot grade
"is this right" when asked "is this supported". Each judgement returns a verdict plus a one-line
justification, which is written into `eval/results/` so a surprising score can be inspected rather
than trusted.

### f.3 Limits of this evaluation approach — read this before trusting the numbers

I would present these limits alongside any score, because a metric whose weaknesses are unstated
invites over-trust:

1. **Twelve cases is a smoke test, not a measurement.** At n=12, one case flipping moves the
   headline number by 8 points. These metrics are sensitive enough to catch a *broken* change and
   far too noisy to adjudicate a *marginal* one. I would not tune a parameter on a 4-point
   difference here. Confidence intervals would be wider than most of the effects I care about.
2. **The test set is written by the system's author, from the corpus.** I know what the documents
   say, so my questions are unconsciously shaped by their vocabulary and structure — the single
   most common way RAG evals flatter themselves. Real support tickets are messier: typos,
   under-specified, multi-part, and framed in the learner's words rather than the policy's. The
   corrective is real logged questions (§g.2), not more author-written ones.
3. **The judge is the same model family as the system under test.** LLM judges are known to favour
   outputs resembling their own, and a shared blind spot — a misread clause both the answerer and
   the judge make identically — is invisible to this setup. Mitigations: deterministic judge settings, judge sees
   only what it needs, justifications logged for spot-checking. A real fix is a different judge
   model and periodic human agreement sampling (§g).
4. **`expected_facts` encodes my reading of the corpus as ground truth.** If I misread a clause, the
   eval enforces my error. A second reader is the only real fix.
5. **Faithfulness and correctness are not the same, and neither alone is sufficient.** A perfectly
   faithful answer to badly-retrieved context is confidently wrong. That is why metric 1 is
   reported alongside, and why I would never publish a faithfulness score on its own.
6. **Nothing here measures what actually matters commercially:** did the agent resolve the ticket
   faster, and did the answer avoid creating a downstream complaint? Offline metrics are a proxy for
   that, and a loose one. The real measurement is a human-rated sample of production traces, plus
   ticket-resolution time.
7. **No adversarial or safety testing.** No prompt-injection cases (a hostile question trying to
   override the system prompt), no PII handling cases, no multi-turn coherence. Out of scope at
   2.5 hours; in scope for production.

Given those limits, the way I would actually use this suite is as a **regression gate**, not a
leaderboard: it must not go down, and every production failure becomes a new case (§e.2 step 5).

---

## g. What I'd do with more time

**1. A cross-encoder reranker between retrieval and generation.** *(highest value)*
Retrieve 20 candidates, then score each `(query, chunk)` pair jointly — with a cross-encoder model
or a cheap LLM call — and keep the best 5. Bi-encoder retrieval compresses the query and the chunk
into vectors *independently*, so it cannot model their interaction; a cross-encoder sees both
together and is dramatically better at precision. This is the standard fix for the failure mode I
am most exposed to: a small 384-dimensional bi-encoder ranking a topically-similar chunk above the
one that actually answers the question, and, when the fallback embedder is active, missing
paraphrases outright. It would let me raise recall by retrieving wider without paying for it in
context precision. I deferred it because it roughly doubles per-query latency and I wanted the
baseline measured first — but it is the first thing I would build on Monday.

**2. Close the loop with production traces.** The instrumentation in §e is already collecting
exactly what is needed; nothing currently consumes it. I would build: a weekly job that clusters
queries where `escalate = true` or `top_score` is low, surfacing *what the corpus does not cover* —
the highest-value output of the whole system, since it tells the content team which document to
write next. Alongside it, a thumbs-up/down control in the agent UI written back to the trace row,
and a promotion path where any thumbs-down becomes a candidate eval case after review. That turns
§f.3's "the author wrote the test set" problem into "production wrote the test set", which is the
only real fix for it. I would also add drift alerting on escalation rate and mean `top_score`.

**3. Systematic tuning of chunking and retrieval, on an eval set large enough to trust.** Right now
`MAX_CHARS=1200`, `OVERLAP=180`, `TOP_K=5`, and the authority penalty are reasoned choices
validated against 12 cases — which, per §f.3, cannot distinguish a real 4-point gain from noise. I
would grow the set to 80–100 cases (mostly from #2), then sweep chunk size, overlap, `TOP_K`, and
candidate depth against context precision and recall together, and hill-climb honestly with
confidence intervals. I would also test two structural variants I suspect would win: **small-to-big
retrieval** (embed small precise chunks, then expand to the parent section before generation, which
decouples retrieval granularity from generation context), and **a query router** that skips the
LLM entirely for exact-lookup questions like "what is the fee for DCE-401" that the pricing YAML
answers deterministically — faster, cheaper, and impossible to hallucinate.

*(Runners-up, in order: swapping `bge-small` for `bge-base` or a hosted embedder and measuring
the delta — one env var, already supported; a second judge model plus human agreement sampling to
de-bias §f;
streaming responses over SSE for perceived latency; prompt-injection hardening on the retrieved
content.)*
