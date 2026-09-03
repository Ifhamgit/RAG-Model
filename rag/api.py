"""FastAPI surface (DESIGN.md §d.3, §e.2).

The rule that shapes this module: **an abstention is a successful answer.**
"The corpus does not cover this" is the correct response to an unanswerable
question, so it returns HTTP 200 with `escalate: true` — the field a real
support tool routes on. Only a genuine fault (index missing, LLM unreachable)
is a 5xx, and it still carries a trace_id so the failure can be looked up.

The engine — embedder, index, HTTP client — is built once in the lifespan
handler. Per-request construction would put ~700 ms of model load into every
query.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import settings
from .llm import LLMError
from .pipeline import QueryEngine, corpus_fingerprint

log = logging.getLogger(__name__)

_engine: Optional[QueryEngine] = None


# --------------------------------------------------------------------------
# boundary models
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceOut(BaseModel):
    id: str                       # the [Sn] label the answer cites
    chunk_id: str
    doc: str
    doc_id: str
    section: str
    page: Optional[int] = None
    authority: str
    defers_to: list[str] = []
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf_score: float = 0.0
    authority_demoted: bool = False
    cited: bool = False
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    sources: list[SourceOut]
    sufficient_context: bool
    escalate: bool
    no_context: bool
    citation_integrity: str
    invalid_citations: list[str]
    unsupported_claim: bool
    reasoning_note: str
    trace_id: str
    expanded_query: str
    top_score: Optional[float]
    score_gap: Optional[float]
    latency_ms: float
    latency_breakdown: dict[str, float]
    input_tokens: int
    output_tokens: int
    model: str
    embedding_backend: str


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    # require_llm=False so the service still starts for /health and /traces when
    # no key is configured; /ask then fails with a clear 503 rather than the
    # process refusing to boot.
    _engine = QueryEngine(settings, require_llm=False)
    log.info("api ready: %d chunks, %s", _engine.store.count(), _engine.embedder.name)
    try:
        yield
    finally:
        _engine.close()
        _engine = None


app = FastAPI(
    title="Meridian Academy RAG",
    description="Grounded, cited Q&A over the Meridian Academy corpus. See DESIGN.md.",
    version="1.0.0",
    lifespan=lifespan,
)

# Local development only. A real deployment would name its origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _engine_or_503() -> QueryEngine:
    if _engine is None:
        raise HTTPException(503, "engine not initialised")
    if _engine.store.count() == 0:
        raise HTTPException(503, "index is empty — run `python main.py --ingest`")
    return _engine


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    engine = _engine_or_503()
    if engine.client is None:
        raise HTTPException(
            503, "no LLM client configured — set SCALER_LLM_API_KEY to answer questions"
        )

    try:
        r = engine.ask(req.question)
    except LLMError as exc:
        # A genuine fault, unlike an abstention. 502 because the failure is
        # upstream, not in the caller's request.
        raise HTTPException(502, f"LLM failure [{exc.stage}]: {exc}") from exc

    cited = set(r.citations)
    sources = [
        SourceOut(
            id=f"S{i}",
            chunk_id=s.chunk.chunk_id,
            doc=s.chunk.source_file,
            doc_id=s.chunk.doc_id,
            section=s.chunk.section,
            page=s.chunk.page,
            authority=s.chunk.authority,
            defers_to=list(s.chunk.defers_to),
            dense_score=s.dense_score,
            dense_rank=s.dense_rank,
            bm25_score=s.bm25_score,
            bm25_rank=s.bm25_rank,
            rrf_score=s.rrf_score,
            authority_demoted=s.authority_demoted,
            cited=f"S{i}" in cited,
            text=s.chunk.body,
        )
        for i, s in enumerate(r.sources, start=1)
    ]

    # Note the status code: 200. An abstention is a correct, successful answer
    # (§d.3); `escalate` is what tells the caller to route it to a human.
    return AskResponse(
        answer=r.answer,
        citations=r.citations,
        sources=sources,
        sufficient_context=r.sufficient_context,
        escalate=r.escalate,
        no_context=r.no_context,
        citation_integrity=r.citation_integrity,
        invalid_citations=r.invalid_citations,
        unsupported_claim=r.unsupported_claim,
        reasoning_note=r.reasoning_note,
        trace_id=r.trace_id,
        expanded_query=r.expanded_query,
        top_score=r.top_score,
        score_gap=r.score_gap,
        latency_ms=round(r.latency_ms, 1),
        latency_breakdown={
            "expand_ms": round(r.latency_expand_ms, 1),
            "embed_ms": round(r.latency_embed_ms, 1),
            "retrieve_ms": round(r.latency_retrieve_ms, 1),
            "llm_ms": round(r.latency_llm_ms, 1),
            "verify_ms": round(r.latency_verify_ms, 1),
        },
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        model=r.model,
        embedding_backend=r.embedding_backend,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """What actually loaded on this machine.

    Deliberately reports the embedding backend and the corpus fingerprint: the
    backend auto-selects (§c.1), so a reviewer can confirm which one is running
    without reading logs, and the fingerprint says whether the index was built
    from the corpus currently on disk.
    """
    if _engine is None:
        return JSONResponse({"status": "starting"}, status_code=503)

    meta = _engine.store.all_meta()
    try:
        live = corpus_fingerprint(settings.corpus_dir)
    except Exception:
        live = ""
    indexed = meta.get("corpus_fingerprint", "")

    return {
        "status": "ok" if _engine.store.count() else "empty index",
        "chunks": _engine.store.count(),
        "embedding": {
            "backend": _engine.embedder.name,
            "dim": _engine.embedder.dim,
            "indexed_with": meta.get("embedding_backend"),
        },
        "llm": {
            "provider": settings.resolved_provider,
            "model": settings.resolved_model,
            "configured": _engine.client is not None,
        },
        "index": {
            "built_at": meta.get("built_at"),
            "source_files": meta.get("source_files", []),
            "chunking": meta.get("chunking"),
            "corpus_fingerprint": indexed[:16],
            "corpus_matches_index": bool(live and indexed and live == indexed),
        },
        "retrieval": settings.describe()["retrieval"],
    }


@app.get("/traces")
def traces(limit: int = Query(20, ge=1, le=500)) -> list[dict[str, Any]]:
    engine = _engine_or_503()
    rows = engine.store.recent_traces(limit)
    return [
        {
            "trace_id": t["trace_id"],
            "timestamp_utc": t["timestamp_utc"],
            "query": t["query"],
            "latency_ms": t["latency_ms"],
            "escalate": bool(t["escalate"]),
            "no_context": bool(t["no_context"]),
            "citations": json.loads(t["citations"] or "[]"),
            "citation_integrity": t["citation_integrity"],
            "top_score": t["top_score"],
            "score_gap": t["score_gap"],
            "retrieval_mode": t["retrieval_mode"],
            "error": t["error"],
        }
        for t in rows
    ]


@app.get("/traces/{trace_id}")
def trace(trace_id: str) -> dict[str, Any]:
    engine = _engine_or_503()
    t = engine.store.get_trace(trace_id)
    if t is None:
        raise HTTPException(404, f"no trace {trace_id}")
    out = dict(t)
    for key in ("retrieved", "citations", "invalid_citations"):
        with contextlib.suppress(Exception):
            out[key] = json.loads(out.get(key) or "[]")
    return out


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UI_HTML


# --------------------------------------------------------------------------
# UI — one self-contained page, no build step and no CDN
# --------------------------------------------------------------------------

UI_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meridian Academy — Support Answers</title>
<style>
 :root{--bg:#fbfaf8;--fg:#1c1b19;--mut:#6b6862;--line:#e3e0da;--card:#fff;
       --ok:#1f6f43;--warn:#8a5a00;--bad:#a02c2c;--accent:#2b5f8f}
 @media(prefers-color-scheme:dark){:root{--bg:#161513;--fg:#eceae6;--mut:#9a968e;
       --line:#302e2a;--card:#1e1d1a;--ok:#63b98a;--warn:#d7a03c;--bad:#e0736d;--accent:#7fb0dd}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
 .wrap{max-width:900px;margin:0 auto;padding:32px 20px 80px}
 h1{font-size:20px;margin:0 0 4px} .sub{color:var(--mut);font-size:13px;margin:0 0 24px}
 form{display:flex;gap:8px;margin-bottom:8px}
 input{flex:1;padding:11px 13px;border:1px solid var(--line);border-radius:8px;
       background:var(--card);color:var(--fg);font-size:15px}
 input:focus{outline:2px solid var(--accent);outline-offset:-1px}
 button{padding:11px 20px;border:0;border-radius:8px;background:var(--accent);color:#fff;
        font-size:15px;font-weight:600;cursor:pointer}
 button:disabled{opacity:.55;cursor:default}
 .ex{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:26px}
 .ex button{background:transparent;color:var(--mut);border:1px solid var(--line);
            font-size:12.5px;font-weight:400;padding:5px 10px;border-radius:20px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;
       padding:18px;margin-bottom:14px}
 .ans{font-size:15.5px;white-space:pre-wrap}
 .badges{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px}
 .b{font-size:11.5px;padding:3px 9px;border-radius:20px;border:1px solid var(--line);
    color:var(--mut);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
 .b.ok{color:var(--ok);border-color:var(--ok)} .b.warn{color:var(--warn);border-color:var(--warn)}
 .b.bad{color:var(--bad);border-color:var(--bad)}
 h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);
    margin:26px 0 10px}
 details{border:1px solid var(--line);border-radius:8px;margin-bottom:8px;background:var(--card)}
 details[open]{border-color:var(--accent)}
 summary{padding:11px 14px;cursor:pointer;display:flex;gap:10px;align-items:baseline;
         flex-wrap:wrap;font-size:13.5px}
 summary::-webkit-details-marker{display:none}
 .sid{font-family:ui-monospace,Menlo,monospace;font-weight:700;color:var(--accent)}
 .sid.cited::after{content:" ●";color:var(--ok)}
 .meta{color:var(--mut);font-size:12px;font-family:ui-monospace,Menlo,monospace;
       margin-left:auto}
 .body{padding:0 14px 14px;white-space:pre-wrap;font-size:13px;color:var(--fg);
       border-top:1px solid var(--line);padding-top:12px;margin-top:2px}
 .err{color:var(--bad)} .mut{color:var(--mut);font-size:12.5px}
 table{border-collapse:collapse;width:100%;font-size:12px;
       font-family:ui-monospace,Menlo,monospace}
 td{padding:2px 10px 2px 0;color:var(--mut)}
</style></head><body><div class="wrap">
<h1>Meridian Academy — support answers</h1>
<p class="sub">Every answer is grounded in the document corpus and cites its sources.
When the corpus does not cover a question, the system says so instead of guessing.</p>

<form id="f"><input id="q" autocomplete="off"
  placeholder="e.g. If someone withdraws on day 45, do they get their GST back?" autofocus>
<button id="go">Ask</button></form>
<div class="ex" id="ex"></div>
<div id="out"></div>

<script>
const EX = ["If someone withdraws on day 45, do they get their GST back?",
            "Can I pay in monthly instalments?",
            "What is the minimum MAT score for the data science program?",
            "Does the Advanced Systems Design course include placement support?",
            "Does Meridian offer a cybersecurity program?"];
const $ = s => document.querySelector(s);
const esc = s => (s??"").toString().replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const num = (v,d=3) => v==null ? "-" : (+v).toFixed(d);

EX.forEach(t => { const b=document.createElement("button"); b.type="button"; b.textContent=t;
  b.onclick=()=>{ $("#q").value=t; $("#f").requestSubmit(); }; $("#ex").append(b); });

$("#f").onsubmit = async e => {
  e.preventDefault();
  const q = $("#q").value.trim(); if(!q) return;
  $("#go").disabled = true; $("#out").innerHTML = '<p class="mut">Searching the corpus…</p>';
  try {
    const r = await fetch("/ask", {method:"POST", headers:{"Content-Type":"application/json"},
                                  body: JSON.stringify({question:q})});
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || ("HTTP " + r.status));
    render(d);
  } catch(err) {
    $("#out").innerHTML = '<div class="card err">'+esc(err.message)+'</div>';
  } finally { $("#go").disabled = false; }
};

function render(d){
  const badge = (t,c="") => '<span class="b '+c+'">'+esc(t)+'</span>';
  let b = "";
  b += d.sufficient_context ? badge("grounded","ok") : badge("not in corpus","warn");
  if(d.escalate)          b += badge("escalate to human","warn");
  if(d.no_context)        b += badge("no LLM call made","warn");
  if(d.citation_integrity!=="ok") b += badge("citation integrity: "+d.citation_integrity,"bad");
  if(d.unsupported_claim) b += badge("unsupported claim","bad");
  b += badge(d.citations.length+" cited");
  b += badge(Math.round(d.latency_ms)+" ms");
  b += badge(d.input_tokens+"/"+d.output_tokens+" tok");

  let h = '<div class="card"><div class="ans">'+esc(d.answer)+'</div>'
        + '<div class="badges">'+b+'</div></div>';

  if(d.sources.length){
    h += '<h2>Sources</h2>';
    d.sources.forEach(s => {
      const loc = [s.section, s.page!=null ? "page "+s.page : ""].filter(Boolean).join(" · ");
      h += '<details'+(s.cited?" open":"")+'><summary>'
         + '<span class="sid'+(s.cited?" cited":"")+'">'+esc(s.id)+'</span>'
         + '<span>'+esc(s.doc)+'</span>'
         + '<span class="mut">'+esc(loc)+'</span>'
         + '<span class="meta">'+esc(s.authority)+(s.authority_demoted?" ↓":"")+'</span>'
         + '</summary><div class="body">'+esc(s.text)
         + '<table><tr><td>dense</td><td>rank '+(s.dense_rank??"-")+'</td><td>cos '
         + num(s.dense_score)+'</td></tr><tr><td>bm25</td><td>rank '+(s.bm25_rank??"-")
         + '</td><td>'+num(s.bm25_score,2)+'</td></tr><tr><td>rrf</td><td colspan="2">'
         + num(s.rrf_score,5)+'</td></tr></table></div></details>';
    });
  }
  h += '<p class="mut" style="margin-top:18px">trace <code>'+esc(d.trace_id)+'</code>'
     + ' · '+esc(d.model)+' · '+esc(d.embedding_backend)
     + (d.expanded_query ? ' · expanded: '+esc(d.expanded_query) : '')+'</p>';
  $("#out").innerHTML = h;
}
</script></div></body></html>"""
